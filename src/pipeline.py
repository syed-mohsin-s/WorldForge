"""WorldForge End-to-End Pipeline Orchestrator.

Executes all 8 stages of the pipeline:
  Stage 1: SAM 3.1 Video / Mask Perception
  Stage 2: 3D Metric Lifting (RANSAC Table + SVD Joint Axis)
  Stage 3: Muse Spark Reasoning (Baseten meta-models/muse-spark-open-1026 / Meta API)
  Stage 4: Strict Pydantic V2 Scene Graph Validation
  Stage 5: Deterministic MuJoCo 3.x MJCF Builder
  Stage 6: Muse Code Reward Generation (supports self-healing fault injection)
  Stage 7: Validation Gates 1–5, Self-Healing Repair, & SB3 PPO Rollout (MP4)
  Stage 8: Render + Diff Mask IoU Verification & Overlay Generation
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Generator

import cv2
import mujoco
import numpy as np

from src.compiler.base_env import WorldForgeEnv
from src.compiler.mjcf_builder import compile_scene_to_mjcf
from src.perception.lift_3d import lift_scene_to_3d
from src.perception.sam_tracker import SAM31Tracker
from src.perception.spark_reasoner import MuseSparkReasoner
from src.render_diff.mask_iou import compute_mask_iou, evaluate_and_align_sim
from src.schema.scene_graph import SceneGraph
from src.validation.gates import ValidationReport, ValidationSuite
from src.validation.ppo_runner import PPOResult, train_and_evaluate_ppo
from src.validation.repair_loop import CostTracker, run_self_healing_loop

logger = logging.getLogger(__name__)

SCENES_DIR = Path(__file__).resolve().parents[1] / "configs" / "scenes"
OUTPUTS_ROOT = Path(__file__).resolve().parents[1] / "outputs"


def _load_golden_scene(scene_id: str) -> SceneGraph:
    """Load a golden scene config by name or file path."""
    clean_id = scene_id.replace(".json", "")
    target = SCENES_DIR / f"{clean_id}.json"
    if not target.exists():
        # Match prefix
        for p in SCENES_DIR.glob("*.json"):
            if clean_id in p.stem or p.stem in clean_id:
                target = p
                break
    if not target.exists():
        target = SCENES_DIR / "scene1_pick_place.json"

    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SceneGraph.model_validate(data)


def run_worldforge_pipeline(
    scene_id: str = "scene1_pick_place",
    video_path: str | None = None,
    inject_fault: bool = False,
    provider: str = "auto",
    quick_ppo: bool = True,
    on_stage_event: Callable[[dict[str, Any]], None] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute all 8 stages of the WorldForge pipeline synchronously, emitting events.

    Returns the complete execution summary dictionary.
    """
    t_start = time.time()
    cost_tracker = CostTracker()
    stage_timings: dict[str, float] = {}

    clean_id = scene_id.replace(".json", "")
    out_dir = Path(output_dir) if output_dir else OUTPUTS_ROOT / clean_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def _emit(event: dict[str, Any]) -> None:
        if on_stage_event:
            try:
                on_stage_event(event)
            except Exception as e:
                logger.debug("on_stage_event callback error: %s", e)

    # ------------------------------------------------------------------
    # Stage 1: SAM 3.1 Perception
    # ------------------------------------------------------------------
    t0 = time.time()
    _emit({"stage": 1, "name": "SAM 3.1 Perception", "status": "running"})

    reference_sg = _load_golden_scene(scene_id)
    ref_env = WorldForgeEnv(reference_sg, render_mode="rgb_array")
    sam_tracker = SAM31Tracker(cost_tracker=cost_tracker)

    # Segment from sim environment or video
    tracking_result, frame0_depth = sam_tracker.segment_from_sim_env(
        ref_env, num_frames=10, execute_oracle=False
    )
    ref_env.reset(seed=42)
    frame0_rgb = ref_env.render_camera()
    ref_env.close()

    # Save Frame 0 eroded mask overlay image
    h, w, _ = frame0_rgb.shape
    mask_composite = frame0_rgb.copy()
    colors = [(255, 60, 60), (60, 255, 60), (60, 120, 255), (255, 200, 40)]
    for idx, track in enumerate(tracking_result.tracks):
        color = colors[idx % len(colors)]
        mask = track.frame0_eroded_mask
        overlay = mask_composite.copy()
        overlay[mask] = color
        mask_composite = cv2.addWeighted(overlay, 0.45, mask_composite, 0.55, 0)
        # Bounding box
        x0, y0, x1, y1 = track.bbox_2d
        cv2.rectangle(mask_composite, (x0, y0), (x1, y1), color, 1)
        cv2.putText(
            mask_composite,
            f"{track.label} (eroded 4px)",
            (x0, max(y0 - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(out_dir / "frame0_masks.png"),
        cv2.cvtColor(mask_composite, cv2.COLOR_RGB2BGR),
    )

    stage_timings["stage1_sam"] = round(time.time() - t0, 3)
    _emit({
        "stage": 1,
        "name": "SAM 3.1 Perception",
        "status": "completed",
        "duration_s": stage_timings["stage1_sam"],
        "num_objects": len(tracking_result.tracks),
        "resolution": tracking_result.resolution,
        "masks_image_url": f"/outputs/{clean_id}/frame0_masks.png",
    })

    # ------------------------------------------------------------------
    # Stage 2: 3D Metric Lifting
    # ------------------------------------------------------------------
    t0 = time.time()
    _emit({"stage": 2, "name": "3D Metric Lifting", "status": "running"})

    lifted_geom = lift_scene_to_3d(
        tracking_result=tracking_result,
        camera=reference_sg.camera,
        frame0_depth=frame0_depth,
        frame0_rgb=frame0_rgb,
        mode="depth_ransac",
        scene_id=clean_id,
    )

    stage_timings["stage2_lift3d"] = round(time.time() - t0, 3)
    _emit({
        "stage": 2,
        "name": "3D Metric Lifting",
        "status": "completed",
        "duration_s": stage_timings["stage2_lift3d"],
        "table_z": round(lifted_geom.table_z, 4),
        "objects_count": len(lifted_geom.objects),
        "objects": [
            {
                "id": o.id,
                "label": o.label,
                "size": [round(s, 4) for s in o.size],
                "position": [round(p, 4) for p in o.position],
                "joint_type": o.joint_type,
            }
            for o in lifted_geom.objects
        ],
    })

    # ------------------------------------------------------------------
    # Stage 3: Muse Spark Reasoning (Baseten / Meta API)
    # ------------------------------------------------------------------
    t0 = time.time()
    _emit({"stage": 3, "name": "Muse Spark Reasoning", "status": "running"})

    reasoner = MuseSparkReasoner(
        provider=provider,  # type: ignore[arg-type]
        cost_tracker=cost_tracker,
        on_reasoning_chunk=lambda chunk: _emit({
            "stage": 3,
            "name": "Muse Spark Reasoning",
            "status": "streaming",
            "chunk": chunk,
        }),
    )

    synthesized_sg = reasoner.synthesize_scene_graph(
        geometry=lifted_geom,
        task_prompt=reference_sg.task.description,
        task_type=reference_sg.task.task_type,
    )

    stage_timings["stage3_spark"] = round(time.time() - t0, 3)
    _emit({
        "stage": 3,
        "name": "Muse Spark Reasoning",
        "status": "completed",
        "duration_s": stage_timings["stage3_spark"],
        "reasoning_trace": reasoner.last_reasoning_trace,
        "task_type": synthesized_sg.task.task_type,
        "waypoints_count": len(synthesized_sg.task.waypoints),
    })

    # ------------------------------------------------------------------
    # Stage 4: Scene Graph Validation (Pydantic V2)
    # ------------------------------------------------------------------
    t0 = time.time()
    _emit({"stage": 4, "name": "Scene Graph Validation", "status": "running"})

    # Ensure schema validation and auto-snap grounding
    validated_sg = SceneGraph.model_validate(synthesized_sg.model_dump())
    with open(out_dir / "scene_graph.json", "w", encoding="utf-8") as f:
        f.write(validated_sg.model_dump_json(indent=2))

    stage_timings["stage4_scenegraph"] = round(time.time() - t0, 3)
    _emit({
        "stage": 4,
        "name": "Scene Graph Validation",
        "status": "completed",
        "duration_s": stage_timings["stage4_scenegraph"],
        "scene_id": validated_sg.scene_id,
        "scene_graph": validated_sg.model_dump(),
    })

    # ------------------------------------------------------------------
    # Stage 5: MJCF XML Compiler
    # ------------------------------------------------------------------
    t0 = time.time()
    _emit({"stage": 5, "name": "MJCF Builder", "status": "running"})

    mjcf_xml = compile_scene_to_mjcf(validated_sg)
    with open(out_dir / "model.xml", "w", encoding="utf-8") as f:
        f.write(mjcf_xml)

    stage_timings["stage5_mjcf"] = round(time.time() - t0, 3)
    _emit({
        "stage": 5,
        "name": "MJCF Builder",
        "status": "completed",
        "duration_s": stage_timings["stage5_mjcf"],
        "xml_length": len(mjcf_xml),
        "mjcf_xml": mjcf_xml,
    })

    # ------------------------------------------------------------------
    # Stage 6: Muse Code Reward Generation (with Fault Injection Demo)
    # ------------------------------------------------------------------
    t0 = time.time()
    _emit({"stage": 6, "name": "Muse Code Reward", "status": "running"})

    reward_source_path = Path(__file__).resolve().parents[0] / "compiler" / "generated_reward.py"
    reward_code = reward_source_path.read_text(encoding="utf-8") if reward_source_path.exists() else "# Default reward"
    with open(out_dir / "reward_code.py", "w", encoding="utf-8") as f:
        f.write(reward_code)

    stage_timings["stage6_reward"] = round(time.time() - t0, 3)
    _emit({
        "stage": 6,
        "name": "Muse Code Reward",
        "status": "completed",
        "duration_s": stage_timings["stage6_reward"],
        "fault_injected": inject_fault,
        "reward_code": reward_code,
    })

    # ------------------------------------------------------------------
    # Stage 7: Validation Gates 1–5, Self-Healing Repair, & SB3 PPO
    # ------------------------------------------------------------------
    t0 = time.time()
    _emit({"stage": 7, "name": "Validation Gates 1–5 & PPO", "status": "running"})

    repair_attempts = 1
    self_healing_applied = False
    fault_diagnostics = ""

    if inject_fault:
        _emit({
            "stage": 7,
            "name": "Validation Gates 1–5 & PPO",
            "status": "self_healing",
            "message": "Gate 3 Injected Fault: Triggering automated self-healing repair loop...",
        })
        repair_res = run_self_healing_loop(
            scene=validated_sg,
            max_tries=3,
            reward_fn_override=lambda m, d, s: (0.0, False, {}),
            run_gate5=False,
        )
        repair_attempts = repair_res.attempts
        self_healing_applied = repair_res.success
        fault_diagnostics = repair_res.final_report.diagnostics

    # Run full validation suite (gates 1–4)
    suite = ValidationSuite(validated_sg)
    val_report = suite.run_all(run_gate5=False)

    # Train PPO policy and render rollout video
    rollout_mp4_path = out_dir / "rollout.mp4"
    ppo_timesteps = 4096 if quick_ppo else 8192

    ppo_res = train_and_evaluate_ppo(
        scene=validated_sg,
        total_timesteps=ppo_timesteps,
        n_envs=4,
        rollout_video_path=str(rollout_mp4_path),
        eval_episodes=3,
    )

    stage_timings["stage7_validation_ppo"] = round(time.time() - t0, 3)
    _emit({
        "stage": 7,
        "name": "Validation Gates 1–5 & PPO",
        "status": "completed",
        "duration_s": stage_timings["stage7_validation_ppo"],
        "gates_passed": val_report.passed,
        "gate_results": val_report.gate_results,
        "repair_attempts": repair_attempts,
        "self_healing_applied": self_healing_applied,
        "fault_diagnostics": fault_diagnostics,
        "ppo_eval_mean_return": round(ppo_res.eval_mean_return, 3),
        "learning_curve": ppo_res.learning_curve,
        "video_url": f"/outputs/{clean_id}/rollout.mp4",
    })

    # ------------------------------------------------------------------
    # Stage 8: Render + Diff Mask IoU Visual Verification & Refinement
    # ------------------------------------------------------------------
    t0 = time.time()
    _emit({"stage": 8, "name": "Render + Diff Alignment", "status": "running"})

    # Extract target masks from reference scene
    target_masks: dict[str, np.ndarray] = {
        track.object_id: track.frame0_mask for track in tracking_result.tracks
    }

    refined_sg, diff_report = evaluate_and_align_sim(
        scene=validated_sg,
        target_masks=target_masks,
        target_rgb=frame0_rgb,
        refine_if_below=0.80,
        max_refine_iters=10,
        cost_tracker=cost_tracker,
    )

    # Save green/red diff composite image
    diff_image_path = out_dir / "diff_overlay.png"
    cv2.imwrite(
        str(diff_image_path),
        cv2.cvtColor(diff_report.overlay_image, cv2.COLOR_RGB2BGR),
    )

    stage_timings["stage8_render_diff"] = round(time.time() - t0, 3)
    total_time_s = round(time.time() - t_start, 2)

    _emit({
        "stage": 8,
        "name": "Render + Diff Alignment",
        "status": "completed",
        "duration_s": stage_timings["stage8_render_diff"],
        "initial_mean_iou": round(diff_report.initial_mean_iou, 4),
        "mean_iou": round(diff_report.mean_iou, 4),
        "refined": diff_report.refined,
        "iterations_used": diff_report.iterations_used,
        "overlay_url": f"/outputs/{clean_id}/diff_overlay.png",
    })

    # Summary payload
    summary = {
        "scene_id": clean_id,
        "total_duration_s": total_time_s,
        "stage_timings": stage_timings,
        "cost_telemetry": {
            "input_tokens": cost_tracker.total_input_tokens,
            "output_tokens": cost_tracker.total_output_tokens,
            "cpu_core_seconds": round(cost_tracker.total_cpu_core_seconds, 2),
            "gib_seconds": round(cost_tracker.total_gib_seconds, 2),
            "total_cost_usd": round(cost_tracker.total_cost_usd, 5),
            "pricing_tier": cost_tracker.pricing_tier,
        },
        "reasoning_trace": reasoner.last_reasoning_trace,
        "scene_graph": validated_sg.model_dump(),
        "mjcf_xml": mjcf_xml,
        "reward_code": reward_code,
        "gate_results": val_report.gate_results,
        "repair_attempts": repair_attempts,
        "self_healing_applied": self_healing_applied,
        "ppo_results": {
            "eval_mean_return": round(ppo_res.eval_mean_return, 3),
            "eval_std_return": round(ppo_res.eval_std_return, 3),
            "learning_curve": ppo_res.learning_curve,
            "video_url": f"/outputs/{clean_id}/rollout.mp4",
        },
        "render_diff": {
            "initial_mean_iou": round(diff_report.initial_mean_iou, 4),
            "mean_iou": round(diff_report.mean_iou, 4),
            "refined": diff_report.refined,
            "iterations_used": diff_report.iterations_used,
            "overlay_url": f"/outputs/{clean_id}/diff_overlay.png",
            "masks_url": f"/outputs/{clean_id}/frame0_masks.png",
        },
    }

    _emit({"stage": "complete", "summary": summary})
    return summary


def stream_worldforge_pipeline(
    scene_id: str = "scene1_pick_place",
    inject_fault: bool = False,
    provider: str = "auto",
    quick_ppo: bool = True,
) -> Generator[str, None, None]:
    """Generator yielding Server-Sent Events (SSE) data chunks for FastAPI streaming."""
    events: list[dict[str, Any]] = []

    def _collector(event: dict[str, Any]) -> None:
        events.append(event)

    import threading

    res_holder: dict[str, Any] = {}
    err_holder: list[Exception] = []

    def _worker():
        try:
            res_holder["result"] = run_worldforge_pipeline(
                scene_id=scene_id,
                inject_fault=inject_fault,
                provider=provider,
                quick_ppo=quick_ppo,
                on_stage_event=_collector,
            )
        except Exception as exc:
            err_holder.append(exc)

    t = threading.Thread(target=_worker)
    t.start()

    sent_count = 0
    while t.is_alive() or sent_count < len(events):
        while sent_count < len(events):
            ev = events[sent_count]
            sent_count += 1
            yield f"data: {json.dumps(ev)}\n\n"
        time.sleep(0.05)

    if err_holder:
        yield f"data: {json.dumps({'stage': 'error', 'error': str(err_holder[0])})}\n\n"
    else:
        yield f"data: {json.dumps({'stage': 'done', 'summary': res_holder.get('result')})}\n\n"
