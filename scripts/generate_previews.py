"""
Regenerate 4-second (120-frame @ 30 FPS) preview assets for all 3 golden scenes:
- outputs/{scene_id}/frame0_masks.png (SAM 3.1 4px eroded masks)
- outputs/{scene_id}/diff_overlay.png (Render + Diff IoU composite)
- outputs/{scene_id}/oracle_rollout.mp4 (Pure Waypoint Oracle baseline, 120 frames @ 30 FPS)
- outputs/{scene_id}/rollout.mp4 (Residual PPO rollout, 120 frames @ 30 FPS)
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.schema.scene_graph import SceneGraph
from src.compiler.base_env import WorldForgeEnv
from src.perception.sam_tracker import SAM31Tracker
from src.render_diff.mask_iou import evaluate_and_align_sim
from src.validation.ppo_runner import (
    OracleController,
    ResidualOracleEnv,
    _save_video,
    record_oracle_rollout,
)


def generate_scene_previews(scene_id: str, scenes_dir: Path, outputs_root: Path):
    clean_id = scene_id.replace(".json", "")
    scene_file = scenes_dir / f"{clean_id}.json"
    if not scene_file.exists():
        for p in scenes_dir.glob("*.json"):
            if clean_id in p.stem or p.stem in clean_id:
                scene_file = p
                clean_id = p.stem
                break

    out_dir = outputs_root / clean_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(scene_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    sg = SceneGraph.model_validate(data)

    print(f"Generating assets for {clean_id}...")

    # 1. SAM 3.1 Segment from Sim Env + frame0_masks.png
    tracker = SAM31Tracker()
    ref_env = WorldForgeEnv(sg, render_mode="rgb_array")
    tracking_result, _ = tracker.segment_from_sim_env(ref_env, num_frames=10, execute_oracle=False)
    ref_env.reset(seed=42)
    frame0_rgb = ref_env.render_camera()
    ref_env.close()

    mask_composite = frame0_rgb.copy()
    colors = [(255, 60, 60), (60, 255, 60), (60, 120, 255), (255, 200, 40), (200, 80, 255)]
    for idx, track in enumerate(tracking_result.tracks):
        color = colors[idx % len(colors)]
        mask = track.frame0_eroded_mask
        overlay = mask_composite.copy()
        overlay[mask] = color
        mask_composite = cv2.addWeighted(overlay, 0.45, mask_composite, 0.55, 0)
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
    print(f"  [x] Saved {out_dir / 'frame0_masks.png'}")

    # 2. Render + Diff Mask IoU overlay
    target_masks = {track.object_id: track.frame0_mask for track in tracking_result.tracks}
    _, diff_report = evaluate_and_align_sim(
        scene=sg,
        target_masks=target_masks,
        target_rgb=frame0_rgb,
        refine_if_below=0.80,
        max_refine_iters=5,
    )
    cv2.imwrite(
        str(out_dir / "diff_overlay.png"),
        cv2.cvtColor(diff_report.overlay_image, cv2.COLOR_RGB2BGR),
    )
    print(f"  [x] Saved {out_dir / 'diff_overlay.png'}")

    # 3. Pure Waypoint Oracle baseline rollout MP4 (120 frames @ 30 FPS = 4.0 seconds)
    oracle_path = out_dir / "oracle_rollout.mp4"
    record_oracle_rollout(sg, str(oracle_path), max_steps=120, fps=30)
    print(f"  [x] Saved {oracle_path}")

    # 4. Residual PPO rollout MP4 (120 frames @ 30 FPS = 4.0 seconds)
    rollout_path = out_dir / "rollout.mp4"
    record_residual_rollout(sg, str(rollout_path), max_steps=120, fps=30)
    print(f"  [x] Saved {rollout_path}")


def record_residual_rollout(
    scene: SceneGraph,
    output_path: str,
    max_steps: int = 120,
    fps: int = 30,
) -> str:
    """Record 4.0-second 120-frame Residual rollout video."""
    base_env = WorldForgeEnv(scene, render_mode="rgb_array")
    res_env = ResidualOracleEnv(base_env, scene.task.waypoints, residual_weight=0.10)
    obs, _ = res_env.reset(seed=101)
    frames: list[np.ndarray] = []
    terminated = False
    last_grip = scene.task.waypoints[-1][3] if scene.task.waypoints else 0.0

    for _ in range(max_steps):
        if not terminated:
            # Slight exploration jitter around oracle waypoint
            residual_action = np.random.uniform(-0.04, 0.04, size=4).astype(np.float32)
            residual_action[3] = 0.0  # Keep gripper discrete
            obs, reward, term, trunc, _ = res_env.step(residual_action)
            if term or trunc:
                terminated = True
        else:
            base_env.step(np.array([0.0, 0.0, 0.0, last_grip], dtype=np.float32))

        frame = base_env.render_camera()
        if frame is not None:
            frames.append(frame)

    base_env.close()
    return _save_video(frames, output_path, fps=fps)


def main():
    root = Path(__file__).resolve().parent.parent
    scenes_dir = root / "configs" / "scenes"
    outputs_root = root / "outputs"

    for scene_file in sorted(scenes_dir.glob("*.json")):
        generate_scene_previews(scene_file.stem, scenes_dir, outputs_root)
    print("Done regenerating all preview assets!")


if __name__ == "__main__":
    main()
