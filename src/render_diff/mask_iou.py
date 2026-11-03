"""WorldForge Render + Diff Visual Verification & Coordinate Refinement.

Stage 8 of the pipeline:
1. Renders Frame 0 from estimated MuJoCo camera ("render_cam").
2. Extracts per-object binary masks via fast MuJoCo geom segmentation IDs or SAM 3.1.
3. Computes per-object Mask IoU against target (real) video masks.
4. If mean_iou < refine_if_below (default 0.75), runs iterative coordinate refinement
   over object (x, y) coordinates and scales to maximize Mask IoU.
5. Generates an (H, W, 3) composite overlay image showing Green=Aligned,
   Red=Video/Ref Only, and Blue=Sim Only mask differences per object.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import cv2
import mujoco
import numpy as np

from src.compiler.base_env import WorldForgeEnv
from src.perception.lift_3d import derive_camera_extrinsics, derive_camera_intrinsics
from src.schema.scene_graph import SceneGraph
from src.validation.repair_loop import CostTracker


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RenderDiffReport:
    """Report summarizing visual diff, per-object Mask IoU, and refinement telemetry."""

    per_object_iou: dict[str, float]
    mean_iou: float
    initial_mean_iou: float
    refined: bool
    iterations_used: int
    overlay_image: np.ndarray  # (H, W, 3) uint8 composite
    passed: bool = True


@dataclass
class MaskDiffReport:
    """Lightweight mask diff report for reconstructed multi-stack scenes."""

    scene_id: str = ""
    mean_iou: float = 1.0
    per_object_iou: dict[str, float] = field(default_factory=dict)
    passed: bool = True
    refined: bool = False
    iterations_used: int = 0
    initial_mean_iou: float = 1.0
    overlay_image: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Mask IoU Metrics
# ---------------------------------------------------------------------------

def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute the Intersection-over-Union (Jaccard Index) of two binary masks."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 1.0
    return float(intersection / union)


@dataclass
class SceneIoUReport:
    """Per-scene mask IoU report."""

    per_object_iou: dict[str, float]
    mean_iou: float
    passed: bool


def compute_scene_mask_iou(
    ref_masks: dict[str, np.ndarray],
    sim_masks: dict[str, np.ndarray],
    threshold: float = 0.60,
) -> SceneIoUReport:
    """Compute per-object and mean Mask IoU between reference and simulated masks."""
    per_object: dict[str, float] = {}
    all_keys = set(ref_masks.keys()) | set(sim_masks.keys())

    if not all_keys:
        return SceneIoUReport(per_object_iou={}, mean_iou=1.0, passed=True)

    for k in all_keys:
        ref_m = ref_masks.get(k)
        sim_m = sim_masks.get(k)
        if ref_m is None or sim_m is None:
            per_object[k] = 0.0
        else:
            per_object[k] = compute_mask_iou(ref_m, sim_m)

    mean_iou = float(np.mean(list(per_object.values()))) if per_object else 1.0
    passed = mean_iou >= threshold

    return SceneIoUReport(
        per_object_iou=per_object,
        mean_iou=mean_iou,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Evaluate & Align Pipeline
# ---------------------------------------------------------------------------

def evaluate_and_align_sim(
    scene: SceneGraph,
    target_masks: dict[str, np.ndarray],
    target_rgb: np.ndarray | None = None,
    refine_if_below: float = 0.75,
    max_refine_iters: int = 20,
    use_sam_resegment: bool = False,
    cost_tracker: CostTracker | None = None,
    pass_threshold: float = 0.60,
) -> tuple[SceneGraph, RenderDiffReport]:
    """Evaluate visual alignment of a SceneGraph against target Frame 0 masks.

    If mean IoU is below ``refine_if_below``, iteratively refines object (x, y)
    coordinates and scales using camera back-projection displacement until aligned.
    """
    current_scene = copy.deepcopy(scene)
    w, h = current_scene.camera.resolution

    def _render_scene(sg: SceneGraph) -> tuple[dict[str, np.ndarray], np.ndarray]:
        env = WorldForgeEnv(sg, render_mode="rgb_array")
        env.reset(seed=42)
        rgb = env.render_camera()

        masks: dict[str, np.ndarray] = {}
        if use_sam_resegment:
            from src.perception.sam_tracker import SAM31Tracker
            tracker = SAM31Tracker(cost_tracker=cost_tracker)
            if tracker.has_configured_backend():
                prompts = [o.label for o in current_scene.objects]
                obj_ids = [o.id for o in current_scene.objects]
                res = tracker.segment_image(rgb, text_prompts=prompts, object_ids=obj_ids)
                for track in res.tracks:
                    masks[track.object_id] = track.frame0_mask
            else:
                res, _ = tracker.segment_from_sim_env(env, num_frames=1, execute_oracle=False)
                for track in res.tracks:
                    masks[track.object_id] = track.frame0_mask
        else:
            renderer = mujoco.Renderer(env.model, height=h, width=w)
            renderer.enable_segmentation_rendering()
            renderer.update_scene(env.data, camera="render_cam")
            seg = renderer.render()[:, :, 0]
            for obj_id in target_masks:
                gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{obj_id}_geom")
                if gid >= 0:
                    masks[obj_id] = (seg == gid)
                else:
                    masks[obj_id] = np.zeros((h, w), dtype=bool)
            renderer.close()

        env.close()
        return masks, rgb

    sim_masks, sim_rgb = _render_scene(current_scene)

    all_keys = set(target_masks.keys()) | set(sim_masks.keys())
    per_obj_iou: dict[str, float] = {
        obj_id: compute_mask_iou(
            sim_masks.get(obj_id, np.zeros((h, w), dtype=bool)),
            target_masks.get(obj_id, np.zeros((h, w), dtype=bool)),
        )
        for obj_id in all_keys
    }
    initial_mean_iou = float(np.mean(list(per_obj_iou.values()))) if per_obj_iou else 1.0

    refined = False
    iterations_used = 0

    fx, fy, cam_cx, cam_cy = derive_camera_intrinsics(current_scene.camera)
    R_cam, _ = derive_camera_extrinsics(current_scene.camera)
    Z_approx = float(np.linalg.norm(np.array(current_scene.camera.pos) - np.array(current_scene.camera.lookat)))

    max_target_area = max((int(np.sum(m)) for m in target_masks.values()), default=0)
    is_closeup_or_multistack = (
        current_scene.task.task_type == "multi_stack" or max_target_area > 8000
    )

    if initial_mean_iou < refine_if_below and not is_closeup_or_multistack:
        refined = True
        for _ in range(max_refine_iters):
            iterations_used += 1

            if all(v >= refine_if_below for v in per_obj_iou.values()):
                break

            scene_updated = False
            for obj_node in current_scene.objects:
                obj_id = obj_node.id
                if obj_id not in target_masks or obj_node.parent_id is not None:
                    continue

                t_mask = target_masks[obj_id]
                s_mask = sim_masks.get(obj_id, np.zeros((h, w), dtype=bool))
                obj_iou = compute_mask_iou(s_mask, t_mask)
                if obj_iou >= refine_if_below or not np.any(t_mask) or not np.any(s_mask):
                    continue

                yt, xt = np.where(t_mask)
                ys, xs = np.where(s_mask)
                du = float(np.mean(xt) - np.mean(xs))
                dv = float(np.mean(yt) - np.mean(ys))

                dx_cam = du * Z_approx / fx
                dy_cam = -dv * Z_approx / fy
                dp_world = R_cam @ np.array([dx_cam, dy_cam, 0.0])

                step_scale = 0.75
                new_x = obj_node.position[0] + float(dp_world[0]) * step_scale
                new_y = obj_node.position[1] + float(dp_world[1]) * step_scale

                area_ratio = float(t_mask.sum()) / max(float(s_mask.sum()), 1.0)
                s_factor = float(np.clip(np.sqrt(area_ratio), 0.8, 1.25))
                obj_node.size = [float(s * (1.0 + 0.6 * (s_factor - 1.0))) for s in obj_node.size]

                half_h = obj_node.size[2] if obj_node.primitive == "box" else obj_node.size[1]
                new_z = current_scene.table_z + half_h + 0.001

                obj_node.position = (new_x, new_y, new_z)
                scene_updated = True

            if not scene_updated:
                break

            sim_masks, sim_rgb = _render_scene(current_scene)
            all_keys = set(target_masks.keys()) | set(sim_masks.keys())
            per_obj_iou = {
                oid: compute_mask_iou(
                    sim_masks.get(oid, np.zeros((h, w), dtype=bool)),
                    target_masks.get(oid, np.zeros((h, w), dtype=bool)),
                )
                for oid in all_keys
            }

    final_mean_iou = float(np.mean(list(per_obj_iou.values()))) if per_obj_iou else 1.0

    overlay_image = _create_overlay_image(
        target_rgb=target_rgb,
        sim_rgb=sim_rgb,
        target_masks=target_masks,
        sim_masks=sim_masks,
        resolution=(w, h),
    )

    report = RenderDiffReport(
        per_object_iou=per_obj_iou,
        mean_iou=final_mean_iou,
        initial_mean_iou=initial_mean_iou,
        refined=refined,
        iterations_used=iterations_used,
        overlay_image=overlay_image,
        passed=(final_mean_iou >= pass_threshold),
    )
    return current_scene, report


# ---------------------------------------------------------------------------
# Composite Overlay Image Generator
# ---------------------------------------------------------------------------

def _create_overlay_image(
    target_rgb: np.ndarray | None,
    sim_rgb: np.ndarray,
    target_masks: dict[str, np.ndarray],
    sim_masks: dict[str, np.ndarray],
    resolution: tuple[int, int],
) -> np.ndarray:
    """Generate (H, W, 3) uint8 composite showing Real Frame 0, Sim Frame 0, and Diff.

    Green (16, 185, 129): Per-object Overlap (Ref_k ∩ Sim_k)
    Red   (239, 68, 68):  Video/Ref Only (Ref_k \\ Sim_k)
    Blue  (59, 130, 246): Sim Only (Sim_k \\ Ref_k)
    """
    w, h = resolution
    base_target = target_rgb if target_rgb is not None else sim_rgb.copy()
    if base_target.shape[:2] != (h, w):
        base_target = cv2.resize(base_target, (w, h), interpolation=cv2.INTER_AREA)

    composite = cv2.addWeighted(base_target, 0.45, sim_rgb, 0.45, 0.0)

    all_overlap = np.zeros((h, w), dtype=bool)
    all_ref_only = np.zeros((h, w), dtype=bool)
    all_sim_only = np.zeros((h, w), dtype=bool)

    all_keys = set(target_masks.keys()) | set(sim_masks.keys())
    for obj_id in all_keys:
        t_mask = target_masks.get(obj_id, np.zeros((h, w), dtype=bool))
        s_mask = sim_masks.get(obj_id, np.zeros((h, w), dtype=bool))
        if t_mask.shape != (h, w):
            t_mask = cv2.resize(t_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        if s_mask.shape != (h, w):
            s_mask = cv2.resize(s_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

        all_overlap |= (t_mask & s_mask)
        all_ref_only |= (t_mask & ~s_mask)
        all_sim_only |= (s_mask & ~t_mask)

    green = np.array([16, 185, 129], dtype=np.uint8)
    red = np.array([239, 68, 68], dtype=np.uint8)
    blue = np.array([59, 130, 246], dtype=np.uint8)

    if np.any(all_ref_only):
        composite[all_ref_only] = cv2.addWeighted(
            composite[all_ref_only], 0.30, np.full_like(composite[all_ref_only], red), 0.70, 0.0
        )
    if np.any(all_sim_only):
        composite[all_sim_only] = cv2.addWeighted(
            composite[all_sim_only], 0.30, np.full_like(composite[all_sim_only], blue), 0.70, 0.0
        )
    if np.any(all_overlap):
        composite[all_overlap] = cv2.addWeighted(
            composite[all_overlap], 0.30, np.full_like(composite[all_overlap], green), 0.70, 0.0
        )

    cv2.putText(
        composite,
        "Diff Overlay: Green=Aligned, Red=Video-Only, Blue=Sim-Only",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return composite
