"""WorldForge Render + Diff Visual Verification & Coordinate Refinement.

Stage 8 of the pipeline:
1. Renders Frame 0 from estimated MuJoCo camera ("render_cam").
2. Extracts per-object binary masks via fast MuJoCo geom segmentation IDs or SAM 3.1.
3. Computes per-object Mask IoU against target (real) video masks.
4. If mean_iou < refine_if_below (default 0.75), runs iterative coordinate refinement
   over object (x, y) coordinates and scales to maximize Mask IoU.
5. Generates an (H, W, 3) composite overlay image showing Real Frame 0, Sim Frame 0,
   and Green=Overlap / Red=Mismatch mask differences.
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


# ---------------------------------------------------------------------------
# Mask IoU Metrics
# ---------------------------------------------------------------------------

def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute the Intersection-over-Union (Jaccard Index) of two binary masks.

    Handles edge cases:
    - If both masks are completely empty, returns 1.0 (perfect agreement).
    - If one mask is empty and the other is not, returns 0.0.
    - If disjoint non-empty masks, returns 0.0.
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)

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
) -> tuple[SceneGraph, RenderDiffReport]:
    """Evaluate visual alignment of a SceneGraph against target Frame 0 masks.

    If mean IoU is below ``refine_if_below``, iteratively refines object (x, y)
    coordinates using camera back-projection displacement until aligned.

    Returns (refined_scene, report).
    """
    current_scene = copy.deepcopy(scene)
    w, h = current_scene.camera.resolution

    # Helper to render masks and RGB for a scene
    def _render_scene(sg: SceneGraph) -> tuple[dict[str, np.ndarray], np.ndarray]:
        env = WorldForgeEnv(sg, render_mode="rgb_array")
        env.reset(seed=42)
        rgb = env.render_camera()

        masks: dict[str, np.ndarray] = {}
        if use_sam_resegment:
            from src.perception.sam_tracker import SAM31Tracker
            tracker = SAM31Tracker(cost_tracker=cost_tracker)
            res, _ = tracker.segment_from_sim_env(env, num_frames=1, execute_oracle=False)
            for track in res.tracks:
                masks[track.object_id] = track.frame0_mask
        else:
            # Fast MuJoCo geom segmentation buffer
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

    # Initial evaluation
    sim_masks, sim_rgb = _render_scene(current_scene)

    per_obj_iou: dict[str, float] = {
        obj_id: compute_mask_iou(sim_masks.get(obj_id, np.zeros((h, w), dtype=bool)), target_masks[obj_id])
        for obj_id in target_masks
    }
    initial_mean_iou = float(np.mean(list(per_obj_iou.values()))) if per_obj_iou else 1.0

    refined = False
    iterations_used = 0

    # Camera math for projection-guided coordinate search
    fx, fy, cam_cx, cam_cy = derive_camera_intrinsics(current_scene.camera)
    R_cam, _ = derive_camera_extrinsics(current_scene.camera)
    Z_approx = float(np.linalg.norm(np.array(current_scene.camera.pos) - np.array(current_scene.camera.lookat)))

    # Refinement loop
    if initial_mean_iou < refine_if_below:
        refined = True
        for it in range(max_refine_iters):
            iterations_used += 1

            # Check if all objects meet threshold
            if all(v >= refine_if_below for v in per_obj_iou.values()):
                break

            # Adjust each object whose IoU is below threshold
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

                # 2D centroid error in pixel space
                yt, xt = np.where(t_mask)
                ys, xs = np.where(s_mask)
                du = float(np.mean(xt) - np.mean(xs))
                dv = float(np.mean(yt) - np.mean(ys))

                # Map to camera frame and rotate to world frame
                dx_cam = du * Z_approx / fx
                dy_cam = -dv * Z_approx / fy
                dp_world = R_cam @ np.array([dx_cam, dy_cam, 0.0])

                step_scale = 0.75
                new_x = obj_node.position[0] + float(dp_world[0]) * step_scale
                new_y = obj_node.position[1] + float(dp_world[1]) * step_scale

                # Scale adjustment based on 2D mask pixel area ratio
                area_ratio = float(t_mask.sum()) / max(float(s_mask.sum()), 1.0)
                s_factor = float(np.clip(np.sqrt(area_ratio), 0.8, 1.25))
                obj_node.size = [float(s * (1.0 + 0.6 * (s_factor - 1.0))) for s in obj_node.size]

                # Keep height grounded to table
                half_h = obj_node.size[2] if obj_node.primitive == "box" else obj_node.size[1]
                new_z = current_scene.table_z + half_h + 0.001

                obj_node.position = (new_x, new_y, new_z)
                scene_updated = True

            if not scene_updated:
                break

            # Re-render with updated scene
            sim_masks, sim_rgb = _render_scene(current_scene)
            per_obj_iou = {
                oid: compute_mask_iou(sim_masks.get(oid, np.zeros((h, w), dtype=bool)), target_masks[oid])
                for oid in target_masks
            }

    final_mean_iou = float(np.mean(list(per_obj_iou.values()))) if per_obj_iou else 1.0

    # Generate composite overlay image
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

    Green: Overlap (A ∩ B)
    Red:   Mismatch (A ⊕ B)
    Base:  Alpha blend of target RGB (Real) and sim RGB.
    """
    w, h = resolution
    base_target = target_rgb if target_rgb is not None else sim_rgb.copy()

    # Blend Real Frame 0 and Sim Frame 0 as background
    composite = cv2.addWeighted(base_target, 0.5, sim_rgb, 0.5, 0.0)

    # Accumulate global overlap and mismatch masks across all objects
    all_overlap = np.zeros((h, w), dtype=bool)
    all_mismatch = np.zeros((h, w), dtype=bool)

    for obj_id, t_mask in target_masks.items():
        s_mask = sim_masks.get(obj_id, np.zeros((h, w), dtype=bool))
        all_overlap |= (t_mask & s_mask)
        all_mismatch |= (t_mask ^ s_mask)

    # Color overlay: Green for Overlap, Red for Mismatch
    green = np.array([0, 255, 0], dtype=np.uint8)
    red = np.array([255, 0, 0], dtype=np.uint8)

    if np.any(all_overlap):
        composite[all_overlap] = cv2.addWeighted(
            composite[all_overlap], 0.35, np.full_like(composite[all_overlap], green), 0.65, 0.0
        )
    if np.any(all_mismatch):
        composite[all_mismatch] = cv2.addWeighted(
            composite[all_mismatch], 0.35, np.full_like(composite[all_mismatch], red), 0.65, 0.0
        )

    # Watermark text
    cv2.putText(
        composite,
        "Diff Overlay: Green=Overlap, Red=Mismatch",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return composite

