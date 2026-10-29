"""WorldForge Phase 4 Test Suite — Render + Diff Visual Verification & Modal Sandbox Cloud Runner.

Tests:
1. compute_mask_iou edge cases (identical, disjoint, both empty, one empty, partial overlap).
2. evaluate_and_align_sim ground truth verification across golden scenes (initial IoU > 0.85).
3. Coordinate refinement on scene1_pick_place perturbed by +4cm in X:
   - Initial IoU < 0.65
   - Refined IoU > 0.80
   - Overlay composite image shape (480, 640, 3) uint8
   - Object table snap preservation (z = table_z + half_height + 0.001).
4. ModalSandboxRunner:
   - Local fallback execution for validation and repair
   - Resource cost tracking (CPU core-seconds, GiB-seconds, USD)
   - Hard budget cap enforcement raising BudgetExceededError.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from src.compiler.base_env import WorldForgeEnv
from src.render_diff.mask_iou import (
    RenderDiffReport,
    compute_mask_iou,
    evaluate_and_align_sim,
)
from src.sandbox.modal_executor import BudgetExceededError, ModalSandboxRunner
from src.schema.scene_graph import SceneGraph
from src.validation.repair_loop import CostTracker

CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "scenes"


# ===========================================================================
# 1. Mask IoU Unit Tests
# ===========================================================================

def test_mask_iou_identical():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:50, 20:50] = True
    assert compute_mask_iou(mask, mask) == 1.0


def test_mask_iou_disjoint():
    mask_a = np.zeros((100, 100), dtype=bool)
    mask_b = np.zeros((100, 100), dtype=bool)
    mask_a[10:30, 10:30] = True
    mask_b[60:80, 60:80] = True
    assert compute_mask_iou(mask_a, mask_b) == 0.0


def test_mask_iou_both_empty():
    mask_a = np.zeros((100, 100), dtype=bool)
    mask_b = np.zeros((100, 100), dtype=bool)
    assert compute_mask_iou(mask_a, mask_b) == 1.0


def test_mask_iou_one_empty():
    mask_a = np.zeros((100, 100), dtype=bool)
    mask_b = np.zeros((100, 100), dtype=bool)
    mask_a[10:20, 10:20] = True
    assert compute_mask_iou(mask_a, mask_b) == 0.0
    assert compute_mask_iou(mask_b, mask_a) == 0.0


def test_mask_iou_partial_overlap():
    mask_a = np.zeros((10, 10), dtype=bool)
    mask_b = np.zeros((10, 10), dtype=bool)
    mask_a[0:4, 0:4] = True  # 16 pixels
    mask_b[2:6, 0:4] = True  # 16 pixels
    # Intersection: rows 2:4, cols 0:4 -> 2 * 4 = 8 pixels
    # Union: rows 0:6, cols 0:4 -> 6 * 4 = 24 pixels
    # IoU = 8 / 24 = 1/3
    expected_iou = 8.0 / 24.0
    assert pytest.approx(compute_mask_iou(mask_a, mask_b), abs=1e-5) == expected_iou


# ===========================================================================
# 2. Render Diff & Alignment Tests
# ===========================================================================

def _extract_ground_truth_masks(scene: SceneGraph) -> tuple[dict[str, np.ndarray], np.ndarray]:
    w, h = scene.camera.resolution
    env = WorldForgeEnv(scene, render_mode="rgb_array")
    env.reset(seed=42)
    rgb = env.render_camera()
    renderer = mujoco.Renderer(env.model, height=h, width=w)
    renderer.enable_segmentation_rendering()
    renderer.update_scene(env.data, camera="render_cam")
    seg = renderer.render()[:, :, 0]
    masks = {}
    for obj in scene.objects:
        gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{obj.id}_geom")
        if gid >= 0:
            masks[obj.id] = (seg == gid)
        else:
            masks[obj.id] = np.zeros((h, w), dtype=bool)
    renderer.close()
    env.close()
    return masks, rgb


def test_render_diff_ground_truth_scene1():
    with open(CONFIGS_DIR / "scene1_pick_place.json") as f:
        sg = SceneGraph.model_validate(json.load(f))

    target_masks, target_rgb = _extract_ground_truth_masks(sg)
    _, report = evaluate_and_align_sim(sg, target_masks=target_masks, target_rgb=target_rgb)

    assert report.initial_mean_iou >= 0.85
    assert report.mean_iou >= 0.85
    assert not report.refined
    assert report.iterations_used == 0
    assert report.overlay_image.shape == (sg.camera.resolution[1], sg.camera.resolution[0], 3)
    assert report.overlay_image.dtype == np.uint8


@pytest.mark.parametrize("scene_name", [
    "scene1_pick_place.json",
    "scene2_push_obstacle.json",
    "scene3_articulated_drawer.json",
])
def test_render_diff_golden_scenes_baseline(scene_name: str):
    with open(CONFIGS_DIR / scene_name) as f:
        sg = SceneGraph.model_validate(json.load(f))

    target_masks, target_rgb = _extract_ground_truth_masks(sg)
    _, report = evaluate_and_align_sim(sg, target_masks=target_masks, target_rgb=target_rgb)

    assert report.initial_mean_iou >= 0.85
    assert report.mean_iou >= 0.85


def test_coordinate_refinement_perturbed_scene():
    with open(CONFIGS_DIR / "scene1_pick_place.json") as f:
        sg = SceneGraph.model_validate(json.load(f))

    target_masks, target_rgb = _extract_ground_truth_masks(sg)

    # Perturb red_cube by +4cm in X
    sg_perturbed = sg.model_copy(deep=True)
    target_obj = None
    for obj in sg_perturbed.objects:
        if obj.id == "red_cube":
            target_obj = obj
            obj.position = (obj.position[0] + 0.04, obj.position[1], obj.position[2])

    assert target_obj is not None

    # Run alignment and coordinate refinement
    refined_sg, report = evaluate_and_align_sim(
        sg_perturbed,
        target_masks=target_masks,
        target_rgb=target_rgb,
        refine_if_below=0.75,
        max_refine_iters=10,
    )

    # Initial IoU must be < 0.65 due to 4cm perturbation
    assert report.initial_mean_iou < 0.65
    # Refined IoU must recover to > 0.80
    assert report.mean_iou > 0.80
    assert report.refined
    assert report.iterations_used > 0
    assert report.overlay_image.shape == (480, 640, 3)
    assert report.overlay_image.dtype == np.uint8

    # Verify tabletop snap condition is strictly preserved
    refined_cube = [o for o in refined_sg.objects if o.id == "red_cube"][0]
    expected_z = refined_sg.table_z + refined_cube.size[2] + 0.001
    assert pytest.approx(refined_cube.position[2], abs=1e-4) == expected_z


# ===========================================================================
# 3. Modal Sandbox Runner & Cost Accounting Tests
# ===========================================================================

def test_modal_runner_local_fallback_validation():
    with open(CONFIGS_DIR / "scene1_pick_place.json") as f:
        sg = SceneGraph.model_validate(json.load(f))

    runner = ModalSandboxRunner(max_budget_usd=2.00)
    assert not runner.is_cloud_available()

    report = runner.run_validation(sg, run_gate5=False)
    assert report.passed
    assert runner.cost_tracker.total_cpu_core_seconds > 0.0
    assert runner.cost_tracker.total_gib_seconds > 0.0
    assert runner.cost_tracker.total_cost_usd > 0.0


def test_modal_runner_repair_execution():
    with open(CONFIGS_DIR / "scene1_pick_place.json") as f:
        sg = SceneGraph.model_validate(json.load(f))

    runner = ModalSandboxRunner(max_budget_usd=2.00)
    # Inject broken reward to exercise repair loop
    result = runner.run_repair(
        sg,
        max_tries=2,
        reward_fn_override=lambda m, d, s: (0.0, False, {}),
    )

    assert result.success
    assert result.attempts == 2
    assert runner.cost_tracker.total_cost_usd > 0.0


def test_modal_runner_budget_exceeded():
    with open(CONFIGS_DIR / "scene1_pick_place.json") as f:
        sg = SceneGraph.model_validate(json.load(f))

    # Pre-seed tracker with cost exceeding cap
    expensive_tracker = CostTracker()
    expensive_tracker.log_compute(core_seconds=100000.0, gib_seconds=100000.0)

    runner = ModalSandboxRunner(max_budget_usd=1.00, cost_tracker=expensive_tracker)

    with pytest.raises(BudgetExceededError) as exc_info:
        runner.run_validation(sg, run_gate5=False)

    assert "Budget cap of $1.00 exceeded" in str(exc_info.value)


def test_modal_runner_micro_budget_enforcement():
    with open(CONFIGS_DIR / "scene1_pick_place.json") as f:
        sg = SceneGraph.model_validate(json.load(f))

    # Micro budget of $0.0000001 will breach during or after validation
    runner = ModalSandboxRunner(max_budget_usd=1e-8)

    with pytest.raises(BudgetExceededError) as exc_info:
        runner.run_validation(sg, run_gate5=False)

    assert "Budget cap of $0.00 exceeded" in str(exc_info.value)

