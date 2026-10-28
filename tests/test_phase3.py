"""Phase 3 integration tests — SAM 3.1 Tracker, 3D Lifter, and Muse Spark Reasoner.

Test matrix
-----------
1. Mask erosion removes boundary flying-pixel edges and handles multiplexed tracking.
2. RANSAC tabletop plane fitting and 3D lifting on simulated RGB-D frames across all
   3 golden scenes recover object 3D positions within < 2.5cm of ground truth and
   snap resting objects to the table without penetration.
3. Multi-frame motion trajectory SVD accurately recovers the drawer slide joint axis
   [0, -1, 0] (or collinear direction) from multi-frame tracks on scene3_articulated_drawer.
4. End-to-end front-to-back pipeline test:
   Synthetic RGB-D -> SAM31Tracker -> lift_scene_to_3d -> MuseSparkReasoner ->
   SceneGraph -> MJCF Builder -> passes Validation Gates 1–4!
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import scipy.ndimage

from src.compiler.base_env import WorldForgeEnv
from src.perception.lift_3d import (
    backproject_pixels,
    derive_camera_extrinsics,
    derive_camera_intrinsics,
    fit_tabletop_plane_ransac,
    lift_scene_to_3d,
)
from src.perception.sam_tracker import ObjectMaskTrack, SAM31Tracker, SamTrackingResult
from src.perception.spark_reasoner import MuseSparkReasoner
from src.schema.scene_graph import SceneGraph
from src.validation.gates import ValidationSuite
from src.validation.repair_loop import CostTracker

SCENE_DIR = Path(__file__).resolve().parents[1] / "configs" / "scenes"

GOLDEN_SCENES = [
    "scene1_pick_place.json",
    "scene2_push_obstacle.json",
    "scene3_articulated_drawer.json",
]


def _load_scene(name: str) -> SceneGraph:
    path = SCENE_DIR / name
    with open(path) as f:
        data = json.load(f)
    return SceneGraph(**data)


# =====================================================================
# 1. SAM 3.1 Tracker & Mask Erosion Tests
# =====================================================================

class TestSAMTracker:
    def test_mask_erosion_removes_boundary_pixels(self) -> None:
        """Verify binary erosion shrinks mask perimeter by ~4px on each side."""
        # Create a synthetic 100x100 square mask in a 200x200 image
        h, w = 200, 200
        mask = np.zeros((h, w), dtype=bool)
        mask[50:150, 50:150] = True

        eroded = scipy.ndimage.binary_erosion(mask, iterations=4)

        # Eroded mask should be smaller
        assert np.sum(eroded) < np.sum(mask)
        # Expected size: (100 - 2*4) x (100 - 2*4) = 92 x 92
        ys, xs = np.where(eroded)
        assert xs.min() == 54
        assert xs.max() == 145
        assert ys.min() == 54
        assert ys.max() == 145

    def test_multiplexed_tracking_up_to_16_objects(self) -> None:
        """Verify tracking data structure supports multiplexing 16 objects."""
        tracks = []
        for i in range(16):
            m = np.zeros((100, 100), dtype=bool)
            m[10 + i * 4 : 14 + i * 4, 10:20] = True
            e = scipy.ndimage.binary_erosion(m, iterations=1)
            tracks.append(
                ObjectMaskTrack(
                    object_id=f"obj_{i:02d}",
                    label=f"primitive_{i}",
                    frame0_mask=m,
                    frame0_eroded_mask=e,
                    bbox_2d=(10, 10 + i * 4, 20, 14 + i * 4),
                    centroids_2d=[(15.0, 12.0 + i * 4) for _ in range(10)],
                )
            )

        res = SamTrackingResult(
            video_path="mock_video.mp4",
            resolution=(100, 100),
            num_frames=10,
            tracks=tracks,
        )
        assert len(res.tracks) == 16
        assert res.tracks[15].object_id == "obj_15"

    def test_segment_from_sim_env(self) -> None:
        """Verify simulation segmenter extracts valid tracks and depth map."""
        scene = _load_scene("scene1_pick_place.json")
        env = WorldForgeEnv(scene)
        tracker = SAM31Tracker()
        result, depth = tracker.segment_from_sim_env(env, num_frames=5)

        assert depth.shape == (480, 640)
        assert depth.min() > 0.0
        assert len(result.tracks) >= 2
        for t in result.tracks:
            assert np.any(t.frame0_mask)
            assert len(t.centroids_2d) == 5
        env.close()


# =====================================================================
# 2. RANSAC Tabletop & 3D Lifting Tests
# =====================================================================

class TestLift3D:
    @pytest.mark.parametrize("scene_file", GOLDEN_SCENES)
    def test_table_plane_and_object_positions_within_2_5cm(self, scene_file: str) -> None:
        """Lifted object centroids must be within < 2.5cm of ground truth."""
        scene = _load_scene(scene_file)
        env = WorldForgeEnv(scene)

        tracker = SAM31Tracker()
        result, depth = tracker.segment_from_sim_env(env, num_frames=3, execute_oracle=False)
        lifted = lift_scene_to_3d(result, scene.camera, frame0_depth=depth, scene_id=scene.scene_id)

        # Table Z should be close to 0.0
        assert abs(lifted.table_z - scene.table_z) < 0.015, (
            f"Table Z error: {lifted.table_z} vs {scene.table_z}"
        )

        # Compare non-child object positions against ground truth
        lifted_dict = {o.id: o for o in lifted.objects}
        for orig_obj in scene.objects:
            if orig_obj.parent_id is not None:
                continue
            assert orig_obj.id in lifted_dict, f"Missing object {orig_obj.id}"
            lifted_obj = lifted_dict[orig_obj.id]

            gt_bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, orig_obj.id)
            gt_pos = env.data.xpos[gt_bid]

            error = float(np.linalg.norm(np.array(lifted_obj.position) - gt_pos))
            assert error < 0.025, (
                f"3D position error for {orig_obj.id} was {error:.4f}m (>= 0.025m). "
                f"Lifted: {lifted_obj.position}, GT: {gt_pos}"
            )

            # Assert resting objects do not penetrate table
            half_h = lifted_obj.size[2] if lifted_obj.primitive == "box" else lifted_obj.size[1]
            bottom_z = lifted_obj.position[2] - half_h
            assert bottom_z >= lifted.table_z - 0.005, (
                f"Object {orig_obj.id} penetrates table: bottom_z={bottom_z:.4f}, table_z={lifted.table_z:.4f}"
            )

        env.close()

    def test_vlm_fallback_mode_generates_annotated_frame(self) -> None:
        """Mode B (vlm_fallback) produces an annotated frame with bounding boxes & scale grid."""
        scene = _load_scene("scene1_pick_place.json")
        env = WorldForgeEnv(scene)
        tracker = SAM31Tracker()
        result, depth = tracker.segment_from_sim_env(env, num_frames=2)
        rgb = env.render_camera()

        lifted = lift_scene_to_3d(
            result, scene.camera, frame0_rgb=rgb, mode="vlm_fallback"
        )
        assert lifted.annotated_rgb_frame is not None
        assert lifted.annotated_rgb_frame.shape == (480, 640, 3)
        assert len(lifted.objects) == len(result.tracks)
        env.close()


# =====================================================================
# 3. Motion Trajectory & Joint Axis SVD Fitting Tests
# =====================================================================

class TestJointAxisEstimation:
    def test_svd_recovers_drawer_slide_axis(self) -> None:
        """Multi-frame tracking on scene3_articulated_drawer recovers slide axis [0, -1, 0]."""
        scene = _load_scene("scene3_articulated_drawer.json")
        env = WorldForgeEnv(scene)

        # Run waypoint oracle across frames to actuate the drawer
        tracker = SAM31Tracker()
        result, depth = tracker.segment_from_sim_env(env, num_frames=15, execute_oracle=True)
        lifted = lift_scene_to_3d(result, scene.camera, frame0_depth=depth)

        drawer = next((o for o in lifted.objects if "drawer" in o.label.lower()), None)
        assert drawer is not None, "Drawer not found in lifted objects"

        # Check joint type and axis
        assert drawer.joint_type == "slide", f"Expected slide joint, got {drawer.joint_type}"
        axis = np.array(drawer.joint_axis)
        target_axis = np.array([0.0, -1.0, 0.0])

        # Collinear alignment check (cosine similarity > 0.85)
        cos_sim = abs(float(np.dot(axis, target_axis)))
        assert cos_sim > 0.85, (
            f"Recovered axis {axis} not collinear with target {target_axis} (cos_sim={cos_sim:.3f})"
        )
        env.close()


# =====================================================================
# 4. End-to-End Front-to-Back Perception Pipeline Tests
# =====================================================================

class TestEndToEndPerception:
    def test_full_front_to_back_pick_and_place(self) -> None:
        """Synthetic RGB-D -> SAM31Tracker -> 3D lift -> MuseSpark -> SceneGraph -> Gates 1–4."""
        orig_scene = _load_scene("scene1_pick_place.json")
        env = WorldForgeEnv(orig_scene)

        cost_tracker = CostTracker(pricing_tier="standard")

        # Stage 1: SAM 3.1 Tracking
        tracker = SAM31Tracker(cost_tracker=cost_tracker)
        result, depth = tracker.segment_from_sim_env(env, num_frames=5)
        assert len(result.tracks) >= 2

        # Stage 2: 3D Lift
        lifted = lift_scene_to_3d(result, orig_scene.camera, frame0_depth=depth)
        assert lifted.table_z <= 0.01

        # Stage 3: Muse Spark Reasoning & Synthesis
        reasoner = MuseSparkReasoner(cost_tracker=cost_tracker)
        synthesized_scene = reasoner.synthesize_scene_graph(
            lifted,
            task_prompt="Pick up red cube and place on coaster",
            task_type="pick_and_place",
        )

        assert isinstance(synthesized_scene, SceneGraph)
        assert synthesized_scene.task.task_type == "pick_and_place"
        assert len(synthesized_scene.task.waypoints) >= 5

        # Validate synthesized scene through Gates 1–4
        suite = ValidationSuite(synthesized_scene, verbose=False)
        report = suite.run_all(run_gate5=False)
        assert report.passed, f"Validation failed at gate {report.failed_gate}: {report.diagnostics}"

        # Cost accounting verification
        assert cost_tracker.total_input_tokens > 0
        assert cost_tracker.total_output_tokens > 0

        env.close()

