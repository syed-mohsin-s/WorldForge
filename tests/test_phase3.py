"""Phase 3 integration tests — SAM 3.1 Tracker, 3D Lifter, and Muse Spark Reasoner."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import mujoco
import numpy as np
import pytest
import scipy.ndimage

from src.compiler.base_env import WorldForgeEnv
from src.perception.lift_3d import lift_scene_to_3d
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


class TestSAMTracker:
    def test_mask_erosion_removes_boundary_pixels(self) -> None:
        h, w = 200, 200
        mask = np.zeros((h, w), dtype=bool)
        mask[50:150, 50:150] = True

        eroded = scipy.ndimage.binary_erosion(mask, iterations=4)

        assert np.sum(eroded) < np.sum(mask)
        ys, xs = np.where(eroded)
        assert xs.min() == 54
        assert xs.max() == 145
        assert ys.min() == 54
        assert ys.max() == 145

    def test_multiplexed_tracking_up_to_16_objects(self) -> None:
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

    def test_sam31_self_hosted_endpoint_tier2(self, tmp_path: Path) -> None:
        vid_path = tmp_path / "sample_stage1.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (640, 480))
        for _ in range(5):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[180:280, 240:340] = [200, 30, 30]
            writer.write(frame)
        writer.release()

        raw_mask = np.zeros((480, 640), dtype=np.uint8)
        raw_mask[180:280, 240:340] = 255
        ok, buf = cv2.imencode(".png", raw_mask)
        assert ok
        mask_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "tracks": [
                {
                    "object_id": "red_cube",
                    "label": "red cube",
                    "mask_png_b64": mask_b64,
                    "bbox_2d": [240, 180, 340, 280],
                    "centroids_2d": [[290.0, 230.0] for _ in range(5)],
                }
            ]
        }
        mock_resp.raise_for_status.return_value = None

        tracker = SAM31Tracker(endpoint_url="https://mock-unsloth-sam31.trycloudflare.com")

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            result = tracker.track_video(
                str(vid_path), text_prompts=["red plastic cube"], object_ids=["red_cube"]
            )

            mock_post.assert_called_once()
            called_url = mock_post.call_args[0][0] if mock_post.call_args[0] else mock_post.call_args[1].get("url")
            assert called_url == "https://mock-unsloth-sam31.trycloudflare.com/segment"

        assert result.sam_provider_used == "unsloth/sam3.1 (Colab/Kaggle T4 GPU)"
        assert len(result.tracks) == 1
        t = result.tracks[0]
        assert t.object_id == "red_cube"
        assert int(np.sum(t.frame0_eroded_mask)) == 92 * 92

        scene = _load_scene("scene1_pick_place.json")
        depth = np.ones((480, 640), dtype=np.float32) * 1.2
        lifted = lift_scene_to_3d(result, scene.camera, frame0_depth=depth, scene_id="mock_unsloth_scene")

        assert len(lifted.objects) == 1
        assert lifted.objects[0].id == "red_cube"

    def test_sam31_tier1_failure_does_not_lie_about_provider(self) -> None:
        """When META_API_KEY is set but the HTTP call fails, provider badge must reflect offline fallback."""
        tracker = SAM31Tracker(api_key="test_meta_key", endpoint_url=None, local_hf=False)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[180:260, 240:320] = [220, 40, 40]
        with patch("httpx.Client.post", side_effect=RuntimeError("API unreachable")):
            res = tracker.segment_image(img, text_prompts=["red plastic cube"], object_ids=["red_cube"])
        assert res.sam_provider_used == "opencv-hsv (offline)"


class TestLift3D:
    @pytest.mark.parametrize("scene_file", GOLDEN_SCENES)
    def test_table_plane_and_object_positions_within_2_5cm(self, scene_file: str) -> None:
        scene = _load_scene(scene_file)
        env = WorldForgeEnv(scene)

        tracker = SAM31Tracker()
        result, depth = tracker.segment_from_sim_env(env, num_frames=3, execute_oracle=False)
        lifted = lift_scene_to_3d(result, scene.camera, frame0_depth=depth, scene_id=scene.scene_id)

        assert abs(lifted.table_z - scene.table_z) < 0.015

        lifted_dict = {o.id: o for o in lifted.objects}
        for orig_obj in scene.objects:
            if orig_obj.parent_id is not None:
                continue
            assert orig_obj.id in lifted_dict
            lifted_obj = lifted_dict[orig_obj.id]

            gt_bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, orig_obj.id)
            gt_pos = env.data.xpos[gt_bid]

            error = float(np.linalg.norm(np.array(lifted_obj.position) - gt_pos))
            assert error < 0.025

            half_h = lifted_obj.size[2] if lifted_obj.primitive == "box" else lifted_obj.size[1]
            bottom_z = lifted_obj.position[2] - half_h
            assert bottom_z >= lifted.table_z - 0.005

        env.close()

    def test_vlm_fallback_mode_lifts_tabletop_coordinates(self) -> None:
        """Mode B (vlm_fallback) lifts 2D masks onto tabletop via ray-plane homography."""
        scene = _load_scene("scene1_pick_place.json")
        env = WorldForgeEnv(scene)
        tracker = SAM31Tracker()
        result, _ = tracker.segment_from_sim_env(env, num_frames=2)
        rgb = env.render_camera()

        lifted = lift_scene_to_3d(
            result, scene.camera, frame0_rgb=rgb, mode="vlm_fallback"
        )
        assert lifted.annotated_rgb_frame is not None
        assert lifted.annotated_rgb_frame.shape == (480, 640, 3)
        assert len(lifted.objects) == len(result.tracks)
        cube = next(o for o in lifted.objects if o.id == "red_cube")
        assert abs(cube.position[0] - 0.15) < 0.03
        assert abs(cube.position[1] - 0.0) < 0.03
        env.close()


class TestJointAxisEstimation:
    def test_svd_recovers_drawer_slide_axis(self) -> None:
        scene = _load_scene("scene3_articulated_drawer.json")
        env = WorldForgeEnv(scene)

        tracker = SAM31Tracker()
        result, depth = tracker.segment_from_sim_env(env, num_frames=15, execute_oracle=True)
        lifted = lift_scene_to_3d(result, scene.camera, frame0_depth=depth)

        drawer = next((o for o in lifted.objects if "drawer" in o.label.lower()), None)
        assert drawer is not None
        assert drawer.joint_type == "slide"
        axis = np.array(drawer.joint_axis)
        target_axis = np.array([0.0, -1.0, 0.0])

        cos_sim = abs(float(np.dot(axis, target_axis)))
        assert cos_sim > 0.85
        env.close()


class TestEndToEndPerception:
    def test_full_front_to_back_pick_and_place(self) -> None:
        orig_scene = _load_scene("scene1_pick_place.json")
        env = WorldForgeEnv(orig_scene)

        cost_tracker = CostTracker(pricing_tier="standard")

        tracker = SAM31Tracker(cost_tracker=cost_tracker)
        result, depth = tracker.segment_from_sim_env(env, num_frames=5)
        assert len(result.tracks) >= 2

        lifted = lift_scene_to_3d(result, orig_scene.camera, frame0_depth=depth)
        assert lifted.table_z <= 0.01

        reasoner = MuseSparkReasoner(provider="local", cost_tracker=cost_tracker)
        synthesized_scene = reasoner.synthesize_scene_graph(
            lifted,
            task_prompt="Pick up red cube and place on coaster",
            task_type="pick_and_place",
        )

        assert isinstance(synthesized_scene, SceneGraph)
        assert synthesized_scene.task.task_type == "pick_and_place"
        assert len(synthesized_scene.task.waypoints) >= 5

        suite = ValidationSuite(synthesized_scene, verbose=False)
        report = suite.run_all(run_gate5=False)
        assert report.passed, f"Validation failed at gate {report.failed_gate}: {report.diagnostics}"

        # Offline synthesis must log zero API tokens
        assert cost_tracker.total_input_tokens == 0
        assert cost_tracker.total_output_tokens == 0

        env.close()
