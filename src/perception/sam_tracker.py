"""WorldForge SAM 3.1 Tracker — Stage 1 of the front-end perception pipeline.

Extracts Frame 0 binary masks and multi-frame 2D centroid trajectories.
Supports:
1. Live Meta Model API calls to ``SAM 3.1`` (supporting up to 16 multiplexed objects).
2. Deterministic ``segment_from_sim_env`` helper using MuJoCo's segmentation & depth
   renderers to generate exact ground-truth tracks and depth for offline test validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import scipy.ndimage

from src.compiler.base_env import WorldForgeEnv
from src.validation.repair_loop import CostTracker


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ObjectMaskTrack:
    """Track information for a single detected tabletop object."""

    object_id: str
    label: str
    frame0_mask: np.ndarray  # bool (H, W)
    frame0_eroded_mask: np.ndarray  # bool (H, W), eroded by 4px
    bbox_2d: tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)
    centroids_2d: list[tuple[float, float]]  # (cx, cy) across T frames


@dataclass
class SamTrackingResult:
    """Result of running SAM 3.1 tracking across a video sequence."""

    video_path: str
    resolution: tuple[int, int]  # (width, height)
    num_frames: int
    tracks: list[ObjectMaskTrack] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SAM 3.1 Tracker
# ---------------------------------------------------------------------------

class SAM31Tracker:
    """Front-end tracker interfacing with SAM 3.1 or simulation renderer."""

    def __init__(
        self,
        api_key: str | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("META_API_KEY")
        self.cost_tracker = cost_tracker or CostTracker()

    def track_video(
        self,
        video_path: str,
        prompts: list[str] | None = None,
        max_objects: int = 16,
    ) -> SamTrackingResult:
        """Call SAM 3.1 Meta Model API to segment and track up to 16 objects.

        If ``META_API_KEY`` is available, makes an HTTP call to the Meta
        Model API. Otherwise raises a RuntimeError advising offline fallback.
        """
        if not self.api_key:
            raise RuntimeError(
                "META_API_KEY not set. Use segment_from_sim_env for deterministic offline tracking."
            )

        import httpx

        # Multiplex up to 16 objects in a single API call
        input_tokens = 512 + (len(prompts) if prompts else 1) * 64
        output_tokens = 1024

        # Model API client
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "sam-3.1",
            "video_path": video_path,
            "max_objects": min(max_objects, 16),
            "prompts": prompts or ["tabletop objects"],
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    "https://api.meta.ai/v1/models/sam-3.1/track",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            # Log attempt and raise informative error
            self.cost_tracker.log_api_call(
                input_tokens=input_tokens,
                output_tokens=32,
                model="sam_3.1",
                purpose=f"sam31_track_failed: {exc}",
            )
            raise RuntimeError(f"SAM 3.1 API request failed: {exc}") from exc

        self.cost_tracker.log_api_call(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model="sam_3.1",
            purpose="sam31_video_tracking",
        )

        return self._parse_api_response(data, video_path)

    @staticmethod
    def _parse_api_response(data: dict[str, Any], video_path: str) -> SamTrackingResult:
        res = tuple(data.get("resolution", [640, 480]))
        num_frames = data.get("num_frames", 1)
        tracks: list[ObjectMaskTrack] = []

        for item in data.get("tracks", []):
            mask = np.array(item["frame0_mask"], dtype=bool)
            eroded = scipy.ndimage.binary_erosion(mask, iterations=4)
            tracks.append(
                ObjectMaskTrack(
                    object_id=item["object_id"],
                    label=item.get("label", item["object_id"]),
                    frame0_mask=mask,
                    frame0_eroded_mask=eroded,
                    bbox_2d=tuple(item["bbox_2d"]),
                    centroids_2d=[tuple(c) for c in item["centroids_2d"]],
                )
            )

        return SamTrackingResult(
            video_path=video_path,
            resolution=res,
            num_frames=num_frames,
            tracks=tracks,
        )

    # ------------------------------------------------------------------
    # Deterministic Simulation Helper (Offline Validation)
    # ------------------------------------------------------------------
    def segment_from_sim_env(
        self,
        env: WorldForgeEnv,
        num_frames: int = 15,
        execute_oracle: bool = True,
    ) -> tuple[SamTrackingResult, np.ndarray]:
        """Step the environment and extract exact segmentation, tracks & Frame 0 depth.

        Uses MuJoCo's segmentation and depth renderers. Filters out background,
        tabletop, and robot gripper bodies to extract only scene objects.
        """
        w, h = env.scene.camera.resolution
        renderer = mujoco.Renderer(env.model, height=h, width=w)

        # 1. Reset and capture Frame 0
        env.reset(seed=42)

        # Depth buffer (in meters)
        renderer.enable_depth_rendering()
        renderer.update_scene(env.data, camera="render_cam")
        frame0_depth = renderer.render().copy()

        # Segmentation buffer (geom IDs)
        renderer.enable_segmentation_rendering()
        renderer.update_scene(env.data, camera="render_cam")
        frame0_seg = renderer.render()[:, :, 0].copy()

        # Identify which geoms belong to tabletop objects (exclude world, table, gripper)
        excluded_bodies = {"world", "table", "ee_mocap", "gripper_base", "finger_L", "finger_R"}
        target_objects = [
            obj for obj in env.scene.objects
            if obj.id not in excluded_bodies
        ]

        # Map each object to its geom ID in MuJoCo
        obj_geom_ids: dict[str, int] = {}
        for obj in target_objects:
            gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{obj.id}_geom")
            if gid >= 0:
                obj_geom_ids[obj.id] = gid

        # Compute Frame 0 masks, eroded masks, and bboxes
        frame0_masks: dict[str, np.ndarray] = {}
        frame0_eroded: dict[str, np.ndarray] = {}
        bboxes: dict[str, tuple[int, int, int, int]] = {}
        centroids_history: dict[str, list[tuple[float, float]]] = {
            obj.id: [] for obj in target_objects
        }

        for obj in target_objects:
            gid = obj_geom_ids.get(obj.id, -1)
            mask = (frame0_seg == gid) if gid >= 0 else np.zeros((h, w), dtype=bool)

            # 4-pixel erosion
            eroded = scipy.ndimage.binary_erosion(mask, iterations=4) if np.any(mask) else mask.copy()

            frame0_masks[obj.id] = mask
            frame0_eroded[obj.id] = eroded

            if np.any(mask):
                ys, xs = np.where(mask)
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                c2d = (float(np.mean(xs)), float(np.mean(ys)))
            else:
                bbox = (0, 0, 0, 0)
                c2d = (0.0, 0.0)

            bboxes[obj.id] = bbox
            centroids_history[obj.id].append(c2d)

        # 2. Step through subsequent frames to record 2D centroids
        waypoints = env.scene.task.waypoints if execute_oracle else []
        wp_idx = 0

        for frame_idx in range(1, num_frames):
            if waypoints and wp_idx < len(waypoints):
                wx, wy, wz, g = waypoints[wp_idx]
                err = np.array([wx, wy, wz]) - env.data.mocap_pos[0]
                dist = np.linalg.norm(err)
                if dist < 0.01:
                    wp_idx = min(wp_idx + 1, len(waypoints) - 1)
                    wx, wy, wz, g = waypoints[wp_idx]
                    err = np.array([wx, wy, wz]) - env.data.mocap_pos[0]
                    dist = np.linalg.norm(err)
                d = err / max(dist, 1e-8)
                action = np.array([d[0], d[1], d[2], g], dtype=np.float32)
                env.step(action)
            else:
                env.step(np.zeros(4, dtype=np.float32))

            renderer.enable_segmentation_rendering()
            renderer.update_scene(env.data, camera="render_cam")
            seg = renderer.render()[:, :, 0]

            for obj in target_objects:
                gid = obj_geom_ids.get(obj.id, -1)
                mask = (seg == gid) if gid >= 0 else np.zeros((h, w), dtype=bool)
                if np.any(mask):
                    ys, xs = np.where(mask)
                    c2d = (float(np.mean(xs)), float(np.mean(ys)))
                else:
                    # Keep previous centroid if occluded or out of view
                    c2d = centroids_history[obj.id][-1]
                centroids_history[obj.id].append(c2d)

        renderer.close()

        # Build ObjectMaskTrack objects
        tracks: list[ObjectMaskTrack] = []
        for obj in target_objects:
            tracks.append(
                ObjectMaskTrack(
                    object_id=obj.id,
                    label=obj.label,
                    frame0_mask=frame0_masks[obj.id],
                    frame0_eroded_mask=frame0_eroded[obj.id],
                    bbox_2d=bboxes[obj.id],
                    centroids_2d=centroids_history[obj.id],
                )
            )

        result = SamTrackingResult(
            video_path="sim://rehearsed_tabletop_rollout",
            resolution=(w, h),
            num_frames=num_frames,
            tracks=tracks,
        )
        return result, frame0_depth

