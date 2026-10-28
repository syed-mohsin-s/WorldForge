"""WorldForge 3D Lifter — Stage 2 of the front-end perception pipeline.

Lifts 2D SAM mask tracks and monocular depth into metric 3D primitives:
1. Derives pinhole intrinsics and camera extrinsics from ``CameraSpec``.
2. Fits the tabletop plane via RANSAC on background pixels.
3. Back-projects eroded object masks to compute 3D bounding sizes, centroids,
   and snaps resting objects to ``z_table + 0.001``.
4. Analyzes multi-frame 3D centroid trajectories using SVD line fitting to
   infer joint types (slide vs hinge) and articulation axes.
5. Mode B (VLM fallback): Annotates Frame 0 with bounding boxes, contours,
   labels, and an A4 metric scale grid for visual reasoning when depth is absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np

from src.perception.sam_tracker import ObjectMaskTrack, SamTrackingResult
from src.schema.scene_graph import CameraSpec


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class LiftedObject:
    """3D geometry and motion attributes estimated for one object."""

    id: str
    label: str
    primitive: Literal["box", "cylinder", "sphere"]
    size: list[float]  # MuJoCo half-sizes: box->[hx, hy, hz], cylinder->[r, hz]
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    joint_type: Literal["free", "hinge", "slide", "fixed"] = "free"
    joint_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    joint_range: tuple[float, float] | None = None
    joint_damping: float = 0.5
    is_prior_guess: bool = False
    parent_id: str | None = None
    is_static: bool = False
    trajectory_3d: list[tuple[float, float, float]] = field(default_factory=list)


@dataclass
class LiftedSceneGeometry:
    """Complete 3D metric scene geometry recovered from perception."""

    scene_id: str
    table_z: float
    camera: CameraSpec
    objects: list[LiftedObject] = field(default_factory=list)
    estimated_motion_type: str | None = None
    trajectory_goal: tuple[float, float, float] | None = None
    annotated_rgb_frame: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Camera Math Utilities
# ---------------------------------------------------------------------------

def derive_camera_intrinsics(camera: CameraSpec) -> tuple[float, float, float, float]:
    """Derive (fx, fy, cx, cy) from CameraSpec."""
    w, h = camera.resolution
    fovy_rad = math.radians(camera.fovy)
    fy = (h / 2.0) / math.tan(fovy_rad / 2.0)
    fx = fy  # Square pixels
    cx = w / 2.0
    cy = h / 2.0
    return fx, fy, cx, cy


def derive_camera_extrinsics(camera: CameraSpec) -> tuple[np.ndarray, np.ndarray]:
    """Derive camera rotation matrix (3x3) and translation vector (3,) from CameraSpec.

    Converts from OpenGL camera coordinate convention (used by MuJoCo) to world frame:
      Column 0: Camera X (right)
      Column 1: Camera Y (up)
      Column 2: Camera Z (backward / -forward)
    """
    pos = np.array(camera.pos, dtype=np.float64)
    lookat = np.array(camera.lookat, dtype=np.float64)

    forward = lookat - pos
    norm_f = np.linalg.norm(forward)
    forward = forward / max(norm_f, 1e-8)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    right = np.cross(forward, world_up)
    norm_r = np.linalg.norm(right)
    if norm_r < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / norm_r

    true_up = np.cross(right, forward)
    true_up = true_up / max(np.linalg.norm(true_up), 1e-8)

    # Rotation matrix where columns are right, up, -forward
    R_cam_to_world = np.column_stack([right, true_up, -forward])
    t_cam_to_world = pos
    return R_cam_to_world, t_cam_to_world


def backproject_pixels(
    xs: np.ndarray,
    ys: np.ndarray,
    depths: np.ndarray,
    camera: CameraSpec,
) -> np.ndarray:
    """Back-project 2D pixel coordinates + depths into 3D world coordinates.

    Returns an (N, 3) array of 3D points in the world frame.
    """
    fx, fy, cx, cy = derive_camera_intrinsics(camera)
    R_cam_to_world, t_cam_to_world = derive_camera_extrinsics(camera)

    # OpenGL camera frame coordinates
    x_cam = (xs - cx) * depths / fx
    y_cam = -(ys - cy) * depths / fy  # Image Y down -> OpenGL Y up
    z_cam = -depths                   # Camera looks along -Z

    p_cam = np.vstack([x_cam, y_cam, z_cam])  # (3, N)
    p_world = t_cam_to_world[:, None] + R_cam_to_world @ p_cam  # (3, N)
    return p_world.T  # (N, 3)


# ---------------------------------------------------------------------------
# RANSAC Tabletop Plane Fitting
# ---------------------------------------------------------------------------

def fit_tabletop_plane_ransac(
    points_3d: np.ndarray,
    distance_threshold: float = 0.003,
    max_iterations: int = 1000,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """Fit dominant horizontal plane (tabletop) using RANSAC.

    Returns (z_table, plane_normal).
    """
    if len(points_3d) < 3:
        return 0.0, np.array([0.0, 0.0, 1.0])

    rng = np.random.default_rng(seed)
    best_inliers = 0
    best_normal = np.array([0.0, 0.0, 1.0])
    best_d = 0.0

    n_pts = len(points_3d)
    for _ in range(max_iterations):
        sample_idx = rng.choice(n_pts, size=3, replace=False)
        p1, p2, p3 = points_3d[sample_idx]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm

        # Orient normal towards +Z
        if normal[2] < 0:
            normal = -normal

        # Plane must be predominantly horizontal (normal_z > 0.85)
        if normal[2] < 0.85:
            continue

        d = -float(np.dot(normal, p1))
        distances = np.abs(points_3d @ normal + d)
        inliers = int(np.sum(distances < distance_threshold))

        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal
            best_d = d

    # Refit on best inliers via least squares
    distances = np.abs(points_3d @ best_normal + best_d)
    inlier_mask = distances < distance_threshold
    if np.sum(inlier_mask) >= 3:
        inlier_pts = points_3d[inlier_mask]
        z_table = float(np.mean(inlier_pts[:, 2]))
    else:
        z_table = -best_d / max(best_normal[2], 1e-6)

    return z_table, best_normal


# ---------------------------------------------------------------------------
# 3D Lifting Pipeline
# ---------------------------------------------------------------------------

def lift_scene_to_3d(
    tracking_result: SamTrackingResult,
    camera: CameraSpec,
    frame0_depth: np.ndarray | None = None,
    frame0_rgb: np.ndarray | None = None,
    mode: Literal["depth_ransac", "vlm_fallback"] = "depth_ransac",
    scene_id: str = "lifted_scene",
) -> LiftedSceneGeometry:
    """Lift 2D tracks and depth into full metric 3D scene geometry.

    Parameters
    ----------
    tracking_result:
        SAM 3.1 tracking output across video frames.
    camera:
        Estimated or calibrated pinhole camera spec.
    frame0_depth:
        Float depth map (meters) for Frame 0. Required if mode="depth_ransac".
    frame0_rgb:
        RGB image for Frame 0 (used for VLM visualization fallback).
    mode:
        "depth_ransac" for metric depth lifting, or "vlm_fallback".
    scene_id:
        Identifier string for the scene.

    Returns
    -------
    LiftedSceneGeometry
    """
    if mode == "vlm_fallback" or frame0_depth is None:
        return _lift_scene_vlm_fallback(
            tracking_result, camera, frame0_rgb, scene_id
        )

    w, h = tracking_result.resolution

    # 1. Backproject background pixels to fit tabletop plane
    all_obj_mask = np.zeros((h, w), dtype=bool)
    for track in tracking_result.tracks:
        all_obj_mask |= track.frame0_mask

    bg_mask = ~all_obj_mask
    bg_ys, bg_xs = np.where(bg_mask)
    if len(bg_ys) > 3000:
        step = len(bg_ys) // 3000
        bg_ys = bg_ys[::step]
        bg_xs = bg_xs[::step]

    bg_depths = frame0_depth[bg_ys, bg_xs]
    valid_depth = (bg_depths > 0.1) & (bg_depths < 5.0)
    bg_points = backproject_pixels(
        bg_xs[valid_depth], bg_ys[valid_depth], bg_depths[valid_depth], camera
    )

    table_z, _ = fit_tabletop_plane_ransac(bg_points)

    lifted_objects: list[LiftedObject] = []

    # 2. Lift each object track using eroded mask
    for track in tracking_result.tracks:
        mask = track.frame0_eroded_mask
        if not np.any(mask):
            mask = track.frame0_mask  # Fallback if erosion eliminated all pixels

        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue

        depths = frame0_depth[ys, xs]
        valid = (depths > 0.1) & (depths < 5.0)
        if not np.any(valid):
            continue

        pts = backproject_pixels(xs[valid], ys[valid], depths[valid], camera)

        # Robust extents (5th to 95th percentiles to avoid boundary outliers)
        p5 = np.percentile(pts, 5, axis=0)
        p95 = np.percentile(pts, 95, axis=0)
        extents = p95 - p5

        # Centroid in XY from 3D bounding extents midpoint
        cx_3d = float((p5[0] + p95[0]) / 2.0)
        cy_3d = float((p5[1] + p95[1]) / 2.0)

        # Height determination
        top_z = float(np.percentile(pts[:, 2], 95))
        obj_height = max(top_z - table_z, 0.01)

        # Centroid Z: resting object snapped to table
        cz_3d = table_z + obj_height / 2.0 + 0.001

        # Classify primitive: box vs cylinder
        width = float(extents[0])
        depth_val = float(extents[1])
        aspect_ratio = width / max(depth_val, 1e-4)
        is_cylindrical = 0.75 <= aspect_ratio <= 1.35 and (
            "cylinder" in track.label.lower()
            or "coaster" in track.label.lower()
            or "can" in track.label.lower()
            or "bottle" in track.label.lower()
        )

        if is_cylindrical:
            primitive = "cylinder"
            radius = (width + depth_val) / 4.0
            size = [radius, obj_height / 2.0]
        else:
            primitive = "box"
            size = [max(width / 2.0, 0.005), max(depth_val / 2.0, 0.005), max(obj_height / 2.0, 0.005)]

        # 3. Lift multi-frame 2D centroid trajectory to 3D via ray-plane intersection
        fx, fy, cam_cx, cam_cy = derive_camera_intrinsics(camera)
        R_cam, t_cam = derive_camera_extrinsics(camera)

        traj_3d: list[tuple[float, float, float]] = []
        z_plane = cz_3d
        for c2d in track.centroids_2d:
            # Ray direction in OpenGL camera frame
            ray_cam = np.array([
                (c2d[0] - cam_cx) / fx,
                -(c2d[1] - cam_cy) / fy,
                -1.0,
            ])
            ray_world = R_cam @ ray_cam
            ray_norm = np.linalg.norm(ray_world)
            if ray_norm > 1e-6:
                ray_world = ray_world / ray_norm

            # Intersect ray with horizontal plane at z = z_plane
            if abs(ray_world[2]) > 1e-4:
                s = (z_plane - t_cam[2]) / ray_world[2]
                pt3d = t_cam + s * ray_world
            else:
                pt3d = np.array([cx_3d, cy_3d, cz_3d])
            traj_3d.append((float(pt3d[0]), float(pt3d[1]), float(pt3d[2])))

        # 4. Trajectory analysis: fit joint axis
        joint_type: Literal["free", "hinge", "slide", "fixed"] = "free"
        joint_axis = (0.0, 0.0, 1.0)
        is_prior_guess = False
        joint_range: tuple[float, float] | None = None

        P = np.array(traj_3d)
        total_displacement = float(np.linalg.norm(P[-1] - P[0])) if len(P) > 1 else 0.0

        if total_displacement > 0.02:
            # SVD linear axis fit
            Pc = P - np.mean(P, axis=0)
            _, S, Vt = np.linalg.svd(Pc)
            axis = Vt[0]
            if np.dot(P[-1] - P[0], axis) < 0:
                axis = -axis

            collinearity = S[1] / max(S[0], 1e-6) if len(S) > 1 else 0.0

            if collinearity < 0.35:
                # High collinearity -> slide joint or linear push
                if "drawer" in track.label.lower() or "slide" in track.label.lower():
                    joint_type = "slide"
                    joint_axis = (float(axis[0]), float(axis[1]), float(axis[2]))
                    joint_range = (0.0, total_displacement * 1.2)
                else:
                    joint_type = "free"
            else:
                # Curved motion -> hinge joint
                joint_type = "hinge"
                normal = np.cross(Vt[0], Vt[1])
                normal = normal / max(np.linalg.norm(normal), 1e-6)
                joint_axis = (float(normal[0]), float(normal[1]), float(normal[2]))
                joint_range = (0.0, math.pi / 2.0)
        else:
            # Static or unmoved
            if "cabinet" in track.label.lower() or "coaster" in track.label.lower() or "marker" in track.label.lower():
                joint_type = "fixed"
            elif "drawer" in track.label.lower() or "door" in track.label.lower():
                joint_type = "slide"
                joint_axis = (0.0, -1.0, 0.0)
                joint_range = (0.0, 0.12)
                is_prior_guess = True
            else:
                joint_type = "free"

        lifted_objects.append(
            LiftedObject(
                id=track.object_id,
                label=track.label,
                primitive=primitive,
                size=size,
                position=(cx_3d, cy_3d, cz_3d),
                joint_type=joint_type,
                joint_axis=joint_axis,
                joint_range=joint_range,
                is_prior_guess=is_prior_guess,
                is_static=(joint_type == "fixed"),
                trajectory_3d=traj_3d,
            )
        )

    # Establish parent-child hierarchies (e.g. handle -> drawer -> cabinet)
    _infer_hierarchies(lifted_objects)

    return LiftedSceneGeometry(
        scene_id=scene_id,
        table_z=table_z,
        camera=camera,
        objects=lifted_objects,
    )


# ---------------------------------------------------------------------------
# Hierarchy Inference
# ---------------------------------------------------------------------------

def _infer_hierarchies(objects: list[LiftedObject]) -> None:
    """Infer articulated parent-child relationships from spatial nesting & labels."""
    cabinet = next((o for o in objects if "cabinet" in o.label.lower()), None)
    drawer = next((o for o in objects if "drawer" in o.label.lower() and "handle" not in o.label.lower()), None)
    handle = next((o for o in objects if "handle" in o.label.lower()), None)

    if drawer and cabinet:
        drawer.parent_id = cabinet.id
        # Relative position
        dx = drawer.position[0] - cabinet.position[0]
        dy = drawer.position[1] - cabinet.position[1]
        dz = drawer.position[2] - cabinet.position[2]
        drawer.position = (dx, dy, dz)

    if handle and drawer:
        handle.parent_id = drawer.id
        hx = handle.position[0] - drawer.position[0]
        hy = handle.position[1] - drawer.position[1]
        hz = handle.position[2] - drawer.position[2]
        handle.position = (hx, hy, hz)


# ---------------------------------------------------------------------------
# Mode B: VLM Fallback Annotation
# ---------------------------------------------------------------------------

def _lift_scene_vlm_fallback(
    tracking_result: SamTrackingResult,
    camera: CameraSpec,
    frame0_rgb: np.ndarray | None,
    scene_id: str,
) -> LiftedSceneGeometry:
    """Annotate Frame 0 with bounding boxes, contours, and metric reference grid."""
    w, h = tracking_result.resolution
    canvas = frame0_rgb.copy() if frame0_rgb is not None else np.zeros((h, w, 3), dtype=np.uint8)

    # Draw metric reference grid (A4 paper scale reference: 210mm x 297mm)
    grid_size = 50  # 50px grid
    for x in range(0, w, grid_size):
        cv2.line(canvas, (x, 0), (x, h), (70, 70, 70), 1)
    for y in range(0, h, grid_size):
        cv2.line(canvas, (0, y), (w, y), (70, 70, 70), 1)

    # Overlay A4 sheet watermark in corner
    cv2.putText(
        canvas,
        "Scale Ref: A4 Sheet (210mm x 297mm)",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    lifted_objects: list[LiftedObject] = []
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    for idx, track in enumerate(tracking_result.tracks):
        color = colors[idx % len(colors)]
        xmin, ymin, xmax, ymax = track.bbox_2d

        # Draw bbox
        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), color, 2)

        # Draw mask contour
        contours, _ = cv2.findContours(
            track.frame0_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, color, 1)

        # Draw label
        cv2.putText(
            canvas,
            f"{track.object_id}: {track.label}",
            (xmin, max(ymin - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        # Estimated default positions
        lifted_objects.append(
            LiftedObject(
                id=track.object_id,
                label=track.label,
                primitive="box",
                size=[0.025, 0.025, 0.025],
                position=(0.0, 0.0, 0.025),
                joint_type="free",
                is_prior_guess=True,
            )
        )

    return LiftedSceneGeometry(
        scene_id=scene_id,
        table_z=0.0,
        camera=camera,
        objects=lifted_objects,
        annotated_rgb_frame=canvas,
    )
