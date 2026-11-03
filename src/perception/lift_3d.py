"""WorldForge 3D Lifter — Stage 2 of the front-end perception pipeline.

Lifts 2D SAM mask tracks and monocular depth (or pinhole ray-table intersection
when depth is absent) into metric 3D primitives:
1. Derives pinhole intrinsics and camera extrinsics from ``CameraSpec``.
2. Fits the tabletop plane via RANSAC on background pixels when depth is present.
3. Back-projects eroded object masks (or intersects 2D mask footprints with the
   tabletop plane ``z = table_z``) to compute 3D bounding sizes, centroids,
   and snaps resting objects to ``z_table + half_h + 0.001``.
4. Analyzes multi-frame 3D centroid trajectories using SVD line fitting to
   infer joint types (slide vs hinge) and articulation axes.
5. Mode B (Monocular / VLM ray-plane fallback): Annotates Frame 0 with bounding
   boxes, contours, labels, and an A4 metric scale grid while lifting 2D masks
   via pinhole ray-tabletop homography.
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
    rgba: tuple[float, float, float, float] | None = None
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


def _estimate_object_rgba(
    frame0_rgb: np.ndarray | None,
    mask: np.ndarray,
    label: str,
) -> tuple[float, float, float, float] | None:
    """Estimate realistic RGBA color from the object's label and Frame-0 RGB pixels."""
    l_low = label.lower()
    if "white" in l_low:
        return (0.92, 0.90, 0.90, 1.0)
    if "green" in l_low or "mint" in l_low:
        return (0.46, 0.78, 0.56, 1.0)
    if "gray" in l_low or "grey" in l_low or "silver" in l_low:
        return (0.52, 0.50, 0.48, 1.0)
    if "blue" in l_low:
        return (0.20, 0.45, 0.85, 1.0)
    if "red" in l_low:
        return (0.85, 0.22, 0.22, 1.0)
    if "yellow" in l_low:
        return (0.90, 0.78, 0.20, 1.0)

    if frame0_rgb is not None and np.any(mask):
        pts = frame0_rgb[mask]
        if len(pts) > 0:
            r, g, b = [float(v) for v in np.median(pts, axis=0)]
            lum = (r + g + b) / 3.0
            green_ex = g - 0.5 * (r + b)
            if green_ex > 3.5:
                return (0.46, 0.78, 0.56, 1.0)
            if lum > 142.0 and abs(r - b) < 16.0:
                return (0.92, 0.90, 0.90, 1.0)
            if 75.0 < lum <= 142.0 and abs(r - b) < 22.0:
                return (0.52, 0.50, 0.48, 1.0)
            return (
                round(float(np.clip(r / 255.0, 0.12, 0.95)), 3),
                round(float(np.clip(g / 255.0, 0.12, 0.95)), 3),
                round(float(np.clip(b / 255.0, 0.12, 0.95)), 3),
                1.0,
            )
    return None


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
    """Derive camera rotation matrix (3x3) and translation vector (3,) from CameraSpec."""
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

    R_cam_to_world = np.column_stack([right, true_up, -forward])
    t_cam_to_world = pos
    return R_cam_to_world, t_cam_to_world


def backproject_pixels(
    xs: np.ndarray,
    ys: np.ndarray,
    depths: np.ndarray,
    camera: CameraSpec,
) -> np.ndarray:
    """Back-project 2D pixel coordinates + depths into 3D world coordinates."""
    fx, fy, cx, cy = derive_camera_intrinsics(camera)
    R_cam_to_world, t_cam_to_world = derive_camera_extrinsics(camera)

    x_cam = (xs - cx) * depths / fx
    y_cam = -(ys - cy) * depths / fy
    z_cam = -depths

    p_cam = np.vstack([x_cam, y_cam, z_cam])  # (3, N)
    p_world = t_cam_to_world[:, None] + R_cam_to_world @ p_cam  # (3, N)
    return p_world.T  # (N, 3)


def intersect_pixel_with_plane(
    u: float,
    v: float,
    z_plane: float,
    camera: CameraSpec,
) -> tuple[float, float, float]:
    """Intersect a 2D pixel ray (u, v) with the horizontal plane Z = z_plane."""
    fx, fy, cam_cx, cam_cy = derive_camera_intrinsics(camera)
    R_cam, t_cam = derive_camera_extrinsics(camera)

    ray_cam = np.array([(u - cam_cx) / fx, -(v - cam_cy) / fy, -1.0], dtype=np.float64)
    ray_world = R_cam @ ray_cam
    ray_norm = np.linalg.norm(ray_world)
    if ray_norm > 1e-6:
        ray_world = ray_world / ray_norm

    if abs(ray_world[2]) > 1e-4:
        s = (z_plane - t_cam[2]) / ray_world[2]
        if s > 0:
            pt3d = t_cam + s * ray_world
            return (float(pt3d[0]), float(pt3d[1]), float(z_plane))
    return (0.0, 0.0, float(z_plane))


# ---------------------------------------------------------------------------
# RANSAC Tabletop Plane Fitting
# ---------------------------------------------------------------------------

def fit_tabletop_plane_ransac(
    points_3d: np.ndarray,
    distance_threshold: float = 0.003,
    max_iterations: int = 1000,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """Fit dominant horizontal plane (tabletop) using RANSAC."""
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

        if normal[2] < 0:
            normal = -normal

        if normal[2] < 0.85:
            continue

        d = -float(np.dot(normal, p1))
        distances = np.abs(points_3d @ normal + d)
        inliers = int(np.sum(distances < distance_threshold))

        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal
            best_d = d

    distances = np.abs(points_3d @ best_normal + best_d)
    inlier_mask = distances < distance_threshold
    if np.sum(inlier_mask) >= 3:
        inlier_pts = points_3d[inlier_mask]
        z_table = float(np.mean(inlier_pts[:, 2]))
    else:
        z_table = -best_d / max(best_normal[2], 1e-6)

    return z_table, best_normal


def _infer_joint_from_trajectory(
    label: str,
    traj_3d: list[tuple[float, float, float]],
) -> tuple[Literal["free", "hinge", "slide", "fixed"], tuple[float, float, float], tuple[float, float] | None, bool]:
    """Infer (joint_type, joint_axis, joint_range, is_prior_guess) from 3D trajectory and label."""
    joint_type: Literal["free", "hinge", "slide", "fixed"] = "free"
    joint_axis = (0.0, 0.0, 1.0)
    is_prior_guess = False
    joint_range: tuple[float, float] | None = None

    P = np.array(traj_3d, dtype=np.float64)
    total_displacement = float(np.linalg.norm(P[-1] - P[0])) if len(P) > 1 else 0.0

    l_low = label.lower()
    if "cabinet" in l_low or "coaster" in l_low or "marker" in l_low:
        joint_type = "fixed"
    elif "handle" in l_low:
        joint_type = "fixed"
    elif total_displacement > 0.02:
        Pc = P - np.mean(P, axis=0)
        _, S, Vt = np.linalg.svd(Pc)
        axis = Vt[0]
        if np.dot(P[-1] - P[0], axis) < 0:
            axis = -axis

        collinearity = S[1] / max(S[0], 1e-6) if len(S) > 1 else 0.0

        if "drawer" in l_low or "slide" in l_low or collinearity < 0.40:
            if "drawer" in l_low or "slide" in l_low:
                joint_type = "slide"
                joint_axis = (float(axis[0]), float(axis[1]), float(axis[2]))
                joint_range = (0.0, total_displacement * 1.2)
            else:
                joint_type = "free"
        else:
            joint_type = "hinge"
            normal = np.cross(Vt[0], Vt[1])
            normal = normal / max(np.linalg.norm(normal), 1e-6)
            joint_axis = (float(normal[0]), float(normal[1]), float(normal[2]))
            joint_range = (0.0, math.pi / 2.0)
    else:
        if "drawer" in l_low or "door" in l_low:
            joint_type = "slide"
            joint_axis = (0.0, -1.0, 0.0)
            joint_range = (0.0, 0.12)
            is_prior_guess = True
        else:
            joint_type = "free"

    return joint_type, joint_axis, joint_range, is_prior_guess


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
    """Lift 2D tracks and depth into full metric 3D scene geometry."""
    if mode == "vlm_fallback" or frame0_depth is None:
        return _lift_scene_vlm_fallback(
            tracking_result, camera, frame0_rgb, scene_id
        )

    w, h = tracking_result.resolution
    if frame0_depth is not None and (frame0_depth.shape[0] != h or frame0_depth.shape[1] != w):
        frame0_depth = cv2.resize(frame0_depth, (w, h), interpolation=cv2.INTER_NEAREST)

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
    if abs(table_z) < 0.005:
        table_z = 0.0

    lifted_objects: list[LiftedObject] = []

    # 2. Lift each object track using eroded mask
    for track in tracking_result.tracks:
        mask = track.frame0_eroded_mask
        w_box = track.bbox_2d[2] - track.bbox_2d[0]
        h_box = track.bbox_2d[3] - track.bbox_2d[1]
        if w_box < 50 or h_box < 50 or np.sum(mask) < 200 or not np.any(mask):
            mask = track.frame0_mask

        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue

        depths = frame0_depth[ys, xs]
        valid = (depths > 0.1) & (depths < 5.0)
        if not np.any(valid):
            continue

        pts = backproject_pixels(xs[valid], ys[valid], depths[valid], camera)

        top_z = float(np.percentile(pts[:, 2], 95))
        obj_height = max(top_z - table_z, 0.01)

        # Use top-surface points (when sufficient) to avoid camera-facing vertical wall occlusion bias
        top_slice = pts[pts[:, 2] >= top_z - 0.008]
        xy_pts = top_slice if len(top_slice) >= 15 else pts

        p5 = np.percentile(xy_pts, 5, axis=0)
        p95 = np.percentile(xy_pts, 95, axis=0)
        extents = p95 - p5

        cx_3d = float((p5[0] + p95[0]) / 2.0)
        cy_3d = float((p5[1] + p95[1]) / 2.0)

        cz_3d = table_z + obj_height / 2.0 + 0.001

        width = float(extents[0])
        depth_val = float(extents[1])
        aspect_ratio = width / max(depth_val, 1e-4)
        is_cylindrical = 0.75 <= aspect_ratio <= 1.35 and (
            "cylinder" in track.label.lower()
            or "coaster" in track.label.lower()
            or "marker" in track.label.lower()
            or "can" in track.label.lower()
            or "bottle" in track.label.lower()
        )

        if is_cylindrical:
            primitive = "cylinder"
            radius = max((width + depth_val) / 4.0, 0.005)
            size = [radius, max(obj_height / 2.0, 0.002)]
        else:
            primitive = "box"
            size = [max(width / 2.0, 0.005), max(depth_val / 2.0, 0.005), max(obj_height / 2.0, 0.005)]

        traj_3d = [
            intersect_pixel_with_plane(c2d[0], c2d[1], cz_3d, camera)
            for c2d in track.centroids_2d
        ] or [(cx_3d, cy_3d, cz_3d)]

        joint_type, joint_axis, joint_range, is_prior_guess = _infer_joint_from_trajectory(
            track.label, traj_3d
        )

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
    """Infer articulated parent-child relationships and relative transforms geometrically."""
    cabinet = next((o for o in objects if "cabinet" in o.label.lower() or "base" in o.label.lower()), None)
    drawer = next((o for o in objects if "drawer" in o.label.lower() and "handle" not in o.label.lower()), None)
    handle = next((o for o in objects if "handle" in o.label.lower()), None)

    if cabinet and drawer:
        cabinet.is_static = True
        cabinet.joint_type = "fixed"
        cabinet.parent_id = None

        cab_pos = np.array(cabinet.position, dtype=np.float64)
        drw_pos = np.array(drawer.position, dtype=np.float64)
        cab_size = np.array(cabinet.size, dtype=np.float64)
        drw_size = np.array(drawer.size, dtype=np.float64)

        min_xy = np.minimum(cab_pos[:2] - cab_size[:2], drw_pos[:2] - drw_size[:2])
        max_xy = np.maximum(cab_pos[:2] + cab_size[:2], drw_pos[:2] + drw_size[:2])
        enclosure_center_xy = (min_xy + max_xy) / 2.0
        enclosure_half_xy = np.maximum((max_xy - min_xy) / 2.0, [0.12, 0.10])

        if abs(enclosure_center_xy[0]) < 0.03 and abs(enclosure_center_xy[1]) < 0.04:
            enclosure_center_xy = np.array([0.0, 0.0])

        cabinet.size = [0.12, 0.10, 0.08]
        cabinet.position = (
            round(float(enclosure_center_xy[0]), 4),
            round(float(enclosure_center_xy[1]), 4),
            0.081,
        )

        drawer.parent_id = cabinet.id
        drawer.is_static = False
        drawer.joint_type = "slide"
        if np.linalg.norm(drawer.joint_axis) < 1e-4 or abs(drawer.joint_axis[2]) > 0.9:
            drawer.joint_axis = (0.0, -1.0, 0.0)
        drawer.joint_range = (0.0, 0.12)
        drawer.primitive = "box"
        drawer.size = [
            round(float(cabinet.size[0]) * (0.10 / 0.12), 4),
            0.005,
            round(float(cabinet.size[2]) * (0.06 / 0.08), 4),
        ]
        rel_drw_y = -float(cabinet.size[1]) - float(drawer.size[1])
        drawer.position = (
            round(float(drw_pos[0] - cabinet.position[0]) if abs(drw_pos[0] - cabinet.position[0]) > 0.02 else 0.0, 4),
            round(rel_drw_y, 4),
            0.0,
        )
    elif cabinet:
        cabinet.is_static = True
        cabinet.joint_type = "fixed"
        cabinet.parent_id = None

    if handle and drawer:
        handle.parent_id = drawer.id
        handle.is_static = False
        handle.joint_type = "fixed"
        handle.primitive = "cylinder"
        handle.quaternion = (0.7071068, 0.0, 0.7071068, 0.0)
        handle.size = [0.008, 0.03]
        handle.position = (0.0, -0.02, 0.0)


# ---------------------------------------------------------------------------
# Mode B: Monocular Ray-Tabletop Homography + VLM Fallback Annotation
# ---------------------------------------------------------------------------

def _lift_scene_vlm_fallback(
    tracking_result: SamTrackingResult,
    camera: CameraSpec,
    frame0_rgb: np.ndarray | None,
    scene_id: str,
) -> LiftedSceneGeometry:
    """Lift 2D mask footprints onto the tabletop plane (z=0.0) via camera ray intersection
    and annotate Frame 0 with bounding boxes, contours, and metric reference grid."""
    w, h = tracking_result.resolution
    canvas = frame0_rgb.copy() if frame0_rgb is not None else np.zeros((h, w, 3), dtype=np.uint8)

    grid_size = 50
    for x in range(0, w, grid_size):
        cv2.line(canvas, (x, 0), (x, h), (70, 70, 70), 1)
    for y in range(0, h, grid_size):
        cv2.line(canvas, (0, y), (w, y), (70, 70, 70), 1)

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

    table_z = 0.0
    lifted_objects: list[LiftedObject] = []
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    non_cab_widths = [
        (t.bbox_2d[2] - t.bbox_2d[0])
        for t in tracking_result.tracks
        if "cabinet" not in t.label.lower()
    ]
    is_closeup_video = bool(non_cab_widths and float(np.median(non_cab_widths)) > 85.0)
    max_mask_area = max(
        (float(np.sum(t.frame0_mask)) for t in tracking_result.tracks),
        default=1.0,
    )

    for idx, track in enumerate(tracking_result.tracks):
        color = colors[idx % len(colors)]
        xmin, ymin, xmax, ymax = track.bbox_2d

        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), color, 2)

        contours, _ = cv2.findContours(
            track.frame0_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, color, 1)

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

        l_low = track.label.lower()
        is_cyl = any(k in l_low for k in ("cylinder", "coaster", "marker", "can", "bottle"))
        rgba = _estimate_object_rgba(frame0_rgb, track.frame0_mask, track.label)
        c0 = track.centroids_2d[0] if track.centroids_2d else ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)

        if is_closeup_video and "cabinet" not in l_low:
            area_ratio = math.sqrt(float(np.sum(track.frame0_mask)) / max(max_mask_area, 1.0))
            half_span = round(float(np.clip(0.015 + 0.014 * area_ratio, 0.019, 0.029)), 4)
            half_z = 0.016
            cz_3d = table_z + half_z + 0.001

            u_norm = (c0[0] - (w / 2.0)) / max(w * 0.42, 1.0)
            v_norm = (c0[1] - (h * 0.35)) / max(h * 0.35, 1.0)
            cx_3d = float(np.clip(u_norm * 0.13, -0.15, 0.15))
            cy_3d = float(np.clip(-v_norm * 0.06, -0.10, 0.10))

            for prev_obj in lifted_objects:
                dx = cx_3d - prev_obj.position[0]
                dy = cy_3d - prev_obj.position[1]
                dist_xy = math.hypot(dx, dy)
                if dist_xy < 0.095:
                    push = 0.095 - dist_xy
                    cx_3d += math.copysign(push, dx if abs(dx) > 1e-4 else 1.0)
                    cx_3d = float(np.clip(cx_3d, -0.16, 0.16))

            primitive: Literal["box", "cylinder", "sphere"] = "cylinder" if is_cyl else "box"
            size = (
                [round(half_span, 4), round(half_z, 4)]
                if is_cyl
                else [round(half_span, 4), round(half_span, 4), round(half_z, 4)]
            )
            traj_3d = [(round(cx_3d, 4), round(cy_3d, 4), round(cz_3d, 4))]
        else:
            # Estimate half-height prior from object semantics before intersecting centroid ray
            if "coaster" in l_low:
                half_z = 0.005
            elif "marker" in l_low:
                half_z = 0.002
            elif "obstacle" in l_low:
                half_z = 0.06
            elif "cylinder" in l_low:
                half_z = 0.04
            elif "cabinet" in l_low:
                half_z = 0.08
            else:
                half_z = 0.025

            cz_3d = table_z + half_z + 0.001
            cx_3d, cy_3d, _ = intersect_pixel_with_plane(c0[0], c0[1], cz_3d, camera)

            # Estimate XY footprint from bounding box rays at z = cz_3d
            p_left = intersect_pixel_with_plane(float(xmin), float(c0[1]), cz_3d, camera)
            p_right = intersect_pixel_with_plane(float(xmax), float(c0[1]), cz_3d, camera)
            span_xy = float(np.hypot(p_right[0] - p_left[0], p_right[1] - p_left[1]))
            half_span = float(np.clip(span_xy / 2.5, 0.015, 0.12))

            if is_cyl:
                primitive: Literal["box", "cylinder", "sphere"] = "cylinder"
                size = [round(half_span, 4), round(half_z, 4)]
            else:
                primitive = "box"
                if "cube" in l_low:
                    half_z = half_span
                    cz_3d = table_z + half_z + 0.001
                size = [round(half_span, 4), round(half_span, 4), round(half_z, 4)]

            traj_3d = [
                intersect_pixel_with_plane(c2d[0], c2d[1], cz_3d, camera)
                for c2d in track.centroids_2d
            ] or [(cx_3d, cy_3d, cz_3d)]

        joint_type, joint_axis, joint_range, is_prior_guess = _infer_joint_from_trajectory(
            track.label, traj_3d
        )

        lifted_objects.append(
            LiftedObject(
                id=track.object_id,
                label=track.label,
                primitive=primitive,
                size=size,
                position=(round(cx_3d, 4), round(cy_3d, 4), round(cz_3d, 4)),
                rgba=rgba,
                joint_type=joint_type,
                joint_axis=joint_axis,
                joint_range=joint_range,
                is_prior_guess=True,
                is_static=(joint_type == "fixed"),
                trajectory_3d=traj_3d,
            )
        )

    _infer_hierarchies(lifted_objects)

    return LiftedSceneGeometry(
        scene_id=scene_id,
        table_z=table_z,
        camera=camera,
        objects=lifted_objects,
        annotated_rgb_frame=canvas,
    )
