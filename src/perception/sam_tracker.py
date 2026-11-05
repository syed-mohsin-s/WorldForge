"""WorldForge SAM 3.1 Video Segmentation & Tracking — Stage 1 of perception.

Provides:
1. ``SAM31Tracker``: 4-tier execution hierarchy:
   - Tier 1: Hosted Meta SAM 3.1 API (META_API_KEY + META_SAM_API_URL)
   - Tier 2: Colab / Kaggle / Modal Self-Hosted unsloth/sam3.1 GPU Tunnel (SAM31_ENDPOINT_URL)
   - Tier 3: Local CPU unsloth/sam3.1 via Hugging Face transformers (SAM31_LOCAL_HF=true)
   - Tier 4: Instant Offline Sim / Depth-Seg / OpenCV Lab+HSV Multi-Frame Tracker
2. 4-pixel morphological boundary erosion (``scipy.ndimage.binary_erosion``)
   before depth back-projection to eliminate flying boundary pixels.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
from scipy.ndimage import binary_erosion

from src.schema.scene_graph import SceneGraph
from src.validation.repair_loop import CostTracker

logger = logging.getLogger(__name__)

# Module-level singletons for Tier 3 Local HF SAM 3.1 model
_LOCAL_SAM3_PROCESSOR: Any = None
_LOCAL_SAM3_MODEL: Any = None

_COLOR_SYNONYMS: dict[str, str] = {
    "grey": "gray",
    "gray": "gray",
    "silver": "gray",
    "metallic": "gray",
    "green": "green",
    "mint": "green",
    "white": "white",
    "red": "red",
    "blue": "blue",
    "yellow": "yellow",
    "black": "black",
    "orange": "orange",
    "purple": "purple",
    "pink": "pink",
    "brown": "brown",
    "cyan": "cyan",
}


def extract_prompts_from_user_intent(
    task_prompt: str | None,
    fallback_prompts: list[str],
    is_stack_intent: bool = False,
) -> list[str]:
    """Extract natural-language visual object prompts from user intent prompt for SAM 3.1."""
    if not task_prompt or not task_prompt.strip():
        if is_stack_intent:
            return ["left object", "center object", "right object"]
        return fallback_prompts

    tokens = re.findall(r"[a-zA-Z]+", task_prompt.lower())
    extracted_colors: list[str] = []
    for tok in tokens:
        canon = _COLOR_SYNONYMS.get(tok)
        if canon and canon not in extracted_colors:
            extracted_colors.append(canon)

    if len(extracted_colors) >= 2:
        return [f"{c} object" for c in extracted_colors]

    if is_stack_intent:
        if len(extracted_colors) == 1:
            c0 = extracted_colors[0]
            return [f"{c0} object", "center object", "right object"]
        return ["left object", "center object", "right object"]

    if len(extracted_colors) == 1:
        return [f"{extracted_colors[0]} object"] + [
            p for p in fallback_prompts if extracted_colors[0] not in p.lower()
        ]

    return fallback_prompts


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute binary mask Intersection-over-Union."""
    inter = int(np.sum(mask_a & mask_b))
    if inter == 0:
        return 0.0
    union = int(np.sum(mask_a | mask_b))
    return float(inter) / max(float(union), 1.0)


def _get_local_sam3_singleton() -> tuple[Any, Any]:
    """Lazy-load unsloth/sam3.1 model and processor onto CPU as a singleton."""
    global _LOCAL_SAM3_PROCESSOR, _LOCAL_SAM3_MODEL
    if _LOCAL_SAM3_PROCESSOR is not None and _LOCAL_SAM3_MODEL is not None:
        return _LOCAL_SAM3_PROCESSOR, _LOCAL_SAM3_MODEL

    import torch

    num_threads = int(os.environ.get("TORCH_NUM_THREADS", min(4, os.cpu_count() or 1)))
    torch.set_num_threads(max(1, num_threads))

    proc_cls = None
    model_cls = None
    try:
        from transformers import Sam3Model, Sam3Processor  # type: ignore

        proc_cls, model_cls = Sam3Processor, Sam3Model
    except ImportError:
        try:
            from transformers import SamModel, SamProcessor  # type: ignore

            proc_cls, model_cls = SamProcessor, SamModel
        except ImportError:
            from transformers import AutoModel, AutoProcessor  # type: ignore

            proc_cls, model_cls = AutoProcessor, AutoModel

    processor = proc_cls.from_pretrained("unsloth/sam3.1")
    model = model_cls.from_pretrained("unsloth/sam3.1", torch_dtype=torch.float32)
    model.to("cpu")
    model.eval()

    _LOCAL_SAM3_PROCESSOR = processor
    _LOCAL_SAM3_MODEL = model
    return _LOCAL_SAM3_PROCESSOR, _LOCAL_SAM3_MODEL


@dataclass
class ObjectMaskTrack:
    """Segmentation and multi-frame tracking data for a single object."""

    object_id: str
    label: str
    frame0_mask: np.ndarray          # (H, W) bool raw mask at Frame 0
    frame0_eroded_mask: np.ndarray   # (H, W) bool 4px eroded mask for depth lifting
    bbox_2d: tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax) at Frame 0
    centroids_2d: list[tuple[float, float]] = field(default_factory=list)  # [(cx, cy), ...] across frames
    masks_over_time: list[np.ndarray] = field(default_factory=list)


@dataclass
class SamTrackingResult:
    """Complete SAM 3.1 segmentation and tracking output for a video."""

    scene_id: str = "video_scene"
    resolution: tuple[int, int] = (640, 480)
    num_frames: int = 1
    tracks: list[ObjectMaskTrack] = field(default_factory=list)
    frame0_rgb: np.ndarray | None = None
    frame0_depth: np.ndarray | None = None
    video_path: str = ""
    sam_provider_used: str = "sim-depth-seg (offline)"

    def __post_init__(self) -> None:
        if self.video_path and (not self.scene_id or self.scene_id == "video_scene"):
            self.scene_id = Path(self.video_path).stem


def erode_mask(mask: np.ndarray, iterations: int = 4) -> np.ndarray:
    """Apply morphological binary erosion by ``iterations`` pixels."""
    if not np.any(mask):
        return mask.copy()
    eroded = binary_erosion(mask, iterations=iterations)
    if not np.any(eroded):
        return mask.copy()
    return eroded.astype(bool)


def compute_mask_centroid_and_bbox(
    mask: np.ndarray,
) -> tuple[tuple[float, float], tuple[int, int, int, int]]:
    """Compute (cx, cy) and (xmin, ymin, xmax, ymax) from a binary mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0.0, 0.0), (0, 0, 0, 0)
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    bbox = (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys)))
    return (cx, cy), bbox


def _resolve_object_id(
    raw_id: str | None,
    label: str,
    idx: int,
    prompts: list[str],
    object_ids: list[str] | None,
) -> str:
    """Resolve a track's canonical object_id from explicit object_ids or prompt mapping."""
    if object_ids:
        # 1. Exact match in object_ids
        if raw_id and raw_id in object_ids:
            return raw_id
        # 2. Match via prompt index
        if idx < len(object_ids) and (not raw_id or raw_id == label.lower().replace(" ", "_")):
            return object_ids[idx]
        # 3. Match via prompt text lookup
        label_norm = label.strip().lower()
        for p_idx, p_text in enumerate(prompts):
            if p_idx < len(object_ids) and p_text.strip().lower() == label_norm:
                return object_ids[p_idx]
        # 4. Token overlap match against object_ids
        cand_tokens = set(re.findall(r"[a-z0-9]+", f"{raw_id or ''} {label}".lower()))
        best_oid = None
        best_score = 0
        for oid in object_ids:
            oid_tokens = set(oid.lower().split("_"))
            overlap = len(cand_tokens & oid_tokens)
            if overlap > best_score:
                best_score = overlap
                best_oid = oid
        if best_oid and best_score > 0:
            return best_oid

    if raw_id:
        return raw_id
    clean_lbl = re.sub(r"\bgrey\b", "gray", label.lower())
    return re.sub(r"[^a-z0-9_]+", "_", clean_lbl).strip("_") or f"object_{idx}"


def _deduplicate_tracks(
    tracks: list[ObjectMaskTrack],
    object_ids: list[str] | None,
) -> list[ObjectMaskTrack]:
    """Deduplicate tracks by object_id and spatial mask overlap."""
    by_id: dict[str, ObjectMaskTrack] = {}
    allowed = set(object_ids) if object_ids else None
    for t in tracks:
        if allowed is not None and t.object_id not in allowed:
            continue
        prev = by_id.get(t.object_id)
        if prev is None or int(np.sum(t.frame0_mask)) > int(np.sum(prev.frame0_mask)):
            by_id[t.object_id] = t
    deduped = list(by_id.values()) if by_id else tracks
    if object_ids is not None:
        return deduped

    # In open discovery mode (object_ids=None), remove duplicate masks with IoU > 0.45
    unique_spatial: list[ObjectMaskTrack] = []
    for t in deduped:
        if any(_mask_iou(t.frame0_mask, u.frame0_mask) > 0.45 for u in unique_spatial):
            continue
        unique_spatial.append(t)
    return unique_spatial


def _propagate_mask_over_frames(
    frame0_mask: np.ndarray,
    frames_rgb: list[np.ndarray],
) -> tuple[list[tuple[float, float]], list[np.ndarray]]:
    """Propagate a Frame-0 binary mask across subsequent RGB frames via optical flow & color matching."""
    c0, bbox0 = compute_mask_centroid_and_bbox(frame0_mask)
    if len(frames_rgb) <= 1 or not np.any(frame0_mask):
        return [c0 for _ in frames_rgb], [frame0_mask for _ in frames_rgb]

    h, w = frame0_mask.shape[:2]
    centroids: list[tuple[float, float]] = [c0]
    masks: list[np.ndarray] = [frame0_mask]

    prev_gray = cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(frame0_mask)
    step = max(1, len(xs) // 80)
    pts = np.column_stack([xs[::step], ys[::step]]).astype(np.float32).reshape(-1, 1, 2)
    cur_mask = frame0_mask.copy()

    for t in range(1, len(frames_rgb)):
        cur_gray = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2GRAY)
        if len(pts) >= 4:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                cur_gray,
                pts,
                None,
                winSize=(15, 15),
                maxLevel=2,
            )
            if next_pts is not None and status is not None:
                good_old = pts[status.flatten() == 1].reshape(-1, 2)
                good_new = next_pts[status.flatten() == 1].reshape(-1, 2)
                if len(good_new) >= 3:
                    dx = float(np.median(good_new[:, 0] - good_old[:, 0]))
                    dy = float(np.median(good_new[:, 1] - good_old[:, 1]))
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    shifted = cv2.warpAffine(
                        cur_mask.astype(np.uint8),
                        M,
                        (w, h),
                        flags=cv2.INTER_NEAREST,
                    ) > 0
                    if np.any(shifted):
                        cur_mask = shifted
                        pts = good_new.reshape(-1, 1, 2)
        c_t, _ = compute_mask_centroid_and_bbox(cur_mask)
        centroids.append(c_t)
        masks.append(cur_mask)
        prev_gray = cur_gray

    return centroids, masks


class SAM31Tracker:
    """SAM 3.1 Video Segmentation and Object Tracker supporting 4-tier hierarchy:
    Tier 1: Hosted Meta SAM 3.1 Model API (META_API_KEY + META_SAM_API_URL)
    Tier 2: Colab / Kaggle / Modal Self-Hosted unsloth/sam3.1 GPU Endpoint (SAM31_ENDPOINT_URL)
    Tier 3: Local CPU unsloth/sam3.1 via HF Transformers (SAM31_LOCAL_HF=true)
    Tier 4: Offline Sim / Depth-Seg / OpenCV Lab+HSV Multi-Frame Tracker
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        endpoint_url: str | None = None,
        local_hf: bool | None = None,
        cost_tracker: CostTracker | None = None,
        erosion_px: int = 4,
    ) -> None:
        raw_key = api_key or os.environ.get("META_API_KEY", "")
        self.api_key = raw_key if (raw_key and not raw_key.startswith("your_")) else None
        self.api_url = api_url or os.environ.get("META_SAM_API_URL")
        self.endpoint_url = endpoint_url or os.environ.get("SAM31_ENDPOINT_URL")
        if local_hf is not None:
            self.local_hf = local_hf
        else:
            self.local_hf = os.environ.get("SAM31_LOCAL_HF", "").lower() in ("true", "1", "yes")
        self.cost_tracker = cost_tracker or CostTracker()
        self.erosion_px = erosion_px

    def has_configured_backend(self) -> bool:
        """Check if any remote or local model tier is configured."""
        if self.api_key and self.api_url:
            return True
        if self.endpoint_url and self.endpoint_url.strip():
            return True
        if self.local_hf:
            return True
        return False

    def track_video(
        self,
        video_path: str,
        text_prompts: list[str] | None = None,
        target_size: tuple[int, int] | None = (640, 480),
        object_ids: list[str] | None = None,
    ) -> SamTrackingResult:
        """Track objects in a video file via the 4-tier SAM 3.1 execution hierarchy."""
        prompts = text_prompts or ["object"]
        frames_rgb = self._decode_video_frames(video_path, max_frames=15, target_size=target_size)
        scene_id = Path(video_path).stem if video_path else "video_scene"

        # Tier 1: Hosted Meta SAM 3.1 API
        if self.api_key:
            try:
                res = self._call_sam_api(video_path, prompts, frames_rgb, object_ids=object_ids)
                if res and res.tracks:
                    res.sam_provider_used = "meta-api/sam-3.1"
                    return res
            except Exception as exc:
                logger.warning("Tier 1 Meta SAM 3.1 API failed: %s; trying Tier 2", exc)

        # Tier 2: Colab / Kaggle / Modal Self-Hosted unsloth/sam3.1 GPU Endpoint
        if self.endpoint_url and self.endpoint_url.strip():
            try:
                res = self._call_self_hosted_endpoint(
                    frames_rgb, prompts, scene_id, video_path, object_ids=object_ids
                )
                if res and res.tracks:
                    res.sam_provider_used = "unsloth/sam3.1 (Colab/Kaggle T4 GPU)"
                    return res
            except Exception as exc:
                logger.warning("Tier 2 unsloth/sam3.1 endpoint failed: %s; trying Tier 3", exc)

        # Tier 3: Local CPU unsloth/sam3.1
        if self.local_hf:
            try:
                res = self._call_local_hf_sam3(
                    frames_rgb, prompts, scene_id, video_path, object_ids=object_ids
                )
                if res and res.tracks:
                    res.sam_provider_used = "unsloth/sam3.1 (Local HF CPU)"
                    return res
            except Exception as exc:
                logger.warning("Tier 3 local HF SAM 3.1 inference failed: %s; falling back to Tier 4", exc)

        # Tier 4: Offline OpenCV Lab+HSV multi-frame contour tracker
        if frames_rgb:
            res = self._segment_opencv_video(scene_id, frames_rgb, prompts, object_ids=object_ids)
            res.video_path = video_path
            res.sam_provider_used = "opencv-hsv (offline)"
            return res

        return SamTrackingResult(
            scene_id=scene_id,
            resolution=target_size or (640, 480),
            num_frames=1,
            tracks=[],
            video_path=video_path,
            sam_provider_used="opencv-hsv (offline)",
        )

    def segment_image(
        self,
        rgb_frame: np.ndarray,
        text_prompts: list[str] | None = None,
        scene_id: str = "image_scene",
        object_ids: list[str] | None = None,
    ) -> SamTrackingResult:
        """Segment a single RGB frame using the 4-tier SAM 3.1 hierarchy."""
        prompts = text_prompts or ["object"]
        frames_rgb = [rgb_frame]

        # Tier 1
        if self.api_key:
            try:
                res = self._call_sam_api("frame0.png", prompts, frames_rgb, object_ids=object_ids)
                if res and res.tracks:
                    res.sam_provider_used = "meta-api/sam-3.1"
                    return res
            except Exception as exc:
                logger.warning("Tier 1 Meta SAM 3.1 API failed: %s; trying Tier 2", exc)

        # Tier 2
        if self.endpoint_url and self.endpoint_url.strip():
            try:
                res = self._call_self_hosted_endpoint(
                    frames_rgb, prompts, scene_id, object_ids=object_ids
                )
                if res and res.tracks:
                    res.sam_provider_used = "unsloth/sam3.1 (Colab/Kaggle T4 GPU)"
                    return res
            except Exception as exc:
                logger.warning("Tier 2 unsloth/sam3.1 endpoint failed: %s; trying Tier 3", exc)

        # Tier 3
        if self.local_hf:
            try:
                res = self._call_local_hf_sam3(
                    frames_rgb, prompts, scene_id, object_ids=object_ids
                )
                if res and res.tracks:
                    res.sam_provider_used = "unsloth/sam3.1 (Local HF CPU)"
                    return res
            except Exception as exc:
                logger.warning("Tier 3 local HF SAM 3.1 failed: %s; falling back to Tier 4", exc)

        # Tier 4
        res = self._segment_opencv_video(scene_id, frames_rgb, prompts, object_ids=object_ids)
        res.sam_provider_used = "opencv-hsv (offline)"
        return res

    @staticmethod
    def _crop_and_resize(rgb: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """Resize to target_size (w, h) while preserving the full horizontal field of view."""
        tw, th = target_size
        h, w = rgb.shape[:2]
        if w == tw and h == th:
            return rgb
        target_aspect = tw / max(th, 1)
        src_aspect = w / max(h, 1)
        # For portrait videos (taller than 4:3), center-crop vertically; for widescreen (16:9),
        # keep the full horizontal width so objects near the left/right edges are never clipped.
        if src_aspect < target_aspect - 0.02:
            new_h = int(round(w / target_aspect))
            y0 = max(0, (h - new_h) // 2)
            rgb = rgb[y0 : y0 + new_h, :]
        return cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)

    @classmethod
    def _decode_video_frames(
        cls,
        video_path: str,
        max_frames: int = 15,
        target_size: tuple[int, int] | None = (640, 480),
    ) -> list[np.ndarray]:
        """Decode evenly spaced RGB frames from an MP4 file while preserving full width."""
        if not video_path or not Path(video_path).exists():
            return []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        raw_frames: list[np.ndarray] = []
        while len(raw_frames) < max(max_frames, 150):
            ret, bgr = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if target_size:
                rgb = cls._crop_and_resize(rgb, target_size)
            raw_frames.append(rgb)
        cap.release()

        if len(raw_frames) <= max_frames:
            return raw_frames
        indices = np.linspace(0, len(raw_frames) - 1, max_frames, dtype=int)
        return [raw_frames[int(i)] for i in indices]

    @staticmethod
    def _extract_sheet_foreground_components(
        rgb: np.ndarray,
        hsv: np.ndarray,
        lab: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Segment objects resting on a bright calibration sheet (e.g. A4 paper) via corridor splitting."""
        h, w = rgb.shape[:2]
        roi_lab = lab[int(h * 0.48) : int(h * 0.74), int(w * 0.18) : int(w * 0.82)]
        roi_hsv = hsv[int(h * 0.48) : int(h * 0.74), int(w * 0.18) : int(w * 0.82)]
        if roi_lab.size == 0:
            return []

        med_hsv = np.median(roi_hsv, axis=(0, 1))
        std_l = float(np.std(roi_lab[:, :, 0]))
        if not (med_hsv[2] > 165.0 and med_hsv[1] < 25.0 and std_l < 18.0):
            return []

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g_minus_rb = rgb[:, :, 1].astype(np.float32) - 0.5 * (
            rgb[:, :, 0].astype(np.float32) + rgb[:, :, 2].astype(np.float32)
        )
        mean_g = cv2.blur(gray, (9, 9))
        local_std = np.sqrt(np.maximum(cv2.blur(gray * gray, (9, 9)) - mean_g * mean_g, 0.0))

        is_paper_or_shadow = (
            (hsv[:, :, 2] > 138.0)
            & (hsv[:, :, 1] < 18.0)
            & (g_minus_rb < 2.2)
            & (local_std < 3.8)
        )
        num_p, p_labels, p_stats, _ = cv2.connectedComponentsWithStats(
            is_paper_or_shadow.astype(np.uint8)
        )
        if num_p <= 1:
            return []

        largest_p = 1 + int(np.argmax(p_stats[1:, cv2.CC_STAT_AREA]))
        if int(p_stats[largest_p, cv2.CC_STAT_AREA]) < (h * w) // 8:
            return []

        paper_cc = p_labels == largest_p
        ys_p, _ = np.where(paper_cc)
        if len(ys_p) < 100:
            return []

        y_min_p = int(np.percentile(ys_p, 1))
        y_max_p = int(np.percentile(ys_p, 98))
        y_obj_top = y_min_p
        y_obj_bot = min(h - 30, int(y_min_p + 0.60 * (y_max_p - y_min_p)))
        band_h = max(1, y_obj_bot - y_obj_top)
        if band_h < 25:
            return []

        paper_col_ratio = np.sum(paper_cc[y_obj_top:y_obj_bot, :], axis=0) / float(band_h)
        corridor_cols = paper_col_ratio > 0.55
        is_dark_cloth = (hsv[:, :, 2] < 110.0) & (
            (np.arange(w)[None, :] < 180) | (np.arange(w)[None, :] > w - 80)
        )

        obj_band = np.zeros((h, w), dtype=bool)
        obj_band[y_obj_top:y_obj_bot, int(w * 0.07) : int(w * 0.96)] = True

        fg = obj_band & (~paper_cc) & (~is_dark_cloth)
        fg[:, corridor_cols] = False

        fg_open = cv2.morphologyEx(
            fg.astype(np.uint8),
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        fg_close = cv2.morphologyEx(
            fg_open,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )

        num_c, c_labels, c_stats, _ = cv2.connectedComponentsWithStats(fg_close)
        sheet_comps: list[dict[str, Any]] = []
        max_area = (h * w) // 4

        for cid in range(1, num_c):
            area = int(c_stats[cid, cv2.CC_STAT_AREA])
            if area < 1200 or area > max_area:
                continue
            mask = c_labels == cid
            ys_m, xs_m = np.where(mask)
            if len(xs_m) == 0:
                continue
            x0_c, x1_c = int(xs_m.min()), int(xs_m.max())
            obj_rgb_med = np.median(rgb[mask], axis=0)
            upper_top = max(35, y_min_p - 32)
            if upper_top < y_obj_top:
                upper_slice = slice(upper_top, y_obj_top)
                col_slice = slice(x0_c, x1_c + 1)
                color_dist = np.linalg.norm(
                    rgb[upper_slice, col_slice].astype(np.float32) - obj_rgb_med[None, None, :],
                    axis=2,
                )
                upper_match = (color_dist < 32.0) & (~paper_cc[upper_slice, col_slice])
                mask[upper_slice, col_slice] |= upper_match

            pts = np.column_stack(np.where(mask))[:, ::-1]
            hull = cv2.convexHull(pts)
            hull_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask, hull, 1)
            hull_bool = hull_mask > 0
            hull_area = int(np.sum(hull_bool))
            if hull_area > max_area:
                continue

            centroid, bbox = compute_mask_centroid_and_bbox(hull_bool)
            mean_rgb = np.mean(rgb[hull_bool], axis=0)
            mean_hsv = np.mean(hsv[hull_bool], axis=0)
            sheet_comps.append(
                {
                    "mask": hull_bool,
                    "area": hull_area,
                    "centroid": centroid,
                    "bbox": bbox,
                    "mean_rgb": mean_rgb,
                    "mean_hsv": mean_hsv,
                }
            )

        if len(sheet_comps) >= 2:
            sheet_comps.sort(key=lambda c: c["area"], reverse=True)
            return sheet_comps
        return []

    @classmethod
    def _extract_foreground_components(
        cls,
        rgb: np.ndarray,
        min_area: int = 25,
    ) -> list[dict[str, Any]]:
        """Extract distinct tabletop foreground objects using adaptive background Lab/HSV differencing."""
        h, w = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float32)

        # First check for objects resting on an A4 calibration sheet / bright workspace mat
        sheet_comps = cls._extract_sheet_foreground_components(rgb, hsv, lab)
        if len(sheet_comps) >= 2:
            return sheet_comps

        # Estimate dominant tabletop background from lower-half border & interior median
        lower_half = slice(int(h * 0.22), int(h * 0.95))
        bg_lab = np.median(lab[lower_half, :], axis=(0, 1))
        bg_hsv = np.median(hsv[lower_half, :], axis=(0, 1))

        # Lab perceptual color distance from dominant tabletop color
        delta_lab = np.linalg.norm(lab - bg_lab[None, None, :], axis=2)
        # Hue/Saturation contrast from tabletop
        dh = np.minimum(np.abs(hsv[:, :, 0] - bg_hsv[0]), 180.0 - np.abs(hsv[:, :, 0] - bg_hsv[0]))
        ds = np.abs(hsv[:, :, 1] - bg_hsv[1])
        dv = np.abs(hsv[:, :, 2] - bg_hsv[2])

        # Also support classic dark-cloth tabletop (low background V < 50)
        if bg_hsv[2] < 50.0:
            fg_mask = (hsv[:, :, 1] > 45) | (hsv[:, :, 2] > 100)
        else:
            fg_mask = (delta_lab > 14.0) | ((dh > 12.0) & (hsv[:, :, 1] > 40.0)) | (ds > 42.0) | (dv > 55.0)
            # Exclude top background horizon strip and neutral gray gripper fingers at top
            fg_mask[: int(h * 0.16), :] = False

        kernel = np.ones((3, 3), np.uint8)
        fg_clean = cv2.morphologyEx(fg_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_CLOSE, kernel)

        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(fg_clean)
        max_area = (h * w) // 4
        components: list[dict[str, Any]] = []
        for comp_id in range(1, num_labels):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if not (min_area <= area <= max_area):
                continue
            mask = labels_im == comp_id
            centroid, bbox = compute_mask_centroid_and_bbox(mask)
            mean_rgb = np.mean(rgb[mask], axis=0)
            mean_hsv = np.mean(hsv[mask], axis=0)
            if centroid[1] < h * 0.34 and mean_hsv[1] < 35.0:
                continue
            components.append({
                "mask": mask,
                "area": area,
                "centroid": centroid,
                "bbox": bbox,
                "mean_rgb": mean_rgb,
                "mean_hsv": mean_hsv,
            })

        components.sort(key=lambda c: c["area"], reverse=True)
        return components

    @staticmethod
    def _match_component_to_prompt(
        prompt: str,
        components: list[dict[str, Any]],
        used_indices: set[int],
    ) -> int | None:
        """Match a prompt/label to the best unused connected component using color, spatial, and size cues."""
        p_low = prompt.lower()
        best_idx: int | None = None
        best_score = -1e9

        for idx, comp in enumerate(components):
            if idx in used_indices:
                continue
            r, g, b = float(comp["mean_rgb"][0]), float(comp["mean_rgb"][1]), float(comp["mean_rgb"][2])
            sat = float(comp["mean_hsv"][1])
            lum = (r + g + b) / 3.0
            green_excess = g - 0.5 * (r + b)
            cx = float(comp["centroid"][0])

            score = float(comp["area"]) * 0.002
            if "red" in p_low:
                score += (r - max(g, b)) * 6.0
            elif "blue" in p_low:
                score += (b - max(r, g)) * 6.0
            elif "green" in p_low or "mint" in p_low:
                score += green_excess * 25.0
            elif "white" in p_low or "bright" in p_low or "light" in p_low:
                score += (lum - 125.0) * 6.0 - sat * 2.5 - max(0.0, green_excess) * 10.0
            elif "gray" in p_low or "grey" in p_low or "silver" in p_low or "metal" in p_low:
                score += (60.0 - abs(lum - 112.0)) * 5.0 - max(0.0, green_excess) * 12.0
            elif "yellow" in p_low or "wood" in p_low or "coaster" in p_low:
                score += ((r + g) * 0.5 - b) * 4.0
            elif "rubber" in p_low or "dark" in p_low or "marker" in p_low or "black" in p_low:
                score += (128.0 - lum) * 4.0
            elif "left" in p_low:
                score += (320.0 - cx) * 2.0
            elif "center" in p_low or "middle" in p_low:
                score += (200.0 - abs(cx - 320.0)) * 2.0
            elif "right" in p_low:
                score += (cx - 320.0) * 2.0

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _segment_opencv_video(
        self,
        scene_id: str,
        frames_rgb: list[np.ndarray],
        text_prompts: list[str] | None,
        object_ids: list[str] | None = None,
    ) -> SamTrackingResult:
        """Segment and track distinct foreground tabletop objects across video frames."""
        f0 = frames_rgb[0]
        h, w = f0.shape[:2]
        per_frame_comps = [self._extract_foreground_components(f) for f in frames_rgb]
        comps0 = per_frame_comps[0]

        prompts = text_prompts or ["manipulated_object", "target_zone"]
        used0: set[int] = set()
        tracks: list[ObjectMaskTrack] = []

        max_tracks = len(object_ids) if object_ids else len(prompts)
        for idx in range(min(max_tracks, len(comps0))):
            label = prompts[idx] if idx < len(prompts) else f"object_{idx}"
            comp_idx = self._match_component_to_prompt(label, comps0, used0)
            if comp_idx is None:
                continue
            used0.add(comp_idx)
            c_info = comps0[comp_idx]
            raw_mask0 = c_info["mask"]
            eroded0 = erode_mask(raw_mask0, iterations=self.erosion_px)
            c0 = c_info["centroid"]
            bbox0 = c_info["bbox"]
            ref_rgb = c_info["mean_rgb"]

            centroids_over_time: list[tuple[float, float]] = [c0]
            masks_over_time: list[np.ndarray] = [raw_mask0]
            prev_c = c0
            prev_mask = raw_mask0

            for t in range(1, len(frames_rgb)):
                comps_t = per_frame_comps[t]
                best_c_info = None
                best_dist = 1e9
                for cand in comps_t:
                    pos_dist = float(np.hypot(cand["centroid"][0] - prev_c[0], cand["centroid"][1] - prev_c[1]))
                    col_dist = float(np.linalg.norm(cand["mean_rgb"] - ref_rgb))
                    cost = pos_dist + 0.5 * col_dist
                    if cost < best_dist and pos_dist < max(w, h) * 0.35:
                        best_dist = cost
                        best_c_info = cand
                if best_c_info is not None:
                    prev_c = best_c_info["centroid"]
                    prev_mask = best_c_info["mask"]
                centroids_over_time.append(prev_c)
                masks_over_time.append(prev_mask)

            obj_id = _resolve_object_id(None, label, idx, prompts, object_ids)
            tracks.append(
                ObjectMaskTrack(
                    object_id=obj_id,
                    label=label,
                    frame0_mask=raw_mask0,
                    frame0_eroded_mask=eroded0,
                    bbox_2d=bbox0,
                    centroids_2d=centroids_over_time,
                    masks_over_time=masks_over_time,
                )
            )

        tracks = _deduplicate_tracks(tracks, object_ids)

        return SamTrackingResult(
            scene_id=scene_id,
            resolution=(w, h),
            num_frames=len(frames_rgb),
            tracks=tracks,
            frame0_rgb=f0,
            frame0_depth=None,
            sam_provider_used="opencv-hsv (offline)",
        )

    def _call_sam_api(
        self,
        video_path: str,
        prompts: list[str],
        frames_rgb: list[np.ndarray],
        object_ids: list[str] | None = None,
    ) -> SamTrackingResult:
        """Call Meta SAM 3.1 REST endpoint with base64-encoded keyframes."""
        import httpx

        api_url = self.api_url or "https://api.meta.ai/v1/models/sam-3.1/track"
        frame0_b64 = ""
        if frames_rgb:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2BGR))
            if ok:
                frame0_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "sam-3.1",
            "video_path": Path(video_path).name if video_path else "frame0.png",
            "frame0_jpeg_base64": frame0_b64,
            "prompts": prompts,
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                api_url,
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            self.cost_tracker.log_api_call(
                input_tokens=500,
                output_tokens=200,
                model="sam_3.1",
                purpose="video_mask_tracking",
            )
            data = resp.json()

        tracks: list[ObjectMaskTrack] = []
        raw_items = data.get("tracks") or data.get("objects") or []
        h, w = (frames_rgb[0].shape[:2]) if frames_rgb else (480, 640)
        for idx, item in enumerate(raw_items):
            label = item.get("label", prompts[idx] if idx < len(prompts) else f"object_{idx}")
            obj_id = _resolve_object_id(item.get("object_id"), label, idx, prompts, object_ids)
            mask_b64 = item.get("mask_png_b64") or item.get("mask_b64")
            if not mask_b64:
                continue
            mask_bytes = base64.b64decode(mask_b64)
            mask_img = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                continue
            raw_mask = (mask_img > 127)
            if raw_mask.shape != (h, w):
                raw_mask = cv2.resize(raw_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            eroded = erode_mask(raw_mask, iterations=self.erosion_px)
            bbox_2d = tuple(item.get("bbox_2d", compute_mask_centroid_and_bbox(raw_mask)[1]))

            centroids = item.get("centroids_2d")
            if centroids and len(centroids) == len(frames_rgb):
                centroids_2d = [tuple(c) for c in centroids]
                masks_over_time = [raw_mask for _ in frames_rgb]
            else:
                centroids_2d, masks_over_time = _propagate_mask_over_frames(raw_mask, frames_rgb)

            tracks.append(
                ObjectMaskTrack(
                    object_id=obj_id,
                    label=label,
                    frame0_mask=raw_mask,
                    frame0_eroded_mask=eroded,
                    bbox_2d=bbox_2d,
                    centroids_2d=centroids_2d,
                    masks_over_time=masks_over_time,
                )
            )

        tracks = _deduplicate_tracks(tracks, object_ids)
        if not tracks:
            raise RuntimeError("Tier 1 Meta SAM 3.1 API returned no valid tracks")

        return SamTrackingResult(
            scene_id=Path(video_path).stem if video_path else "sam_api_scene",
            resolution=(w, h),
            num_frames=len(frames_rgb) if frames_rgb else 1,
            tracks=tracks,
            frame0_rgb=frames_rgb[0] if frames_rgb else None,
            video_path=video_path,
            sam_provider_used="meta-api/sam-3.1",
        )

    def _call_self_hosted_endpoint(
        self,
        frames_rgb: list[np.ndarray],
        prompts: list[str],
        scene_id: str,
        video_path: str = "",
        object_ids: list[str] | None = None,
    ) -> SamTrackingResult:
        """Call self-hosted unsloth/sam3.1 GPU endpoint (Colab/Kaggle/Modal) via POST /segment."""
        import httpx

        if not frames_rgb:
            raise ValueError("No frames provided for self-hosted SAM 3.1 endpoint")

        f0 = frames_rgb[0]
        h, w = f0.shape[:2]

        ok, buf = cv2.imencode(".png", cv2.cvtColor(f0, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Failed to encode Frame 0 as PNG")
        img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        endpoint = (self.endpoint_url or "").rstrip("/")
        url = f"{endpoint}/segment"

        payload = {
            "image_b64": img_b64,
            "prompts": prompts,
            "max_frames": len(frames_rgb),
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        self.cost_tracker.log_api_call(
            input_tokens=800,
            output_tokens=400,
            model="unsloth/sam3.1",
            purpose="self_hosted_gpu_segmentation",
        )

        tracks: list[ObjectMaskTrack] = []
        raw_items = (
            data.get("tracks")
            or data.get("detections")
            or data.get("objects")
            or data.get("masks")
            or []
        )

        max_mask_area = int(h * w * 0.25)
        min_mask_area = 250 if object_ids is None else 20

        for idx, item in enumerate(raw_items):
            label = item.get("label", prompts[idx] if idx < len(prompts) else f"object_{idx}")
            obj_id = _resolve_object_id(item.get("object_id"), label, idx, prompts, object_ids)

            mask_b64 = item.get("mask_png_b64") or item.get("mask_b64")
            if not mask_b64:
                continue

            mask_bytes = base64.b64decode(mask_b64)
            mask_img = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                continue

            raw_mask = (mask_img > 127)
            if raw_mask.shape != (h, w):
                raw_mask = cv2.resize(raw_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

            area = int(np.sum(raw_mask))
            if object_ids is None and (area < min_mask_area or area > max_mask_area):
                continue

            eroded = erode_mask(raw_mask, iterations=self.erosion_px)

            if "bbox_2d" in item and len(item["bbox_2d"]) == 4:
                bbox_2d = tuple(int(v) for v in item["bbox_2d"])
            else:
                _, bbox_2d = compute_mask_centroid_and_bbox(raw_mask)

            centroids = item.get("centroids_2d")
            if centroids and len(centroids) == len(frames_rgb):
                centroids_2d = [tuple(c) for c in centroids]
                masks_over_time = [raw_mask for _ in frames_rgb]
            else:
                centroids_2d, masks_over_time = _propagate_mask_over_frames(raw_mask, frames_rgb)

            tracks.append(
                ObjectMaskTrack(
                    object_id=obj_id,
                    label=label,
                    frame0_mask=raw_mask,
                    frame0_eroded_mask=eroded,
                    bbox_2d=bbox_2d,
                    centroids_2d=centroids_2d,
                    masks_over_time=masks_over_time,
                )
            )

        tracks = _deduplicate_tracks(tracks, object_ids)

        # In open multi-object discovery (e.g. multi_stack video with 3 objects), if the
        # single-argmax Colab endpoint returned fewer distinct objects than requested prompts,
        # supplement the missing prompts from adaptive foreground components.
        if object_ids is None and len(tracks) < len(prompts):
            comps0 = self._extract_foreground_components(f0)
            if len(comps0) >= len(prompts):
                # Re-match all prompts cleanly against the extracted foreground components
                # so color-specific prompts ("gray object", "green object", "white object")
                # map 1-to-1 to the right physical objects without duplicate/shifted labels.
                cv_res = self._segment_opencv_video(scene_id, frames_rgb, prompts, object_ids=None)
                if len(cv_res.tracks) >= len(prompts):
                    tracks = cv_res.tracks

        return SamTrackingResult(
            scene_id=scene_id,
            resolution=(w, h),
            num_frames=len(frames_rgb),
            tracks=tracks,
            frame0_rgb=f0,
            frame0_depth=None,
            video_path=video_path,
            sam_provider_used="unsloth/sam3.1 (Colab/Kaggle T4 GPU)",
        )

    def _call_local_hf_sam3(
        self,
        frames_rgb: list[np.ndarray],
        prompts: list[str],
        scene_id: str,
        video_path: str = "",
        object_ids: list[str] | None = None,
    ) -> SamTrackingResult:
        """Run local Hugging Face unsloth/sam3.1 inference on CPU."""
        import torch
        from PIL import Image

        if not frames_rgb:
            raise ValueError("No frames provided for local HF SAM 3.1")

        processor, model = _get_local_sam3_singleton()

        f0 = frames_rgb[0]
        h, w = f0.shape[:2]
        pil_img = Image.fromarray(f0)

        tracks: list[ObjectMaskTrack] = []

        with torch.no_grad():
            try:
                inputs = processor(images=pil_img, text=prompts, return_tensors="pt")
            except TypeError:
                inputs = processor(pil_img, return_tensors="pt")

            inputs = {k: v.to("cpu") if hasattr(v, "to") else v for k, v in inputs.items()}
            outputs = model(**inputs)

            pred_masks = getattr(outputs, "pred_masks", None)
            if pred_masks is None and hasattr(outputs, "logits"):
                pred_masks = outputs.logits > 0

            if pred_masks is not None:
                masks_np = pred_masks.squeeze().detach().cpu().numpy()
                if masks_np.ndim == 2:
                    masks_np = masks_np[np.newaxis, ...]
                for idx, raw_mask in enumerate(masks_np[:len(prompts)]):
                    bool_mask = (raw_mask > 0.5) if raw_mask.dtype != bool else raw_mask
                    if bool_mask.shape != (h, w):
                        bool_mask = cv2.resize(bool_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                    eroded = erode_mask(bool_mask, iterations=self.erosion_px)
                    _, bbox0 = compute_mask_centroid_and_bbox(bool_mask)
                    label = prompts[idx] if idx < len(prompts) else f"object_{idx}"
                    obj_id = _resolve_object_id(None, label, idx, prompts, object_ids)
                    centroids_2d, masks_over_time = _propagate_mask_over_frames(bool_mask, frames_rgb)
                    tracks.append(
                        ObjectMaskTrack(
                            object_id=obj_id,
                            label=label,
                            frame0_mask=bool_mask,
                            frame0_eroded_mask=eroded,
                            bbox_2d=bbox0,
                            centroids_2d=centroids_2d,
                            masks_over_time=masks_over_time,
                        )
                    )

        tracks = _deduplicate_tracks(tracks, object_ids)
        if not tracks:
            raise RuntimeError("Local HF SAM 3.1 produced no tracks")

        return SamTrackingResult(
            scene_id=scene_id,
            resolution=(w, h),
            num_frames=len(frames_rgb),
            tracks=tracks,
            frame0_rgb=f0,
            frame0_depth=None,
            video_path=video_path,
            sam_provider_used="unsloth/sam3.1 (Local HF CPU)",
        )

    def segment_from_sim_env(
        self,
        env: Any,
        scene: SceneGraph | None = None,
        num_frames: int = 15,
        execute_oracle: bool = False,
    ) -> tuple[SamTrackingResult, np.ndarray]:
        """Generate realistic multi-frame SAM-like mask tracks and depth from a MuJoCo env."""
        return segment_from_sim_env(
            env=env,
            scene=scene,
            num_frames=num_frames,
            execute_oracle=execute_oracle,
        )


def segment_from_sim_env(
    env: Any,
    scene: SceneGraph | None = None,
    num_frames: int = 15,
    execute_oracle: bool = False,
) -> tuple[SamTrackingResult, np.ndarray]:
    """Generate realistic multi-frame SAM-like mask tracks and depth from a MuJoCo env."""
    from src.validation.ppo_runner import OracleController

    if scene is None:
        scene = getattr(env, "scene", None)
        if scene is None:
            raise ValueError("scene must be provided or env must have a .scene attribute")

    w, h = scene.camera.resolution
    env.reset(seed=42)

    body_id_to_obj: dict[int, tuple[str, str]] = {}
    for obj in scene.objects:
        bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, obj.id)
        if bid >= 0:
            body_id_to_obj[bid] = (obj.id, obj.label)

    renderer = mujoco.Renderer(env.model, height=h, width=w)
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "render_cam")

    renderer.update_scene(env.data, camera=cam_id if cam_id >= 0 else -1)
    frame0_rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(env.data, camera=cam_id if cam_id >= 0 else -1)
    frame0_depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    tracks_dict: dict[str, ObjectMaskTrack] = {}
    renderer.enable_segmentation_rendering()
    renderer.update_scene(env.data, camera=cam_id if cam_id >= 0 else -1)
    seg0 = renderer.render().copy()
    renderer.disable_segmentation_rendering()

    geom_ids_0 = seg0[:, :, 0]
    obj_types_0 = seg0[:, :, 1]

    for bid, (obj_id, label) in body_id_to_obj.items():
        mask = np.zeros((h, w), dtype=bool)
        for gid in range(env.model.ngeom):
            if env.model.geom_bodyid[gid] == bid:
                mask |= (obj_types_0 == int(mujoco.mjtObj.mjOBJ_GEOM)) & (geom_ids_0 == gid)

        eroded = erode_mask(mask, iterations=4)
        centroid, bbox = compute_mask_centroid_and_bbox(mask)
        tracks_dict[obj_id] = ObjectMaskTrack(
            object_id=obj_id,
            label=label,
            frame0_mask=mask,
            frame0_eroded_mask=eroded,
            bbox_2d=bbox,
            centroids_2d=[centroid],
            masks_over_time=[mask],
        )

    controller = OracleController(scene.task.waypoints) if execute_oracle else None
    steps_per_frame = max(1, 60 // max(num_frames - 1, 1))
    idle_action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    for _ in range(1, num_frames):
        for _ in range(steps_per_frame):
            action = controller.get_action(env) if controller is not None else idle_action
            env.step(action)

        renderer.enable_segmentation_rendering()
        renderer.update_scene(env.data, camera=cam_id if cam_id >= 0 else -1)
        seg_t = renderer.render().copy()
        renderer.disable_segmentation_rendering()

        g_ids = seg_t[:, :, 0]
        o_types = seg_t[:, :, 1]

        for bid, (obj_id, _) in body_id_to_obj.items():
            mask_t = np.zeros((h, w), dtype=bool)
            for gid in range(env.model.ngeom):
                if env.model.geom_bodyid[gid] == bid:
                    mask_t |= (o_types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (g_ids == gid)
            c_t, _ = compute_mask_centroid_and_bbox(mask_t)
            tracks_dict[obj_id].centroids_2d.append(c_t)
            tracks_dict[obj_id].masks_over_time.append(mask_t)

    renderer.close()
    env.reset(seed=42)

    result = SamTrackingResult(
        scene_id=scene.scene_id,
        resolution=(w, h),
        num_frames=num_frames,
        tracks=list(tracks_dict.values()),
        frame0_rgb=frame0_rgb,
        frame0_depth=frame0_depth,
        sam_provider_used="sim-depth-seg (offline)",
    )
    return result, frame0_depth
