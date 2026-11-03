"""WorldForge FastAPI Backend Server.

Provides REST and Server-Sent Event (SSE) endpoints for:
- Listing golden scenes and environment presets.
- Executing the complete 8-stage pipeline in Python with MuJoCo 3.17 physics,
  Render+Diff Mask IoU, and Stable-Baselines3 PPO policy training.
- Live streaming reasoning traces from Baseten's meta-models/muse-spark-open-1026.
- Static file serving for real generated MP4 rollouts, mask images, and diff overlays.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import io
import json
import logging
import os
import re
import shutil
import threading
import zipfile
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import mujoco
import numpy as np
from src.compiler.base_env import WorldForgeEnv
from src.compiler.generated_reward import compute_reward
from src.compiler.mjcf_builder import SceneCompiler
from collections import OrderedDict
from src.pipeline import (
    OUTPUTS_ROOT,
    SCENES_DIR,
    run_worldforge_pipeline,
    sanitize_scene_id,
    stream_worldforge_pipeline,
    validate_safe_video_path,
)
from src.schema.scene_graph import SceneGraph
from src.validation.ppo_runner import OracleController

load_dotenv()
logger = logging.getLogger(__name__)

_ALLOWED_VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".mov", ".avi", ".mkv"})
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def ensure_preview_assets() -> None:
    """Ensure cached preview assets exist for all golden scenes."""
    for file_path in SCENES_DIR.glob("*.json"):
        clean_id = file_path.stem
        out_dir = OUTPUTS_ROOT / clean_id
        if (
            not (out_dir / "frame0_masks.png").exists()
            or not (out_dir / "diff_overlay.png").exists()
            or not (out_dir / "oracle_rollout.mp4").exists()
        ):
            try:
                from scripts.generate_previews import generate_scene_previews
                generate_scene_previews(clean_id, SCENES_DIR, OUTPUTS_ROOT)
            except Exception as exc:
                logger.warning("Could not auto-generate preview assets for %s: %s", clean_id, exc)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Generate missing preview assets in a background daemon thread so cold start is non-blocking."""
    if os.environ.get("WORLDFORGE_SYNC_PREVIEWS", "").lower() in ("1", "true", "yes"):
        ensure_preview_assets()
    else:
        threading.Thread(target=ensure_preview_assets, daemon=True, name="preview-init").start()
    yield


app = FastAPI(
    title="WorldForge-RL Backend API",
    version="1.0.0",
    description="Studio-Grade MuJoCo 3.17 + Baseten / Meta RL Environment Synthesis API",
    lifespan=_lifespan,
)

# Restrict CORS origins instead of reflecting arbitrary origins with allow_credentials=True
_raw_origins = os.environ.get(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000",
)
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip() and o.strip() != "*"]
if not _allowed_origins:
    _allowed_origins = ["http://localhost:5173", "http://localhost:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.middleware("http")
async def _security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/outputs/"):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "sandbox"
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_ROOT)), name="outputs")

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
FRONTEND_ASSETS = FRONTEND_DIST / "assets"
if FRONTEND_ASSETS.is_dir():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_ASSETS)), name="frontend-assets")


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    """Execution request payload."""
    scene_id: str = Field(default="scene1_pick_place", description="Golden scene ID or custom scene")
    inject_fault: bool = Field(default=False, description="Whether to inject Gate 3 zero-reward fault to demo self-healing")
    provider: Literal["auto", "baseten", "meta", "local"] = Field(default="auto", description="LLM provider")
    quick_ppo: bool = Field(default=True, description="Use rapid interactive training horizon")
    task_prompt: str | None = Field(default=None, description="Optional custom natural-language manipulation prompt")
    video_path: str | None = Field(default=None, description="Optional uploaded tabletop MP4 path")


class SimActionRequest(BaseModel):
    """Real MuJoCo 3.17 simulation action request."""
    scene_id: str = Field(default="scene1_pick_place", description="Target scene ID")
    session_id: str = Field(default="default", description="Client session ID for physics isolation")
    action: tuple[float, float, float, float] = Field(
        default=(0.0, 0.0, 0.0, 1.0),
        description="4-DoF [dx, dy, dz, grip_cmd] action executed by MuJoCo 3.17 mocap weld",
    )


class SimResetRequest(BaseModel):
    """Real MuJoCo 3.17 simulation reset request."""
    scene_id: str = Field(default="scene1_pick_place", description="Target scene ID")
    session_id: str = Field(default="default", description="Client session ID for physics isolation")


active_sim_sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()
_SIM_SESSIONS_LOCK = threading.Lock()
_MAX_SIM_SESSIONS = 32


def _resolve_safe_scene_path(scene_id: str, fallback_to_default: bool = False) -> tuple[str, Path]:
    """Validate scene_id against path traversal and resolve to a file inside SCENES_DIR."""
    try:
        clean_id = sanitize_scene_id(scene_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    scenes_root = SCENES_DIR.resolve()
    target = (SCENES_DIR / f"{clean_id}.json").resolve()
    if target.parent != scenes_root:
        raise HTTPException(status_code=400, detail="Invalid scene_id")

    if not target.exists():
        for p in sorted(SCENES_DIR.glob("*.json")):
            if clean_id in p.stem or p.stem in clean_id:
                target = p.resolve()
                clean_id = p.stem
                break

    if not target.exists():
        if fallback_to_default:
            clean_id = "scene1_pick_place"
            target = (SCENES_DIR / "scene1_pick_place.json").resolve()
        else:
            raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found")

    return clean_id, target


def _load_scene_graph(scene_id: str, fallback_to_default: bool = True) -> SceneGraph:
    clean_id, target = _resolve_safe_scene_path(scene_id, fallback_to_default=fallback_to_default)
    recon_path = (OUTPUTS_ROOT / clean_id / "reconstructed_scene.json").resolve()
    if recon_path.exists():
        try:
            with open(recon_path, "r", encoding="utf-8") as f:
                return SceneGraph.model_validate(json.load(f))
        except Exception as exc:
            logger.warning("Could not load reconstructed_scene.json for %s: %s", clean_id, exc)
    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SceneGraph.model_validate(data)


def _extract_sim_state(
    env: WorldForgeEnv,
    sg: SceneGraph,
    step: int,
    grip_cmd: float = 1.0,
) -> dict[str, Any]:
    """Extract ground-truth MuJoCo 3.17 physics state, body positions, aperture, and rewards."""
    grip_pos = env.data.mocap_pos[0].tolist()
    obj_pos: dict[str, list[float]] = {}
    obj_quat: dict[str, list[float]] = {}

    for obj in sg.objects:
        try:
            bid = env.model.body(obj.id).id
            obj_pos[obj.id] = env.data.xpos[bid].tolist()
            obj_quat[obj.id] = env.data.xquat[bid].tolist()
        except Exception:
            obj_pos[obj.id] = list(obj.position)
            obj_quat[obj.id] = list(obj.quaternion)

    aperture = 0.04
    try:
        lf_id = env.model.body("finger_L").id
        rf_id = env.model.body("finger_R").id
        aperture = float(np.linalg.norm(env.data.xpos[lf_id] - env.data.xpos[rf_id]))
    except Exception:
        pass

    reward, term, r_info = compute_reward(env.model, env.data, sg)

    d_reach = float(r_info.get("d_reach", 0.0))
    d_goal = float(r_info.get("d_place", r_info.get("d_goal", 0.0)))
    grasp_bonus = float(r_info.get("grasp_bonus", 0.0))
    lift_reward = float(r_info.get("lift_reward", 0.0))
    place_reward = float(r_info.get("place_reward", 0.0))
    reach_reward = float(r_info.get("reach_reward", 0.0))

    return {
        "step": step,
        "gripperPos": grip_pos,
        "gripperCmd": float(grip_cmd),
        "fingerAperture": aperture,
        "isGrasped": bool(grasp_bonus > 0.5),
        "objectPositions": obj_pos,
        "objectQuaternions": obj_quat,
        "diagnostics": {
            "dReach": d_reach,
            "dGoal": d_goal,
        },
        "reward": float(reward),
        "terminated": bool(term),
        "stageBreakdown": {
            "reach": reach_reward,
            "graspOrClearance": grasp_bonus,
            "liftOrProgress": lift_reward,
            "placeOrGoal": place_reward,
            "terminalBonus": 10.0 if term else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
@app.get("/api/status")
def get_system_status() -> dict[str, Any]:
    """Return system readiness, hardware, and connected LLM provider telemetry."""
    baseten_key = os.environ.get("BASETEN_API_KEY", "")
    meta_key = os.environ.get("META_API_KEY", "")
    meta_sam_url = os.environ.get("META_SAM_API_URL", "")
    modal_token = os.environ.get("MODAL_TOKEN_ID", "")
    sam_endpoint = os.environ.get("SAM31_ENDPOINT_URL", "")

    sam_provider = "sim-depth-seg (offline)"
    if sam_endpoint and sam_endpoint.strip():
        sam_provider = "unsloth/sam3.1 (Colab/Kaggle T4 GPU)"
    elif meta_key and not meta_key.startswith("your_") and meta_sam_url:
        sam_provider = "meta-api/sam-3.1"
    elif os.environ.get("SAM31_LOCAL_HF", "").lower() in ("true", "1", "yes"):
        sam_provider = "unsloth/sam3.1 (Local HF CPU)"

    return {
        "status": "online",
        "engine": "MuJoCo 3.17.0",
        "mujoco_version": getattr(mujoco, "__version__", "3.17.0"),
        "baseten": {
            "configured": bool(baseten_key and not baseten_key.startswith("your_")),
            "model": "meta-models/muse-spark-open-1026",
            "base_url": os.environ.get("BASETEN_BASE_URL", "https://inference.baseten.co/v1"),
            "grant": "$100 Baseten Hackathon Credit",
        },
        "meta": {
            "configured": bool(meta_key and not meta_key.startswith("your_")),
            "model": "muse-spark-1.2",
            "grant": "$150 Meta Hackathon Credit",
        },
        "modal": {
            "configured": bool(modal_token),
            "grant": "$130 Modal Cloud Credit",
        },
        "sam31": {
            "configured": bool(sam_endpoint or (meta_key and not meta_key.startswith("your_") and meta_sam_url)),
            "provider": sam_provider,
            "endpoint_url": sam_endpoint,
        },
    }


@app.get("/api/scenes")
def list_scenes() -> list[dict[str, Any]]:
    """List the 3 verified golden test scenes with rich metadata."""
    scenes_meta: list[dict[str, Any]] = []

    for file_path in sorted(SCENES_DIR.glob("*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sg = SceneGraph.model_validate(data)
            clean_stem = file_path.stem
            scenes_meta.append({
                "scene_id": sg.scene_id,
                "file_id": clean_stem,
                "file_name": file_path.name,
                "task_type": sg.task.task_type,
                "description": sg.task.description,
                "num_objects": len(sg.objects),
                "objects": [
                    {
                        "id": o.id,
                        "label": o.label,
                        "primitive": o.primitive,
                        "material": o.material,
                        "joint_type": o.joint.type,
                    }
                    for o in sg.objects
                ],
                "table_z": sg.table_z,
                "has_articulated_joint": any(o.joint.type in ("slide", "hinge") for o in sg.objects),
                "masks_url": f"/outputs/{clean_stem}/frame0_masks.png",
                "diff_url": f"/outputs/{clean_stem}/diff_overlay.png",
                "rollout_url": f"/outputs/{clean_stem}/rollout.mp4",
                "oracle_url": f"/outputs/{clean_stem}/oracle_rollout.mp4",
            })
        except Exception as exc:
            logger.warning("Could not load scene file %s: %s", file_path, exc)

    return scenes_meta


@app.get("/api/scene-preview/{scene_id}")
def get_scene_preview(scene_id: str) -> dict[str, Any]:
    """Retrieve scene-specific Stage 1 perception masks, Stage 8 render+diff overlay, and 4.0s rollouts."""
    clean_id, _ = _resolve_safe_scene_path(scene_id, fallback_to_default=True)

    return {
        "scene_id": clean_id,
        "masks_url": f"/outputs/{clean_id}/frame0_masks.png",
        "diff_url": f"/outputs/{clean_id}/diff_overlay.png",
        "rollout_url": f"/outputs/{clean_id}/rollout.mp4",
        "oracle_url": f"/outputs/{clean_id}/oracle_rollout.mp4",
    }


@app.get("/api/scenes/{scene_id}")
def get_scene(scene_id: str) -> dict[str, Any]:
    """Retrieve full SceneGraph JSON for a given scene ID or file ID."""
    clean_id, target = _resolve_safe_scene_path(scene_id, fallback_to_default=False)
    recon_path = (OUTPUTS_ROOT / clean_id / "reconstructed_scene.json").resolve()
    if recon_path.exists():
        try:
            recon_path.unlink()
        except OSError:
            pass

    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Real MuJoCo 3.17 Interactive Simulation Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/sim/reset")
def sim_reset(req: SimResetRequest) -> dict[str, Any]:
    """Reset the live MuJoCo 3.17 simulation environment and return ground-truth state."""
    clean_id, _ = _resolve_safe_scene_path(req.scene_id, fallback_to_default=True)
    sg = _load_scene_graph(clean_id)
    session_key = f"{req.session_id}:{clean_id}"

    with _SIM_SESSIONS_LOCK:
        prev = active_sim_sessions.get(session_key)
        if prev and "env" in prev:
            try:
                prev["env"].close()
            except Exception:
                pass

        if len(active_sim_sessions) >= _MAX_SIM_SESSIONS and session_key not in active_sim_sessions:
            _, oldest = active_sim_sessions.popitem(last=False)
            if "env" in oldest:
                try:
                    oldest["env"].close()
                except Exception:
                    pass

        lock = threading.Lock()
        with lock:
            env = WorldForgeEnv(sg)
            env.reset(seed=42)
            ctrl = OracleController(sg.task.waypoints)
            active_sim_sessions[session_key] = {
                "env": env,
                "sg": sg,
                "ctrl": ctrl,
                "step": 0,
                "grip_cmd": 1.0,
                "lock": lock,
            }
            active_sim_sessions.move_to_end(session_key)
            return _extract_sim_state(env, sg, step=0, grip_cmd=1.0)


@app.post("/api/sim/step")
def sim_step(req: SimActionRequest) -> dict[str, Any]:
    """Execute a single real physics step in MuJoCo 3.17 with the given 4-DoF action."""
    clean_id, _ = _resolve_safe_scene_path(req.scene_id, fallback_to_default=True)
    session_key = f"{req.session_id}:{clean_id}"

    with _SIM_SESSIONS_LOCK:
        if session_key not in active_sim_sessions:
            sim_reset(SimResetRequest(scene_id=clean_id, session_id=req.session_id))
        session = active_sim_sessions[session_key]
        active_sim_sessions.move_to_end(session_key)

    with session["lock"]:
        env: WorldForgeEnv = session["env"]
        sg: SceneGraph = session["sg"]

        action_arr = np.array(req.action, dtype=np.float32)
        obs, r, term, trunc, info = env.step(action_arr)
        session["step"] += 1
        session["grip_cmd"] = float(req.action[3])

        return _extract_sim_state(env, sg, step=session["step"], grip_cmd=session["grip_cmd"])


@app.post("/api/sim/oracle-step")
def sim_oracle_step(req: SimResetRequest) -> dict[str, Any]:
    """Step the deterministic Waypoint Oracle controller inside MuJoCo 3.17."""
    clean_id, _ = _resolve_safe_scene_path(req.scene_id, fallback_to_default=True)
    session_key = f"{req.session_id}:{clean_id}"

    with _SIM_SESSIONS_LOCK:
        if session_key not in active_sim_sessions:
            sim_reset(SimResetRequest(scene_id=clean_id, session_id=req.session_id))
        session = active_sim_sessions[session_key]
        active_sim_sessions.move_to_end(session_key)

    with session["lock"]:
        env: WorldForgeEnv = session["env"]
        sg: SceneGraph = session["sg"]
        ctrl: OracleController = session["ctrl"]

        action = ctrl.get_action(env)
        obs, r, term, trunc, info = env.step(action)
        session["step"] += 1
        session["grip_cmd"] = float(action[3])

        return _extract_sim_state(env, sg, step=session["step"], grip_cmd=session["grip_cmd"])


@app.get("/api/sim/trajectory")
def sim_trajectory(
    scene_id: str = Query("scene1_pick_place"),
    max_steps: int = Query(120, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Compute and return the full high-fidelity MuJoCo 3.17 physics trajectory."""
    clean_id, _ = _resolve_safe_scene_path(scene_id, fallback_to_default=True)
    sg = _load_scene_graph(clean_id)
    env = WorldForgeEnv(sg)
    ctrl = OracleController(sg.task.waypoints)
    env.reset(seed=42)

    trajectory: list[dict[str, Any]] = []
    terminated = False
    last_grip = sg.task.waypoints[-1][3] if sg.task.waypoints else 0.0

    effective_steps = max(max_steps, 220) if (sg.task.task_type == "multi_stack" and max_steps >= 120) else max_steps
    for s in range(effective_steps):
        if not ctrl.completed:
            action = ctrl.get_action(env)
            obs, r, term, trunc, info = env.step(action)
            if term or trunc:
                terminated = True
            grip_cmd = float(action[3])
        else:
            idle_act = np.array([0.0, 0.0, 0.0, last_grip], dtype=np.float32)
            env.step(idle_act)
            grip_cmd = float(last_grip)

        trajectory.append(_extract_sim_state(env, sg, step=s, grip_cmd=grip_cmd))

    env.close()
    return trajectory


@app.post("/api/run")
def run_pipeline(req: RunRequest) -> dict[str, Any]:
    """Execute the complete 8-stage pipeline synchronously and return all telemetry."""
    clean_id, _ = _resolve_safe_scene_path(req.scene_id, fallback_to_default=False)
    if req.video_path:
        try:
            validate_safe_video_path(req.video_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        result = run_worldforge_pipeline(
            scene_id=clean_id,
            inject_fault=req.inject_fault,
            provider=req.provider,
            quick_ppo=req.quick_ppo,
            video_path=req.video_path,
            task_prompt=req.task_prompt,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Pipeline run failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/run/stream")
def stream_pipeline_get(
    scene_id: str = Query("scene1_pick_place"),
    inject_fault: bool = Query(False),
    provider: str = Query("auto"),
    quick_ppo: bool = Query(True),
    task_prompt: str | None = Query(None),
    video_path: str | None = Query(None),
):
    """Stream the 8-stage pipeline as Server-Sent Events (SSE)."""
    clean_id, _ = _resolve_safe_scene_path(scene_id, fallback_to_default=False)
    if video_path:
        try:
            validate_safe_video_path(video_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StreamingResponse(
        stream_worldforge_pipeline(
            scene_id=clean_id,
            inject_fault=inject_fault,
            provider=provider,
            quick_ppo=quick_ppo,
            video_path=video_path,
            task_prompt=task_prompt,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/run/stream")
def stream_pipeline_post(req: RunRequest):
    """Stream the 8-stage pipeline via POST as Server-Sent Events (SSE)."""
    clean_id, _ = _resolve_safe_scene_path(req.scene_id, fallback_to_default=False)
    if req.video_path:
        try:
            validate_safe_video_path(req.video_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StreamingResponse(
        stream_worldforge_pipeline(
            scene_id=clean_id,
            inject_fault=req.inject_fault,
            provider=req.provider,
            quick_ppo=req.quick_ppo,
            video_path=req.video_path,
            task_prompt=req.task_prompt,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/export/{scene_id}")
def export_gym_package(scene_id: str) -> Response:
    """Export a self-contained Gymnasium + MuJoCo 3.17 environment ZIP package."""
    clean_id, _ = _resolve_safe_scene_path(scene_id, fallback_to_default=False)
    sg = _load_scene_graph(clean_id, fallback_to_default=False)
    out_dir = (OUTPUTS_ROOT / clean_id).resolve()

    mjcf_path = out_dir / "environment.xml"
    if mjcf_path.exists():
        mjcf_xml = mjcf_path.read_text(encoding="utf-8")
    else:
        mjcf_xml = SceneCompiler(sg).compile()

    reward_src_path = Path(__file__).resolve().parents[1] / "compiler" / "generated_reward.py"
    reward_code = reward_src_path.read_text(encoding="utf-8") if reward_src_path.exists() else ""

    standalone_runner = f'''"""Standalone WorldForge-RL Gymnasium Environment ({sg.scene_id})."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent


class WorldForgeStandaloneEnv(gym.Env):
    """Self-contained Gymnasium environment wrapping {sg.scene_id} in MuJoCo 3.x."""

    metadata = {{"render_modes": ["rgb_array"], "render_fps": 25}}

    def __init__(
        self,
        xml_path: str | Path = HERE / "environment.xml",
        scene_json_path: str | Path = HERE / "scene_graph.json",
        render_mode: str | None = None,
        n_substeps: int = 10,
        max_delta_m: float = 0.02,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.n_substeps = n_substeps
        self.max_delta = max_delta_m

        with open(scene_json_path, "r", encoding="utf-8") as f:
            self.scene_data = json.load(f)

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self._obj_body_ids = []
        for obj in self.scene_data.get("objects", []):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj["id"])
            if bid >= 0:
                self._obj_body_ids.append(bid)

        n_objs = len(self._obj_body_ids)
        obs_dim = 3 + 2 + 3 * n_objs + 6 * n_objs + 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self._mocap_id = 0
        self._finger_act_L = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_actuator")
        self._finger_act_R = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_actuator_R")
        self._finger_jnt_L = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint_L")
        self._finger_jnt_R = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint_R")
        self._renderer: mujoco.Renderer | None = None

    def _obs(self) -> np.ndarray:
        parts = [self.data.mocap_pos[self._mocap_id].copy()]
        parts.append(np.array([
            self.data.qpos[self.model.jnt_qposadr[self._finger_jnt_L]],
            self.data.qpos[self.model.jnt_qposadr[self._finger_jnt_R]],
        ], dtype=np.float64))
        for bid in self._obj_body_ids:
            parts.append(self.data.xpos[bid].copy())
        for bid in self._obj_body_ids:
            parts.append(self.data.cvel[bid].copy())
        target = self.scene_data.get("task", {{}}).get("target_position")
        parts.append(np.array(target, dtype=np.float64) if target is not None else np.zeros(3))
        return np.concatenate(parts).astype(np.float32)

    def _compute_reward(self) -> tuple[float, bool, dict[str, Any]]:
        task = self.scene_data.get("task", {{}})
        obj_id = task.get("manipulated_object_id", "")
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_id)
        obj_pos = self.data.xpos[bid].copy() if bid >= 0 else np.zeros(3)
        grip_pos = self.data.mocap_pos[self._mocap_id].copy()
        d_reach = float(np.linalg.norm(grip_pos - obj_pos))
        r_reach = 1.0 - math.tanh(4.0 * d_reach)

        target = task.get("target_position")
        if target is not None:
            d_goal = float(np.linalg.norm(obj_pos - np.array(target, dtype=np.float64)))
            r_goal = 1.0 - math.tanh(4.0 * d_goal)
            terminated = d_goal < 0.04
            reward = 0.3 * r_reach + 0.7 * r_goal + (10.0 if terminated else 0.0)
        else:
            terminated = False
            reward = r_reach
        return float(reward), bool(terminated), {{"d_reach": d_reach}}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {{}}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.data.mocap_pos[self._mocap_id] += action[:3] * self.max_delta
        grip_target = float((action[3] + 1.0) / 2.0 * 0.04)
        self.data.ctrl[self._finger_act_L] = grip_target
        self.data.ctrl[self._finger_act_R] = grip_target
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        obs = self._obs()
        reward, terminated, info = self._compute_reward()
        return obs, reward, terminated, False, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        w, h = self.scene_data.get("camera", {{}}).get("resolution", [640, 480])
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "render_cam")
        self._renderer.update_scene(self.data, camera=cam_id if cam_id >= 0 else -1)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    env = WorldForgeStandaloneEnv()
    obs, _ = env.reset(seed=42)
    total_return = 0.0
    for step in range(250):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_return += reward
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    print(f"Simulated 250 Gymnasium steps of {sg.scene_id} (return={{total_return:.2f}})")
'''

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{clean_id}/environment.xml", mjcf_xml)
        zf.writestr(f"{clean_id}/scene_graph.json", json.dumps(sg.model_dump(), indent=2))
        zf.writestr(f"{clean_id}/reward_code.py", reward_code)
        zf.writestr(f"{clean_id}/run_standalone.py", standalone_runner)
        if (out_dir / "frame0_masks.png").exists():
            zf.write(out_dir / "frame0_masks.png", arcname=f"{clean_id}/frame0_masks.png")
        if (out_dir / "diff_overlay.png").exists():
            zf.write(out_dir / "diff_overlay.png", arcname=f"{clean_id}/diff_overlay.png")
        if (out_dir / "rollout.mp4").exists():
            zf.write(out_dir / "rollout.mp4", arcname=f"{clean_id}/rollout.mp4")

    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="worldforge_{clean_id}_gym_pkg.zip"'
        },
    )


@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accept an uploaded tabletop video for perception analysis with extension and content-type validation."""
    raw_name = Path(file.filename or "").name
    if not raw_name:
        raise HTTPException(status_code=400, detail="Missing filename")

    clean_name = _SAFE_FILENAME_RE.sub("_", raw_name).lstrip(".")
    ext = Path(clean_name).suffix.lower()
    if ext not in _ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Only video files ({sorted(_ALLOWED_VIDEO_EXTENSIONS)}) are allowed.",
        )

    content_type = (file.content_type or "").lower()
    if content_type and not (
        content_type.startswith("video/") or content_type == "application/octet-stream"
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Content-Type '{content_type}'. Expected video/*.",
        )

    uploads_dir = (OUTPUTS_ROOT / "uploads").resolve()
    uploads_dir.mkdir(parents=True, exist_ok=True)

    dest_path = (uploads_dir / clean_name).resolve()
    if dest_path.parent != uploads_dir:
        raise HTTPException(status_code=400, detail="Invalid upload filename")

    max_bytes = 25 * 1024 * 1024  # 25 MB cap
    total_bytes = 0
    try:
        with open(dest_path, "wb") as buffer:
            while True:
                chunk = await file.read(65536)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds maximum allowed upload size of 25 MB",
                    )
                buffer.write(chunk)
    except Exception:
        if dest_path.exists():
            try:
                dest_path.unlink()
            except OSError:
                pass
        raise

    return {
        "status": "success",
        "filename": clean_name,
        "video_path": str(dest_path),
        "video_url": f"/outputs/uploads/{clean_name}",
        "size_bytes": dest_path.stat().st_size,
    }


@app.get("/", include_in_schema=False)
def serve_frontend() -> Response:
    """Serve the image's compiled Studio entry point when it is available."""
    index_html = FRONTEND_DIST / "index.html"
    if not index_html.is_file():
        raise HTTPException(status_code=404, detail="Frontend build is not available.")
    return FileResponse(index_html, headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
