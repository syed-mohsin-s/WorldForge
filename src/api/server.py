"""WorldForge FastAPI Backend Server.

Provides REST and Server-Sent Event (SSE) endpoints for:
- Listing golden scenes and environment presets.
- Executing the complete 8-stage pipeline in Python with MuJoCo 3.17 physics,
  Render+Diff Mask IoU, and Stable-Baselines3 PPO policy training.
- Live streaming reasoning traces from Baseten's meta-models/muse-spark-open-1026.
- Static file serving for real generated MP4 rollouts, mask images, and diff overlays.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import src.compat  # noqa: F401
import mujoco
import numpy as np
from src.compiler.base_env import WorldForgeEnv
from src.compiler.generated_reward import compute_reward
from src.pipeline import OUTPUTS_ROOT, SCENES_DIR, run_worldforge_pipeline, stream_worldforge_pipeline
from src.schema.scene_graph import SceneGraph
from src.validation.ppo_runner import OracleController

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="WorldForge-RL Backend API",
    version="1.0.0",
    description="Studio-Grade MuJoCo 3.17 + Baseten / Meta RL Environment Synthesis API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_ROOT)), name="outputs")


def ensure_preview_assets():
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


ensure_preview_assets()


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
    action: tuple[float, float, float, float] = Field(
        default=(0.0, 0.0, 0.0, 1.0),
        description="4-DoF [dx, dy, dz, grip_cmd] action executed by MuJoCo 3.17 mocap weld",
    )


class SimResetRequest(BaseModel):
    """Real MuJoCo 3.17 simulation reset request."""
    scene_id: str = Field(default="scene1_pick_place", description="Target scene ID")


active_sim_sessions: dict[str, dict[str, Any]] = {}


def _load_scene_graph(scene_id: str) -> SceneGraph:
    clean_id = scene_id.replace(".json", "")
    target = SCENES_DIR / f"{clean_id}.json"
    if not target.exists():
        for p in SCENES_DIR.glob("*.json"):
            if clean_id in p.stem or p.stem in clean_id:
                target = p
                break
    if not target.exists():
        target = SCENES_DIR / "scene1_pick_place.json"
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
    modal_token = os.environ.get("MODAL_TOKEN_ID", "")

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
    clean_id = scene_id.replace(".json", "")
    target = SCENES_DIR / f"{clean_id}.json"
    if not target.exists():
        for p in SCENES_DIR.glob("*.json"):
            if clean_id in p.stem or p.stem in clean_id:
                clean_id = p.stem
                target = p
                break
    if not target.exists():
        clean_id = "scene1_pick_place"

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
    clean_id = scene_id.replace(".json", "")
    target = SCENES_DIR / f"{clean_id}.json"
    if not target.exists():
        for p in SCENES_DIR.glob("*.json"):
            if clean_id in p.stem or p.stem in clean_id:
                target = p
                break
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found")

    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Real MuJoCo 3.17 Interactive Simulation Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/sim/reset")
def sim_reset(req: SimResetRequest) -> dict[str, Any]:
    """Reset the live MuJoCo 3.17 simulation environment and return ground-truth state."""
    clean_id = req.scene_id.replace(".json", "")
    sg = _load_scene_graph(clean_id)
    prev = active_sim_sessions.get(clean_id)
    lock = prev["lock"] if (prev and "lock" in prev) else threading.Lock()
    with lock:
        if prev and "env" in prev:
            try:
                prev["env"].close()
            except Exception:
                pass
        env = WorldForgeEnv(sg)
        env.reset(seed=42)
        ctrl = OracleController(sg.task.waypoints)
        active_sim_sessions[clean_id] = {
            "env": env,
            "sg": sg,
            "ctrl": ctrl,
            "step": 0,
            "grip_cmd": 1.0,
            "lock": lock,
        }
        return _extract_sim_state(env, sg, step=0, grip_cmd=1.0)


@app.post("/api/sim/step")
def sim_step(req: SimActionRequest) -> dict[str, Any]:
    """Execute a single real physics step in MuJoCo 3.17 with the given 4-DoF action."""
    clean_id = req.scene_id.replace(".json", "")
    if clean_id not in active_sim_sessions:
        sim_reset(SimResetRequest(scene_id=clean_id))

    session = active_sim_sessions[clean_id]
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
    clean_id = req.scene_id.replace(".json", "")
    if clean_id not in active_sim_sessions:
        sim_reset(SimResetRequest(scene_id=clean_id))

    session = active_sim_sessions[clean_id]
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
    max_steps: int = Query(120),
) -> list[dict[str, Any]]:
    """Compute and return the full high-fidelity MuJoCo 3.17 physics trajectory."""
    clean_id = scene_id.replace(".json", "")
    sg = _load_scene_graph(clean_id)
    env = WorldForgeEnv(sg)
    ctrl = OracleController(sg.task.waypoints)
    env.reset(seed=42)

    trajectory: list[dict[str, Any]] = []
    terminated = False
    last_grip = sg.task.waypoints[-1][3] if sg.task.waypoints else 0.0

    for s in range(max_steps):
        if not terminated:
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
    try:
        result = run_worldforge_pipeline(
            scene_id=req.scene_id,
            inject_fault=req.inject_fault,
            provider=req.provider,
            quick_ppo=req.quick_ppo,
        )
        return result
    except Exception as exc:
        logger.exception("Pipeline run failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/run/stream")
def stream_pipeline_get(
    scene_id: str = Query("scene1_pick_place"),
    inject_fault: bool = Query(False),
    provider: str = Query("auto"),
    quick_ppo: bool = Query(True),
):
    """Stream the 8-stage pipeline as Server-Sent Events (SSE)."""
    return StreamingResponse(
        stream_worldforge_pipeline(
            scene_id=scene_id,
            inject_fault=inject_fault,
            provider=provider,
            quick_ppo=quick_ppo,
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
    return StreamingResponse(
        stream_worldforge_pipeline(
            scene_id=req.scene_id,
            inject_fault=req.inject_fault,
            provider=req.provider,
            quick_ppo=req.quick_ppo,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accept an uploaded tabletop MP4 video for perception analysis."""
    uploads_dir = OUTPUTS_ROOT / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename or "uploaded_tabletop.mp4").name
    dest_path = uploads_dir / safe_name
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "success",
        "filename": safe_name,
        "video_url": f"/outputs/uploads/{safe_name}",
        "size_bytes": dest_path.stat().st_size,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
