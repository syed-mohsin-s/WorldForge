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
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import mujoco
from src.pipeline import OUTPUTS_ROOT, SCENES_DIR, run_worldforge_pipeline, stream_worldforge_pipeline
from src.schema.scene_graph import SceneGraph

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="WorldForge-RL Backend API",
    version="1.0.0",
    description="Studio-Grade MuJoCo 3.17 + Baseten / Meta RL Environment Synthesis API",
)

# Enable CORS for frontend UI running on any port (e.g. 5173, 3000, 8080)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs directory for static files (rollout.mp4, frame0_masks.png, diff_overlay.png)
OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_ROOT)), name="outputs")


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    """Execution request payload."""
    scene_id: str = Field(default="scene1_pick_place", description="Golden scene ID or custom scene")
    inject_fault: bool = Field(default=False, description="Whether to inject Gate 3 zero-reward fault to demo self-healing")
    provider: Literal["auto", "baseten", "meta", "local"] = Field(default="auto", description="LLM provider")
    quick_ppo: bool = Field(default=True, description="Use 4096 timesteps for rapid interactive training")


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
            scenes_meta.append({
                "scene_id": sg.scene_id,
                "file_id": file_path.stem,
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
            })
        except Exception as exc:
            logger.warning("Could not load scene file %s: %s", file_path, exc)

    return scenes_meta


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

    dest_path = uploads_dir / file.filename
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "success",
        "filename": file.filename,
        "video_url": f"/outputs/uploads/{file.filename}",
        "size_bytes": dest_path.stat().st_size,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
