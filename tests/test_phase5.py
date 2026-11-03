"""WorldForge Phase 5 Test Suite.

Tests:
1. Baseten integration and hard-enforced model string ("meta-models/muse-spark-open-1026").
2. MuseSparkReasoner live reasoning parsing & deterministic offline fallback (with zero phantom tokens).
3. Complete 8-stage pipeline execution (run_worldforge_pipeline).
4. FastAPI endpoints (/api/status, /api/scenes, /api/run, /api/upload-video, /api/export)
   plus security hardening (CORS, upload extension filtering, path traversal protection).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.perception.lift_3d import LiftedObject, LiftedSceneGeometry
from src.perception.spark_reasoner import BASETEN_MODEL, MuseSparkReasoner
from src.pipeline import run_worldforge_pipeline
from src.schema.scene_graph import CameraSpec, SceneGraph

client = TestClient(app)


def test_baseten_model_string_enforced():
    """Verify that the exact hackathon model string is hard-coded and configured."""
    assert BASETEN_MODEL == "meta-models/muse-spark-open-1026"
    reasoner = MuseSparkReasoner(provider="local")
    assert reasoner.provider == "local"


def test_muse_spark_reasoner_offline_synthesis():
    """Verify deterministic fallback produces a valid SceneGraph with zero phantom API tokens."""
    geom = LiftedSceneGeometry(
        scene_id="test_offline_scene",
        table_z=0.0,
        camera=CameraSpec(pos=(0.6, 0.0, 0.5), lookat=(0.0, 0.0, 0.0)),
        objects=[
            LiftedObject(
                id="cube_1",
                label="red cube",
                primitive="box",
                size=[0.025, 0.025, 0.025],
                position=(0.0, 0.0, 0.026),
                quaternion=(1, 0, 0, 0),
                joint_type="free",
                joint_axis=(0, 0, 1),
                joint_range=None,
                joint_damping=0.5,
                is_static=False,
                is_prior_guess=False,
                parent_id=None,
            )
        ],
    )

    reasoner = MuseSparkReasoner(provider="local")
    sg = reasoner.synthesize_scene_graph(geom)

    assert isinstance(sg, SceneGraph)
    assert sg.scene_id == "test_offline_scene"
    assert len(sg.objects) == 1
    assert sg.objects[0].id == "cube_1"
    assert len(sg.task.waypoints) > 0
    assert "Analyzing 3D scene" in reasoner.last_reasoning_trace
    assert reasoner.cost_tracker.total_input_tokens == 0
    assert reasoner.cost_tracker.total_output_tokens == 0


def test_run_worldforge_pipeline_scene1():
    """Execute the full 8-stage pipeline on Scene 1 in quick_ppo mode."""
    events_received: list[dict] = []

    result = run_worldforge_pipeline(
        scene_id="scene1_pick_place",
        inject_fault=False,
        provider="local",
        quick_ppo=True,
        on_stage_event=lambda ev: events_received.append(ev),
    )

    stage_numbers = [e.get("stage") for e in events_received if isinstance(e.get("stage"), int)]
    for stage_num in range(1, 9):
        assert stage_num in stage_numbers

    assert result["scene_id"] == "scene1_pick_place"
    assert result["total_duration_s"] > 0
    assert "scene_graph" in result
    assert "mjcf_xml" in result
    assert "reward_code" in result
    assert result["render_diff"]["mean_iou"] >= 0.75
    assert result["gate_results"]["5"]["oracle_return"] > 0.0
    assert result["cost_telemetry"]["input_tokens"] == 0
    assert result["cost_telemetry"]["output_tokens"] == 0
    assert len(result["ppo_results"]["learning_curve"]) > 0


def test_pipeline_with_fault_injection_demonstrates_self_healing():
    """Verify that when inject_fault=True, self-healing diagnoses and patches the reward."""
    events: list[dict] = []
    result = run_worldforge_pipeline(
        scene_id="scene1_pick_place",
        inject_fault=True,
        provider="local",
        quick_ppo=True,
        on_stage_event=lambda ev: events.append(ev),
    )

    assert result["repair_attempts"] >= 2
    assert result["self_healing_applied"] is True
    assert "--- generated_reward.py" in result["repair_diff"]


def test_api_status():
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert data["engine"] == "MuJoCo 3.17.0"
    assert "baseten" in data
    assert data["baseten"]["model"] == "meta-models/muse-spark-open-1026"


def test_api_scenes():
    resp = client.get("/api/scenes")
    assert resp.status_code == 200
    scenes = resp.json()
    assert len(scenes) >= 3
    file_ids = [s["file_id"] for s in scenes]
    assert "scene1_pick_place" in file_ids
    assert "scene2_push_obstacle" in file_ids
    assert "scene3_articulated_drawer" in file_ids


def test_api_scene_preview():
    resp = client.get("/api/scene-preview/scene2_push_obstacle")
    assert resp.status_code == 200
    data = resp.json()
    assert data["scene_id"] == "scene2_push_obstacle"


def test_api_get_scene_details():
    resp = client.get("/api/scenes/scene1_pick_place")
    assert resp.status_code == 200
    sg_data = resp.json()
    assert sg_data["scene_id"] == "pick_place_cube"


def test_api_rejects_path_traversal():
    resp = client.get("/api/scenes/..%2F..%2Ffrontend%2Fpackage")
    assert resp.status_code in (400, 404)
    resp_run = client.post("/api/run", json={"scene_id": "../../frontend/package", "provider": "local"})
    assert resp_run.status_code in (400, 404)


def test_api_cors_blocks_untrusted_origin_reflection():
    resp = client.options(
        "/api/status",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.headers.get("access-control-allow-origin") != "http://evil.example"
    assert resp.headers.get("access-control-allow-credentials") != "true"


def test_api_upload_video_and_reject_html():
    dummy_video_bytes = b"fake mp4 video bytes for testing"
    resp = client.post(
        "/api/upload-video",
        files={"file": ("test_tabletop.mp4", io.BytesIO(dummy_video_bytes), "video/mp4")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["filename"] == "test_tabletop.mp4"

    # Reject .html uploads
    bad_resp = client.post(
        "/api/upload-video",
        files={"file": ("exploit.html", io.BytesIO(b"<script>alert(1)</script>"), "text/html")},
    )
    assert bad_resp.status_code == 400


def test_api_export_contains_gymnasium_env():
    resp = client.get("/api/export/scene1_pick_place")
    assert resp.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    runner_src = zf.read("scene1_pick_place/run_standalone.py").decode("utf-8")
    assert "import gymnasium as gym" in runner_src
    assert "class WorldForgeStandaloneEnv(gym.Env):" in runner_src
    assert "env.reset(" in runner_src
    assert "env.step(" in runner_src


def test_api_run_pipeline():
    payload = {
        "scene_id": "scene1_pick_place",
        "inject_fault": False,
        "provider": "local",
        "quick_ppo": True,
    }
    resp = client.post("/api/run", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["scene_id"] == "scene1_pick_place"


def test_api_sim_endpoints():
    resp = client.post("/api/sim/reset", json={"scene_id": "scene1_pick_place"})
    assert resp.status_code == 200
    st = resp.json()
    assert st["step"] == 0
    assert "red_cube" in st["objectPositions"]

    resp = client.post(
        "/api/sim/step",
        json={"scene_id": "scene1_pick_place", "action": [0.0, 0.0, -0.5, 1.0]},
    )
    assert resp.status_code == 200
    assert resp.json()["step"] == 1

    resp = client.post("/api/sim/oracle-step", json={"scene_id": "scene1_pick_place"})
    assert resp.status_code == 200
    assert resp.json()["step"] == 2

    resp = client.get("/api/sim/trajectory?scene_id=scene1_pick_place&max_steps=10")
    assert resp.status_code == 200
    assert len(resp.json()) == 10


def test_multi_stack_reasoner_prompt_driven():
    """Verify prompt-driven 3-object footprint sorting and multi_stack trajectory synthesis without hardcoded scene configs."""
    geom = LiftedSceneGeometry(
        scene_id="custom_3_particles",
        table_z=0.0,
        camera=CameraSpec(pos=(0.55, -0.55, 0.72), lookat=(0.0, 0.0, 0.06)),
        objects=[
            LiftedObject(id="particle_0", label="small blue cube", primitive="box", size=[0.018, 0.018, 0.018], position=(-0.12, -0.04, 0.019)),
            LiftedObject(id="particle_1", label="wide wooden base", primitive="box", size=[0.036, 0.036, 0.020], position=(0.0, 0.08, 0.021)),
            LiftedObject(id="particle_2", label="medium red block", primitive="box", size=[0.026, 0.026, 0.020], position=(0.12, -0.04, 0.021)),
        ],
    )
    reasoner = MuseSparkReasoner(provider="local")
    synth = reasoner.synthesize_scene_graph(
        geom,
        user_prompt="Stack the 3 particles from widest base to smallest top by understanding their geometry and stability",
    )
    assert synth.task.task_type == "multi_stack"
    assert synth.task.stack_order == ["particle_1", "particle_2", "particle_0"]
    assert "Center-of-Mass stability" in reasoner.last_reasoning_trace
    assert len(synth.task.waypoints) == 16

def test_multi_stack_color_order_and_prompt_extraction():
    """Verify SAM color prompt extraction and relative 'X on top of Y' stacking order."""
    from src.perception.sam_tracker import extract_prompts_from_user_intent

    prompt = "stack the object one by one gray on top of green and then white on top of grey with stability"
    sam_prompts = extract_prompts_from_user_intent(prompt, ["red cube", "coaster"], is_stack_intent=True)
    assert sam_prompts == ["gray object", "green object", "white object"]

    geom = LiftedSceneGeometry(
        scene_id="real_video_stack",
        table_z=0.0,
        camera=CameraSpec(pos=(0.6, -0.6, 0.8), lookat=(0.0, 0.0, 0.05)),
        objects=[
            LiftedObject(
                id="gray_object",
                label="gray object",
                primitive="box",
                size=[0.0276, 0.0276, 0.02],
                position=(-0.0135, 0.0089, 0.021),
                rgba=(0.52, 0.5, 0.48, 1.0),
            ),
            LiftedObject(
                id="green_object",
                label="green object",
                primitive="box",
                size=[0.028, 0.028, 0.02],
                position=(0.0928, 0.0073, 0.021),
                rgba=(0.46, 0.78, 0.56, 1.0),
            ),
            LiftedObject(
                id="white_object",
                label="white object",
                primitive="box",
                size=[0.0238, 0.0238, 0.02],
                position=(-0.1085, 0.007, 0.021),
                rgba=(0.92, 0.9, 0.9, 1.0),
            ),
        ],
    )
    reasoner = MuseSparkReasoner(provider="local")
    synth = reasoner.synthesize_scene_graph(geom, user_prompt=prompt)
    assert synth.task.task_type == "multi_stack"
    assert synth.task.stack_order == ["green_object", "gray_object", "white_object"]
    assert synth.task.manipulated_object_id == "white_object"
    assert len(synth.task.waypoints) == 16
    assert synth.objects[0].rgba == (0.52, 0.5, 0.48, 1.0)
    assert synth.objects[1].rgba == (0.46, 0.78, 0.56, 1.0)
    assert synth.objects[2].rgba == (0.92, 0.9, 0.9, 1.0)

