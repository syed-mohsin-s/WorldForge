"""Phase 1 integration tests — Scene Graph → MJCF → Gymnasium env.

For each of the 3 golden scenes:
  1. Load the JSON and validate it through ``SceneGraph``.
  2. Compile to MJCF XML and check it parses with ``mujoco.MjModel``.
  3. Instantiate ``WorldForgeEnv``, step 500 times with random actions,
     and assert zero exceptions / NaN observations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from src.compiler.base_env import WorldForgeEnv
from src.compiler.mjcf_builder import compile_scene_to_mjcf
from src.schema.scene_graph import SceneGraph

SCENE_DIR = Path(__file__).resolve().parents[1] / "configs" / "scenes"

GOLDEN_SCENES = [
    "scene1_pick_place.json",
    "scene2_push_obstacle.json",
    "scene3_articulated_drawer.json",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_scene(name: str) -> SceneGraph:
    path = SCENE_DIR / name
    with open(path) as f:
        data = json.load(f)
    return SceneGraph(**data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSceneGraphValidation:
    """Pydantic model validation and auto-snap behaviour."""

    @pytest.mark.parametrize("scene_file", GOLDEN_SCENES)
    def test_loads_without_error(self, scene_file: str) -> None:
        scene = _load_scene(scene_file)
        assert scene.scene_id

    def test_auto_snap_raises_floor(self) -> None:
        """An object placed below the table surface should be snapped up."""
        scene = _load_scene("scene1_pick_place.json")
        for obj in scene.objects:
            if obj.parent_id is not None:
                continue
            # bottom of object must be >= table_z + 0.001
            if obj.primitive == "box":
                half_h = obj.size[2]
            elif obj.primitive == "cylinder":
                half_h = obj.size[1]
            else:
                half_h = obj.size[0]
            bottom = obj.position[2] - half_h
            assert bottom >= scene.table_z + 0.0005, (
                f"Object {obj.id} bottom={bottom:.4f} is below table"
            )


class TestMJCFCompilation:
    """MJCF XML generation and MuJoCo loading."""

    @pytest.mark.parametrize("scene_file", GOLDEN_SCENES)
    def test_compiles_to_valid_xml(self, scene_file: str) -> None:
        scene = _load_scene(scene_file)
        xml = compile_scene_to_mjcf(scene)
        assert xml.startswith("<?xml")
        # Must load without error
        model = mujoco.MjModel.from_xml_string(xml)
        assert model.nq > 0

    @pytest.mark.parametrize("scene_file", GOLDEN_SCENES)
    def test_has_render_cam(self, scene_file: str) -> None:
        scene = _load_scene(scene_file)
        xml = compile_scene_to_mjcf(scene)
        model = mujoco.MjModel.from_xml_string(xml)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "render_cam")
        assert cam_id >= 0

    @pytest.mark.parametrize("scene_file", GOLDEN_SCENES)
    def test_has_gripper(self, scene_file: str) -> None:
        scene = _load_scene(scene_file)
        xml = compile_scene_to_mjcf(scene)
        model = mujoco.MjModel.from_xml_string(xml)
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mocap")
        grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        assert ee_id >= 0
        assert grip_id >= 0

    @pytest.mark.parametrize("scene_file", GOLDEN_SCENES)
    def test_deterministic_output(self, scene_file: str) -> None:
        """Same scene must produce identical XML on two runs."""
        scene = _load_scene(scene_file)
        xml1 = compile_scene_to_mjcf(scene, randomize=False)
        xml2 = compile_scene_to_mjcf(scene, randomize=False)
        assert xml1 == xml2


class TestWorldForgeEnv:
    """Full integration: step the env with random actions."""

    @pytest.mark.parametrize("scene_file", GOLDEN_SCENES)
    def test_step_500_no_exceptions(self, scene_file: str) -> None:
        scene = _load_scene(scene_file)
        env = WorldForgeEnv(scene, render_mode=None)

        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert not np.any(np.isnan(obs)), "NaN in initial observation"

        for i in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert not np.any(np.isnan(obs)), f"NaN in observation at step {i}"
            assert math.isfinite(reward), f"Non-finite reward at step {i}"
            if terminated or truncated:
                obs, info = env.reset()

        env.close()

    @pytest.mark.parametrize("scene_file", GOLDEN_SCENES)
    def test_observation_space_match(self, scene_file: str) -> None:
        scene = _load_scene(scene_file)
        env = WorldForgeEnv(scene, render_mode=None)
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), (
            f"obs shape {obs.shape} not in space {env.observation_space.shape}"
        )
        env.close()

    @pytest.mark.parametrize("scene_file", GOLDEN_SCENES)
    def test_action_space_bounds(self, scene_file: str) -> None:
        scene = _load_scene(scene_file)
        env = WorldForgeEnv(scene, render_mode=None)
        assert env.action_space.shape == (4,)
        assert np.allclose(env.action_space.low, -1.0)
        assert np.allclose(env.action_space.high, 1.0)
        env.close()

