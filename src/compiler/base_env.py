"""WorldForge Base Gymnasium Environment.

Wraps a compiled MuJoCo scene with 4-DoF Cartesian mocap delta control
and a flat observation vector.

Action space
    ``Box([-1,-1,-1,-1], [1,1,1,1], float32)``
    Dimensions: ``[dx, dy, dz, gripper_cmd]``
    • xyz deltas are scaled to a maximum of 2 cm per ``step()`` call.
    • ``gripper_cmd``:  −1 → fully closed, +1 → fully open.

Observation space (flat float32 vector)
    [gripper_pos(3), finger_state(2), obj_pos(3*N), obj_vel(6*N), target(3)]

Reward
    Pluggable via ``reward_fn(model, data, scene) -> (reward, terminated, info)``.
    Default uses ``generated_reward.compute_reward`` if available, else
    negative Euclidean distance from manipulated object to target.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

import gymnasium as gym
import mujoco
import numpy as np

from src.compiler.mjcf_builder import compile_scene_to_mjcf
from src.schema.scene_graph import SceneGraph


# Type alias for the pluggable reward function
RewardFn = Callable[[mujoco.MjModel, mujoco.MjData, SceneGraph], tuple[float, bool, dict]]


def _get_default_reward() -> RewardFn:
    """Return the best available default reward function.

    Prefers ``generated_reward.compute_reward`` (Phase 2 dense reward);
    falls back to simple negative-distance if unavailable.
    """
    try:
        from src.compiler.generated_reward import compute_reward
        return compute_reward
    except ImportError:
        return _fallback_reward


def _fallback_reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: SceneGraph,
) -> tuple[float, bool, dict]:
    """Negative L2 distance from manipulated object to target (fallback)."""
    obj_id = scene.task.manipulated_object_id
    target = scene.task.target_position

    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_id)
        obj_pos = data.xpos[body_id].copy()
    except Exception:
        obj_pos = np.zeros(3)

    if target is not None:
        dist = float(np.linalg.norm(obj_pos - np.array(target)))
        reward = -dist
        terminated = dist < 0.02
    else:
        reward = 0.0
        terminated = False

    return reward, terminated, {"obj_pos": obj_pos.tolist()}


class WorldForgeEnv(gym.Env):
    """Gymnasium environment backed by a WorldForge SceneGraph."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        scene: SceneGraph,
        reward_fn: RewardFn | None = None,
        randomize: bool = False,
        render_mode: str | None = "rgb_array",
        n_substeps: int = 10,
        max_delta_m: float = 0.02,
    ) -> None:
        super().__init__()

        self.scene = scene
        self.reward_fn = reward_fn if reward_fn is not None else _get_default_reward()
        self.randomize = randomize
        self.render_mode = render_mode
        self.n_substeps = n_substeps
        self.max_delta = max_delta_m

        # Compile and load
        xml = compile_scene_to_mjcf(scene, randomize=randomize)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Cache body ids for observation building
        self._obj_body_ids: list[int] = []
        for obj in scene.objects:
            try:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj.id)
                self._obj_body_ids.append(bid)
            except Exception:
                pass

        n_objs = len(self._obj_body_ids)

        # gripper_pos(3) + finger_state(2) + obj_pos(3*N) + obj_vel(6*N) + target(3)
        obs_dim = 3 + 2 + 3 * n_objs + 6 * n_objs + 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 4-DoF: [dx, dy, dz, gripper_cmd]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Renderer (lazy — created on first render)
        self._renderer: mujoco.Renderer | None = None
        self._cam_id: int = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "render_cam"
        )

        # Mocap body index (always 0 in our XML — only one mocap body)
        self._mocap_id = 0

        # Gripper actuator indices
        self._finger_act_L = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_actuator"
        )
        self._finger_act_R = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_actuator_R"
        )

        # Gripper joint indices (for observation)
        self._finger_jnt_L = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint_L"
        )
        self._finger_jnt_R = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint_R"
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0)

        # Scale xyz deltas
        delta = action[:3] * self.max_delta

        # Move mocap target
        self.data.mocap_pos[self._mocap_id] += delta

        # Gripper command: map [-1,1] → [0, 0.04] (finger aperture)
        grip_target = float((action[3] + 1.0) / 2.0 * 0.04)
        self.data.ctrl[self._finger_act_L] = grip_target
        self.data.ctrl[self._finger_act_R] = grip_target

        # Step physics
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._obs()
        reward, terminated, info = self.reward_fn(self.model, self.data, self.scene)
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        return self.render_camera()

    def render_camera(self) -> np.ndarray:
        """Render an RGB image from ``render_cam``.

        Works on both Windows (wgl default) and Linux (egl).
        On headless Linux, falls back to ``MUJOCO_GL=egl`` or ``osmesa``.
        """
        h = self.scene.camera.resolution[1]
        w = self.scene.camera.resolution[0]
        if self._renderer is None:
            # Ensure offscreen rendering works on headless Linux
            if sys.platform == "linux" and "DISPLAY" not in os.environ:
                if "MUJOCO_GL" not in os.environ:
                    os.environ["MUJOCO_GL"] = "egl"
            try:
                self._renderer = mujoco.Renderer(self.model, height=h, width=w)
            except Exception:
                if sys.platform == "linux" and os.environ.get("MUJOCO_GL") != "osmesa":
                    os.environ["MUJOCO_GL"] = "osmesa"
                    self._renderer = mujoco.Renderer(self.model, height=h, width=w)
                else:
                    raise
        self._renderer.update_scene(self.data, camera=self._cam_id)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        parts: list[np.ndarray] = []

        # Gripper position (mocap pos)
        parts.append(self.data.mocap_pos[self._mocap_id].copy())

        # Finger state (joint positions for L and R)
        finger_state = np.array(
            [
                self.data.qpos[self.model.jnt_qposadr[self._finger_jnt_L]],
                self.data.qpos[self.model.jnt_qposadr[self._finger_jnt_R]],
            ],
            dtype=np.float64,
        )
        parts.append(finger_state)

        # Object positions
        for bid in self._obj_body_ids:
            parts.append(self.data.xpos[bid].copy())

        # Object velocities (6-DoF: 3 linear + 3 angular)
        for bid in self._obj_body_ids:
            parts.append(self.data.cvel[bid].copy())

        # Target
        target = self.scene.task.target_position
        if target is not None:
            parts.append(np.array(target, dtype=np.float64))
        else:
            parts.append(np.zeros(3))

        return np.concatenate(parts).astype(np.float32)
