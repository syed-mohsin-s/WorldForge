"""WorldForge PPO Runner — short training + evaluation + optional video.

Provides ``train_and_evaluate_ppo`` which:
1. Creates a ``DummyVecEnv`` of ``ResidualOracleEnv`` instances.
2. Trains ``PPO("MlpPolicy", ...)`` for a configurable number of timesteps.
3. Evaluates the trained policy for 3 episodes and returns mean return.
4. Optionally records a ~4-second (100–120 frame at 30 FPS) rollout video (MP4).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import gymnasium as gym

from src.compiler.base_env import WorldForgeEnv
from src.schema.scene_graph import SceneGraph


class OracleController:
    """Deterministic proportional controller following scene waypoints."""

    def __init__(
        self,
        waypoints: list[list[float]] | list[tuple[float, float, float, float]],
    ) -> None:
        self.waypoints = [list(wp) for wp in waypoints]
        self.reset()

    def reset(self) -> None:
        self.wp_idx = 0
        self.dwell_remaining = 0
        self.prev_grip_cmd: float | None = None
        self.step_in_wp = 0
        self.completed = False

    def get_action(self, env: Any) -> np.ndarray:
        if not self.waypoints or self.completed:
            last_grip = self.waypoints[-1][3] if self.waypoints else 0.0
            return np.array([0.0, 0.0, 0.0, last_grip], dtype=np.float32)

        if self.wp_idx >= len(self.waypoints):
            self.completed = True
            last_grip = self.waypoints[-1][3] if self.waypoints else 0.0
            return np.array([0.0, 0.0, 0.0, last_grip], dtype=np.float32)

        wx, wy, wz, grip_cmd = self.waypoints[self.wp_idx]

        # Trigger 15-step dwell when grip command changes (e.g. open -> closed)
        if (
            self.prev_grip_cmd is not None
            and grip_cmd != self.prev_grip_cmd
            and self.dwell_remaining == 0
            and self.step_in_wp == 0
        ):
            self.dwell_remaining = 15

        if self.dwell_remaining > 0:
            self.dwell_remaining -= 1
            if self.dwell_remaining == 0:
                self.prev_grip_cmd = grip_cmd
            return np.array([0.0, 0.0, 0.0, grip_cmd], dtype=np.float32)

        self.prev_grip_cmd = grip_cmd
        self.step_in_wp += 1

        grip_pos = env.data.mocap_pos[0].copy()
        err = np.array([wx, wy, wz], dtype=np.float64) - grip_pos
        dist = float(np.linalg.norm(err))

        if dist < 0.005 or self.step_in_wp >= 60:
            self.wp_idx += 1
            self.step_in_wp = 0
            if self.wp_idx >= len(self.waypoints):
                self.completed = True
                return np.array([0.0, 0.0, 0.0, grip_cmd], dtype=np.float32)
            return self.get_action(env)

        direction = err / max(dist, 1e-8)
        max_delta = getattr(env, "max_delta", 0.02)
        gain = min(dist / max_delta, 1.0)
        action_xyz = direction * gain
        action = np.array(
            [action_xyz[0], action_xyz[1], action_xyz[2], grip_cmd],
            dtype=np.float32,
        )
        return np.clip(action, -1.0, 1.0)


class ResidualOracleEnv(gym.Wrapper):
    """Gym wrapper wrapping WorldForgeEnv with bounded oracle residual action."""

    def __init__(
        self,
        env: Any,
        waypoints: list[list[float]] | list[tuple[float, float, float, float]],
        residual_weight: float = 0.15,
    ) -> None:
        super().__init__(env)
        self.controller = OracleController(waypoints)
        self.residual_weight = residual_weight

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        self.controller.reset()
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        u_oracle = self.controller.get_action(self.env.unwrapped)
        u_total = np.clip(u_oracle + self.residual_weight * action, -1.0, 1.0)
        return self.env.step(u_total)


@dataclass
class PPOResult:
    """Result of a PPO training + evaluation run."""

    eval_mean_return: float
    eval_std_return: float
    learning_curve: list[dict[str, float]] = field(default_factory=list)
    video_path: str | None = None


def train_and_evaluate_ppo(
    scene: SceneGraph,
    total_timesteps: int = 1024,
    n_envs: int = 4,
    rollout_video_path: str | None = None,
    eval_episodes: int = 1,
    eval_steps: int = 120,
    render_stride: int = 2,
) -> PPOResult:
    """Train PPO on a WorldForge scene and evaluate.

    Parameters
    ----------
    scene:
        The scene graph to train on.
    total_timesteps:
        Total environment steps for PPO training.
    n_envs:
        Number of parallel envs in DummyVecEnv.
    rollout_video_path:
        If provided, record an MP4/GIF rollout here.
    eval_episodes:
        Number of evaluation episodes.
    eval_steps:
        Max steps per evaluation episode (defaults to 120 for a 4s video at 30 fps).

    Returns
    -------
    PPOResult
        Training and evaluation metrics.
    """
    # Lazy imports to keep module importable without SB3
    import gymnasium as gym
    import torch
    torch.set_num_threads(2)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    # ── Build vectorized residual env ──────────────────────────────────
    def _make_env(idx: int):
        def _init():
            base_e = WorldForgeEnv(scene, render_mode=None)
            res_e = ResidualOracleEnv(base_e, scene.task.waypoints, residual_weight=0.15)
            return gym.wrappers.TimeLimit(res_e, max_episode_steps=60)
        return _init

    vec_env = DummyVecEnv([_make_env(i) for i in range(n_envs)])

    # ── Learning curve callback ───────────────────────────────────────
    learning_curve: list[dict[str, float]] = []

    class _CurveCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._episode_rewards: list[float] = []
            self._current_rewards = np.zeros(n_envs)

        def _on_step(self) -> bool:
            self._current_rewards += self.locals["rewards"]
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    self._episode_rewards.append(float(self._current_rewards[i]))
                    self._current_rewards[i] = 0.0

            if self.num_timesteps % 256 == 0 and self._episode_rewards:
                mean_r = float(np.mean(self._episode_rewards[-10:]))
                learning_curve.append({
                    "timesteps": self.num_timesteps,
                    "mean_reward": mean_r,
                })
            return True

    callback = _CurveCallback()

    # ── Train ─────────────────────────────────────────────────────────
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=64,
        batch_size=64,
        n_epochs=3,
        learning_rate=3e-4,
        ent_coef=0.01,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)
    vec_env.close()

    if not learning_curve and callback._episode_rewards:
        learning_curve.append({
            "timesteps": total_timesteps,
            "mean_reward": float(np.mean(callback._episode_rewards[-10:])),
        })

    # ── Evaluate ──────────────────────────────────────────────────────
    eval_base_env = WorldForgeEnv(
        scene,
        render_mode="rgb_array" if rollout_video_path else None,
    )
    eval_res_env = ResidualOracleEnv(eval_base_env, scene.task.waypoints, residual_weight=0.15)
    eval_env = gym.wrappers.TimeLimit(eval_res_env, max_episode_steps=eval_steps)

    episode_returns: list[float] = []
    video_frames: list[np.ndarray] = []
    last_grip = scene.task.waypoints[-1][3] if scene.task.waypoints else 0.0

    for ep in range(eval_episodes):
        obs, _ = eval_env.reset(seed=100 + ep)
        ep_return = 0.0
        terminated = False
        truncated = False

        for step_i in range(eval_steps):
            if not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                ep_return += reward
            elif ep == 0 and rollout_video_path:
                # Settle physics with idle action to complete the 120-frame (4s) recording
                eval_base_env.step(np.array([0.0, 0.0, 0.0, last_grip], dtype=np.float32))

            # Record frames for video on first episode only (strided for 2x faster export)
            if ep == 0 and rollout_video_path and (step_i % render_stride == 0):
                try:
                    frame = eval_base_env.render_camera()
                    if frame is not None:
                        video_frames.append(frame)
                except Exception:
                    pass

            if (terminated or truncated) and (not rollout_video_path or ep > 0):
                break

        episode_returns.append(ep_return)

    eval_env.close()

    # ── Save video ────────────────────────────────────────────────────
    video_path_out: str | None = None
    if rollout_video_path and video_frames:
        fps_out = 20 if render_stride == 2 else 30
        video_path_out = _save_video(video_frames, rollout_video_path, fps=fps_out)

    return PPOResult(
        eval_mean_return=float(np.mean(episode_returns)),
        eval_std_return=float(np.std(episode_returns)),
        learning_curve=learning_curve,
        video_path=video_path_out,
    )


def record_oracle_rollout(
    scene: SceneGraph,
    output_path: str,
    max_steps: int = 120,
    fps: int = 30,
    render_stride: int = 2,
) -> str:
    """Record pure Waypoint Oracle baseline rollout video (MP4)."""
    env = WorldForgeEnv(scene, render_mode="rgb_array")
    controller = OracleController(scene.task.waypoints)
    obs, _ = env.reset(seed=42)
    frames: list[np.ndarray] = []
    terminated = False
    last_grip = scene.task.waypoints[-1][3] if scene.task.waypoints else 0.0

    for step_i in range(max_steps):
        if not terminated:
            action = controller.get_action(env)
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                terminated = True
        else:
            env.step(np.array([0.0, 0.0, 0.0, last_grip], dtype=np.float32))

        if step_i % render_stride == 0:
            frame = env.render_camera()
            if frame is not None:
                frames.append(frame)

    env.close()
    fps_out = 20 if render_stride == 2 else fps
    return _save_video(frames, output_path, fps=fps_out)


def _save_video(frames: list[np.ndarray], path: str, fps: int = 30) -> str:
    """Save frames as MP4 (preferred) or GIF fallback."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v3 as iio
        if out_path.suffix.lower() in (".mp4", ""):
            if out_path.suffix == "":
                out_path = out_path.with_suffix(".mp4")
            iio.imwrite(str(out_path), frames, fps=fps)
            return str(out_path)
    except Exception:
        pass

    try:
        import imageio.v3 as iio
        gif_path = out_path.with_suffix(".gif")
        iio.imwrite(str(gif_path), frames, duration=int(1000 / fps), loop=0)
        return str(gif_path)
    except Exception:
        pass

    try:
        import imageio.v3 as iio
        png_path = out_path.with_suffix(".png")
        iio.imwrite(str(png_path), frames[0])
        return str(png_path)
    except Exception:
        return ""
