"""WorldForge PPO Runner — short training + evaluation + optional video.

Provides ``train_and_evaluate_ppo`` which:
1. Creates a ``DummyVecEnv`` of ``WorldForgeEnv`` instances.
2. Trains ``PPO("MlpPolicy", ...)`` for a configurable number of timesteps.
3. Evaluates the trained policy for 3 episodes and returns mean return.
4. Optionally records a 60-step rollout video (MP4 via imageio, GIF fallback).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.compiler.base_env import WorldForgeEnv
from src.schema.scene_graph import SceneGraph


@dataclass
class PPOResult:
    """Result of a PPO training + evaluation run."""

    eval_mean_return: float
    eval_std_return: float
    learning_curve: list[dict[str, float]] = field(default_factory=list)
    video_path: str | None = None


def train_and_evaluate_ppo(
    scene: SceneGraph,
    total_timesteps: int = 8192,
    n_envs: int = 4,
    rollout_video_path: str | None = None,
    eval_episodes: int = 3,
    eval_steps: int = 60,
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
        Max steps per evaluation episode.

    Returns
    -------
    PPOResult
        Training and evaluation metrics.
    """
    # Lazy imports to keep module importable without SB3
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback

    # ── Build vectorized env ──────────────────────────────────────────
    def _make_env(idx: int):
        def _init():
            env = WorldForgeEnv(scene, render_mode=None)
            return gym.wrappers.TimeLimit(env, max_episode_steps=50)
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
            # Accumulate rewards
            self._current_rewards += self.locals["rewards"]
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    self._episode_rewards.append(float(self._current_rewards[i]))
                    self._current_rewards[i] = 0.0

            # Log every 1024 steps
            if self.num_timesteps % 1024 == 0 and self._episode_rewards:
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
        n_steps=128,
        batch_size=64,
        n_epochs=5,
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
    eval_env = gym.wrappers.TimeLimit(eval_base_env, max_episode_steps=eval_steps)

    episode_returns: list[float] = []
    video_frames: list[np.ndarray] = []

    for ep in range(eval_episodes):
        obs, _ = eval_env.reset(seed=100 + ep)
        ep_return = 0.0
        for step_i in range(eval_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_return += reward

            # Collect frames for video on first episode only
            if ep == 0 and rollout_video_path:
                try:
                    frame = eval_base_env.render_camera()
                    if frame is not None:
                        video_frames.append(frame)
                except Exception:
                    pass  # Skip frame on render failure

            if terminated or truncated:
                break
        episode_returns.append(ep_return)

    eval_env.close()

    # ── Save video ────────────────────────────────────────────────────
    video_path_out: str | None = None
    if rollout_video_path and video_frames:
        video_path_out = _save_video(video_frames, rollout_video_path)

    return PPOResult(
        eval_mean_return=float(np.mean(episode_returns)),
        eval_std_return=float(np.std(episode_returns)),
        learning_curve=learning_curve,
        video_path=video_path_out,
    )


def _save_video(frames: list[np.ndarray], path: str) -> str:
    """Save frames as MP4 (preferred) or GIF fallback."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v3 as iio
        # Try MP4 first
        if out_path.suffix.lower() in (".mp4", ""):
            if out_path.suffix == "":
                out_path = out_path.with_suffix(".mp4")
            iio.imwrite(str(out_path), frames, fps=25)
            return str(out_path)
    except Exception:
        pass

    try:
        import imageio.v3 as iio
        # GIF fallback
        gif_path = out_path.with_suffix(".gif")
        iio.imwrite(str(gif_path), frames, duration=40, loop=0)
        return str(gif_path)
    except Exception:
        pass

    # Last resort: save first frame as PNG
    try:
        import imageio.v3 as iio
        png_path = out_path.with_suffix(".png")
        iio.imwrite(str(png_path), frames[0])
        return str(png_path)
    except Exception:
        return ""

