"""WorldForge Validation Gates 1–5.

Five sequential gates that a compiled scene must pass before being
considered a valid, trainable environment.

Gate 1 — Stability / Zero NaNs
Gate 2 — Static Equilibrium / No Resting Drift
Gate 3 — Non-Degenerate Reward Variance
Gate 4 — Waypoint Oracle Solvability
Gate 5 — Short PPO Convergence Beats Random
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from src.compiler.base_env import WorldForgeEnv
from src.schema.scene_graph import SceneGraph


@dataclass
class ValidationReport:
    """Result of running the full validation suite."""

    passed: bool
    failed_gate: int | None
    gate_results: dict[int, dict] = field(default_factory=dict)
    diagnostics: str = ""


class ValidationSuite:
    """Run gates 1–5 sequentially; stop on first failure."""

    def __init__(self, scene: SceneGraph, verbose: bool = False) -> None:
        self.scene = scene
        self.verbose = verbose
        # Populated by gate 3 for use in gates 4 and 5
        self._mean_random_return: float = 0.0

    def run_all(
        self,
        run_gate5: bool = True,
        start_from_gate: int = 1,
        cached_gate_results: dict[int, dict] | None = None,
    ) -> ValidationReport:
        """Run all gates sequentially starting from start_from_gate. Returns on first failure."""
        report = ValidationReport(passed=True, failed_gate=None)
        if cached_gate_results:
            report.gate_results.update(cached_gate_results)

        for gate_num in range(start_from_gate, 6):
            if gate_num == 5 and not run_gate5:
                report.gate_results[5] = {"skipped": True}
                continue

            method = getattr(self, f"_gate_{gate_num}")
            try:
                result = method()
            except Exception as exc:
                result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

            report.gate_results[gate_num] = result

            if not result.get("passed", False):
                report.passed = False
                report.failed_gate = gate_num
                report.diagnostics = self._format_diagnostics(gate_num, result)
                if self.verbose:
                    print(f"  ✗ Gate {gate_num} FAILED: {report.diagnostics}")
                break
            elif self.verbose:
                print(f"  ✓ Gate {gate_num} passed")

        return report

    # ------------------------------------------------------------------
    # Gate 1: Stability / Zero NaNs
    # ------------------------------------------------------------------
    def _gate_1(self) -> dict[str, Any]:
        env = WorldForgeEnv(self.scene, render_mode=None)
        env.reset(seed=42)

        max_qacc = 0.0
        for step_i in range(500):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)

            if np.any(np.isnan(env.data.qpos)) or np.any(np.isinf(env.data.qpos)):
                env.close()
                return {"passed": False, "reason": f"NaN/Inf in qpos at step {step_i}"}
            if np.any(np.isnan(env.data.qvel)) or np.any(np.isinf(env.data.qvel)):
                env.close()
                return {"passed": False, "reason": f"NaN/Inf in qvel at step {step_i}"}
            if np.any(np.isnan(env.data.qacc)) or np.any(np.isinf(env.data.qacc)):
                env.close()
                return {"passed": False, "reason": f"NaN/Inf in qacc at step {step_i}"}

            max_qacc = max(max_qacc, float(np.max(np.abs(env.data.qacc))))

            if terminated or truncated:
                env.reset()

        env.close()

        if max_qacc >= 1e6:
            return {"passed": False, "reason": f"max(|qacc|) = {max_qacc:.1f} >= 1e6"}

        return {"passed": True, "max_qacc": max_qacc}

    # ------------------------------------------------------------------
    # Gate 2: Static Equilibrium / No Resting Drift
    # ------------------------------------------------------------------
    def _gate_2(self) -> dict[str, Any]:
        env = WorldForgeEnv(self.scene, render_mode=None)
        env.reset(seed=42)

        # Record initial positions of non-manipulated, non-child,
        # non-gripper resting objects
        manipulated_id = self.scene.task.manipulated_object_id
        resting_ids: list[str] = []
        for obj in self.scene.objects:
            if obj.parent_id is not None:
                continue
            if obj.id == manipulated_id:
                continue
            if obj.is_static:
                continue
            resting_ids.append(obj.id)

        # Initial settle: 25 steps holding gripper stationary so resting objects settle
        stationary_action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        for _ in range(25):
            env.step(stationary_action)

        initial_pos: dict[str, np.ndarray] = {}
        for oid in resting_ids:
            bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, oid)
            initial_pos[oid] = env.data.xpos[bid].copy()

        # Step 250 times with gripper stationary and closed
        for _ in range(250):
            env.step(stationary_action)

        # Check drift
        max_horiz_drift = 0.0
        max_vert_drift = 0.0
        drifters: list[str] = []
        for oid in resting_ids:
            bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, oid)
            final_pos = env.data.xpos[bid].copy()
            diff = final_pos - initial_pos[oid]
            horiz = float(np.linalg.norm(diff[:2]))
            vert = abs(float(diff[2]))
            max_horiz_drift = max(max_horiz_drift, horiz)
            max_vert_drift = max(max_vert_drift, vert)
            if horiz > 0.002 or vert > 0.001:
                drifters.append(f"{oid}(h={horiz:.4f},v={vert:.4f})")

        env.close()

        if drifters:
            return {
                "passed": False,
                "reason": f"Resting objects drifted: {', '.join(drifters)}",
                "max_horiz_drift": max_horiz_drift,
                "max_vert_drift": max_vert_drift,
            }

        return {
            "passed": True,
            "max_horiz_drift": max_horiz_drift,
            "max_vert_drift": max_vert_drift,
        }

    # ------------------------------------------------------------------
    # Gate 3: Non-Degenerate Reward Variance
    # ------------------------------------------------------------------
    def _gate_3(self) -> dict[str, Any]:
        env = WorldForgeEnv(self.scene, render_mode=None)

        all_rewards: list[float] = []
        episode_returns: list[float] = []

        for ep in range(3):
            obs, _ = env.reset(seed=42 + ep)
            ep_return = 0.0
            for _ in range(30):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)

                if not math.isfinite(reward):
                    env.close()
                    return {"passed": False, "reason": f"Non-finite reward in episode {ep}"}

                all_rewards.append(reward)
                ep_return += reward

                if terminated or truncated:
                    obs, _ = env.reset()
            episode_returns.append(ep_return)

        env.close()

        reward_std = float(np.std(all_rewards))
        mean_return = float(np.mean(episode_returns))

        if reward_std <= 1e-4:
            return {
                "passed": False,
                "reason": f"Reward std={reward_std:.6f} <= 1e-4 (degenerate)",
                "reward_std": reward_std,
                "mean_random_return": mean_return,
            }

        # Store for gates 4 and 5
        self._mean_random_return = mean_return

        return {
            "passed": True,
            "reward_std": reward_std,
            "mean_random_return": mean_return,
        }

    # ------------------------------------------------------------------
    # Gate 4: Waypoint Oracle Solvability
    # ------------------------------------------------------------------
    def _gate_4(self) -> dict[str, Any]:
        waypoints = self.scene.task.waypoints
        if not waypoints:
            return {"passed": True, "reason": "No waypoints defined; skipped."}

        env = WorldForgeEnv(self.scene, render_mode=None)
        obs, _ = env.reset(seed=42)

        total_reward = 0.0
        terminated_flag = False
        max_steps_per_wp = 60
        dwell_steps = 15
        tolerance = 0.005  # 5 mm

        prev_grip_cmd: float | None = None

        for wp_idx, wp in enumerate(waypoints):
            wx, wy, wz, grip_cmd = wp

            # Detect gripper command change → dwell
            if prev_grip_cmd is not None and grip_cmd != prev_grip_cmd:
                for _ in range(dwell_steps):
                    action = np.array([0.0, 0.0, 0.0, grip_cmd], dtype=np.float32)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if terminated:
                        terminated_flag = True
                        break
                if terminated_flag:
                    break

            prev_grip_cmd = grip_cmd

            # Drive toward waypoint with proportional controller
            for step_i in range(max_steps_per_wp):
                grip = env.data.mocap_pos[0].copy()
                err = np.array([wx, wy, wz]) - grip
                dist = float(np.linalg.norm(err))

                if dist < tolerance:
                    break

                # Proportional control: normalize error, clamp to [-1,1]
                direction = err / max(dist, 1e-8)
                # Scale so that max_delta (0.02m) maps to action=1.0
                gain = min(dist / env.max_delta, 1.0)
                action_xyz = direction * gain
                action = np.array(
                    [action_xyz[0], action_xyz[1], action_xyz[2], grip_cmd],
                    dtype=np.float32,
                )
                action = np.clip(action, -1.0, 1.0)

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if terminated:
                    terminated_flag = True
                    break

            if terminated_flag:
                break

        env.close()

        # Success criteria: terminated OR return > mean_random_return + 10
        threshold = self._mean_random_return + 10.0
        oracle_passed = terminated_flag or (total_reward > threshold)

        if not oracle_passed:
            return {
                "passed": False,
                "reason": (
                    f"Oracle failed: terminated={terminated_flag}, "
                    f"return={total_reward:.2f}, "
                    f"threshold={threshold:.2f} "
                    f"(mean_random={self._mean_random_return:.2f}+10)"
                ),
                "total_reward": total_reward,
                "terminated": terminated_flag,
            }

        return {
            "passed": True,
            "total_reward": total_reward,
            "terminated": terminated_flag,
            "threshold": threshold,
        }

    # ------------------------------------------------------------------
    # Gate 5: Short PPO Convergence Beats Random
    # ------------------------------------------------------------------
    def _gate_5(self) -> dict[str, Any]:
        from src.validation.ppo_runner import train_and_evaluate_ppo

        result = train_and_evaluate_ppo(
            self.scene,
            total_timesteps=8192,
            n_envs=4,
        )

        ppo_return = result.eval_mean_return
        beats_random = ppo_return > self._mean_random_return

        if not beats_random:
            return {
                "passed": False,
                "reason": (
                    f"PPO return ({ppo_return:.2f}) does not beat "
                    f"random baseline ({self._mean_random_return:.2f})"
                ),
                "ppo_return": ppo_return,
                "random_return": self._mean_random_return,
                "learning_curve": result.learning_curve,
            }

        return {
            "passed": True,
            "ppo_return": ppo_return,
            "random_return": self._mean_random_return,
            "learning_curve": result.learning_curve,
        }

    # ------------------------------------------------------------------
    # Diagnostics formatter
    # ------------------------------------------------------------------
    @staticmethod
    def _format_diagnostics(gate: int, result: dict) -> str:
        reason = result.get("reason", result.get("error", "Unknown failure"))
        lines = [f"Gate {gate} failed: {reason}"]
        for k, v in result.items():
            if k not in ("passed", "reason", "error"):
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

