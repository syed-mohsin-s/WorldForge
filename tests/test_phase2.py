"""Phase 2 integration tests — Reward, Validation Gates 1–5, PPO, Repair Loop.

Test matrix
-----------
* Gates 1–4 on all 3 golden scenes
* Gate 5 (PPO short run) end-to-end
* Self-healing loop with injected broken reward
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from src.compiler.base_env import WorldForgeEnv
from src.compiler.generated_reward import compute_reward
from src.schema.scene_graph import SceneGraph
from src.validation.gates import ValidationSuite, ValidationReport
from src.validation.repair_loop import (
    CostTracker,
    run_self_healing_loop,
)

SCENE_DIR = Path(__file__).resolve().parents[1] / "configs" / "scenes"

GOLDEN_SCENES = [
    "scene1_pick_place.json",
    "scene2_push_obstacle.json",
    "scene3_articulated_drawer.json",
]


def _load_scene(name: str) -> SceneGraph:
    path = SCENE_DIR / name
    with open(path) as f:
        data = json.load(f)
    return SceneGraph(**data)


# =====================================================================
# Dense Reward Tests
# =====================================================================

class TestDenseReward:
    """Verify the multi-stage reward functions are well-behaved."""

    @pytest.mark.parametrize("scene_file", GOLDEN_SCENES)
    def test_reward_returns_finite(self, scene_file: str) -> None:
        scene = _load_scene(scene_file)
        env = WorldForgeEnv(scene, reward_fn=compute_reward, render_mode=None)
        env.reset(seed=42)
        for _ in range(50):
            action = env.action_space.sample()
            _, reward, terminated, _, info = env.step(action)
            assert math.isfinite(reward), f"Non-finite reward: {reward}"
            if terminated:
                env.reset()
        env.close()

    @pytest.mark.parametrize("scene_file", GOLDEN_SCENES)
    def test_reward_has_variance(self, scene_file: str) -> None:
        """Dense reward must not be constant across random actions."""
        scene = _load_scene(scene_file)
        env = WorldForgeEnv(scene, reward_fn=compute_reward, render_mode=None)
        env.reset(seed=42)
        rewards = []
        for _ in range(100):
            action = env.action_space.sample()
            _, reward, terminated, _, _ = env.step(action)
            rewards.append(reward)
            if terminated:
                env.reset()
        env.close()
        assert np.std(rewards) > 1e-4, f"Reward std too low: {np.std(rewards)}"


# =====================================================================
# Validation Gate Tests (Gates 1–4 on all 3 scenes)
# =====================================================================

class TestGate1Stability:
    @pytest.mark.parametrize("scene_file", GOLDEN_SCENES)
    def test_gate1_passes(self, scene_file: str) -> None:
        scene = _load_scene(scene_file)
        suite = ValidationSuite(scene)
        result = suite._gate_1()
        assert result["passed"], f"Gate 1 failed: {result}"


class TestGate2StaticEquilibrium:
    @pytest.mark.parametrize("scene_file", GOLDEN_SCENES)
    def test_gate2_passes(self, scene_file: str) -> None:
        scene = _load_scene(scene_file)
        suite = ValidationSuite(scene)
        result = suite._gate_2()
        assert result["passed"], f"Gate 2 failed: {result}"


class TestGate3RewardVariance:
    @pytest.mark.parametrize("scene_file", GOLDEN_SCENES)
    def test_gate3_passes(self, scene_file: str) -> None:
        scene = _load_scene(scene_file)
        suite = ValidationSuite(scene)
        result = suite._gate_3()
        assert result["passed"], f"Gate 3 failed: {result}"
        assert result["reward_std"] > 1e-4


class TestGate4WaypointOracle:
    @pytest.mark.parametrize("scene_file", GOLDEN_SCENES)
    def test_gate4_passes(self, scene_file: str) -> None:
        scene = _load_scene(scene_file)
        suite = ValidationSuite(scene)
        # Run gate 3 first to populate _mean_random_return
        g3 = suite._gate_3()
        assert g3["passed"], f"Gate 3 prerequisite failed: {g3}"
        result = suite._gate_4()
        assert result["passed"], f"Gate 4 failed: {result}"


# =====================================================================
# Gate 5: PPO Short Run (single scene for speed)
# =====================================================================

class TestGate5PPO:
    def test_gate5_ppo_runs_and_beats_random(self) -> None:
        """PPO short run on scene1 — must beat random baseline."""
        from src.validation.ppo_runner import train_and_evaluate_ppo

        scene = _load_scene("scene1_pick_place.json")

        # Get random baseline
        suite = ValidationSuite(scene)
        g3 = suite._gate_3()
        assert g3["passed"]
        random_return = g3["mean_random_return"]

        # Train PPO
        result = train_and_evaluate_ppo(
            scene,
            total_timesteps=8192,
            n_envs=4,
        )

        assert result.eval_mean_return > random_return, (
            f"PPO return {result.eval_mean_return:.2f} does not beat "
            f"random {random_return:.2f}"
        )
        # Verify learning curve was logged
        assert len(result.learning_curve) > 0, "No learning curve data"

    def test_gate5_exports_rendered_frame(self) -> None:
        """Verify render_camera produces a valid numpy array."""
        scene = _load_scene("scene1_pick_place.json")
        env = WorldForgeEnv(scene, render_mode="rgb_array")
        env.reset(seed=42)
        env.step(env.action_space.sample())
        frame = env.render_camera()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3  # H x W x 3
        assert frame.shape[2] == 3
        assert frame.shape[0] > 0 and frame.shape[1] > 0
        env.close()


# =====================================================================
# Self-Healing Repair Loop
# =====================================================================

class TestRepairLoop:
    def test_cost_tracker_basics(self) -> None:
        ct = CostTracker(pricing_tier="standard")
        ct.log_api_call(1_000_000, 500_000, purpose="test")
        ct.log_compute(100.0)
        cost = ct.total_cost_usd
        # Standard: 1M * 1.25/1M + 0.5M * 4.25/1M + 100 * 0.00003942
        expected = 1.25 + 2.125 + 0.003942
        assert abs(cost - expected) < 0.01, f"Cost {cost} != expected {expected}"

    def test_cost_tracker_contributor_tier(self) -> None:
        ct = CostTracker(pricing_tier="contributor")
        ct.log_api_call(1_000_000, 1_000_000, purpose="test")
        cost = ct.total_cost_usd
        # Contributor: 1M * 0.10/1M + 1M * 0.20/1M
        expected = 0.10 + 0.20
        assert abs(cost - expected) < 0.01

    def test_self_healing_catches_broken_reward(self) -> None:
        """Inject a constant-zero reward (fails Gate 3), verify auto-repair."""
        scene = _load_scene("scene1_pick_place.json")

        # Broken reward: always returns 0.0
        def broken_reward(model, data, sg):
            return 0.0, False, {}

        result = run_self_healing_loop(
            scene,
            max_tries=4,
            use_live_llm=False,
            reward_fn_override=broken_reward,
            run_gate5=False,
            verbose=False,
        )

        assert result.success, (
            f"Self-healing failed after {result.attempts} attempts. "
            f"Repairs: {result.repairs_applied}. "
            f"Final report: gate {result.final_report.failed_gate}, "
            f"{result.final_report.diagnostics}"
        )
        # Should have been repaired — the broken reward was on attempt 1 only
        assert result.attempts >= 2, "Should have needed at least 2 attempts"
        assert len(result.repairs_applied) >= 1, "Should have applied a repair"
        assert result.cost_tracker.total_cost_usd >= 0

    def test_healthy_scene_passes_without_repair(self) -> None:
        """A valid scene should pass on attempt 1 with no repairs."""
        scene = _load_scene("scene1_pick_place.json")
        result = run_self_healing_loop(
            scene,
            max_tries=4,
            use_live_llm=False,
            run_gate5=False,
            verbose=False,
        )
        assert result.success
        assert result.attempts == 1
        assert len(result.repairs_applied) == 0


# =====================================================================
# Full Suite Run (Gates 1–4, skipping 5 for speed)
# =====================================================================

class TestFullSuiteGates1to4:
    @pytest.mark.parametrize("scene_file", GOLDEN_SCENES)
    def test_gates_1_through_4_pass(self, scene_file: str) -> None:
        scene = _load_scene(scene_file)
        suite = ValidationSuite(scene, verbose=False)
        report = suite.run_all(run_gate5=False)
        assert report.passed, (
            f"Validation failed at gate {report.failed_gate}: "
            f"{report.diagnostics}"
        )

