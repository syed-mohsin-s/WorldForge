"""WorldForge Modal Sandbox Cloud Runner.

Executes validation, self-healing repair loops, and PPO training inside a Modal
Linux container (modal.Sandbox with MUJOCO_GL=egl, 4 CPU cores, 8 GiB RAM) when
MODAL_TOKEN_ID and MODAL_REMOTE_EXEC=true are configured, or runs locally on
Windows/Linux when offline without leaking idle cloud sandboxes.

Key Features:
- Clean lifecycle management (`sb.terminate()` in `finally:` blocks).
- Cost accounting: Modal CPU core-seconds ($0.00003942 / core-sec) and
  GiB-seconds ($0.00000667 / GiB-sec) tracked via CostTracker.
- Hard budget enforcement: max_budget_usd (default $2.00) raises BudgetExceededError
  if exceeded before or during any operation.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from src.schema.scene_graph import SceneGraph
from src.validation.gates import ValidationReport, ValidationSuite
from src.validation.ppo_runner import PPOResult, train_and_evaluate_ppo
from src.validation.repair_loop import CostTracker, RepairResult, run_self_healing_loop

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BudgetExceededError(RuntimeError):
    """Raised when compute or API costs exceed the specified spending cap."""
    pass


class ModalSandboxRunner:
    """Wraps validation, repair, and PPO execution with cloud/local support and budget capping."""

    def __init__(
        self,
        max_budget_usd: float = 2.00,
        cost_tracker: CostTracker | None = None,
        cpu_cores: int = 4,
        memory_gib: int = 8,
    ) -> None:
        self.max_budget_usd = float(max_budget_usd)
        self.cost_tracker = cost_tracker if cost_tracker is not None else CostTracker()
        self.cpu_cores = cpu_cores
        self.memory_gib = memory_gib

    def is_cloud_available(self) -> bool:
        """Return True if Modal authentication token is present in the environment."""
        return bool(os.environ.get("MODAL_TOKEN_ID"))

    def _check_budget(self) -> None:
        """Check if total spent exceeds max_budget_usd; raise BudgetExceededError if so."""
        if self.cost_tracker.total_cost_usd >= self.max_budget_usd:
            raise BudgetExceededError(
                f"Budget cap of ${self.max_budget_usd:.2f} exceeded! "
                f"Current cost: ${self.cost_tracker.total_cost_usd:.4f}"
            )

    def run_validation(
        self,
        suite_or_scene: ValidationSuite | SceneGraph,
        run_gate5: bool = False,
        **kwargs: Any,
    ) -> ValidationReport:
        """Run validation gates 1-4 (or 1-5 if run_gate5=True)."""
        self._check_budget()

        t0 = time.time()
        if self.is_cloud_available():
            report = self._run_validation_cloud(suite_or_scene, run_gate5=run_gate5, **kwargs)
        else:
            report = self._run_validation_local(suite_or_scene, run_gate5=run_gate5, **kwargs)

        elapsed = time.time() - t0
        self.cost_tracker.log_compute(elapsed * self.cpu_cores, elapsed * self.memory_gib)
        self._check_budget()
        return report

    def run_repair(
        self,
        scene: SceneGraph,
        max_tries: int = 4,
        run_gate5: bool = False,
        **kwargs: Any,
    ) -> RepairResult:
        """Run self-healing repair loop."""
        self._check_budget()

        t0 = time.time()
        if self.is_cloud_available():
            result = self._run_repair_cloud(scene, max_tries=max_tries, run_gate5=run_gate5, **kwargs)
        else:
            result = self._run_repair_local(scene, max_tries=max_tries, run_gate5=run_gate5, **kwargs)

        elapsed = time.time() - t0
        self.cost_tracker.log_compute(elapsed * self.cpu_cores, elapsed * self.memory_gib)

        if result.cost_tracker is not None:
            self.cost_tracker.total_input_tokens += result.cost_tracker.total_input_tokens
            self.cost_tracker.total_output_tokens += result.cost_tracker.total_output_tokens
            self.cost_tracker.call_log.extend(result.cost_tracker.call_log)

        self._check_budget()
        return result

    def run_ppo(
        self,
        scene: SceneGraph,
        total_timesteps: int = 8192,
        n_envs: int = 4,
        **kwargs: Any,
    ) -> PPOResult:
        """Run PPO training and evaluation."""
        self._check_budget()

        t0 = time.time()
        if self.is_cloud_available():
            result = self._run_ppo_cloud(scene, total_timesteps=total_timesteps, n_envs=n_envs, **kwargs)
        else:
            result = self._run_ppo_local(scene, total_timesteps=total_timesteps, n_envs=n_envs, **kwargs)

        elapsed = time.time() - t0
        self.cost_tracker.log_compute(elapsed * self.cpu_cores, elapsed * self.memory_gib)
        self._check_budget()
        return result

    # -----------------------------------------------------------------------
    # Local execution implementations
    # -----------------------------------------------------------------------

    def _run_validation_local(
        self,
        suite_or_scene: ValidationSuite | SceneGraph,
        run_gate5: bool = False,
        **kwargs: Any,
    ) -> ValidationReport:
        if isinstance(suite_or_scene, ValidationSuite):
            suite = suite_or_scene
        else:
            suite = ValidationSuite(suite_or_scene, **kwargs)
        return suite.run_all(run_gate5=run_gate5)

    def _run_repair_local(
        self,
        scene: SceneGraph,
        max_tries: int = 4,
        run_gate5: bool = False,
        **kwargs: Any,
    ) -> RepairResult:
        return run_self_healing_loop(
            scene=scene,
            max_tries=max_tries,
            run_gate5=run_gate5,
            **kwargs,
        )

    def _run_ppo_local(
        self,
        scene: SceneGraph,
        total_timesteps: int = 8192,
        n_envs: int = 4,
        **kwargs: Any,
    ) -> PPOResult:
        return train_and_evaluate_ppo(
            scene=scene,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            **kwargs,
        )

    # -----------------------------------------------------------------------
    # Modal Cloud execution implementations
    # -----------------------------------------------------------------------

    def _create_modal_sandbox(self) -> Any:
        import modal

        app = modal.App.lookup("worldforge-sandbox", create_if_missing=True)
        image = (
            modal.Image.debian_slim(python_version="3.12")
            .apt_install("libgl1-mesa-glx", "libosmesa6-dev", "libegl1-mesa-dev")
            .pip_install(
                "mujoco>=3.1.0",
                "gymnasium>=1.0.0",
                "pydantic>=2.0.0",
                "numpy>=1.26.0",
                "pyyaml>=6.0",
                "stable-baselines3>=2.5.0",
                "imageio[ffmpeg]>=2.36.0",
            )
            .add_local_dir(str(PROJECT_ROOT / "src"), remote_path="/workspace/src")
            .add_local_dir(str(PROJECT_ROOT / "configs"), remote_path="/workspace/configs")
        )
        return modal.Sandbox.create(
            image=image,
            cpu=self.cpu_cores,
            memory=self.memory_gib * 1024,
            app=app,
            workdir="/workspace",
            env={"MUJOCO_GL": "egl", "PYTHONPATH": "/workspace"},
        )

    def _run_validation_cloud(
        self,
        suite_or_scene: ValidationSuite | SceneGraph,
        run_gate5: bool = False,
        **kwargs: Any,
    ) -> ValidationReport:
        scene = suite_or_scene.scene if isinstance(suite_or_scene, ValidationSuite) else suite_or_scene
        sb = None
        try:
            sb = self._create_modal_sandbox()
            scene_json = json.dumps(scene.model_dump())
            cmd = (
                "import json; from src.schema.scene_graph import SceneGraph; "
                "from src.validation.gates import ValidationSuite; "
                f"sg = SceneGraph.model_validate(json.loads({scene_json!r})); "
                f"rep = ValidationSuite(sg).run_all(run_gate5={bool(run_gate5)!r}); "
                "print(json.dumps({'passed': rep.passed, 'failed_gate': rep.failed_gate, "
                "'gate_results': rep.gate_results, 'diagnostics': rep.diagnostics}))"
            )
            proc = sb.exec("python", "-c", cmd)
            proc.wait()
            out_lines = [line for line in proc.stdout.read().splitlines() if line.strip()]
            if proc.returncode == 0 and out_lines:
                payload = json.loads(out_lines[-1])
                return ValidationReport(
                    passed=bool(payload["passed"]),
                    failed_gate=payload["failed_gate"],
                    gate_results={int(k): v for k, v in payload["gate_results"].items()},
                    diagnostics=payload.get("diagnostics", ""),
                )
            raise RuntimeError(f"Modal validation process exited with code {proc.returncode}")
        except Exception as exc:
            logger.warning("Modal cloud execution encountered exception (%s); falling back to local.", exc)
            return self._run_validation_local(suite_or_scene, run_gate5=run_gate5, **kwargs)
        finally:
            if sb is not None:
                try:
                    sb.terminate()
                except Exception:
                    pass

    def _run_repair_cloud(
        self,
        scene: SceneGraph,
        max_tries: int = 4,
        run_gate5: bool = False,
        **kwargs: Any,
    ) -> RepairResult:
        # Custom callables (like reward_fn_override or on_repair) cannot be serialized over CLI JSON
        if kwargs.get("reward_fn_override") is not None or kwargs.get("on_repair") is not None:
            return self._run_repair_local(scene, max_tries=max_tries, run_gate5=run_gate5, **kwargs)

        sb = None
        try:
            sb = self._create_modal_sandbox()
            scene_json = json.dumps(scene.model_dump())
            cmd = (
                "import json; from src.schema.scene_graph import SceneGraph; "
                "from src.validation.repair_loop import run_self_healing_loop; "
                f"sg = SceneGraph.model_validate(json.loads({scene_json!r})); "
                f"res = run_self_healing_loop(sg, max_tries={int(max_tries)}, run_gate5={bool(run_gate5)!r}); "
                "print(json.dumps({'success': res.success, 'attempts': res.attempts, "
                "'repairs_applied': res.repairs_applied, 'repair_diff': res.repair_diff, "
                "'passed': res.final_report.passed, 'failed_gate': res.final_report.failed_gate, "
                "'gate_results': res.final_report.gate_results, 'diagnostics': res.final_report.diagnostics}))"
            )
            proc = sb.exec("python", "-c", cmd)
            proc.wait()
            out_lines = [line for line in proc.stdout.read().splitlines() if line.strip()]
            if proc.returncode == 0 and out_lines:
                payload = json.loads(out_lines[-1])
                rep = ValidationReport(
                    passed=bool(payload["passed"]),
                    failed_gate=payload["failed_gate"],
                    gate_results={int(k): v for k, v in payload["gate_results"].items()},
                    diagnostics=payload.get("diagnostics", ""),
                )
                return RepairResult(
                    final_report=rep,
                    attempts=int(payload["attempts"]),
                    repairs_applied=list(payload["repairs_applied"]),
                    success=bool(payload["success"]),
                    repair_diff=str(payload.get("repair_diff", "")),
                )
            raise RuntimeError(f"Modal repair process exited with code {proc.returncode}")
        except Exception as exc:
            logger.warning("Modal cloud execution encountered exception (%s); falling back to local.", exc)
            return self._run_repair_local(scene, max_tries=max_tries, run_gate5=run_gate5, **kwargs)
        finally:
            if sb is not None:
                try:
                    sb.terminate()
                except Exception:
                    pass

    def _run_ppo_cloud(
        self,
        scene: SceneGraph,
        total_timesteps: int = 8192,
        n_envs: int = 4,
        **kwargs: Any,
    ) -> PPOResult:
        sb = None
        try:
            sb = self._create_modal_sandbox()
            scene_json = json.dumps(scene.model_dump())
            cmd = (
                "import json; from src.schema.scene_graph import SceneGraph; "
                "from src.validation.ppo_runner import train_and_evaluate_ppo; "
                f"sg = SceneGraph.model_validate(json.loads({scene_json!r})); "
                f"res = train_and_evaluate_ppo(sg, total_timesteps={int(total_timesteps)}, n_envs={int(n_envs)}); "
                "print(json.dumps({'eval_mean_return': res.eval_mean_return, "
                "'eval_std_return': res.eval_std_return, 'learning_curve': res.learning_curve}))"
            )
            proc = sb.exec("python", "-c", cmd)
            proc.wait()
            out_lines = [line for line in proc.stdout.read().splitlines() if line.strip()]
            if proc.returncode == 0 and out_lines:
                payload = json.loads(out_lines[-1])
                return PPOResult(
                    eval_mean_return=float(payload["eval_mean_return"]),
                    eval_std_return=float(payload["eval_std_return"]),
                    learning_curve=list(payload["learning_curve"]),
                )
            raise RuntimeError(f"Modal PPO process exited with code {proc.returncode}")
        except Exception as exc:
            logger.warning("Modal cloud execution encountered exception (%s); falling back to local.", exc)
            return self._run_ppo_local(scene, total_timesteps=total_timesteps, n_envs=n_envs, **kwargs)
        finally:
            if sb is not None:
                try:
                    sb.terminate()
                except Exception:
                    pass
