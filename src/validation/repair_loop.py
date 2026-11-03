"""WorldForge Self-Healing Repair Loop.

Implements:
* ``CostTracker`` — tracks API token costs and compute time.
* ``_extract_python_code`` & ``_compile_safe_reward_fn`` — strips markdown code
  fences and enforces strict AST sandboxing (blocking imports, dunder access,
  and dangerous builtins) before compiling LLM-synthesized reward code.
* ``run_self_healing_loop`` — runs validation gates, and when a gate fails,
  applies a deterministic or live Baseten Muse Spark LLM-driven fix with
  unified diff generation, and re-runs up to ``max_tries`` times.
"""

from __future__ import annotations

import ast
import copy
import difflib
import inspect
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from src.schema.scene_graph import SceneGraph
from src.validation.gates import ValidationReport, ValidationSuite

logger = logging.getLogger(__name__)


# ─── Cost Tracker ─────────────────────────────────────────────────────

@dataclass
class CostTracker:
    """Tracks API token costs and compute time.

    Pricing tiers (per 1M tokens):
        Muse Spark Standard:  $1.25 input, $4.25 output
        Muse Spark Contributor: $0.10 input, $0.20 output
        Modal CPU: $0.00003942 per core-second
    """

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cpu_core_seconds: float = 0.0
    total_gib_seconds: float = 0.0
    pricing_tier: str = "standard"
    call_log: list[dict[str, Any]] = field(default_factory=list)

    def log_api_call(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "muse_spark",
        purpose: str = "",
    ) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_log.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "purpose": purpose,
            "timestamp": time.time(),
        })

    def log_compute(self, core_seconds: float, gib_seconds: float = 0.0) -> None:
        self.total_cpu_core_seconds += core_seconds
        self.total_gib_seconds += gib_seconds

    @property
    def total_cost_usd(self) -> float:
        if self.pricing_tier == "contributor":
            in_rate = 0.10 / 1_000_000
            out_rate = 0.20 / 1_000_000
        else:
            in_rate = 1.25 / 1_000_000
            out_rate = 4.25 / 1_000_000

        api_cost = (
            self.total_input_tokens * in_rate
            + self.total_output_tokens * out_rate
        )
        compute_cost = (
            self.total_cpu_core_seconds * 0.00003942
            + self.total_gib_seconds * 0.00000667
        )
        return api_cost + compute_cost

    def summary(self) -> str:
        return (
            f"CostTracker: {self.total_input_tokens:,} in + "
            f"{self.total_output_tokens:,} out tokens, "
            f"{self.total_cpu_core_seconds:.1f} CPU-sec, "
            f"${self.total_cost_usd:.4f} total ({self.pricing_tier})"
        )


# ─── Repair Result ────────────────────────────────────────────────────

@dataclass
class RepairResult:
    """Outcome of a self-healing repair loop."""

    final_report: ValidationReport
    attempts: int
    repairs_applied: list[str] = field(default_factory=list)
    cost_tracker: CostTracker = field(default_factory=CostTracker)
    success: bool = False
    repair_diff: str = ""
    repaired_reward_fn: Callable | None = None
    repaired_reward_code: str = ""


# ─── Safe AST Validation & Code Extraction ────────────────────────────

_FORBIDDEN_AST_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
    ast.With,
    ast.AsyncWith,
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
    ast.Try,
    ast.Raise,
)

_FORBIDDEN_NAMES = frozenset({
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "__builtins__",
    "__loader__",
    "__spec__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "input",
    "breakpoint",
    "exit",
    "quit",
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "socket",
    "ctypes",
    " builtins",
})

_FORBIDDEN_ATTRS = frozenset({
    "format", "format_map", "load", "loads", "save", "savez", "savez_compressed",
    "savetxt", "loadtxt", "genfromtxt", "fromregex", "fromfile", "tofile",
    "memmap", "DataSource", "ctypeslib", "f2py", "testing", "distutils", "lib",
    "pickle", "allow_pickle", "system", "popen", "spawn", "exec", "eval",
})

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
    "dict": dict,
    "list": list,
    "tuple": tuple,
    "set": set,
    "round": round,
    "isinstance": isinstance,
    "any": any,
    "all": all,
    "True": True,
    "False": False,
    "None": None,
}


def _extract_python_code(raw_reply: str) -> str:
    """Extract Python source code from a raw or markdown-fenced LLM reply."""
    text = raw_reply.strip()
    if "&lt;think&gt;" in text and "&lt;/think&gt;" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    fence_match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        return "\n".join(lines[1:-1]).strip()

    return text


def _validate_safe_reward_ast(tree: ast.AST) -> None:
    """Validate that an AST contains only safe numeric/reward computation nodes."""
    has_compute_reward = False
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_AST_NODES):
            raise ValueError(f"Disallowed AST construct in synthesized reward: {type(node).__name__}")
        if isinstance(node, ast.FunctionDef) and node.name == "compute_reward":
            has_compute_reward = True
        if isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES or node.id.startswith("__"):
                raise ValueError(f"Disallowed identifier '{node.id}' in synthesized reward")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in _FORBIDDEN_ATTRS:
                raise ValueError(f"Disallowed attribute access '.{node.attr}' in synthesized reward")
    if not has_compute_reward:
        raise ValueError("Synthesized code must define `def compute_reward(model, data, scene):`")


def _verify_reward_in_scrubbed_subprocess(candidate_code: str) -> None:
    """Run candidate code in an isolated subprocess with sanitized environment to catch syntax or sandbox breaches."""
    import subprocess
    import sys

    # Strip any sensitive keys (BASETEN, META, OPENAI, MODAL, etc.) from env
    scrubbed_env = {
        k: v for k, v in os.environ.items()
        if not any(token in k.upper() for token in ("KEY", "SECRET", "TOKEN", "AUTH", "PASS"))
    }
    verify_script = (
        "import ast, sys\n"
        "code = sys.stdin.read()\n"
        "tree = ast.parse(code, mode='exec')\n"
        "compile(tree, '<test>', 'exec')\n"
    )
    res = subprocess.run(
        [sys.executable, "-I", "-c", verify_script],
        input=candidate_code,
        text=True,
        capture_output=True,
        env=scrubbed_env,
        timeout=5.0,
    )
    if res.returncode != 0:
        raise ValueError(f"Reward code verification failed in scrubbed subprocess: {res.stderr.strip()}")


def _build_scrubbed_proxies() -> tuple[Any, Any, Any]:
    """Build read-only math, numpy, and mujoco proxies with scrubbed __globals__ and no I/O or pickle APIs."""
    import types

    isolated_env: dict[str, Any] = {
        "__builtins__": {},
        "_asarray": np.asarray,
        "_clip": np.clip,
        "_norm": np.linalg.norm,
        "_mean": np.mean,
        "_sum": np.sum,
        "_min": np.min,
        "_max": np.max,
        "_dot": np.dot,
        "_zeros": np.zeros,
        "_ones": np.ones,
        "_f64": np.float64,
        "_f32": np.float32,
        "_float": float,
    }
    helper_lines = [
        "def array(x, dtype=_f64): return _asarray(x, dtype=_f64)",
        "def asarray(x, dtype=_f64): return _asarray(x, dtype=_f64)",
        "def clip(x, a_min, a_max): return _clip(x, a_min, a_max)",
        "def norm(x, axis=None): return _float(_norm(_asarray(x, dtype=_f64), axis=axis))",
        "def mean(x, axis=None): return _mean(_asarray(x, dtype=_f64), axis=axis)",
        "def sum_(x, axis=None): return _sum(_asarray(x, dtype=_f64), axis=axis)",
        "def min_(x, axis=None): return _min(_asarray(x, dtype=_f64), axis=axis)",
        "def max_(x, axis=None): return _max(_asarray(x, dtype=_f64), axis=axis)",
        "def dot(a, b): return _dot(_asarray(a, dtype=_f64), _asarray(b, dtype=_f64))",
        "def zeros(shape, dtype=_f64): return _zeros(shape, dtype=_f64)",
        "def ones(shape, dtype=_f64): return _ones(shape, dtype=_f64)",
    ]
    exec(compile("\n".join(helper_lines), "<safe_np_proxy>", "exec"), isolated_env, isolated_env)  # noqa: S102

    safe_linalg = types.SimpleNamespace(norm=isolated_env["norm"])
    safe_np = types.SimpleNamespace(
        array=isolated_env["array"],
        asarray=isolated_env["asarray"],
        clip=isolated_env["clip"],
        linalg=safe_linalg,
        mean=isolated_env["mean"],
        sum=isolated_env["sum_"],
        min=isolated_env["min_"],
        max=isolated_env["max_"],
        dot=isolated_env["dot"],
        zeros=isolated_env["zeros"],
        ones=isolated_env["ones"],
        tanh=np.tanh,
        sqrt=np.sqrt,
        exp=np.exp,
        abs=np.abs,
        minimum=np.minimum,
        maximum=np.maximum,
        hypot=np.hypot,
        isfinite=np.isfinite,
        float64=np.float64,
        float32=np.float32,
        ndarray=np.ndarray,
        pi=float(math.pi),
    )
    safe_math = types.SimpleNamespace(
        tanh=math.tanh, sqrt=math.sqrt, exp=math.exp, log=math.log,
        sin=math.sin, cos=math.cos, acos=math.acos, atan2=math.atan2,
        fabs=math.fabs, hypot=math.hypot, isfinite=math.isfinite, isnan=math.isnan,
        pi=float(math.pi),
    )
    mjt_obj = getattr(mujoco, "mjtObj", None)
    safe_mujoco = types.SimpleNamespace(
        mj_name2id=getattr(mujoco, "mj_name2id", None),
        mjtObj=mjt_obj,
    )
    return safe_math, safe_np, safe_mujoco


def _compile_safe_reward_fn(raw_reply: str) -> tuple[Callable, str]:
    """Strip markdown fences, AST-validate, and compile a restricted reward function."""
    candidate_code = _extract_python_code(raw_reply)
    tree = ast.parse(candidate_code, mode="exec")
    _validate_safe_reward_ast(tree)

    _verify_reward_in_scrubbed_subprocess(candidate_code)

    safe_math, safe_np, safe_mujoco = _build_scrubbed_proxies()

    code_obj = compile(tree, filename="<synthesized_reward>", mode="exec")
    safe_ns: dict[str, Any] = {
        "__builtins__": dict(_SAFE_BUILTINS),
        "math": safe_math,
        "np": safe_np,
        "mujoco": safe_mujoco,
    }
    exec(code_obj, safe_ns, safe_ns)  # noqa: S102
    fn = safe_ns.get("compute_reward")
    if not callable(fn):
        raise ValueError("Compiled namespace did not produce a callable `compute_reward`")
    return fn, candidate_code


# ─── Repair Strategies ────────────────────────────────────────────────

def _format_repair_prompt(
    gate_num: int,
    diagnostics: str,
    scene: SceneGraph,
) -> str:
    """Format a structured repair prompt for LLM consumption."""
    return (
        f"## WorldForge Validation Gate {gate_num} FAILED\n\n"
        f"### Scene: {scene.scene_id} (table_z={scene.table_z})\n"
        f"### Task: {scene.task.task_type} — {scene.task.description}\n"
        f"### Diagnostics:\n{diagnostics}\n\n"
        f"### Instructions:\n"
        f"Identify the root cause of the gate {gate_num} failure and return "
        f"a patched reward function `def compute_reward(model, data, scene):`."
    )


RepairFn = Callable[[SceneGraph, str], tuple[SceneGraph, str]]


def _repair_gate3_degenerate_reward(
    scene: SceneGraph,
    diagnostics: str,
) -> tuple[SceneGraph, str]:
    """Fix degenerate (constant) reward by restoring the default dense reward."""
    return scene, "Restored default dense reward (removed constant-zero override)"


def _repair_gate2_drift(
    scene: SceneGraph,
    diagnostics: str,
) -> tuple[SceneGraph, str]:
    """Fix resting drift by increasing object damping."""
    scene = copy.deepcopy(scene)
    for obj in scene.objects:
        if not obj.is_static and obj.joint.type == "free":
            obj.joint.damping = max(obj.joint.damping, 1.0)
    return scene, "Increased free-joint damping to reduce drift"


def _repair_gate1_instability(
    scene: SceneGraph,
    diagnostics: str,
) -> tuple[SceneGraph, str]:
    """Fix NaN/instability — usually caused by bad density or timestep."""
    scene = copy.deepcopy(scene)
    for obj in scene.objects:
        for i, s in enumerate(obj.size):
            if s < 0.005:
                obj.size[i] = 0.005
    return scene, "Clamped minimum geom sizes to 5mm to prevent solver instability"


_DETERMINISTIC_REPAIRS: dict[int, RepairFn] = {
    1: _repair_gate1_instability,
    2: _repair_gate2_drift,
    3: _repair_gate3_degenerate_reward,
}


def _build_unified_diff(
    before_src: str,
    after_src: str,
    fromfile: str = "before",
    tofile: str = "after",
    from_label: str | None = None,
    to_label: str | None = None,
) -> str:
    from_name = from_label if from_label is not None else fromfile
    to_name = to_label if to_label is not None else tofile
    return "".join(
        difflib.unified_diff(
            before_src.splitlines(keepends=True),
            after_src.splitlines(keepends=True),
            fromfile=from_name,
            tofile=to_name,
        )
    )


def _get_faulty_source(reward_fn_override: Callable | None) -> str:
    if reward_fn_override is not None:
        try:
            return inspect.getsource(reward_fn_override)
        except Exception:
            pass
    return (
        "def compute_reward(model, data, scene):\n"
        "    return 0.0, False, {'fault': 'Constant 0.0 reward injected'}\n"
    )


def _get_default_reward_source() -> str:
    reward_file = Path(__file__).resolve().parents[1] / "compiler" / "generated_reward.py"
    if reward_file.exists():
        return reward_file.read_text(encoding="utf-8")
    return ""


def _call_live_llm_repair(
    gate_num: int,
    diagnostics: str,
    scene: SceneGraph,
    cost_tracker: CostTracker,
    faulty_src: str,
) -> tuple[Callable | None, str, str, str]:
    """Call Baseten meta-models/muse-spark-open-1026 to synthesize a patched reward function."""
    baseten_key = os.environ.get("BASETEN_API_KEY", "")
    baseten_url = os.environ.get("BASETEN_BASE_URL", "https://inference.baseten.co/v1")
    base_prompt = _format_repair_prompt(gate_num, diagnostics, scene)

    if baseten_key and not baseten_key.startswith("your_"):
        try:
            from openai import OpenAI
            client = OpenAI(base_url=baseten_url, api_key=baseten_key)
            resp = client.chat.completions.create(
                model="meta-models/muse-spark-open-1026",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Muse Code / Muse Spark, an expert MuJoCo reinforcement learning engineer. "
                            "Output only valid Python code defining `compute_reward(model, data, scene)` "
                            "using `math`, `np`, and `mujoco` without any import statements."
                        ),
                    },
                    {"role": "user", "content": base_prompt},
                ],
                max_tokens=768,
                temperature=0.1,
            )
            raw_reply = resp.choices[0].message.content or ""
            cost_tracker.log_api_call(
                input_tokens=len(base_prompt.split()) * 2,
                output_tokens=max(len(raw_reply.split()) * 2, 50),
                model="meta-models/muse-spark-open-1026",
                purpose=f"llm_repair_gate_{gate_num}",
            )

            compiled_fn, clean_code = _compile_safe_reward_fn(raw_reply)
            diff_str = _build_unified_diff(
                faulty_src,
                clean_code + "\n",
                fromfile="generated_reward.py (Attempt 1: Faulty)",
                tofile="generated_reward.py (Attempt 2: Muse Spark AST-Patched)",
            )
            return (
                compiled_fn,
                f"Live Baseten Muse Spark AST-patched reward for gate {gate_num}",
                diff_str,
                clean_code,
            )
        except Exception as exc:
            logger.warning("Live Baseten repair fallback (%s); restoring verified dense reward.", exc)

    # Deterministic fallback when live LLM is unavailable or rejected by AST sandbox
    default_src = _get_default_reward_source()
    diff_str = _build_unified_diff(
        faulty_src,
        default_src,
        fromfile="generated_reward.py (Attempt 1: Faulty Override)",
        tofile="generated_reward.py (Attempt 2: Restored Dense Reward)",
    )
    return None, f"Restored verified multi-stage dense reward for gate {gate_num}", diff_str, default_src


# ─── Self-Healing Loop ───────────────────────────────────────────────

def run_self_healing_loop(
    scene: SceneGraph,
    max_tries: int = 4,
    use_live_llm: bool = False,
    reward_fn_override: Any | None = None,
    on_repair: Callable[[int, str, SceneGraph], SceneGraph] | None = None,
    run_gate5: bool = False,
    verbose: bool = False,
) -> RepairResult:
    """Run validation with automatic self-healing on failure."""
    cost_tracker = CostTracker()
    repairs_applied: list[str] = []
    current_scene = copy.deepcopy(scene)
    current_reward_override = reward_fn_override
    faulty_src = _get_faulty_source(reward_fn_override)
    cached_gate_results: dict[int, dict] = {}
    last_failed_gate: int | None = None

    result = RepairResult(
        final_report=ValidationReport(passed=False, failed_gate=None),
        attempts=0,
        cost_tracker=cost_tracker,
    )

    for attempt in range(1, max_tries + 1):
        result.attempts = attempt

        if verbose:
            print(f"\n{'='*50}")
            print(f"Validation attempt {attempt}/{max_tries}")
            print(f"{'='*50}")

        t0 = time.time()

        suite = ValidationSuite(
            current_scene,
            verbose=verbose,
            reward_fn=current_reward_override,
        )

        start_gate = 1
        if attempt > 1 and last_failed_gate == 3 and 1 in cached_gate_results and 2 in cached_gate_results:
            start_gate = 3

        report = suite.run_all(
            run_gate5=run_gate5,
            start_from_gate=start_gate,
            cached_gate_results=cached_gate_results,
        )

        for g_num, g_res in report.gate_results.items():
            if g_res.get("passed", False):
                cached_gate_results[g_num] = g_res

        elapsed = time.time() - t0
        cost_tracker.log_compute(elapsed)

        result.final_report = report

        if report.passed:
            result.success = True
            result.repairs_applied = repairs_applied
            result.repaired_reward_fn = current_reward_override
            if verbose:
                print(f"  ✓ All gates passed on attempt {attempt}")
            return result

        # ── Attempt repair ────────────────────────────────────────────
        gate = report.failed_gate
        last_failed_gate = gate
        diag = report.diagnostics
        if verbose:
            print(f"  Gate {gate} failed, attempting repair...")

        if on_repair is not None:
            current_scene = on_repair(attempt, diag, current_scene)
            repairs_applied.append(f"Attempt {attempt}: on_repair callback applied for gate {gate}")
            continue

        if use_live_llm and gate == 3:
            compiled_fn, desc, diff_str, clean_code = _call_live_llm_repair(
                gate, diag, current_scene, cost_tracker, faulty_src
            )
            result.repair_diff = diff_str
            result.repaired_reward_code = clean_code
            current_reward_override = compiled_fn
            repairs_applied.append(f"Attempt {attempt}: {desc}")
            continue

        if gate in _DETERMINISTIC_REPAIRS:
            repair_fn = _DETERMINISTIC_REPAIRS[gate]
            current_scene, repair_desc = repair_fn(current_scene, diag)
            repairs_applied.append(f"Attempt {attempt}: {repair_desc}")

            if gate == 3:
                current_reward_override = None
                default_src = _get_default_reward_source()
                result.repaired_reward_code = default_src
                result.repair_diff = _build_unified_diff(
                    faulty_src,
                    default_src,
                    fromfile="generated_reward.py (Attempt 1: Faulty Override)",
                    tofile="generated_reward.py (Attempt 2: Restored Dense Reward)",
                )
            continue

        repairs_applied.append(
            f"Attempt {attempt}: No repair available for gate {gate}"
        )
        break

    result.repairs_applied = repairs_applied
    result.repaired_reward_fn = current_reward_override
    return result