# WorldForge-RL

> **Back-to-Front RL Environment Synthesis for Tabletop Manipulation**
>
> Converts video specifications into validated, trainable MuJoCo 3.x Gymnasium environments.

[![CI Tests](https://img.shields.io/badge/tests-52%20passed-brightgreen.svg)]()
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)]()
[![MuJoCo 3.x](https://img.shields.io/badge/mujoco-3.17.0-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture Overview

```
SceneGraph (Pydantic V2)
        │
        ▼ (Deterministic)
MJCF Builder (MuJoCo 3.x XML)
        │
        ▼
WorldForgeEnv (Gymnasium Env, 4-DoF Cartesian Mocap Weld)
        │
        ├── Multi-Stage Shaped Dense Reward
        │     • pick_and_place (reach → grasp → lift → place → success)
        │     • push_to_target (reach → cradle push → clearance → success)
        │     • articulate_joint (reach → displacement → success)
        │
        ├── 5-Gate Validation Pipeline
        │     • Gate 1: Numerical Stability (1,000 steps, zero NaNs, |qacc| < 1e6)
        │     • Gate 2: Static Equilibrium (250 steps, resting drift < 2mm h, < 1mm v)
        │     • Gate 3: Non-Degenerate Reward Variance (σ > 1e-4)
        │     • Gate 4: Closed-Loop Waypoint Oracle Solvability
        │     • Gate 5: Fast PPO Convergence (beats random baseline)
        │
        └── Self-Healing Repair Loop + CostTracker
```

---

## Directory Structure

```text
WorldForge-RL/
├── configs/
│   ├── materials.yaml                # Density, friction & RGBA lookup tables
│   └── scenes/
│       ├── scene1_pick_place.json    # Golden Scene 1: Red cube pick & place
│       ├── scene2_push_obstacle.json  # Golden Scene 2: Cylinder push around obstacle
│       └── scene3_articulated_drawer.json # Golden Scene 3: Articulated slide drawer
├── src/
│   ├── schema/
│   │   └── scene_graph.py            # Pydantic V2 models with table auto-snap
│   ├── compiler/
│   │   ├── mjcf_builder.py           # Deterministic SceneGraph -> MJCF XML compiler
│   │   ├── base_env.py               # WorldForgeEnv (Gymnasium 4-DoF Mocap Env)
│   │   └── generated_reward.py       # Modular multi-stage dense reward functions
│   └── validation/
│       ├── gates.py                  # Validation Gates 1–5
│       ├── ppo_runner.py             # DummyVecEnv PPO trainer & MP4 rollout recorder
│       └── repair_loop.py            # CostTracker + automated self-healing loop
└── tests/
    ├── test_phase1.py                # 25 Phase 1 tests (Schema, MJCF, Env)
    └── test_phase2.py                # 27 Phase 2 tests (Rewards, Gates 1–5, PPO, Repair)
```

---

## Quickstart

### Prerequisites
- Python 3.12 (CPython)
- `uv` or `pip`

### Setup

```bash
# Clone the repository
git clone https://github.com/syed-mohsin-s/WorldForge.git
cd WorldForge

# Create virtual environment and install dependencies
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe mujoco gymnasium pydantic pyyaml numpy scipy pytest stable-baselines3 "imageio[ffmpeg]"
```

### Running Tests

```bash
# Run full test suite (52 tests)
.venv/Scripts/python.exe -m pytest tests/test_phase1.py tests/test_phase2.py -v
```

---

## Features

- **MuJoCo 3.x Physics Integrity**: Implicit-fast integrator (`timestep="0.002"`), elliptic contact friction cones, `condim="4"` contact solvers, and stiff mocap weld equality constraints (`solref="0.01 1"`).
- **Auto-Snap Height Grounding**: Eliminates Frame-0 penetration explosions by grounding tabletop bodies at `table_z + half_height + 0.001m`.
- **Cross-Platform Offscreen Rendering**: Automatic WGL (Windows) and EGL/OSMesa (Linux) headless camera support.
- **Cost Accounting**: Built-in `CostTracker` instrumenting LLM tokens and cloud compute core-seconds.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
