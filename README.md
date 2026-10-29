# WorldForge-RL

> **Back-to-Front RL Environment Synthesis for Tabletop Manipulation**
>
> Converts video specifications into validated, trainable MuJoCo 3.x Gymnasium environments.

[![CI Tests](https://img.shields.io/badge/tests-83%20passed-brightgreen.svg)]()
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)]()
[![MuJoCo 3.x](https://img.shields.io/badge/mujoco-3.17.0-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture Overview

```
Monocular Video / RGB-D
        │
        ▼ (Stage 1: SAM 3.1)
SAM 3.1 Tracker (Mask Segmentation, 4px Boundary Erosion, 2D Trajectories)
        │
        ▼ (Stage 2: 3D Lift)
3D Lifter (Pinhole Back-Projection, RANSAC Plane Fit, SVD Trajectory Fit)
        │
        ▼ (Stage 3: Muse Spark - Baseten / Meta)
Muse Spark Reasoner (Baseten meta-models/muse-spark-open-1026 + Meta API)
        │
        ▼ (Stage 4: Scene Graph)
SceneGraph (Pydantic V2 Strict Models + Auto-Snap Height Grounding)
        │
        ▼ (Stage 5: MJCF Compiler)
MJCF Builder (MuJoCo 3.x XML, Mocap Weld Equality, condim="4")
        │
        ▼ (Stage 6: Multi-Stage Shaped Dense Rewards)
WorldForgeEnv (Gymnasium 4-DoF Mocap Cartesian Env, Cross-Platform WGL/EGL)
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
        ├── Stage 7: Self-Healing Repair Loop + CostTracker
        │
        └── Stage 8: Render + Diff Visual Verification & Modal Sandbox Cloud Runner
              • Fast MuJoCo geom ID & SAM 3.1 segmentation extraction
              • Mask IoU evaluation against real video Frame 0
              • Projection-guided coordinate refinement loop (restoring IoU > 0.80)
              • Modal Sandbox container runner (4 CPUs, 8 GiB RAM, EGL)
              • Strict cost accounting & hard spending cap enforcement ($2.00)
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
│   ├── api/
│   │   └── server.py                 # FastAPI REST & SSE backend (CORS, /outputs)
│   ├── pipeline.py                   # 8-Stage end-to-end autonomous orchestrator
│   ├── schema/
│   │   └── scene_graph.py            # Pydantic V2 models with table auto-snap
│   ├── perception/
│   │   ├── sam_tracker.py            # SAM 3.1 tracking, 4px erosion & sim ground-truth
│   │   ├── lift_3d.py                # Pinhole ray-plane math, RANSAC table & SVD trajectory
│   │   └── spark_reasoner.py         # Baseten meta-models/muse-spark-open-1026 reasoner
│   ├── compiler/
│   │   ├── mjcf_builder.py           # Deterministic SceneGraph -> MJCF XML compiler
│   │   ├── base_env.py               # WorldForgeEnv (Gymnasium 4-DoF Mocap Env)
│   │   └── generated_reward.py       # Modular multi-stage dense reward functions
│   ├── validation/
│   │   ├── gates.py                  # Validation Gates 1–5
│   │   ├── ppo_runner.py             # DummyVecEnv PPO trainer & MP4 rollout recorder
│   │   └── repair_loop.py            # CostTracker + automated self-healing loop
│   ├── render_diff/
│   │   └── mask_iou.py               # Mask IoU, coordinate refinement & diff overlays
│   └── sandbox/
│       └── modal_executor.py         # Modal Sandbox cloud runner & budget capping
├── frontend/                         # Studio-Grade React 18 + Vite + Tailwind CSS + Recharts
├── run_demo.py                       # Single-command launcher for API + Frontend
├── start_studio.ps1                  # PowerShell launcher for Windows
└── tests/
    ├── test_phase1.py                # 25 Phase 1 tests (Schema, MJCF, Env)
    ├── test_phase2.py                # 27 Phase 2 tests (Rewards, Gates 1–5, PPO, Repair)
    ├── test_phase3.py                # 9 Phase 3 tests (SAM 3.1, 3D lift, Muse Spark, End-to-End)
    ├── test_phase4.py                # 14 Phase 4 tests (IoU, coordinate refinement, Modal sandbox)
    └── test_phase5.py                # 8 Phase 5 tests (Baseten, Pipeline, FastAPI endpoints)
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
uv pip install --python .venv/Scripts/python.exe mujoco gymnasium pydantic pyyaml numpy scipy pytest stable-baselines3 "imageio[ffmpeg]" opencv-python-headless httpx
```

### Running Tests

```bash
# Run full test suite across all 5 phases (83 tests)
.venv/Scripts/python.exe -m pytest tests/ -v
```

### Launching WorldForge Studio 2.0 (FastAPI + React UI)

```bash
# Option 1: Python Launcher (Cross-platform)
python run_demo.py

# Option 2: PowerShell (Windows)
.\start_studio.ps1
```
- **Studio UI**: `http://localhost:5173`
- **FastAPI Docs**: `http://localhost:8000/docs`

---

## Features

- **MuJoCo 3.x Physics Integrity**: Implicit-fast integrator (`timestep="0.002"`), elliptic contact friction cones, `condim="4"` contact solvers, and stiff mocap weld equality constraints (`solref="0.01 1"`).
- **Auto-Snap Height Grounding**: Eliminates Frame-0 penetration explosions by grounding tabletop bodies at `table_z + half_height + 0.001m`.
- **Cross-Platform Offscreen Rendering**: Automatic WGL (Windows) and EGL/OSMesa (Linux) headless camera support.
- **Cost Accounting**: Built-in `CostTracker` instrumenting LLM tokens and cloud compute core-seconds.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

