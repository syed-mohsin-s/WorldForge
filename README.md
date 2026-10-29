# WorldForge-RL: Autonomous Video-to-Environment Synthesis

> **Autonomous Synthesis of Validated, Trainable MuJoCo 3.x Gymnasium Environments from Monocular Video & 3D Scene Graphs**

[![CI Tests](https://img.shields.io/badge/tests-86%20passed%20(100%25)-brightgreen.svg)]()
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)]()
[![MuJoCo 3.x](https://img.shields.io/badge/mujoco-3.17.0-orange.svg)]()
[![Baseten](https://img.shields.io/badge/Baseten-meta--models%2Fmuse--spark--open--1026-violet.svg)]()
[![React 18](https://img.shields.io/badge/frontend-React%2018%20%2B%20Three.js-cyan.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is WorldForge?

**WorldForge** transforms uncalibrated monocular tabletop videos or task prompts into fully validated, physics-accurate, and trainable **MuJoCo 3.17 Gymnasium RL environments** in under 30 seconds—reducing manual robot simulation setup time from **~6.0 hours to under 30 seconds** (a **>700× speedup**).

### Core Highlights
- **100% Python MuJoCo 3.17 Single Source of Truth**: All physics, contact dynamics, Cartesian mocap kinematics, dense rewards, and PPO training execute in real Python MuJoCo 3.17. The frontend contains **zero** client-side mock physics or synthetic telemetry.
- **Baseten Hackathon Integration**: Native live streaming reasoning powered by `meta-models/muse-spark-open-1026` via OpenAI-compatible endpoints at `https://inference.baseten.co/v1`.
- **Automated Self-Healing Loop**: If physics gates fail (e.g. zero-reward variance, static drift, or pen-explosion), the Muse Spark reasoner diagnoses the runtime failure, generates an automated AST patch diff, hot-reloads the reward module, and re-validates.
- **5-Gate Quality Pipeline**: Zero-flakiness automated validation ensuring numerical stability, static equilibrium, non-degenerate dense rewards, closed-loop waypoint oracle solvability, and residual PPO policy learning.
- **Render + Diff Visual Alignment**: Validates MuJoCo Frame 0 against the source video via SAM 3.1 mask IoU with automated coordinate refinement.
- **Dual-Cockpit Studio UI**: A professional React 18 + Vite dashboard combining an **8-Stage Autonomous Pipeline Studio** with an interactive **Full-Fidelity 3D Three.js PBR Workbench**.

---

## 8-Stage Autonomous Pipeline

```
Uncalibrated Monocular Video / RGB-D
        │
        ▼ (Stage 1: SAM 3.1 Perception)
SAM 3.1 Multiplexed Tracker ── 4px Boundary Erosion, Mask Extraction, Centroid Trajectories
        │
        ▼ (Stage 2: 3D Metric Lifting)
3D Metric Lifter ── Pinhole Ray-Plane Back-Projection, RANSAC Table Fit, SVD Joint Axis Fitting
        │
        ▼ (Stage 3: Muse Spark Reasoning)
Baseten meta-models/muse-spark-open-1026 ── Physical reasoning trace & domain parameter synthesis
        │
        ▼ (Stage 4: Scene Graph Synthesis)
Pydantic V2 SceneGraph ── Auto-Snap Height Grounding (z = table_z + h/2 + 0.001m)
        │
        ▼ (Stage 5: MJCF XML Compilation)
MJCF Builder ── MuJoCo 3.17 XML with Mocap Welds, Elliptic Friction Cones, condim="4"
        │
        ▼ (Stage 6: Multi-Stage Dense Reward Synthesis)
Modular Reward Engine ── Reach → Grasp/Push → Lift/Progress → Place/Displace → Terminal Success
        │
        ▼ (Stage 7: 5-Gate Validation & Residual PPO Training)
Validation Suite ── Gate 1: Stability (1,000 steps, zero NaNs, |qacc| < 1e6)
                 ── Gate 2: Equilibrium (250 steps, drift < 2mm h, < 1mm v)
                 ── Gate 3: Non-Degenerate Reward Variance (σ > 1e-4)
                 ── Gate 4: Closed-Loop Waypoint Oracle (120 frames @ 30 FPS)
                 ── Gate 5: Residual PPO Learning (beats random baseline)
                 ── Self-Healing Loop: Auto-patches broken reward functions
        │
        ▼ (Stage 8: Render + Diff Visual Verification)
Render + Diff Alignment ── Mask IoU evaluation against video Frame 0 + Coordinate Refinement
        │
        ▼
4.0-Second 120-Frame Rollout MP4 & Deployable MuJoCo Gymnasium Environment
```

---

## Golden Verification Scenes

WorldForge ships with 3 pre-configured golden validation benchmarks covering fundamental manipulation primitives:

| Scene Preset | Task Type | Manipulated Object | Distractors / Articulation | Waypoints | Target Goal |
|---|---|---|---|---|---|
| **Scene 1: Red Cube Pick & Place** | `pick_and_place` | 3cm Red Plastic Cube | Optical table, blue coaster target | 5 Waypoints | Pick cube, lift 15cm, place precisely on coaster |
| **Scene 2: Push Around Obstacle** | `push_to_target` | Yellow Wood Cylinder | Heavy metal tall obstacle block, goal marker | 7 Waypoints | Pre-pose, navigate around obstacle, push to goal |
| **Scene 3: Articulated Slide Drawer** | `articulate_joint` | Sliding Drawer Box | Cabinet base, prismatic slide joint (range: `[0, 0.12]m`) | 5 Waypoints | Approach handle, hook gripper, pull drawer open 10cm |

---

## 5-Gate Validation Pipeline

Every synthesized environment must pass five strict, objective physics and learning gates before deployment:

1. **Gate 1: Numerical Stability & Integration Health**
   - Simulates 1,000 physics steps (`timestep=0.002`, `implicitfast` integrator).
   - Enforces **zero NaNs / Infs** in `qpos`, `qvel`, and generalized accelerations (`|qacc| < 1e6`).
2. **Gate 2: Static Equilibrium & Penetration Check**
   - Settles objects for 250 steps without actuation.
   - Enforces resting drift **< 2.0 mm horizontally** and **< 1.0 mm vertically**.
   - Eliminates explosive contact forces from interpenetrating geoms.
3. **Gate 3: Non-Degenerate Reward Variance**
   - Steps 20 random exploratory actions.
   - Requires reward standard deviation $\sigma > 10^{-4}$ to ensure the RL policy receives a rich learning signal.
   - Automatically triggers the **Self-Healing Loop** if broken reward logic is detected.
4. **Gate 4: Closed-Loop Waypoint Oracle Solvability**
   - Executes deterministic proportional cartesian control along estimated 3D waypoints.
   - Enforces task completion and terminal goal attainment within 120 steps.
5. **Gate 5: Residual PPO Convergence**
   - Pre-trains a residual PPO policy ($\mathbf{u} = \mathbf{u}_{\text{oracle}} + 0.15 \cdot \mathbf{u}_{\text{ppo}}$) with Stable-Baselines3.
   - Verifies that trained evaluation return significantly outperforms random actions.

---

## Repository Architecture

```text
WorldForge-RL/
├── configs/
│   ├── materials.yaml                  # Physical density, friction & PBR material lookups
│   └── scenes/
│       ├── scene1_pick_place.json      # Golden Scene 1 specification
│       ├── scene2_push_obstacle.json   # Golden Scene 2 specification
│       └── scene3_articulated_drawer.json # Golden Scene 3 specification
├── src/
│   ├── api/
│   │   └── server.py                   # FastAPI REST & SSE backend + Live MuJoCo 3.17 simulation endpoints
│   ├── pipeline.py                     # 8-Stage autonomous end-to-end orchestrator
│   ├── schema/
│   │   └── scene_graph.py              # Pydantic V2 strict models + Auto-Snap height grounding
│   ├── perception/
│   │   ├── sam_tracker.py              # SAM 3.1 tracking, 4px boundary erosion & ground-truth sim extractor
│   │   ├── lift_3d.py                  # Pinhole back-projection, RANSAC table & SVD joint axis estimator
│   │   └── spark_reasoner.py           # Baseten meta-models/muse-spark-open-1026 reasoner + token stream
│   ├── compiler/
│   │   ├── mjcf_builder.py             # Deterministic SceneGraph -> MuJoCo 3.17 MJCF XML compiler
│   │   ├── base_env.py                 # WorldForgeEnv (Gymnasium 4-DoF Mocap Cartesian Env)
│   │   └── generated_reward.py         # Modular multi-stage dense reward functions
│   ├── validation/
│   │   ├── gates.py                    # Physics Validation Gates 1–5
│   │   ├── ppo_runner.py               # ResidualOracleEnv, SB3 PPO trainer & 120-frame MP4 recorder
│   │   └── repair_loop.py              # CostTracker + automated self-healing AST repair loop
│   ├── render_diff/
│   │   └── mask_iou.py                 # Mask IoU, coordinate refinement & green/red diff overlays
│   └── sandbox/
│       └── modal_executor.py           # Modal Sandbox cloud runner & micro-budget cap enforcement
├── frontend/                           # Studio-Grade React 18 + Vite + Tailwind CSS + Three.js
│   ├── src/
│   │   ├── App.tsx                     # 3-Pane Studio 2.0 dashboard + full-screen view switcher
│   │   ├── components/
│   │   │   └── SceneCanvas.tsx         # Interactive Three.js PBR WebGL stage with 4-DoF manual jogging
│   │   └── engine/
│   │       └── worldforge.ts           # Typed API client (ZERO mock math; all sim computed on backend)
├── outputs/                            # Static asset repository (rollout MP4s, masks, diff overlays)
├── scripts/
│   └── generate_previews.py            # Offline asset regeneration for all 3 golden scenes
├── run_demo.py                         # Single-command launcher for FastAPI backend + Vite frontend
├── start_studio.ps1                    # Native Windows PowerShell launcher
└── tests/
    ├── test_phase1.py                  # 25 Phase 1 tests (Schema, MJCF Builder, Base Env)
    ├── test_phase2.py                  # 27 Phase 2 tests (Dense Rewards, Gates 1–5, Self-Healing)
    ├── test_phase3.py                  # 9 Phase 3 tests (SAM 3.1, 3D Lift, SVD Axis, Muse Spark)
    ├── test_phase4.py                  # 14 Phase 4 tests (Render+Diff IoU, Modal Sandbox)
    └── test_phase5.py                  # 11 Phase 5 tests (Baseten, SSE Streaming, API Endpoints)
```

---

## Quickstart

### Prerequisites
- **Python 3.12**
- **Node.js 18+** & `npm`
- Windows 10/11 or Linux (Ubuntu 22.04+)

### 1. Clone & Set Up Python Environment

```bash
git clone https://github.com/syed-mohsin-s/WorldForge.git
cd WorldForge

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install Python backend dependencies
pip install --upgrade pip
pip install mujoco gymnasium pydantic pyyaml numpy scipy pytest stable-baselines3 "imageio[ffmpeg]" opencv-python-headless httpx openai fastapi "uvicorn[standard]" python-dotenv python-multipart
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Set your API keys in `.env` (optional for local/mock offline runs; required for live Baseten / Meta model streaming):

```env
# Baseten Hackathon $100 Grant
BASETEN_API_KEY="your_baseten_api_key_here"
BASETEN_BASE_URL="https://inference.baseten.co/v1"

# Meta Model API $150 Grant
META_API_KEY="your_meta_api_key_here"

# Modal Cloud Sandbox $130 Grant
MODAL_TOKEN_ID=""
MODAL_TOKEN_SECRET=""
```

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
npm run build
cd ..
```

---

## Running WorldForge Studio 2.0

Launch both the **FastAPI Python Backend** (port 8000) and **React Vite Frontend** (port 5173) with a single command:

```bash
# Cross-Platform Python Launcher:
python run_demo.py

# Or on Windows via PowerShell:
.\start_studio.ps1
```

Once running:
- **Interactive Studio UI**: [http://localhost:5173](http://localhost:5173)
- **FastAPI OpenAPI Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Studio 2.0 Interface Walkthrough

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ WorldForge Studio 2.0  [⚡ 8-Stage Pipeline Studio]  [🎮 3D PBR Workbench]  $0.0088   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 8-STAGE AUTONOMOUS PIPELINE:                                                           │
│ [1. SAM 3.1] [2. 3D LIFT] [3. MUSE SPARK] [4. SCENEGRAPH] [5. MJCF] [6. REWARD] ...   │
├───────────────────────┬────────────────────────────────┬───────────────────────────────┤
│ LEFT: PERCEPTION      │ CENTER: SYNTHESIS ARTIFACTS    │ RIGHT: VALIDATION & ROLLOUT   │
│ • SAM 3.1 4px Masks   │ • Spark Reasoning Token Stream │ • 4.0s Trained PPO MP4 (30fps)│
│ • Zero Flying Pixels  │ • Pydantic V2 SceneGraph JSON  │ • Pure Waypoint Oracle MP4    │
│ • Stage 8 Render+Diff │ • Compiled MuJoCo 3.17 MJCF    │ • 5-Gate Telemetry Checklist  │
│   Mask IoU (99.4%)    │ • Dense Reward Code & Diff     │ • Learning Return Curve       │
│ • Green/Red Overlays  │ • [🎮 Open 3D Workbench ↗]     │ • Self-Healing Retry Badge    │
└───────────────────────┴────────────────────────────────┴───────────────────────────────┘
```

1. **Scene Preset Selector**: Switch seamlessly between **Red Cube Pick & Place**, **Push Around Obstacle**, and **Articulated Slide Drawer** with automatic cache-busted image and rollout updates.
2. **Interactive 8-Stage Pipeline Bar**: Displays real-time status and timing badges across Meta AI and Python stages with zero vertical clipping.
3. **Live Baseten Reasoning Stream**: Watch `meta-models/muse-spark-open-1026` generate chain-of-thought domain reasoning tokens live via Server-Sent Events (SSE).
4. **Self-Healing Simulation**: Check **"Inject Gate 3 Fault (Demo Self-Healing)"** to watch Gate 3 fail, emit diagnostic logs, synthesize an automated AST patch, and re-validate 100% green.
5. **Interactive 3D PBR Workbench**: Switch into the full-screen Three.js WebGL workbench with 4-DoF manual jogging controls, waypoint spline trajectories, and real-time multi-channel reward waveforms—all executed directly by MuJoCo 3.17 on the backend.

---

## Running the Automated Test Suite

WorldForge includes an exhaustive test suite covering all 5 architectural phases:

```bash
# Run all 86 unit and integration tests:
.venv/Scripts/python.exe -m pytest tests/ -v
```

```text
======================== 86 passed in 70.61s (0:01:10) ========================
- tests/test_phase1.py : 25 PASSED (Schema validation, MJCF compilation, Gymnasium env)
- tests/test_phase2.py : 27 PASSED (Reward shaping, Gates 1–5, PPO convergence, self-healing)
- tests/test_phase3.py : 9 PASSED (SAM 3.1 erosion, depth back-projection, SVD joint axis)
- tests/test_phase4.py : 14 PASSED (Render+Diff IoU, coordinate refinement, Modal sandbox)
- tests/test_phase5.py : 11 PASSED (Baseten enforcement, SSE streaming, FastAPI live sim endpoints)
```

---

## Live Simulation Endpoints (FastAPI)

The backend exposes native endpoints for real-time MuJoCo 3.17 physics execution:

- `POST /api/sim/reset`: Initializes/resets a scene and returns real Cartesian gripper positions, body transforms, finger aperture, grasp flags, and dense reward.
- `POST /api/sim/step`: Steps MuJoCo physics with a 4-DoF action `[dx, dy, dz, grip]`.
- `POST /api/sim/oracle-step`: Steps the closed-loop Waypoint Oracle in MuJoCo physics.
- `GET /api/sim/trajectory`: Runs the complete oracle episode in MuJoCo physics (~60ms) and returns 120 steps of physics states for 30 FPS playback.
- `GET /api/scene-preview/{scene_id}`: Returns scene-specific preview image and rollout video URLs.

---

## Tech Stack

- **Physics & Simulation Engine**: MuJoCo 3.17.0, Gymnasium, Stable-Baselines3
- **Perception & 3D Math**: SAM 3.1 (Meta Model API), OpenCV, SciPy (RANSAC, SVD)
- **Generative AI & Reasoning**: Baseten `meta-models/muse-spark-open-1026` (OpenAI-compatible client), Meta Model API
- **Backend API**: FastAPI, Uvicorn, SSE Starlette, Pydantic V2
- **Frontend Workbench**: React 18, Vite, Three.js, OrbitControls, Tailwind CSS, Lucide Icons, Recharts
- **Testing & Verification**: Pytest, NumPy, ImageIO

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
