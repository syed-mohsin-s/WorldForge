# WorldForge-RL: Autonomous Video-to-Environment Synthesis

> **Autonomous Synthesis of Validated, Trainable MuJoCo 3.x Gymnasium Environments from Monocular Video & Natural Language Prompts**

[![CI Tests](https://img.shields.io/badge/tests-102%20passed%20(100%25)-brightgreen.svg)]()
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)]()
[![MuJoCo 3.x](https://img.shields.io/badge/mujoco-3.17.0-orange.svg)]()
[![Baseten](https://img.shields.io/badge/Baseten-meta--models%2Fmuse--spark--open--1026-violet.svg)]()
[![React 19](https://img.shields.io/badge/frontend-React%2019%20%2B%20Three.js-cyan.svg)]()
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is WorldForge?

**WorldForge** transforms uncalibrated monocular tabletop videos or natural language task intents into fully grounded, physics-accurate, and trainable **MuJoCo 3.17 Gymnasium RL environments** in under 30 seconds. By automating vision tracking, metric 3D lifting, physical spatial reasoning, MJCF compilation, dense reward generation, and multi-gate physics verification, WorldForge reduces manual robot simulation setup time from **~6 hours to under 30 seconds** (a **>700× acceleration**).

```
Uncalibrated Video / Prompt  ──►  WorldForge Studio  ──►  Deployable Gymnasium Package (.zip)
   (Real phone video or          (8-stage autonomous        (MuJoCo MJCF + Gymnasium Env
    task description)             verification pipeline)     + Dense Rewards + PPO Policy)
```

---

## Key Capabilities

- **100% Python MuJoCo 3.17 Ground Truth**: Contact dynamics, elliptic friction cones (`condim="4"`), Cartesian mocap kinematics, and residual PPO policy training execute in real MuJoCo 3.17. The frontend contains zero synthetic telemetry or simulated client-side mock physics.
- **Baseten Muse Spark Reasoning**: Native integration with `meta-models/muse-spark-open-1026` via Baseten's OpenAI-compatible inference endpoint, streaming live spatial chain-of-thought traces with concise executive summaries.
- **Metric Camera Calibration via A4 SolvePnP**: Detects metric A4 calibration sheets ($0.297\,\text{m} \times 0.210\,\text{m}$ at $z = 0$) in Frame 0 to solve phone camera extrinsics and intrinsics using `cv2.solvePnP`, lifting realistic 3D extents via ray-plane intersection and achieving up to **75%+ Mask IoU** on real phone video.
- **Prompt-Driven Multi-Stacking with CoM Stability**: Synthesizes complex multi-object stacking towers (`multi_stack`) from user prompts. Evaluates Center-of-Mass (CoM) stability, footprint bounding box areas, and 2D visual extent tiebreaking to order objects from stable wide bases to lighter tops.
- **Automated Self-Healing Loop**: If physics gates fail (such as static drift, penetration explosion, or degenerate zero-variance reward), Muse Spark diagnoses runtime stack traces and auto-generates unified AST patches to hot-reload and re-verify the environment.
- **Hardened AST Sandbox with Subprocess Timeouts**: Restricts synthesized reward code to mathematical and kinematic operations, disallowing dunder escapes, file I/O, and unsafe pickling, guarded by POSIX `SIGALRM` and per-call execution deadlines that prevent runaway loops.
- **1-Click Self-Contained Gymnasium Export**: Generates a complete, downloadable `.zip` package with `gym.register(...)`, `run_standalone.py`, modular reward code, 10-step consecutive velocity settlement verification, and domain randomization (`±1 cm` initial reset perturbation).
- **Dual-Cockpit Studio UI**: Built with React 19, Vite, Tailwind CSS, and Three.js PBR rendering, featuring real-time Server-Sent Events (SSE) telemetry, 4-DoF manual jogging, and visual diff mask overlays.

---

## 8-Stage Autonomous Pipeline Architecture

```
Uncalibrated Monocular Video / Natural Language Prompt
         │
         ▼  (Stage 1: SAM 3.1 Perception)
 4-Tier Tracker: Meta API ──► Self-Hosted Unsloth SAM 3.1 ──► Local OpenCV HSV Fallback
 4px Boundary Erosion, Mask Centroid & Bounding Box Extraction
         │
         ▼  (Stage 2: 3D Metric Lifting & Calibration)
 Pinhole Ray-Plane Homography + A4 Sheet solvePnP Calibration (0.297m x 0.210m)
 RANSAC Table Plane Fitting (z <= 0.01m) + SVD Prismatic/Revolute Joint Axis Fitting
         │
         ▼  (Stage 3: Muse Spark Spatial Reasoning)
 Baseten meta-models/muse-spark-open-1026 ── Physical reasoning & CoM stability ordering
 Material prior mapping (configs/materials.yaml) & Waypoint Oracle synthesis
         │
         ▼  (Stage 4: Scene Graph Synthesis & Grounding)
 Pydantic V2 SceneGraph ── Auto-Snap Elevation Grounding (z = table_z + h/2 + 0.001m)
 Coordinate alignment & joint constraint hierarchy verification
         │
         ▼  (Stage 5: MJCF XML Compilation)
 SceneCompiler ── MuJoCo 3.17 XML with Mocap Welds, Elliptic Cones, and 2.5m x 2.5m Table
 Inner finger pads mounted flush (0.0 cm gap closed, 8.0 cm span open)
         │
         ▼  (Stage 6: Multi-Stage Dense Reward Synthesis)
 Modular Reward Engine ── Dense tanh reach/grasp/lift shaping + 10-step velocity settlement
 AST Security Validator + Scrubbed Proxy Environment
         │
         ▼  (Stage 7: 5-Gate Validation & Residual PPO)
 Gate 1: Numerical Stability (500 steps, zero NaNs, |qacc| < 1e6)
 Gate 2: Static Equilibrium (250 steps, drift < 2mm h, < 1mm v)
 Gate 3: Reward Variance (sigma > 1e-4, non-degenerate gradient)
 Gate 4: Closed-Loop Waypoint Oracle (deterministic solvability within horizon)
 Gate 5: Residual PPO Smoke Test (SB3 PPO beats random baseline)
 [Self-Healing Loop]: Hot-patches broken reward functions on Gate failure
         │
         ▼  (Stage 8: Render + Diff Visual Alignment)
 MuJoCo Offscreen EGL/OSMesa Rendering vs. Source Video Masks
 Closed-loop coordinate refinement optimizing Mean Mask IoU (threshold >= 0.60)
         │
         ▼
 1-Click Exportable Gymnasium Environment Package (.zip) & 4.0s Verification MP4
```

---

## 5-Gate Validation Suite

Every synthesized environment is subjected to five objective validation gates before acceptance:

| Gate | Verification Target | Pass Criteria | Failure Remediation |
|---|---|---|---|
| **Gate 1** | Numerical Stability & Integration | 500 steps, `timestep=0.002s`, zero NaNs/Infs, $\|qacc\| < 10^6$ | Damps joint velocities, relaxes stiff contact constraints |
| **Gate 2** | Static Equilibrium & Penetration | 250 unactuated steps, horizontal drift $< 2.0\,\text{mm}$, vertical drift $< 1.0\,\text{mm}$ | Auto-snaps object elevation to $z_{\text{table}} + \frac{h}{2} + 0.001\,\text{m}$ |
| **Gate 3** | Non-Degenerate Reward Variance | 3 exploratory rollouts, standard deviation $\sigma > 10^{-4}$ | Triggers **Self-Healing Loop**: Muse Spark diagnoses failure & applies unified diff |
| **Gate 4** | Closed-Loop Waypoint Oracle | Proportional Cartesian control along 3D waypoints reaches terminal goal | Adjusts approach heights, grasp dwells, and waypoints |
| **Gate 5** | Residual PPO Trainability | Residual policy ($\mathbf{u} = \mathbf{u}_{\text{oracle}} + 0.15 \cdot \mathbf{u}_{\text{ppo}}$) outperforms random actions | Calibrates dense shaping weights and dwell steps |

---

## Golden Verification Benchmark Scenes

WorldForge includes pre-calibrated golden benchmarks covering core manipulation primitives:

| Scene Preset | Task Type | Manipulated Object | Distractors / Articulation | Waypoints | Objective |
|---|---|---|---|---|---|
| **Scene 1: Pick & Place** | `pick_and_place` | $5\times 5\times 5\,\text{cm}$ Red Cube | Table, $\varnothing 8\times 1\,\text{cm}$ Wooden Coaster | 7 Waypoints | Grasp cube at $(15, 0)\,\text{cm}$, lift $14\,\text{cm}$, place on coaster at $(-15, 10)\,\text{cm}$ |
| **Scene 2: Push Obstacle** | `push_to_target` | $\varnothing 4\times 8\,\text{cm}$ Blue Cylinder | $6\times 6\times 12\,\text{cm}$ Metal Box, Rubber Marker | 6 Waypoints | Pre-pose, navigate around metal obstacle, push cylinder to target marker |
| **Scene 3: Articulated Drawer** | `articulate_joint` | $20\times 1\times 12\,\text{cm}$ Drawer Panel | $24\times 20\times 16\,\text{cm}$ Cabinet, Prismatic Slide | 5 Waypoints | Approach handle, hook gripper, pull drawer open $10\,\text{cm}$ along $-Y$ axis |
| **Scene 4: Multi-Object Stacking** | `multi_stack` | 3 Geometric Objects (Cube / Prism) | Tabletop workspace | 16 Waypoints | Order objects by CoM stability, grasp, lift, and stack sequentially into a stable tower |

---

## 1-Click Exported Gymnasium Package

Clicking **Export Gym Package** in Studio 2.0 generates an isolated, self-contained Python package:

```
WorldForge-scene1_pick_place-v0/
├── environment.xml      # Compiled MuJoCo 3.17 MJCF with geometries, contacts, mocap weld
├── scene_graph.json     # Declarative SceneGraph specification
├── reward_code.py       # Standalone dense reward function (works with dicts & Pydantic)
├── run_standalone.py    # Gymnasium environment wrapper + test rollout script
├── requirements.txt     # Minimal dependencies (mujoco, gymnasium, numpy)
└── README.md            # Quickstart documentation & sample usage
```

### Using the Exported Package

```python
import gymnasium as gym
import run_standalone  # Auto-registers WorldForge-{scene_id}-v0

# Initialize environment with optional reset domain randomization
env = gym.make("WorldForge-scene1_pick_place-v0", randomize_reset=True)
obs, info = env.reset(seed=42)

for step in range(400):
    action = env.action_space.sample()  # 4-DoF [dx, dy, dz, grip_cmd]
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(f"Episode completed at step {step} with reward {reward:.2f}")
        break

env.close()
```

---

## Repository Structure

```text
WorldForge-RL/
├── configs/
│   ├── materials.yaml                  # Physical density, friction & PBR material properties
│   └── scenes/
│       ├── scene1_pick_place.json      # Golden Scene 1 specification
│       ├── scene2_push_obstacle.json   # Golden Scene 2 specification
│       └── scene3_articulated_drawer.json # Golden Scene 3 specification
├── src/
│   ├── api/
│   │   └── server.py                   # FastAPI REST, SSE streaming & live MuJoCo 3.17 simulation
│   ├── pipeline.py                     # 8-Stage end-to-end master orchestrator
│   ├── schema/
│   │   └── scene_graph.py              # Pydantic V2 models, physical grounding & validation
│   ├── perception/
│   │   ├── sam_tracker.py              # SAM 3.1 4-tier tracking & 4px boundary erosion
│   │   ├── lift_3d.py                  # A4 solvePnP calibration, RANSAC table & ray-plane lifting
│   │   └── spark_reasoner.py           # Baseten Muse Spark reasoner & CoM stability logic
│   ├── compiler/
│   │   ├── mjcf_builder.py             # Deterministic SceneGraph -> MuJoCo 3.17 MJCF compiler
│   │   ├── base_env.py                 # WorldForgeEnv (Gymnasium 4-DoF Cartesian Mocap Env)
│   │   └── generated_reward.py         # Modular multi-stage dense rewards & settlement checks
│   ├── validation/
│   │   ├── gates.py                    # Physics Validation Gates 1–5
│   │   ├── ppo_runner.py               # ResidualOracleEnv, SB3 PPO trainer & 8-checkpoint curve
│   │   └── repair_loop.py              # AST sandbox, SIGALRM/subprocess timeout & self-healing
│   ├── render_diff/
│   │   └── mask_iou.py                 # Offscreen MuJoCo rendering, Mask IoU & visual overlays
│   └── sandbox/
│       └── modal_executor.py           # Modal cloud runner & micro-budget accounting ($30 grant)
├── frontend/                           # Studio 2.0 cockpit (React 19, Vite, Tailwind, Three.js)
│   ├── src/
│   │   ├── App.tsx                     # 3-Pane Studio dashboard & telemetry viewer
│   │   ├── components/
│   │   │   └── SceneCanvas.tsx         # Interactive Three.js WebGL PBR stage with 4-DoF jogging
│   │   └── engine/
│   │       └── worldforge.ts           # Typed API engine client (backend-driven simulation)
├── outputs/                            # Rollout MP4s, masks, and render-diff comparison assets
├── scripts/
│   ├── generate_previews.py            # Headless asset regeneration for golden scenes
│   └── e2e_verify.cjs                  # Headless Puppeteer end-to-end frontend verification
├── Dockerfile                          # Multi-stage production container (Node 20 + Python 3.12 + uv)
├── docker-compose.yml                  # Docker Compose stack with volume mounts & healthcheck
├── requirements.txt                    # Top-level Python dependency specification
├── requirements.lock                   # Deterministic hermetic lockfile (compiled via uv)
├── run_demo.py                         # Unified launcher for FastAPI backend + Vite frontend
├── start_studio.ps1                    # Native Windows PowerShell one-click launcher
└── tests/
    ├── test_phase1.py                  # Phase 1: Schema, MJCF Builder, Base Environment
    ├── test_phase2.py                  # Phase 2: Dense Rewards, Gates 1–5, Self-Healing, AST Sandbox
    ├── test_phase3.py                  # Phase 3: SAM 3.1 Tracking, A4 solvePnP, 3D Lifting, SVD Axis
    ├── test_phase4.py                  # Phase 4: Render + Diff Mask IoU, Modal Sandbox Accounting
    └── test_phase5.py                  # Phase 5: Baseten Live Synthesis, SSE Streaming, API Endpoints
```

---

## Quickstart Guide

### Prerequisites
- **Python 3.12**
- **Node.js 18+** & `npm`
- Windows 10/11, macOS, or Linux (Ubuntu 22.04+)

### 1. Environment Setup

#### Option A: Fast Setup with Astral `uv` (Recommended)
```bash
# Clone the repository
git clone https://github.com/syed-mohsin-s/WorldForge.git
cd WorldForge

# Create and activate virtual environment
uv venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install deterministic dependencies
uv pip sync --torch-backend cpu requirements.lock
```

#### Option B: Standard Python `venv` + `pip`
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Credentials

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` with your API credentials (all features cleanly fall back to deterministic local physics if keys are not set):

```env
# Baseten Hackathon Inference ($100 Grant)
BASETEN_API_KEY="your_baseten_api_key_here"
BASETEN_BASE_URL="https://inference.baseten.co/v1"

# Meta Model API ($150 Grant)
META_API_KEY="your_meta_api_key_here"
META_SAM_API_URL=""

# Modal Cloud Sandbox ($30 Grant)
MODAL_TOKEN_ID=""
MODAL_TOKEN_SECRET=""
```

### 3. Build Frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### 4. Launch WorldForge Studio 2.0

Run the combined backend and frontend servers with one command:

```bash
# Cross-platform launcher:
python run_demo.py

# Or on Windows via PowerShell:
.\start_studio.ps1
```

Once started:
- **WorldForge Studio 2.0**: [http://localhost:5173](http://localhost:5173)
- **FastAPI Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Verification & Testing

The comprehensive test suite covers all 5 phases of the system with **102 integration and unit tests**:

```bash
# Run the complete test suite:
pytest tests/ -v

# Run specific phase tests:
pytest tests/test_phase1.py -v   # Schema & MJCF Builder
pytest tests/test_phase2.py -v   # Physics Gates & Self-Healing Loop
pytest tests/test_phase3.py -v   # SAM 3.1 & 3D Metric Lifting
pytest tests/test_phase4.py -v   # Render + Diff & Modal Sandbox
pytest tests/test_phase5.py -v   # Baseten Reasoner & API Endpoints
```

### Headless Browser End-to-End Verification
```bash
node scripts/e2e_verify.cjs
```

---

## Docker Deployment

WorldForge includes a production-ready, multi-stage Dockerfile with headless OSMesa rendering:

```bash
# Build and start via Docker Compose:
docker compose up --build

# Run in background:
docker compose up -d
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
