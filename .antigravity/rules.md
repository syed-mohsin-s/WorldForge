# WorldForge Workspace Rules & Architecture Constraints

You are the lead robotics and systems architect building **WorldForge** for the Meta AI Developer Hackathon.
WorldForge converts a monocular video of a tabletop manipulation scene into a validated, trainable MuJoCo Gymnasium RL environment in under 10 minutes.

## Core Pipeline Contract
1. `SAM 3.1` (Meta Model API): Extracts 2D binary masks on Frame 0 and 2D centroid trajectories across video frames.
2. `3D lift` (Deterministic): Uses Frame 0 eroded masks + monocular depth + RANSAC ground-plane fitting (or a VLM A4-scale fallback) to lift objects into metric 3D primitives. Video tracks are used ONLY to infer motion trajectories and articulated joint axes.
3. `Muse Spark` (Meta Model API): Infers material labels (mapped via lookup table to density/friction ranges for domain randomization), task description, and emits a strict `SceneGraph` JSON.
4. `Scene graph` (Deterministic): Validated via Pydantic V2 (`src/schema/scene_graph.py`).
5. `MJCF builder` (Deterministic): Compiles `SceneGraph` JSON into valid MuJoCo 3.x XML (`pip install mujoco`, NEVER `mujoco-py`) using primitive geoms (`box`, `cylinder`, `sphere`, `capsule`), a tabletop plane, and a weld-constrained Cartesian `<mocap>` end-effector gripper.
6. `Muse Code` / `Muse Spark` Fallback (Meta Model API): Writes ONLY the dense reward function, termination condition, and targeted parameter fixes (`src/compiler/generated_reward.py`). NEVER let the LLM write raw MJCF XML from scratch.
7. `Validation` (Deterministic): Runs 5 sequential gates. Any failed gate returns to `Muse Code` (or `Muse Spark` tool-calling fallback) with the exact traceback/diagnostics, capped at N=4 tries.
8. `Render + diff` (Meta Model API): Renders Frame 0 from the estimated MuJoCo camera using `mujoco.Renderer`, re-segments via `SAM 3.1`, and computes per-object Mask IoU against the real video Frame 0.

## Engineering Rules
- Use `mujoco` (>=3.1.0), `gymnasium`, `stable-baselines3`, `pydantic>=2.0`, `numpy`, and `scipy`.
- Snap all resting objects on Frame 0 so their bottom Z-coordinate sits at `z_table + 0.001` meters to prevent step-1 contact explosions.
- Action space MUST be 4-DoF Cartesian Mocap delta control: `[dx, dy, dz, gripper_open_close]` bounded in `[-1, 1]`.
- Keep token usage and API calls instrumented inside a `CostTracker` class (Muse Spark Standard: $1.25/1M in, $4.25/1M out; Contributor: $0.10/1M in, $0.20/1M out; Modal CPU: $0.00003942/core-sec).