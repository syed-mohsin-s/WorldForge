"""WorldForge Dense Reward Module — multi-stage task rewards.

Supports four task types with smooth, shaped dense rewards:
  * ``pick_and_place``:   reach → grasp → lift → place → success
  * ``push_to_target``:   reach → push progress → success
  * ``articulate_joint``: reach → joint displacement → success
  * ``multi_stack``:      Stage 1 (obj_1 onto base obj_0, +5.0 subgoal) →
                          Stage 2 (obj_2 onto obj_1, +10.0 terminal success)

The ``compute_reward`` function is the default reward for ``WorldForgeEnv``
when no custom ``reward_fn`` is passed.  It can also be overridden at
runtime by the self-healing repair loop.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from src.schema.scene_graph import ObjectNode, SceneGraph


def compute_reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: SceneGraph,
) -> tuple[float, bool, dict]:
    """Compute a dense shaped reward for the given task type.

    Returns ``(reward, terminated, info_dict)``.
    """
    task_type = scene.task.task_type
    if task_type == "pick_and_place":
        return _pick_and_place_reward(model, data, scene)
    elif task_type == "push_to_target":
        return _push_to_target_reward(model, data, scene)
    elif task_type == "articulate_joint":
        return _articulate_joint_reward(model, data, scene)
    elif task_type == "multi_stack":
        return _multi_stack_reward(model, data, scene)
    else:
        raise ValueError(f"Unknown task type: {task_type!r}")


# ─── Helper to get body position ──────────────────────────────────────

def _body_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"Body {name!r} not found in compiled MuJoCo model")
    return data.xpos[bid].copy()


def _gripper_pos(data: mujoco.MjData) -> np.ndarray:
    """Return the current gripper (mocap) position."""
    return data.mocap_pos[0].copy()


def _finger_aperture(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Sum of left + right finger joint positions (0 = closed, ~0.08 = open)."""
    jl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint_L")
    jr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint_R")
    return float(
        data.qpos[model.jnt_qposadr[jl]] + data.qpos[model.jnt_qposadr[jr]]
    )


def _tanh_reward(dist: float, scale: float = 5.0) -> float:
    """Smooth tanh-shaped reward: 1.0 at dist=0, ~0 at dist >> 1/scale."""
    return 1.0 - math.tanh(scale * dist)


def _node_half_height(obj: ObjectNode) -> float:
    if obj.primitive == "box":
        return float(obj.size[2])
    elif obj.primitive == "cylinder":
        return float(obj.size[1])
    return float(obj.size[0])


# ─── Pick and Place ───────────────────────────────────────────────────

def _pick_and_place_reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: SceneGraph,
) -> tuple[float, bool, dict]:
    obj_id = scene.task.manipulated_object_id
    target = np.array(scene.task.target_position if scene.task.target_position is not None else (-0.15, 0.1, 0.035))

    grip = _gripper_pos(data)
    obj = _body_pos(model, data, obj_id)
    aperture = _finger_aperture(model, data)

    d_reach = float(np.linalg.norm(grip - obj))
    d_place = float(np.linalg.norm(obj - target))

    # Stage 1: Reach — move gripper toward object
    reach_reward = _tanh_reward(d_reach, scale=4.0)

    # Stage 2: Grasp bonus — fingers closed AND near object
    grasp_bonus = 0.0
    if d_reach < 0.065 and aperture < 0.055:
        grasp_bonus = 1.0

    # Stage 3: Lift reward — object Z above table
    lift_reward = 0.0
    obj_z = float(obj[2])
    if obj_z > scene.table_z + 0.04:
        lift_reward = min((obj_z - scene.table_z - 0.04) * 20.0, 1.0)

    # Stage 4: Place — move object toward target
    place_reward = _tanh_reward(d_place, scale=4.0)

    # Combine with stage weighting
    reward = 0.25 * reach_reward + 0.15 * grasp_bonus + 0.1 * lift_reward + 0.5 * place_reward

    # Terminal success: object within 4 cm of target
    terminated = d_place < 0.045
    if terminated:
        reward += 10.0

    info: dict[str, Any] = {
        "d_reach": d_reach,
        "d_place": d_place,
        "reach_reward": reach_reward,
        "grasp_bonus": grasp_bonus,
        "lift_reward": lift_reward,
        "place_reward": place_reward,
        "obj_pos": obj.tolist(),
    }
    return reward, terminated, info


# ─── Push to Target ──────────────────────────────────────────────────

def _push_to_target_reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: SceneGraph,
) -> tuple[float, bool, dict]:
    obj_id = scene.task.manipulated_object_id
    target = np.array(scene.task.target_position if scene.task.target_position is not None else (-0.15, 0.1, 0.04))

    grip = _gripper_pos(data)
    obj = _body_pos(model, data, obj_id)

    d_reach = float(np.linalg.norm(grip[:2] - obj[:2]))  # XY only for push
    d_goal = float(np.linalg.norm(obj[:2] - target[:2]))

    # Stage 1: Reach — move gripper near object (XY plane)
    reach_reward = _tanh_reward(d_reach, scale=8.0)

    # Stage 2: Push progress — object moving toward target
    progress_reward = _tanh_reward(d_goal, scale=6.0)

    # Obstacle clearance penalty — penalise gripper being far above table
    grip_z = float(grip[2])
    height_penalty = -0.1 * max(grip_z - scene.table_z - 0.08, 0.0)

    reward = 0.3 * reach_reward + 0.6 * progress_reward + height_penalty

    # Terminal success: object within 5 cm of target (XY)
    terminated = d_goal < 0.05
    if terminated:
        reward += 10.0

    info: dict[str, Any] = {
        "d_reach": d_reach,
        "d_goal": d_goal,
        "reach_reward": reach_reward,
        "grasp_bonus": float(max(0.0, 1.0 + height_penalty)),
        "lift_reward": progress_reward * 0.5,
        "place_reward": progress_reward,
        "obj_pos": obj.tolist(),
    }
    return reward, terminated, info


# ─── Articulate Joint ────────────────────────────────────────────────

def _articulate_joint_reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: SceneGraph,
) -> tuple[float, bool, dict]:
    obj_id = scene.task.manipulated_object_id
    target_delta = scene.task.target_joint_delta or 0.10

    grip = _gripper_pos(data)
    aperture = _finger_aperture(model, data)

    # Find the joint on the manipulated object
    joint_name = f"{obj_id}_joint"
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"Joint {joint_name!r} not found in compiled MuJoCo model")
    joint_pos = float(data.qpos[model.jnt_qposadr[jid]])

    # Use the handle or object body for reach target
    reach_target_name = obj_id
    for obj_node in scene.objects:
        if obj_node.parent_id == obj_id or (
            obj_node.parent_id is not None
            and any(
                o.id == obj_node.parent_id and o.parent_id == obj_id
                for o in scene.objects
            )
        ):
            reach_target_name = obj_node.id
    for obj_node in scene.objects:
        if obj_node.parent_id is not None:
            parent = next((o for o in scene.objects if o.id == obj_node.parent_id), None)
            if parent and parent.parent_id == obj_id:
                reach_target_name = obj_node.id
                break
        if obj_node.parent_id == obj_id and "handle" in obj_node.label.lower():
            reach_target_name = obj_node.id
            break

    reach_body = _body_pos(model, data, reach_target_name)
    d_reach = float(np.linalg.norm(grip - reach_body))

    # Stage 1: Reach
    reach_reward = _tanh_reward(d_reach, scale=10.0)

    # Stage 2: Joint displacement toward target
    progress = abs(joint_pos) / max(abs(target_delta), 1e-6)
    progress = min(progress, 1.0)
    joint_reward = progress

    grasp_bonus = 1.0 if (d_reach < 0.065 and aperture < 0.045) else 0.0
    reward = 0.3 * reach_reward + 0.7 * joint_reward

    # Terminal success: joint reached target threshold (within 20% tolerance)
    terminated = abs(joint_pos) >= abs(target_delta) * 0.8
    if terminated:
        reward += 10.0

    info: dict[str, Any] = {
        "d_reach": d_reach,
        "d_goal": max(0.0, abs(target_delta) - abs(joint_pos)),
        "joint_pos": joint_pos,
        "target_delta": target_delta,
        "progress": progress,
        "reach_reward": reach_reward,
        "grasp_bonus": grasp_bonus,
        "lift_reward": progress * 0.5,
        "place_reward": joint_reward,
    }
    return reward, terminated, info


# ─── Progressive Multi-Object Tower Stacking ─────────────────────────

def _multi_stack_reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: SceneGraph,
) -> tuple[float, bool, dict]:
    """Two-stage progressive dense reward for stacking 3 objects [base, mid, top]."""
    obj_map = {o.id: o for o in scene.objects}
    order = scene.task.stack_order
    if not order or len(order) < 3:
        order = [o.id for o in scene.objects[:3]]
    if len(order) < 3:
        return _pick_and_place_reward(model, data, scene)

    id_0, id_1, id_2 = order[0], order[1], order[2]
    pos_0 = _body_pos(model, data, id_0)
    pos_1 = _body_pos(model, data, id_1)
    pos_2 = _body_pos(model, data, id_2)

    hz_0 = _node_half_height(obj_map[id_0]) if id_0 in obj_map else 0.02
    hz_1 = _node_half_height(obj_map[id_1]) if id_1 in obj_map else 0.02
    hz_2 = _node_half_height(obj_map[id_2]) if id_2 in obj_map else 0.02

    grip = _gripper_pos(data)
    aperture = _finger_aperture(model, data)

    target_1_on_0 = np.array([pos_0[0], pos_0[1], pos_0[2] + hz_0 + hz_1], dtype=np.float64)
    d_xy_10 = float(np.linalg.norm(pos_1[:2] - pos_0[:2]))
    d_place_1 = float(np.linalg.norm(pos_1 - target_1_on_0))
    stage1_stacked = (d_xy_10 < 0.035) and (float(pos_1[2]) > float(pos_0[2]) + hz_0 * 0.75)

    target_2_on_1 = np.array([pos_1[0], pos_1[1], pos_1[2] + hz_1 + hz_2], dtype=np.float64)
    d_xy_21 = float(np.linalg.norm(pos_2[:2] - pos_1[:2]))
    d_place_2 = float(np.linalg.norm(pos_2 - target_2_on_1))
    stage2_stacked = (
        stage1_stacked
        and (d_xy_21 < 0.035)
        and (float(pos_2[2]) > float(pos_1[2]) + hz_1 * 0.75)
    )

    if not stage1_stacked:
        # Stage 1: Reach, grasp, lift, and place obj_1 onto base obj_0
        d_reach = float(np.linalg.norm(grip - pos_1))
        reach_reward = _tanh_reward(d_reach, scale=4.0)
        grasp_bonus = 1.0 if (d_reach < 0.065 and aperture < 0.060) else 0.0
        lift_reward = min(max(float(pos_1[2]) - (scene.table_z + hz_1 + 0.015), 0.0) * 20.0, 1.0)
        place_reward = _tanh_reward(d_place_1, scale=4.0)
        reward = 0.25 * reach_reward + 0.15 * grasp_bonus + 0.10 * lift_reward + 0.50 * place_reward
        d_goal = d_place_1
    else:
        # Stage 2: Subgoal bonus (+5.0) locked in; now stack obj_2 onto obj_1
        d_reach = float(np.linalg.norm(grip - pos_2))
        reach_reward = _tanh_reward(d_reach, scale=4.0)
        grasp_bonus = 1.0 if (d_reach < 0.065 and aperture < 0.060) else 0.0
        lift_reward = min(max(float(pos_2[2]) - (scene.table_z + hz_2 + 0.025), 0.0) * 15.0, 1.0)
        place_reward = _tanh_reward(d_place_2, scale=4.0)
        reward = 5.0 + 0.25 * reach_reward + 0.15 * grasp_bonus + 0.10 * lift_reward + 0.50 * place_reward
        d_goal = d_place_2

    terminated = bool(stage2_stacked)
    if terminated:
        reward += 10.0

    info: dict[str, Any] = {
        "d_reach": d_reach,
        "d_place": d_goal,
        "d_goal": d_goal,
        "stage1_stacked": stage1_stacked,
        "stage2_stacked": stage2_stacked,
        "reach_reward": reach_reward,
        "grasp_bonus": grasp_bonus,
        "lift_reward": lift_reward,
        "place_reward": place_reward,
        "obj_pos": pos_2.tolist(),
    }
    return reward, terminated, info