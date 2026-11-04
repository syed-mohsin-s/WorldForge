"""WorldForge Dense Reward Module — standalone multi-stage task rewards.

Supports four task types with smooth, shaped dense rewards and zero runtime
dependency on internal repository packages (works with both Pydantic SceneGraph
instances and raw JSON dicts in exported Gymnasium packages):
  * ``pick_and_place``:   reach → grasp → lift → place → settled success
  * ``push_to_target``:   reach → push progress → settled success
  * ``articulate_joint``: reach → joint displacement → success
  * ``multi_stack``:      Stage 1 (obj_1 onto base obj_0) →
                          Stage 2 (obj_2 onto obj_1 with <=1.8cm XY alignment,
                          open jaws, and settled velocity)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

if TYPE_CHECKING:
    from src.schema.scene_graph import ObjectNode, SceneGraph


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read an attribute or dict key from either a Pydantic model or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def compute_reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: "SceneGraph | dict[str, Any]",
) -> tuple[float, bool, dict[str, Any]]:
    """Compute a dense shaped reward for the given task type."""
    task = _get(scene, "task", {})
    task_type = _get(task, "task_type", "pick_and_place")
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


def _body_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"Body {name!r} not found in compiled MuJoCo model")
    return data.xpos[bid].copy()


def _body_speed(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        return 0.0
    return float(np.linalg.norm(data.cvel[bid][3:6]))


def _gripper_pos(data: mujoco.MjData) -> np.ndarray:
    """Return the current gripper (mocap) position."""
    return data.mocap_pos[0].copy()


def _finger_aperture(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Inner gap between left and right finger pads (0.0 = flush closed, ~0.08 = fully open)."""
    jl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint_L")
    jr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint_R")
    return float(
        data.qpos[model.jnt_qposadr[jl]] + data.qpos[model.jnt_qposadr[jr]]
    )


def _tanh_reward(dist: float, scale: float = 5.0) -> float:
    """Smooth tanh-shaped reward: 1.0 at dist=0, ~0 at dist >> 1/scale."""
    return 1.0 - math.tanh(scale * dist)


def _node_half_height(obj: "ObjectNode | dict[str, Any]") -> float:
    prim = _get(obj, "primitive", "box")
    size = _get(obj, "size", [0.025, 0.025, 0.025])
    if prim == "box":
        return float(size[2])
    elif prim == "cylinder":
        return float(size[1])
    return float(size[0])


# ─── Pick and Place ───────────────────────────────────────────────────

def _pick_and_place_reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: "SceneGraph | dict[str, Any]",
) -> tuple[float, bool, dict[str, Any]]:
    task = _get(scene, "task", {})
    table_z = float(_get(scene, "table_z", 0.0))
    obj_id = _get(task, "manipulated_object_id", "")
    raw_target = _get(task, "target_position", None)
    target = np.array(raw_target if raw_target is not None else (-0.15, 0.1, 0.035), dtype=np.float64)

    grip = _gripper_pos(data)
    obj = _body_pos(model, data, obj_id)
    aperture = _finger_aperture(model, data)

    d_reach = float(np.linalg.norm(grip - obj))
    d_place = float(np.linalg.norm(obj - target))

    reach_reward = _tanh_reward(d_reach, scale=4.0)

    grasp_bonus = 0.0
    if d_reach < 0.065 and aperture < 0.055:
        grasp_bonus = 1.0

    lift_reward = 0.0
    obj_z = float(obj[2])
    if obj_z > table_z + 0.04:
        lift_reward = min((obj_z - table_z - 0.04) * 20.0, 1.0)

    place_reward = _tanh_reward(d_place, scale=4.0)

    reward = 0.25 * reach_reward + 0.15 * grasp_bonus + 0.1 * lift_reward + 0.5 * place_reward

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
        "success": bool(terminated),
        "is_success": bool(terminated),
    }
    return reward, terminated, info


# ─── Push to Target ──────────────────────────────────────────────────

def _push_to_target_reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: "SceneGraph | dict[str, Any]",
) -> tuple[float, bool, dict[str, Any]]:
    task = _get(scene, "task", {})
    table_z = float(_get(scene, "table_z", 0.0))
    obj_id = _get(task, "manipulated_object_id", "")
    raw_target = _get(task, "target_position", None)
    target = np.array(raw_target if raw_target is not None else (-0.15, 0.1, 0.04), dtype=np.float64)

    grip = _gripper_pos(data)
    obj = _body_pos(model, data, obj_id)

    d_reach = float(np.linalg.norm(grip[:2] - obj[:2]))
    d_goal = float(np.linalg.norm(obj[:2] - target[:2]))

    reach_reward = _tanh_reward(d_reach, scale=8.0)
    progress_reward = _tanh_reward(d_goal, scale=6.0)

    grip_z = float(grip[2])
    height_penalty = -0.1 * max(grip_z - table_z - 0.08, 0.0)

    reward = 0.3 * reach_reward + 0.6 * progress_reward + height_penalty

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
        "success": bool(terminated),
        "is_success": bool(terminated),
    }
    return reward, terminated, info


# ─── Articulate Joint ────────────────────────────────────────────────

def _articulate_joint_reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: "SceneGraph | dict[str, Any]",
) -> tuple[float, bool, dict[str, Any]]:
    task = _get(scene, "task", {})
    objects = _get(scene, "objects", [])
    obj_id = _get(task, "manipulated_object_id", "")
    target_delta = _get(task, "target_joint_delta", None) or 0.10

    grip = _gripper_pos(data)
    aperture = _finger_aperture(model, data)

    joint_name = f"{obj_id}_joint"
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"Joint {joint_name!r} not found in compiled MuJoCo model")
    joint_pos = float(data.qpos[model.jnt_qposadr[jid]])

    reach_target_name = obj_id
    for obj_node in objects:
        pid = _get(obj_node, "parent_id", None)
        nid = _get(obj_node, "id", "")
        nlabel = str(_get(obj_node, "label", "")).lower()
        if pid is not None:
            parent = next((o for o in objects if _get(o, "id", "") == pid), None)
            if parent and _get(parent, "parent_id", None) == obj_id:
                reach_target_name = nid
                break
        if pid == obj_id and "handle" in nlabel:
            reach_target_name = nid
            break

    reach_body = _body_pos(model, data, reach_target_name)
    d_reach = float(np.linalg.norm(grip - reach_body))

    reach_reward = _tanh_reward(d_reach, scale=10.0)

    progress = abs(joint_pos) / max(abs(target_delta), 1e-6)
    progress = min(progress, 1.0)
    joint_reward = progress

    grasp_bonus = 1.0 if (d_reach < 0.065 and aperture < 0.045) else 0.0
    reward = 0.3 * reach_reward + 0.7 * joint_reward

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
        "success": bool(terminated),
        "is_success": bool(terminated),
    }
    return reward, terminated, info


# ─── Progressive Multi-Object Tower Stacking ─────────────────────────

def _multi_stack_reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: "SceneGraph | dict[str, Any]",
) -> tuple[float, bool, dict[str, Any]]:
    """Two-stage progressive dense reward for stacking 3 objects [base, mid, top]."""
    task = _get(scene, "task", {})
    objects = _get(scene, "objects", [])
    table_z = float(_get(scene, "table_z", 0.0))

    obj_map = {_get(o, "id", ""): o for o in objects}
    order = _get(task, "stack_order", None)
    if not order or len(order) < 3:
        order = [_get(o, "id", "") for o in objects[:3]]
    if len(order) < 3:
        return _pick_and_place_reward(model, data, scene)

    id_0, id_1, id_2 = order[0], order[1], order[2]
    pos_0 = _body_pos(model, data, id_0)
    pos_1 = _body_pos(model, data, id_1)
    pos_2 = _body_pos(model, data, id_2)

    speed_1 = _body_speed(model, data, id_1)
    speed_2 = _body_speed(model, data, id_2)

    hz_0 = _node_half_height(obj_map[id_0]) if id_0 in obj_map else 0.02
    hz_1 = _node_half_height(obj_map[id_1]) if id_1 in obj_map else 0.02
    hz_2 = _node_half_height(obj_map[id_2]) if id_2 in obj_map else 0.02

    grip = _gripper_pos(data)
    aperture = _finger_aperture(model, data)

    target_1_on_0 = np.array([pos_0[0], pos_0[1], pos_0[2] + hz_0 + hz_1], dtype=np.float64)
    d_xy_10 = float(np.linalg.norm(pos_1[:2] - pos_0[:2]))
    d_z_10 = abs(float(pos_1[2]) - float(target_1_on_0[2]))
    d_place_1 = float(np.linalg.norm(pos_1 - target_1_on_0))
    stage1_stacked = (d_xy_10 < 0.018) and (d_z_10 < 0.015) and (speed_1 < 0.12)

    target_2_on_1 = np.array([pos_1[0], pos_1[1], pos_1[2] + hz_1 + hz_2], dtype=np.float64)
    d_xy_21 = float(np.linalg.norm(pos_2[:2] - pos_1[:2]))
    d_z_21 = abs(float(pos_2[2]) - float(target_2_on_1[2]))
    d_place_2 = float(np.linalg.norm(pos_2 - target_2_on_1))
    stage2_stacked = (
        stage1_stacked
        and (d_xy_21 < 0.018)
        and (d_z_21 < 0.015)
        and (speed_2 < 0.10)
        and (aperture > 0.045)
    )

    if not stage1_stacked:
        # Stage 1: Reach, grasp, lift, and place obj_1 onto base obj_0 (per-step in [0, 1])
        d_reach = float(np.linalg.norm(grip - pos_1))
        reach_reward = _tanh_reward(d_reach, scale=4.0)
        grasp_bonus = 1.0 if (d_reach < 0.065 and aperture < 0.060) else 0.0
        lift_reward = min(max(float(pos_1[2]) - (table_z + hz_1 + 0.015), 0.0) * 20.0, 1.0)
        place_reward = _tanh_reward(d_place_1, scale=4.0)
        reward = 0.25 * reach_reward + 0.15 * grasp_bonus + 0.10 * lift_reward + 0.50 * place_reward
        d_goal = d_place_1
    else:
        # Stage 2: Stage 1 subgoal active (+1.0 base); now stack obj_2 onto obj_1 (per-step in [1, 2])
        d_reach = float(np.linalg.norm(grip - pos_2))
        reach_reward = _tanh_reward(d_reach, scale=4.0)
        grasp_bonus = 1.0 if (d_reach < 0.065 and aperture < 0.060) else 0.0
        lift_reward = min(max(float(pos_2[2]) - (table_z + hz_2 + 0.025), 0.0) * 15.0, 1.0)
        place_reward = _tanh_reward(d_place_2, scale=4.0)
        reward = 1.0 + 0.25 * reach_reward + 0.15 * grasp_bonus + 0.10 * lift_reward + 0.50 * place_reward
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
        "success": bool(terminated),
        "is_success": bool(terminated),
    }
    return reward, terminated, info