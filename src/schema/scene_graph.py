"""WorldForge Scene Graph — Pydantic V2 strict models.

Defines the canonical intermediate representation between vision/VLM
perception and the deterministic MJCF compiler.  Every SceneGraph
instance is serializable to JSON and round-trips losslessly.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
class CameraSpec(BaseModel):
    """Pinhole camera parameters for MuJoCo rendering."""

    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fovy: float = Field(default=45.0, gt=1.0, lt=179.0)
    resolution: tuple[int, int] = (640, 480)


# ---------------------------------------------------------------------------
# Joint
# ---------------------------------------------------------------------------
class JointSpec(BaseModel):
    """Joint attached to an ObjectNode body."""

    type: Literal["free", "hinge", "slide", "fixed"] = "free"
    axis: tuple[float, float, float] = (0, 0, 1)
    range: tuple[float, float] | None = None
    damping: float = Field(default=0.5, ge=0.0)
    is_prior_guess: bool = False

    @model_validator(mode="after")
    def _validate_joint(self) -> "JointSpec":
        ax_norm = math.sqrt(sum(float(a) * float(a) for a in self.axis))
        if not math.isfinite(ax_norm) or ax_norm < 1e-6:
            raise ValueError(f"Joint axis {self.axis} must have non-zero norm")
        if self.range is not None:
            r0, r1 = float(self.range[0]), float(self.range[1])
            if not (math.isfinite(r0) and math.isfinite(r1)) or r0 > r1:
                raise ValueError(f"Invalid joint range {self.range}: min must be <= max")
        return self


# ---------------------------------------------------------------------------
# Object
# ---------------------------------------------------------------------------
class ObjectNode(BaseModel):
    """One rigid-body primitive on the tabletop."""

    id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    primitive: Literal["box", "cylinder", "sphere"]
    size: list[float] = Field(
        ...,
        description=(
            "MuJoCo half-sizes.  box→[hx, hy, hz], "
            "cylinder→[radius, half_length], sphere→[radius]."
        ),
    )
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float] = (1, 0, 0, 0)
    material: str = "plastic"
    rgba: tuple[float, float, float, float] | None = None
    is_static: bool = False
    joint: JointSpec = Field(default_factory=JointSpec)
    parent_id: str | None = None

    @model_validator(mode="after")
    def _validate_geometry_and_pose(self) -> "ObjectNode":
        expected_len = {"box": 3, "cylinder": 2, "sphere": 1}[self.primitive]
        if len(self.size) != expected_len:
            raise ValueError(
                f"Primitive '{self.primitive}' on object '{self.id}' requires "
                f"exactly {expected_len} size element(s), got {len(self.size)}: {self.size}"
            )
        for idx, s in enumerate(self.size):
            if not math.isfinite(float(s)) or float(s) <= 0.0:
                raise ValueError(
                    f"Object '{self.id}' size[{idx}]={s} must be finite and strictly positive (> 0)"
                )

        if any(not math.isfinite(float(p)) for p in self.position):
            raise ValueError(f"Object '{self.id}' position {self.position} contains non-finite values")

        q_norm = math.sqrt(sum(float(q) * float(q) for q in self.quaternion))
        if not math.isfinite(q_norm) or q_norm < 1e-6:
            raise ValueError(
                f"Object '{self.id}' quaternion {self.quaternion} has zero or non-finite norm"
            )
        return self


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
class TaskSpec(BaseModel):
    """Describes the manipulation task the RL agent should solve."""

    task_type: Literal["pick_and_place", "push_to_target", "articulate_joint", "multi_stack"]
    description: str
    manipulated_object_id: str = Field(..., min_length=1)
    target_position: tuple[float, float, float] | None = None
    target_joint_delta: float | None = None
    stack_order: list[str] | None = Field(
        default=None,
        description="Ordered object_ids from bottom base [0] to middle [1] to top [2] for multi_stack.",
    )
    user_prompt: str | None = Field(
        default=None,
        description="Optional natural-language user intent prompt.",
    )
    waypoints: list[tuple[float, float, float, float]] = Field(
        default_factory=list,
        description="Oracle keyframes as [x, y, z, gripper_cmd].",
    )


# ---------------------------------------------------------------------------
# Scene Graph (top-level)
# ---------------------------------------------------------------------------
class SceneGraph(BaseModel):
    """Complete scene description consumed by the MJCF compiler."""

    scene_id: str = Field(..., min_length=1)
    table_z: float = 0.0
    camera: CameraSpec
    objects: list[ObjectNode]
    task: TaskSpec

    @model_validator(mode="after")
    def _validate_and_snap_objects(self) -> "SceneGraph":
        """Validate object IDs, parent hierarchy acyclicity, manipulated object
        existence, stack_order references, and snap resting objects to ``table_z + ε``."""
        obj_map: dict[str, ObjectNode] = {}
        for obj in self.objects:
            if obj.id in obj_map:
                raise ValueError(f"Duplicate object id '{obj.id}' in scene '{self.scene_id}'")
            obj_map[obj.id] = obj

        for obj in self.objects:
            if obj.parent_id is not None:
                if obj.parent_id not in obj_map:
                    raise ValueError(
                        f"Object '{obj.id}' references nonexistent parent_id '{obj.parent_id}'"
                    )
                if obj.parent_id == obj.id:
                    raise ValueError(f"Object '{obj.id}' cannot be its own parent")

                visited = {obj.id}
                curr = obj.parent_id
                while curr is not None:
                    if curr in visited:
                        raise ValueError(
                            f"Cyclic parent_id hierarchy detected involving object '{obj.id}'"
                        )
                    visited.add(curr)
                    curr = obj_map[curr].parent_id

        if self.objects and self.task.manipulated_object_id not in obj_map:
            raise ValueError(
                f"task.manipulated_object_id '{self.task.manipulated_object_id}' "
                f"not found among scene objects {list(obj_map.keys())}"
            )

        if self.objects and self.task.stack_order:
            for sid in self.task.stack_order:
                if sid not in obj_map:
                    raise ValueError(
                        f"task.stack_order id '{sid}' not found among scene objects {list(obj_map.keys())}"
                    )

        eps = 0.001  # 1 mm clearance
        for obj in self.objects:
            if obj.parent_id is not None:
                continue

            half_h = _half_height(obj)
            floor = self.table_z + half_h + eps
            x, y, z = obj.position
            if z < floor:
                obj.position = (x, y, floor)
        return self


def _half_height(obj: ObjectNode) -> float:
    """Return the half-height of *obj* along its local Z axis."""
    prim = obj.primitive
    if prim == "box":
        return obj.size[2]
    elif prim == "cylinder":
        return obj.size[1]
    elif prim == "sphere":
        return obj.size[0]
    raise ValueError(f"Unknown primitive {prim!r}")
