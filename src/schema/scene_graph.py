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
    fovy: float = 45.0
    resolution: tuple[int, int] = (640, 480)


# ---------------------------------------------------------------------------
# Joint
# ---------------------------------------------------------------------------
class JointSpec(BaseModel):
    """Joint attached to an ObjectNode body.

    ``is_prior_guess`` flags joints whose type/axis was inferred by the VLM
    rather than measured from video tracks — downstream validators may
    apply extra checks or domain-randomize these.
    """

    type: Literal["free", "hinge", "slide", "fixed"] = "free"
    axis: tuple[float, float, float] = (0, 0, 1)
    range: tuple[float, float] | None = None
    damping: float = 0.5
    is_prior_guess: bool = False


# ---------------------------------------------------------------------------
# Object
# ---------------------------------------------------------------------------
class ObjectNode(BaseModel):
    """One rigid-body primitive on the tabletop."""

    id: str
    label: str
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
    is_static: bool = False
    joint: JointSpec = Field(default_factory=JointSpec)
    parent_id: str | None = None


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
class TaskSpec(BaseModel):
    """Describes the manipulation task the RL agent should solve."""

    task_type: Literal["pick_and_place", "push_to_target", "articulate_joint"]
    description: str
    manipulated_object_id: str
    target_position: tuple[float, float, float] | None = None
    target_joint_delta: float | None = None
    waypoints: list[tuple[float, float, float, float]] = Field(
        default_factory=list,
        description="Oracle keyframes as [x, y, z, gripper_cmd].",
    )


# ---------------------------------------------------------------------------
# Scene Graph (top-level)
# ---------------------------------------------------------------------------
class SceneGraph(BaseModel):
    """Complete scene description consumed by the MJCF compiler."""

    scene_id: str
    table_z: float = 0.0
    camera: CameraSpec
    objects: list[ObjectNode]
    task: TaskSpec

    # -- Auto-snap validator ------------------------------------------------
    @model_validator(mode="after")
    def _snap_objects_to_table(self) -> "SceneGraph":
        """Ensure every non-child, non-static resting object has its bottom
        surface at ``table_z + ε`` so MuJoCo's contact solver never starts
        with deep penetration (which causes step-1 force explosions).

        Half-height computation follows MuJoCo 3.x geom semantics:
        * ``box``      → size[2]              (half-size along local Z)
        * ``cylinder`` → size[1]              (half-length along local Z)
        * ``sphere``   → size[0]              (radius — symmetric)
        """
        eps = 0.001  # 1 mm clearance
        for obj in self.objects:
            if obj.parent_id is not None:
                continue  # child bodies are positioned relative to parent

            half_h = _half_height(obj)
            floor = self.table_z + half_h + eps
            x, y, z = obj.position
            if z < floor:
                obj.position = (x, y, floor)
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _half_height(obj: ObjectNode) -> float:
    """Return the half-height of *obj* along its local Z axis.

    For a rotated object this is an approximation — the full AABB would
    require quaternion math — but for Frame-0 snapping of axis-aligned
    primitives on a flat table it is exact and keeps the code simple.
    """
    prim = obj.primitive
    if prim == "box":
        return obj.size[2]
    elif prim == "cylinder":
        return obj.size[1]
    elif prim == "sphere":
        return obj.size[0]
    raise ValueError(f"Unknown primitive {prim!r}")

