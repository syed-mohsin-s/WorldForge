"""WorldForge MJCF Builder — Deterministic compiler from SceneGraph to MuJoCo 3.x XML.

Design notes
------------
* Coordinate frame: MuJoCo world-Z is up. The tabletop is a plane geom at
  ``z = scene.table_z``.
* Gripper control: a ``<body mocap="true">`` named ``ee_mocap`` is welded to
  ``gripper_base`` via an equality constraint. The weld's ``solref="0.01 1"``
  gives a stiff-but-damped spring so the gripper tracks the mocap target
  without exciting the contact solver.  ``solimp`` is left at defaults —
  MuJoCo 3.x's implicit-fast integrator handles it well.
* Contact: ``condim="4"`` on the floor geom enables tangential + torsional
  friction for realistic sliding/pivoting.  Object geoms use ``condim="3"``
  (tangential only) by default.
* Every object centroid and the task target position get a ``<site>`` marker
  for reward computation / visualization.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from src.schema.scene_graph import SceneGraph

# ---------------------------------------------------------------------------
# Material look-up
# ---------------------------------------------------------------------------
_MATERIALS_PATH = Path(__file__).resolve().parents[2] / "configs" / "materials.yaml"


def _load_materials() -> dict:
    with open(_MATERIALS_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_scene_to_mjcf(
    scene: "SceneGraph",
    randomize: bool = False,
    seed: int | None = None,
) -> str:
    """Compile a validated ``SceneGraph`` into a MuJoCo 3.x XML string.

    Parameters
    ----------
    scene:
        Validated scene graph (objects already snapped to table).
    randomize:
        If *True*, sample density and friction uniformly from the material
        ranges in ``configs/materials.yaml``.  Otherwise use range midpoints.
    seed:
        RNG seed used when *randomize* is True (for reproducibility).

    Returns
    -------
    str
        A complete, loadable MuJoCo XML document.
    """
    if seed is not None:
        random.seed(seed)

    mat_db = _load_materials()

    root = ET.Element("mujoco", model=f"worldforge_{scene.scene_id}")

    # ── compiler & option ──────────────────────────────────────────────
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        root,
        "option",
        timestep="0.002",
        integrator="implicitfast",
        cone="elliptic",
        impratio="10",
    )

    # ── visual defaults ───────────────────────────────────────────────
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3")

    # ── defaults ──────────────────────────────────────────────────────
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "geom",
        condim="3",
        solref="0.01 1",
        solimp="0.9 0.95 0.001 0.5 2",
    )
    ET.SubElement(default, "joint", damping="0.5")
    gripper_cls = ET.SubElement(default, "default", **{"class": "gripper"})
    ET.SubElement(
        gripper_cls,
        "geom",
        condim="4",
        friction="1.5 0.02 0.01",
        rgba="0.3 0.3 0.3 1",
    )

    # ── assets (materials) ────────────────────────────────────────────
    asset = ET.SubElement(root, "asset")
    used_materials: set[str] = set()
    for obj in scene.objects:
        mat_name = obj.material
        if mat_name in used_materials:
            continue
        used_materials.add(mat_name)
        info = mat_db.get(mat_name, mat_db["plastic"])
        rgba = " ".join(str(c) for c in info["rgba"])
        ET.SubElement(
            asset,
            "material",
            name=mat_name,
            rgba=rgba,
            specular="0.3",
            shininess="0.3",
        )
    # Table material
    ET.SubElement(
        asset,
        "material",
        name="table_mat",
        rgba="0.45 0.35 0.25 1",
        specular="0.1",
        shininess="0.1",
    )
    # Target marker material
    ET.SubElement(
        asset,
        "material",
        name="target_mat",
        rgba="0.0 1.0 0.0 0.4",
    )

    # ── worldbody ─────────────────────────────────────────────────────
    worldbody = ET.SubElement(root, "worldbody")

    # Light
    ET.SubElement(
        worldbody,
        "light",
        pos="0 0 2.0",
        dir="0 0 -1",
        diffuse="0.8 0.8 0.8",
        directional="true",
    )

    # Camera
    cam = scene.camera
    ET.SubElement(
        worldbody,
        "camera",
        name="render_cam",
        pos=_v3(cam.pos),
        xyaxes=_camera_xyaxes(cam.pos, cam.lookat),
        fovy=str(cam.fovy),
    )

    # Table plane
    ET.SubElement(
        worldbody,
        "geom",
        name="table",
        type="plane",
        pos=f"0 0 {scene.table_z}",
        size="1 1 0.02",
        material="table_mat",
        condim="4",
        friction="0.6 0.005 0.0001",
    )

    # ── scene objects ─────────────────────────────────────────────────
    _obj_map = {o.id: o for o in scene.objects}
    # Build children lookup
    children: dict[str | None, list] = {None: []}
    for o in scene.objects:
        children.setdefault(o.parent_id, [])
        children[None] = children.get(None, [])
    for o in scene.objects:
        children.setdefault(o.parent_id, []).append(o)
    # Deduplicate (we added to None twice in the setup)
    children[None] = [o for o in scene.objects if o.parent_id is None]

    def _add_body(parent_el: ET.Element, obj) -> None:  # noqa: ANN001
        info = mat_db.get(obj.material, mat_db["plastic"])

        # Density / friction
        if randomize:
            density = random.uniform(*info["density_range"])
            friction_t = random.uniform(*info["sliding_friction_range"])
        else:
            density = sum(info["density_range"]) / 2.0
            friction_t = sum(info["sliding_friction_range"]) / 2.0
        friction = f"{friction_t} 0.005 0.0001"

        body = ET.SubElement(
            parent_el,
            "body",
            name=obj.id,
            pos=_v3(obj.position),
            quat=_v4(obj.quaternion),
        )

        # Joint
        jt = obj.joint
        if obj.is_static or jt.type == "fixed":
            pass  # no joint → welded to world / parent
        elif jt.type == "free":
            ET.SubElement(body, "freejoint", name=f"{obj.id}_free")
        else:
            jattrs: dict[str, str] = {
                "name": f"{obj.id}_joint",
                "type": jt.type,
                "axis": _v3(jt.axis),
                "damping": str(jt.damping),
            }
            if jt.range is not None:
                jattrs["range"] = f"{jt.range[0]} {jt.range[1]}"
                jattrs["limited"] = "true"
            ET.SubElement(body, "joint", **jattrs)

        # Geom
        geom_attrs: dict[str, str] = {
            "name": f"{obj.id}_geom",
            "type": obj.primitive,
            "size": " ".join(str(s) for s in obj.size),
            "material": obj.material,
            "density": str(density),
            "friction": friction,
        }
        ET.SubElement(body, "geom", **geom_attrs)

        # Centroid site (at body origin — which IS the centroid for MuJoCo prims)
        ET.SubElement(
            body,
            "site",
            name=f"{obj.id}_site",
            pos="0 0 0",
            size="0.005",
            rgba="1 0 0 0.5",
        )

        # Recurse into children
        for child in children.get(obj.id, []):
            _add_body(body, child)

    for obj in children[None]:
        _add_body(worldbody, obj)

    # ── target site ───────────────────────────────────────────────────
    if scene.task.target_position is not None:
        ET.SubElement(
            worldbody,
            "site",
            name="target_site",
            pos=_v3(scene.task.target_position),
            size="0.015",
            rgba="0 1 0 0.6",
            type="sphere",
        )

    # ── mocap gripper ─────────────────────────────────────────────────
    _build_gripper(worldbody)

    # ── equality constraints ──────────────────────────────────────────
    equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "weld",
        body1="ee_mocap",
        body2="gripper_base",
        solref="0.01 1",
        solimp="0.9 0.95 0.001 0.5 2",
    )

    # ── actuators ─────────────────────────────────────────────────────
    actuator = ET.SubElement(root, "actuator")
    ET.SubElement(
        actuator,
        "position",
        name="finger_actuator",
        joint="finger_joint_L",
        ctrlrange="0 0.04",
        kp="250",
    )
    ET.SubElement(
        actuator,
        "position",
        name="finger_actuator_R",
        joint="finger_joint_R",
        ctrlrange="0 0.04",
        kp="250",
    )

    # ── serialize ─────────────────────────────────────────────────────
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


# ---------------------------------------------------------------------------
# Gripper builder
# ---------------------------------------------------------------------------

def _build_gripper(worldbody: ET.Element) -> None:
    """Add a two-finger parallel-jaw gripper driven by mocap weld.

    Layout (all sizes in metres):
    * ``ee_mocap``    — the mocap body the user drives via ``data.mocap_pos``
    * ``gripper_base``— the physical body welded to ``ee_mocap``
    * Two symmetric finger bodies with slide joints (±Y)
    """
    # Mocap body (kinematic, not simulated)
    mocap = ET.SubElement(
        worldbody,
        "body",
        name="ee_mocap",
        mocap="true",
        pos="0 0 0.25",
        quat="1 0 0 0",
    )
    ET.SubElement(
        mocap,
        "geom",
        name="ee_mocap_vis",
        type="box",
        size="0.01 0.01 0.01",
        rgba="0.9 0.1 0.1 0.3",
        contype="0",
        conaffinity="0",
    )

    # Physical gripper base
    gripper_base = ET.SubElement(
        worldbody,
        "body",
        name="gripper_base",
        pos="0 0 0.25",
        quat="1 0 0 0",
    )
    ET.SubElement(gripper_base, "freejoint", name="gripper_free")
    ET.SubElement(
        gripper_base,
        "geom",
        name="gripper_base_geom",
        type="box",
        size="0.03 0.02 0.015",
        rgba="0.3 0.3 0.3 1",
        mass="0.2",
        **{"class": "gripper"},
    )
    # Inertial for the base — keeps the solver happy
    ET.SubElement(
        gripper_base,
        "inertial",
        pos="0 0 0",
        mass="0.4",
        diaginertia="0.001 0.001 0.001",
    )

    # Left finger
    finger_l = ET.SubElement(
        gripper_base,
        "body",
        name="finger_L",
        pos="0 -0.02 -0.02",
    )
    ET.SubElement(
        finger_l,
        "joint",
        name="finger_joint_L",
        type="slide",
        axis="0 -1 0",
        damping="10",
        range="0 0.04",
        limited="true",
    )
    ET.SubElement(
        finger_l,
        "geom",
        name="finger_L_geom",
        type="box",
        size="0.01 0.005 0.03",
        rgba="0.4 0.4 0.4 1",
        mass="0.05",
        **{"class": "gripper"},
    )

    # Right finger (mirror)
    finger_r = ET.SubElement(
        gripper_base,
        "body",
        name="finger_R",
        pos="0 0.02 -0.02",
    )
    ET.SubElement(
        finger_r,
        "joint",
        name="finger_joint_R",
        type="slide",
        axis="0 1 0",
        damping="10",
        range="0 0.04",
        limited="true",
    )
    ET.SubElement(
        finger_r,
        "geom",
        name="finger_R_geom",
        type="box",
        size="0.01 0.005 0.03",
        rgba="0.4 0.4 0.4 1",
        mass="0.05",
        **{"class": "gripper"},
    )


# ---------------------------------------------------------------------------
# Math / formatting helpers
# ---------------------------------------------------------------------------

def _v3(v: tuple[float, float, float]) -> str:
    return f"{v[0]} {v[1]} {v[2]}"


def _v4(v: tuple[float, float, float, float]) -> str:
    return f"{v[0]} {v[1]} {v[2]} {v[3]}"


def _camera_xyaxes(
    pos: tuple[float, float, float],
    lookat: tuple[float, float, float],
) -> str:
    """Compute MuJoCo ``xyaxes`` from *pos* and *lookat*.

    The camera looks from *pos* towards *lookat*.  MuJoCo's ``xyaxes``
    attribute expects the camera X-axis (right) and Y-axis (up) as a
    6-element vector.  We derive them from the forward vector and world-up.
    """
    import math

    # Forward = lookat - pos (normalised)
    fx = lookat[0] - pos[0]
    fy = lookat[1] - pos[1]
    fz = lookat[2] - pos[2]
    norm = math.sqrt(fx * fx + fy * fy + fz * fz) or 1e-8
    fx, fy, fz = fx / norm, fy / norm, fz / norm

    # World up
    ux, uy, uz = 0.0, 0.0, 1.0

    # Right = forward × up  (normalised)
    rx = fy * uz - fz * uy
    ry = fz * ux - fx * uz
    rz = fx * uy - fy * ux
    norm = math.sqrt(rx * rx + ry * ry + rz * rz) or 1e-8
    rx, ry, rz = rx / norm, ry / norm, rz / norm

    # True up = right × forward
    tux = ry * fz - rz * fy
    tuy = rz * fx - rx * fz
    tuz = rx * fy - ry * fx

    return f"{rx} {ry} {rz} {tux} {tuy} {tuz}"

