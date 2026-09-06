#!/usr/bin/env python3
"""Sync URDF <collision> blocks from Unitree ``g1_actuated.xml`` (official sim MJCF).

Visual meshes stay untouched. MJCF capsules become URDF cylinders (Isaac converts to
capsules via ``replace_cylinders_with_capsules=True``).
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent
MJCF_PATH = ASSET_DIR / "xmls" / "g1_actuated.xml"
URDF_PATH = ASSET_DIR / "urdf" / "g1_29dof_mode_15.urdf"


def _parse_floats(text: str) -> list[float]:
    return [float(v) for v in text.split()]


def quat_wxyz_to_rpy(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _is_collision_geom(geom: ET.Element) -> bool:
    cls = geom.attrib.get("class", "")
    name = geom.attrib.get("name", "")
    if "collision" in cls or "foot_capsule" in cls:
        return True
    if name.endswith("_collision"):
        return True
    return geom.attrib.get("type") in {"sphere", "capsule", "cylinder", "box"}


def _collect_mjcf_collisions(body: ET.Element, out: dict[str, list[dict]]) -> None:
    name = body.attrib["name"]
    for geom in body.findall("geom"):
        if not _is_collision_geom(geom):
            continue
        gtype = geom.attrib.get("type", "capsule")
        size = _parse_floats(geom.attrib.get("size", "0"))
        pos = _parse_floats(geom.attrib.get("pos", "0 0 0"))
        quat = _parse_floats(geom.attrib.get("quat", "1 0 0 0"))
        entry = {
            "name": geom.attrib.get("name", f"{name}_collision"),
            "type": gtype,
            "pos": pos,
            "rpy": quat_wxyz_to_rpy(*quat) if len(quat) == 4 else (0.0, 0.0, 0.0),
        }
        if gtype == "sphere":
            entry["radius"] = size[0]
        else:
            # MJCF capsule: size = radius half_length
            entry["radius"] = size[0]
            entry["length"] = 2.0 * size[1]
        out.setdefault(name, []).append(entry)
    for child in body.findall("body"):
        _collect_mjcf_collisions(child, out)


def load_mjcf_collisions(path: Path) -> dict[str, list[dict]]:
    root = ET.parse(path).getroot()
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError(f"no worldbody in {path}")
    out: dict[str, list[dict]] = {}
    for body in world.findall("body"):
        _collect_mjcf_collisions(body, out)
    return out


def _fmt(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def _make_collision(entry: dict) -> ET.Element:
    col = ET.Element("collision")
    if entry.get("name"):
        col.set("name", entry["name"])
    origin = ET.SubElement(col, "origin")
    origin.set("xyz", _fmt(entry["pos"]))
    origin.set("rpy", _fmt(entry["rpy"]))
    geometry = ET.SubElement(col, "geometry")
    if entry["type"] == "sphere":
        sphere = ET.SubElement(geometry, "sphere")
        sphere.set("radius", f"{entry['radius']:.9g}")
    else:
        cylinder = ET.SubElement(geometry, "cylinder")
        cylinder.set("radius", f"{entry['radius']:.9g}")
        cylinder.set("length", f"{entry['length']:.9g}")
    return col


def patch_urdf(urdf_path: Path, collisions: dict[str, list[dict]]) -> dict[str, int]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    stats = {"patched": 0, "cleared": 0, "mesh_removed": 0}
    for link in root.findall("link"):
        name = link.attrib["name"]
        for old in list(link.findall("collision")):
            geom = old.find("geometry/*")
            if geom is not None and geom.tag == "mesh":
                stats["mesh_removed"] += 1
            link.remove(old)
        entries = collisions.get(name)
        if not entries:
            stats["cleared"] += 1
            continue
        insert_at = len(link.findall("visual"))
        for idx, entry in enumerate(entries):
            link.insert(insert_at + idx, _make_collision(entry))
        stats["patched"] += 1
    tree.write(urdf_path, encoding="unicode", xml_declaration=False)
    return stats


def main() -> None:
    collisions = load_mjcf_collisions(MJCF_PATH)
    stats = patch_urdf(URDF_PATH, collisions)
    total_cols = sum(len(v) for v in collisions.values())
    print(f"Patched {URDF_PATH.name}: links={stats['patched']} geoms={total_cols} mesh_removed={stats['mesh_removed']}")


if __name__ == "__main__":
    main()
