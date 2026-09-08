"""Inventory project Python sources and reject robot dependencies in shared algorithms."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path


def imports(tree, path):
    package = path.parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = package[: len(package) - node.level + 1] if node.level else ()
            module = (*prefix, *(node.module or "").split("."))
            module = ".".join(part for part in module if part)
            yield module
            yield from (f"{module}.{alias.name}" for alias in node.names if alias.name != "*")


def shared_algorithm(path):
    return path.startswith(("legged_lab/locomotion/", "legged_lab/motion_tracking/", "rsl_rl/rsl_rl/")) or path in (
        "legged_lab/config.py", "legged_lab/utils/rsl_rl_compat.py",
    )


def classify(path):
    if shared_algorithm(path):
        return "shared_algorithm"
    if path.startswith("legged_lab/assets/"):
        return "robot_asset_or_deployment_contract"
    if path.startswith("legged_lab/envs/"):
        return "robot_recipe_compatibility_or_legacy_task"
    if path.startswith("legged_lab/scripts/"):
        return "task_entrypoint"
    if path.startswith("tests/"):
        return "test"
    if path.startswith("artifacts/"):
        return "diagnostic_artifact"
    if path.startswith("legged_lab/"):
        return "framework_sensor_terrain_or_mdp"
    return "development_tool"


def audit(root):
    robots = {
        path.name for path in (root / "legged_lab/envs").iterdir()
        if path.is_dir() and path.name not in ("base", "__pycache__")
    }
    files, violations = [], []
    excluded = {".git", "__pycache__", ".pytest_cache", ".venv"}
    for directory, dirs, names in os.walk(root):
        dirs[:] = sorted(
            name for name in dirs if name not in excluded
            and not (Path(directory) == root / "artifacts" and name.startswith("pytest-"))
        )
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix != ".py":
                continue
            relative = path.relative_to(root)
            key = relative.as_posix()
            source = path.read_text(encoding="utf-8-sig")
            try:
                tree = ast.parse(source)
            except SyntaxError as error:
                violations.append({"path": key, "error": f"cannot parse source: {error}"})
                continue
            dependencies = sorted(set(imports(tree, relative)))
            robot_imports = [
                item for item in dependencies
                if item.startswith("legged_lab.assets.")
                or any(item == f"legged_lab.envs.{robot}" or item.startswith(f"legged_lab.envs.{robot}.")
                       for robot in robots)
            ]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            named_classes = [name for name in classes if name.lower().startswith(tuple(robots))]
            files.append({
                "path": key, "role": classify(key),
                "sha256": hashlib.sha256(source.replace("\r\n", "\n").encode()).hexdigest(),
                "robot_imports": robot_imports, "robot_named_classes": named_classes,
            })
            if shared_algorithm(key):
                forbidden = [item for item in dependencies if item.startswith(("legged_lab.assets", "legged_lab.envs"))]
                if forbidden or named_classes:
                    violations.append({"path": key, "error": "shared algorithm depends on robot or task registry",
                                       "imports": forbidden, "classes": named_classes})
            if len(relative.parts) >= 4 and relative.parts[:2] == ("legged_lab", "envs"):
                owner = relative.parts[2]
                if owner in robots:
                    foreign = [item for item in robot_imports if any(
                        item == f"legged_lab.envs.{robot}" or item.startswith(f"legged_lab.envs.{robot}.")
                        for robot in robots - {owner}
                    )]
                    if "legged_lab.envs" in dependencies:
                        foreign.append("legged_lab.envs (registry-level import inside a robot recipe)")
                    if foreign:
                        violations.append({"path": key, "error": "robot recipe imports another robot", "imports": foreign})
    return {"schema_version": "robot_boundary_audit.v1", "robot_packages": sorted(robots),
            "excluded_directory_names": sorted(excluded), "excluded_globs": ["artifacts/pytest-*/"],
            "files": sorted(files, key=lambda item: item["path"]),
            "violations": violations}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[1])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scanned_files": len(result["files"]), "violations": result["violations"]}, indent=2))
    raise SystemExit(bool(result["violations"]))
