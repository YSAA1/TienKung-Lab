"""Offline provenance guard; runtime asset values are checked by the Isaac probe."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_official_configuration_matches_pinned_upstream_ast():
    source = (ROOT / "legged_lab/assets/unitree_g1/official_g1.py").read_text()
    assignment = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign))
    evidence = json.loads((ROOT / "artifacts/portability/v4/official_source.json").read_text())
    digest = hashlib.sha256(json.dumps(_canonical(assignment), sort_keys=True).encode()).hexdigest()
    assert digest == evidence["config_canonical_sha256"]
    assert evidence["commit"] == "b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8"


def _canonical(value):
    if isinstance(value, ast.AST):
        return {
            "type": type(value).__name__,
            **{key: _canonical(item) for key, item in ast.iter_fields(value) if item is not None and item != []},
        }
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value
