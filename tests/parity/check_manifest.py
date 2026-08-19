#!/usr/bin/env python3
"""Validate the frozen JAX oracle and the cross-language numerical contract."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "tests" / "fixtures" / "manifest.json"
METHODS = (
    "loopy-vbp",
    "loopy",
    "region-extended",
    "dyn-channel",
    "nuijten",
    "vbp-channel",
    "precise-info-seeking",
    "active-inference",
)
EXPECTED_PLANNER_PATHS = {
    "loopy-vbp": ["dense"],
    "loopy": ["dense", "deterministic_sparse"],
    "region-extended": ["dense"],
    "dyn-channel": ["dense", "deterministic_sparse"],
    "nuijten": ["dense", "deterministic_sparse"],
    "vbp-channel": ["dense", "deterministic_sparse"],
    "precise-info-seeking": ["dense"],
    "active-inference": ["dense", "precomputed_deterministic_sparse"],
}
EXPECTED_TOLERANCES = {
    "float32_probability_atol": 1e-5,
    "float32_probability_rtol": 1e-5,
    "probability_sum_environment_atol": 1e-6,
    "probability_sum_inference_atol": 1e-5,
    "region_channel_atol": 1e-4,
    "minigrid_float16_atol_min": 1e-3,
    "minigrid_float16_atol_max": 1e-2,
    "masked_action_probability_max": 1e-6,
    "argmax_margin_min": 1e-4,
}
ENVIRONMENTS = (
    "frozen_lake",
    "wumpus_world",
    "rocksample",
    "minigrid",
)
EXPECTED_AXES = {
    "transition_dense": ["x_new", "x_old", "theta", "action"],
    "transition_sparse": ["x_old", "action", "theta"],
    "observation": ["channel", "outcome", "state", "theta"],
    "state_belief": ["state"],
    "static_belief": ["theta"],
    "terminal_goal": ["state"],
    "theta_goal": ["state", "theta"],
    "action_distribution": ["time", "action"],
    "minigrid_observation": [
        "fov_x",
        "fov_y",
        "cell_type",
        "state",
        "theta",
    ],
    "minigrid_orientation_observation": ["orientation", "state"],
}


def _run(command: list[str], cwd: Path) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _oracle_runtime(python: Path, cwd: Path) -> dict[str, Any]:
    source = (
        "import json,platform,jax,jaxlib,numpy;"
        "print(json.dumps({'python':platform.python_version(),"
        "'jax':jax.__version__,'jaxlib':jaxlib.__version__,"
        "'numpy':numpy.__version__,"
        "'jax_backend':jax.default_backend(),"
        "'jax_enable_x64':bool(jax.config.jax_enable_x64)}))"
    )
    return json.loads(_run([str(python), "-c", source], cwd))


def check_manifest(path: Path = DEFAULT_MANIFEST) -> list[str]:
    manifest = json.loads(path.read_text())
    _require(manifest["schema_version"] == 1, "unsupported schema version")

    oracle = manifest["oracle"]
    oracle_dir = ROOT / oracle["relative_path"]
    _require(oracle_dir.is_dir(), f"oracle checkout missing: {oracle_dir}")
    actual_sha = _run(["git", "rev-parse", "HEAD"], oracle_dir)
    _require(actual_sha == oracle["commit"], "oracle SHA does not match manifest")
    if oracle["required_clean"]:
        status = _run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            oracle_dir,
        )
        _require(not status, f"oracle checkout is dirty:\n{status}")

    runtime = manifest["runtime"]
    oracle_python = oracle_dir / ".venv" / "bin" / "python"
    _require(oracle_python.is_file(), "oracle .venv Python is missing")
    actual_runtime = _oracle_runtime(oracle_python, oracle_dir)
    for key in (
        "python",
        "jax",
        "jaxlib",
        "numpy",
        "jax_backend",
        "jax_enable_x64",
    ):
        _require(
            actual_runtime[key] == runtime[key],
            f"runtime mismatch for {key}: {actual_runtime[key]!r}",
        )
    mojo_output = _run(["pixi", "run", "mojo", "--version"], ROOT)
    mojo_match = re.search(r"^Mojo\s+(\S+)", mojo_output)
    _require(bool(mojo_match), f"cannot parse Mojo version: {mojo_output!r}")
    _require(mojo_match.group(1) == runtime["mojo"], "Mojo version mismatch")
    _require(runtime["dtype"] == "Float32", "production dtype must be Float32")
    _require(runtime["jax_backend"] == "cpu", "JAX backend must remain CPU")
    _require(runtime["jax_enable_x64"] is False, "JAX x64 must remain disabled")

    _require(manifest["axes"] == EXPECTED_AXES, "canonical tensor axes changed")
    _require(
        set(manifest["goal_forms"]) == {"terminal", "theta_preference"},
        "goal form set changed",
    )
    _require(
        manifest["goal_forms"]["terminal"]["axes"] == ["state"],
        "terminal goal axes changed",
    )
    _require(
        manifest["goal_forms"]["theta_preference"]["axes"]
        == ["state", "theta"],
        "theta preference axes changed",
    )
    _require(
        tuple(manifest["planner_paths"]) == METHODS,
        "retained planner names or order changed",
    )
    _require(
        manifest["planner_paths"] == EXPECTED_PLANNER_PATHS,
        "planner path mapping changed",
    )
    fixture = manifest["planner_smoke_fixture"]
    _require(fixture["dtype"] == "Float32", "smoke fixture dtype changed")
    _require(
        fixture["goal_form"] == "theta_preference"
        and fixture["transition_path"] == "dense",
        "smoke fixture execution contract changed",
    )
    _require(
        len(fixture["action_prior"]) == fixture["n_actions"]
        and len(fixture["action_mask"]) == fixture["n_actions"],
        "smoke fixture action vectors changed",
    )
    environment_fixtures = manifest["experiment_smoke_fixtures"]
    _require(
        tuple(environment_fixtures) == ENVIRONMENTS,
        "experiment smoke fixture set changed",
    )
    expected_actions = {
        "frozen_lake": 5,
        "wumpus_world": 5,
        "rocksample": 6,
        "minigrid": 7,
    }
    for environment, action_count in expected_actions.items():
        environment_fixture = environment_fixtures[environment]
        _require(
            environment_fixture["n_actions"] == action_count,
            f"action cardinality changed for {environment}",
        )
        _require(
            environment_fixture["generator"].startswith("generate_"),
            f"native generator missing for {environment}",
        )
    _require(
        tuple(manifest["deterministic_tapes"]) == ENVIRONMENTS,
        "deterministic tape set changed",
    )
    for environment, tape in manifest["deterministic_tapes"].items():
        _require(
            len(tape["actions"]) == len(tape["states"]) > 0,
            f"invalid deterministic tape for {environment}",
        )
        if "realized_movement_actions" in tape:
            _require(
                len(tape["realized_movement_actions"]) == len(tape["actions"]),
                f"invalid realized-action tape for {environment}",
            )
        draws = tape.get("observation_draws")
        if isinstance(draws, dict):
            _require(
                draws.get("channels", 0) > 0
                and 0.0 <= draws.get("default", -1.0) < 1.0,
                f"invalid observation draws for {environment}",
            )
        elif draws is not None:
            _require(
                draws and all(0.0 <= value < 1.0 for value in draws),
                f"invalid observation draws for {environment}",
            )
    _require(
        manifest["tolerances"] == EXPECTED_TOLERANCES,
        "numerical tolerances changed",
    )

    return [
        f"oracle={actual_sha}",
        f"python={runtime['python']} jax={runtime['jax']} numpy={runtime['numpy']}",
        f"mojo={runtime['mojo']} dtype={runtime['dtype']} backend=cpu x64=false",
        f"methods={len(METHODS)} tapes={len(ENVIRONMENTS)}",
    ]


def main() -> int:
    try:
        checks = check_manifest()
    except (AssertionError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(f"FAIL manifest: {exc}", file=sys.stderr)
        return 1
    print("PASS manifest: " + "; ".join(checks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
