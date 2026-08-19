#!/usr/bin/env python3
"""Run the bounded native-planner smoke matrix and write reference-shaped JSON."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCE = ROOT / "scripts" / "native_experiment_matrix.mojo"
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
ENVIRONMENTS = (
    "frozen_lake",
    "wumpus_world",
    "rocksample",
    "minigrid",
)
ENVIRONMENT_ACTIONS = {
    "frozen_lake": 5,
    "wumpus_world": 5,
    "rocksample": 6,
    "minigrid": 7,
}

COMMON_RESULTS_SCHEMA = {
    "success_rate": float,
    "avg_steps": float,
    "avg_reward": float,
    "successes": int,
    "total_time_s": float,
}
RESULT_SCHEMAS = {
    "frozen_lake": {
        "config": {
            "environment": str,
            "grid_size": int,
            "n_configs": int,
            "hole_fraction": float,
            "base_noise": float,
            "noise_range": float,
            "slip_prob": float,
            "hole_penalty": float,
            "goal_temperature": float,
            "scan_cost": float,
            "planning_method": str,
            "n_episodes": int,
            "max_steps": int,
            "planning_horizon": int,
            "planning_iterations": int,
            "receding_horizon": bool,
            "seed_start": int,
        },
        "results": COMMON_RESULTS_SCHEMA,
    },
    "wumpus_world": {
        "config": {
            "environment": str,
            "grid_size": int,
            "n_configs": int,
            "n_pits": int,
            "obs_noise": float,
            "pos_noise": float,
            "slip_prob": float,
            "sense_cost": float,
            "planning_method": str,
            "n_episodes": int,
            "max_steps": int,
            "planning_horizon": int,
            "planning_iterations": int,
            "receding_horizon": bool,
            "seed_start": int,
        },
        "results": COMMON_RESULTS_SCHEMA,
    },
    "rocksample": {
        "config": {
            "environment": str,
            "grid_size": int,
            "n_rocks": int,
            "n_configs": int,
            "half_eff_dist": float,
            "pos_noise": float,
            "slip_prob": float,
            "good_reward": float,
            "bad_penalty": float,
            "exit_reward": float,
            "good_logit": float,
            "bad_logit": float,
            "exit_logit": float,
            "goal_temperature": float,
            "sense_cost": float,
            "sample_cost": float,
            "planning_method": str,
            "n_episodes": int,
            "max_steps": int,
            "planning_horizon": int,
            "planning_iterations": int,
            "receding_horizon": bool,
            "seed_start": int,
        },
        "results": {
            "success_rate": float,
            "avg_steps": float,
            "avg_reward": float,
            "std_reward": float,
            "good_rock_retrieval": float,
            "total_good_rocks": int,
            "total_good_collected": int,
            "episodes_with_good_rocks": int,
            "successes": int,
            "total_time_s": float,
        },
    },
    "minigrid": {
        "config": {
            "grid_size": int,
            "env_name": str,
            "planning_method": str,
            "fov_size": int,
            "n_episodes": int,
            "max_steps": int,
            "planning_horizon": int,
            "receding_horizon": bool,
            "inference_iterations": int,
            "planning_iterations": int,
            "no_orientation": bool,
            "damping": float,
            "obs_alpha": float,
            "seed_start": int,
        },
        "results": COMMON_RESULTS_SCHEMA,
    },
}


def _config(environment: str, method: str) -> dict[str, Any]:
    common = {
        "planning_method": method,
        "n_episodes": 0,
        "max_steps": 0,
        "planning_horizon": 1,
        "planning_iterations": 1,
        "receding_horizon": False,
        "seed_start": 0,
    }
    if environment == "frozen_lake":
        return {
            "environment": environment,
            "grid_size": 2,
            "n_configs": 2,
            "hole_fraction": 0.25,
            "base_noise": 0.05,
            "noise_range": 0.15,
            "slip_prob": 0.0,
            "hole_penalty": 1.0,
            "goal_temperature": 1.0,
            "scan_cost": 0.5,
            **common,
        }
    if environment == "wumpus_world":
        return {
            "environment": environment,
            "grid_size": 2,
            "n_configs": 2,
            "n_pits": 1,
            "obs_noise": 0.1,
            "pos_noise": 0.1,
            "slip_prob": 0.0,
            "sense_cost": 0.5,
            **common,
        }
    if environment == "rocksample":
        return {
            "environment": environment,
            "grid_size": 2,
            "n_rocks": 1,
            "n_configs": 2,
            "half_eff_dist": 2.0,
            "pos_noise": 0.1,
            "slip_prob": 0.0,
            "good_reward": 10.0,
            "bad_penalty": 10.0,
            "exit_reward": 10.0,
            "good_logit": 2.0,
            "bad_logit": 4.0,
            "exit_logit": 2.0,
            "goal_temperature": 1.0,
            "sense_cost": 0.5,
            "sample_cost": 0.5,
            **common,
        }
    if environment == "minigrid":
        return {
            "grid_size": 3,
            "env_name": "MiniGrid-DoorKey-5x5-v0",
            "planning_method": method,
            "fov_size": 3,
            "n_episodes": 0,
            "max_steps": 0,
            "planning_horizon": 1,
            "receding_horizon": False,
            "inference_iterations": 0,
            "planning_iterations": 1,
            "no_orientation": False,
            "damping": 0.5,
            "obs_alpha": 0.0,
            "seed_start": 0,
        }
    raise ValueError(f"unknown environment: {environment}")


def _results(environment: str, elapsed: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "success_rate": 0.0,
        "avg_steps": 0.0,
        "avg_reward": 0.0,
    }
    if environment == "rocksample":
        result.update(
            {
                "std_reward": 0.0,
                "good_rock_retrieval": 0.0,
                "total_good_rocks": 0,
                "total_good_collected": 0,
                "episodes_with_good_rocks": 0,
            }
        )
    result.update({"successes": 0, "total_time_s": float(elapsed)})
    return result


def make_result_document(
    environment: str, method: str, elapsed: float = 0.0
) -> dict[str, Any]:
    document = {
        "config": _config(environment, method),
        "results": _results(environment, elapsed),
    }
    validate_result_document(environment, document)
    return document


def _exact_type(value: Any, expected: type[Any]) -> bool:
    return type(value) is expected


def validate_result_document(environment: str, document: dict[str, Any]) -> None:
    if environment not in RESULT_SCHEMAS:
        raise ValueError(f"unknown environment: {environment}")
    if set(document) != {"config", "results"}:
        raise ValueError("result root must contain exactly config and results")
    schema = RESULT_SCHEMAS[environment]
    for section in ("config", "results"):
        values = document[section]
        expected = schema[section]
        if set(values) != set(expected):
            raise ValueError(f"{environment} {section} fields changed")
        for key, expected_type in expected.items():
            if not _exact_type(values[key], expected_type):
                raise TypeError(
                    f"{environment}.{section}.{key} must be "
                    f"{expected_type.__name__}"
                )
    config = document["config"]
    results = document["results"]
    if config["planning_method"] not in METHODS:
        raise ValueError("unknown planning method")
    if environment != "minigrid" and config["environment"] != environment:
        raise ValueError("environment discriminator changed")
    if environment == "minigrid" and not config["env_name"].startswith(
        "MiniGrid-DoorKey-"
    ):
        raise ValueError("MiniGrid environment name changed")
    if not 0.0 <= results["success_rate"] <= 1.0:
        raise ValueError("success_rate must be in [0, 1]")
    nonnegative = {
        "avg_steps",
        "std_reward",
        "good_rock_retrieval",
        "total_good_rocks",
        "total_good_collected",
        "episodes_with_good_rocks",
        "successes",
        "total_time_s",
    }
    for key in nonnegative & results.keys():
        if results[key] < 0:
            raise ValueError(f"negative result field: {key}")


def parse_native_policies(output: str) -> dict[str, list[float]]:
    policies: dict[str, list[float]] = {}
    for line in output.splitlines():
        fields = line.split()
        if not fields or fields[0] not in {"POLICY", "ACTION"}:
            continue
        if len(fields) != 4:
            raise ValueError(f"invalid native policy line: {line!r}")
        method = fields[1]
        if fields[0] == "POLICY":
            if method in policies:
                raise ValueError(f"duplicate native policy: {method}")
            policies[method] = [float(fields[2]), float(fields[3])]
        else:
            action_idx, probability = int(fields[2]), float(fields[3])
            policy = policies.setdefault(method, [])
            if action_idx != len(policy):
                raise ValueError(f"non-contiguous native actions: {method}")
            policy.append(probability)
    if tuple(policies) != METHODS:
        raise ValueError(f"native methods changed: {tuple(policies)!r}")
    for method, policy in policies.items():
        if not all(math.isfinite(value) and value >= 0.0 for value in policy):
            raise ValueError(f"invalid native policy values: {method}")
        if not math.isclose(sum(policy), 1.0, abs_tol=1e-5):
            raise ValueError(f"unnormalized native policy: {method}")
    return policies


def compile_native(binary: Path, source: Path = NATIVE_SOURCE) -> float:
    start = time.perf_counter()
    subprocess.run(
        [
            "pixi",
            "run",
            "mojo",
            "build",
            "-I",
            str(ROOT / "src"),
            str(source),
            "-o",
            str(binary),
        ],
        cwd=ROOT,
        check=True,
    )
    return time.perf_counter() - start


def invoke_native(
    binary: Path, environment: str
) -> tuple[dict[str, list[float]], float]:
    start = time.perf_counter()
    completed = subprocess.run(
        [str(binary), environment],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.perf_counter() - start
    policies = parse_native_policies(completed.stdout)
    if any(
        len(policy) != ENVIRONMENT_ACTIONS[environment]
        for policy in policies.values()
    ):
        raise ValueError(f"native action count changed for {environment}")
    return policies, elapsed


def write_matrix(
    output_dir: Path,
    native_runs: dict[str, tuple[dict[str, list[float]], float]],
    compile_seconds: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        policies, elapsed = native_runs[environment]
        environment_dir = output_dir / environment
        environment_dir.mkdir(exist_ok=True)
        for method in METHODS:
            relative = Path(environment) / f"{method}.json"
            document = make_result_document(
                environment, method, elapsed / len(METHODS)
            )
            (output_dir / relative).write_text(json.dumps(document, indent=2) + "\n")
            policy = policies[method]
            cases.append(
                {
                    "environment": environment,
                    "planning_method": method,
                    "native_policy": policy,
                    "selected_action": max(
                        range(len(policy)), key=policy.__getitem__
                    ),
                    "native_execution_share_s": elapsed / len(METHODS),
                    "result_file": str(relative),
                }
            )
    summary = {
        "schema_version": 1,
        "kind": "native_policy_smoke",
        "fixtures": {
            "frozen_lake": {
                "grid_size": 2,
                "n_states": 8,
                "n_static": 2,
                "n_actions": 5,
                "n_observation_channels": 12,
                "n_observation_types": 2,
                "goal_form": "theta_preference",
            },
            "wumpus_world": {
                "grid_size": 2,
                "n_states": 8,
                "n_static": 2,
                "n_actions": 5,
                "n_observation_channels": 7,
                "n_observation_types": 2,
                "goal_form": "theta_preference",
            },
            "rocksample": {
                "grid_size": 2,
                "n_states": 24,
                "n_static": 2,
                "n_actions": 6,
                "n_observation_channels": 1,
                "n_observation_types": 3,
                "goal_form": "theta_preference",
            },
            "minigrid": {
                "grid_size": 3,
                "n_states": 108,
                "n_static": 9,
                "n_actions": 7,
                "n_observation_channels": 9,
                "n_observation_types": 11,
                "goal_form": "terminal",
            },
        },
        "dtype": "Float32",
        "transition_path": "dense",
        "planning_horizon": 1,
        "planning_iterations": 1,
        "compile_seconds": compile_seconds,
        "case_count": len(cases),
        "cases": cases,
    }
    summary_path = output_dir / "_smoke_matrix.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path


def run_matrix(
    output_dir: Path, native_binary: Path | None = None
) -> tuple[Path, float]:
    with tempfile.TemporaryDirectory(prefix="aif-mojo-matrix-") as temporary:
        binary = native_binary or Path(temporary) / "planner_benchmark"
        compile_seconds = 0.0 if native_binary else compile_native(binary)
        native_runs = {
            environment: invoke_native(binary, environment)
            for environment in ENVIRONMENTS
        }
        return write_matrix(output_dir, native_runs, compile_seconds), compile_seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data" / "smoke_matrix"
    )
    parser.add_argument(
        "--native-binary",
        type=Path,
        help="Reuse a previously built planner_benchmark executable",
    )
    args = parser.parse_args()
    summary, compile_seconds = run_matrix(args.output_dir, args.native_binary)
    print(
        f"PASS native smoke matrix: {len(METHODS) * len(ENVIRONMENTS)} cases; "
        f"compile={compile_seconds:.3f}s; summary={summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
