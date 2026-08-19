#!/usr/bin/env python3
"""Benchmark the eight compiled native planners on the shared tiny fixture."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCE = ROOT / "benchmarks" / "planner_benchmark.mojo"
sys.path.insert(0, str(ROOT / "scripts"))

from run_experiment_matrix import METHODS, compile_native  # noqa: E402


def invoke(binary: Path, method: str) -> tuple[list[float], float]:
    start = time.perf_counter()
    completed = subprocess.run(
        [str(binary), method],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.perf_counter() - start
    lines = [line.split() for line in completed.stdout.splitlines() if line]
    if len(lines) != 1 or lines[0][:2] != ["POLICY", method]:
        raise ValueError(f"invalid native benchmark output for {method}")
    return [float(lines[0][2]), float(lines[0][3])], elapsed


def invoke_minigrid_sparse(binary: Path) -> tuple[list[float], float]:
    name = "minigrid-active-sparse"
    start = time.perf_counter()
    completed = subprocess.run(
        [str(binary), name],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.perf_counter() - start
    policy = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if fields[:2] != ["SPARSE_ACTION", name] or int(fields[2]) != len(
            policy
        ):
            raise ValueError("invalid native MiniGrid sparse output")
        policy.append(float(fields[3]))
    if len(policy) != 7 or not math.isclose(sum(policy), 1.0, abs_tol=1e-5):
        raise ValueError("invalid native MiniGrid sparse policy")
    return policy, elapsed


def quartiles(samples: list[float]) -> tuple[float, float]:
    ordered = sorted(samples)
    midpoint = len(ordered) // 2
    lower = ordered[:midpoint] or ordered
    upper = ordered[(len(ordered) + 1) // 2 :] or ordered
    return statistics.median(lower), statistics.median(upper)


def benchmark(binary: Path, runs: int, compile_seconds: float) -> dict[str, Any]:
    planners = []
    for method in METHODS:
        policy, first_seconds = invoke(binary, method)
        warm_seconds = [invoke(binary, method)[1] for _ in range(runs)]
        q1, q3 = quartiles(warm_seconds)
        planners.append(
            {
                "planning_method": method,
                "native_policy": policy,
                "first_execution_seconds": first_seconds,
                "warm_execution_seconds": warm_seconds,
                "warm_median_seconds": statistics.median(warm_seconds),
                "warm_q1_seconds": q1,
                "warm_q3_seconds": q3,
                "warm_iqr_seconds": q3 - q1,
            }
        )
    sparse_policy, sparse_first_seconds = invoke_minigrid_sparse(binary)
    sparse_runs = min(runs, 3)
    sparse_warm_seconds = [
        invoke_minigrid_sparse(binary)[1] for _ in range(sparse_runs)
    ]
    sparse_q1, sparse_q3 = quartiles(sparse_warm_seconds)
    return {
        "schema_version": 1,
        "fixture": {
            "name": "tiny-s2-k2-a2-h2-i2",
            "dtype": "Float32",
            "transition_path": "dense",
            "goal_form": "theta_preference",
        },
        "methodology": {
            "compile_separated": True,
            "jit": False,
            "warm_sample_scope": "compiled_native_process_wall",
            "includes_process_startup": True,
            "warm_runs_per_planner": runs,
        },
        "compile_seconds": compile_seconds,
        "planners": planners,
        "representative_samples": [
            {
                "name": "minigrid-3x3-active-precomputed-sparse",
                "environment": "minigrid",
                "planning_method": "active-inference",
                "transition_path": "precomputed_deterministic_sparse",
                "n_states": 108,
                "n_static": 9,
                "n_actions": 7,
                "planning_horizon": 1,
                "planning_iterations": 1,
                "native_policy": sparse_policy,
                "first_execution_seconds": sparse_first_seconds,
                "warm_execution_seconds": sparse_warm_seconds,
                "warm_median_seconds": statistics.median(
                    sparse_warm_seconds
                ),
                "warm_q1_seconds": sparse_q1,
                "warm_q3_seconds": sparse_q3,
                "warm_iqr_seconds": sparse_q3 - sparse_q1,
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--native-binary", type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    with tempfile.TemporaryDirectory(prefix="aif-mojo-benchmark-") as temporary:
        binary = args.native_binary or Path(temporary) / "planner_benchmark"
        compile_seconds = (
            0.0
            if args.native_binary
            else compile_native(binary, source=NATIVE_SOURCE)
        )
        result = benchmark(binary, args.runs, compile_seconds)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"PASS planner benchmark: {len(METHODS)} methods; output={args.output}")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
