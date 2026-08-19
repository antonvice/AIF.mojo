#!/usr/bin/env python3
"""In-process JAX eager or warm-JIT Loopy-BP benchmark worker."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "UAI-MP-AIF-JAX"
sys.path.insert(0, str(ORACLE))

from inference.loopy_bp import loopy_bp_planning  # noqa: E402


FIXTURES = {"small": 8, "large": 64}
N_STATIC = 2
N_ACTIONS = 4
HORIZON = 3
PLANNING_ITERATIONS = 3
WARMUP_ITERATIONS = 3


def fixture(n_states: int) -> tuple[jax.Array, ...]:
    q_state = np.zeros(n_states, dtype=np.float32)
    q_state[0] = 1.0
    q_static = np.array([0.6, 0.4], dtype=np.float32)
    transition = np.zeros(
        (n_states, n_states, N_STATIC, N_ACTIONS), dtype=np.float32
    )
    for old_state in range(n_states):
        for theta in range(N_STATIC):
            for action in range(N_ACTIONS):
                transition[
                    (old_state + theta + action) % n_states,
                    old_state,
                    theta,
                    action,
                ] = 1.0
    goal = np.arange(1, n_states + 1, dtype=np.float32)
    goal /= goal.sum(dtype=np.float32)
    action_prior = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    return tuple(
        jnp.asarray(value)
        for value in (q_state, q_static, transition, goal, action_prior)
    )


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "jit"), required=True)
    parser.add_argument("--fixture", choices=tuple(FIXTURES), required=True)
    parser.add_argument("--min-duration", type=float, default=0.25)
    parser.add_argument("--max-iterations", type=int, default=10_000)
    args = parser.parse_args()

    process = psutil.Process(os.getpid())
    rss_after_import = process.memory_info().rss
    values = fixture(FIXTURES[args.fixture])
    rss_after_fixture = process.memory_info().rss

    def call() -> np.ndarray:
        result = loopy_bp_planning(
            values[0],
            values[1],
            values[2],
            values[3],
            HORIZON,
            PLANNING_ITERATIONS,
            action_prior=values[4],
        )
        result.block_until_ready()
        return np.asarray(result)

    compile_seconds = 0.0
    context = jax.disable_jit() if args.mode == "eager" else nullcontext()
    with context:
        if args.mode == "jit":
            start = time.perf_counter()
            loopy_bp_planning.lower(
                values[0],
                values[1],
                values[2],
                values[3],
                HORIZON,
                PLANNING_ITERATIONS,
                action_prior=values[4],
            ).compile()
            compile_seconds = time.perf_counter() - start
        rss_after_compile = process.memory_info().rss
        start = time.perf_counter()
        policy = call()
        first_execution_seconds = time.perf_counter() - start
        warmup_start = time.perf_counter()
        for _ in range(WARMUP_ITERATIONS):
            policy = call()
        warmup_seconds = time.perf_counter() - warmup_start
        rss_after_warmup = process.memory_info().rss

        samples: list[float] = []
        measurement_start = time.perf_counter()
        while (
            len(samples) < 5
            or time.perf_counter() - measurement_start < args.min_duration
        ) and len(samples) < args.max_iterations:
            start = time.perf_counter()
            policy = call()
            samples.append(time.perf_counter() - start)
        duration = time.perf_counter() - measurement_start

    mean = statistics.fmean(samples)
    payload = {
        "backend": f"jax-{args.mode}",
        "fixture": args.fixture,
        "n_states": FIXTURES[args.fixture],
        "compile_seconds": compile_seconds,
        "first_execution_seconds": first_execution_seconds,
        "warmup_iterations": WARMUP_ITERATIONS,
        "warmup_seconds": warmup_seconds,
        "measurement_iterations": len(samples),
        "measurement_duration_seconds": duration,
        "latency_mean_seconds": mean,
        "latency_median_seconds": statistics.median(samples),
        "latency_min_seconds": min(samples),
        "latency_max_seconds": max(samples),
        "latency_p95_seconds": percentile(samples, 0.95),
        "throughput_calls_per_second": 1.0 / mean,
        "policy": policy.tolist(),
        "rss_bytes": {
            "after_import": rss_after_import,
            "after_fixture": rss_after_fixture,
            "after_compile": rss_after_compile,
            "after_warmup": rss_after_warmup,
            "after_measurement": process.memory_info().rss,
        },
        "versions": {
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "numpy": np.__version__,
            "platform": jax.default_backend(),
            "x64": bool(jax.config.jax_enable_x64),
        },
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
