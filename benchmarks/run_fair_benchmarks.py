#!/usr/bin/env python3
"""Compare identical Loopy-BP fixtures in JAX eager/JIT and native Mojo."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCE = ROOT / "benchmarks" / "fair_planner_benchmark.mojo"
JAX_WORKER = ROOT / "benchmarks" / "jax_fair_benchmark.py"
ORACLE_PYTHON = ROOT / "UAI-MP-AIF-JAX" / ".venv" / "bin" / "python"
FIXTURES = ("small", "large")
WARMUP_ITERATIONS = 3


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    wall_seconds: float
    peak_rss_bytes: int


def run_measured(
    command: list[str], *, env: dict[str, str] | None = None
) -> CommandResult:
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        start = time.perf_counter()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
        _, status, usage = os.wait4(process.pid, 0)
        wall_seconds = time.perf_counter() - start
        returncode = os.waitstatus_to_exitcode(status)
        stdout.seek(0)
        stderr.seek(0)
        output = stdout.read().decode()
        errors = stderr.read().decode()
    if returncode:
        raise subprocess.CalledProcessError(
            returncode, command, output=output, stderr=errors
        )
    peak = int(usage.ru_maxrss)
    if sys.platform != "darwin":
        peak *= 1024
    return CommandResult(output, errors, wall_seconds, peak)


def compile_native(binary: Path) -> CommandResult:
    return run_measured(
        [
            "pixi",
            "run",
            "mojo",
            "build",
            "--Werror",
            "-I",
            str(ROOT / "src"),
            str(NATIVE_SOURCE),
            "-o",
            str(binary),
        ]
    )


def native_result(binary: Path, fixture: str) -> dict[str, Any]:
    measured = run_measured([str(binary), fixture])
    fields = measured.stdout.strip().split()
    if len(fields) != 13 or fields[:2] != ["RESULT", fixture]:
        raise ValueError(f"invalid native output: {measured.stdout!r}")
    mean = float(fields[3])
    result = {
        "backend": "mojo-native",
        "fixture": fixture,
        "n_states": int(fields[2]),
        "compile_seconds": None,
        "first_execution_seconds": None,
        "warmup_iterations": WARMUP_ITERATIONS,
        "warmup_seconds": int(fields[8]) / 1_000_000_000,
        "measurement_iterations": int(fields[7]),
        "measurement_duration_seconds": float(fields[6]),
        "latency_mean_seconds": mean,
        "latency_median_seconds": None,
        "latency_min_seconds": float(fields[4]),
        "latency_max_seconds": float(fields[5]),
        "latency_p95_seconds": None,
        "throughput_calls_per_second": 1.0 / mean,
        "runtime_peak_rss_bytes": measured.peak_rss_bytes,
        "policy": [float(value) for value in fields[9:13]],
    }
    return result


def jax_result(mode: str, fixture: str) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "false",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
        }
    )
    measured = run_measured(
        [
            str(ORACLE_PYTHON),
            str(JAX_WORKER),
            "--mode",
            mode,
            "--fixture",
            fixture,
        ],
        env=environment,
    )
    result = json.loads(measured.stdout)
    result["runtime_peak_rss_bytes"] = measured.peak_rss_bytes
    return result


def validate(fixture: str, results: list[dict[str, Any]]) -> None:
    expected_states = 8 if fixture == "small" else 64
    reference = results[0]["policy"]
    for result in results:
        if result["n_states"] != expected_states:
            raise AssertionError(f"state count changed for {fixture}")
        if not math.isclose(sum(result["policy"]), 1.0, abs_tol=1e-5):
            raise AssertionError(f"unnormalized policy: {result['backend']}")
        if max(
            abs(actual - expected)
            for actual, expected in zip(result["policy"], reference)
        ) > 1e-5:
            raise AssertionError(f"policy mismatch: {result['backend']}")


def print_table(payload: dict[str, Any]) -> None:
    print(
        "fixture  backend          compile(s)  latency(ms)  calls/s     peak MiB"
    )
    for fixture in payload["fixtures"]:
        for result in fixture["results"]:
            compile_seconds = result["compile_seconds"]
            compile_text = "-" if compile_seconds is None else f"{compile_seconds:.3f}"
            print(
                f"{fixture['name']:<8} {result['backend']:<16} "
                f"{compile_text:>10}  "
                f"{result['latency_mean_seconds'] * 1000:>11.3f}  "
                f"{result['throughput_calls_per_second']:>10.1f}  "
                f"{result['runtime_peak_rss_bytes'] / 2**20:>10.1f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "fair_latest.json",
    )
    args = parser.parse_args()
    if not ORACLE_PYTHON.is_file():
        raise SystemExit("missing oracle environment; run `pixi run sync-oracle`")

    with tempfile.TemporaryDirectory(prefix="aif-fair-benchmark-") as temporary:
        binary = Path(temporary) / "fair_planner_benchmark"
        compilation = compile_native(binary)
        fixtures = []
        for fixture in FIXTURES:
            native = native_result(binary, fixture)
            eager = jax_result("eager", fixture)
            jit = jax_result("jit", fixture)
            native["compile_seconds"] = compilation.wall_seconds
            results = [native, eager, jit]
            validate(fixture, results)
            fixtures.append(
                {
                    "name": fixture,
                    "n_states": native["n_states"],
                    "results": results,
                }
            )

    payload = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "planner": "loopy-bp-dense-terminal-goal",
        "fixture_contract": {
            "n_static": 2,
            "n_actions": 4,
            "horizon": 3,
            "planning_iterations": 3,
            "dtype": "Float32",
            "transition_axes": ["new", "old", "theta", "action"],
            "small_n_states": 8,
            "large_n_states": 64,
        },
        "methodology": {
            "same_fixture_and_policy_verified": True,
            "warmup_iterations": WARMUP_ITERATIONS,
            "repeated_calls_are_in_process": True,
            "jax_jit_compilation_is_separate": True,
            "mojo_aot_compilation_is_separate": True,
            "cpu_protocol": "single-threaded Mojo loops; JAX CPU with BLAS threads=1 and XLA Eigen multithreading disabled",
            "memory_metric": "maximum resident set size of each complete runtime process",
            "mojo_latency": "std.benchmark mean with max_batch_size=1",
            "jax_latency": "wall time per synchronous block_until_ready call",
        },
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "versions": {
            "mojo": subprocess.check_output(
                ["pixi", "run", "mojo", "--version"], cwd=ROOT, text=True
            ).strip(),
            "aif_mojo_git": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "jax_oracle_git": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT / "UAI-MP-AIF-JAX",
                text=True,
            ).strip(),
        },
        "mojo_compile": {
            "seconds": compilation.wall_seconds,
            "peak_rss_bytes": compilation.peak_rss_bytes,
        },
        "fixtures": fixtures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print_table(payload)
    print(f"PASS fair benchmark: output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
