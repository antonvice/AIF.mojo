#!/usr/bin/env python3
"""Publication-grade Loopy-BP comparison for JAX eager/JIT and native Mojo."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
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
FIXTURES = {"small": 8, "medium": 32, "large": 64, "xlarge": 128}
BACKENDS = ("mojo-native", "jax-eager", "jax-jit")
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
            command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr
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


def native_baseline(binary: Path) -> int:
    measured = run_measured([str(binary), "baseline"])
    if measured.stdout.strip() != "BASELINE":
        raise ValueError(f"invalid native baseline output: {measured.stdout!r}")
    return measured.peak_rss_bytes


def native_result(
    binary: Path, fixture: str, baseline_rss_bytes: int
) -> dict[str, Any]:
    measured = run_measured([str(binary), fixture])
    fields = measured.stdout.strip().split()
    if len(fields) != 13 or fields[:2] != ["RESULT", fixture]:
        raise ValueError(f"invalid native output: {measured.stdout!r}")
    mean = float(fields[3])
    return {
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
        "latency_min_seconds": float(fields[4]),
        "latency_max_seconds": float(fields[5]),
        "throughput_calls_per_second": 1.0 / mean,
        "runtime_baseline_rss_bytes": baseline_rss_bytes,
        "runtime_peak_rss_bytes": measured.peak_rss_bytes,
        "planner_incremental_rss_bytes": max(
            0, measured.peak_rss_bytes - baseline_rss_bytes
        ),
        "policy": [float(value) for value in fields[9:13]],
    }


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
    baseline = int(result["rss_bytes"]["after_import"])
    result["runtime_baseline_rss_bytes"] = baseline
    result["runtime_peak_rss_bytes"] = measured.peak_rss_bytes
    result["planner_incremental_rss_bytes"] = max(
        0, measured.peak_rss_bytes - baseline
    )
    return result


def bootstrap_median_ci(values: list[float]) -> list[float]:
    """Deterministic percentile bootstrap 95% CI for a process-level median."""
    rng = random.Random(0)
    medians = sorted(
        statistics.median(rng.choices(values, k=len(values))) for _ in range(10_000)
    )
    return [medians[249], medians[9749]]


def aggregate(
    backend: str,
    fixture: str,
    runs: list[dict[str, Any]],
    mojo_compile_seconds: float,
) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        return [float(run[key]) for run in runs if run.get(key) is not None]

    latency = values("latency_mean_seconds")
    throughput = values("throughput_calls_per_second")
    peak = values("runtime_peak_rss_bytes")
    baseline = values("runtime_baseline_rss_bytes")
    incremental = values("planner_incremental_rss_bytes")
    compile_values = (
        [mojo_compile_seconds]
        if backend == "mojo-native"
        else values("compile_seconds")
    )
    return {
        "backend": backend,
        "fixture": fixture,
        "n_states": FIXTURES[fixture],
        "process_count": len(runs),
        "compile_seconds_median": statistics.median(compile_values),
        "compile_seconds_ci95": bootstrap_median_ci(compile_values),
        "warmup_seconds_median": statistics.median(values("warmup_seconds")),
        "latency_seconds_median": statistics.median(latency),
        "latency_seconds_ci95": bootstrap_median_ci(latency),
        "throughput_calls_per_second_median": statistics.median(throughput),
        "throughput_calls_per_second_ci95": bootstrap_median_ci(throughput),
        "runtime_peak_rss_bytes_median": int(statistics.median(peak)),
        "runtime_baseline_rss_bytes_median": int(statistics.median(baseline)),
        "planner_incremental_rss_bytes_median": int(
            statistics.median(incremental)
        ),
        "policy": runs[0]["policy"],
        "processes": runs,
    }


def validate(fixture: str, runs: list[dict[str, Any]]) -> None:
    reference = runs[0]["policy"]
    for run in runs:
        if run["n_states"] != FIXTURES[fixture]:
            raise AssertionError(f"state count changed for {fixture}")
        if not math.isclose(sum(run["policy"]), 1.0, abs_tol=1e-5):
            raise AssertionError(f"unnormalized policy: {run['backend']}")
        if max(
            abs(actual - expected)
            for actual, expected in zip(run["policy"], reference)
        ) > 1e-5:
            raise AssertionError(f"policy mismatch: {run['backend']}")


def command_text(command: list[str], cwd: Path = ROOT) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def system_metadata() -> dict[str, Any]:
    hardware: dict[str, str] = {}
    try:
        output = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType", "-json"], text=True
        )
        hardware = json.loads(output)["SPHardwareDataType"][0]
    except (OSError, subprocess.CalledProcessError, KeyError, json.JSONDecodeError):
        pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_brand": hardware.get("chip_type", "unknown"),
        "machine_model": hardware.get("machine_model", "unknown"),
        "physical_memory": hardware.get("physical_memory", "unknown"),
    }


def print_table(payload: dict[str, Any]) -> None:
    print(
        "fixture  backend          compile(s)  latency(ms)  calls/s   "
        "peak MiB  delta MiB"
    )
    for fixture in payload["fixtures"]:
        for result in fixture["results"]:
            print(
                f"{fixture['name']:<8} {result['backend']:<16} "
                f"{result['compile_seconds_median']:>10.3f}  "
                f"{result['latency_seconds_median'] * 1000:>11.3f}  "
                f"{result['throughput_calls_per_second_median']:>7.1f}  "
                f"{result['runtime_peak_rss_bytes_median'] / 2**20:>8.1f}  "
                f"{result['planner_incremental_rss_bytes_median'] / 2**20:>9.1f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "fair_latest.json",
    )
    parser.add_argument("--process-runs", type=int, default=5)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="also refresh the dated JSON, README table, and SVG",
    )
    args = parser.parse_args()
    if args.process_runs < 2:
        raise SystemExit("--process-runs must be at least 2")
    if not ORACLE_PYTHON.is_file():
        raise SystemExit("missing oracle environment; run `pixi run sync-oracle`")

    with tempfile.TemporaryDirectory(prefix="aif-fair-benchmark-") as temporary:
        binary = Path(temporary) / "fair_planner_benchmark"
        compilation = compile_native(binary)
        native_baselines = [
            native_baseline(binary) for _ in range(args.process_runs)
        ]
        fixtures = []
        for fixture in FIXTURES:
            runs_by_backend = {
                "mojo-native": [
                    native_result(binary, fixture, native_baselines[index])
                    for index in range(args.process_runs)
                ],
                "jax-eager": [
                    jax_result("eager", fixture) for _ in range(args.process_runs)
                ],
                "jax-jit": [
                    jax_result("jit", fixture) for _ in range(args.process_runs)
                ],
            }
            all_runs = [run for runs in runs_by_backend.values() for run in runs]
            validate(fixture, all_runs)
            fixtures.append(
                {
                    "name": fixture,
                    "n_states": FIXTURES[fixture],
                    "results": [
                        aggregate(
                            backend,
                            fixture,
                            runs_by_backend[backend],
                            compilation.wall_seconds,
                        )
                        for backend in BACKENDS
                    ],
                }
            )

    payload = {
        "schema_version": 2,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "planner": "loopy-bp-dense-terminal-goal",
        "fixture_contract": {
            "n_static": 2,
            "n_actions": 4,
            "horizon": 3,
            "planning_iterations": 3,
            "dtype": "Float32",
            "transition_axes": ["new", "old", "theta", "action"],
            "state_spaces": FIXTURES,
        },
        "methodology": {
            "same_fixture_and_policy_verified": True,
            "independent_processes_per_backend_and_size": args.process_runs,
            "warmup_iterations_per_process": WARMUP_ITERATIONS,
            "repeated_calls_are_in_process": True,
            "confidence_interval": "deterministic percentile bootstrap 95% CI of process medians; 10000 resamples; seed 0",
            "jax_jit_compilation_is_separate": True,
            "mojo_aot_compilation_is_separate": True,
            "cpu_protocol": "single-threaded Mojo loops; JAX CPU with BLAS threads=1 and XLA Eigen multithreading disabled",
            "memory_metric": "complete process peak RSS plus runtime baseline and non-negative planner delta",
            "mojo_baseline": "peak RSS of an independent no-op invocation of the same AOT binary",
            "jax_baseline": "same-process RSS after JAX and planner import, before fixture allocation",
            "mojo_latency": "std.benchmark mean per process with max_batch_size=1",
            "jax_latency": "mean synchronous block_until_ready call per process",
        },
        "system": system_metadata(),
        "versions": {
            "mojo": command_text(["pixi", "run", "mojo", "--version"]),
            "aif_mojo_git": command_text(["git", "rev-parse", "HEAD"]),
            "jax_oracle_git": command_text(
                ["git", "rev-parse", "HEAD"], ROOT / "UAI-MP-AIF-JAX"
            ),
        },
        "mojo_compile": {
            "seconds": compilation.wall_seconds,
            "peak_rss_bytes": compilation.peak_rss_bytes,
        },
        "fixtures": fixtures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.publish:
        from render_fair_results import publish

        publish(args.output)
    print_table(payload)
    print(f"PASS fair benchmark: output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
