#!/usr/bin/env python3
"""Run the paper-exact Appendix F.5 convergence matrix with native Mojo."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / ".generated"
EXECUTABLE = GENERATED / "appendix_f5_sweep"
DAMPING = (0.25, 0.4, 0.5, 0.6, 0.75, 0.9)
ENVIRONMENTS = ("frozen-lake", "rocksample", "wumpus-world")
METHODS = ("bp", "vbp", "rm-mp", "aif-mp")


def _build() -> None:
    subprocess.run(
        [str(ROOT / "UAI-MP-AIF-JAX" / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "generate_paper_inputs.py")],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            "mojo",
            "build",
            "--Werror",
            "-I",
            "src",
            "-I",
            ".generated",
            "scripts/appendix_f5_sweep.mojo",
            "-o",
            str(EXECUTABLE),
        ],
        cwd=ROOT,
        check=True,
    )


def _parse(stdout: str, elapsed: float, adaptive: bool) -> dict[str, object]:
    result_line = next(line for line in stdout.splitlines() if line.startswith("RESULT "))
    fields = result_line.split()
    actions = [float(line.split()[2]) for line in stdout.splitlines() if line.startswith("ACTION ")]
    return {
        "environment": fields[1],
        "method": fields[2],
        "seed": int(fields[3]),
        "initial_damping": float(fields[4]),
        "converged": bool(int(float(fields[5]))),
        "iterations": int(float(fields[6])),
        "final_residual": float(fields[7]),
        "final_damping": float(fields[8]),
        "adaptive_damping": adaptive,
        "action_distribution": actions,
        "elapsed_s": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--environment", choices=ENVIRONMENTS, action="append")
    parser.add_argument("--method", choices=METHODS, action="append")
    parser.add_argument("--seed", type=int, choices=range(5), action="append")
    parser.add_argument("--damping", type=float, choices=(*DAMPING, 1.0), action="append")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "appendix_f5")
    args = parser.parse_args()
    _build()
    environments = args.environment or (("frozen-lake",) if args.mode == "smoke" else ENVIRONMENTS)
    methods = args.method or METHODS
    seeds = args.seed or ((0,) if args.mode == "smoke" else tuple(range(5)))
    rows: list[dict[str, object]] = []
    for environment in environments:
        for method in methods:
            dampings = (1.0,) if method == "bp" else (tuple(args.damping) if args.damping else DAMPING)
            for damping in dampings:
                for seed in seeds:
                    command = [
                        str(EXECUTABLE),
                        environment,
                        method,
                        str(damping),
                        str(seed),
                        args.mode,
                        "adaptive" if args.adaptive and method != "bp" else "fixed",
                    ]
                    started = time.perf_counter()
                    completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
                    row = _parse(completed.stdout, time.perf_counter() - started, args.adaptive and method != "bp")
                    rows.append(row)
                    print(
                        f"{environment:13} {method:6} λ={damping:<4} seed={seed} "
                        f"iter={row['iterations']} residual={row['final_residual']:.3e}"
                    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.output_dir / ("adaptive.json" if args.adaptive else "fixed.json")
    artifact.write_text(json.dumps({"mode": args.mode, "criterion": 1e-4, "maximum_iterations": 1000 if args.mode == "full" else 2, "runs": rows}, indent=2) + "\n")
    summary = args.output_dir / ("adaptive.csv" if args.adaptive else "fixed.csv")
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key != "action_distribution"])
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key != "action_distribution"} for row in rows)
    print(artifact)
    print(summary)


if __name__ == "__main__":
    main()
