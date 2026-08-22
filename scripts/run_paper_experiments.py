#!/usr/bin/env python3
"""Run the native five-method, three-environment authors-code protocol."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / ".generated"
EXECUTABLE = GENERATED / "paper_episode_runner"
ENVIRONMENTS = ("frozen-lake", "rocksample", "wumpus-world")
METHODS = ("bp", "vbp", "rm-mp", "nuijten-mp", "aif-mp")


def _build() -> None:
    oracle_python = ROOT / "UAI-MP-AIF-JAX" / ".venv" / "bin" / "python"
    subprocess.run([str(oracle_python), "scripts/generate_paper_inputs.py"], cwd=ROOT, check=True)
    subprocess.run(
        [
            "mojo",
            "build",
            "--Werror",
            "-I",
            "src",
            "-I",
            ".generated",
            "scripts/paper_episode_runner.mojo",
            "-o",
            str(EXECUTABLE),
        ],
        cwd=ROOT,
        check=True,
    )


def _parse(stdout: str, elapsed: float) -> dict[str, object]:
    line = next(line for line in stdout.splitlines() if line.startswith("RESULT "))
    fields = line.split()
    episodes = int(fields[3])
    total_good = int(fields[8])
    return {
        "environment": fields[1],
        "method": fields[2],
        "episodes": episodes,
        "successes": int(fields[4]),
        "success_rate": int(fields[4]) / episodes,
        "avg_reward": float(fields[5]) / episodes,
        "avg_steps": int(fields[6]) / episodes,
        "good_rock_retrieval": int(fields[7]) / total_good if total_good else None,
        "total_good_collected": int(fields[7]),
        "total_good_rocks": total_good,
        "episodes_with_good_rocks": int(fields[9]),
        "elapsed_s": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--environment", choices=ENVIRONMENTS, action="append")
    parser.add_argument("--method", choices=METHODS, action="append")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "paper_experiments" / "results.json")
    args = parser.parse_args()
    _build()
    if args.environment or args.method or args.mode == "full":
        environments = args.environment or ENVIRONMENTS
        methods = args.method or METHODS
        cases = [(environment, method) for environment in environments for method in methods]
    else:
        # Cheap CI/manual smoke: all five planners plus all three simulators.
        cases = [("frozen-lake", method) for method in METHODS]
        cases.extend((("wumpus-world", "bp"), ("rocksample", "bp")))
    rows: list[dict[str, object]] = []
    for environment, method in cases:
        started = time.perf_counter()
        completed = subprocess.run(
            [str(EXECUTABLE), environment, method, args.mode],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        row = _parse(completed.stdout, time.perf_counter() - started)
        rows.append(row)
        print(
            f"{environment:13} {method:10} success={row['success_rate']:.3f} "
            f"reward={row['avg_reward']:.3f} steps={row['avg_steps']:.2f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "profile": "authors-public-code-at-30ee6f0",
        "paper": "arXiv:2606.04935v4",
        "mode": args.mode,
        "rng": "Mojo std.random, seeded per episode; not NumPy bitstream parity",
        "rocksample_discrepancy": "paper prose says 16 states/9 actions; public code uses 640 states/8 actions",
        "results": rows,
    }
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    print(csv_path)


if __name__ == "__main__":
    main()
