#!/usr/bin/env python3
"""Clone the frozen JAX oracle once, then enforce clean exact-SHA reuse."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/biaslab/UAI-MP-AIF-JAX"
COMMIT = "30ee6f0ebce32c6a430fa7c25f1c01390415a797"


def run(command: list[str], cwd: Path = ROOT) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def verify_existing(destination: Path) -> None:
    if not (destination / ".git").exists():
        raise RuntimeError(f"existing path is not a Git checkout: {destination}")
    status = run(
        ["git", "status", "--porcelain", "--untracked-files=all"], destination
    )
    if status:
        raise RuntimeError(f"existing oracle checkout is dirty:\n{status}")
    actual = run(["git", "rev-parse", "HEAD"], destination)
    if actual != COMMIT:
        raise RuntimeError(
            f"existing oracle is at {actual}; required {COMMIT}; refusing mutation"
        )


def bootstrap(destination: Path) -> str:
    if destination.exists():
        verify_existing(destination)
        return "verified"
    run(["git", "clone", REPOSITORY, str(destination)])
    run(["git", "checkout", "--detach", COMMIT], destination)
    verify_existing(destination)
    return "cloned"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=ROOT / "UAI-MP-AIF-JAX"
    )
    args = parser.parse_args()
    try:
        action = bootstrap(args.destination.resolve())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"FAIL oracle bootstrap: {exc}", file=sys.stderr)
        return 1
    print(f"PASS oracle {action}: {args.destination.resolve()} @ {COMMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
