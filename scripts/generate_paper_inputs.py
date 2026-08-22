#!/usr/bin/env python3
"""Freeze the paper's five random environment configurations for native Mojo.

The optional JAX oracle is used only to sample the exact seed-0..4 configs.
The generated Mojo module contains configs, not inference outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "UAI-MP-AIF-JAX"
DEFAULT_OUTPUT = ROOT / ".generated" / "paper_inputs.mojo"
SEEDS = range(5)


def _float_literal(value: float) -> str:
    return f"Float32({float(np.float32(value))!r})"


def _emit_function(name: str, values: dict[int, np.ndarray], dtype: str) -> str:
    lines = [f"def {name}(seed: Int) -> List[{dtype}]:", f"    var result = List[{dtype}]()"]
    for seed, array in values.items():
        lines.append(f"    if seed == {seed}:")
        for value in array.ravel():
            literal = str(int(value)) if dtype == "Int" else _float_literal(float(value))
            lines.append(f"        result.append({literal})")
        lines.append("        return result^")
    lines.extend(["    debug_assert(False, \"paper seed must be 0..4\")", "    return result^"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not ORACLE.exists():
        raise SystemExit("missing UAI-MP-AIF-JAX; run `pixi run bootstrap-oracle`")
    sys.path.insert(0, str(ORACLE))
    from environments.frozen_lake import sample_configs as sample_frozen
    from environments.rocksample import sample_rock_positions
    from environments.wumpus_world import sample_configs as sample_wumpus

    frozen = {
        seed: sample_frozen(4, 15, hole_fraction=0.2, seed=seed, min_hamming=4)
        for seed in SEEDS
    }
    pits: dict[int, np.ndarray] = {}
    wumpus: dict[int, np.ndarray] = {}
    gold: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        pits[seed], wumpus[seed], gold[seed] = sample_wumpus(
            5, 25, n_pits=4, seed=seed
        )
    rocks = {seed: sample_rock_positions(4, 3, seed=seed) for seed in SEEDS}
    sections = [
        "from std.collections import List",
        _emit_function("paper_frozen_holes", frozen, "Float32"),
        _emit_function("paper_wumpus_pits", pits, "Float32"),
        _emit_function("paper_wumpus_monsters", wumpus, "Float32"),
        _emit_function("paper_wumpus_gold", gold, "Float32"),
        _emit_function("paper_rock_positions", rocks, "Int"),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n\n".join(sections) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
