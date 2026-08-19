#!/usr/bin/env python3
"""Publish a dated benchmark record and regenerate README/SVG summaries."""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SVG = ROOT / "docs" / "backend_comparison.svg"
DATED_RESULTS = ROOT / "docs" / "benchmarks"
START = "<!-- BENCHMARK_RESULTS_START -->"
END = "<!-- BENCHMARK_RESULTS_END -->"
LABELS = {
    "mojo-native": "Mojo native",
    "jax-eager": "JAX eager",
    "jax-jit": "JAX warm-JIT",
}
COLORS = {
    "mojo-native": "#f97316",
    "jax-eager": "#94a3b8",
    "jax-jit": "#2563eb",
}


def fmt_ci(value: float, interval: list[float], scale: float = 1.0) -> str:
    return (
        f"{value * scale:.3f} "
        f"[{interval[0] * scale:.3f}, {interval[1] * scale:.3f}]"
    )


def markdown(payload: dict[str, Any], dated_path: Path) -> str:
    lines = [
        "Median and bootstrap 95% CI across five independent processes. "
        "Lower latency and memory are better.",
        "",
        "| States | Backend | Compile (s) | Latency, ms (95% CI) | Calls/s (95% CI) | Peak RSS | Planner RSS delta |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for fixture in payload["fixtures"]:
        for result in fixture["results"]:
            lines.append(
                "| {states} | {backend} | {compile:.3f} | {latency} | "
                "{throughput} | {peak:.1f} MiB | {delta:.1f} MiB |".format(
                    states=fixture["n_states"],
                    backend=LABELS[result["backend"]],
                    compile=result["compile_seconds_median"],
                    latency=fmt_ci(
                        result["latency_seconds_median"],
                        result["latency_seconds_ci95"],
                        1000,
                    ),
                    throughput=fmt_ci(
                        result["throughput_calls_per_second_median"],
                        result["throughput_calls_per_second_ci95"],
                    ),
                    peak=result["runtime_peak_rss_bytes_median"] / 2**20,
                    delta=result["planner_incremental_rss_bytes_median"] / 2**20,
                )
            )
    date = payload["recorded_at_utc"][:10]
    system = payload["system"]
    lines.extend(
        [
            "",
            f"Snapshot: {system['cpu_brand']} ({system['machine']}), "
            f"{payload['versions']['mojo']}, {date}. "
            f"[Full process-level JSON]({dated_path.resolve().relative_to(ROOT).as_posix()}).",
            "These are machine-specific measurements, not universal language claims.",
        ]
    )
    return "\n".join(lines)


def svg(payload: dict[str, Any]) -> str:
    rows = [
        (fixture["n_states"], result)
        for fixture in payload["fixtures"]
        for result in fixture["results"]
    ]
    latencies = [row[1]["latency_seconds_median"] * 1000 for row in rows]
    memories = [row[1]["runtime_peak_rss_bytes_median"] / 2**20 for row in rows]
    latency_floor = min(latencies) * 0.75
    latency_ceiling = max(latencies) * 1.15
    memory_ceiling = max(memories) * 1.08

    def latency_width(value: float) -> float:
        fraction = math.log(value / latency_floor) / math.log(
            latency_ceiling / latency_floor
        )
        return max(8.0, 430.0 * fraction)

    def memory_width(value: float) -> float:
        return max(8.0, 430.0 * value / memory_ceiling)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="860" viewBox="0 0 1100 860" role="img" aria-labelledby="title desc">',
        '<title id="title">AIF.mojo publication benchmark</title>',
        '<desc id="desc">Median latency and peak process memory across five independent processes for four Loopy BP state spaces.</desc>',
        '<rect width="1100" height="860" rx="18" fill="#f8fafc"/>',
        '<text x="55" y="50" font-family="system-ui,sans-serif" font-size="28" font-weight="700" fill="#111827">Mojo native and JAX runtime tradeoffs</text>',
        '<text x="55" y="78" font-family="system-ui,sans-serif" font-size="15" fill="#4b5563">Same Float32 Loopy-BP planner · median of 5 processes · lower is better</text>',
    ]

    def panel(title: str, top: int, values: list[float], width_fn, suffix: str) -> None:
        parts.append(
            f'<text x="55" y="{top}" font-family="system-ui,sans-serif" font-size="20" font-weight="700" fill="#111827">{html.escape(title)}</text>'
        )
        for index, ((states, result), value) in enumerate(zip(rows, values)):
            y = top + 28 + index * 24
            label = f"S={states} {LABELS[result['backend']]}"
            width = width_fn(value)
            color = COLORS[result["backend"]]
            parts.extend(
                [
                    f'<text x="55" y="{y + 14}" font-family="system-ui,sans-serif" font-size="12" fill="#334155">{html.escape(label)}</text>',
                    f'<rect x="225" y="{y}" width="{width:.1f}" height="18" rx="4" fill="{color}"/>',
                    f'<text x="{235 + width:.1f}" y="{y + 14}" font-family="system-ui,sans-serif" font-size="12" fill="#111827">{value:.3f}{suffix}</text>',
                ]
            )

    panel("Median latency (log scale)", 120, latencies, latency_width, " ms")
    panel("Median peak process memory", 475, memories, memory_width, " MiB")
    parts.extend(
        [
            '<rect x="55" y="820" width="14" height="14" rx="3" fill="#f97316"/><text x="77" y="832" font-family="system-ui,sans-serif" font-size="13" fill="#475569">Mojo native</text>',
            '<rect x="190" y="820" width="14" height="14" rx="3" fill="#94a3b8"/><text x="212" y="832" font-family="system-ui,sans-serif" font-size="13" fill="#475569">JAX eager</text>',
            '<rect x="310" y="820" width="14" height="14" rx="3" fill="#2563eb"/><text x="332" y="832" font-family="system-ui,sans-serif" font-size="13" fill="#475569">JAX warm-JIT</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def publish(input_path: Path) -> Path:
    payload = json.loads(input_path.read_text())
    if payload.get("schema_version") != 2:
        raise ValueError("publication requires benchmark schema_version=2")
    date = payload["recorded_at_utc"][:10]
    DATED_RESULTS.mkdir(parents=True, exist_ok=True)
    dated_path = DATED_RESULTS / f"{date}.json"
    shutil.copyfile(input_path, dated_path)

    readme = README.read_text()
    if readme.count(START) != 1 or readme.count(END) != 1:
        raise ValueError("README benchmark markers are missing or duplicated")
    before, remainder = readme.split(START)
    _, after = remainder.split(END)
    README.write_text(
        before + START + "\n" + markdown(payload, dated_path) + "\n" + END + after
    )
    SVG.write_text(svg(payload))
    return dated_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    path = publish(args.input)
    print(f"published benchmark artifacts: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
