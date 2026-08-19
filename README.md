# AIF.mojo 🔥

[![CI](https://github.com/antonvice/AIF.mojo/actions/workflows/ci.yml/badge.svg)](https://github.com/antonvice/AIF.mojo/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Mojo](https://img.shields.io/badge/Mojo-1.0.0b2-orange.svg)](https://mojolang.org/)

A standalone, CPU-first toolkit for **active inference and message-passing
planning, written in Mojo**. It provides native belief updates, finite-horizon
planners, environment models, and agent steps for partially observable worlds.

Use it when you want explicit, inspectable Active Inference kernels without a
Python or JAX runtime in the numerical core. The project currently targets
correctness, deterministic behavior, and low-memory CPU execution.

## What it includes

- Eight planners: Loopy BP, Loopy VBP, Region-Extended Loopy BP,
  Dynamic-Channel Loopy BP, Nuijten MP, VBP Channel, Precise Information
  Seeking, and Active Inference.
- Dense execution for all eight, plus deterministic sparse/precomputed paths
  where the model permits them.
- Terminal-state goals and theta-dependent preferences.
- Native models for Frozen Lake, Wumpus World, RockSample, and MiniGrid
  DoorKey.
- Belief updates, planner dispatch, convergence/VFE traces, pure simulator
  steps, runnable examples, and a precompiled `.mojoc` package.
- Flat, row-major `List[Float32]` and `List[Int]` tensors with documented axes.
  There is no hidden NumPy dependency or general tensor framework.

## Quick start

Install [Pixi](https://pixi.sh/), then:

```bash
pixi install
pixi run bootstrap-oracle
pixi run sync-oracle
pixi run manifest
pixi run example-frozen
```

The last command runs a visible native episode. The agent observes a noisy 2x2
Frozen Lake, updates its belief over two possible hole layouts, plans with
Loopy BP, and renders every action:

```text
STEP 0                         STEP 1                         FINAL
+-----+-----+                 +-----+-----+                 +-----+-----+
|  A  |  H  |                 |  .  |  H  |                 |  .  |  H  |
+-----+-----+                 +-----+-----+                 +-----+-----+
|  .  |  G  |  -- DOWN -->    |  A  |  G  |  -- RIGHT -->   |  .  | A/G |
+-----+-----+                 +-----+-----+                 +-----+-----+

posterior theta: [0.98, 0.02]  posterior theta: [0.99973, 0.00027]
SUCCESS — goal reached in 2 steps
```

## Use the planners

The smallest examples are:

```bash
pixi run example-loopy   # deterministic-sparse Loopy BP
pixi run example-all     # all eight planners on one explicit model
pixi run example-frozen  # complete belief-plan-step episode
```

Mojo callers import from `aif_mojo` and pass flat buffers plus dimensions. For
example, the public dense Loopy-BP entry point returns the normalized first
action distribution:

```mojo
from aif_mojo.loopy_bp import loopy_bp_planning_dense

var q_u = loopy_bp_planning_dense(
    q_state,
    q_theta,
    transition,  # T(new_state, old_state, theta, action)
    goal,        # C(state)
    action_prior,
    horizon,
    iterations,
    n_states,
    n_actions,
    n_theta,
)
```

See [`examples/`](examples/) for complete inputs and
[`TECHNICAL_DESIGN.md`](TECHNICAL_DESIGN.md) for every tensor contract
and planner path.

## Mojo versus JAX

![Local JAX versus Mojo benchmark](docs/backend_comparison.svg)

This is a fair CPU comparison of the **same** dense Float32 Loopy-BP fixture.
AOT/JIT compilation and three warm-up calls are separated from repeated
in-process calls. Five independent processes are measured for every backend at
four state-space sizes, and every process must return the same normalized policy
within `1e-5`. Both backends use their default CPU worker pools; the native
kernel parallelizes independent dense reductions from 32 states upward.

<!-- BENCHMARK_RESULTS_START -->
Median and bootstrap 95% CI across five independent processes. Lower latency and memory are better.

| States | Backend | Compile (s) | Latency, ms (95% CI) | Calls/s (95% CI) | Peak RSS | Planner RSS delta |
|---:|---|---:|---:|---:|---:|---:|
| 8 | Mojo native | 4.796 | 0.070 [0.058, 0.071] | 14317.388 [14058.650, 17282.871] | 12.3 MiB | 0.9 MiB |
| 8 | JAX eager | 0.000 | 82.350 [67.221, 97.144] | 12.143 [10.294, 14.876] | 343.1 MiB | 231.8 MiB |
| 8 | JAX warm-JIT | 0.410 | 0.110 [0.100, 0.125] | 9131.472 [8025.559, 10014.269] | 214.8 MiB | 103.8 MiB |
| 32 | Mojo native | 4.796 | 1.507 [1.127, 1.985] | 663.423 [503.818, 887.020] | 12.4 MiB | 1.0 MiB |
| 32 | JAX eager | 0.000 | 95.672 [72.186, 112.277] | 10.452 [8.907, 13.853] | 343.6 MiB | 232.7 MiB |
| 32 | JAX warm-JIT | 0.606 | 0.656 [0.582, 1.115] | 1525.196 [896.536, 1717.124] | 226.6 MiB | 115.7 MiB |
| 64 | Mojo native | 4.796 | 3.241 [3.196, 3.402] | 308.573 [293.982, 312.850] | 12.6 MiB | 1.2 MiB |
| 64 | JAX eager | 0.000 | 76.872 [67.855, 85.713] | 13.009 [11.667, 14.737] | 348.1 MiB | 237.1 MiB |
| 64 | JAX warm-JIT | 0.463 | 3.759 [3.554, 4.393] | 266.011 [227.657, 281.334] | 227.2 MiB | 116.2 MiB |
| 128 | Mojo native | 4.796 | 10.552 [9.876, 11.249] | 94.765 [88.900, 101.257] | 13.9 MiB | 2.5 MiB |
| 128 | JAX eager | 0.000 | 103.585 [68.354, 113.865] | 9.654 [8.782, 14.630] | 360.4 MiB | 249.5 MiB |
| 128 | JAX warm-JIT | 0.521 | 8.540 [7.693, 9.768] | 117.100 [102.377, 129.990] | 231.1 MiB | 120.2 MiB |

Snapshot: Apple M4 (arm64), Mojo 1.0.0b2 (2cf4d08a), 2026-08-19. [Full process-level JSON](docs/benchmarks/2026-08-19.json).
These are machine-specific measurements, not universal language claims.
<!-- BENCHMARK_RESULTS_END -->

Relative to the [v0.1.0 benchmark](https://github.com/antonvice/AIF.mojo/blob/v0.1.0/docs/benchmarks/2026-08-19.json)
on the same Apple M4, the optimized Mojo median is 3.3×, 2.0×, 3.5×, and 6.7×
faster at 8, 32, 64, and 128 states.
In this snapshot Mojo has the lower warm-call median at 8 and 64 states; JAX
warm-JIT leads at 32 and 128 states.

Choose **Mojo native** when low process memory, predictable native deployment,
or avoiding a Python/XLA runtime matters. Choose **JAX warm-JIT** when peak CPU
throughput on repeated array programs matters and its runtime/compile footprint
is acceptable. JAX eager is valuable for research iteration and debugging, but
is not the fast deployment path in this fixture.

For repeated dense terminal-goal planning, construct `PreparedDenseLoopyBP`
once and call `plan()` at every agent step. It caches the static transition,
goal, and action-prior logs and reuses one scratch workspace:

```mojo
from aif_mojo.loopy_bp import PreparedDenseLoopyBP

var planner = PreparedDenseLoopyBP(
    transition, goal, action_prior, horizon, iterations,
    n_states, n_actions, n_theta,
)
var policy = planner.plan(q_state, q_theta)
```

The prepared planner owns mutable scratch storage; use one instance per
concurrently executing agent.

Reproduce the comparison on your machine:

```bash
pixi run benchmark
# JSON: benchmarks/results/fair_latest.json

pixi run benchmark-publication  # five processes, dated JSON, README + SVG
```

`pixi run benchmark-process` is a separate launch-time regression benchmark;
its samples include process startup and must not be mixed with the in-process
table above. Methodology and optimization notes are in
[`MOJO_UPGRADE_NOTES.md`](MOJO_UPGRADE_NOTES.md).

## Verify it

```bash
pixi run test                # native Mojo suites
pixi run parity              # planner/message differentials
pixi run parity-env          # environment and agent differentials
pixi run parity-convergence  # VFE and full convergence traces
pixi run check               # native release gate
pixi run check-all           # also run the frozen JAX suite
```

The nested JAX checkout is a read-only test oracle, not a runtime dependency.
The manifest pins its commit, Python/JAX versions, tensor axes, tolerances, and
deterministic tapes. Python is retained only for oracle comparison, JSON
orchestration, and research-host adapters such as plotting and video.

## Project boundaries

The numerical core is native Mojo and CPU-first. General NDArray semantics,
autodiff, GPU kernels, native plotting/video, and cross-runtime random-stream
identity are intentionally out of scope until a measured use case justifies
them. See the [technical design](TECHNICAL_DESIGN.md) for details.

## Acknowledgements

This standalone Mojo implementation was validated against
[`biaslab/UAI-MP-AIF-JAX`](https://github.com/biaslab/UAI-MP-AIF-JAX), companion
code for [*What Type of Inference Is Active Inference?*](https://arxiv.org/abs/2606.04935)
by Wouter W. L. Nuijten, Mykola Lukashchuk, Thijs van de Laar, and Bert de
Vries. Thank you to its Git
contributors [Wouter Nuijten](https://github.com/biaslab/UAI-MP-AIF-JAX/commits?author=wouternuijten)
and [Mykola Lukashchuk](https://github.com/biaslab/UAI-MP-AIF-JAX/commits?author=nikola-lukashuk)
for the original research implementation and canonical environment semantics.
The reference repository remains unmodified at commit
`30ee6f0ebce32c6a430fa7c25f1c01390415a797`.

## Contributing, citation, and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for the correctness and validation
contract, [CITATION.cff](CITATION.cff) for software and paper citations, and
[API stability](docs/API_STABILITY.md) for compatibility guarantees.

AIF.mojo is licensed under [Apache-2.0](LICENSE). The separately fetched JAX
oracle is not distributed under this license; see [NOTICE](NOTICE).
