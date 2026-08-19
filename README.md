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
within `1e-5`.

<!-- BENCHMARK_RESULTS_START -->
Run `pixi run benchmark-publication` to generate the publication table.
<!-- BENCHMARK_RESULTS_END -->

Choose **Mojo native** when low process memory, predictable native deployment,
or avoiding a Python/XLA runtime matters. Choose **JAX warm-JIT** when peak CPU
throughput on repeated array programs matters and its runtime/compile footprint
is acceptable. JAX eager is valuable for research iteration and debugging, but
is not the fast deployment path in this fixture.

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
