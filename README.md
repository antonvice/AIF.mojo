# AIF-MOJO

CPU-first, test-first Mojo port of the numerical core of
`UAI-MP-AIF-JAX`. The nested JAX checkout is a frozen behavioral oracle; native
Mojo performs the state inference, planning, environment tensor generation, and
VFE/convergence calculations.

## Port status

The native core is complete at the current public surface:

- All eight retained planners are native: Loopy BP, Loopy VBP,
  Region-Extended Loopy BP, Dynamic-Channel Loopy BP, Nuijten MP, VBP Channel,
  Precise Information Seeking, and Active Inference.
- Dense paths are available for all eight. Deterministic sparse paths are
  available for Loopy BP, Dynamic-Channel, Nuijten, and VBP Channel; Active
  Inference also has the precomputed sparse path used by MiniGrid.
- Both terminal `(state)` goals and theta-dependent `(state, theta)`
  preferences are covered.
- Frozen Lake, Wumpus World, RockSample, and MiniGrid have native state/index
  rules and transition, observation, and goal tensor generation. Their native
  tests cover deterministic tapes and environment-specific semantics. Frozen
  Lake and RockSample additionally expose pure draw-injected simulator steps
  with reward, termination, and truncation results.
- `agent.mojo` consolidates the eight-kind planner dispatcher and shared belief
  updates. It includes Wumpus and RockSample reset/step paths, Frozen Lake
  reset plus all-eight-planner steps, and probabilistic MiniGrid multimodal
  inference with all eight dense planner routes and all five supported sparse
  routes, including precomputed Active Inference.
- `convergence.mojo` contains the full entropy, energy, and VFE formulas plus
  instrumented traces for the seven convergence planners. Differential coverage
  exercises terminal and theta-dependent goals, giving 14 trace groups.
- The fixture manifest, result schemas, clean-oracle check, native 8-by-4 policy
  smoke matrix, package smoke test, examples, and benchmark runner are wired into
  Pixi.

The implementation intentionally uses flat row-major `List[Float32]` and
`List[Int]` buffers with explicit dimensions. That is a YAGNI decision: the
current kernels do not justify a general `NDArray`, broadcasting system, or
NumPy replacement. GPU kernels and cross-runtime RNG stream parity are also out
of scope.

## Setup

```bash
pixi install
pixi run bootstrap-oracle
pixi run sync-oracle
pixi run manifest
```

`bootstrap-oracle` clones the reference only when absent. An existing checkout
must already be clean and at
`30ee6f0ebce32c6a430fa7c25f1c01390415a797`; the script refuses to mutate a
dirty or wrong-SHA checkout. `sync-oracle` selects the manifest-pinned Python
3.13.13 instead of whichever newer 3.13 patch happens to be installed.
`manifest` verifies that SHA, cleanliness, runtime
versions, Float32/x64 settings, axes, tolerances, goal forms, sparse/dense paths,
and deterministic tapes.

## Checks and runnable surfaces

```bash
# Native and differential correctness
pixi run test
pixi run parity
pixi run parity-env
pixi run parity-convergence
pixi run test-schema

# Reproducibility and distribution boundaries
pixi run experiment-smoke
pixi run package-smoke
pixi run benchmark

# Aggregates
pixi run check
pixi run test-ref
pixi run check-all

# Examples
pixi run example-loopy
pixi run example-all
```

`check` runs the manifest, schema, native, differential, policy-smoke, and
package-smoke gates. `check-all` additionally runs the canonical frozen JAX
suite. `example-loopy` shows the deterministic-sparse Loopy BP API;
`example-all` dispatches all eight planners on one explicit tiny model.

`experiment-smoke` runs every retained planner against native tensors from all
four environment generators and writes 32 reference-shaped result files plus a
summary under `data/smoke_matrix/`. It is deliberately a policy-only smoke
matrix: every result records `n_episodes=0`. It does not claim episode success or
reward measurements.

`benchmark` separates compilation, first execution, and repeated execution. Its
timings are compiled-native **process wall time including process startup**, not
in-process steady-state planner latency. The JSON retains every warm sample and
reports median, quartiles, and IQR. It covers all eight planners on the shared
tiny fixture and one bounded 3x3 MiniGrid Active Inference sample using
`T_idx`, a sparse precomputed dynamics base, and precomputed observation
messages.

## Python boundary and oracle policy

Python remains intentionally responsible for the frozen JAX oracle, pytest
differential driving, JSON orchestration, DVC/YAML workflows, plots, and
Gymnasium/MiniGrid hosting, rendering, and video. These are adapters around the
native numerical core; the experiment and benchmark drivers invoke Mojo rather
than reimplementing inference in Python.

The JAX checkout is unchanged at the pinned SHA. One known reference inconsistency
is documented rather than patched locally: its RockSample Loopy BP agent alone
bypasses `terminal_goal_only` goal marginalization. The consolidated Mojo agent
applies the documented behavior consistently to all eight planners. That is an
intentional consistency fix and a candidate upstream JAX pull request.

See [MOJO_PORTING_TECHDOC.md](MOJO_PORTING_TECHDOC.md) for tensor contracts,
test rationale, milestone closure, limitations, and the definition of done. The
original JAX baseline record is in
[baseline/jax/2026-08-07/BASELINE.md](baseline/jax/2026-08-07/BASELINE.md).
