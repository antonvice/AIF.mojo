# Mojo release and API review

Reviewed: 2026-08-22

The repository is pinned to Mojo `1.0.0`, the current stable release. The
upgrade from `1.0.0b2` was validated with the package gate and all native tests
before the numerical differential gates were rerun.

## Adopted now

- Native measurements use `std.benchmark.run` with explicit warm-up and
  `std.benchmark.keep` so repeated planner work is measured in-process and its
  output remains observable.
- Packaging uses `--Werror --warn-on-unstable-apis`; examples use `--Werror`.
- Dense terminal-goal Loopy BP uses a reusable contiguous workspace,
  allocation-free streaming reductions, and four-lane action SIMD with scalar
  tails.
- `PreparedDenseLoopyBP` caches static log tensors and the workspace across
  agent steps. The original probability-space API delegates to the same fast
  kernel.
- `PreparedDenseVBP[H, I]`, `PreparedDenseRMMP[H, I]`, and
  `PreparedDenseAIFMP[H, I]` cache static log inputs and make horizon/iteration
  specialization explicit in Mojo's compile-time parameter system.
- Residual-aware BP, VBP, RM-MP, and AIF-MP paths support early stopping;
  channel methods additionally expose bounded adaptive damping.
- Tests retain `TestSuite.discover_tests()` plus `mojo run`, matching the
  current documented testing workflow.

## Deferred until a measured consumer exists

- `LayoutTensor`, `Span`, and a general tensor abstraction: flat Lists plus
  `unsafe_ptr()` provide the measured SIMD path without expanding the public
  tensor surface.
- CPU `parallelize`: standalone Mojo 1.0.0 no longer exports this API. Adding
  MAX solely for the thread pool would introduce a proprietary dependency, so
  the pure-Mojo kernel currently uses its serial fallback. Revisit concurrency
  only with an independently measured, dependency-compatible implementation.
- Full scratch-buffer reuse for the channel planners and GPU kernels: add only
  with their own differential and benchmark contracts.
- Batched execution: one `PreparedDenseLoopyBP` instance intentionally owns one
  mutable workspace. A serial wrapper is not a batch optimization; promote
  independent workspace lanes only for a measured multi-agent consumer.
- Nightly compiler features: evaluate in a branch and run `check-all` plus the
  fair benchmark before changing `pixi.lock` or the manifest.

## Official references

- [Mojo releases](https://mojolang.org/releases/)
- [Mojo 1.0.0 changes](https://mojolang.org/releases/v1.0.0/)
- [`std.benchmark`](https://mojolang.org/docs/std/benchmark/)
- [`List` capacity and value semantics](https://mojolang.org/docs/std/collections/list/List/)
- [`mojo build` diagnostics](https://mojolang.org/docs/cli/build/)
- [Testing with `TestSuite`](https://mojolang.org/docs/tools/testing/)
- [Layout, batching, and accelerator evaluation](docs/ACCELERATOR_EVALUATION.md)
