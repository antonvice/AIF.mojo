# Mojo release and API review

Reviewed: 2026-08-19

The repository is already pinned to Mojo `1.0.0b2`, the current stable release
listed by the official release index at review time. An immediate compiler
upgrade is therefore unnecessary; nightly adoption would weaken the frozen
reproducibility contract.

## Adopted now

- Native measurements use `std.benchmark.run` with explicit warm-up and
  `std.benchmark.keep` so repeated planner work is measured in-process and its
  output remains observable.
- Packaging uses `--Werror --warn-on-unstable-apis`; examples use `--Werror`.
- Exact `List(capacity=...)` allocation was added to the measured dense
  Loopy-BP scratch paths. Native and differential tests guard the change.
- Tests retain `TestSuite.discover_tests()` plus `mojo run`, matching the
  current documented testing workflow.

## Deferred until a measured consumer exists

- `LayoutTensor`, `Span`, and a general tensor abstraction: current flat Lists
  expose axes clearly and the measured allocation fix did not require a new
  public type.
- SIMD `vectorize` and parallel/GPU kernels: profile the remaining inner
  reductions first, then add a backend-specific correctness and benchmark
  contract.
- Nightly compiler features: evaluate in a branch and run `check-all` plus the
  fair benchmark before changing `pixi.lock` or the manifest.

## Official references

- [Mojo releases](https://mojolang.org/releases/)
- [Mojo 1.0.0b2 changes](https://mojolang.org/releases/v1.0.0b2/)
- [`std.benchmark`](https://mojolang.org/docs/std/benchmark/)
- [`List` capacity and value semantics](https://mojolang.org/docs/std/collections/list/List/)
- [`mojo build` diagnostics](https://mojolang.org/docs/cli/build/)
- [Testing with `TestSuite`](https://mojolang.org/docs/tools/testing/)
