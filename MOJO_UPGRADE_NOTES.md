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
- Dense terminal-goal Loopy BP now uses a reusable contiguous workspace,
  allocation-free streaming reductions, four-lane action SIMD with scalar
  tails, and coarse CPU parallelism for independent rows at 32 or more states.
- `PreparedDenseLoopyBP` caches static log tensors and the workspace across
  agent steps. The original probability-space API delegates to the same fast
  kernel.
- Tests retain `TestSuite.discover_tests()` plus `mojo run`, matching the
  current documented testing workflow.

## Deferred until a measured consumer exists

- `LayoutTensor`, `Span`, and a general tensor abstraction: flat Lists plus
  `unsafe_ptr()` provide the measured SIMD path without expanding the public
  tensor surface.
- Fine-grained nested parallel forward/backward rows: Mojo `1.0.0b2` crashed
  when parameter closures captured loop-local time indices, so only stable
  coarse-grained closures are retained.
- Theta-goal workspace and GPU kernels: add only with their own differential
  and benchmark contracts.
- Nightly compiler features: evaluate in a branch and run `check-all` plus the
  fair benchmark before changing `pixi.lock` or the manifest.

## Official references

- [Mojo releases](https://mojolang.org/releases/)
- [Mojo 1.0.0b2 changes](https://mojolang.org/releases/v1.0.0b2/)
- [`std.benchmark`](https://mojolang.org/docs/std/benchmark/)
- [`List` capacity and value semantics](https://mojolang.org/docs/std/collections/list/List/)
- [`mojo build` diagnostics](https://mojolang.org/docs/cli/build/)
- [Testing with `TestSuite`](https://mojolang.org/docs/tools/testing/)
