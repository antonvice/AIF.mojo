# Changelog

All notable changes to AIF.mojo are documented here.

## [Unreleased]

- Replaced dense terminal-goal Loopy-BP temporary reductions with a reusable
  contiguous workspace, SIMD action blocks, and coarse CPU parallelism.
- Added `PreparedDenseLoopyBP` for cached static logs and repeated agent-step
  planning without workspace allocation.
- Updated the publication benchmark to compare default CPU worker pools.
- Added an optional, leakage-safe motor-imagery EEG research harness with
  synthetic smoke coverage, auditable pilot/tuning results, and an explicit
  separation from the native Mojo core.
- Added an optional surrogate-gradient Spiking-EEGNet training experiment with
  subject-separated epoch selection and focused PyTorch tests; no trained
  performance result is claimed yet.

## [0.1.0] - 2026-08-19

Initial public release.

- Eight native Active Inference and message-passing planners with dense and
  supported deterministic-sparse paths.
- Frozen Lake, Wumpus World, RockSample, and MiniGrid environment models and
  consolidated agent steps.
- Native VFE helpers and full convergence traces.
- 127 native tests plus message, planner, environment, agent, and convergence
  differential validation against a manifest-frozen JAX oracle.
- Visible Frozen Lake episode and precompiled Mojo package.
- Five-process JAX eager, JAX warm-JIT, and Mojo native benchmark across four
  state-space sizes.

[0.1.0]: https://github.com/antonvice/AIF.mojo/releases/tag/v0.1.0
