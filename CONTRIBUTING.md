# Contributing

Thank you for improving AIF.mojo. Correct tensor semantics and reproducible
behavior take priority over API breadth.

## Set up

```bash
pixi install
pixi run bootstrap-oracle
pixi run sync-oracle
pixi run manifest
```

The nested JAX checkout is a read-only behavioral oracle. Do not commit changes
inside it.

## Before opening a pull request

```bash
pixi run check
```

Run `pixi run check-all` when changing tensor semantics, environment behavior,
or oracle fixtures. Benchmarks are evidence, not correctness gates; include the
before/after JSON when claiming a performance improvement.

Keep changes narrow, document every flat-buffer axis, preserve Float32 behavior,
and add a native test plus a differential or invariant test for new numerical
paths. See [TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md) and
[API stability](docs/API_STABILITY.md).

By contributing, you agree that your contribution is licensed under
Apache-2.0.
