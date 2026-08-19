# API stability

AIF.mojo follows semantic versioning from `v0.1.0` onward.

- Public functions imported by the examples and documented in
  `TECHNICAL_DESIGN.md` are the supported surface.
- `PreparedDenseLoopyBP`, its `plan()` method, and the dense workspace sizing
  and prelogged entry points follow the same pre-1.0 compatibility policy.
- Flat-buffer axis order, planner IDs, action IDs, return layouts, and explicit
  terminal/theta goal selection are compatibility contracts.
- Names beginning with `_`, diagnostic trace packing, benchmark schemas, and
  Python oracle adapters may change between minor `0.x` releases.
- Mojo `.mojoc` packages are compiler-version-sensitive. Release artifacts name
  their Mojo version, operating system, and architecture.

Before `1.0`, a breaking public change requires a changelog entry, a migration
note, and a minor-version increment.
