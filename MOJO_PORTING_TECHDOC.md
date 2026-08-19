# AIF-MOJO: technical design and validation history

Status: standalone native numerical core complete; Python validation adapters retained
Prepared: 2026-08-07
Updated: 2026-08-19
Reference implementation: `UAI-MP-AIF-JAX` at commit
`30ee6f0ebce32c6a430fa7c25f1c01390415a797`

## 1. Decision and outcome

AIF-MOJO is a standalone native, CPU-first implementation of active inference,
message-passing planners, and partially observable environment models. The JAX
research repository is retained only as a frozen behavioral oracle. AIF-MOJO
preserves its validated mathematical semantics without depending on the JAX API
or execution model at runtime.

The test-first sequence is now closed for the native-core scope:

1. Freeze and record an exact JAX oracle.
2. Pin the Mojo compiler and establish native and differential gates.
3. Port Float32 log-domain numerics, dense messages, and deterministic-sparse
   messages.
4. Port state inference and all eight retained planners.
5. Port all four environment models and consolidate shared agent behavior.
6. Port the complete VFE helper surface and instrumented convergence traces.
7. Lock the schema, policy-smoke, package, example, and benchmark boundaries.

Python remains by design for the frozen oracle, pytest driving, JSON
orchestration, DVC/YAML, plots, and Gymnasium/MiniGrid hosting and rendering.
Those are adapters around the native numerical core. Python does not implement
the policies produced by the Mojo experiment or benchmark runners.

## 2. Goals, achieved scope, and non-goals

### 2.1 Achieved native scope

- Preserve the canonical tensor axes, state encodings, action IDs, observation
  semantics, goal semantics, masks, damping, and smallest-index argmax rule.
- Provide all eight retained planning methods:
  - Loopy BP;
  - Loopy VBP;
  - Region-Extended Loopy BP;
  - Dynamic-Channel Loopy BP;
  - Nuijten MP;
  - VBP Channel;
  - Precise Information Seeking;
  - Active Inference.
- Preserve dense execution for all eight and deterministic-sparse execution
  where the JAX design supports it.
- Support terminal `(state)` goals and theta-dependent `(state, theta)`
  preferences.
- Provide native tensor/state models for Frozen Lake, Wumpus World, RockSample,
  and MiniGrid DoorKey.
- Provide pure draw-injected Frozen Lake and RockSample simulator steps with
  reward, termination, and truncation semantics.
- Provide consolidated belief updates and planner dispatch, including Frozen
  Lake all-eight dispatch and MiniGrid multimodal state inference across all
  eight dense and five supported sparse planner paths.
- Match the reference VFE formulas and 14 convergence trace groups.
- Emit reference-compatible experiment JSON and reproducible benchmark JSON.

### 2.2 Intentional non-goals

- Bitwise equality with JAX.
- Matching NumPy/JAX and Mojo random streams from the same seed.
- A general `NDArray`, broadcasting framework, arbitrary `einsum`, NumPy
  replacement, or autodiff system.
- GPU support, handwritten GPU kernels, or GPU parity claims.
- Native reimplementations of Gymnasium, MiniGrid rendering, Matplotlib, tqdm,
  YAML, DVC, plotting, or video tooling.
- Reproducing `jit`, `vmap`, `lax.scan`, and `.at[]` as framework abstractions.
- Treating process-launch timings as steady-state in-process planner latency.

These boundaries follow YAGNI. The current port has explicit, auditable loops
for every contraction it needs; adding a generic tensor or accelerator layer
without a measured consumer would increase semantic risk without completing a
missing user-facing contract.

## 3. Runtime and tensor contracts

### 3.1 Runtime flow

The common flow is:

```text
configuration -> native transition/observation/goal tensors
              -> belief reset or supplied beliefs
observation   -> native state/static-state update
              -> native finite-horizon planner
              -> normalized first-action distribution
              -> smallest-index argmax action
              -> environment adapter or deterministic tape
```

State inference and planning remain separate. Frozen Lake, Wumpus World, and
RockSample use dense categorical or binary Bayesian updates. MiniGrid uses
iterative multimodal state inference with a deterministic transition index,
vision observations, and a separate orientation tensor.

### 3.2 Canonical axes

| Symbol | Meaning | Canonical shape |
|---|---|---|
| `S` | dynamic states | scalar |
| `K` | static configurations, theta | scalar |
| `A` | actions | scalar |
| `F` | observation channels/FOV cells | scalar |
| `O` | outcomes per observation channel | scalar |
| `H` | planning horizon | scalar |
| `T` | dense transition probabilities | `(x_new=S, x_old=S, theta=K, action=A)` |
| `T_idx` | deterministic transition index | `(x_old=S, action=A, theta=K)` |
| `B` | observation probabilities | `(channel=F, outcome=O, state=S, theta=K)` |
| `q_x` | dynamic-state belief | `(S)` |
| `q_theta` | static-state belief | `(K)` |
| `C` | terminal goal or theta preference | `(S)` or `(S, K)` |
| `q_u` | action distribution by time | `(H, A)` |

The dense and sparse transition orders intentionally differ. This is recorded in
`tests/fixtures/manifest.json` and checked by `pixi run manifest`; callers must
not reinterpret a flat buffer from length alone.

MiniGrid additionally uses
`B_fov(fov_x, fov_y, cell_type, state, theta)` and
`B_orientation(orientation, state)`. Its FOV cells are flattened into the
canonical observation-channel axis at planner boundaries.

### 3.3 Environment state spaces

| Environment | Dynamic states | Static states | Actions | Observations |
|---|---:|---:|---:|---|
| Frozen Lake | `2 * grid^2` | configured hole layouts | 5 | `3 * grid^2` binary channels |
| Wumpus World | `2 * grid^2` | pit/wumpus/gold layouts | 5 | `3 + grid^2` binary channels |
| RockSample | `grid^2 * 2^rocks * (rocks + 2)` | `2^rocks` quality assignments | `rocks + 5` | `grid^2 + rocks` channels, 3 outcomes |
| MiniGrid | `grid^2 * 4 * 3` | valid `(key_pos, door_pos)` pairs | 7 | `fov^2` channels, 11 outcomes; orientation separate |

The doubled Frozen Lake and Wumpus spaces encode transient SCAN/SENSE modes.
RockSample includes the sampled-rock mask and last-event mode. MiniGrid includes
orientation and key/door state. These are inference states, not simulator-only
metadata.

### 3.4 Dense and sparse paths

| Planner | Dense | Deterministic sparse/precomputed |
|---|---:|---:|
| Loopy BP | yes | yes |
| Loopy VBP | yes | no |
| Region-Extended Loopy BP | yes | no |
| Dynamic-Channel Loopy BP | yes | yes |
| Nuijten MP | yes | yes |
| VBP Channel | yes | yes |
| Precise Information Seeking | yes | no |
| Active Inference | yes | yes, via precomputed `log_base` and `log_local_to_x` |

Deterministic sparse storage is invalid for stochastic slip unless generalized to
store multiple weighted outcomes. The Mojo environment modules therefore retain
dense stochastic transition generation and expose `T_idx` only for deterministic
cases.

Active Inference has two real execution contracts:

- the dense path updates theta-dependent dynamics and observation channels;
- the optimized path consumes a theta-marginalized dynamics base and a
  precomputed local state message. MiniGrid uses this path to avoid carrying its
  large theta dimension through the planning loop.

## 4. Implemented native surface

### 4.1 Numerical and planner modules

| Mojo module | Implemented responsibility |
|---|---|
| `numerics.mojo` | finite-zero log contract, stable reductions, softmax, normalization |
| `messages.mojo` | explicit dense 2-D/3-D/4-D contractions and marginalization |
| `sparse_messages.mojo` | deterministic reductions, theta messages, scatter-add pairs, channels, EFE priors |
| `state_inference.mojo` | dense and sparse state/static inference with vision and orientation |
| `loopy_bp.mojo` | dense/sparse planning for both goal forms |
| `loopy_vbp.mojo` | dense max/argmax value-BP planning for both goal forms |
| `region_extended_loopy_bp.mojo` | dense observation-factor planning and regions |
| `dyn_channel_loopy_bp.mojo` | dense/sparse dynamics/action-channel planning |
| `nuijten_mp.mojo` | dense/sparse region beliefs and EFE action priors |
| `vbp_channel.mojo` | dense/sparse conditional action-channel planning |
| `precise_info_seeking.mojo` | dense information-ratio and observation-channel planning |
| `active_inference.mojo` | dynamics kernels, observation/preference precompute, dense and optimized planning |

Action masks, iteration changes, damping, channel normalization, terminal goals,
theta preferences, and environment-derived fixtures are covered at the module
boundaries rather than inferred only from episode behavior.

### 4.2 Environment modules

| Module | Native coverage |
|---|---|
| `frozen_lake.mojo` | state encoding, deterministic transition, dense/sparse tensors, observation sampling from supplied draws, theta goal, reward/terminal/truncation simulator step |
| `wumpus_world.mojo` | state encoding, sensing/hazards, transition, observation sampling from supplied draws, goal, reward, terminal |
| `rocksample.mojo` | position/mask/event encoding, sensing accuracy, dense/sparse transition, observation sampling from supplied draws, quality configurations, goal, reward/terminal/truncation simulator step |
| `minigrid.mojo` | coordinates/configs, FOV/visibility, orientation/key/door/movement rules, `T_idx`, tiny dense transition, hard/soft observations, orientation tensor, conversions, goal |

Each environment has an explicit deterministic tape in the native tests and in
the reproducibility manifest. Random stream equality is not used as an oracle.

### 4.3 Consolidated agent module

`agent.mojo` replaces the JAX class-per-planner duplication with one eight-kind
dispatcher plus shared inference helpers. The current native agent surface
contains:

- dense categorical and binary belief updates;
- action-prior handling, observation-channel slicing, goal marginalization, and
  smallest-index argmax;
- generic dense dispatch for all eight planners;
- generic deterministic-sparse dispatch for Loopy BP, Dynamic-Channel,
  Nuijten, VBP Channel, and precomputed Active Inference;
- Wumpus reset/step and RockSample reset/step functions;
- Frozen Lake reset and generic all-eight-planner step;
- MiniGrid reset, probabilistic dense/sparse multimodal belief updates, generic
  all-eight dense step, and generic five-method sparse step.

Full live episode loops, seeded environment sampling, Gym reset/step translation,
rendering, and video remain Python adapters. This avoids duplicating mature host
libraries while keeping belief updates and planning native.

### 4.4 VFE formulas and convergence traces

`convergence.mojo` ports normalized and unnormalized entropy, factor energy,
conditional action/dynamics/observation entropy, and all reference VFE formulas:

- Bethe/Loopy BP;
- Region-Extended;
- Dynamic-Channel;
- VBP Channel;
- Precise Information Seeking;
- Nuijten MP;
- Active Inference.

The same seven reference convergence planners expose instrumented
`planning_with_vfe` traces. Terminal and theta-dependent goal modes are covered
for each, producing 14 differential trace groups. Loopy VBP is one of the eight
native planners, but the frozen reference does not expose a corresponding
convergence API, so it is not an omitted trace port.

`pixi run parity-convergence` compares the complete trace vectors and VFE helper
fixtures against the frozen JAX functions. The focused native convergence suite
also covers masks and the flattened `K=1` goal-shape ambiguity.
The convergence-focused suite and all seven affected planner regression suites
have been rerun successfully; no convergence gate remains pending.

## 5. Oracle, fixtures, and comparison contract

### 5.1 Frozen reference

The oracle is unchanged at
`30ee6f0ebce32c6a430fa7c25f1c01390415a797`. The reproducibility manifest locks:

| Component | Locked state |
|---|---|
| Python | 3.13.13 |
| JAX / jaxlib | 0.9.0 / 0.9.0 |
| NumPy | 2.4.2 |
| JAX backend | CPU, `jax_enable_x64=False` |
| Mojo | 1.0.0b2 |
| Production tensor dtype | Float32 |

The canonical reference command remains:

```bash
cd UAI-MP-AIF-JAX
uv run pytest -q --ignore=reference --ignore=tests/test_minigrid_groundtruth.py
```

The exclusions are intentional and recorded in the original baseline: the
legacy `reference/` directory shadows the installed MiniGrid package under plain
collection, `run_tests.py` contains stale RockSample method names, and the
scenario-argument MiniGrid ground-truth functions are not ordinary pytest tests.
The Mojo port did not repair or commit into the nested reference checkout.

`scripts/bootstrap_oracle.py` clones the repository only when absent. If the
path already exists it must be a clean Git checkout at the exact SHA; wrong or
dirty state is rejected instead of repaired silently. `check_manifest.py`
independently verifies SHA, cleanliness, tool versions, dtype/x64, axes,
tolerances, paths, goals, environment action cardinalities, and tapes.

### 5.2 Three test layers

1. Native Mojo unit tests cover numerical edge cases, indexing, normalization,
   planners, environments, agents, and convergence formulas.
2. Python-driven differential executables evaluate frozen JAX and native Mojo
   on the same explicit tensors and compare structured output.
3. Invariants cover stochastic tables, conditional normalization, masks,
   terminal behavior, and deterministic episode tapes where exact samples would
   be the wrong contract.

Python is only the cross-language driver in differential tests; planner
arithmetic remains in Mojo.

### 5.3 Numerical tolerances

| Output | Comparison |
|---|---|
| Shapes, IDs, indices, masks, terminal flags | exact |
| Deterministic transition/observation entries | exact |
| Float32 probabilities and log messages | `atol=1e-5`, `rtol=1e-5` |
| Region/channel normalization | `atol=1e-4` |
| MiniGrid Float16-derived reference tensors | test-specific `1e-3` to `1e-2` |
| Probability sums | environment `1e-6`; inference `1e-5` unless declared otherwise |
| Masked action probability | `< 1e-6` |

Exact argmax is required only when the reference top-two margin exceeds `1e-4`.
Otherwise the distribution is compared and both implementations must retain the
smallest-index tie rule.

### 5.4 Reference inconsistency: RockSample Loopy BP

The frozen JAX `RockSampleLoopyBPAgent` uniquely bypasses its `_planning_goal`
when `terminal_goal_only=True`; the other seven RockSample planner agents use the
marginalized terminal goal. The consolidated Mojo implementation applies the
documented behavior uniformly to all eight methods. A dedicated regression test
locks this intentional consistency fix.

The oracle was not edited. This narrow divergence is a JAX pull-request
candidate, not a reason to reproduce class-specific drift in the consolidated
Mojo dispatcher.

## 6. Mojo architecture and ownership

### 6.1 Actual repository layout

```text
AIF-MOJO/
  pixi.toml
  pixi.lock
  src/aif_mojo/
    numerics.mojo
    messages.mojo
    sparse_messages.mojo
    state_inference.mojo
    loopy_bp.mojo
    loopy_vbp.mojo
    region_extended_loopy_bp.mojo
    dyn_channel_loopy_bp.mojo
    nuijten_mp.mojo
    vbp_channel.mojo
    precise_info_seeking.mojo
    active_inference.mojo
    convergence.mojo
    frozen_lake.mojo
    wumpus_world.mojo
    rocksample.mojo
    minigrid.mojo
    agent.mojo
  tests/mojo/
  tests/parity/
  tests/fixtures/manifest.json
  examples/
  scripts/
  benchmarks/
  UAI-MP-AIF-JAX/              # ignored frozen oracle checkout
```

The planner files remain flat in one Mojo package. A nested planner/environment
package hierarchy would add relocation work without changing the public
contracts.

### 6.2 Flat Lists are intentional

The correctness implementation uses owned, contiguous row-major
`List[Float32]` and `List[Int]` values plus explicit named dimensions and offset
helpers. Public functions validate shapes; inner loops use direct indexing.

This is the final native-core design, not an unfinished placeholder. No current
kernel needs general slicing, broadcasting, rank-polymorphic views, or arbitrary
strides. A small layout type or `LayoutTensor` should be introduced only if a
measured hotspot needs shared view logic. Until then, flat buffers make axis
transposes, copies, and scatter-add semantics visible during review.

The implementation follows these ownership rules:

- Float32 for reference-compatible production probability/log tensors;
- `Int` for host indices and deterministic transition destinations;
- read-only input borrowing where possible;
- explicit copies when a planner iteration needs independent state;
- scratch reuse and no intentionally hidden full-tensor copies in inner loops.

### 6.3 JAX-to-Mojo mapping

| JAX construct | Mojo implementation |
|---|---|
| `@jax.jit` | ordinary compiled Mojo function |
| static horizon/iterations | runtime loops |
| `lax.fori_loop` / reverse `scan` | explicit forward/backward loops and buffers |
| `vmap` | explicit outer loops |
| fixed `einsum` signatures | named nested contractions |
| `.at[idx].add` | explicit scatter-add |
| `.at[idx].set` | indexed assignment only for unique destinations |
| transpose/broadcast | explicit offsets or tiling helpers |
| `jax.nn.softmax` | project stable softmax |

The absence of JIT static arguments simplifies correctness. Specialization is a
future performance option, not part of semantic parity.

## 7. Result schemas, experiment smoke, examples, and packaging

### 7.1 Exact result schemas

`run_experiment_matrix.py` defines and validates the exact `config` and `results`
field sets emitted by the four reference experiment scripts. Golden schema cases
cover Frozen Lake, Wumpus World, RockSample, and MiniGrid. Unknown fields and
Python type ambiguities such as `bool` masquerading as `int` are rejected.

### 7.2 Native 8-by-4 policy smoke matrix

`pixi run experiment-smoke` compiles `native_experiment_matrix.mojo`, invokes
native numerical inference, and crosses all eight planner names with four native
environment fixtures:

| Fixture | Generator | States | Static | Actions |
|---|---|---:|---:|---:|
| Frozen Lake 2x2 | `generate_frozen_lake_transition` | 8 | 2 | 5 |
| Wumpus World 2x2 | `generate_wumpus_transition` | 8 | 2 | 5 |
| RockSample 2x2, one rock | `generate_rocksample_transition` | 24 | 2 | 6 |
| MiniGrid 3x3 | `generate_minigrid_transition_tensor` | 108 | 9 | 7 |

Python launches the compiled binary, validates normalization/cardinality, and
writes JSON; it does not calculate policies. The output contains 32
reference-shaped result files and `_smoke_matrix.json`.

This boundary is deliberately **policy-only**. Every result config records
`n_episodes=0`, and success, step, and reward fields remain zero. These files
prove native planner invocation and schema compatibility, not episode quality or
paper reproduction.

### 7.3 Benchmark contract

`pixi run benchmark` performs a like-for-like CPU comparison of the dense
terminal-goal Loopy-BP planner in Mojo native, JAX eager, and JAX warm-JIT modes.
All modes consume the same generated Float32 tensors and use `K=2`, `A=4`,
`H=3`, three planning iterations, and the same normalized policy oracle. Two
state sizes are measured: `S=8` and `S=64`.

Mojo AOT compilation and JAX lowering/compilation are recorded separately.
Each runtime then performs three warm-up calls followed by adaptive repeated
calls inside one process. JAX calls synchronize with `block_until_ready`; Mojo
uses `std.benchmark.run` with `max_batch_size=1` and keeps an output live to
prevent dead-code elimination. Results include mean/min/max latency,
calls/second, measurement duration and count, warm-up duration, and complete
process peak RSS. Median and p95 are additionally available for the individually
timed JAX calls; the pinned Mojo benchmark API reports aggregate batches.

`pixi run benchmark-process` retains the previous eight-planner plus bounded
3x3 MiniGrid Active runner. Its process-wall samples include startup and remain
useful for launch-time regression tracking, but they are not mixed with the fair
steady-state comparison.

### 7.4 Examples and package smoke

- `example-loopy` exercises the public deterministic-sparse Loopy BP API.
- `example-all` exercises the consolidated eight-planner dispatcher.
- `example-frozen` renders a full native Frozen Lake belief-plan-step episode.
- `package-smoke` precompiles `aif_mojo.mojoc` and reruns the public Loopy example
  against the packaged artifact.

## 8. Milestone closure

| Milestone | Status | Closure evidence |
|---|---|---|
| M0: freeze oracle | complete | exact SHA, clean-check/refusal bootstrap, manifest, canonical reference command |
| M1: scaffold/numerics | complete | pinned Mojo via Pixi, package, stable Float32 numerics, native tests |
| M2: dense/sparse messages | complete | full retained message surface and differential/invariant coverage |
| M3: first planner | complete | Loopy BP dense/sparse, both goals, horizons, iterations, masks |
| M4: all planners | complete | all eight dense; supported sparse/precomputed paths; planner parity gates |
| M5: environments/agents | complete for native-core scope | four native models and tapes; pure Frozen/Rock simulators; shared belief updates; Frozen all-eight and MiniGrid dense-eight/sparse-five dispatch; Python host adapters retained |
| M6: convergence/schema/performance boundary | complete | seven VFE formulas, 14 traces, exact schemas, policy matrix, package smoke, fair eager/JIT/native benchmark |

The original plan proposed a general `NDArray` in M1. Implementation evidence
showed it was unnecessary, so milestone closure uses flat Lists instead. This is
a documented YAGNI refinement, not missing work.

Likewise, M5 does not require replacing Gymnasium or the research repository's
DVC/plot/video layer. The porting goal is native inference and environment
semantics with adapters at the system boundary, not a language-purity rewrite.

## 9. High-risk semantic details

### 9.1 Numerical support

- Preserve `LOG_ZERO = -1e12`; negative infinity changes `0/0` handling and can
  introduce NaNs.
- `safe_log_div` maps both `0/0` and nonzero-over-zero to impossible support.
- Log-domain channel damping is a geometric mixture followed by normalization
  along the exact conditional axis, not a linear probability mixture.
- Reduction order differs from JAX; declared tolerances are part of the API.

### 9.2 Goals and observation selection

- A one-dimensional goal is a terminal-state distribution.
- A `(state, theta)` preference is marginalized with theta beliefs and injected
  according to planner semantics; it is not interchangeable with a terminal
  goal.
- RockSample planning sees only its theta-dependent rock channels, while full
  position and rock observations update state beliefs.
- `terminal_goal_only` must be applied before dispatch for every RockSample
  planner, including Loopy BP in the consistency-fixed Mojo behavior.

### 9.3 Sparse transitions

- `T` and `T_idx` have different axes.
- Sparse state prediction uses scatter-add because multiple old-state/theta
  pairs can reach one new state.
- A deterministic index cannot represent nonzero slip probabilities.
- Large MiniGrid execution must avoid materializing dense theta-dependent
  planning carries; use the supported sparse/precomputed path.

### 9.4 Randomness and episodes

- Equal integer seeds do not imply equal NumPy and Mojo random sequences.
- Generated probability tables are compared separately from samples.
- End-to-end semantic tests inject explicit action/outcome tapes.
- Native RNG reproducibility, if added later, is a new contract rather than a
  prerequisite for this port.

## 10. Performance policy

Correctness comes before optimization. Current code uses contiguous Float32
buffers, explicit loops, stable reductions, and deterministic sparse transitions
where valid.

The fair benchmark now isolates compilation and warm-up from repeated
in-process calls, compares two state-space sizes, and records latency,
throughput, and peak RSS. The latest local artifact is
`benchmarks/results/fair_latest.json`; results remain machine-specific. On the
development Mac, warm-JIT JAX wins latency at both sizes, Mojo beats eager JAX
and has much lower process peak RSS. This supports workload-specific choices,
not a general language ranking.

The first measured optimization was exact-capacity preallocation for temporary
Lists in the dense Loopy-BP hot path. It preserves the public flat-buffer model,
passes the differential gates, and avoids repeated geometric growth. Further
work follows measured hotspots in this order:

If performance work is requested, optimize in this order:

1. remove measured full-tensor copies and allocations;
2. improve contiguous iteration order;
3. cache theta-marginalized quantities;
4. vectorize proven inner reductions;
5. introduce a narrow layout/view type for a demonstrated hotspot;
6. add parallel CPU or GPU kernels only with new correctness and benchmark
   contracts.

The release/API review is recorded in `MOJO_UPGRADE_NOTES.md`. No GPU claim is
made by this CPU benchmark.

## 11. Toolchain and exact commands

Mojo is pinned to `1.0.0b2` by `pixi.toml` and `pixi.lock`. Mojo packages are
compiler-version-sensitive, so compiler updates require a deliberate full gate.

### 11.1 Bootstrap and oracle lock

```bash
pixi install
pixi run bootstrap-oracle
pixi run sync-oracle
pixi run manifest
```

### 11.2 Native and differential gates

```bash
pixi run test
pixi run parity
pixi run parity-env
pixi run parity-convergence
pixi run test-schema
```

### 11.3 Runnable artifacts

```bash
pixi run experiment-smoke
pixi run benchmark
pixi run benchmark-process
pixi run package-smoke
pixi run example-loopy
pixi run example-all
pixi run example-frozen
```

### 11.4 Aggregate gates

```bash
pixi run check
pixi run test-ref
pixi run check-all
```

`check` depends on the manifest, schema, native unit, message/planner parity,
environment parity, convergence parity, experiment-smoke, and package-smoke
gates. The benchmark is intentionally separate because timing variability should
not fail correctness CI. `check-all` adds the canonical frozen JAX suite.

The standard generated outputs are:

- `aif_mojo.mojoc` from packaging;
- `data/smoke_matrix/` from the policy matrix;
- `benchmarks/results/fair_latest.json` from the fair comparison;
- `benchmarks/results/latest.json` from the legacy process benchmark.

## 12. Definition of done

The native-core port is complete when all of the following remain true:

- the nested JAX oracle is clean and at the manifest SHA;
- the pinned toolchain, dtype/x64 setting, axes, goal forms, tolerances, paths,
  and tapes pass manifest validation;
- native unit and differential gates cover the retained message surface, all
  eight planners, all four environment models, consolidated agent slices, and
  the seven reference VFE/convergence planners;
- supported dense/sparse pairs agree within their contracts;
- Active Inference dense and optimized paths are covered, including MiniGrid's
  sparse precomputed path, while MiniGrid also covers the other four supported
  sparse planners and all eight dense planners;
- deterministic environment tapes pass without cross-runtime RNG assumptions;
- the 14 terminal/theta convergence traces match the frozen reference within
  declared tolerances;
- all four exact experiment schemas pass golden tests;
- the 8-by-4 matrix invokes native policies and labels its zero-episode scope;
- the precompiled package and all three examples run through public APIs;
- the fair benchmark separates compilation, warm-up, and repeated in-process
  measurements and validates identical policies;
- no Python/JAX inference runs inside the Mojo numerical core.

The following are intentionally retained boundaries, not incomplete core-port
items:

- Python/JAX oracle and pytest orchestration;
- Python DVC/YAML, plots, Gymnasium/MiniGrid hosting, rendering, and video;
- reference-compatible full episode drivers and seeded random environment
  selection in Python;
- flat row-major Lists instead of a general NDArray;
- no GPU backend and no NumPy/Mojo RNG-stream parity;
- no episode-quality claim from `experiment-smoke`;
- machine-specific benchmark results rather than universal backend claims.

Future work may replace an adapter, add native episode loops, introduce a narrow
layout optimization, or propose the RockSample Loopy BP fix upstream. Those are
new extensions. They are not required to call the native numerical-core port
complete.
