# JAX executable baseline

Captured: 2026-08-07
Reference checkout: `UAI-MP-AIF-JAX`
Reference commit: `30ee6f0ebce32c6a430fa7c25f1c01390415a797`
Backend: CPU

## Outcome

- Canonical tests: **235 passed**.
- Primary example smoke matrix: **32/32 passed**.
- Result artifacts: **32 valid JSON files**, with no `NaN`, `Infinity`, traceback, or error text.
- Repository-native `pytest` collection: **fails with two import errors**.
- Repository-native `run_tests.py`: **fails on stale RockSample test names** after its earlier suites pass.
- Repository-native `benchmark_planning.py`: **fails against the current MiniGrid tensor API**.
- Corrected bounded warm benchmark: completed for all eight planners.

The smoke runs prove that every planner can be constructed and executed through every primary environment CLI. They are deliberately too short to measure policy quality.

## Environment

| Component | Version/state |
|---|---|
| Python | 3.13.13 |
| JAX | 0.9.0 |
| jaxlib | 0.9.0 |
| NumPy | 2.4.2 |
| pytest | 9.0.2 |
| Gymnasium | 1.2.3 |
| MiniGrid | 3.0.0 |
| JAX device | `CpuDevice(id=0)` |
| `jax_enable_x64` | `False` |

## Tests

### Canonical suite

Command:

```bash
cd UAI-MP-AIF-JAX
/usr/bin/time -p .venv/bin/pytest -q \
  --ignore=reference \
  --ignore=tests/test_minigrid_groundtruth.py
```

Result:

```text
235 passed in 57.29s
real 58.75
user 162.58
sys 3.40
```

This is the authoritative green baseline until the repository's collection and custom-runner defects are repaired.

### Plain pytest collection

Command:

```bash
.venv/bin/pytest --collect-only -q
```

Result: exit code 2, 235 tests collected, two collection errors.

Both errors occur because `reference/minigrid.py` shadows the installed `minigrid` package while pytest collects:

- `reference/test_minigrid_fov.py`
- `tests/test_minigrid_groundtruth.py`

The stale reference module then fails with:

```text
ImportError: attempted relative import with no known parent package
```

### Custom runner

Command:

```bash
/usr/bin/time -p .venv/bin/python run_tests.py
```

Result: exit code 1 after 42.54 seconds. MiniGrid, inference, and the selected MiniGrid ground-truth scenarios pass. The run aborts at the start of RockSample tests with:

```text
AttributeError: 'TestRockSampleTensors' object has no attribute
'test_rock_quality_independence'
```

It also emits Gymnasium warnings that reset/step observations are outside the declared observation space.

## Primary example smoke matrix

The reproducible runner is [run_smoke_matrix.sh](run_smoke_matrix.sh). It executes all eight methods:

```text
loopy-vbp
loopy
region-extended
dyn-channel
nuijten
vbp-channel
precise-info-seeking
active-inference
```

against all four primary CLIs:

- Frozen Lake: grid 3, three configurations, one episode, maximum three steps.
- Wumpus World: grid 3, three configurations, one pit, one episode, maximum three steps.
- RockSample: grid 3, one rock, terminal-only goal, one episode, maximum three steps.
- MiniGrid: internal grid 3, FOV 3, one episode, maximum three steps.

All cases use seed 0, horizon 2, two planning iterations, receding horizon, and damping 0.5. MiniGrid also uses two state-inference iterations. Environment slip is disabled where available.

### Matrix result

| Environment | Passed | Failed | Process-time range |
|---|---:|---:|---:|
| Frozen Lake | 8 | 0 | 1.07-1.39s |
| Wumpus World | 8 | 0 | 1.13-1.40s |
| RockSample | 8 | 0 | 1.07-1.47s |
| MiniGrid | 8 | 0 | 1.02-2.42s |
| Total | 32 | 0 | 1.02-2.42s |

These process times include Python startup, tensor generation, and JAX compilation. They are not warm planner timings.

The complete per-case table is [examples/summary.tsv](examples/summary.tsv). Each row links to a console log and JSON result. All 32 JSON files parse successfully.

### Aggregate smoke behavior

- Frozen Lake: all eight methods ran three steps; none reached the goal in this deliberately short episode.
- Wumpus World: all eight methods ran three steps; none reached gold in this deliberately short episode.
- RockSample: all eight methods exited successfully with reward 10; Loopy VBP used three steps and the other methods used two.
- MiniGrid: all eight methods ran three steps; none solved DoorKey in this deliberately short episode.

Do not use these single-episode success rates to rank planners. The useful baseline is that construction, inference, planning, stepping, receding horizon, and JSON serialization all execute without errors.

### MiniGrid warnings

Every MiniGrid smoke run emits:

- an environment-registration override warning;
- reset observation outside declared observation space;
- step observation outside declared observation space.

These warnings do not change the exit code, but they should be resolved or explicitly accepted before MiniGrid observations become cross-language golden fixtures.

## Planning benchmark

### Repository benchmark status

The checked-in `benchmark_planning.py` exits before benchmarking:

```text
TypeError: generate_transition_tensor() missing 1 required positional
argument: 'valid_configs'
```

Its setup still uses an older MiniGrid tensor-generation signature and also assumes 64 static configurations for grid 4; the current generator returns 48 valid configurations.

### Corrected bounded benchmark

The external [benchmark_current.py](benchmark_current.py) uses current APIs without modifying the JAX checkout.

Command:

```bash
cd UAI-MP-AIF-JAX
PYTHONPATH=. /usr/bin/time -p .venv/bin/python \
  ../baseline/jax/2026-08-07/benchmark_current.py
```

Configuration:

```text
grid=4, states=192, static=48, actions=7
horizon=5, iterations=3, warm runs=10, backend=cpu
```

| Planner | First call / compile (ms) | Warm median (ms) | Warm IQR (ms) |
|---|---:|---:|---:|
| Loopy VBP | 449.649 | 163.949 | 4.673 |
| Loopy BP | 467.755 | 250.441 | 7.220 |
| Region Extended | 719.306 | 269.803 | 7.544 |
| Dynamic Channel | 817.295 | 332.924 | 14.289 |
| Nuijten MP | 1,165.745 | 783.182 | 74.455 |
| VBP Channel | 781.054 | 357.782 | 4.625 |
| Precise Information Seeking | 627.689 | 254.794 | 11.572 |
| Active Inference | 693.850 | 69.029 | 23.382 |

Whole benchmark process:

```text
real 32.25
user 116.98
sys 1.81
```

These numbers are suitable as a local CPU regression baseline for this exact machine and dependency lock. They are not portable performance claims.

## Artifact map

```text
baseline/jax/2026-08-07/
  BASELINE.md
  benchmark_current.py
  run_smoke_matrix.sh
  examples/
    summary.tsv
    logs/       32 command/output logs
    results/    32 JSON result files
```

The timing fields inside JSON and logs are expected to change on rerun. For Mojo correctness comparisons, compare configuration and semantic outputs while excluding wall-clock fields.

## What was not run

- The paper-scale DVC pipeline with hundreds or thousands of episodes.
- The 1,000-iteration convergence sweep.
- Video recording and plotting scripts.
- Cluster/SLURM scripts.
- Full diagnostics for every planner.

Those are expensive experiment/reproduction jobs rather than basic executable examples. They should be run after the test oracle is repaired and before claiming paper-result parity.
