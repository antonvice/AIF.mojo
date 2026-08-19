# Motor-imagery EEG benchmark

This harness compares logistic regression, a current-coded LIF reservoir, and a
sequential Active Inference decoder under leave-one-subject-out evaluation.

The AIF decoder borrows the belief-plan-act semantics of UAI-MP-AIF-JAX. It is
not a direct port of that repository's gridworld factor graphs. Gaussian EEG
likelihoods update a posterior over left/right intent; the agent chooses among
`wait`, `choose-left`, and `choose-right` by comparing expected decision risk.

## Smoke benchmark

```bash
pixi run test-eeg
pixi run eeg-smoke
```

The default generator is a deterministic motor-imagery-like surrogate. Its
results validate code paths only, not physiological EEG or BCI performance.

## Real EEG data

Download and epoch the public [PhysioNet EEG Motor Movement/Imagery dataset](https://physionet.org/content/eegmmidb/1.0.0/)
(runs 4, 8, and 12 are left/right fist imagery):

```bash
uv run --with mne python benchmarks/eeg_motor_imagery/prepare_eegbci.py \
  --subjects 1 2 3 4 5 \
  --output data/eegbci_left_right.npz

UAI-MP-AIF-JAX/.venv/bin/python -m benchmarks.eeg_motor_imagery.benchmark \
  --data data/eegbci_left_right.npz \
  --output benchmarks/results/eegbci_left_right.json
```

For another dataset, provide an NPZ containing `X[trial, channel, time]`, binary
`y[trial]`, `subject[trial]`, and optional scalar `fs` (128 Hz by default):

```bash
UAI-MP-AIF-JAX/.venv/bin/python -m benchmarks.eeg_motor_imagery.benchmark \
  --data /absolute/path/motor_imagery.npz \
  --output benchmarks/results/eeg_real.json
```

All feature scaling and fitting happen inside each subject-wise fold. Balanced
accuracy and macro F1 are the primary metrics. Decision windows measure latency;
SNN spike rate is an activity proxy, not hardware energy consumption.

## Five-subject pilot result

The checked local run used 225 trials from subjects 1-5, three channels (C3,
Cz, C4), and four 500 ms windows. These are pilot results, not a full-dataset
claim.

| Model | Balanced accuracy | Macro F1 | Log loss | Mean decision window |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.601 | 0.576 | 0.689 | 4.00 |
| LIF SNN | 0.600 | 0.571 | 0.738 | 4.00 |
| Sequential AIF | 0.551 | 0.447 | 0.816 | 1.53 |

The SNN tied the linear baseline on accuracy but was less calibrated. The AIF
decoder committed much earlier and lost accuracy. Sweeping its wait cost did
not reverse that result, so the benchmark retains the predeclared value. The
full fold-level output is in
[`benchmarks/results/eegbci_left_right.json`](../results/eegbci_left_right.json).

## Tune the LIF reservoir

```bash
pixi run eeg-prepare-20
pixi run eeg-tune
```

The tuning protocol keeps subjects 11-20 untouched. Optuna TPE searches 100
configurations using leave-one-subject-out validation on subjects 1-10 and
three fixed reservoir seeds. It tunes temporal windows, reservoir size, ticks,
decay, threshold, input gain, recurrent spectral radius and sparsity, ridge
regularization, reset behavior, and probability temperature.

Trials with mean firing rates below 0.01 or above 0.60 are pruned. The selected
configuration is frozen before the default and tuned three-seed ensembles are
evaluated on subjects 11-20. The objective is balanced accuracy minus 0.01 times
log loss; test-subject metrics never influence selection.

### 100-trial result

The completed search took 607 seconds: 95 trials completed and five infeasible
firing-rate configurations were pruned. The selected reservoir used four
windows, 96 neurons, 24 ticks, decay 0.936, threshold 0.847, spectral radius
0.539, recurrent sparsity 0.185, zero reset, and mean test spike rate 0.181.

| Untouched subjects 11-20 | Balanced accuracy | Macro F1 | Log loss | Spike rate |
| --- | ---: | ---: | ---: | ---: |
| Default three-seed ensemble | 0.5244 | 0.5214 | 0.6972 | 0.3013 |
| Tuned three-seed ensemble | 0.5333 | 0.5310 | 0.6897 | 0.1810 |

Tuning reduced spike activity by 39.9% and improved six of ten test subjects.
The balanced-accuracy gain was only 0.9 percentage points; its paired
subject-bootstrap 95% interval was -4.0 to +5.3 points. The accuracy result is
therefore inconclusive, while the activity reduction is clear for this model
and cohort. The full protocol, selected configuration, per-subject results, and
paired analysis are in
[`benchmarks/results/eeg_lif_tuning.json`](../results/eeg_lif_tuning.json).

The raw PhysioNet epochs are intentionally not redistributed. Recreate them
from EEGMMIDB 1.0.0 with the preparation command above.

## Trainable Spiking EEGNet

```bash
pixi run eeg-prepare-30
pixi run test-eeg-trainable
pixi run eeg-train-snn
```

The trainable model uses an EEGNet-style temporal/spatial convolutional front
end, a 64-neuron LIF layer, a fast-sigmoid surrogate derivative, learnable
membrane decay and threshold, AdamW, and spike-rate regularization. Subjects
1-16 train epoch selection, subjects 17-20 provide validation, and the selected
epoch count is used to refit subjects 1-20. Newly downloaded subjects 21-30 are
evaluated only after all three seeds are trained.

The output includes three model checkpoints, complete validation histories,
per-subject metrics, and paired subject bootstraps against both baselines.

### Fresh-cohort result

The three seeds selected 47, 26, and 23 epochs using subjects 17-20, then were
refit on all 900 development trials. Subjects 21-30 were evaluated once.

| Model | Balanced accuracy | Macro F1 | Log loss | Spike rate |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.5319 | 0.5299 | 0.6883 | n/a |
| Tuned LIF reservoir | 0.5452 | 0.5435 | 0.6892 | 0.1442 |
| Spiking EEGNet | **0.6052** | **0.6035** | **0.6737** | **0.0966** |

Spiking EEGNet improved eight of ten subjects versus logistic regression. Its
mean subject balanced-accuracy gain was 7.1 points with a paired bootstrap 95%
interval of +2.4 to +12.3 points. It improved nine of ten subjects versus the
tuned reservoir, with a mean gain of 5.6 points and interval of +2.7 to +8.7.
