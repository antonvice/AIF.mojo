#!/usr/bin/env python3
"""Build the locked 109-subject Kaggle benchmark notebook."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parent


def cell(kind: str, source: str) -> dict:
    value = {
        "cell_type": kind,
        "id": hashlib.sha1(source.encode()).hexdigest()[:8],
        "metadata": {},
        "source": source.splitlines(True),
    }
    if kind == "code":
        value |= {"execution_count": None, "outputs": []}
    return value


cells = [
    cell(
        "markdown",
        """# Locked SNN benchmark on 109 EEGBCI subjects

This notebook asks a stricter question than the earlier 30-subject pilot: does a tuned Spiking EEGNet still help when compared with established motor-imagery baselines and evaluated on genuinely new people?

Five models share the same EEG and subject split:

- windowed bandpower logistic regression;
- filter-bank common spatial patterns (FBCSP);
- Riemannian tangent-space logistic regression;
- parameter-matched analog EEGNet;
- Spiking EEGNet.

The protocol is fail-closed. Hyperparameter search can only access subjects 1-20. The resulting configuration and per-seed epoch counts are SHA-256 locked before subjects 31-50 are evaluated. Subjects 21-30 are excluded because they were already observed in the previous experiment. Subjects 31-109 are then reported without further tuning.

Methods follow the [MNE CSP implementation](https://mne.tools/stable/generated/mne.decoding.CSP.html), [pyRiemann tangent-space classifiers](https://pyriemann.readthedocs.io/), and [EEGNet](https://arxiv.org/abs/1611.08024).
""",
    ),
    cell(
        "code",
        """import json, os, shutil, subprocess, sys
from pathlib import Path

WORK = Path('/kaggle/working')
REPO = WORK / 'AIF-MOJO'
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run([
    'git', 'clone', '--depth', '1', '--branch', 'null/eeg-research-harness',
    'https://github.com/antonvice/AIF.mojo.git', str(REPO)
], check=True)
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'mne>=1.10,<1.13', 'scikit-learn>=1.7,<1.9', 'pyriemann==0.12',
    'optuna>=4.5,<5', 'rich>=14,<15', 'nvidia-ml-py>=13,<14'
], check=True)
gpu_name = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True
).strip()
if 'P100' in gpu_name:
    print('P100 detected: installing PyTorch 2.7.1 CUDA 11.8 for sm_60 support')
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall',
        'torch==2.7.1', '--index-url', 'https://download.pytorch.org/whl/cu118'
    ], check=True)
os.chdir(REPO)
print('Source commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
""",
    ),
    cell(
        "code",
        """import torch

assert torch.cuda.is_available(), 'This benchmark requires a Kaggle GPU session'
print('PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0))
print('Capability:', torch.cuda.get_device_capability(0))
print('Compiled architectures:', torch.cuda.get_arch_list())
print('CUDA probe:', (torch.ones(1, device='cuda') * 2).item())
""",
    ),
    cell(
        "markdown",
        """## Recreate the data

EEGBCI runs 4, 8, and 12 provide left/right fist motor imagery. Each subject contributes 45 two-second trials. We keep C3, Cz, and C4, resample to 128 Hz, and download independent subjects concurrently. Raw data comes directly from PhysioNet and is not republished by this notebook.
""",
    ),
    cell(
        "code",
        """data_path = WORK / 'eegbci_left_right_109.npz'
os.environ['MNE_DATASETS_EEGBCI_PATH'] = str(WORK / 'mne_data')
subprocess.run([
    sys.executable, 'benchmarks/eeg_motor_imagery/prepare_eegbci.py',
    '--subject-range', '1-109', '--jobs', '4', '--output', str(data_path)
], check=True)
""",
    ),
    cell(
        "markdown",
        """## Tune on subjects 1-20, then lock

Optuna TPE searches temporal filters, depth multiplier, hidden width, temporal kernel, pooling, LIF decay and threshold, surrogate slope, learning rate, weight decay, batch size, spike-activity target and penalty, and input noise. Trials train on subjects 1-16 and validate on 17-20. No array from subject 21 or later is preprocessed by the tuner.

After selecting the configuration, three fixed seeds choose epoch counts for both the spiking and analog networks using the same development split. The complete protocol becomes immutable once its SHA-256 lock is written.
""",
    ),
    cell(
        "code",
        """lock_path = WORK / 'eeg_snn_lock.json'
subprocess.run([
    sys.executable, '-m', 'benchmarks.eeg_motor_imagery.tune_locked_snn',
    '--data', str(data_path), '--trials', '30', '--trial-epochs', '35',
    '--lock-epochs', '60', '--device', 'cuda', '--output', str(lock_path)
], check=True)
lock = json.loads(lock_path.read_text())
print('LOCK SHA-256:', lock['lock_sha256'])
print('Configuration:', json.dumps(lock['config'], indent=2))
print('Selected epochs:', json.dumps(lock['selected_epochs'], indent=2))
""",
    ),
    cell(
        "markdown",
        """## One fresh evaluation, then fixed expansion

The evaluator verifies the lock, trains every model only on subjects 1-20, and first computes the primary result on subjects 31-50. It then applies those same fitted models and configuration to subjects 31-109. Existing result receipts cannot be overwritten.

Latency is measured after warm-up at batch sizes 1 and 64. For analog and spiking networks, NVML samples real GPU board power every 20 ms during repeated inference. Gross and idle-subtracted joules per trial are reported separately. These measurements describe dense PyTorch on this GPU, not neuromorphic hardware.
""",
    ),
    cell(
        "code",
        """result_path = WORK / 'eeg_locked_31_109.json'
subprocess.run([
    sys.executable, '-m', 'benchmarks.eeg_motor_imagery.evaluate_locked',
    '--data', str(data_path), '--lock', str(lock_path), '--device', 'cuda',
    '--energy-seconds', '5', '--output', str(result_path)
], check=True)
result = json.loads(result_path.read_text())
assert result['protocol']['lock_sha256'] == lock['lock_sha256']
""",
    ),
    cell(
        "code",
        """import pandas as pd
from IPython.display import display

def metrics_table(cohort):
    rows = []
    for name, model in result[cohort]['models'].items():
        row = {'model': name} | model['aggregate']
        rows.append(row)
    columns = ['model', 'balanced_accuracy', 'macro_f1', 'log_loss', 'brier', 'mean_spike_rate']
    return pd.DataFrame(rows).reindex(columns=columns).set_index('model')

print('PRIMARY FRESH COHORT: SUBJECTS 31-50')
display(metrics_table('primary_fresh_evaluation').style.format('{:.4f}', na_rep='n/a'))
print('FIXED-CONFIG EXPANSION: SUBJECTS 31-109')
display(metrics_table('fixed_config_expansion').style.format('{:.4f}', na_rep='n/a'))
""",
    ),
    cell(
        "code",
        """for cohort in ('primary_fresh_evaluation', 'fixed_config_expansion'):
    print('\n' + cohort)
    rows = []
    for name, value in result[cohort]['paired_comparisons'].items():
        rows.append({
            'comparison': name,
            'mean_subject_delta': value['mean_subject_delta'],
            'ci_low': value['ci95'][0],
            'ci_high': value['ci95'][1],
            'improved': value['subjects_improved'],
            'tied': value['subjects_tied'],
            'worse': value['subjects_worse'],
        })
    display(pd.DataFrame(rows).set_index('comparison').style.format({
        'mean_subject_delta': '{:+.4f}', 'ci_low': '{:+.4f}', 'ci_high': '{:+.4f}'
    }))
""",
    ),
    cell(
        "code",
        """rows = []
for model_name, measurement in result['neural_efficiency'].items():
    for batch_size, values in measurement['batches'].items():
        rows.append({
            'model': model_name,
            'batch_size': int(batch_size),
            'parameters': measurement['parameter_count_per_model'],
            'latency_ms_per_trial': values['latency_ms_per_trial'],
            'mean_power_watts': values.get('mean_power_watts'),
            'gross_joules_per_trial': values.get('gross_joules_per_trial'),
            'idle_subtracted_joules_per_trial': values.get('idle_subtracted_joules_per_trial'),
            'energy_samples': values.get('energy_samples'),
        })
display(pd.DataFrame(rows).set_index(['model', 'batch_size']).style.format('{:.6f}', na_rep='unsupported'))
""",
    ),
    cell(
        "markdown",
        """## Reading the result

The primary 31-50 table is the confirmatory result. The 31-109 table measures how the frozen choice scales across a larger population. A positive SNN accuracy delta does not by itself establish efficiency. The energy table is required for that comparison, and it only applies to this dense software implementation and Kaggle GPU.

Remaining work after this run includes broader electrode sets, session-aware adaptation, artifact rejection, and deployment on event-driven neuromorphic hardware. None of those are implied by the result here.
""",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "locked-snn-eeg-109-subject-benchmark.ipynb").write_text(
    json.dumps(notebook, indent=1) + "\n"
)

