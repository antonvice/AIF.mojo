#!/usr/bin/env python3
"""Build the public Kaggle walkthrough notebook."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parent


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": hashlib.sha1(source.encode()).hexdigest()[:8],
        "metadata": {},
        "source": source.splitlines(True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": hashlib.sha1(source.encode()).hexdigest()[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    markdown(
        """# Spiking EEGNet on motor-imagery EEG

Can a compact spiking neural network decode imagined left-versus-right hand movement better than simple non-spiking and fixed-reservoir baselines?

This notebook reproduces a subject-held-out experiment on the public PhysioNet EEG Motor Movement/Imagery dataset. It documents the completed local run and reruns the same protocol on Kaggle GPU.

## Completed result

| Fresh subjects 21-30 | Balanced accuracy | Macro F1 | Log loss | Spike rate |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.5319 | 0.5299 | 0.6883 | n/a |
| Tuned fixed LIF reservoir | 0.5452 | 0.5435 | 0.6892 | 0.1442 |
| **Spiking EEGNet** | **0.6052** | **0.6035** | **0.6737** | **0.0966** |

Across subjects, Spiking EEGNet gained 7.1 balanced-accuracy points over logistic regression (paired bootstrap 95% CI: +2.4 to +12.3) and 5.6 points over the tuned reservoir (+2.7 to +8.7). It improved 8/10 and 9/10 subjects, respectively.

This is evidence on one held-out cohort, not a state-of-the-art or energy-efficiency claim. Spike rate is an activity proxy, not measured power.
"""
    ),
    markdown(
        """## Protocol

- Dataset: EEGBCI runs 4, 8, and 12, which contain left/right fist motor imagery.
- Input: C3, Cz, C4; 2-second epochs; 128 Hz; 4-38 Hz FFT band-pass.
- Development: subjects 1-20.
- Epoch selection: train on 1-16, validate on 17-20.
- Final refit: subjects 1-20 using each seed's selected epoch count.
- Test: subjects 21-30, evaluated only after model selection.
- Ensemble: seeds 11, 29, 47.
- Metrics: balanced accuracy, macro F1, log loss, per-subject paired bootstrap.

The trainable model combines an EEGNet-style temporal/spatial convolutional front end with 64 learnable LIF neurons. It uses a fast-sigmoid surrogate gradient, learnable decay and threshold, AdamW, Gaussian input noise, gradient clipping, and spike-activity regularization.
"""
    ),
    code(
        """import os, shutil, subprocess, sys
from pathlib import Path

WORK = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
REPO = WORK / 'AIF-MOJO'
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run([
    'git', 'clone', '--depth', '1', '--branch', 'main',
    'https://github.com/antonvice/AIF.mojo.git', str(REPO)
], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'mne', 'rich'], check=True)
gpu_name = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True
).strip() if shutil.which('nvidia-smi') else ''
if 'P100' in gpu_name:
    print('P100 detected: installing PyTorch 2.7.1 CUDA 11.8 for sm_60 support')
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall',
        'torch==2.7.1', '--index-url', 'https://download.pytorch.org/whl/cu118'
    ], check=True)
os.chdir(REPO)
print('Source commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
"""
    ),
    code(
        """import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('PyTorch:', torch.__version__)
print('Device:', device)
if device == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU capability:', torch.cuda.get_device_capability(0))
    print('Compiled CUDA architectures:', torch.cuda.get_arch_list())
    print('CUDA probe:', (torch.ones(1, device='cuda') * 2).item())
"""
    ),
    markdown(
        """## Download and epoch EEGBCI

The raw epochs are recreated from PhysioNet rather than redistributed. This downloads 30 subjects, keeps only C3/Cz/C4, resamples to 128 Hz, and writes a compressed local NPZ for the run.
"""
    ),
    code(
        """subjects = [str(i) for i in range(1, 31)]
data_path = WORK / 'eegbci_left_right_30.npz'
subprocess.run([
    sys.executable, 'benchmarks/eeg_motor_imagery/prepare_eegbci.py',
    '--subjects', *subjects, '--output', str(data_path)
], check=True)
"""
    ),
    markdown(
        """## Train, refit, and evaluate

Each seed chooses its epoch count without seeing subjects 21-30. The selected model is then retrained on all development subjects. The test probabilities are averaged across seeds.

The logistic baseline uses windowed bandpower features. The fixed LIF reservoir uses the configuration selected by the earlier 100-trial Optuna search on subjects 1-10 and frozen before subjects 11-20 were evaluated.
"""
    ),
    code(
        """result_path = WORK / 'spiking_eegnet_kaggle.json'
subprocess.run([
    sys.executable, '-m', 'benchmarks.eeg_motor_imagery.train_spiking_eegnet',
    '--data', str(data_path),
    '--development-subjects', '1-20',
    '--validation-subjects', '17-20',
    '--test-subjects', '21-30',
    '--seeds', '11', '29', '47',
    '--epochs', '60',
    '--device', device,
    '--reservoir-result', 'benchmarks/results/eeg_lif_tuning.json',
    '--output', str(result_path),
], check=True)
"""
    ),
    code(
        """import json
import pandas as pd
from IPython.display import display

result = json.loads(result_path.read_text())
rows = []
for name, model in result['models'].items():
    metrics = model['aggregate']
    rows.append({
        'model': name,
        'balanced_accuracy': metrics['balanced_accuracy'],
        'macro_f1': metrics['macro_f1'],
        'log_loss': metrics['log_loss'],
        'spike_rate': metrics.get('mean_spike_rate'),
    })
metrics_table = pd.DataFrame(rows).set_index('model')
display(metrics_table.style.format('{:.4f}', na_rep='n/a').highlight_max(
    subset=['balanced_accuracy', 'macro_f1'], color='#9fffd0'
).highlight_min(subset=['log_loss', 'spike_rate'], color='#9fffd0'))

for name, comparison in result['paired_comparisons'].items():
    low, high = comparison['ci95']
    print(f"{name}: delta={comparison['mean_subject_delta']:+.4f}, "
          f"95% CI=[{low:+.4f}, {high:+.4f}], "
          f"improved={comparison['subjects_improved']}/10")
"""
    ),
    code(
        """import matplotlib.pyplot as plt

subject_rows = []
for name, model in result['models'].items():
    for fold in model['folds']:
        subject_rows.append({
            'subject': fold['subject'], 'model': name,
            'balanced_accuracy': fold['metrics']['balanced_accuracy']
        })
subject_table = pd.DataFrame(subject_rows)
ax = subject_table.pivot(index='subject', columns='model', values='balanced_accuracy').plot(
    kind='bar', figsize=(13, 5), width=0.82
)
ax.axhline(0.5, color='black', linestyle='--', linewidth=1, label='chance')
ax.set_ylabel('Balanced accuracy')
ax.set_ylim(0, 1)
ax.set_title('Fresh-cohort performance by subject')
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """## What was optimized

The fixed reservoir search covered 100 Optuna trials over temporal windows, reservoir width, simulation ticks, decay, threshold, input gain, recurrent spectral radius and sparsity, ridge regularization, reset mode, and probability temperature. That search reduced spike activity by 39.9%, but its held-out accuracy gain was inconclusive.

The trainable SNN then replaced random fixed features with supervised temporal/spatial filters and surrogate-gradient learning. Three independently trained seeds all passed through the same validation and fresh-cohort protocol.

## What this does not establish

We have not exhausted the design space. This run uses only 30 of 109 EEGBCI subjects, three electrodes, one binary task, one trainable architecture, and a modest hand-chosen training recipe. It does not compare against strong EEG-specific baselines such as FBCSP, Riemannian tangent-space classifiers, or a parameter-matched analog EEGNet. It also does not measure neuromorphic latency or joules.

Now that subjects 21-30 have been observed, they should not guide another claimed fresh test. The next serious experiment is to tune only on subjects 1-20, lock everything, and evaluate once on new subjects 31-50 (then scale to 31-109). Add FBCSP+LDA, Riemannian, and analog EEGNet; search channel sets, filterbanks, crop augmentation, temporal kernel/width, LIF decay/threshold/surrogate slope, regularization, learning rate, and weight decay. Only after that should we discuss a ceiling or an efficiency advantage.

Source and full JSON protocol: [antonvice/AIF.mojo](https://github.com/antonvice/AIF.mojo/tree/main/benchmarks/eeg_motor_imagery)
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "spiking-eegnet-motor-imagery-eeg.ipynb").write_text(
    json.dumps(notebook, indent=1) + "\n"
)
