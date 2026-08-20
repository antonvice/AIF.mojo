#!/usr/bin/env python3
"""Train a compact surrogate-gradient Spiking EEGNet on subject-held-out EEG."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .benchmark import (
    LIFConfig,
    LIFReservoir,
    LogisticRegression,
    Standardizer,
    bandpower_features,
    classification_metrics,
    load_npz,
)

CONSOLE = Console()


@dataclass(frozen=True)
class TrainConfig:
    temporal_filters: int = 8
    depth_multiplier: int = 2
    hidden: int = 64
    temporal_kernel: int = 33
    pool: int = 4
    beta: float = 0.90
    threshold: float = 1.0
    surrogate_slope: float = 10.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 64
    activity_target: float = 0.10
    activity_weight: float = 0.10
    noise_std: float = 0.02
    patience: int = 12


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane_minus_threshold, slope):
        ctx.save_for_backward(membrane_minus_threshold)
        ctx.slope = slope
        return (membrane_minus_threshold >= 0).to(membrane_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (distance,) = ctx.saved_tensors
        derivative = 1.0 / (1.0 + ctx.slope * distance.abs()).pow(2)
        return gradient * derivative, None


class TrainableLIF(nn.Module):
    def __init__(self, size: int, beta: float, threshold: float, slope: float):
        super().__init__()
        self.beta_logit = nn.Parameter(torch.full((size,), math.log(beta / (1 - beta))))
        inverse_softplus = math.log(math.exp(threshold) - 1.0)
        self.threshold_raw = nn.Parameter(torch.full((size,), inverse_softplus))
        self.slope = slope

    def forward(self, current: torch.Tensor, membrane: torch.Tensor):
        beta = torch.sigmoid(self.beta_logit)
        threshold = F.softplus(self.threshold_raw) + 1e-4
        membrane = beta * membrane + current
        spike = SurrogateSpike.apply(membrane - threshold, self.slope)
        membrane = membrane - spike.detach() * threshold
        return spike, membrane


class SpikingEEGNet(nn.Module):
    def __init__(self, channels: int, classes: int = 2, config: TrainConfig | None = None):
        super().__init__()
        self.config = config or TrainConfig()
        f1 = self.config.temporal_filters
        f2 = f1 * self.config.depth_multiplier
        self.frontend = nn.Sequential(
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, self.config.temporal_kernel),
                padding=(0, self.config.temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
            nn.Conv2d(
                f1,
                f2,
                kernel_size=(channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.config.pool)),
            nn.Dropout(0.25),
        )
        self.projection = nn.Linear(f2, self.config.hidden)
        self.lif = TrainableLIF(
            self.config.hidden,
            self.config.beta,
            self.config.threshold,
            self.config.surrogate_slope,
        )
        self.readout = nn.Linear(self.config.hidden, classes)

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.frontend(eeg[:, None]).squeeze(2).transpose(1, 2)
        membrane = torch.zeros(
            eeg.shape[0], self.config.hidden, device=eeg.device, dtype=eeg.dtype
        )
        logits, spike_count = [], torch.zeros((), device=eeg.device)
        for step in range(features.shape[1]):
            spikes, membrane = self.lif(self.projection(features[:, step]), membrane)
            logits.append(self.readout(spikes))
            spike_count = spike_count + spikes.mean()
        return torch.stack(logits).mean(dim=0), spike_count / features.shape[1]


class AnalogEEGNet(nn.Module):
    """Parameter-matched non-spiking control for the Spiking EEGNet."""

    def __init__(self, channels: int, classes: int = 2, config: TrainConfig | None = None):
        super().__init__()
        self.config = config or TrainConfig()
        f1 = self.config.temporal_filters
        f2 = f1 * self.config.depth_multiplier
        self.frontend = nn.Sequential(
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, self.config.temporal_kernel),
                padding=(0, self.config.temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, kernel_size=(channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.config.pool)),
            nn.Dropout(0.25),
        )
        self.projection = nn.Linear(f2, self.config.hidden)
        self.activation_scale = nn.Parameter(torch.ones(self.config.hidden))
        self.activation_bias = nn.Parameter(torch.zeros(self.config.hidden))
        self.readout = nn.Linear(self.config.hidden, classes)

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.frontend(eeg[:, None]).squeeze(2).transpose(1, 2)
        hidden = F.elu(self.projection(features))
        hidden = hidden * self.activation_scale + self.activation_bias
        logits = self.readout(hidden).mean(dim=1)
        return logits, torch.zeros((), device=eeg.device, dtype=eeg.dtype)


def build_neural_model(
    model_type: str, channels: int, config: TrainConfig
) -> nn.Module:
    if model_type == "spiking":
        return SpikingEEGNet(channels, config=config)
    if model_type == "analog":
        return AnalogEEGNet(channels, config=config)
    raise ValueError(f"unknown neural model type: {model_type}")


class ChannelStandardizer:
    def fit(self, X: np.ndarray) -> "ChannelStandardizer":
        self.mean = X.mean(axis=(0, 2), keepdims=True)
        self.scale = X.std(axis=(0, 2), keepdims=True) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.scale).astype(np.float32)


def fft_bandpass(X: np.ndarray, fs: float, low: float = 4.0, high: float = 38.0):
    spectrum = np.fft.rfft(X, axis=2)
    frequency = np.fft.rfftfreq(X.shape[2], 1.0 / fs)
    spectrum[:, :, (frequency < low) | (frequency > high)] = 0
    return np.fft.irfft(spectrum, n=X.shape[2], axis=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    for eeg, labels in loader:
        eeg, labels = eeg.to(device), labels.to(device)
        if model.config.noise_std:
            eeg = eeg + torch.randn_like(eeg) * model.config.noise_std
        optimizer.zero_grad(set_to_none=True)
        logits, spike_rate = model(eeg)
        loss = F.cross_entropy(logits, labels)
        if isinstance(model, SpikingEEGNet):
            activity = (spike_rate - model.config.activity_target).pow(2)
            loss = loss + model.config.activity_weight * activity
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach()) * labels.shape[0]
    return total / len(loader.dataset)


@torch.no_grad()
def predict(
    model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device
) -> tuple[np.ndarray, float, float]:
    model.eval()
    loader = make_loader(X, np.zeros(X.shape[0]), batch_size, 0, False)
    probabilities, spike_rates = [], []
    for eeg, _ in loader:
        logits, spike_rate = model(eeg.to(device))
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        spike_rates.append(float(spike_rate.cpu()) * eeg.shape[0])
    probability = np.concatenate(probabilities)
    return probability, float(sum(spike_rates) / X.shape[0]), 0.0


def select_epoch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    channels: int,
    config: TrainConfig,
    seed: int,
    max_epochs: int,
    device: torch.device,
    model_type: str = "spiking",
) -> tuple[int, list[dict]]:
    set_seed(seed)
    model = build_neural_model(model_type, channels, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = make_loader(X_train, y_train, config.batch_size, seed, True)
    best_epoch, best_score, stale, history = 1, -np.inf, 0, []
    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, loader, optimizer, device)
        probability, spike_rate, _ = predict(
            model, X_validation, config.batch_size, device
        )
        metrics = classification_metrics(y_validation, probability)
        score = metrics["balanced_accuracy"] - 0.01 * metrics["log_loss"]
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "spike_rate": spike_rate} | metrics
        )
        if score > best_score + 1e-5:
            best_epoch, best_score, stale = epoch, score, 0
            CONSOLE.log(
                f"seed {seed} epoch {epoch}: validation balanced accuracy "
                f"{metrics['balanced_accuracy']:.3f}, spikes {spike_rate:.3f}"
            )
        else:
            stale += 1
        if stale >= config.patience:
            break
    return best_epoch, history


def fit_fixed_epochs(
    X: np.ndarray,
    y: np.ndarray,
    channels: int,
    config: TrainConfig,
    seed: int,
    epochs: int,
    device: torch.device,
    model_type: str = "spiking",
) -> nn.Module:
    set_seed(seed)
    model = build_neural_model(model_type, channels, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = make_loader(X, y, config.batch_size, seed, True)
    for _ in range(epochs):
        train_epoch(model, loader, optimizer, device)
    return model


def per_subject_metrics(
    y: np.ndarray, probability: np.ndarray, subject: np.ndarray, subjects: list[int]
) -> list[dict]:
    return [
        {
            "subject": value,
            "metrics": classification_metrics(y[subject == value], probability[subject == value]),
        }
        for value in subjects
    ]


def paired_bootstrap(reference: list[dict], candidate: list[dict]) -> dict:
    deltas = np.asarray(
        [
            right["metrics"]["balanced_accuracy"]
            - left["metrics"]["balanced_accuracy"]
            for left, right in zip(reference, candidate)
        ]
    )
    rng = np.random.default_rng(2026)
    samples = rng.choice(deltas, size=(20_000, deltas.size), replace=True).mean(axis=1)
    return {
        "mean_subject_delta": float(deltas.mean()),
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "subjects_improved": int(np.sum(deltas > 0)),
        "subjects_tied": int(np.sum(deltas == 0)),
        "subjects_worse": int(np.sum(deltas < 0)),
    }


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_subjects(value: str) -> list[int]:
    start, end = (int(item) for item in value.split("-", maxsplit=1))
    return list(range(start, end + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--development-subjects", default="1-20")
    parser.add_argument("--validation-subjects", default="17-20")
    parser.add_argument("--test-subjects", default="21-30")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reservoir-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_npz(args.data)
    development_subjects = parse_subjects(args.development_subjects)
    validation_subjects = parse_subjects(args.validation_subjects)
    test_subjects = parse_subjects(args.test_subjects)
    if set(development_subjects) & set(test_subjects):
        raise ValueError("development and test subjects must be disjoint")
    if not set(validation_subjects) <= set(development_subjects):
        raise ValueError("validation subjects must be a subset of development")
    available = set(int(value) for value in np.unique(dataset.subject))
    requested = set(development_subjects + test_subjects)
    if missing := requested - available:
        raise ValueError(f"missing subjects: {sorted(missing)}")

    device = choose_device(args.device)
    CONSOLE.rule(f"Spiking EEGNet on {device}")
    filtered = fft_bandpass(dataset.X, dataset.fs)
    development = np.isin(dataset.subject, development_subjects)
    validation = np.isin(dataset.subject, validation_subjects)
    selection_train = development & ~validation
    test = np.isin(dataset.subject, test_subjects)
    selection_scaler = ChannelStandardizer().fit(filtered[selection_train])
    X_selection = selection_scaler.transform(filtered[selection_train])
    X_validation = selection_scaler.transform(filtered[validation])
    final_scaler = ChannelStandardizer().fit(filtered[development])
    X_development = final_scaler.transform(filtered[development])
    X_test = final_scaler.transform(filtered[test])
    config = TrainConfig()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    seed_probabilities, seed_spike_rates, histories, checkpoints = [], [], {}, []
    for seed in args.seeds:
        best_epoch, history = select_epoch(
            X_selection,
            dataset.y[selection_train],
            X_validation,
            dataset.y[validation],
            dataset.X.shape[1],
            config,
            seed,
            args.epochs,
            device,
        )
        CONSOLE.log(f"seed {seed}: refitting all development subjects for {best_epoch} epochs")
        model = fit_fixed_epochs(
            X_development,
            dataset.y[development],
            dataset.X.shape[1],
            config,
            seed,
            best_epoch,
            device,
        )
        probability, spike_rate, _ = predict(model, X_test, config.batch_size, device)
        checkpoint = args.output.with_name(f"{args.output.stem}_seed{seed}.pt")
        torch.save(model.to("cpu").state_dict(), checkpoint)
        seed_probabilities.append(probability)
        seed_spike_rates.append(spike_rate)
        histories[str(seed)] = {"selected_epoch": best_epoch, "epochs": history}
        checkpoints.append(str(checkpoint))

    probability = np.mean(seed_probabilities, axis=0)
    y_test, subject_test = dataset.y[test], dataset.subject[test]
    spiking_folds = per_subject_metrics(y_test, probability, subject_test, test_subjects)
    spiking = {
        "aggregate": classification_metrics(y_test, probability)
        | {"mean_spike_rate": float(np.mean(seed_spike_rates))},
        "folds": spiking_folds,
    }

    features = bandpower_features(dataset.X, dataset.fs, 4)
    feature_scaler = Standardizer().fit(features[development])
    train_features = feature_scaler.transform(features[development])
    test_features = feature_scaler.transform(features[test])
    logistic = LogisticRegression().fit(
        train_features.reshape(train_features.shape[0], -1), dataset.y[development]
    )
    logistic_probability = logistic.predict_proba(
        test_features.reshape(test_features.shape[0], -1)
    )
    logistic_folds = per_subject_metrics(
        y_test, logistic_probability, subject_test, test_subjects
    )
    baselines = {
        "logistic_regression": {
            "aggregate": classification_metrics(y_test, logistic_probability),
            "folds": logistic_folds,
        }
    }

    if args.reservoir_result:
        reservoir = json.loads(args.reservoir_result.read_text())
        reservoir_config = LIFConfig(
            **reservoir["best_development_trial"]["config"]
        )
        predictions, spike_rates = [], []
        for seed in args.seeds:
            model = LIFReservoir(train_features.shape[-1], reservoir_config, seed).fit(
                train_features, dataset.y[development]
            )
            seed_probability, spikes = model.predict_proba(test_features)
            predictions.append(seed_probability)
            spike_rates.append(float(spikes[:, -1].mean()))
        reservoir_probability = np.mean(predictions, axis=0)
        reservoir_folds = per_subject_metrics(
            y_test, reservoir_probability, subject_test, test_subjects
        )
        baselines["tuned_lif_reservoir"] = {
            "aggregate": classification_metrics(y_test, reservoir_probability)
            | {"mean_spike_rate": float(np.mean(spike_rates))},
            "folds": reservoir_folds,
        }

    result = {
        "protocol": {
            "dataset": dataset.source,
            "development_subjects": development_subjects,
            "validation_subjects": validation_subjects,
            "test_subjects": test_subjects,
            "seeds": args.seeds,
            "device": str(device),
            "test_evaluated_after_epoch_selection": True,
        },
        "config": asdict(config),
        "checkpoints": checkpoints,
        "selection_histories": histories,
        "models": baselines | {"spiking_eegnet": spiking},
        "paired_comparisons": {
            f"spiking_eegnet_vs_{name}": paired_bootstrap(
                baseline["folds"], spiking_folds
            )
            for name, baseline in baselines.items()
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name, model_result in result["models"].items():
        metrics = model_result["aggregate"]
        CONSOLE.print(
            f"{name:24} balanced_accuracy={metrics['balanced_accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} log_loss={metrics['log_loss']:.4f}"
        )
    CONSOLE.print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
