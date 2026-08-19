#!/usr/bin/env python3
"""Compare a linear baseline, an LIF SNN, and sequential AIF on EEG trials."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPS = 1e-9


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    fs: float
    source: str


@dataclass(frozen=True)
class LIFConfig:
    neurons: int = 48
    ticks: int = 12
    decay: float = 0.86
    threshold: float = 1.0
    input_gain: float = 0.42
    recurrent_scale: float = 0.12
    spectral_radius: float | None = None
    recurrent_sparsity: float = 1.0
    ridge: float = 0.2
    temperature: float = 4.0
    reset: str = "subtract"


def load_npz(path: Path) -> Dataset:
    """Load X[trial, channel, time], y, subject, and optional scalar fs."""
    with np.load(path, allow_pickle=False) as data:
        missing = {"X", "y", "subject"} - set(data.files)
        if missing:
            raise ValueError(f"missing NPZ arrays: {sorted(missing)}")
        X = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["y"]).reshape(-1)
        subject = np.asarray(data["subject"]).reshape(-1)
        fs = float(np.asarray(data["fs"]).item()) if "fs" in data else 128.0
    if X.ndim != 3 or X.shape[0] != y.size or y.size != subject.size:
        raise ValueError("expected X[trial, channel, time], y[trial], subject[trial]")
    labels = np.unique(y)
    if labels.size != 2:
        raise ValueError(f"expected exactly two classes, got {labels.tolist()}")
    y = (y == labels[1]).astype(np.int64)
    if not np.isfinite(X).all() or fs <= 0:
        raise ValueError("X must be finite and fs must be positive")
    return Dataset(X, y, subject, fs, str(path))


def synthetic_dataset(
    *, subjects: int = 6, trials_per_subject: int = 40, seed: int = 7
) -> Dataset:
    """Generate a motor-imagery-like surrogate for pipeline smoke testing."""
    rng = np.random.default_rng(seed)
    fs, samples, channels = 128.0, 256, 4
    t = np.arange(samples) / fs
    rows, labels, subject_ids = [], [], []
    for subject in range(subjects):
        phase = rng.uniform(0, 2 * np.pi, size=channels)
        gain = rng.uniform(0.85, 1.15, size=channels)
        for trial in range(trials_per_subject):
            label = trial % 2
            signal = rng.normal(0, 0.9, size=(channels, samples))
            signal += 0.25 * np.sin(2 * np.pi * 6 * t + phase[:, None])
            active, opposite = ((1, 0) if label else (0, 1))
            signal[active] += 1.25 * np.sin(2 * np.pi * 10 * t + phase[active])
            signal[active] += 0.60 * np.sin(2 * np.pi * 20 * t)
            signal[opposite] += 0.35 * np.sin(2 * np.pi * 10 * t)
            rows.append(signal * gain[:, None])
            labels.append(label)
            subject_ids.append(subject)
    order = rng.permutation(len(rows))
    return Dataset(
        np.asarray(rows)[order],
        np.asarray(labels, dtype=np.int64)[order],
        np.asarray(subject_ids)[order],
        fs,
        "synthetic motor-imagery surrogate (not physiological evidence)",
    )


def bandpower_features(X: np.ndarray, fs: float, windows: int = 4) -> np.ndarray:
    """Return log mu/beta power as [trial, window, channel * 2]."""
    if X.shape[2] % windows:
        raise ValueError("time samples must divide evenly into windows")
    features = []
    for chunk in np.split(X, windows, axis=2):
        chunk = chunk - chunk.mean(axis=2, keepdims=True)
        spectrum = np.abs(np.fft.rfft(chunk, axis=2)) ** 2
        frequencies = np.fft.rfftfreq(chunk.shape[2], 1.0 / fs)
        bands = []
        for low, high in ((8.0, 13.0), (13.0, 30.0)):
            mask = (frequencies >= low) & (frequencies < high)
            bands.append(np.log(spectrum[:, :, mask].mean(axis=2) + EPS))
        features.append(np.concatenate(bands, axis=1))
    return np.stack(features, axis=1)


class Standardizer:
    def fit(self, X: np.ndarray) -> "Standardizer":
        flat = X.reshape(-1, X.shape[-1])
        self.mean = flat.mean(axis=0)
        self.scale = flat.std(axis=0) + EPS
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


class LogisticRegression:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        design = np.column_stack((np.ones(X.shape[0]), X))
        self.weights = np.zeros(design.shape[1])
        for _ in range(500):
            probabilities = sigmoid(design @ self.weights)
            gradient = design.T @ (probabilities - y) / y.size
            gradient[1:] += 1e-3 * self.weights[1:]
            self.weights -= 0.08 * gradient
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(np.column_stack((np.ones(X.shape[0]), X)) @ self.weights)


class LIFReservoir:
    """Current-coded leaky integrate-and-fire reservoir with a ridge readout."""

    def __init__(self, inputs: int, config: LIFConfig | None = None, seed: int = 11):
        self.config = config or LIFConfig()
        rng = np.random.default_rng(seed)
        neurons = self.config.neurons
        self.input_weights = rng.normal(
            0, self.config.input_gain, size=(inputs, neurons)
        )
        self.recurrent_weights = rng.normal(
            0, self.config.recurrent_scale, size=(neurons, neurons)
        )
        mask = rng.random((neurons, neurons)) < self.config.recurrent_sparsity
        self.recurrent_weights *= mask
        np.fill_diagonal(self.recurrent_weights, 0)
        if self.config.spectral_radius is not None:
            radius = approximate_spectral_radius(self.recurrent_weights)
            if radius > EPS:
                self.recurrent_weights *= self.config.spectral_radius / radius

    def encode(self, X: np.ndarray) -> np.ndarray:
        outputs = np.zeros((X.shape[0], X.shape[1], self.input_weights.shape[1]))
        for trial in range(X.shape[0]):
            membrane = np.zeros(self.input_weights.shape[1])
            spikes = np.zeros_like(membrane)
            cumulative = np.zeros_like(membrane)
            for window in range(X.shape[1]):
                current = X[trial, window] @ self.input_weights
                for _ in range(self.config.ticks):
                    membrane = (
                        self.config.decay * membrane
                        + current
                        + spikes @ self.recurrent_weights
                    )
                    spikes = (membrane >= self.config.threshold).astype(np.float64)
                    if self.config.reset == "subtract":
                        membrane -= spikes * self.config.threshold
                    elif self.config.reset == "zero":
                        membrane *= 1.0 - spikes
                    else:
                        raise ValueError(f"unknown reset mechanism: {self.config.reset}")
                    cumulative += spikes
                outputs[trial, window] = cumulative / (
                    (window + 1) * self.config.ticks
                )
        return outputs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LIFReservoir":
        encoded = self.encode(X)[:, -1]
        design = np.column_stack((np.ones(encoded.shape[0]), encoded))
        regularizer = np.eye(design.shape[1]) * self.config.ridge
        regularizer[0, 0] = 0
        self.readout = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
        return self

    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        encoded = self.encode(X)
        flat = encoded.reshape(-1, encoded.shape[-1])
        scores = np.column_stack((np.ones(flat.shape[0]), flat)) @ self.readout
        probabilities = sigmoid(
            self.config.temperature * (scores.reshape(encoded.shape[:2]) - 0.5)
        )
        return probabilities[:, -1], encoded


def approximate_spectral_radius(matrix: np.ndarray, iterations: int = 30) -> float:
    """Estimate the dominant eigenvalue magnitude without an O(n^3) solve."""
    if not np.any(matrix):
        return 0.0
    vector = np.full(matrix.shape[0], 1.0 / np.sqrt(matrix.shape[0]))
    radius = 0.0
    for _ in range(iterations):
        projected = matrix @ vector
        radius = float(np.linalg.norm(projected))
        if radius <= EPS:
            return 0.0
        vector = projected / radius
    return radius


class SequentialAIF:
    """A belief-plan-act decoder with wait/left/right actions."""

    def __init__(self, wait_cost: float = 0.025, samples: int = 24, seed: int = 19):
        self.wait_cost = wait_cost
        self.samples = samples
        self.rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SequentialAIF":
        self.means = np.stack([X[y == label].mean(axis=0) for label in (0, 1)])
        self.variances = np.stack([X[y == label].var(axis=0) + 0.15 for label in (0, 1)])
        counts = np.bincount(y, minlength=2).astype(np.float64)
        self.prior = counts / counts.sum()
        return self

    def _log_likelihood(self, observation: np.ndarray, window: int) -> np.ndarray:
        delta = observation - self.means[:, window]
        return -0.5 * np.sum(
            np.log(2 * np.pi * self.variances[:, window])
            + delta * delta / self.variances[:, window], axis=1
        )

    def _update(self, belief: np.ndarray, observation: np.ndarray, window: int) -> np.ndarray:
        logits = np.log(belief + EPS) + self._log_likelihood(observation, window)
        logits -= logits.max()
        posterior = np.exp(logits)
        return posterior / posterior.sum()

    def _expected_next_risk(self, belief: np.ndarray, window: int) -> float:
        risks = []
        for _ in range(self.samples):
            label = int(self.rng.random() >= belief[0])
            observation = self.rng.normal(
                self.means[label, window], np.sqrt(self.variances[label, window])
            )
            risks.append(self._update(belief, observation, window).min())
        return float(np.mean(risks))

    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        probabilities = np.empty(X.shape[0])
        decision_windows = np.empty(X.shape[0], dtype=np.int64)
        actions = []
        for trial in range(X.shape[0]):
            belief = self.prior.copy()
            for window in range(X.shape[1]):
                belief = self._update(belief, X[trial, window], window)
                commit_costs = np.array([belief[1], belief[0]])
                if window < X.shape[1] - 1:
                    wait_cost = self.wait_cost + self._expected_next_risk(belief, window + 1)
                    if wait_cost < commit_costs.min():
                        continue
                choice = int(np.argmin(commit_costs))
                decision_windows[trial] = window + 1
                actions.append("choose-left" if choice == 0 else "choose-right")
                break
            probabilities[trial] = belief[1]
        return probabilities, decision_windows, actions


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def classification_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    predicted = (probabilities >= 0.5).astype(np.int64)
    recalls, f1s = [], []
    for label in (0, 1):
        tp = np.sum((predicted == label) & (y == label))
        fp = np.sum((predicted == label) & (y != label))
        fn = np.sum((predicted != label) & (y == label))
        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        recalls.append(recall)
        f1s.append(2 * precision * recall / max(EPS, precision + recall))
    return {
        "accuracy": float(np.mean(predicted == y)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "log_loss": float(-np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))),
        "brier": float(np.mean((probabilities - y) ** 2)),
    }


def evaluate_fold(features: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> dict:
    scaler = Standardizer().fit(features[train])
    train_X, test_X = scaler.transform(features[train]), scaler.transform(features[test])
    flattened_train = train_X.reshape(train_X.shape[0], -1)
    flattened_test = test_X.reshape(test_X.shape[0], -1)
    results = {}

    start = time.perf_counter()
    baseline = LogisticRegression().fit(flattened_train, y[train])
    probabilities = baseline.predict_proba(flattened_test)
    results["logistic_regression"] = classification_metrics(y[test], probabilities) | {
        "decision_windows": float(features.shape[1]),
        "elapsed_seconds": time.perf_counter() - start,
    }

    start = time.perf_counter()
    snn = LIFReservoir(train_X.shape[-1]).fit(train_X, y[train])
    probabilities, spikes = snn.predict_proba(test_X)
    results["lif_snn"] = classification_metrics(y[test], probabilities) | {
        "decision_windows": float(features.shape[1]),
        "mean_spike_rate": float(spikes[:, -1].mean()),
        "elapsed_seconds": time.perf_counter() - start,
    }

    start = time.perf_counter()
    aif = SequentialAIF().fit(train_X, y[train])
    probabilities, decisions, _ = aif.predict_proba(test_X)
    results["sequential_aif"] = classification_metrics(y[test], probabilities) | {
        "decision_windows": float(decisions.mean()),
        "elapsed_seconds": time.perf_counter() - start,
    }
    return results


def run_benchmark(dataset: Dataset, windows: int = 4, holdout: str | None = None) -> dict:
    features = bandpower_features(dataset.X, dataset.fs, windows)
    subjects = np.unique(dataset.subject)
    if holdout is not None:
        subjects = np.asarray([subject for subject in subjects if str(subject) == holdout])
        if subjects.size == 0:
            raise ValueError(f"unknown holdout subject {holdout!r}")
    folds = []
    for subject in subjects:
        test = dataset.subject == subject
        train = ~test
        if np.unique(dataset.y[train]).size != 2 or np.unique(dataset.y[test]).size != 2:
            raise ValueError(f"subject {subject!r} does not provide a valid binary fold")
        folds.append({"subject": str(subject), "models": evaluate_fold(features, dataset.y, train, test)})
    aggregate = {}
    for model in folds[0]["models"]:
        aggregate[model] = {
            key: float(np.mean([fold["models"][model][key] for fold in folds]))
            for key in folds[0]["models"][model]
        }
    return {
        "dataset": {
            "source": dataset.source,
            "trials": int(dataset.X.shape[0]),
            "channels": int(dataset.X.shape[1]),
            "samples": int(dataset.X.shape[2]),
            "sampling_hz": dataset.fs,
            "subjects": int(np.unique(dataset.subject).size),
            "evaluation": "leave-one-subject-out" if holdout is None else "held-out subject",
        },
        "feature_extraction": f"{windows} windows; log mu (8-13 Hz) and beta (13-30 Hz) power",
        "models": aggregate,
        "folds": folds,
    }


def format_table(result: dict) -> str:
    lines = ["model                 bal_acc   macro_f1  log_loss  windows", "-" * 62]
    for name, metrics in result["models"].items():
        lines.append(
            f"{name:21} {metrics['balanced_accuracy']:8.3f} "
            f"{metrics['macro_f1']:10.3f} {metrics['log_loss']:9.3f} "
            f"{metrics['decision_windows']:8.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", type=Path, help="NPZ containing X, y, subject, optional fs")
    source.add_argument("--synthetic", action="store_true", help="run the deterministic smoke dataset")
    parser.add_argument("--subjects", type=int, default=6)
    parser.add_argument("--trials-per-subject", type=int, default=40)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--holdout-subject")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.data is None and not args.synthetic:
        parser.error("choose --data FILE or --synthetic")
    dataset = load_npz(args.data) if args.data else synthetic_dataset(
        subjects=args.subjects, trials_per_subject=args.trials_per_subject
    )
    result = run_benchmark(dataset, args.windows, args.holdout_subject)
    print(f"source: {result['dataset']['source']}")
    print(format_table(result))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
