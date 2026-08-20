#!/usr/bin/env python3
"""Evaluate locked EEG models on fresh subjects and measure real GPU inference."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .benchmark import (
    LogisticRegression,
    Standardizer,
    bandpower_features,
    classification_metrics,
    load_npz,
)
from .lock import SEEDS, verify_lock
from .strong_baselines import FBCSPClassifier, RiemannianTSClassifier
from .train_spiking_eegnet import (
    ChannelStandardizer,
    TrainConfig,
    choose_device,
    fft_bandpass,
    fit_fixed_epochs,
    paired_bootstrap,
    predict,
)


def parse_subjects(value: str) -> list[int]:
    subjects = []
    for part in value.split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            subjects.extend(range(start, end + 1))
        else:
            subjects.append(int(part))
    return sorted(set(subjects))


def validate_subjects(available: np.ndarray, requested: list[int], name: str) -> None:
    missing = sorted(set(requested) - set(int(value) for value in available))
    if missing:
        raise ValueError(f"missing {name} subjects: {missing}")


def per_subject_metrics(
    y: np.ndarray, probability: np.ndarray, subject: np.ndarray, subjects: list[int]
) -> list[dict]:
    return [
        {
            "subject": value,
            "metrics": classification_metrics(
                y[subject == value], probability[subject == value]
            ),
        }
        for value in subjects
    ]


def cohort_result(
    y: np.ndarray,
    subject: np.ndarray,
    probability: np.ndarray,
    subjects: list[int],
    extra: dict | None = None,
) -> dict:
    mask = np.isin(subject, subjects)
    return {
        "aggregate": classification_metrics(y[mask], probability[mask]) | (extra or {}),
        "folds": per_subject_metrics(y, probability, subject, subjects),
    }


def timed_prediction(model, X: np.ndarray) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    probability = model.predict_proba(X)
    elapsed = time.perf_counter() - started
    return probability, {
        "inference_seconds": elapsed,
        "milliseconds_per_trial": elapsed * 1_000 / X.shape[0],
        "hardware": "CPU",
    }


class PowerSampler:
    def __init__(self, device_index: int = 0, interval: float = 0.02):
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self.stop_event = threading.Event()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            timestamp = time.perf_counter()
            watts = self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1_000.0
            self.samples.append((timestamp, watts))
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.samples = []
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> list[tuple[float, float]]:
        self.stop_event.set()
        self.thread.join()
        return self.samples

    def close(self) -> None:
        self.nvml.nvmlShutdown()


def sample_idle_power(seconds: float = 1.0) -> tuple[float | None, str | None]:
    try:
        sampler = PowerSampler()
        sampler.start()
        time.sleep(seconds)
        samples = sampler.stop()
        sampler.close()
        return float(np.mean([watts for _, watts in samples])), None
    except Exception as error:  # NVML is optional outside NVIDIA workers.
        return None, f"{type(error).__name__}: {error}"


@torch.inference_mode()
def ensemble_forward(models: list[torch.nn.Module], batch: torch.Tensor) -> torch.Tensor:
    return torch.stack([model(batch)[0].softmax(dim=1) for model in models]).mean(0)


def neural_efficiency(
    models: list[torch.nn.Module],
    X: np.ndarray,
    device: torch.device,
    batch_sizes: tuple[int, ...] = (1, 64),
    energy_seconds: float = 3.0,
) -> dict:
    for model in models:
        model.eval().to(device)
    result = {
        "device": str(device),
        "parameter_count_per_model": sum(
            parameter.numel() for parameter in models[0].parameters()
        ),
        "ensemble_size": len(models),
        "batches": {},
    }
    idle_watts, nvml_error = sample_idle_power() if device.type == "cuda" else (None, None)
    result["idle_power_watts"] = idle_watts
    if nvml_error:
        result["energy_unavailable_reason"] = nvml_error

    for batch_size in batch_sizes:
        values = np.resize(X, (batch_size, X.shape[1], X.shape[2])).astype(np.float32)
        batch = torch.from_numpy(values).to(device)
        for _ in range(10):
            ensemble_forward(models, batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        repeats = 100 if batch_size == 1 else 30
        started = time.perf_counter()
        for _ in range(repeats):
            ensemble_forward(models, batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metrics = {
            "latency_ms_per_batch": elapsed * 1_000 / repeats,
            "latency_ms_per_trial": elapsed * 1_000 / (repeats * batch_size),
            "latency_repeats": repeats,
        }

        if device.type == "cuda" and idle_watts is not None:
            try:
                sampler = PowerSampler()
                sampler.start()
                loops = 0
                energy_started = time.perf_counter()
                while time.perf_counter() - energy_started < energy_seconds:
                    for _ in range(10):
                        ensemble_forward(models, batch)
                    loops += 10
                    torch.cuda.synchronize()
                duration = time.perf_counter() - energy_started
                samples = sampler.stop()
                sampler.close()
                mean_watts = float(np.mean([watts for _, watts in samples]))
                gross_joules = mean_watts * duration
                dynamic_joules = max(0.0, (mean_watts - idle_watts) * duration)
                inferences = loops * batch_size
                metrics |= {
                    "energy_method": "NVML 20 ms power sampling integrated over repeated inference",
                    "energy_duration_seconds": duration,
                    "energy_samples": len(samples),
                    "mean_power_watts": mean_watts,
                    "gross_joules_per_trial": gross_joules / inferences,
                    "idle_subtracted_joules_per_trial": dynamic_joules / inferences,
                    "energy_inferences": inferences,
                }
            except Exception as error:
                metrics["energy_unavailable_reason"] = f"{type(error).__name__}: {error}"
        result["batches"][str(batch_size)] = metrics
    for model in models:
        model.to("cpu")
    return result


def fit_classical_models(
    dataset, development: np.ndarray, evaluation: np.ndarray
) -> tuple[dict[str, np.ndarray], dict]:
    probabilities, timing = {}, {}

    features = bandpower_features(dataset.X, dataset.fs, 4)
    scaler = Standardizer().fit(features[development])
    train_features = scaler.transform(features[development]).reshape(
        development.sum(), -1
    )
    test_features = scaler.transform(features[evaluation]).reshape(evaluation.sum(), -1)
    started = time.perf_counter()
    logistic = LogisticRegression().fit(train_features, dataset.y[development])
    training_seconds = time.perf_counter() - started
    probabilities["logistic_regression"], timing["logistic_regression"] = timed_prediction(
        logistic, test_features
    )
    timing["logistic_regression"]["training_seconds"] = training_seconds

    for name, model in (
        ("fbcsp", FBCSPClassifier(dataset.fs)),
        ("riemannian_tangent_space", RiemannianTSClassifier(dataset.fs)),
    ):
        started = time.perf_counter()
        model.fit(dataset.X[development], dataset.y[development])
        training_seconds = time.perf_counter() - started
        probabilities[name], timing[name] = timed_prediction(model, dataset.X[evaluation])
        timing[name]["training_seconds"] = training_seconds
    return probabilities, timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--development-subjects", default="1-20")
    parser.add_argument("--primary-subjects", default="31-50")
    parser.add_argument("--expansion-subjects", default="31-109")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--energy-seconds", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation receipt: {args.output}")
    lock = json.loads(args.lock.read_text())
    verify_lock(lock)
    dataset = load_npz(args.data)
    development_subjects = parse_subjects(args.development_subjects)
    primary_subjects = parse_subjects(args.primary_subjects)
    expansion_subjects = parse_subjects(args.expansion_subjects)
    if development_subjects != lock["tuning_protocol"]["development_subjects"]:
        raise ValueError("development subjects differ from the evaluation lock")
    if primary_subjects != lock["evaluation_plan"]["primary_fresh_subjects"]:
        raise ValueError("primary cohort differs from the evaluation lock")
    if expansion_subjects != lock["evaluation_plan"]["fixed_config_expansion_subjects"]:
        raise ValueError("expansion cohort differs from the evaluation lock")
    if set(range(21, 31)) & set(primary_subjects + expansion_subjects):
        raise ValueError("subjects 21-30 are prior evidence and cannot be fresh evaluation")
    if set(development_subjects) & set(expansion_subjects):
        raise ValueError("development and evaluation cohorts overlap")
    validate_subjects(np.unique(dataset.subject), development_subjects, "development")
    validate_subjects(np.unique(dataset.subject), expansion_subjects, "expansion")
    expected = lock["dataset"]
    actual_shape = (dataset.X.shape[1], dataset.X.shape[2], dataset.fs)
    if actual_shape != (expected["channels"], expected["samples"], expected["fs"]):
        raise ValueError(f"dataset geometry {actual_shape} differs from lock")

    development = np.isin(dataset.subject, development_subjects)
    evaluation = np.isin(dataset.subject, expansion_subjects)
    device = choose_device(args.device)
    classical_probabilities, classical_timing = fit_classical_models(
        dataset, development, evaluation
    )

    filtered = fft_bandpass(dataset.X, dataset.fs)
    scaler = ChannelStandardizer().fit(filtered[development])
    X_development = scaler.transform(filtered[development])
    X_evaluation = scaler.transform(filtered[evaluation])
    y_evaluation = dataset.y[evaluation]
    subject_evaluation = dataset.subject[evaluation]
    config = TrainConfig(**lock["config"])
    neural_probabilities, neural_activity, neural_models = {}, {}, {}
    checkpoints = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for model_type in ("analog", "spiking"):
        probabilities, activities, models, model_checkpoints = [], [], [], []
        for seed in SEEDS:
            epochs = lock["selected_epochs"][model_type][str(seed)]
            model = fit_fixed_epochs(
                X_development,
                dataset.y[development],
                dataset.X.shape[1],
                config,
                seed,
                epochs,
                device,
                model_type=model_type,
            )
            probability, activity, _ = predict(
                model, X_evaluation, config.batch_size, device
            )
            checkpoint = args.output.with_name(
                f"{args.output.stem}_{model_type}_seed{seed}.pt"
            )
            torch.save(model.to("cpu").state_dict(), checkpoint)
            model.to(device)
            probabilities.append(probability)
            activities.append(activity)
            models.append(model)
            model_checkpoints.append(str(checkpoint))
        neural_probabilities[model_type] = np.mean(probabilities, axis=0)
        neural_activity[model_type] = float(np.mean(activities))
        neural_models[model_type] = models
        checkpoints[model_type] = model_checkpoints

    all_probabilities = classical_probabilities | {
        "analog_eegnet": neural_probabilities["analog"],
        "spiking_eegnet": neural_probabilities["spiking"],
    }

    def evaluate(subjects: list[int]) -> dict:
        models = {}
        for name, probability in all_probabilities.items():
            extra = {}
            if name == "spiking_eegnet":
                extra["mean_spike_rate"] = neural_activity["spiking"]
            models[name] = cohort_result(
                y_evaluation, subject_evaluation, probability, subjects, extra
            )
        return {
            "subjects": subjects,
            "models": models,
            "paired_comparisons": {
                f"spiking_eegnet_vs_{name}": paired_bootstrap(
                    model["folds"], models["spiking_eegnet"]["folds"]
                )
                for name, model in models.items()
                if name != "spiking_eegnet"
            },
        }

    efficiency = {
        model_type: neural_efficiency(
            models,
            X_evaluation,
            device,
            energy_seconds=args.energy_seconds,
        )
        for model_type, models in neural_models.items()
    }
    result = {
        "protocol": {
            "dataset": dataset.source,
            "lock_sha256": lock["lock_sha256"],
            "development_subjects": development_subjects,
            "primary_evaluated_first": True,
            "primary_subjects": primary_subjects,
            "expansion_subjects": expansion_subjects,
            "subjects_21_30_excluded": True,
            "device": str(device),
        },
        "config": lock["config"],
        "selected_epochs": lock["selected_epochs"],
        "primary_fresh_evaluation": evaluate(primary_subjects),
        "fixed_config_expansion": evaluate(expansion_subjects),
        "timing": classical_timing,
        "neural_efficiency": efficiency,
        "checkpoints": checkpoints,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote immutable evaluation receipt {args.output}")
    for cohort_name in ("primary_fresh_evaluation", "fixed_config_expansion"):
        print(cohort_name)
        for name, model in result[cohort_name]["models"].items():
            metrics = model["aggregate"]
            print(
                f"  {name:28} balanced_accuracy={metrics['balanced_accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f} log_loss={metrics['log_loss']:.4f}"
            )


if __name__ == "__main__":
    main()
