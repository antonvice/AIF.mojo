#!/usr/bin/env python3
"""Tune the LIF reservoir on development subjects and evaluate a frozen test cohort."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import optuna

from .benchmark import (
    LIFConfig,
    LIFReservoir,
    Standardizer,
    bandpower_features,
    classification_metrics,
    load_npz,
)

SEEDS = (11, 29, 47)


def parse_subjects(value: str) -> list[int]:
    """Parse comma-separated IDs and inclusive ranges such as 1-5,8."""
    subjects = []
    for part in value.split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", maxsplit=1))
            subjects.extend(range(start, end + 1))
        else:
            subjects.append(int(part))
    return sorted(set(subjects))


def suggested_config(trial: optuna.Trial) -> tuple[int, LIFConfig]:
    windows = trial.suggest_categorical("windows", [2, 4, 8])
    return windows, LIFConfig(
        neurons=trial.suggest_categorical("neurons", [32, 64, 96, 128]),
        ticks=trial.suggest_categorical("ticks", [4, 8, 12, 16, 24]),
        decay=trial.suggest_float("decay", 0.60, 0.98),
        threshold=trial.suggest_float("threshold", 0.5, 2.0, log=True),
        input_gain=trial.suggest_float("input_gain", 0.05, 1.5, log=True),
        recurrent_scale=1.0,
        spectral_radius=trial.suggest_float("spectral_radius", 0.05, 1.25),
        recurrent_sparsity=trial.suggest_float("recurrent_sparsity", 0.05, 0.50),
        ridge=trial.suggest_float("ridge", 1e-4, 10.0, log=True),
        temperature=trial.suggest_float("temperature", 0.5, 10.0, log=True),
        reset=trial.suggest_categorical("reset", ["subtract", "zero"]),
    )


def config_from_params(params: dict) -> tuple[int, LIFConfig]:
    windows = int(params["windows"])
    fields = {key: value for key, value in params.items() if key != "windows"}
    fields["recurrent_scale"] = 1.0
    return windows, LIFConfig(**fields)


def ensemble_predictions(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    config: LIFConfig,
    seeds: tuple[int, ...] = SEEDS,
) -> tuple[np.ndarray, float]:
    probabilities, spike_rates = [], []
    for seed in seeds:
        model = LIFReservoir(train_X.shape[-1], config=config, seed=seed)
        model.fit(train_X, train_y)
        prediction, spikes = model.predict_proba(test_X)
        probabilities.append(prediction)
        spike_rates.append(float(spikes[:, -1].mean()))
    return np.mean(probabilities, axis=0), float(np.mean(spike_rates))


def subject_cv(
    features: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    subjects: list[int],
    config: LIFConfig,
) -> tuple[dict[str, float], float]:
    development = np.isin(subject, subjects)
    labels, probabilities, spike_rates = [], [], []
    for held_out in subjects:
        test = subject == held_out
        train = development & ~test
        scaler = Standardizer().fit(features[train])
        train_X = scaler.transform(features[train])
        test_X = scaler.transform(features[test])
        prediction, spike_rate = ensemble_predictions(
            train_X, y[train], test_X, config
        )
        labels.append(y[test])
        probabilities.append(prediction)
        spike_rates.append(spike_rate)
    metrics = classification_metrics(
        np.concatenate(labels), np.concatenate(probabilities)
    )
    return metrics, float(np.mean(spike_rates))


def held_out_evaluation(
    features: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    development_subjects: list[int],
    test_subjects: list[int],
    config: LIFConfig,
) -> dict:
    train = np.isin(subject, development_subjects)
    scaler = Standardizer().fit(features[train])
    train_X = scaler.transform(features[train])
    folds, all_labels, all_probabilities = [], [], []
    for held_out in test_subjects:
        test = subject == held_out
        test_X = scaler.transform(features[test])
        probabilities, spike_rate = ensemble_predictions(
            train_X, y[train], test_X, config
        )
        metrics = classification_metrics(y[test], probabilities)
        folds.append(
            {"subject": held_out, "metrics": metrics, "mean_spike_rate": spike_rate}
        )
        all_labels.append(y[test])
        all_probabilities.append(probabilities)
    aggregate = classification_metrics(
        np.concatenate(all_labels), np.concatenate(all_probabilities)
    )
    aggregate["mean_spike_rate"] = float(
        np.mean([fold["mean_spike_rate"] for fold in folds])
    )
    return {"aggregate": aggregate, "folds": folds}


def paired_subject_analysis(default: dict, tuned: dict) -> dict:
    deltas = np.asarray(
        [
            tuned_fold["metrics"]["balanced_accuracy"]
            - default_fold["metrics"]["balanced_accuracy"]
            for default_fold, tuned_fold in zip(default["folds"], tuned["folds"])
        ]
    )
    rng = np.random.default_rng(2026)
    bootstrap = np.mean(
        rng.choice(deltas, size=(20_000, deltas.size), replace=True), axis=1
    )
    return {
        "mean_subject_balanced_accuracy_delta": float(deltas.mean()),
        "subject_bootstrap_ci95": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
        "subjects_improved": int(np.sum(deltas > 0)),
        "subjects_tied": int(np.sum(deltas == 0)),
        "subjects_worse": int(np.sum(deltas < 0)),
    }


def validate_subjects(available: np.ndarray, requested: list[int], name: str) -> None:
    missing = sorted(set(requested) - set(int(value) for value in available))
    if missing:
        raise ValueError(f"missing {name} subjects: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--development-subjects", default="1-10")
    parser.add_argument("--test-subjects", default="11-20")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_npz(args.data)
    development_subjects = parse_subjects(args.development_subjects)
    test_subjects = parse_subjects(args.test_subjects)
    if set(development_subjects) & set(test_subjects):
        raise ValueError("development and test subjects must be disjoint")
    available = np.unique(dataset.subject)
    validate_subjects(available, development_subjects, "development")
    validate_subjects(available, test_subjects, "test")
    feature_cache = {
        windows: bandpower_features(dataset.X, dataset.fs, windows)
        for windows in (2, 4, 8)
    }

    def objective(trial: optuna.Trial) -> float:
        windows, config = suggested_config(trial)
        metrics, spike_rate = subject_cv(
            feature_cache[windows],
            dataset.y,
            dataset.subject,
            development_subjects,
            config,
        )
        trial.set_user_attr("balanced_accuracy", metrics["balanced_accuracy"])
        trial.set_user_attr("log_loss", metrics["log_loss"])
        trial.set_user_attr("mean_spike_rate", spike_rate)
        if spike_rate < 0.01 or spike_rate > 0.60:
            raise optuna.TrialPruned(f"invalid spike rate {spike_rate:.4f}")
        return metrics["balanced_accuracy"] - 0.01 * metrics["log_loss"]

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=2026)
    )
    started = time.perf_counter()
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    elapsed = time.perf_counter() - started
    completed_trials = [
        trial for trial in study.trials if trial.state.name == "COMPLETE"
    ]
    if not completed_trials:
        raise SystemExit(
            "no feasible trials completed; increase --trials or relax the "
            "spike-rate constraint"
        )
    best_windows, best_config = config_from_params(study.best_trial.params)
    default_windows, default_config = 4, LIFConfig()
    default_result = held_out_evaluation(
        feature_cache[default_windows],
        dataset.y,
        dataset.subject,
        development_subjects,
        test_subjects,
        default_config,
    )
    tuned_result = held_out_evaluation(
        feature_cache[best_windows],
        dataset.y,
        dataset.subject,
        development_subjects,
        test_subjects,
        best_config,
    )

    result = {
        "protocol": {
            "dataset": dataset.source,
            "development_subjects": development_subjects,
            "test_subjects": test_subjects,
            "seeds": list(SEEDS),
            "sampler": "Optuna TPE",
            "trials_requested": args.trials,
            "trials_completed": len(completed_trials),
            "trials_pruned": len(
                [trial for trial in study.trials if trial.state.name == "PRUNED"]
            ),
            "elapsed_seconds": elapsed,
            "objective": "balanced_accuracy - 0.01 * log_loss",
            "spike_rate_constraint": [0.01, 0.60],
        },
        "best_development_trial": {
            "number": study.best_trial.number,
            "objective": study.best_value,
            "metrics": study.best_trial.user_attrs,
            "windows": best_windows,
            "config": asdict(best_config),
        },
        "untouched_test": {
            "default": {
                "windows": default_windows,
                "config": asdict(default_config),
                "result": default_result,
            },
            "tuned": {
                "windows": best_windows,
                "config": asdict(best_config),
                "result": tuned_result,
            },
            "paired_subject_analysis": paired_subject_analysis(
                default_result, tuned_result
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["best_development_trial"], indent=2))
    for name in ("default", "tuned"):
        evaluation = result["untouched_test"][name]
        metrics = evaluation["result"]["aggregate"]
        print(
            f"{name}: balanced_accuracy={metrics['balanced_accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} log_loss={metrics['log_loss']:.4f} "
            f"spike_rate={metrics['mean_spike_rate']:.4f}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
