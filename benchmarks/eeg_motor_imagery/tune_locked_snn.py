#!/usr/bin/env python3
"""Tune the SNN on subjects 1-20 and emit an immutable evaluation lock."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna

from .benchmark import load_npz
from .lock import LOCK_VERSION, SEEDS, canonical_hash, verify_lock
from .train_spiking_eegnet import (
    ChannelStandardizer,
    TrainConfig,
    choose_device,
    fft_bandpass,
    select_epoch,
)
from .tune_lif import parse_subjects, validate_subjects


def suggested_config(trial: optuna.Trial) -> TrainConfig:
    return TrainConfig(
        temporal_filters=trial.suggest_categorical("temporal_filters", [4, 8, 12, 16]),
        depth_multiplier=trial.suggest_categorical("depth_multiplier", [1, 2]),
        hidden=trial.suggest_categorical("hidden", [32, 64, 96, 128]),
        temporal_kernel=trial.suggest_categorical(
            "temporal_kernel", [17, 25, 33, 49, 65]
        ),
        pool=trial.suggest_categorical("pool", [2, 4, 8]),
        beta=trial.suggest_float("beta", 0.75, 0.98),
        threshold=trial.suggest_float("threshold", 0.5, 1.5),
        surrogate_slope=trial.suggest_categorical(
            "surrogate_slope", [5.0, 10.0, 20.0, 40.0]
        ),
        learning_rate=trial.suggest_float("learning_rate", 2e-4, 3e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        activity_target=trial.suggest_float("activity_target", 0.04, 0.20),
        activity_weight=trial.suggest_float("activity_weight", 1e-3, 0.30, log=True),
        noise_std=trial.suggest_float("noise_std", 0.0, 0.05),
        patience=8,
    )


def best_history_score(history: list[dict]) -> tuple[float, dict]:
    row = max(
        history,
        key=lambda item: item["balanced_accuracy"] - 0.01 * item["log_loss"],
    )
    return row["balanced_accuracy"] - 0.01 * row["log_loss"], row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--development-subjects", default="1-20")
    parser.add_argument("--validation-subjects", default="17-20")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--trial-epochs", type=int, default=35)
    parser.add_argument("--lock-epochs", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_npz(args.data)
    development_subjects = parse_subjects(args.development_subjects)
    validation_subjects = parse_subjects(args.validation_subjects)
    allowed = set(range(1, 21))
    if not set(development_subjects) <= allowed:
        raise ValueError("SNN tuning is locked to subjects 1-20")
    if not set(validation_subjects) < set(development_subjects):
        raise ValueError("validation subjects must be a strict development subset")
    validate_subjects(np.unique(dataset.subject), development_subjects, "development")

    device = choose_device(args.device)
    development = np.isin(dataset.subject, development_subjects)
    tuning_X = fft_bandpass(dataset.X[development], dataset.fs)
    tuning_y = dataset.y[development]
    tuning_subject = dataset.subject[development]
    validation = np.isin(tuning_subject, validation_subjects)
    selection_train = ~validation
    scaler = ChannelStandardizer().fit(tuning_X[selection_train])
    X_train = scaler.transform(tuning_X[selection_train])
    X_validation = scaler.transform(tuning_X[validation])
    y_train = tuning_y[selection_train]
    y_validation = tuning_y[validation]

    def objective(trial: optuna.Trial) -> float:
        config = suggested_config(trial)
        epoch, history = select_epoch(
            X_train,
            y_train,
            X_validation,
            y_validation,
            dataset.X.shape[1],
            config,
            seed=2026,
            max_epochs=args.trial_epochs,
            device=device,
        )
        score, best = best_history_score(history)
        trial.set_user_attr("selected_epoch", epoch)
        trial.set_user_attr("balanced_accuracy", best["balanced_accuracy"])
        trial.set_user_attr("log_loss", best["log_loss"])
        trial.set_user_attr("spike_rate", best["spike_rate"])
        if not 0.01 <= best["spike_rate"] <= 0.50:
            raise optuna.TrialPruned(f"invalid spike rate {best['spike_rate']:.4f}")
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=2026, multivariate=True),
    )
    started = time.perf_counter()
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    elapsed = time.perf_counter() - started
    config = TrainConfig(**study.best_trial.params, patience=8)

    selected_epochs: dict[str, dict[str, int]] = {"spiking": {}, "analog": {}}
    selection_summaries: dict[str, dict[str, dict]] = {"spiking": {}, "analog": {}}
    for model_type in ("spiking", "analog"):
        for seed in SEEDS:
            epoch, history = select_epoch(
                X_train,
                y_train,
                X_validation,
                y_validation,
                dataset.X.shape[1],
                config,
                seed=seed,
                max_epochs=args.lock_epochs,
                device=device,
                model_type=model_type,
            )
            _, best = best_history_score(history)
            selected_epochs[model_type][str(seed)] = epoch
            selection_summaries[model_type][str(seed)] = {
                "selected_epoch": epoch,
                "balanced_accuracy": best["balanced_accuracy"],
                "macro_f1": best["macro_f1"],
                "log_loss": best["log_loss"],
                "activity": best["spike_rate"],
            }

    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    payload = {
        "lock_version": LOCK_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "source": dataset.source,
            "tuning_trials": int(development.sum()),
            "channels": int(dataset.X.shape[1]),
            "samples": int(dataset.X.shape[2]),
            "fs": dataset.fs,
        },
        "tuning_protocol": {
            "development_subjects": development_subjects,
            "selection_train_subjects": sorted(
                set(development_subjects) - set(validation_subjects)
            ),
            "validation_subjects": validation_subjects,
            "forbidden_during_tuning": list(range(21, 110)),
            "objective": "balanced_accuracy - 0.01 * log_loss",
            "sampler": "Optuna TPE",
            "trials_requested": args.trials,
            "trials_completed": len(completed),
            "trials_pruned": sum(
                trial.state.name == "PRUNED" for trial in study.trials
            ),
            "elapsed_seconds": elapsed,
            "device": str(device),
        },
        "config": asdict(config),
        "selected_epochs": selected_epochs,
        "selection_summaries": selection_summaries,
        "best_trial": {
            "number": study.best_trial.number,
            "objective": study.best_value,
            "attributes": study.best_trial.user_attrs,
        },
        "evaluation_plan": {
            "primary_fresh_subjects": list(range(31, 51)),
            "fixed_config_expansion_subjects": list(range(31, 110)),
            "subjects_21_30_are_prior_evidence_only": True,
        },
    }
    lock = payload | {"lock_sha256": canonical_hash(payload)}
    verify_lock(lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"locked {args.output} sha256={lock['lock_sha256']}")


if __name__ == "__main__":
    main()
