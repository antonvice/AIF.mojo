#!/usr/bin/env python3
"""Download PhysioNet EEGBCI left/right motor-imagery epochs into benchmark NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        import mne
        from mne.datasets import eegbci
    except ImportError as error:
        raise SystemExit("MNE is required: run this script with `uv run --with mne`") from error

    trials, labels, subject_ids = [], [], []
    for subject in args.subjects:
        files = eegbci.load_data(subject, runs=[4, 8, 12], update_path=False)
        raw = mne.concatenate_raws(
            [mne.io.read_raw_edf(path, preload=True, verbose="ERROR") for path in files]
        )
        eegbci.standardize(raw)
        raw.pick(["C3", "Cz", "C4"]).resample(128, verbose="ERROR")
        events, _ = mne.events_from_annotations(
            raw, event_id={"T1": 0, "T2": 1}, verbose="ERROR"
        )
        epochs = mne.Epochs(
            raw,
            events,
            event_id={"left": 0, "right": 1},
            tmin=0.0,
            tmax=2.0,
            baseline=None,
            preload=True,
            verbose="ERROR",
        )
        data = epochs.get_data(copy=True)[:, :, :256] * 1e6
        trials.append(data)
        labels.append(epochs.events[:, -1].astype(np.int64))
        subject_ids.append(np.full(data.shape[0], subject, dtype=np.int64))
        print(f"subject {subject}: {data.shape[0]} trials")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=np.concatenate(trials),
        y=np.concatenate(labels),
        subject=np.concatenate(subject_ids),
        fs=np.asarray(128.0),
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
