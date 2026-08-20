#!/usr/bin/env python3
"""Download PhysioNet EEGBCI left/right motor-imagery epochs into benchmark NPZ."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--subject-range", help="inclusive range such as 1-109")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.subjects and args.subject_range:
        raise ValueError("use either --subjects or --subject-range")
    if args.subject_range:
        start, end = (int(value) for value in args.subject_range.split("-", 1))
        subjects = list(range(start, end + 1))
    else:
        subjects = args.subjects or [1, 2, 3, 4, 5]

    try:
        import mne
        from mne.datasets import eegbci
    except ImportError as error:
        raise SystemExit("MNE is required: run this script with `uv run --with mne`") from error

    def prepare_subject(subject: int):
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
        print(f"subject {subject}: {data.shape[0]} trials")
        return (
            data,
            epochs.events[:, -1].astype(np.int64),
            np.full(data.shape[0], subject, dtype=np.int64),
        )

    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        prepared = list(executor.map(prepare_subject, subjects))
    trials, labels, subject_ids = zip(*prepared)

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
