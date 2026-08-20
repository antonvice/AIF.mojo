"""Established motor-imagery EEG baselines with development-only fitting."""

from __future__ import annotations

import numpy as np


FILTER_BANK = (
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 24.0),
    (24.0, 32.0),
    (32.0, 40.0),
)


def fft_bandpass(
    X: np.ndarray, fs: float, low: float, high: float
) -> np.ndarray:
    spectrum = np.fft.rfft(X, axis=2)
    frequency = np.fft.rfftfreq(X.shape[2], 1.0 / fs)
    spectrum[:, :, (frequency < low) | (frequency > high)] = 0
    return np.fft.irfft(spectrum, n=X.shape[2], axis=2)


class FBCSPClassifier:
    """Filter-bank CSP features followed by balanced logistic regression."""

    def __init__(
        self,
        fs: float,
        bands: tuple[tuple[float, float], ...] = FILTER_BANK,
        components: int = 2,
    ):
        self.fs = fs
        self.bands = bands
        self.components = components

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FBCSPClassifier":
        from mne.decoding import CSP
        from sklearn.linear_model import LogisticRegression

        self.transforms = []
        features = []
        for low, high in self.bands:
            csp = CSP(
                n_components=min(self.components, X.shape[1]),
                reg="ledoit_wolf",
                log=True,
                norm_trace=False,
            )
            features.append(csp.fit_transform(fft_bandpass(X, self.fs, low, high), y))
            self.transforms.append((low, high, csp))
        self.classifier = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2_000, random_state=2026
        ).fit(np.concatenate(features, axis=1), y)
        return self

    def _features(self, X: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                csp.transform(fft_bandpass(X, self.fs, low, high))
                for low, high, csp in self.transforms
            ],
            axis=1,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self._features(X))[:, 1]


class RiemannianTSClassifier:
    """OAS covariance, affine-invariant tangent space, and logistic regression."""

    def __init__(self, fs: float, low: float = 8.0, high: float = 30.0):
        self.fs = fs
        self.low = low
        self.high = high

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RiemannianTSClassifier":
        from pyriemann.classification import TSClassifier
        from pyriemann.estimation import Covariances
        from sklearn.linear_model import LogisticRegression

        self.covariances = Covariances(estimator="oas")
        matrices = self.covariances.fit_transform(
            fft_bandpass(X, self.fs, self.low, self.high)
        )
        self.classifier = TSClassifier(
            metric="riemann",
            tsupdate=False,
            clf=LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2_000, random_state=2026
            ),
        ).fit(matrices, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        matrices = self.covariances.transform(
            fft_bandpass(X, self.fs, self.low, self.high)
        )
        return self.classifier.predict_proba(matrices)[:, 1]
