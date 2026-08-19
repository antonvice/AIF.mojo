import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks.eeg_motor_imagery.benchmark import (
    LIFConfig,
    LIFReservoir,
    SequentialAIF,
    bandpower_features,
    load_npz,
    run_benchmark,
    synthetic_dataset,
)


class EEGBenchmarkTests(unittest.TestCase):
    def test_loader_and_feature_shape(self):
        dataset = synthetic_dataset(subjects=3, trials_per_subject=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eeg.npz"
            np.savez(path, X=dataset.X, y=dataset.y, subject=dataset.subject, fs=dataset.fs)
            loaded = load_npz(path)
        self.assertEqual(loaded.X.shape, dataset.X.shape)
        self.assertEqual(bandpower_features(loaded.X, loaded.fs).shape, (24, 4, 8))

    def test_aif_posterior_normalizes_and_always_commits(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 4, 3))
        y = np.tile([0, 1], 10)
        model = SequentialAIF(samples=4).fit(X, y)
        posterior = model._update(np.array([0.5, 0.5]), X[0, 0], 0)
        probabilities, windows, actions = model.predict_proba(X[:4])
        self.assertAlmostEqual(float(posterior.sum()), 1.0)
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        self.assertTrue(np.all((windows >= 1) & (windows <= 4)))
        self.assertNotIn("wait", actions)

    def test_configurable_lif_is_deterministic(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(12, 4, 6))
        y = np.tile([0, 1], 6)
        config = LIFConfig(
            neurons=16,
            ticks=4,
            spectral_radius=0.5,
            recurrent_sparsity=0.25,
            ridge=0.1,
            reset="zero",
        )
        first = LIFReservoir(6, config=config, seed=9).fit(X, y)
        second = LIFReservoir(6, config=config, seed=9).fit(X, y)
        first_probability, first_spikes = first.predict_proba(X)
        second_probability, second_spikes = second.predict_proba(X)
        np.testing.assert_allclose(first_probability, second_probability)
        np.testing.assert_allclose(first_spikes, second_spikes)
        self.assertTrue(np.all((first_spikes >= 0) & (first_spikes <= 1)))

    def test_subject_wise_smoke_benchmark(self):
        dataset = synthetic_dataset(subjects=3, trials_per_subject=12)
        result = run_benchmark(dataset, holdout="0")
        self.assertEqual(result["dataset"]["evaluation"], "held-out subject")
        self.assertEqual(
            set(result["models"]),
            {"logistic_regression", "lif_snn", "sequential_aif"},
        )
        for metrics in result["models"].values():
            self.assertTrue(all(value == value for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
