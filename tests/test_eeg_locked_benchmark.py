import unittest

import numpy as np
import torch

from benchmarks.eeg_motor_imagery.strong_baselines import (
    FBCSPClassifier,
    RiemannianTSClassifier,
)
from benchmarks.eeg_motor_imagery.lock import canonical_hash, verify_lock
from benchmarks.eeg_motor_imagery.train_spiking_eegnet import (
    AnalogEEGNet,
    SpikingEEGNet,
    TrainConfig,
)


class LockedEEGBenchmarkTests(unittest.TestCase):
    def test_analog_control_matches_spiking_parameter_count(self):
        config = TrainConfig(
            temporal_filters=2,
            depth_multiplier=1,
            hidden=8,
            temporal_kernel=9,
            pool=4,
        )
        spiking = SpikingEEGNet(3, config=config)
        analog = AnalogEEGNet(3, config=config)
        self.assertEqual(
            sum(parameter.numel() for parameter in spiking.parameters()),
            sum(parameter.numel() for parameter in analog.parameters()),
        )
        logits, activity = analog(torch.randn(5, 3, 64))
        self.assertEqual(tuple(logits.shape), (5, 2))
        self.assertEqual(float(activity), 0.0)

    def test_lock_hash_rejects_mutation(self):
        payload = {"config": {"hidden": 64}, "subjects": list(range(1, 21))}
        lock = payload | {"lock_sha256": canonical_hash(payload)}
        verify_lock(lock)
        lock["config"]["hidden"] = 128
        with self.assertRaisesRegex(ValueError, "hash"):
            verify_lock(lock)

    def test_strong_baselines_return_probabilities(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(24, 3, 128))
        y = np.tile([0, 1], 12)
        t = np.arange(128) / 128.0
        X[y == 0, 0] += np.sin(2 * np.pi * 10 * t)
        X[y == 1, 2] += np.sin(2 * np.pi * 10 * t)
        for model in (FBCSPClassifier(128.0), RiemannianTSClassifier(128.0)):
            probability = model.fit(X[:20], y[:20]).predict_proba(X[20:])
            self.assertEqual(probability.shape, (4,))
            self.assertTrue(np.all((0 <= probability) & (probability <= 1)))


if __name__ == "__main__":
    unittest.main()
