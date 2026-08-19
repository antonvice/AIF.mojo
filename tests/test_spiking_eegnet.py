import unittest

import numpy as np
import torch

from benchmarks.eeg_motor_imagery.train_spiking_eegnet import (
    SpikingEEGNet,
    TrainConfig,
    fft_bandpass,
)


class SpikingEEGNetTests(unittest.TestCase):
    def test_forward_and_surrogate_backward(self):
        torch.manual_seed(0)
        config = TrainConfig(
            temporal_filters=2,
            depth_multiplier=1,
            hidden=8,
            temporal_kernel=9,
            pool=4,
        )
        model = SpikingEEGNet(channels=3, config=config)
        eeg = torch.randn(5, 3, 64)
        labels = torch.tensor([0, 1, 0, 1, 0])
        logits, spike_rate = model(eeg)
        torch.nn.functional.cross_entropy(logits, labels).backward()
        self.assertEqual(tuple(logits.shape), (5, 2))
        self.assertTrue(0 <= float(spike_rate.detach()) <= 1)
        self.assertTrue(
            all(
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )

    def test_fft_bandpass_preserves_shape_and_is_finite(self):
        rng = np.random.default_rng(0)
        eeg = rng.normal(size=(4, 3, 256))
        filtered = fft_bandpass(eeg, 128.0)
        self.assertEqual(filtered.shape, eeg.shape)
        self.assertTrue(np.isfinite(filtered).all())


if __name__ == "__main__":
    unittest.main()
