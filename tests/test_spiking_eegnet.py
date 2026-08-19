import json
import unittest
from pathlib import Path

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

    def test_published_result_preserves_subject_separation(self):
        result = json.loads(
            Path("benchmarks/results/spiking_eegnet.json").read_text()
        )
        protocol = result["protocol"]
        development = set(protocol["development_subjects"])
        validation = set(protocol["validation_subjects"])
        test = set(protocol["test_subjects"])
        self.assertTrue(validation <= development)
        self.assertTrue(development.isdisjoint(test))
        self.assertTrue(protocol["test_evaluated_after_epoch_selection"])
        self.assertEqual(
            set(result["models"]),
            {"logistic_regression", "tuned_lif_reservoir", "spiking_eegnet"},
        )
        self.assertEqual(
            set(result["paired_comparisons"]),
            {
                "spiking_eegnet_vs_logistic_regression",
                "spiking_eegnet_vs_tuned_lif_reservoir",
            },
        )

    def test_kaggle_notebook_uses_public_main(self):
        notebook = json.loads(
            Path(
                "benchmarks/eeg_motor_imagery/kaggle_kernel/"
                "spiking-eegnet-motor-imagery-eeg.ipynb"
            ).read_text()
        )
        source = "".join(
            line
            for cell in notebook["cells"]
            for line in cell.get("source", [])
        )
        self.assertIn("'--branch', 'main'", source)
        self.assertNotIn("null/eeg-research-harness", source)


if __name__ == "__main__":
    unittest.main()
