from __future__ import annotations

import unittest
import tempfile
import json
from pathlib import Path

import numpy as np
import joblib

from evaluation.calibrations import PlattCalibrator
from evaluation.metrics import evaluate
from evaluation.thresholds import (
    optimal_significance_threshold,
    optimal_asimov_threshold,
    optimal_punzi_threshold,
    optimal_fixed_efficiency_threshold,
)
from models.MLP_model import MLPModel
from trainer import _save_run


class EvaluationTests(unittest.TestCase):
    def test_platt_handles_boundary_probabilities(self) -> None:
        calibrator = PlattCalibrator()
        calibrator.fit(
            np.array([0.0, 0.1, 0.8, 1.0]),
            np.array([0, 0, 1, 1]),
        )
        calibrated = calibrator.predict(np.array([0.0, 0.5, 1.0]))
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue(np.all((calibrated >= 0.0) & (calibrated <= 1.0)))

    def test_evaluate_calibration_metrics_and_guards(self) -> None:
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.35, 0.9])
        metrics = evaluate(y, p)
        for key in ("brier_score", "log_loss", "ece"):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]))

        with self.assertRaisesRegex(ValueError, "within \\[0, 1\\]"):
            evaluate(y, np.array([0.1, 0.2, 1.1, 0.8]))
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            evaluate(y, np.array([0.1, 0.2, np.nan, 0.8]))

        one_class = evaluate(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
        self.assertTrue(np.isnan(one_class["pr_auc"]))
        self.assertTrue(np.isnan(one_class["roc_auc"]))

    def test_save_run_persists_calibrator_and_inference_manifest(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(20, 2)).astype(np.float32)
        y = np.tile([0, 1], 10)
        model = MLPModel({
            "hidden_units": [4],
            "epochs": 1,
            "batch_size": 4,
            "patience": 1,
            "use_bn": False,
        }).fit(X, y, X, y)
        model.feature_names = ["a", "b"]
        raw = model.predict_proba(X)
        calibrator = PlattCalibrator()
        calibrator.fit(raw, y)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            cfg = {"outputs": {"save_model": True, "save_metrics": False, "save_study": False}}
            _save_run(
                run_dir,
                model,
                calibrator,
                0.5,
                {"brier_score": 0.1},
                {"brier_score": 0.2},
                ["a", "b"],
                model.params,
                "platt",
                model.params,
                None,
                cfg,
            )
            restored_calibrator = joblib.load(run_dir / "model" / "calibrator.joblib")
            with open(run_dir / "model" / "inference_config.json", "r", encoding="utf-8") as f:
                manifest = json.load(f)
            restored_model = MLPModel.load(run_dir / "model")
            raw_restored = restored_model.predict_proba(X)

        np.testing.assert_allclose(raw_restored, raw, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            restored_calibrator.predict(raw),
            calibrator.predict(raw),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertEqual(manifest["feature_names"], ["a", "b"])
        self.assertEqual(manifest["dtype"], "float32")
        self.assertEqual(restored_model.feature_names, ["a", "b"])

    def test_physics_threshold_strategies(self) -> None:
        y = np.array([1, 1, 0, 0, 1, 0])
        p = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1])

        significance_threshold, significance = optimal_significance_threshold(
            y, p, n_steps=20
        )
        asimov_threshold, asimov = optimal_asimov_threshold(y, p, n_steps=20)
        punzi_threshold, punzi = optimal_punzi_threshold(
            y, p, punzi_a=3.0, n_steps=20
        )
        fixed_threshold, recall = optimal_fixed_efficiency_threshold(
            y, p, target_efficiency=0.95, n_steps=20
        )

        self.assertTrue(0.0 <= significance_threshold <= 1.0)
        self.assertTrue(np.isfinite(significance))
        self.assertTrue(0.0 <= asimov_threshold <= 1.0)
        self.assertTrue(np.isfinite(asimov))
        self.assertTrue(0.0 <= punzi_threshold <= 1.0)
        self.assertTrue(np.isfinite(punzi))
        self.assertTrue(0.0 <= fixed_threshold <= 1.0)
        self.assertGreaterEqual(recall, 0.95)


if __name__ == "__main__":
    unittest.main()
