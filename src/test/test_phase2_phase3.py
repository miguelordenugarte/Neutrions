from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features import augment_derived_features
from evaluation.metrics import compute_spectral_distortion
from models.MLP_model import MLPModel
from scripts import run_feature_ablation, verify_distributions
from trainer import _load_data, _load_false_training_data


ACTIVE_FEATURES = [
    "E_1", "z_1", "r2_1", "E_2", "z_2", "r2_2",
    "dt_shifted", "dR", "dwall_1", "dwall_2",
]


def _make_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    on_features: list[dict[str, float]] = []
    labels: list[dict[str, int]] = []
    for block in range(24):
        for pair in range(8):
            signal = pair % 2
            row = {
                "t_1": float(block * 1_000 + 500),
                "t_2": float(block * 1_000 + 500 + pair * 100 + 1),
                "E_1": float(block * 100 + pair),
                "E_2": float(1.0 + signal * 0.2 + pair * 0.01),
                "x_1": float(pair * 0.01),
                "y_1": float(pair * 0.02),
                "z_1": float(block * 0.01 + signal * 0.1),
                "z_2": float(block * 0.01 + signal * 0.1 + 0.02),
                "r_1": float(0.1 + pair * 0.01),
                "r_2": float(0.12 + pair * 0.01),
                "r2_1": float(0.01 + pair * 0.001),
                "r2_2": float(0.02 + pair * 0.001),
                "phi_1": 0.05,
                "phi_2": 2 * np.pi - 0.05,
                "D_1": 1.0,
                "D_2": 1.1,
                "dwall_1": 0.1 + pair * 0.001,
                "dwall_2": 0.2 + pair * 0.001,
                "dt": float(pair * 100 + 1),
                "dE": float(signal * 0.2 + pair * 0.01),
                "dx": 0.01,
                "dy": 0.02,
                "dz": float(signal * 0.1 + 0.02),
                "dr": 0.03 + pair * 0.001,
                "dr2": 0.04 + pair * 0.001,
                "dphi": float((2 * np.pi - 0.1) if pair % 2 else 0.1),
                "dD": 0.1,
                "dR": 0.2 + pair * 0.001,
                "k_window": 0,
                "dt_shifted": float((pair + 1) * 200_000),
                "n_before": 0,
                "n_on": 0,
                "n_gap": 0,
                "n_in_window": 1,
                "is_dc": 0,
            }
            on_features.append(row)
            event_1 = block * 1_000 + pair * 2
            labels.append({
                "is_ibd": signal,
                "false_ibd": 0,
                "event_id_1": event_1,
                "event_id_2": event_1 + 1,
                "time_block_id_1": block,
                "time_block_id_2": block,
            })

    on_features_df = pd.DataFrame(on_features)
    labels_df = pd.DataFrame(labels)
    off_features_df = on_features_df.copy()
    off_features_df["E_1"] += 10_000
    off_features_df["k_window"] = np.tile(np.arange(1, 9), 24)
    off_labels_df = labels_df.copy()
    off_labels_df["is_ibd"] = 0
    off_labels_df["false_ibd"] = np.tile([1, 0, 0, 0, 0, 0, 0, 0], 24)
    return on_features_df, labels_df, off_features_df, off_labels_df


class Phase2Phase3Tests(unittest.TestCase):
    def test_derived_features_are_periodic_stateless_and_validated(self) -> None:
        dphi = np.array([0.1, 2 * np.pi - 0.1])
        source = pd.DataFrame({"dphi": dphi, "dt_shifted": [0.0, 1_000_000.0]})
        result = augment_derived_features(
            source, ["cos_dphi", "sin_dphi", "log1p_dt_shifted"]
        )
        self.assertAlmostEqual(result.loc[0, "cos_dphi"], result.loc[1, "cos_dphi"])
        self.assertAlmostEqual(result.loc[0, "sin_dphi"], -result.loc[1, "sin_dphi"])
        np.testing.assert_allclose(
            result["log1p_dt_shifted"].to_numpy(),
            np.log1p(np.array([0.0, 1_000_000.0])),
        )
        self.assertNotIn("cos_dphi", source.columns)
        self.assertIs(
            augment_derived_features(source, ["dphi", "dt_shifted"]),
            source,
        )

        with self.assertRaisesRegex(ValueError, "non-negative"):
            augment_derived_features(
                pd.DataFrame({"dt_shifted": [1.0, -1.0]}),
                ["log1p_dt_shifted"],
            )

    def test_trainer_loads_derived_columns_for_on_time_and_off_time(self) -> None:
        on_features, on_labels, off_features, off_labels = _make_tables()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {}
            for name, frame in (
                ("on_features", on_features),
                ("on_labels", on_labels),
                ("off_features", off_features),
                ("off_labels", off_labels),
            ):
                path = root / f"{name}.parquet"
                pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
                paths[name] = str(path)
            config = {
                "data": {
                    "features_path": paths["on_features"],
                    "labels_path": paths["on_labels"],
                    "false_features_path": paths["off_features"],
                    "false_labels_path": paths["off_labels"],
                    "target_col": "is_ibd",
                    "positive_label": 1,
                    "split_strategy": "chronological_time_blocks",
                },
                "features": {"include": ["cos_dphi", "log1p_dt_shifted"], "exclude": []},
            }
            on_X, _, names, _ = _load_data(config)
            off_X, off_y, off_metadata = _load_false_training_data(
                config, names
            )

        self.assertEqual(names, ["cos_dphi", "log1p_dt_shifted"])
        self.assertIn("event_id_1", off_metadata)
        self.assertEqual(on_X.shape[1], 2)
        self.assertEqual(off_X.shape[1], 2)
        self.assertTrue(np.all(off_y == 0))
        np.testing.assert_allclose(on_X[:, 0], np.cos(on_features["dphi"]))

    def test_spectral_distortion_identical_shape_and_low_energy_loss(self) -> None:
        all_energies = np.repeat(np.arange(10, dtype=np.float64), 100)
        uniformly_selected = np.repeat(np.arange(10, dtype=np.float64), 50)
        same = compute_spectral_distortion(all_energies, uniformly_selected, bins=10)
        self.assertEqual(same["js_distance"], 0.0)
        self.assertEqual(same["max_relative_efficiency_deviation"], 0.0)

        low_energy_dropped = all_energies[all_energies >= 2]
        distorted = compute_spectral_distortion(
            all_energies, low_energy_dropped, bins=10
        )
        self.assertGreater(distorted["js_distance"], 0.0)
        self.assertGreater(distorted["max_relative_efficiency_deviation"], 0.0)

    def test_distribution_audit_cli_writes_report_and_figures(self) -> None:
        on_features, on_labels, off_features, off_labels = _make_tables()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {}
            for name, frame in (
                ("events_features", on_features),
                ("events_labels", on_labels),
                ("false_events_features", off_features),
                ("false_events_labels", off_labels),
            ):
                path = root / f"{name}.parquet"
                pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
                paths[name] = str(path)
            config = {
                "data": {
                    "features_path": paths["events_features"],
                    "labels_path": paths["events_labels"],
                    "false_features_path": paths["false_events_features"],
                    "false_labels_path": paths["false_events_labels"],
                    "target_col": "is_ibd",
                },
                "features": {"include": ACTIVE_FEATURES, "exclude": []},
            }
            config_path = root / "training.yaml"
            original_config = yaml.safe_dump(config)
            config_path.write_text(original_config, encoding="utf-8")
            output_dir = root / "diagnostics"
            result = verify_distributions.main([
                "--config", str(config_path),
                "--output-dir", str(output_dir),
                "--sample", "180",
            ])

            self.assertEqual(result, 0)
            report = json.loads((output_dir / "distribution_audit.json").read_text())
            sampled_off_labels = off_labels.sample(n=180, random_state=42)
            self.assertEqual(
                report["row_counts"]["off_time_false_ibd_nonzero"],
                int((sampled_off_labels["false_ibd"] != 0).sum()),
            )
            self.assertIn("dt", report["background_invariance"]["global"])
            self.assertIn(
                "first_1_percent",
                report["dt_shifted_edge_occupancy"]["on_time_background"],
            )
            self.assertTrue((output_dir / "dt_shifted_boundaries.png").exists())
            self.assertEqual(len(list(output_dir.glob("distribution_*.png"))), 10)
            self.assertEqual(config_path.read_text(encoding="utf-8"), original_config)

    def test_ablation_cli_only_scores_validation_and_does_not_mutate_config(self) -> None:
        on_features, on_labels, off_features, off_labels = _make_tables()

        class RecordingModel(MLPModel):
            predictions: list[np.ndarray] = []
            fitted_train: list[np.ndarray] = []

            def fit(self, X_train, y_train, X_val, y_val):
                self.__class__.fitted_train.append(X_train.copy())
                return super().fit(X_train, y_train, X_val, y_val)

            def predict_proba(self, X):
                self.__class__.predictions.append(X.copy())
                return super().predict_proba(X)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {}
            for name, frame in (
                ("events_features", on_features),
                ("events_labels", on_labels),
                ("false_events_features", off_features),
                ("false_events_labels", off_labels),
            ):
                path = root / f"{name}.parquet"
                pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
                paths[name] = str(path)
            parameters_path = root / "parameters.npz"
            np.savez(parameters_path, time_origin=0)
            config = {
                "data": {
                    "features_path": paths["events_features"],
                    "labels_path": paths["events_labels"],
                    "false_features_path": paths["false_events_features"],
                    "false_labels_path": paths["false_events_labels"],
                    "parameters_path": str(parameters_path),
                    "target_col": "is_ibd",
                    "positive_label": 1,
                    "split_strategy": "chronological_time_blocks",
                    "block_duration_ns": 1_000,
                    "boundary_guard_ns": 100,
                    "random_seed": 9,
                    "splits": {
                        "train": 0.60, "val": 0.15,
                        "calibration": 0.10, "test": 0.15,
                    },
                },
                "features": {"include": ACTIVE_FEATURES, "exclude": []},
                "threshold": {"n_steps": 100},
            }
            config_path = root / "training.yaml"
            original_config = yaml.safe_dump(config)
            config_path.write_text(original_config, encoding="utf-8")
            params_path = root / "model_params.json"
            params_path.write_text(json.dumps({
                "hidden_units": [4],
                "epochs": 1,
                "batch_size": 16,
                "patience": 1,
                "use_bn": False,
            }), encoding="utf-8")
            output_dir = root / "ablations"

            with patch.object(run_feature_ablation, "MLPModel", RecordingModel):
                result = run_feature_ablation.main([
                    "--config", str(config_path),
                    "--model-params-json", str(params_path),
                    "--output-dir", str(output_dir),
                    "--target-efficiency", "0.95",
                    "--seed", "9",
                ])
            self.assertEqual(config_path.read_text(encoding="utf-8"), original_config)
            self.assertEqual(result, 0)
            report = json.loads((output_dir / "ablation_summary.json").read_text())
            self.assertFalse(report["test_split_accessed"])
            self.assertEqual(len(report["feature_sets"]), 5)
            self.assertEqual(len(RecordingModel.predictions), 5)
            for prediction_input in RecordingModel.predictions:
                self.assertTrue(np.all((prediction_input[:, 0] >= 1_400)
                                       & (prediction_input[:, 0] < 1_800)))
            for training_input in RecordingModel.fitted_train:
                self.assertTrue(np.any(training_input[:, 0] >= 10_000))
            self.assertTrue((output_dir / "ablation_summary.csv").exists())
            self.assertTrue((output_dir / "validation_efficiency_vs_E1.png").exists())


if __name__ == "__main__":
    unittest.main()
