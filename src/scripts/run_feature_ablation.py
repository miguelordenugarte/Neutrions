#!/usr/bin/env python3
"""Compare predefined feature sets using only the chronological validation split."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Type

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from evaluation.metrics import compute_spectral_distortion, evaluate
from evaluation.thresholds import optimal_fixed_efficiency_threshold
from models.MLP_model import MLPModel
from models.base_model import BaseModel
from trainer import (
    _chronological_time_block_split,
    _load_data,
    _load_false_training_data,
)


BASELINE_10 = [
    "E_1", "z_1", "r2_1", "E_2", "z_2", "r2_2",
    "dt_shifted", "dR", "dwall_1", "dwall_2",
]
CANDIDATE_FEATURES: dict[str, list[str]] = {
    "baseline_10": BASELINE_10,
    "no_dwall_8": [
        "E_1", "z_1", "r2_1", "E_2", "z_2", "r2_2",
        "dt_shifted", "dR",
    ],
    "with_contrasts_13": BASELINE_10 + ["dE", "dz", "dr"],
    "relative_periodic_7": [
        "E_1", "E_2", "dt_shifted", "dR", "dr", "dz", "cos_dphi",
    ],
    "log_time_10": [
        "E_1", "z_1", "r2_1", "E_2", "z_2", "r2_2",
        "log1p_dt_shifted", "dR", "dwall_1", "dwall_2",
    ],
}


def _resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_config(config_path: str | Path) -> dict[str, Any]:
    path = _resolve_path(config_path)
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return config


def _resolve_model_params(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return MLPModel.resolve_params({})
    params_path = _resolve_path(path)
    with params_path.open(encoding="utf-8") as file:
        params = json.load(file)
    if not isinstance(params, dict):
        raise ValueError("Model parameters JSON must contain a JSON object.")
    if isinstance(params.get("resolved_params"), dict):
        params = params["resolved_params"]
    else:
        params = dict(params)
    params.pop("feature_names", None)
    params.pop("resolved_params", None)
    class_weight = params.get("class_weight")
    if isinstance(class_weight, dict):
        params["class_weight"] = {
            int(label): weight for label, weight in class_weight.items()
        }
    return MLPModel.resolve_params(params)


def _load_ablation_rows(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, np.ndarray], np.ndarray]:
    """Load shared rows once and return a mask identifying training-only pairs."""
    data_config = config["data"]
    if data_config.get("split_strategy") != "chronological_time_blocks":
        raise ValueError("Feature ablation requires chronological_time_blocks splitting.")
    if "false_features_path" not in data_config or "false_labels_path" not in data_config:
        raise ValueError("Feature ablation requires configured off-time feature and label files.")

    all_features = list(dict.fromkeys(
        feature
        for feature_set in CANDIDATE_FEATURES.values()
        for feature in feature_set
    ))
    load_config = copy.deepcopy(config)
    load_config["features"]["include"] = all_features
    load_config["features"]["exclude"] = []

    X_on, y_on, feature_names, metadata_on = _load_data(load_config)
    if metadata_on is None:
        raise ValueError("Chronological ablation requires on-time split metadata.")

    X_false, y_false, false_metadata = _load_false_training_data(
        load_config, feature_names
    )
    if false_metadata.keys() != metadata_on.keys():
        raise ValueError("On-time and sanitized off-time metadata schemas differ.")
    on_time_count = len(y_on)
    X = np.concatenate((X_on, X_false), axis=0).astype(np.float32, copy=False)
    y = np.concatenate((y_on, y_false))
    metadata = {
        key: np.concatenate((metadata_on[key], false_metadata[key]))
        for key in metadata_on
    }
    training_only_mask = np.concatenate((
        np.zeros(on_time_count, dtype=bool),
        np.ones(len(y_false), dtype=bool),
    ))
    return X, y, feature_names, metadata, training_only_mask


def _time_origin(config: dict[str, Any]) -> int:
    parameters_path = _resolve_path(config["data"]["parameters_path"])
    if not parameters_path.exists():
        raise FileNotFoundError(f"Preprocessing parameters not found: {parameters_path}.")
    with np.load(parameters_path) as parameters:
        return int(parameters["time_origin"])


def _validation_efficiency_curve(
    energies: np.ndarray,
    selected: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    all_counts, _ = np.histogram(energies, bins=edges)
    selected_counts, _ = np.histogram(energies[selected], bins=edges)
    efficiency = np.divide(
        selected_counts,
        all_counts,
        out=np.full(all_counts.shape, np.nan, dtype=np.float64),
        where=all_counts > 0,
    )
    return (edges[:-1] + edges[1:]) / 2, efficiency


def _evaluate_validation_only(
    model: BaseModel,
    X_val: np.ndarray,
    y_val: np.ndarray,
    E1_val: np.ndarray,
    *,
    target_efficiency: float,
    n_threshold_steps: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Score only validation rows and derive its target-efficiency operating point."""
    scores = np.asarray(model.predict_proba(X_val), dtype=np.float64).ravel()
    if len(scores) != len(y_val):
        raise ValueError("Model returned a different number of scores than validation rows.")

    threshold, _ = optimal_fixed_efficiency_threshold(
        y_val,
        scores,
        target_efficiency=target_efficiency,
        n_steps=n_threshold_steps,
    )
    predictions = scores >= threshold
    metrics = evaluate(y_val, scores, threshold=threshold)
    true_signal = y_val == 1
    selected_signal = true_signal & predictions
    signal_count = int(selected_signal.sum())
    background_count = int(((y_val == 0) & predictions).sum())
    total_selected = signal_count + background_count
    significance = (
        float(signal_count / np.sqrt(total_selected))
        if total_selected > 0
        else 0.0
    )
    distortion = compute_spectral_distortion(
        E1_val[true_signal],
        E1_val[selected_signal],
    )
    metrics.update({
        "threshold": float(threshold),
        "signal_efficiency": float(metrics["recall"]),
        "background_rejection": float(metrics["specificity"]),
        "significance": significance,
        "spectral_distortion": distortion,
    })
    return metrics, predictions


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _save_efficiency_plot(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    for name, (centers, efficiencies) in curves.items():
        axis.plot(centers, efficiencies, marker="o", label=name)
    axis.set_xlabel("Prompt energy E_1")
    axis.set_ylabel("Validation signal efficiency")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Validation signal efficiency vs. prompt energy")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print("\nValidation-only feature ablation")
    columns = (
        ("Feature set", "feature_set"),
        ("PR-AUC", "pr_auc"),
        ("ROC-AUC", "roc_auc"),
        ("Brier", "brier_score"),
        ("Log loss", "log_loss"),
        ("Threshold", "threshold"),
        ("Signal eff.", "signal_efficiency"),
        ("Bkg reject.", "background_rejection"),
        ("Precision", "precision"),
        ("S/sqrt(S+B)", "significance"),
        ("JS dist.", "js_distance"),
        ("Red. chi2", "reduced_chi2"),
        ("Max rel. eff. dev.", "max_relative_efficiency_deviation"),
    )
    print(" | ".join(f"{title:>13}" for title, _ in columns))
    for row in rows:
        values = []
        for title, key in columns:
            if key == "feature_set":
                values.append(f"{row[key]:>13}")
            else:
                value = row.get(key)
                values.append(f"{value:13.5g}" if value is not None else f"{'nan':>13}")
        print(" | ".join(values))


def run_ablation(
    config: dict[str, Any],
    *,
    output_dir: Path,
    target_efficiency: float = 0.95,
    seed: int | None = None,
    model_params: dict[str, Any] | None = None,
    model_cls: Type[BaseModel] = MLPModel,
) -> dict[str, Any]:
    """Train five candidate models and score only the on-time validation split."""
    if not 0.0 <= target_efficiency <= 1.0:
        raise ValueError("target_efficiency must be within [0, 1].")
    run_config = copy.deepcopy(config)
    run_config["data"]["split_strategy"] = "chronological_time_blocks"
    run_config["data"]["include_false_events"] = True
    for path_key in (
        "features_path", "labels_path", "false_features_path",
        "false_labels_path", "parameters_path",
    ):
        if path_key in run_config["data"]:
            run_config["data"][path_key] = str(
                _resolve_path(run_config["data"][path_key])
            )
    if seed is not None:
        run_config["data"]["random_seed"] = int(seed)

    X, y, feature_names, metadata, training_only_mask = _load_ablation_rows(run_config)
    row_ids = np.arange(len(y), dtype=np.int64).reshape(-1, 1)
    requested_splits = _chronological_time_block_split(
        row_ids,
        y,
        metadata,
        run_config,
        time_origin=_time_origin(run_config),
        training_only_mask=training_only_mask,
        return_splits=("train", "val"),
    )
    if set(requested_splits) != {"train", "val"}:
        raise AssertionError("Ablation must materialize only train and validation row indices.")
    train_indices = requested_splits["train"][0][:, 0].astype(np.int64)
    validation_indices = requested_splits["val"][0][:, 0].astype(np.int64)

    feature_index = {name: index for index, name in enumerate(feature_names)}
    missing = sorted({
        feature
        for feature_set in CANDIDATE_FEATURES.values()
        for feature in feature_set
        if feature not in feature_index
    })
    if missing:
        raise ValueError("Ablation features were not loaded: " + ", ".join(missing))

    params = model_params or MLPModel.resolve_params({})
    threshold_steps = int(run_config.get("threshold", {}).get("n_steps", 500))
    energy_column = feature_index["E_1"]
    validation_labels = y[validation_indices]
    validation_energies = X[validation_indices, energy_column].astype(np.float64)
    signal_energies = validation_energies[validation_labels == 1]
    if signal_energies.size == 0:
        raise ValueError("Chronological validation split contains no true IBD pairs.")
    low, high = float(np.min(signal_energies)), float(np.max(signal_energies))
    if low == high:
        low -= 0.5
        high += 0.5
    energy_edges = np.linspace(low, high, 11)

    rows: list[dict[str, Any]] = []
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for feature_set_name, selected_features in CANDIDATE_FEATURES.items():
        np.random.seed(run_config["data"]["random_seed"])
        tf.keras.utils.set_random_seed(run_config["data"]["random_seed"])
        selected_columns = [feature_index[name] for name in selected_features]
        X_train = X[train_indices][:, selected_columns]
        X_validation = X[validation_indices][:, selected_columns]
        y_train = y[train_indices]
        y_validation = y[validation_indices]

        model = model_cls(copy.deepcopy(params))
        model.fit(X_train, y_train, X_validation, y_validation)
        metrics, selected_validation = _evaluate_validation_only(
            model,
            X_validation,
            y_validation,
            validation_energies,
            target_efficiency=target_efficiency,
            n_threshold_steps=threshold_steps,
        )
        curve = _validation_efficiency_curve(
            validation_energies,
            (y_validation == 1) & selected_validation,
            energy_edges,
        )
        curves[feature_set_name] = curve
        rows.append({
            "feature_set": feature_set_name,
            "features": list(selected_features),
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)),
            "off_time_training_rows": int(training_only_mask[train_indices].sum()),
            "target_efficiency": float(target_efficiency),
            **metrics,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "evaluation_population": "on-time chronological validation split only",
        "model_scores": "raw predict_proba scores; no calibration fitted",
        "threshold_selection_and_metrics_share_validation_split": True,
        "test_split_accessed": False,
        "training_only_false_pairs": "false_ibd == 0",
        "seed": run_config["data"]["random_seed"],
        "target_efficiency": target_efficiency,
        "resolved_model_params": params,
        "feature_sets": rows,
    }
    with (output_dir / "ablation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(_json_safe(report), file, indent=2, allow_nan=False)
    csv_rows = []
    for row in rows:
        csv_row = {key: value for key, value in row.items() if key != "spectral_distortion"}
        csv_row.update(row["spectral_distortion"])
        csv_row["features"] = ",".join(row["features"])
        csv_rows.append(csv_row)
    pd.DataFrame(csv_rows).to_csv(output_dir / "ablation_summary.csv", index=False)
    _save_efficiency_plot(
        curves,
        output_dir / "validation_efficiency_vs_E1.png",
    )
    _print_summary([
        {
            **row,
            **row["spectral_distortion"],
        }
        for row in rows
    ])
    print(f"\nSummary and plot: {output_dir}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "training.yaml"))
    parser.add_argument("--model-params-json")
    parser.add_argument("--target-efficiency", type=float, default=0.95)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs/ablations"),
    )
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    params = _resolve_model_params(args.model_params_json)
    run_ablation(
        config,
        output_dir=_resolve_path(args.output_dir),
        target_efficiency=args.target_efficiency,
        seed=args.seed,
        model_params=params,
        model_cls=MLPModel,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
