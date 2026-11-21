#!/usr/bin/env python3
"""Audit preprocessed on-time and off-time pair distributions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy.stats import ks_2samp, wasserstein_distance

from constants import (
    DETECTOR_HALF_HEIGHT_M,
    DETECTOR_RADIUS_M,
    PAIR_THRESHOLD_NS,
)
from features import DERIVED_FEATURE_SOURCES, augment_derived_features


RANGE_FEATURES = (
    "E_1", "E_2", "r_1", "r_2", "z_1", "z_2", "dR",
    "dwall_1", "dwall_2", "dt_shifted",
)
TIME_QUANTILE_COLUMN = "time_block_id_1"


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


def _read_pair_table(
    feature_path: Path,
    label_path: Path,
    requested_features: list[str],
    *,
    target_col: str,
    sample: int | None,
    seed: int,
) -> pd.DataFrame:
    feature_schema = pq.read_schema(feature_path).names
    label_schema = pq.read_schema(label_path).names
    required_features = list(dict.fromkeys(
        source
        for name in (
            *requested_features,
            *RANGE_FEATURES,
            "dphi",
            "dt",
            "k_window",
        )
        for source in DERIVED_FEATURE_SOURCES.get(name, (name,))
    ))
    missing_features = [name for name in required_features if name not in feature_schema]
    required_labels = [target_col, TIME_QUANTILE_COLUMN]
    if "false_ibd" in label_schema:
        required_labels.append("false_ibd")
    missing_labels = [name for name in required_labels if name not in label_schema]
    if missing_features or missing_labels:
        details = []
        if missing_features:
            details.append("feature columns: " + ", ".join(missing_features))
        if missing_labels:
            details.append("label columns: " + ", ".join(missing_labels))
        raise ValueError(
            f"Required columns are missing from {feature_path} / {label_path}: "
            + "; ".join(details)
        )

    features = pq.read_table(feature_path, columns=required_features).to_pandas()
    labels = pq.read_table(label_path, columns=required_labels).to_pandas()
    if len(features) != len(labels):
        raise ValueError(
            f"Features and labels row counts differ: {len(features)} != {len(labels)}."
        )
    features = augment_derived_features(features, requested_features)
    data = features.join(labels.reset_index(drop=True))
    if sample is not None and len(data) > sample:
        data = data.sample(n=sample, random_state=seed).reset_index(drop=True)
    return data


def _finite_values(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).ravel()
    return array[np.isfinite(array)]


def _distribution_stats(
    on_values: pd.Series | np.ndarray,
    off_values: pd.Series | np.ndarray,
    ks_threshold: float,
    wasserstein_threshold: float,
) -> dict[str, Any]:
    on = _finite_values(on_values)
    off = _finite_values(off_values)
    if on.size == 0 or off.size == 0:
        return {
            "n_on_time": int(on.size),
            "n_off_time": int(off.size),
            "ks_statistic": None,
            "ks_pvalue": None,
            "wasserstein": None,
            "pooled_iqr": None,
            "iqr_normalized_wasserstein": None,
            "flagged": None,
        }

    ks = ks_2samp(on, off, alternative="two-sided", method="auto")
    distance = float(wasserstein_distance(on, off))
    pooled_iqr = float(np.subtract(*np.quantile(np.concatenate((on, off)), [0.75, 0.25])))
    normalized_distance = (
        distance / pooled_iqr
        if pooled_iqr > 0
        else (0.0 if distance == 0 else None)
    )
    flagged = (
        float(ks.statistic) > ks_threshold
        or (normalized_distance is None and distance > 0)
        or (
            normalized_distance is not None
            and normalized_distance > wasserstein_threshold
        )
    )
    return {
        "n_on_time": int(on.size),
        "n_off_time": int(off.size),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "wasserstein": distance,
        "pooled_iqr": pooled_iqr,
        "iqr_normalized_wasserstein": normalized_distance,
        "flagged": bool(flagged),
    }


def _summary_statistics(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, Any]:
    quantiles = (0.001, 0.01, 0.5, 0.99, 0.999)
    result: dict[str, Any] = {}
    for column in columns:
        values = _finite_values(frame[column])
        if values.size == 0:
            result[column] = {"count": 0, "min": None, "max": None, "quantiles": {}}
            continue
        result[column] = {
            "count": int(values.size),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "quantiles": {
                str(q): float(np.quantile(values, q))
                for q in quantiles
            },
        }
    return result


def _wall_negative_counts(frame: pd.DataFrame) -> dict[str, Any]:
    n = len(frame)
    wall_1 = frame["dwall_1"].to_numpy(dtype=np.float64) < 0
    wall_2 = frame["dwall_2"].to_numpy(dtype=np.float64) < 0
    return {
        "rows": int(n),
        "dwall_1_negative": {
            "count": int(wall_1.sum()),
            "fraction": float(wall_1.mean()) if n else None,
        },
        "dwall_2_negative": {
            "count": int(wall_2.sum()),
            "fraction": float(wall_2.mean()) if n else None,
        },
        "either_negative": {
            "count": int((wall_1 | wall_2).sum()),
            "fraction": float((wall_1 | wall_2).mean()) if n else None,
        },
    }


def _zero_time_counts(frame: pd.DataFrame, threshold_ns: int) -> dict[str, Any]:
    result: dict[str, Any] = {"rows": int(len(frame))}
    for column, target in (
        ("dt", 0.0),
        ("dt_shifted", 0.0),
        ("dt_shifted", float(threshold_ns)),
    ):
        key = f"{column}_equals_{target:g}"
        values = frame[column].to_numpy(dtype=np.float64)
        result[key] = {
            "count": int(np.count_nonzero(values == target)),
            "fraction": float(np.mean(values == target)) if len(values) else None,
        }
    for column in ("dt", "dt_shifted"):
        values = frame[column].to_numpy(dtype=np.float64)
        nonzero = values[np.isfinite(values) & (values != 0)]
        result[f"{column}_minimum_nonzero"] = (
            float(np.min(nonzero)) if nonzero.size else None
        )
    return result


def _edge_occupancy(frame: pd.DataFrame, threshold_ns: int) -> dict[str, Any]:
    values = frame["dt_shifted"].to_numpy(dtype=np.float64)
    n = len(values)
    first = (values >= 0) & (values <= 0.01 * threshold_ns)
    last = (values >= 0.99 * threshold_ns) & (values <= threshold_ns)
    return {
        "rows": int(n),
        "first_1_percent": {
            "range_ns": [0.0, 0.01 * threshold_ns],
            "count": int(first.sum()),
            "fraction": float(first.mean()) if n else None,
        },
        "last_1_percent": {
            "range_ns": [0.99 * threshold_ns, float(threshold_ns)],
            "count": int(last.sum()),
            "fraction": float(last.mean()) if n else None,
        },
    }


def _time_quantile_ids(
    on_time: pd.DataFrame,
    off_time: pd.DataFrame,
) -> tuple[np.ndarray, dict[int, int]]:
    blocks = np.unique(np.concatenate((
        on_time[TIME_QUANTILE_COLUMN].to_numpy(dtype=np.int64),
        off_time[TIME_QUANTILE_COLUMN].to_numpy(dtype=np.int64),
    )))
    quartiles = np.array_split(blocks, 4)
    mapping = {
        int(block): quartile
        for quartile, values in enumerate(quartiles, start=1)
        for block in values
    }
    return blocks, mapping


def _compare_distributions(
    on_background: pd.DataFrame,
    off_background: pd.DataFrame,
    features: list[str],
    *,
    ks_threshold: float,
    wasserstein_threshold: float,
) -> tuple[dict[str, Any], list[str]]:
    global_results: dict[str, Any] = {}
    flagged_features: set[str] = set()
    flagged_global_features: set[str] = set()
    flagged_stratified_features: set[str] = set()
    k_window_results: dict[str, Any] = {}
    time_results: dict[str, Any] = {}

    for feature in features:
        global_stats = _distribution_stats(
            on_background[feature],
            off_background[feature],
            ks_threshold,
            wasserstein_threshold,
        )
        global_results[feature] = global_stats
        if global_stats["flagged"]:
            flagged_features.add(feature)
            flagged_global_features.add(feature)

        per_window: dict[str, Any] = {}
        for k_window in sorted(
            int(value)
            for value in off_background["k_window"].dropna().unique()
        ):
            window_values = off_background.loc[
                off_background["k_window"] == k_window, feature
            ]
            stats = _distribution_stats(
                on_background[feature],
                window_values,
                ks_threshold,
                wasserstein_threshold,
            )
            per_window[str(k_window)] = stats
            if stats["flagged"]:
                flagged_features.add(feature)
                flagged_stratified_features.add(feature)
        k_window_results[feature] = per_window

        _, block_to_quartile = _time_quantile_ids(on_background, off_background)
        on_quartiles = on_background[TIME_QUANTILE_COLUMN].map(block_to_quartile)
        off_quartiles = off_background[TIME_QUANTILE_COLUMN].map(block_to_quartile)
        per_quartile: dict[str, Any] = {}
        for quartile in range(1, 5):
            on_values = on_background.loc[on_quartiles == quartile, feature]
            off_values = off_background.loc[off_quartiles == quartile, feature]
            stats = _distribution_stats(
                on_values,
                off_values,
                ks_threshold,
                wasserstein_threshold,
            )
            per_quartile[str(quartile)] = stats
            if stats["flagged"]:
                flagged_features.add(feature)
                flagged_stratified_features.add(feature)
        time_results[feature] = per_quartile

    _, block_to_quartile = _time_quantile_ids(on_background, off_background)
    report = {
        "global": global_results,
        "by_k_window_vs_on_time_background": k_window_results,
        "by_chronological_time_block_quartile": time_results,
        "time_block_quartiles": {
            str(quartile): [
                int(block)
                for block, assigned in block_to_quartile.items()
                if assigned == quartile
            ]
            for quartile in range(1, 5)
        },
        "flagged_features": sorted(flagged_features),
        "flagged_global_features": sorted(flagged_global_features),
        "flagged_in_stratified_checks": sorted(flagged_stratified_features),
    }
    return report, sorted(flagged_features)


def _histogram_edges(values: np.ndarray, bins: int = 80) -> np.ndarray:
    values = _finite_values(values)
    low, high = float(np.min(values)), float(np.max(values))
    if low == high:
        low -= 0.5
        high += 0.5
    return np.linspace(low, high, bins + 1)


def _save_feature_plot(
    feature: str,
    on_values: np.ndarray,
    off_values: np.ndarray,
    output_dir: Path,
) -> None:
    on_values = _finite_values(on_values)
    off_values = _finite_values(off_values)
    if on_values.size == 0 or off_values.size == 0:
        fig, axis = plt.subplots(figsize=(9, 4))
        axis.text(
            0.5, 0.5, "Insufficient finite values for both populations",
            ha="center", va="center", transform=axis.transAxes,
        )
        axis.set_title(f"Background distribution comparison: {feature}")
        axis.set_xlabel(feature)
        fig.tight_layout()
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature)
        fig.savefig(output_dir / f"distribution_{safe_name}.png", dpi=140)
        plt.close(fig)
        return
    edges = _histogram_edges(np.concatenate((on_values, off_values)))
    on_density, _ = np.histogram(on_values, bins=edges, density=True)
    off_density, _ = np.histogram(off_values, bins=edges, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    ratio = np.divide(
        on_density,
        off_density,
        out=np.full_like(on_density, np.nan, dtype=np.float64),
        where=off_density > 0,
    )

    fig, (hist_ax, ratio_ax) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    hist_ax.hist(on_values, bins=edges, density=True, alpha=0.55, label="On-time background")
    hist_ax.hist(off_values, bins=edges, density=True, alpha=0.55, label="Off-time background")
    hist_ax.set_ylabel("Normalized density")
    hist_ax.set_title(f"Background distribution comparison: {feature}")
    hist_ax.legend()
    ratio_ax.plot(centers, ratio, drawstyle="steps-mid")
    ratio_ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ratio_ax.set_ylabel("On / off")
    ratio_ax.set_xlabel(feature)
    fig.tight_layout()
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature)
    fig.savefig(output_dir / f"distribution_{safe_name}.png", dpi=140)
    plt.close(fig)


def _save_dt_boundary_plot(
    on_values: np.ndarray,
    off_values: np.ndarray,
    output_dir: Path,
    threshold_ns: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    regions = (
        ("First 1% of window", 0.0, 0.01 * threshold_ns),
        ("Last 1% of window", 0.99 * threshold_ns, float(threshold_ns)),
    )
    for axis, (title, low, high) in zip(axes, regions):
        on_edge = on_values[(on_values >= low) & (on_values <= high)]
        off_edge = off_values[(off_values >= low) & (off_values <= high)]
        if on_edge.size:
            axis.hist(
                on_edge, bins=40, density=True, alpha=0.55,
                label="On-time background",
            )
        if off_edge.size:
            axis.hist(
                off_edge, bins=40, density=True, alpha=0.55,
                label="Off-time background",
            )
        if not on_edge.size and not off_edge.size:
            axis.text(0.5, 0.5, "No entries in this edge range",
                      ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        axis.set_xlabel("dt_shifted (ns)")
        axis.set_xlim(low, high)
        axis.set_ylabel("Conditional density")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels)
    fig.suptitle("dt_shifted coincidence-window boundary audit")
    fig.tight_layout()
    fig.savefig(output_dir / "dt_shifted_boundaries.png", dpi=140)
    plt.close(fig)


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


def run_audit(
    config: dict[str, Any],
    *,
    output_dir: Path,
    sample: int | None = None,
    seed: int = 42,
    ks_threshold: float = 0.02,
    wasserstein_threshold: float = 0.05,
) -> dict[str, Any]:
    data_cfg = config["data"]
    feature_names = list(config["features"].get("include") or [])
    for feature in ("dphi", "dt"):
        if feature not in feature_names:
            feature_names.append(feature)
    if sample is not None and sample <= 0:
        raise ValueError("--sample must be a positive integer.")
    if ks_threshold < 0 or wasserstein_threshold < 0:
        raise ValueError("Drift thresholds must be non-negative.")

    on_time = _read_pair_table(
        _resolve_path(data_cfg["features_path"]),
        _resolve_path(data_cfg["labels_path"]),
        feature_names,
        target_col=data_cfg.get("target_col", "is_ibd"),
        sample=sample,
        seed=seed,
    )
    off_time = _read_pair_table(
        _resolve_path(data_cfg["false_features_path"]),
        _resolve_path(data_cfg["false_labels_path"]),
        feature_names,
        target_col=data_cfg.get("target_col", "is_ibd"),
        sample=sample,
        seed=seed,
    )
    target_col = data_cfg.get("target_col", "is_ibd")
    if "false_ibd" in off_time.columns:
        contaminated_false_count = int((off_time["false_ibd"] != 0).sum())
        off_background = off_time.loc[off_time["false_ibd"] == 0].copy()
    else:
        contaminated_false_count = 0
        off_background = off_time.loc[off_time[target_col] == 0].copy()

    on_signal = on_time.loc[on_time[target_col] == 1].copy()
    on_background = on_time.loc[on_time[target_col] == 0].copy()
    if on_background.empty or off_background.empty:
        raise ValueError("Both on-time and sanitized off-time backgrounds must be non-empty.")

    stats_features = list(dict.fromkeys(feature_names))
    comparison, flagged = _compare_distributions(
        on_background,
        off_background,
        stats_features,
        ks_threshold=ks_threshold,
        wasserstein_threshold=wasserstein_threshold,
    )
    groups = {
        "on_time_signal": on_signal,
        "on_time_background": on_background,
        "off_time_background": off_background,
    }
    report: dict[str, Any] = {
        "inputs": {
            "on_time_features": str(_resolve_path(data_cfg["features_path"])),
            "on_time_labels": str(_resolve_path(data_cfg["labels_path"])),
            "off_time_features": str(_resolve_path(data_cfg["false_features_path"])),
            "off_time_labels": str(_resolve_path(data_cfg["false_labels_path"])),
            "sample_per_file": sample,
            "seed": seed,
            "active_features": list(config["features"].get("include") or []),
            "comparison_features": stats_features,
        },
        "constants": {
            "pair_threshold_ns": PAIR_THRESHOLD_NS,
            "detector_radius_m": DETECTOR_RADIUS_M,
            "detector_half_height_m": DETECTOR_HALF_HEIGHT_M,
        },
        "row_counts": {
            "on_time_total": int(len(on_time)),
            "on_time_signal": int(len(on_signal)),
            "on_time_background": int(len(on_background)),
            "off_time_total": int(len(off_time)),
            "off_time_false_ibd_nonzero": contaminated_false_count,
            "off_time_sanitized_background": int(len(off_background)),
        },
        "drift_thresholds": {
            "ks_statistic": ks_threshold,
            "iqr_normalized_wasserstein": wasserstein_threshold,
        },
        "normalization": (
            "1D Wasserstein distance divided by the pooled on-time/off-time IQR."
        ),
        "background_invariance": comparison,
        "range_quantiles": {
            name: _summary_statistics(group, RANGE_FEATURES)
            for name, group in groups.items()
        },
        "negative_wall_distance": {
            name: _wall_negative_counts(group)
            for name, group in groups.items()
        },
        "zero_time": {
            name: _zero_time_counts(group, PAIR_THRESHOLD_NS)
            for name, group in groups.items()
        },
        "dt_shifted_edge_occupancy": {
            name: _edge_occupancy(group, PAIR_THRESHOLD_NS)
            for name, group in {
                "on_time_background": on_background,
                "off_time_background": off_background,
            }.items()
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "distribution_audit.json").open("w", encoding="utf-8") as file:
        json.dump(_json_safe(report), file, indent=2, allow_nan=False)

    for feature in list(config["features"].get("include") or []):
        _save_feature_plot(
            feature,
            on_background[feature].to_numpy(),
            off_background[feature].to_numpy(),
            output_dir,
        )
    _save_dt_boundary_plot(
        on_background["dt_shifted"].to_numpy(dtype=np.float64),
        off_background["dt_shifted"].to_numpy(dtype=np.float64),
        output_dir,
        PAIR_THRESHOLD_NS,
    )

    print("\nOn-time background vs. sanitized off-time background")
    print(
        f"{'Feature':<22} {'KS':>9} {'p-value':>12} {'W1/IQR':>10} "
        f"{'Global':>8} {'Any stratum':>12}"
    )
    stratified_flags = set(comparison["flagged_in_stratified_checks"])
    for feature in stats_features:
        row = comparison["global"][feature]
        if row["ks_statistic"] is None:
            print(f"{feature:<22} insufficient data")
            continue
        normalized_wasserstein = row["iqr_normalized_wasserstein"]
        normalized_text = (
            f"{normalized_wasserstein:>10.4f}"
            if normalized_wasserstein is not None
            else f"{'undefined':>10}"
        )
        print(
            f"{feature:<22} {row['ks_statistic']:>9.4f} "
            f"{row['ks_pvalue']:>12.3g} "
            f"{normalized_text} "
            f"{'DRIFT' if row['flagged'] else 'ok':>8} "
            f"{'DRIFT' if feature in stratified_flags else 'ok':>12}"
        )
    print(
        f"\nRows: on-time signal={len(on_signal)}, on-time background={len(on_background)}, "
        f"off-time total={len(off_time)}, sanitized off-time={len(off_background)}, "
        f"false_ibd != 0={contaminated_false_count}"
    )
    print(f"Flagged features: {', '.join(flagged) if flagged else 'none'}")
    print(f"Report and figures: {output_dir}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "training.yaml"))
    parser.add_argument("--sample", type=int, help="sample this many rows per Parquet pair")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs/diagnostics"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ks-threshold", type=float, default=0.02)
    parser.add_argument("--wasserstein-threshold", type=float, default=0.05)
    args = parser.parse_args(argv)
    run_audit(
        _load_config(args.config),
        output_dir=_resolve_path(args.output_dir),
        sample=args.sample,
        seed=args.seed,
        ks_threshold=args.ks_threshold,
        wasserstein_threshold=args.wasserstein_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
