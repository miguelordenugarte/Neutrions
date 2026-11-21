"""
evaluation/metrics.py
─────────────────────────────────────────────────────────────────────────────
Classification metrics for binary imbalanced problems.

All functions accept plain numpy arrays and return either a scalar or a dict.
No model-specific logic lives here — this module is intentionally stateless.

Recommended primary metric for threshold-independent evaluation: PR-AUC.
Accuracy is deliberately excluded: it is misleading under class imbalance.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_spectral_distortion(
    energies_all: np.ndarray,
    energies_selected: np.ndarray,
    bins: int | np.ndarray = 50,
    energy_range: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Measure prompt-energy shape changes caused by an event selection.

    Histograms use identical edges and are separately normalized to unit
    probability. ``reduced_chi2`` is a descriptive normalized-shape statistic:
    ``sum((Q_i - P_i)**2 / P_i) / (n_populated_bins - 1)`` over bins with
    ``P_i > 0``; it is not an inferential chi-square test because normalized
    bins are statistically dependent. The relative efficiency deviation is
    ``max(abs(epsilon_i - epsilon_global) / epsilon_global)`` for bins with at
    least 20 unselected true signal events.

    If the selection is empty, the shape metrics are undefined and returned as
    NaN. Energies are expected to be prompt energies for true signal pairs.
    """
    all_energy = np.asarray(energies_all, dtype=np.float64).ravel()
    selected_energy = np.asarray(energies_selected, dtype=np.float64).ravel()
    nan_metrics = {
        "js_distance": float("nan"),
        "reduced_chi2": float("nan"),
        "max_relative_efficiency_deviation": float("nan"),
    }

    if not np.isfinite(all_energy).all() or not np.isfinite(selected_energy).all():
        raise ValueError("Energy arrays must contain only finite values.")
    if all_energy.size == 0 or selected_energy.size == 0:
        return nan_metrics

    if energy_range is None:
        low = float(np.min(all_energy))
        high = float(np.max(all_energy))
        if low == high:
            low -= 0.5
            high += 0.5
        histogram_range = (low, high)
        bin_edges: int | np.ndarray = bins
    else:
        low, high = map(float, energy_range)
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError("energy_range must contain finite values with low < high.")
        histogram_range = (low, high)
        bin_edges = bins

    if isinstance(bins, (int, np.integer)) and bins <= 0:
        raise ValueError("bins must be a positive integer or increasing edge array.")
    if not isinstance(bins, (int, np.integer)):
        edge_array = np.asarray(bins, dtype=np.float64)
        if (
            edge_array.ndim != 1
            or len(edge_array) < 2
            or not np.isfinite(edge_array).all()
            or np.any(np.diff(edge_array) <= 0)
        ):
            raise ValueError("bins must be a positive integer or increasing edge array.")
        bin_edges = edge_array
        histogram_range = None

    all_counts, edges = np.histogram(
        all_energy, bins=bin_edges, range=histogram_range
    )
    selected_counts, _ = np.histogram(
        selected_energy, bins=edges
    )
    total_all = int(all_counts.sum())
    total_selected = int(selected_counts.sum())
    if total_all == 0 or total_selected == 0:
        return nan_metrics

    p = all_counts.astype(np.float64) / total_all
    q = selected_counts.astype(np.float64) / total_selected
    populated = p > 0
    chi_bins = int(populated.sum())
    reduced_chi2 = (
        float(np.sum((q[populated] - p[populated]) ** 2 / p[populated]) / (chi_bins - 1))
        if chi_bins > 1
        else float("nan")
    )

    global_efficiency = total_selected / total_all
    efficiency_bins = all_counts >= 20
    if global_efficiency > 0 and np.any(efficiency_bins):
        bin_efficiencies = selected_counts[efficiency_bins] / all_counts[efficiency_bins]
        max_relative_deviation = float(
            np.max(np.abs(bin_efficiencies - global_efficiency)) / global_efficiency
        )
    else:
        max_relative_deviation = float("nan")

    return {
        "js_distance": float(jensenshannon(p, q, base=2)),
        "reduced_chi2": reduced_chi2,
        "max_relative_efficiency_deviation": max_relative_deviation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute a full suite of binary classification metrics.

    Parameters
    ----------
    y_true    : Ground-truth binary labels (0 / 1).
    y_prob    : Predicted probabilities for the positive class.
    threshold : Decision threshold applied to y_prob.

    Returns
    -------
    dict with keys:
        pr_auc      — area under the Precision-Recall curve (primary metric)
        roc_auc     — area under the ROC curve
        f1          — F1 score at *threshold*
        precision   — precision at *threshold*
        recall      — recall (sensitivity) at *threshold*
        specificity — true negative rate at *threshold*
        mcc         — Matthews Correlation Coefficient
        g_mean      — geometric mean of sensitivity and specificity
        threshold   — the threshold used
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(
            f"y_true and y_prob length differ: {y_true.shape[0]} != {y_prob.shape[0]}."
        )
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob contains NaN or infinite values.")
    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("y_prob values must be within [0, 1].")
    if np.isin(y_true, (0, 1)).all() is False:
        raise ValueError("y_true must contain only binary labels 0 and 1.")

    y_pred = (y_prob >= threshold).astype(int)

    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    g_mean      = float(np.sqrt(sensitivity * specificity))

    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        pr_auc = float("nan")
        roc_auc = float("nan")
        log_loss_value = float("nan")
    else:
        pr_auc = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))
        log_loss_value = float(log_loss(y_true, y_prob, labels=[0, 1]))

    if unique_labels.size < 2:
        brier = float(brier_score_loss(y_true, y_prob))
    else:
        brier = float(brier_score_loss(y_true, y_prob))

    bin_edges = np.linspace(0.0, 1.0, 16)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=False), 0, 14)
    ece = 0.0
    for bin_id in range(15):
        mask = bin_ids == bin_id
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(y_true[mask].mean()) - float(y_prob[mask].mean())
            )

    return {
        "pr_auc":      pr_auc,
        "roc_auc":     roc_auc,
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":      float(sensitivity),
        "specificity": float(specificity),
        "mcc":         float(matthews_corrcoef(y_true, y_pred)),
        "g_mean":      g_mean,
        "threshold":   float(threshold),
        "brier_score": brier,
        "log_loss":    log_loss_value,
        "ece":         float(ece),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def evaluate_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Evaluate metrics at multiple thresholds.

    Useful for threshold sweeps and for building calibration curves.

    Parameters
    ----------
    y_true     : Ground-truth binary labels.
    y_prob     : Predicted probabilities for the positive class.
    thresholds : Array of thresholds to evaluate.  Defaults to 100 points
                 in [0, 1].

    Returns
    -------
    List of metric dicts, one per threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 100)
    return [evaluate(y_true, y_prob, t) for t in thresholds]