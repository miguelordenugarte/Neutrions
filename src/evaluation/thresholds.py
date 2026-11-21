"""
evaluation/thresholds.py
─────────────────────────────────────────────────────────────────────────────
Threshold selection strategies for binary classification under class imbalance.

The default threshold of 0.5 is almost never optimal when classes are
imbalanced or when false negatives and false positives carry different costs.
Use the functions here to find a principled operating point after calibration.

Supported strategies:

* ``optimal_f1_threshold``       — maximises F1.
* ``optimal_gmean_threshold``    — maximises sensitivity × specificity.
* ``optimal_cost_threshold``     — minimises weighted FP and FN cost.
* ``optimal_significance_threshold`` — maximises S / sqrt(S + B).
* ``optimal_asimov_threshold``   — maximises the Asimov significance Z_A.
* ``optimal_punzi_threshold``    — maximises efficiency / (punzi_a/2 + sqrt(B)).
* ``optimal_fixed_efficiency_threshold`` — selects the highest threshold with recall
                                           at or above the target efficiency.

All strategies operate on held-out validation or calibration probabilities,
never on the test split, and return a single float threshold ready for deployment.
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Cost function
# ─────────────────────────────────────────────────────────────────────────────

def business_cost(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    threshold: float,
    fp_cost:   float = 1.0,
    fn_cost:   float = 10.0,
) -> float:
    """Compute the total business cost at a given threshold.

    Parameters
    ----------
    y_true    : Ground-truth binary labels.
    y_prob    : Predicted probabilities for the positive class.
    threshold : Decision threshold applied to y_prob.
    fp_cost   : Cost of one false positive (e.g. unnecessary follow-up).
    fn_cost   : Cost of one false negative (e.g. missed detection).
                Should be larger than fp_cost for signal-detection problems.

    Returns
    -------
    Total cost as a float.
    """
    y_pred = (y_prob >= threshold).astype(int)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return float(fp_cost * fp + fn_cost * fn)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold search
# ─────────────────────────────────────────────────────────────────────────────

def optimal_cost_threshold(
    y_true:     np.ndarray,
    y_prob:     np.ndarray,
    fp_cost:    float = 1.0,
    fn_cost:    float = 10.0,
    n_steps:    int   = 500,
) -> tuple[float, float]:
    """Find the threshold that minimises total business cost.

    Parameters
    ----------
    y_true   : Ground-truth binary labels.
    y_prob   : Predicted probabilities for the positive class.
    fp_cost  : Cost per false positive.
    fn_cost  : Cost per false negative.
    n_steps  : Number of candidate thresholds in [0, 1].

    Returns
    -------
    (best_threshold, best_cost)
    """
    thresholds = np.linspace(0.0, 1.0, n_steps)
    costs      = [
        business_cost(y_true, y_prob, t, fp_cost, fn_cost)
        for t in thresholds
    ]
    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), float(costs[best_idx])


def optimal_f1_threshold(
    y_true:  np.ndarray,
    y_prob:  np.ndarray,
    n_steps: int = 500,
) -> tuple[float, float]:
    """Find the threshold that maximises the F1 score.

    Parameters
    ----------
    y_true   : Ground-truth binary labels.
    y_prob   : Predicted probabilities for the positive class.
    n_steps  : Number of candidate thresholds in [0, 1].

    Returns
    -------
    (best_threshold, best_f1)
    """
    from sklearn.metrics import f1_score

    thresholds = np.linspace(0.0, 1.0, n_steps)
    f1_scores  = [
        f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def optimal_gmean_threshold(
    y_true:  np.ndarray,
    y_prob:  np.ndarray,
    n_steps: int = 500,
) -> tuple[float, float]:
    """Find the threshold that maximises the geometric mean (sensitivity × specificity).

    G-mean is robust to class imbalance and penalises models that ignore
    the minority class.  Recommended as a complement to F1 when both
    classes matter equally.

    Parameters
    ----------
    y_true   : Ground-truth binary labels.
    y_prob   : Predicted probabilities for the positive class.
    n_steps  : Number of candidate thresholds in [0, 1].

    Returns
    -------
    (best_threshold, best_g_mean)
    """
    thresholds = np.linspace(0.0, 1.0, n_steps)
    g_means    = []

    for t in thresholds:
        y_pred      = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        g_means.append(np.sqrt(sensitivity * specificity))

    best_idx = int(np.argmax(g_means))
    return float(thresholds[best_idx]), float(g_means[best_idx])


def optimal_significance_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_steps: int = 500,
) -> tuple[float, float]:
    """Maximise S / sqrt(S + B), where S and B are thresholded counts."""
    thresholds = np.linspace(0.0, 1.0, n_steps)
    scores: list[float] = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        signal = int(((y_pred == 1) & (y_true == 1)).sum())
        background = int(((y_pred == 1) & (y_true == 0)).sum())
        denominator = np.sqrt(signal + background)
        scores.append(float(signal / denominator) if denominator > 0 else 0.0)
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def optimal_asimov_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_steps: int = 500,
    background_floor: float = 1e-12,
) -> tuple[float, float]:
    """Maximise the Asimov significance Z_A."""
    thresholds = np.linspace(0.0, 1.0, n_steps)
    scores: list[float] = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        signal = float(((y_pred == 1) & (y_true == 1)).sum())
        background = float(((y_pred == 1) & (y_true == 0)).sum())
        background = max(background, background_floor)
        if signal + background <= 0:
            scores.append(0.0)
            continue
        inner = (signal + background) * np.log1p(signal / background) - signal
        scores.append(float(np.sqrt(max(2.0 * inner, 0.0))))
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def optimal_punzi_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    punzi_a: float = 3.0,
    n_steps: int = 500,
) -> tuple[float, float]:
    """Maximise signal efficiency divided by punzi_a/2 + sqrt(background)."""
    if not np.isfinite(punzi_a) or punzi_a < 0:
        raise ValueError("punzi_a must be finite and non-negative.")
    total_signal = int((y_true == 1).sum())
    thresholds = np.linspace(0.0, 1.0, n_steps)
    scores: list[float] = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        signal = int(((y_pred == 1) & (y_true == 1)).sum())
        background = int(((y_pred == 1) & (y_true == 0)).sum())
        efficiency = signal / total_signal if total_signal > 0 else 0.0
        denominator = punzi_a / 2.0 + np.sqrt(background)
        scores.append(float(efficiency / denominator) if denominator > 0 else 0.0)
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def optimal_fixed_efficiency_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_efficiency: float = 0.95,
    n_steps: int = 500,
) -> tuple[float, float]:
    """Select the highest threshold whose recall is at least target_efficiency."""
    if not np.isfinite(target_efficiency) or not 0.0 <= target_efficiency <= 1.0:
        raise ValueError("target_efficiency must be finite and within [0, 1].")
    total_signal = int((y_true == 1).sum())
    if total_signal == 0:
        return 1.0, 0.0
    thresholds = np.linspace(0.0, 1.0, n_steps)
    eligible: list[tuple[float, float]] = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        signal = int(((y_pred == 1) & (y_true == 1)).sum())
        recall = signal / total_signal
        if recall >= target_efficiency:
            eligible.append((float(threshold), float(recall)))
    if eligible:
        threshold, recall = eligible[-1]
        return threshold, recall
    return 1.0, 0.0