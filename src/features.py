"""Stateless, on-demand feature derivations for paired-event data."""

from __future__ import annotations

import numpy as np
import pandas as pd


DERIVED_FEATURE_SOURCES = {
    "cos_dphi": ("dphi",),
    "sin_dphi": ("dphi",),
    "log1p_dt_shifted": ("dt_shifted",),
}


def augment_derived_features(
    df: pd.DataFrame,
    requested_features: list[str],
) -> pd.DataFrame:
    """Return ``df`` with requested derived columns computed on demand.

    ``cos_dphi`` and ``sin_dphi`` use the periodic angle ``dphi`` directly,
    so angles on opposite sides of the 0 / 2π seam remain close in feature
    space. ``log1p_dt_shifted`` uses time measured in nanoseconds: its
    dimensionless transform is ``log1p(dt_shifted / 1 ns)``, implemented as
    ``log1p(dt_shifted)`` when the stored numeric values are in ns.

    The input frame is not modified. If no derived feature is requested, the
    original frame is returned unchanged.
    """
    requested = set(requested_features)
    derived = requested.intersection(DERIVED_FEATURE_SOURCES)
    if not derived:
        return df

    required_sources = {
        source
        for feature in derived
        for source in DERIVED_FEATURE_SOURCES[feature]
    }
    missing = sorted(required_sources.difference(df.columns))
    if missing:
        raise ValueError(
            "Cannot compute requested derived features; missing source columns: "
            + ", ".join(missing)
        )

    result = df.copy()
    if "cos_dphi" in derived or "sin_dphi" in derived:
        dphi = df["dphi"].to_numpy(dtype=np.float64)
        if "cos_dphi" in derived:
            result["cos_dphi"] = np.cos(dphi)
        if "sin_dphi" in derived:
            result["sin_dphi"] = np.sin(dphi)

    if "log1p_dt_shifted" in derived:
        dt_shifted = df["dt_shifted"].to_numpy(dtype=np.float64)
        if np.any(dt_shifted < 0):
            raise ValueError("dt_shifted must be non-negative for log1p transformation.")
        result["log1p_dt_shifted"] = np.log1p(dt_shifted)

    return result
