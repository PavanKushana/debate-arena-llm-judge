"""
krippendorff.py — Krippendorff's alpha (inter-annotator agreement).

Belongs in: judge_api/validation/krippendorff.py

WHY THIS IS HERE
----------------
Validity is a claim about correlation with human judgment, and that correlation
has a CEILING: a metric cannot be more consistent with "the human construct"
than the humans are with each other. Krippendorff's alpha measures exactly that
reliability of the human annotations. For the Dagstuhl cogency sub-dimensions,
expert agreement is only fair (alpha ~ .44–.47), so alpha is the number every
model correlation must be read against — a Spearman of .5 against a dimension
whose human alpha is .45 is near the achievable ceiling, not a weak result.

Self-contained (numpy only) so the validation harness has no heavy dependency.
Implements the coincidence-matrix formulation (Krippendorff 2011) with three
difference metrics and full missing-data tolerance.

INPUT CONVENTION
----------------
reliability_data is shaped (n_units, n_raters): one ROW per item, one COLUMN
per annotator. Missing ratings are np.nan. This matches how the Dagstuhl loader
yields per-dimension score matrices (row per argument, column per annotator).
(Note: some other libraries use raters × units — the transpose.)
"""
from __future__ import annotations

import numpy as np


def _coincidence_matrix(data: np.ndarray, values: list):
    """Build the symmetric coincidence matrix o[c,k] and marginals n_c.

    data : (n_units, n_raters) with np.nan for missing.
    Only units with >= 2 present ratings contribute; each within-unit ordered
    pair contributes 1/(m_u - 1), where m_u is that unit's number of ratings.
    """
    idx = {v: i for i, v in enumerate(values)}
    V = len(values)
    o = np.zeros((V, V), dtype=float)

    for row in data:
        ratings = row[~np.isnan(row)]
        m = len(ratings)
        if m < 2:
            continue
        w = 1.0 / (m - 1)
        for i in range(m):
            ci = idx[ratings[i]]
            for j in range(m):
                if i == j:
                    continue
                o[ci, idx[ratings[j]]] += w

    n_c = o.sum(axis=1)
    return o, n_c


def _delta2(i: int, j: int, level: str, values: list, n_c: np.ndarray) -> float:
    """Squared difference between category i and j under the chosen metric."""
    if level == "nominal":
        return 0.0 if i == j else 1.0
    if level == "interval":
        return float((values[i] - values[j]) ** 2)
    if level == "ordinal":
        lo, hi = (i, j) if i <= j else (j, i)
        # sum of marginal counts across the interval, minus half the endpoints
        s = float(n_c[lo:hi + 1].sum() - (n_c[i] + n_c[j]) / 2.0)
        return s * s
    raise ValueError(f"unknown level {level!r} (use nominal/ordinal/interval)")


def alpha(reliability_data, level: str = "ordinal") -> float:
    """
    Krippendorff's alpha for reliability_data shaped (n_units, n_raters).

    level : "nominal", "ordinal" (default; right for 1–3 Likert dimensions),
            or "interval".

    Returns a float in (-inf, 1].  1.0 = perfect agreement; ~0 = chance;
    negative = systematic disagreement. Returns nan if fewer than 2 pairable
    ratings exist. If there is no expected disagreement (all values identical),
    returns 1.0 by convention.
    """
    data = np.asarray(reliability_data, dtype=float)
    if data.ndim != 2:
        raise ValueError("reliability_data must be 2D (n_units, n_raters)")

    present = data[~np.isnan(data)]
    if present.size == 0:
        return float("nan")

    values = sorted(np.unique(present).tolist())
    o, n_c = _coincidence_matrix(data, values)
    n = n_c.sum()
    if n < 2:
        return float("nan")

    V = len(values)
    # Observed disagreement
    Do = 0.0
    for c in range(V):
        for k in range(V):
            if o[c, k]:
                Do += o[c, k] * _delta2(c, k, level, values, n_c)
    Do /= n

    # Expected disagreement
    De = 0.0
    for c in range(V):
        for k in range(V):
            De += n_c[c] * n_c[k] * _delta2(c, k, level, values, n_c)
    De /= (n * (n - 1))

    if De == 0:
        return 1.0
    return 1.0 - Do / De


def alpha_from_raters_by_units(matrix, level: str = "ordinal") -> float:
    """Convenience wrapper if your matrix is (n_raters, n_units) instead."""
    return alpha(np.asarray(matrix, dtype=float).T, level=level)
