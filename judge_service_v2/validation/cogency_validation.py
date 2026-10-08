"""
cogency_validation.py — validate the Logical Cogency metric against human labels.

Belongs in: judge_api/validation/cogency_validation.py

This is the harness that turns "we have a cogency metric" into "we have EVIDENCE
the cogency metric measures cogency." It reads the cogency telemetry your
pipeline already writes, lines each record up with the Dagstuhl human labels,
and reports four things a reviewer will ask for:

1. CONSTRUCT / CONVERGENT validity — does each model sub-signal correlate with
   its matching human sub-dimension?
        acceptability  ↔  local_acceptability
        relevance      ↔  local_relevance
        sufficiency    ↔  local_sufficiency
        cogency_score  ↔  cogency
   (Spearman rank correlation — robust to the metric's arbitrary scale.)

2. DISCRIMINANT validity — the full model×human correlation matrix. Each
   sub-signal should correlate MOST with its own dimension, not equally with
   everything. If `relevance` correlates with cogency as strongly as `sufficiency`
   does, the signals aren't measuring distinct things.

3. The human-agreement CEILING — Krippendorff's alpha per dimension from the
   per-annotator scores. A model correlation is "good" relative to this ceiling,
   not in absolute terms.

4. INCREMENTAL validity / data-driven weights — refit cogency = w·[acc,rel,suff]
   by least squares against the human cogency score, and compare that fitted
   composite (and the best single sub-signal) to the current HAND-SET weights.
   This is the "let the data set the weights" step; the hand weights are only a
   starting point.

Reads telemetry only — no torch, no model load. Run your scoring over the
Dagstuhl corpus first (scoring each argument with argument_id "dagstuhl:<id>"),
then point this at the telemetry file.
"""
from __future__ import annotations

import argparse
import json
from typing import Optional

import numpy as np

from .dagstuhl_loader import load_dagstuhl, DagstuhlData
from . import krippendorff

# Model sub-signal → human dimension it should track.
SIGNAL_TO_DIM = {
    "acceptability": "local_acceptability",
    "relevance":     "local_relevance",
    "sufficiency":   "local_sufficiency",
    "cogency_score": "cogency",
}
_SUBSIGNALS = ("acceptability", "relevance", "sufficiency")


# ── correlation helpers (numpy; scipy used for p-values if available) ─────────

def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of a 1D array (ties share the mean rank)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average tied ranks
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def correlate(x, y, method: str = "spearman"):
    """Return (coefficient, p_value_or_None, n) over paired, non-nan values."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = int(len(x))
    if n < 3:
        return float("nan"), None, n

    p = None
    try:                                   # exact p-values if scipy present
        from scipy import stats
        if method == "spearman":
            coef, p = stats.spearmanr(x, y)
        else:
            coef, p = stats.pearsonr(x, y)
        return float(coef), float(p), n
    except Exception:
        if method == "spearman":
            coef = _pearson(_rankdata(x), _rankdata(y))
        else:
            coef = _pearson(x, y)
        return float(coef), None, n


# ── telemetry loading ─────────────────────────────────────────────────────────

def load_cogency_telemetry(path: str, id_prefix: Optional[str] = "dagstuhl:",
                           keep: str = "last") -> dict:
    """
    Read the telemetry JSONL and return {join_key: {acceptability, relevance,
    sufficiency, cogency_score}} for every logical_cogency 'score' record.

    id_prefix : stripped from argument_id to form the join key (so scoring with
                argument_id="dagstuhl:42" joins to Dagstuhl id "42"). Pass None
                to use argument_id verbatim.
    keep      : "last" (default) or "first" when an id was scored more than once.
    """
    out: dict = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("metric") != "logical_cogency" or rec.get("event") != "score":
                continue
            resp = rec.get("response") or {}
            aid = rec.get("argument_id")
            if aid is None:
                continue
            key = aid[len(id_prefix):] if (id_prefix and aid.startswith(id_prefix)) else aid
            row = {k: resp.get(k) for k in
                   ("acceptability", "relevance", "sufficiency", "cogency_score")}
            if keep == "first" and key in out:
                continue
            out[key] = row
    return out


# ── the join + report ─────────────────────────────────────────────────────────

def _aligned(telemetry: dict, data: DagstuhlData, signal: str, dim: str):
    """Paired (model, gold) arrays over ids present in both, for one pair."""
    model, gold = [], []
    gold_by_id = dict(zip(data.ids, data.gold[dim]))
    for key, row in telemetry.items():
        if key in gold_by_id and row.get(signal) is not None:
            model.append(row[signal]); gold.append(gold_by_id[key])
    return np.asarray(model, float), np.asarray(gold, float)


def validate_cogency(telemetry_path: str, dagstuhl_source,
                     id_prefix: Optional[str] = "dagstuhl:",
                     corr_method: str = "spearman",
                     alpha_level: str = "ordinal",
                     aggregate: str = "mean",
                     column_map: Optional[dict] = None) -> dict:
    """Run the full validation and return a structured report dict."""
    data = (dagstuhl_source if isinstance(dagstuhl_source, DagstuhlData)
            else load_dagstuhl(dagstuhl_source, column_map=column_map,
                               aggregate=aggregate))
    telemetry = load_cogency_telemetry(telemetry_path, id_prefix=id_prefix)

    report: dict = {"n_telemetry": len(telemetry), "n_gold": data.n_units(),
                    "corr_method": corr_method, "alpha_level": alpha_level}

    # 1. convergent: each signal vs its own dimension
    convergent = {}
    for signal, dim in SIGNAL_TO_DIM.items():
        if dim not in data.gold:
            continue
        m, g = _aligned(telemetry, data, signal, dim)
        coef, p, n = correlate(m, g, corr_method)
        convergent[signal] = {"dimension": dim, "coef": coef, "p": p, "n": n}
    report["convergent_validity"] = convergent

    # 2. discriminant: full signal × dimension matrix
    matrix = {}
    for signal in SIGNAL_TO_DIM:
        matrix[signal] = {}
        for dim in data.gold:
            m, g = _aligned(telemetry, data, signal, dim)
            coef, _, n = correlate(m, g, corr_method)
            matrix[signal][dim] = {"coef": coef, "n": n}
    report["discriminant_matrix"] = matrix

    # 3. human-agreement ceiling: alpha per dimension
    ceiling = {}
    for dim in data.gold:
        if data.has_raw(dim):
            ceiling[dim] = {
                "krippendorff_alpha": krippendorff.alpha(data.raw[dim], alpha_level),
                "n_raters": int(data.raw[dim].shape[1]),
            }
        else:
            ceiling[dim] = {"krippendorff_alpha": None,
                            "note": "no per-annotator columns found"}
    report["human_ceiling"] = ceiling

    # 4. incremental validity / data-driven weights (fit Cog ~ acc,rel,suff)
    report["incremental_validity"] = _incremental(telemetry, data, corr_method)

    return report


def _incremental(telemetry: dict, data: DagstuhlData, corr_method: str) -> dict:
    if "cogency" not in data.gold:
        return {"note": "no cogency gold column; skipped"}

    gold_by_id = dict(zip(data.ids, data.gold["cogency"]))
    rows, y = [], []
    for key, r in telemetry.items():
        if key not in gold_by_id:
            continue
        if any(r.get(s) is None for s in _SUBSIGNALS):
            continue
        rows.append([r[s] for s in _SUBSIGNALS]); y.append(gold_by_id[key])
    X = np.asarray(rows, float); y = np.asarray(y, float)
    m = ~np.isnan(y)
    X, y = X[m], y[m]
    if len(y) < 4:
        return {"note": f"too few paired items ({len(y)}) to fit weights"}

    # best single sub-signal vs cogency gold
    singles = {}
    for j, s in enumerate(_SUBSIGNALS):
        coef, _, n = correlate(X[:, j], y, corr_method)
        singles[s] = coef
    best_single = max(singles, key=lambda k: (singles[k] if not np.isnan(singles[k]) else -9))

    # current hand-weighted composite (cogency_score) vs gold
    hand_m, hand_g = [], []
    for key, r in telemetry.items():
        if key in gold_by_id and r.get("cogency_score") is not None:
            hand_m.append(r["cogency_score"]); hand_g.append(gold_by_id[key])
    hand_coef, _, _ = correlate(hand_m, hand_g, corr_method)

    # least-squares refit (with intercept)
    A = np.column_stack([X, np.ones(len(X))])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    w = sol[:len(_SUBSIGNALS)]; b = sol[-1]
    fitted = A @ sol
    fit_coef, _, _ = correlate(fitted, y, corr_method)
    w_norm = (w / w.sum()).tolist() if w.sum() != 0 else w.tolist()

    return {
        "n": int(len(y)),
        "single_signal_corr": singles,
        "best_single_signal": best_single,
        "hand_weighted_corr": hand_coef,
        "fitted_weights": dict(zip(_SUBSIGNALS, w.tolist())),
        "fitted_weights_normalized": dict(zip(_SUBSIGNALS, w_norm)),
        "fitted_intercept": float(b),
        "fitted_composite_corr": fit_coef,
        "verdict": ("composite beats best single signal"
                    if (not np.isnan(fit_coef) and not np.isnan(singles[best_single])
                        and fit_coef > singles[best_single])
                    else "composite does NOT beat best single signal — "
                         "reconsider combining"),
    }


# ── pretty printer + CLI ──────────────────────────────────────────────────────

def format_report(report: dict) -> str:
    L = []
    L.append(f"Cogency validation  (method={report['corr_method']}, "
             f"alpha={report['alpha_level']})")
    L.append(f"  matched telemetry ids: {report['n_telemetry']}, "
             f"gold arguments: {report['n_gold']}")

    L.append("\n1. Convergent validity (signal vs its own dimension):")
    for sig, r in report["convergent_validity"].items():
        p = f", p={r['p']:.3g}" if r["p"] is not None else ""
        L.append(f"   {sig:<14} ↔ {r['dimension']:<20} "
                 f"r={r['coef']:.3f} (n={r['n']}{p})")

    L.append("\n2. Discriminant matrix (rows=signal, cols=dimension; want the "
             "diagonal to dominate):")
    dims = list(next(iter(report["discriminant_matrix"].values())).keys())
    L.append("   " + " " * 14 + "".join(f"{d[:12]:>14}" for d in dims))
    for sig, cols in report["discriminant_matrix"].items():
        L.append(f"   {sig:<14}" + "".join(f"{cols[d]['coef']:>14.3f}" for d in dims))

    L.append("\n3. Human-agreement ceiling (Krippendorff's alpha per dimension):")
    for dim, r in report["human_ceiling"].items():
        a = r.get("krippendorff_alpha")
        if a is None:
            L.append(f"   {dim:<20} (no per-annotator data)")
        else:
            L.append(f"   {dim:<20} alpha={a:.3f} (raters={r['n_raters']})")

    iv = report["incremental_validity"]
    L.append("\n4. Incremental validity / data-driven weights (target=cogency):")
    if "note" in iv:
        L.append(f"   {iv['note']}")
    else:
        L.append(f"   single-signal corr: " +
                 ", ".join(f"{s}={c:.3f}" for s, c in iv["single_signal_corr"].items()))
        L.append(f"   best single signal: {iv['best_single_signal']} "
                 f"(r={iv['single_signal_corr'][iv['best_single_signal']]:.3f})")
        L.append(f"   hand-weighted cogency_score: r={iv['hand_weighted_corr']:.3f}")
        L.append(f"   least-squares refit: r={iv['fitted_composite_corr']:.3f}")
        L.append(f"     fitted weights (normalized): " +
                 ", ".join(f"{s}={w:.3f}" for s, w in iv["fitted_weights_normalized"].items()))
        L.append(f"   → {iv['verdict']}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate Logical Cogency against Dagstuhl.")
    ap.add_argument("--telemetry", required=True, help="path to judge_api_telemetry.jsonl")
    ap.add_argument("--dagstuhl", required=True, help="Dagstuhl table (.csv/.tsv/.xlsx)")
    ap.add_argument("--id-prefix", default="dagstuhl:",
                    help="prefix stripped from argument_id to join (default 'dagstuhl:')")
    ap.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    ap.add_argument("--alpha-level", default="ordinal",
                    choices=["nominal", "ordinal", "interval"])
    ap.add_argument("--aggregate", default="mean", choices=["mean", "median", "majority"])
    args = ap.parse_args(argv)

    report = validate_cogency(
        args.telemetry, args.dagstuhl, id_prefix=args.id_prefix,
        corr_method=args.method, alpha_level=args.alpha_level,
        aggregate=args.aggregate)
    print(format_report(report))
    return report


if __name__ == "__main__":
    main()
