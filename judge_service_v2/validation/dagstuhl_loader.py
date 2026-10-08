"""
dagstuhl_loader.py — load the Dagstuhl-15512 ArgQuality corpus into the shape
the cogency validation harness needs.

Belongs in: judge_api/validation/dagstuhl_loader.py

WHAT IT PRODUCES
----------------
For each of the four logical dimensions we validate against —

    cogency               (Cog)  ← validates the composite cogency_score
    local_acceptability   (LAc)  ← validates the `acceptability` sub-signal
    local_relevance       (LRe)  ← validates the `relevance` sub-signal
    local_sufficiency     (LSu)  ← validates the `sufficiency` sub-signal

— it returns BOTH:
    gold : one aggregated human score per argument  (for model↔human correlation)
    raw  : the per-annotator scores per argument     (for Krippendorff's alpha,
           i.e. the human-agreement CEILING)

FORMAT NOTE
-----------
The official corpus ships as annotated XMI, where each dimension stores the
majority score, the mean, and all annotators' individual scores. This loader
does NOT parse XMI directly (the type system is release-specific); instead it
consumes a TABLE — a CSV/TSV/XLSX export or a pandas DataFrame — and maps its
columns to the canonical dimensions by alias. That keeps it robust to whichever
export you use. Columns are matched case/spacing/punctuation-insensitively, so
"Local Sufficiency", "local_sufficiency", "LSu", "ls" all resolve to the same
dimension. Per-annotator columns are detected by an annotator suffix
(e.g. "cogency_ann1", "cogency.1", "Local relevance (annotator 2)").

Pass an explicit `column_map` to override auto-detection entirely.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Canonical dimensions and their accepted aliases (normalised) ──────────────
_CANON_DIMS = ("cogency", "local_acceptability", "local_relevance",
               "local_sufficiency")

_DIM_ALIASES = {
    "cogency":             {"cogency", "cog", "co"},
    "local_acceptability": {"localacceptability", "acceptability", "lac", "la"},
    "local_relevance":     {"localrelevance", "relevance", "lre", "lr"},
    "local_sufficiency":   {"localsufficiency", "sufficiency", "lsu", "ls"},
}

_ID_ALIASES   = {"id", "argid", "argumentid", "argumentnr", "aid", "unitid"}
_TEXT_ALIASES = {"text", "argument", "argumenttext", "comment", "content"}

# Aggregate suffixes to ignore when detecting per-annotator columns.
_AGG_TOKENS = {"mean", "avg", "average", "majority", "maj", "median",
               "gold", "agg", "aggregate", "score"}
# Annotator suffix patterns: trailing "ann1", "annotator2", "a1", "rater3", ".1", "_1"
_ANN_SUFFIX = re.compile(r"(?:ann(?:otator)?|rater|coder|a|r)?\s*_?\.?\s*(\d+)$")


def _norm(s: str) -> str:
    """Lowercase, drop all non-alphanumerics — for alias matching."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _split_dim_and_annotator(col: str):
    """Return (canonical_dim, annotator_key_or_None) for a column, or (None,None).

    annotator_key is None for an aggregated column, an int for a per-annotator
    column, or the string "agg" for an explicit mean/majority column.
    """
    raw = _norm(col)

    # Try each dimension alias as a prefix of the normalised column.
    for dim, aliases in _DIM_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if raw == alias:
                return dim, None                      # bare dimension = gold
            if raw.startswith(alias):
                rest = raw[len(alias):]
                if rest in _AGG_TOKENS:
                    return dim, "agg"                 # e.g. cogency_mean
                m = _ANN_SUFFIX.match(rest)
                if m:
                    return dim, int(m.group(1))       # e.g. cogency_ann2 -> 2
                # rest is something else (another dim word) → not a match here
    return None, None


def _find_special(columns, aliases):
    for c in columns:
        if _norm(c) in aliases:
            return c
    return None


@dataclass
class DagstuhlData:
    ids: list
    texts: Optional[list]
    gold: dict           # dim -> np.ndarray (U,)   aggregated human score
    raw: dict            # dim -> np.ndarray (U, R)  per-annotator, nan-padded
    dims: tuple = field(default=_CANON_DIMS)

    def n_units(self) -> int:
        return len(self.ids)

    def has_raw(self, dim: str) -> bool:
        return dim in self.raw and self.raw[dim] is not None


def _aggregate(per_ann: np.ndarray, how: str) -> np.ndarray:
    """Aggregate a (U,R) per-annotator matrix to (U,) gold, ignoring nan."""
    with np.errstate(invalid="ignore"):
        if how == "mean":
            return np.nanmean(per_ann, axis=1)
        if how == "median":
            return np.nanmedian(per_ann, axis=1)
        if how == "majority":
            out = np.full(per_ann.shape[0], np.nan)
            for i, row in enumerate(per_ann):
                vals = row[~np.isnan(row)]
                if vals.size:
                    u, c = np.unique(vals, return_counts=True)
                    out[i] = u[np.argmax(c)]
            return out
    raise ValueError(f"unknown aggregation {how!r}")


def load_dagstuhl(source, column_map: Optional[dict] = None,
                  aggregate: str = "mean",
                  id_col: Optional[str] = None,
                  text_col: Optional[str] = None) -> DagstuhlData:
    """
    Load the corpus from a path (.csv/.tsv/.xlsx) or a pandas DataFrame.

    Parameters
    ----------
    source      : path string or a pandas.DataFrame.
    column_map  : optional {canonical_dim: column | [per-annotator columns]}.
                  Overrides auto-detection. A single column is treated as gold;
                  a list is treated as per-annotator scores (aggregated for gold,
                  kept raw for alpha).
    aggregate   : "mean" (default), "median", or "majority" — how per-annotator
                  scores become the gold value. If your table already has a
                  mean/majority column, auto-detection uses it as gold and still
                  keeps any per-annotator columns as raw.
    id_col, text_col : override id/text column detection.

    Returns
    -------
    DagstuhlData(ids, texts, gold, raw, dims)
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("dagstuhl_loader needs pandas (pip install pandas)") from exc

    if hasattr(source, "columns"):          # already a DataFrame
        df = source
    else:
        s = str(source).lower()
        if s.endswith((".xlsx", ".xls")):
            df = pd.read_excel(source)
        elif s.endswith(".tsv"):
            df = pd.read_csv(source, sep="\t")
        else:
            df = pd.read_csv(source)

    columns = list(df.columns)

    # ── id / text ────────────────────────────────────────────────────────────
    id_col = id_col or _find_special(columns, _ID_ALIASES)
    text_col = text_col or _find_special(columns, _TEXT_ALIASES)
    if id_col is not None:
        ids = df[id_col].astype(str).tolist()
    else:
        ids = [str(i) for i in range(len(df))]   # fall back to row index
    texts = df[text_col].astype(str).tolist() if text_col is not None else None

    # ── build per-dim column groups ──────────────────────────────────────────
    if column_map is not None:
        groups = {}
        for dim, cols in column_map.items():
            groups[dim] = {"gold": None, "ann": []}
            if isinstance(cols, (list, tuple)):
                groups[dim]["ann"] = list(cols)
            else:
                groups[dim]["gold"] = cols
    else:
        groups = {d: {"gold": None, "ann": []} for d in _CANON_DIMS}
        for col in columns:
            dim, ann = _split_dim_and_annotator(col)
            if dim is None:
                continue
            if ann is None or ann == "agg":
                # prefer an explicit gold/mean column if several appear
                if groups[dim]["gold"] is None:
                    groups[dim]["gold"] = col
            else:
                groups[dim]["ann"].append((ann, col))

    gold: dict = {}
    raw: dict = {}
    for dim in _CANON_DIMS:
        g = groups.get(dim, {"gold": None, "ann": []})
        ann_cols = g["ann"]
        # sort per-annotator columns by their detected annotator number
        if ann_cols and isinstance(ann_cols[0], tuple):
            ann_cols = [c for _, c in sorted(ann_cols, key=lambda t: t[0])]

        if ann_cols:
            per_ann = df[ann_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            raw[dim] = per_ann
            gold[dim] = (df[g["gold"]].to_numpy(dtype=float) if g["gold"]
                         else _aggregate(per_ann, aggregate))
        elif g["gold"] is not None:
            raw[dim] = None
            gold[dim] = pd.to_numeric(df[g["gold"]], errors="coerce").to_numpy(dtype=float)
        # else: dimension absent — skipped

    found = [d for d in _CANON_DIMS if d in gold]
    if not found:
        raise ValueError(
            "No cogency dimensions found. Columns seen: "
            f"{columns}. Pass column_map={{'cogency': 'YourCol', ...}} to map "
            "them explicitly.")

    return DagstuhlData(ids=ids, texts=texts, gold=gold, raw=raw,
                        dims=tuple(found))
