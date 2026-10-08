"""
consistency_score.py — did the speaker contradict their own earlier claims?

A dialectical property, not a logical one: cogency asks whether THIS turn's
reasoning holds; consistency asks whether this claim sits with what the same
speaker said before. Different inputs, different question — hence its own module.

    consistency = 1 - max P(contradiction | current claim, prior claim)

MAX, not mean: contradicting one earlier claim makes a speaker inconsistent, and
averaging would let two agreements dilute one flat contradiction.

Every pair is scored independently and returned in per_prior_claim. The headline
says a contradiction exists; the breakdown says WHICH claim it hits.
"""
from __future__ import annotations

import logging
from typing import Optional

from ..utils.telemetry import log_call, arg_id
from .stance_support_score import _nli_probs

logger = logging.getLogger(__name__)


def score_consistency(claim: str, prior_claims,
                      argument_id: Optional[str] = None) -> dict:
    """
    prior_claims : the SAME speaker's earlier claims. Plain strings, or dicts
                   {"claim": str, "turn_no": int} so the breakdown can name the
                   turn rather than quote the text.

    consistency is None with no priors — an opening turn has nothing to be
    consistent with, and scoring it 1.0 would inflate every speaker average.
    """
    if not claim or not claim.strip():
        raise ValueError("claim must be a non-empty string")

    priors = []
    for p in (prior_claims or []):
        if isinstance(p, dict):
            text, turn_no = str(p.get("claim", "")).strip(), p.get("turn_no")
        else:
            text, turn_no = str(p).strip(), None
        if text:
            priors.append((text, turn_no))

    aid = argument_id or arg_id(claim)
    with log_call("consistency", "nli_consistency", argument_id=aid,
                  claim=claim, n_prior_claims=len(priors)) as rec:
        if not priors:
            result = {"consistency": None, "most_contradicted": None,
                      "per_prior_claim": [], "n_prior_claims": 0}
            rec["response"] = result
            return result

        per = []
        for text, turn_no in priors:
            # Current as premise: does what he says NOW contradict what he said
            # BEFORE. NLI is asymmetric, so the order matters.
            probs = _nli_probs(claim.strip(), text)
            row = {"claim": text,
                   "contradiction": round(probs["contradiction"], 4),
                   "entailment": round(probs["entailment"], 4)}
            if turn_no is not None:
                row["turn_no"] = turn_no
            per.append(row)

        worst = max(per, key=lambda r: r["contradiction"])
        result = {"consistency": round(1.0 - worst["contradiction"], 4),
                  "most_contradicted": worst,
                  "per_prior_claim": per,
                  "n_prior_claims": len(per)}
        rec["response"] = result
        return result


def aggregate_consistency_by_speaker(turn_results: list) -> dict:
    """Per speaker: mean consistency and the turns that contradicted an earlier
    claim. Items need "speaker"; "turn_no" names the offending turns."""
    by_speaker: dict = {}
    for r in turn_results:
        sp = r.get("speaker")
        if sp is None or r.get("consistency") is None:
            continue
        by_speaker.setdefault(sp, []).append(r)

    out: dict = {}
    for sp, rows in by_speaker.items():
        vals = [r["consistency"] for r in rows]
        contradicting = [r.get("turn_no") for r in rows if r["consistency"] < 0.5]
        out[sp] = {
            "turns_scored": len(rows),
            "mean_consistency": round(sum(vals) / len(vals), 4),
            "min_consistency": round(min(vals), 4),
            "contradicting_turns": contradicting,
            "self_consistent": not contradicting,
        }
    return out
