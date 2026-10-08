"""
delta_votes_score.py — Delta (threshold bonus) and Votes scorers.

Both signals are DERIVED FROM ELASTICSEARCH. Delta needs a delta-flagged
reference match; Votes needs cluster-quality from matched reference clusters.
Neither can be computed without ES results — so when ES returns nothing, this
module does NOT fabricate a score. It returns a "not applicable" marker and the
composite excludes both metrics and renormalises the remaining weights onto
ArgQuality + NLI (see utils.judge_config.renormalized_individual_weights).

This replaces the previous behaviour, where an empty result set fell back to a
keyword-based `_evidence_grounding_heuristic` that stood in for Votes. That
proxy measured a DIFFERENT construct (surface evidence cues) while wearing the
Votes label, muddying what the number meant. It has been retired: an argument
with no ES match is now scored purely on the two ES-independent model signals.

Design
------
1. ABSOLUTE similarity scale. All similarities use es_norm(raw) = raw - 1.0,
   the true cosine, never a within-batch min-max (which made every top match
   normalise to 1.0 regardless of its real quality).

2. Delta is a RARE-EVENT BONUS, not a weighted component (spec).
   delta_earned = True iff any ES match is delta-flagged (quality_signal) AND
   its ABSOLUTE similarity >= DELTA_SIM_THRESHOLD (default 0.8 = raw ES 1.8).
   The composite ADDS DELTA_BONUS when earned and does nothing otherwise —
   no neutral-0.5 filler. Because it requires an ES match, delta can only ever
   be earned WHEN es_results exist.

3. Votes: similarity-weighted average of cluster quality across top-3, using
   ABSOLUTE similarity weights.

4. Empty es_results → applicable=False, delta_score=None, votes_score=None.
   Not an exception, not a hard error: an unknown argument is a legitimate,
   common case. The caller treats "not applicable" as "exclude and renormalise".

Return contract
---------------
Every response carries an "applicable" boolean so callers can branch cleanly:
  applicable=True  → delta_*/votes_* are real numbers
  applicable=False → delta_*/votes_* are None; "reason" explains why
"""

import logging

from ..utils.judge_config import es_norm, DELTA_SIM_THRESHOLD
from ..utils.telemetry import log_event

logger = logging.getLogger(__name__)

_VOTES_TOP_K = 3   # aggregate cluster quality across top-3, not just top-1


def _not_applicable(reason: str) -> dict:
    """
    Response when Delta/Votes cannot be computed (no ES results). No score is
    invented — every ES-derived field is None so a caller can never mistake
    "not applicable" for a real (low) score.
    """
    return {
        "applicable":                False,
        "delta_earned":              False,
        "delta_score":               None,
        "delta_best_similarity":     None,
        "delta_threshold":           DELTA_SIM_THRESHOLD,
        "votes_score":               None,
        "delta_confidence":          None,
        "votes_confidence":          None,
        "delta_flagged_count":       0,
        "total_matches":             0,
        "top_match_cluster_quality": None,
        "fallback_used":             True,
        "reason":                    reason,
    }


def score_delta_votes(es_results: list) -> dict:
    """
    Compute the Delta bonus signal and Votes Score from /granular-similarity
    results.

    Parameters
    ----------
    es_results : list of dicts, each with:
        - overall_score       : float  RAW ES score, cosine + 1.0 (1.0–2.0)
        - quality_signal      : bool   True = delta-flagged on CMV
        - cluster_quality_max : float  log-scaled votes proxy, 0-1

    Returns
    -------
    When es_results is empty  → _not_applicable(...) (applicable=False).
    Otherwise:
    {
        "applicable":            True,
        "delta_earned":          bool,    the argument earned the delta bonus
        "delta_score":           float,   1.0 if earned else 0.0 (binary)
        "delta_best_similarity": float,   best ABSOLUTE sim among delta matches
        "delta_threshold":       float,   the gate that had to be cleared
        "votes_score":           float,   0.0-1.0
        "delta_confidence":      float,
        "votes_confidence":      float,
        "delta_flagged_count":   int,
        "total_matches":         int,
        "top_match_cluster_quality": float,
        "fallback_used":         bool,
    }
    """
    # ── No ES results → Delta and Votes are not applicable ────────────────────
    # Neither metric is defined without a reference match. We return a marker,
    # NOT a fabricated score; the composite drops both and renormalises.
    if not es_results:
        return _not_applicable(
            "No ES results — Delta not earnable and Votes undefined; "
            "both excluded from the composite (weights renormalised onto "
            "ArgQuality + NLI)."
        )

    # ABSOLUTE similarities on the true cosine scale (0-1)
    abs_sims = [es_norm(r.get("overall_score", 0.0)) for r in es_results]

    # ── Delta: rare-event threshold bonus ─────────────────────────────────────
    delta_sims = [
        abs_sims[i] for i, r in enumerate(es_results)
        if r.get("quality_signal", False)
    ]
    delta_flagged  = len(delta_sims)
    best_delta_sim = max(delta_sims) if delta_sims else 0.0

    delta_earned = best_delta_sim >= DELTA_SIM_THRESHOLD
    delta_score  = 1.0 if delta_earned else 0.0

    # Confidence: how close the best delta-flagged match actually is
    delta_confidence = round(best_delta_sim, 3)

    # ── Votes Score: similarity-weighted cluster quality across top-3 ─────────
    top_results = es_results[:_VOTES_TOP_K]
    top_sims    = abs_sims[:_VOTES_TOP_K]
    top_sim_sum = sum(top_sims) or 1e-9

    cluster_qualities = [
        r.get("cluster_quality_max", 0.0) or 0.0
        for r in top_results
    ]

    votes_score = sum(
        top_sims[i] * cluster_qualities[i]
        for i in range(len(top_results))
    ) / top_sim_sum
    votes_score = round(min(max(votes_score, 0.0), 1.0), 3)

    # Votes confidence: ABSOLUTE similarity of the best match
    votes_confidence = round(abs_sims[0] if abs_sims else 0.0, 3)

    return {
        "applicable":                True,
        "delta_earned":              delta_earned,
        "delta_score":               delta_score,
        "delta_best_similarity":     round(best_delta_sim, 3),
        "delta_threshold":           DELTA_SIM_THRESHOLD,
        "votes_score":               votes_score,
        "delta_confidence":          delta_confidence,
        "votes_confidence":          votes_confidence,
        "delta_flagged_count":       delta_flagged,
        "total_matches":             len(es_results),
        "top_match_cluster_quality": round(
            cluster_qualities[0] if cluster_qualities else 0.0, 3),
        "fallback_used":             False,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Speaker-level Delta attribution (ATTRIBUTION ONLY — affects no score)
# ──────────────────────────────────────────────────────────────────────────────
#
# The per-turn calculation above is unchanged. Delta is earned per turn from an
# ES match; this rolls those per-turn outcomes up to the speaker so the project
# can attribute "how many deltas did each side earn" — a count + a flag. It does
# NOT feed the composite, add a bonus, or break ties: it is reporting only.
#
# (The turn-by-turn Votes comparison BETWEEN speakers is intentionally NOT built
# here yet — per current direction, Votes stays a per-turn number for now.)

def aggregate_delta_by_speaker(turn_results: list) -> dict:
    """
    Roll per-turn Delta outcomes up to each speaker, for attribution only.

    Parameters
    ----------
    turn_results : list of per-turn records. Each record identifies a speaker
        and carries that turn's Delta/Votes result. Two shapes are accepted:
          * flat  — the score_delta_votes(...) output with a "speaker" key
                    merged in (and optional "turn_no"), i.e. {**result,
                    "speaker": "PRO", "turn_no": 1}
          * nested — {"speaker": "PRO", "turn_no": 1, "result": <score_delta_votes output>}

    Returns
    -------
    dict keyed by speaker → {
        "speaker":              the speaker,
        "delta_earned":         bool,  earned a delta on at least one turn,
        "delta_earned_count":   int,   number of turns that earned a delta,
        "delta_turns":          list,  turn ids/indices that earned a delta,
        "turns_with_es_match":  int,   turns that had an ES match (applicable),
        "total_turns":          int,   all turns seen for this speaker,
    }

    No scores are produced or modified; this is a reporting aggregate.
    """
    by_speaker: dict = {}

    for idx, item in enumerate(turn_results):
        speaker = item.get("speaker", "UNKNOWN")
        res = item.get("result", item)          # accept flat or nested shape
        applicable = bool(res.get("applicable", False))
        earned = bool(res.get("delta_earned", False))
        turn_no = item.get("turn_no", idx)

        agg = by_speaker.setdefault(speaker, {
            "speaker":             speaker,
            "delta_earned":        False,
            "delta_earned_count":  0,
            "delta_turns":         [],
            "turns_with_es_match": 0,
            "total_turns":         0,
        })
        agg["total_turns"] += 1
        if applicable:
            agg["turns_with_es_match"] += 1
        if earned:
            agg["delta_earned"] = True
            agg["delta_earned_count"] += 1
            agg["delta_turns"].append(turn_no)

    # Log the attribution to BOTH files: a lean per-speaker fact in telemetry
    # (no external call → log_event, not log_call) and a narrative summary line.
    for speaker, agg in by_speaker.items():
        log_event("delta_votes", "speaker_delta_attribution",
                  argument_id=None, **agg)
        logger.info("Delta attribution | speaker=%s | earned %d/%d turns %s",
                    speaker, agg["delta_earned_count"], agg["total_turns"],
                    agg["delta_turns"] or "")

    return by_speaker
