"""
debate_winner.py — deterministic whole-debate winner determination.

Winner hierarchy
----------------
Level 1 — Core debate performance
    * Logical Cogency: acceptability, relevance, sufficiency
    * Rebuttal Effectiveness

Level 2 — Supporting argument quality
    * Stance Support
    * Consistency
    * Information Density

Decision rule
-------------
1. Compare Level 1 first.
2. A speaker dominates a level when they are not meaningfully worse on ANY
   metric and are meaningfully better on AT LEAST ONE metric.
3. If Level 1 does not distinguish the speakers, compare Level 2.
4. If Level 2 does not distinguish them, return Tie.

There is no weighted sum and no cross-metric average in winner determination.
Level-2 continuous metrics use a 0.10 meaningful-difference threshold.
Level-1 metrics are native 1-3 scores, so no threshold is applied.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


LEVEL2_THRESHOLDS: Dict[str, float] = {
    "stance_support": 0.10,
    "consistency": 0.10,
    "information_density": 0.10,
}

LEVEL1_METRICS = (
    "acceptability",
    "relevance",
    "sufficiency",
    "rebuttal_effectiveness",
)

LEVEL2_METRICS = (
    "stance_support",
    "consistency",
    "information_density",
)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_numeric(data: Any, keys: Sequence[str]) -> Optional[float]:
    """Read the first numeric value from a dict using preferred field names."""
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = _number(data.get(key))
        if value is not None:
            return value
    return None


def _extract_v2_profile(speaker_result: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Convert the existing judge_service_v2 by_speaker structure into the
    seven canonical fields used by the winner engine.

    This is only extraction/normalisation. It does not calculate a composite
    score.
    """
    cog = speaker_result.get("logical_cogency") or {}
    reb = speaker_result.get("rebuttal") or {}
    stance = speaker_result.get("stance_support")
    consistency = speaker_result.get("consistency")
    info = speaker_result.get("information_density")

    return {
        "acceptability": _first_numeric(
            cog,
            ("mean_local_acceptability", "local_acceptability", "acceptability"),
        ),
        "relevance": _first_numeric(
            cog,
            ("mean_local_relevance", "local_relevance", "relevance"),
        ),
        "sufficiency": _first_numeric(
            cog,
            ("mean_local_sufficiency", "local_sufficiency", "sufficiency"),
        ),
        "rebuttal_effectiveness": _first_numeric(
            reb if isinstance(reb, dict) else {},
            ("mean_effectiveness", "rebuttal_effectiveness"),
        ),
        "stance_support": _first_numeric(
            stance if isinstance(stance, dict) else {},
            (
                "stance_support_score",
                "mean_stance_support_score",
                "stance_support",
                "score",
            ),
        ) if isinstance(stance, dict) else _number(stance),
        "consistency": _first_numeric(
            consistency if isinstance(consistency, dict) else {},
            (
                "consistency_score",
                "mean_consistency",
                "consistency",
                "score",
            ),
        ) if isinstance(consistency, dict) else _number(consistency),
        "information_density": _first_numeric(
            info if isinstance(info, dict) else {},
            ("information_density", "score"),
        ) if isinstance(info, dict) else _number(info),
    }


def _dominates(
    a: Dict[str, Optional[float]],
    b: Dict[str, Optional[float]],
    metrics: Sequence[str],
    thresholds: Optional[Dict[str, float]] = None,
) -> bool:
    """
    A dominates B iff:
      * A is not meaningfully worse on any metric, and
      * A is meaningfully better on at least one metric.

    If a required metric is missing for either speaker, no dominance is claimed.
    """
    thresholds = thresholds or {}

    for metric in metrics:
        av = a.get(metric)
        bv = b.get(metric)
        if av is None or bv is None:
            return False

        threshold = thresholds.get(metric, 0.0)

        # A may not be meaningfully worse.
        # Equality is NOT worse. A difference must be strictly greater than
        # the threshold to count as a meaningful disadvantage.
        if (bv - av) > threshold:
            return False

    for metric in metrics:
        av = a.get(metric)
        bv = b.get(metric)
        threshold = thresholds.get(metric, 0.0)
        if av is not None and bv is not None and (av - bv) > threshold:
            return True

    return False


def _compare_level(
    a: Dict[str, Optional[float]],
    b: Dict[str, Optional[float]],
    metrics: Sequence[str],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """Return an auditable pairwise comparison for one winner level."""
    equal: list[str] = []
    a_better: list[str] = []
    b_better: list[str] = []
    missing: list[str] = []

    for metric in metrics:
        av, bv = a.get(metric), b.get(metric)
        if av is None or bv is None:
            missing.append(metric)
            continue
        threshold = thresholds.get(metric, 0.0)
        diff = av - bv
        if abs(diff) <= threshold:
            equal.append(metric)
        elif diff > 0:
            a_better.append(metric)
        else:
            b_better.append(metric)

    return {
        "equal_metrics": equal,
        "first_speaker_better_metrics": a_better,
        "second_speaker_better_metrics": b_better,
        "missing_metrics": missing,
        "first_speaker_dominates": not missing and not b_better and bool(a_better),
        "second_speaker_dominates": not missing and not a_better and bool(b_better),
    }


def determine_debate_winner(
    by_speaker: Dict[str, Dict[str, Any]],
    speaker_names: Sequence[str],
) -> Dict[str, Any]:
    """Determine a winner directly from judge_service_v2's `by_speaker` output.

    Exactly two speakers are required. Level 1 is evaluated first. If one
    speaker dominates Level 1, Level 2 is not consulted.
    """
    if len(speaker_names) != 2:
        raise ValueError(
            "Winner determination currently requires exactly two speakers."
        )

    speaker_a, speaker_b = speaker_names
    if speaker_a not in by_speaker or speaker_b not in by_speaker:
        raise KeyError(
            f"Winner metrics missing for one or both speakers: "
            f"{speaker_a!r}, {speaker_b!r}"
        )

    profile_a = _extract_v2_profile(by_speaker[speaker_a])
    profile_b = _extract_v2_profile(by_speaker[speaker_b])

    # Level 1: native 1-3 metric scales. Equality is neutral; it is not a
    # disadvantage. A strictly better value is enough to establish dominance
    # when the other Level-1 metrics are equal or better.
    level1_thresholds = {metric: 0.0 for metric in LEVEL1_METRICS}
    l1 = _compare_level(profile_a, profile_b, LEVEL1_METRICS, level1_thresholds)

    if l1["first_speaker_dominates"]:
        better = ", ".join(l1["first_speaker_better_metrics"])
        return {
            "winner": speaker_a,
            "level": "Level 1",
            "reason": (
                f"{speaker_a} dominates {speaker_b} at Level 1: "
                f"equal or better on every core metric and better on {better}."
            ),
            "profiles": {speaker_a: profile_a, speaker_b: profile_b},
            "level_1_comparison": l1,
            "thresholds": LEVEL2_THRESHOLDS,
        }

    if l1["second_speaker_dominates"]:
        better = ", ".join(l1["second_speaker_better_metrics"])
        return {
            "winner": speaker_b,
            "level": "Level 1",
            "reason": (
                f"{speaker_b} dominates {speaker_a} at Level 1: "
                f"equal or better on every core metric and better on {better}."
            ),
            "profiles": {speaker_a: profile_a, speaker_b: profile_b},
            "level_1_comparison": l1,
            "thresholds": LEVEL2_THRESHOLDS,
        }

    # Level 2: continuous metrics with metric-specific meaningful differences.
    l2 = _compare_level(profile_a, profile_b, LEVEL2_METRICS, LEVEL2_THRESHOLDS)

    if l2["first_speaker_dominates"]:
        better = ", ".join(l2["first_speaker_better_metrics"])
        return {
            "winner": speaker_a,
            "level": "Level 2",
            "reason": (
                f"{speaker_a} dominates {speaker_b} at Level 2: "
                f"no meaningful disadvantage and better on {better}."
            ),
            "profiles": {speaker_a: profile_a, speaker_b: profile_b},
            "level_1_comparison": l1,
            "level_2_comparison": l2,
            "thresholds": LEVEL2_THRESHOLDS,
        }

    if l2["second_speaker_dominates"]:
        better = ", ".join(l2["second_speaker_better_metrics"])
        return {
            "winner": speaker_b,
            "level": "Level 2",
            "reason": (
                f"{speaker_b} dominates {speaker_a} at Level 2: "
                f"no meaningful disadvantage and better on {better}."
            ),
            "profiles": {speaker_a: profile_a, speaker_b: profile_b},
            "level_1_comparison": l1,
            "level_2_comparison": l2,
            "thresholds": LEVEL2_THRESHOLDS,
        }

    return {
        "winner": "Tie",
        "level": "Tie",
        "reason": "Neither speaker dominates at Level 1 or Level 2.",
        "profiles": {speaker_a: profile_a, speaker_b: profile_b},
        "level_1_comparison": l1,
        "level_2_comparison": l2,
        "thresholds": LEVEL2_THRESHOLDS,
    }

