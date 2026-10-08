"""
orchestrator.py — score a whole debate through every metric in one call.

Claim + premises are mined ONCE per turn and reused. Stance hypotheses are
generated ONCE per topic. Rebuttal runs once and argument survival is a view of
its output, not a second pass. Every per-turn call is wrapped so one failure
degrades one field instead of sinking the debate.
"""
from __future__ import annotations

import logging
from typing import Optional

from .utils.argument_miner import extract_claim_and_premises
from .utils.telemetry import log_event
from .utils.topic_hypotheses import generate_topic_hypotheses
from .turn_level_scoring.stance_support_score import (
    score_stance_support, aggregate_stance_by_speaker)
from .turn_level_scoring.logical_cogency_score import score_logical_cogency
from .turn_level_scoring.consistency_score import (
    score_consistency, aggregate_consistency_by_speaker)
from .turn_level_scoring.delta_votes_score import (
    score_delta_votes, aggregate_delta_by_speaker)
from .debate_level_scoring.rebuttal_effectiveness_dl import score_rebuttal_debate
from .debate_level_scoring.information_density_dl import (
    score_information_density_by_speaker)
from .debate_level_scoring.argument_survival_dl import score_argument_survival
from .debate_level_scoring.debate_winner import determine_debate_winner

logger = logging.getLogger(__name__)


def _mean(values) -> Optional[float]:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 4) if nums else None


def _resolve_stances(turns, stance_by_speaker):
    """{speaker: 'PRO'|'CON'} plus speakers in first-appearance order."""
    order = []
    for t in turns:
        sp = t.get("speaker", "UNKNOWN")
        if sp not in order:
            order.append(sp)

    if stance_by_speaker:
        return {sp: stance_by_speaker.get(sp, "PRO").upper() for sp in order}, order, None

    stances, note = {}, None
    for t in turns:
        sp = t.get("speaker", "UNKNOWN")
        if sp not in stances and t.get("stance"):
            stances[sp] = t["stance"].upper()
    missing = [sp for sp in order if sp not in stances]
    if missing:
        for sp in missing:
            stances[sp] = "PRO" if order.index(sp) % 2 == 0 else "CON"
        note = ("stance was not supplied for every speaker; defaulted by order "
                "(first=PRO, second=CON). Pass stance_by_speaker to be explicit.")
    return stances, order, note


def score_debate(
    topic: str,
    turns: list,
    stance_by_speaker: Optional[dict] = None,
    model: Optional[str] = None,
    use_llm_rebuttal: bool = True,
    exclude_openers: bool = True,
    es_results_by_turn: Optional[dict] = None,
    include_detail: bool = True,
    write_report: bool = True,
) -> dict:
    """
    topic : the debate topic, in any form (claim, question, motion).
    turns : ordered dicts {"speaker", "text", "turn_no"?, "stance"?,
            "claim"?, "premises"?}. Supplying claim/premises skips mining.
    stance_by_speaker : {speaker: "PRO"|"CON"}; recommended.
    model : Ollama model for every LLM call in this debate.
    es_results_by_turn : {turn_no: [es_result,...]} for Delta/Votes.

    Returns by_speaker profiles, per_turn detail, debate-level outputs, and
    deterministic winner_determination. No separate reasoning-model call is
    made for winner determination.
    """
    for i, t in enumerate(turns):
        t.setdefault("turn_no", i)
    stances, order, note = _resolve_stances(turns, stance_by_speaker)
    es_map = es_results_by_turn or {}
    llm_kw = {"model": model} if model else {}

    # One LLM call per topic; every turn reuses the pair.
    hypotheses, hyp_error = None, None
    try:
        h = generate_topic_hypotheses(topic=topic, use_cache=True, **llm_kw)
        hypotheses = {"PRO": h.pro_hypothesis, "CON": h.con_hypothesis}
    except Exception as e:
        hyp_error = str(e)
        logger.warning("topic hypotheses failed (%s) — stance support will be "
                       "unavailable for every turn", e)

    per_turn: list = []
    enriched: list = []
    stance_rows: list = []
    consistency_rows: list = []
    prior_claims: dict = {}
    delta_records: list = []
    mined_count = 0

    for t in turns:
        sp = t.get("speaker", "UNKNOWN")
        tno = t["turn_no"]
        text = t.get("text", "") or ""
        stance = stances.get(sp, "PRO")

        claim = t.get("claim")
        premises = t.get("premises")
        if not claim:
            try:
                mined = extract_claim_and_premises(text, **llm_kw)
                claim, premises = mined.claim, mined.premises
                mined_count += 1
            except Exception as e:
                logger.warning("mining failed for turn %s: %s", tno, e)
                claim, premises = text, []

        if hypotheses:
            try:
                ss = score_stance_support(topic, text, stance, claim=claim,
                                          hypothesis=hypotheses[stance],
                                          model=model)
            except Exception as e:
                ss = {"error": str(e), "stance_support_score": None}
        else:
            ss = {"error": f"topic hypotheses unavailable: {hyp_error}",
                  "stance_support_score": None}
        stance_rows.append({**ss, "speaker": sp, "turn_no": tno})

        try:
            cog = score_logical_cogency(claim, premises, model=model)
        except Exception as e:
            cog = {"error": str(e), "local_acceptability": None,
                   "local_relevance": None, "local_sufficiency": None}

        try:
            cons = score_consistency(claim, prior_claims.get(sp, []))
        except Exception as e:
            cons = {"error": str(e), "consistency": None}
        consistency_rows.append({**cons, "speaker": sp, "turn_no": tno})
        prior_claims.setdefault(sp, []).append({"claim": claim, "turn_no": tno})

        dv = score_delta_votes(es_map.get(tno, es_map.get(str(tno))) or [])
        delta_records.append({**dv, "speaker": sp, "turn_no": tno})

        per_turn.append({
            "turn_no": tno, "speaker": sp, "stance": stance,
            "claim": claim, "premises": premises,
            "stance_support": ss, "logical_cogency": cog,
            "consistency": cons, "delta_votes": dv,
        })
        enriched.append({"speaker": sp, "turn_no": tno, "text": text,
                         "claim": claim, "premises": premises, "stance": stance})

    try:
        rebuttal = score_rebuttal_debate(
            enriched, model=model, exclude_openers=exclude_openers,
            use_llm=use_llm_rebuttal)
    except Exception as e:
        rebuttal = {"error": str(e)}

    # Survival reads the rebuttal links; it cannot run if rebuttal failed.
    if "error" in rebuttal:
        survival = {"error": f"rebuttal unavailable: {rebuttal['error']}"}
    else:
        try:
            survival = score_argument_survival(enriched, rebuttal_result=rebuttal)
        except Exception as e:
            survival = {"error": str(e)}

    try:
        information_density = score_information_density_by_speaker(
            [{"speaker": e["speaker"], "turn": e["text"]} for e in enriched],
            topic=topic, model=model)
    except Exception as e:
        information_density = {"error": str(e)}

    stance_agg = aggregate_stance_by_speaker(stance_rows)
    consistency_agg = aggregate_consistency_by_speaker(consistency_rows)
    delta_by_speaker = aggregate_delta_by_speaker(delta_records)

    by_speaker: dict = {}
    for sp in order:
        sp_turns = [p for p in per_turn if p["speaker"] == sp]
        by_speaker[sp] = {
            "stance": stances.get(sp),
            "turns": len(sp_turns),
            "stance_support": stance_agg.get(sp),
            # The three cogency dimensions are averaged separately — they answer
            # different questions and are never combined into one score.
            "logical_cogency": {
                "mean_local_acceptability": _mean(
                    p["logical_cogency"].get("local_acceptability") for p in sp_turns),
                "mean_local_relevance": _mean(
                    p["logical_cogency"].get("local_relevance") for p in sp_turns),
                "mean_local_sufficiency": _mean(
                    p["logical_cogency"].get("local_sufficiency") for p in sp_turns),
            },
            "consistency": consistency_agg.get(sp),
            "delta": delta_by_speaker.get(sp),
            "votes": _mean(p["delta_votes"].get("votes_score") for p in sp_turns),
            "rebuttal": (rebuttal.get("by_speaker") or {}).get(sp),
            "information_density": (information_density or {}).get(sp),
            "argument_survival": (survival.get("by_speaker") or {}).get(sp)
                                 if isinstance(survival, dict) else None,
        }

    log_event("orchestrator", "debate_scored", argument_id=None,
              topic=topic, n_turns=len(turns), speakers=order,
              turns_mined=mined_count)

    result = {
        "topic": topic,
        "stances": stances,
        "hypotheses": hypotheses,
        "by_speaker": by_speaker,
        "debate_level": {
            "rebuttal": rebuttal,
            "information_density": information_density,
            "argument_survival": survival,
        },
    }
    if note:
        result["note"] = note
    if include_detail:
        result["per_turn"] = per_turn

    # ── deterministic winner determination ──────────────────────────────────
    # Uses only the already-computed metric profiles. No LLM call is made here.
    if len(order) == 2:
        try:
            result["winner_determination"] = determine_debate_winner(
                by_speaker=by_speaker,
                speaker_names=order,
            )
        except Exception as e:
            logger.warning("winner determination failed: %s", e)
            result["winner_determination"] = {
                "winner": None,
                "level": None,
                "error": str(e),
            }
    else:
        result["winner_determination"] = {
            "winner": None,
            "level": None,
            "error": (
                "Winner determination requires exactly two speakers; "
                f"received {len(order)}."
            ),
        }

    if write_report:
        try:
            from .debate_report import write_debate_report
            result["report_path"] = write_debate_report(result)
        except Exception as e:
            logger.warning("debate report not written: %s", e)

    return result
