"""
argument_survival_dl.py — did each argument survive the debate?

Pure bookkeeping over the rebuttal result. It runs NO model of its own: every
attack it reads was already found and scored by rebuttal_effectiveness_dl, so
survival costs nothing beyond the rebuttal pass.

WHAT CHANGED, AND WHY
An earlier version ran its own linking pass and kept a decimal survival value:

    attack  : survival <- survival * (1 - eff)
    defense : survival <- survival + (1 - survival) * eff

That was removed. Three problems: it re-scored links the rebuttal pass had
already found (duplicate NLI and LLM calls); it allowed only ONE move per turn,
so a turn attacking three arguments updated one and ignored two; and the decay
compounded, so three moderate hits produced 0.125 from the multiplication rather
than from anything measured. Chained arbitrary coefficients are arbitrary
several times over.

Status is discrete instead. An argument is:

    unchallenged  never attacked
    defended      attacked, and its owner answered the attacker
    weakened      attacked with low or moderate force, never answered
    refuted       attacked with high force (3/3), never answered

All four are countable from the rebuttal links. No coefficients, no compounding,
and the per-argument history still shows exactly what happened.

A "defense" here means the owner attacked back at the turn that attacked them —
the only defensive move visible in the rebuttal output. It does not detect an
owner restating or shoring up their own point without engaging the attacker.
"""
from __future__ import annotations

import logging
from typing import Optional

from ..utils.telemetry import log_event

logger = logging.getLogger(__name__)

STATUS_ORDER = ("unchallenged", "defended", "weakened", "refuted")


def score_argument_survival(turns: list,
                            rebuttal_result: Optional[dict] = None,
                            model: Optional[str] = None) -> dict:
    """
    turns           : ordered turn dicts {speaker, turn_no, claim, premises}.
    rebuttal_result : the output of score_rebuttal_debate over the SAME turns.
                      Required — survival is a view of those links, not a second
                      scoring pass.

    Returns per-argument status and per-speaker counts.
    """
    if not rebuttal_result or "rebuttals" not in rebuttal_result:
        raise ValueError(
            "rebuttal_result from score_rebuttal_debate is required — survival "
            "reads the links that pass already found rather than recomputing them")

    for i, t in enumerate(turns):
        t.setdefault("turn_no", i)

    rebuttals = rebuttal_result.get("rebuttals") or []

    # turn_no -> the attacks that hit that turn's CLAIM
    attacks_on: dict = {}
    for r in rebuttals:
        tgt = r.get("target", {})
        if tgt.get("kind") != "claim":
            continue
        attacks_on.setdefault(tgt.get("turn"), []).append(r)

    # Who attacked whom, so a counter-attack can be recognised as a defense.
    attacked_turns_by: dict = {}
    for r in rebuttals:
        attacked_turns_by.setdefault(r["from"]["speaker"], set()).add(
            r["target"]["turn"])

    arguments: list = []
    for t in turns:
        tno, owner = t["turn_no"], t.get("speaker", "UNKNOWN")
        claim = (t.get("claim") or "").strip()
        if not claim:
            continue

        hits = attacks_on.get(tno, [])
        # Defense: the owner attacked back at a turn that attacked this claim.
        attacker_turns = {h["from"]["turn"] for h in hits}
        defended = bool(attacker_turns & attacked_turns_by.get(owner, set()))
        strongest = max((h.get("effectiveness") or 0) for h in hits) if hits else 0

        if not hits:
            status = "unchallenged"
        elif defended:
            status = "defended"
        elif strongest >= 3:
            status = "refuted"
        else:
            status = "weakened"

        arguments.append({
            "turn_no": tno,
            "owner": owner,
            "claim": claim,
            "status": status,
            "attacks": [{"from_turn": h["from"]["speaker"] and h["from"]["turn"],
                         "from_speaker": h["from"]["speaker"],
                         "attack_type": h.get("attack_type"),
                         "effectiveness": h.get("effectiveness"),
                         "sentence": h["from"]["sentence"]} for h in hits],
            "strongest_attack": strongest or None,
            "defended": defended,
        })
        log_event("argument_survival", "argument", argument_id=None,
                  turn=tno, owner=owner, status=status,
                  n_attacks=len(hits), strongest_attack=strongest or None)

    by_speaker: dict = {}
    for sp in sorted({a["owner"] for a in arguments}):
        mine = [a for a in arguments if a["owner"] == sp]
        counts = {s: sum(1 for a in mine if a["status"] == s)
                  for s in STATUS_ORDER}
        survived = counts["unchallenged"] + counts["defended"]
        by_speaker[sp] = {
            "arguments_introduced": len(mine),
            "survived": survived,
            "survival_rate": (round(survived / len(mine), 4) if mine else None),
            **counts,
            "refuted_turns": [a["turn_no"] for a in mine
                              if a["status"] == "refuted"],
        }
        log_event("argument_survival", "speaker_summary", argument_id=None,
                  speaker=sp, **by_speaker[sp])
        logger.info("Argument survival | %s | %d/%d survived (%s)",
                    sp, survived, len(mine),
                    ", ".join(f"{k}={v}" for k, v in counts.items() if v))

    return {"arguments": arguments, "by_speaker": by_speaker}
