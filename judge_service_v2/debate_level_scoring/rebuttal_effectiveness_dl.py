"""
rebuttal_effectiveness_dl.py — sentence-level rebuttal across a debate.

WHAT IT PRODUCES
    Which SENTENCE attacks which earlier opponent point, what it targets, and
    how well it lands (1-3, with a quoting rationale). Non-adjacent links are
    found automatically: turn 5 can rebut turn 1.

HOW
    1. TARGETS — every earlier opponent point (claim + each premise). A
       speaker's opening turn cannot rebut, but opening CLAIMS are attackable.
    2. NLI SCREEN — contradiction over all (sentence, target) pairs, top-K kept.
       The one place NLI genuinely fits: a rebuttal IS a contradiction, unlike
       relevance, where the neutral class turned out to mean "does not entail"
       rather than "is not about this". A screen only — never a score.
    3. LLM CONFIRM + SCORE — one call per sentence with candidates: is this a
       real rebuttal, what does it target, how effective, and why.

ATTACK TYPES
    Three, following structured argumentation (Pollock 1987; Prakken 2010;
    Modgil & Prakken 2014):

        undermining   attacks a premise
        rebuttal      attacks the conclusion
        undercut      attacks the inference step

    These are the only structural targets an argument has — its premises, its
    conclusion, and the step between them. An earlier version used five types,
    which conflated TARGET with METHOD: "challenges_evidence" is a sub-case of
    undermining (evidence backs a premise, so attacking it attacks that
    premise's acceptability), and "counterexample" is not a target at all. A
    counterexample can rebut a conclusion or undermine a premise, so it is now
    a separate boolean alongside the type. Three-way classification is also
    measurably more reliable than five-way with a small model.

EFFECTIVENESS
    1-3. Strength of disagreement is NOT effectiveness: "you are completely
    wrong" contradicts as hard as language allows and scores 1, because it
    gives the reader no reason to change their mind. The question is what the
    attack OFFERS.

WHAT WAS REMOVED
    An earlier version computed

        attack_force = max(contra_claim, w * max_premise_contra)
        damage       = relevance * attack_force * counter_strength
        after        = before * (1 - damage)

    Three fatal problems. COLLINEARITY: relevance was 1 - P(neutral) =
    P(entailment) + P(contradiction) and attack_force was essentially
    P(contradiction), so relevance ALREADY CONTAINED attack_force and could
    never be smaller — multiplying them squared one signal rather than
    combining two. SCALE COLLAPSE: a product of three 0-1 values shrinks fast,
    so a solid rebuttal scored 0.25 and banded MEDIUM against a HIGH cut of
    0.35. INVENTED CAUSAL MODEL: `after = before * (1 - damage)` describes how
    a reader revises belief, which nothing measures and no theory predicts.

    Global sufficiency (the per-speaker 1-3 judgment of whether a speaker
    adequately met the opposing case) was also removed. The per-speaker
    coverage COUNTS remain — faced, answered, unaddressed — because they are
    arithmetic over links already computed and cost nothing.

NOTE ON CALIBRATION
    Not validated against human annotation. The 1-3 scale matches the Dagstuhl
    ArgQuality corpus by construction, but nothing here has been checked
    against expert ratings.

Prompt file (MUST stay in step with its schema — `format` is a grammar
constraint, and a prompt asking for a field the schema forbids is a usual cause
of unparseable output):
    rebuttal_sentence_system.txt  ->  _SentenceVerdicts
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Literal, Optional

import requests
from pydantic import BaseModel, Field, ValidationError

from ..utils.telemetry import log_call, log_event, arg_id
from ..utils.judge_config import OLLAMA_MODEL, OLLAMA_HOST
from ..utils.prompts import load_prompt
# Reuse the single shared DeBERTa NLI — no second model is loaded.
from ..turn_level_scoring.stance_support_score import _nli_probs

logger = logging.getLogger(__name__)

_SENTENCE_PROMPT_FILE = "rebuttal_sentence_system.txt"

# Pollock (1987): rebuttal and undercut. ASPIC+ (Prakken 2010) adds
# undermining. The three structural targets an argument has.
_ATTACK_TYPES = ("undermining", "undercut", "rebuttal")

# ── Ollama transport (env-overridable) ───────────────────────────────────────
REBUTTAL_CONNECT_TIMEOUT = float(os.getenv("REBUTTAL_CONNECT_TIMEOUT", "10"))
REBUTTAL_READ_TIMEOUT = float(os.getenv("REBUTTAL_READ_TIMEOUT", "300"))
REBUTTAL_RETRIES = int(os.getenv("REBUTTAL_RETRIES", "1"))
REBUTTAL_KEEP_ALIVE = os.getenv("REBUTTAL_KEEP_ALIVE", "30m")
# Reasoning models emit a thinking block the grammar constraint forbids, and
# num_predict caps thinking AND answer together — the budget goes to reasoning
# and `content` comes back empty. Set to "" for plain instruct models.
REBUTTAL_THINK = os.getenv("REBUTTAL_THINK", "false")

# ── NLI screen (candidate shortlisting only, never a score) ──────────────────
REBUTTAL_SCREEN_MIN = float(os.getenv("REBUTTAL_SCREEN_MIN", "0.40"))
REBUTTAL_TOPK = int(os.getenv("REBUTTAL_TOPK", "2"))
# A speaker's opening turn is an intro, not a rebuttal. Its CLAIM stays
# attackable — only its role as a rebutter is excluded.
REBUTTAL_EXCLUDE_OPENERS = os.getenv("REBUTTAL_EXCLUDE_OPENERS", "1") == "1"

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\b\w+\b")


# ══════════════════════════════════════════════════════════════════════════════
#  Pure helpers
# ══════════════════════════════════════════════════════════════════════════════

def _as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, str):
        return [x] if x.strip() else []
    return [str(i).strip() for i in x if str(i).strip()]


def _sentences(text: str) -> list:
    """Split a turn into rebutting units. Fragments under three words are
    dropped — they carry no attack and only waste NLI passes."""
    parts = [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]
    return [s for s in parts if len(_WORD.findall(s)) >= 3]


def _score_1to3(x) -> Optional[int]:
    try:
        return max(1, min(3, int(round(float(x)))))
    except (TypeError, ValueError):
        return None


def _band_word(score: Optional[int]) -> str:
    return {1: "low", 2: "moderate", 3: "high"}.get(score, "not measured")


# ══════════════════════════════════════════════════════════════════════════════
#  Ollama transport
# ══════════════════════════════════════════════════════════════════════════════

def _chat_json(system_prompt: str, user: str, schema: dict,
               model: str, host: str, temperature: float = 0.0,
               num_predict: int = 1024) -> str:
    """One schema-constrained chat call. Returns the raw JSON string.

    Empty content after a completed generation is diagnosed here rather than
    surfacing downstream as "unparseable output" — the causes look identical
    from the parser's side but need opposite fixes.
    """
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user}],
        "stream": False,
        "format": schema,
        "keep_alive": REBUTTAL_KEEP_ALIVE,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if REBUTTAL_THINK.lower() in ("false", "0", "no"):
        payload["think"] = False

    last: Optional[Exception] = None
    for attempt in range(REBUTTAL_RETRIES + 1):
        try:
            r = requests.post(f"{host}/api/chat", json=payload,
                              timeout=(REBUTTAL_CONNECT_TIMEOUT,
                                       REBUTTAL_READ_TIMEOUT))
            r.raise_for_status()
            body = r.json()
            msg = body.get("message", {})
            content = (msg.get("content") or "").strip()
            if content:
                return content

            thinking = msg.get("thinking") or ""
            reason = body.get("done_reason", "")
            used = body.get("eval_count", 0)
            logger.error("Empty content from %s | message keys=%s | body=%s",
                         model, list(msg.keys()), json.dumps(body)[:800])
            if thinking:
                raise RuntimeError(
                    f"Model returned a reasoning block and no answer "
                    f"({len(thinking)} chars of thinking, {used}/{num_predict} "
                    f"tokens). num_predict caps thinking AND answer together. "
                    f"Set REBUTTAL_THINK=false, raise num_predict, or use a "
                    f"non-reasoning model.")
            raise RuntimeError(
                f"Model generated {used} tokens but returned empty content "
                f"(done_reason={reason!r}). Its output is not reaching "
                f"Ollama's `content` field — usually a reasoning model that "
                f"does not work with the `format` grammar constraint. Use a "
                f"plain instruct model such as qwen2.5:7b.")

        except requests.Timeout as exc:
            last = exc
            if attempt < REBUTTAL_RETRIES:
                logger.warning("Ollama timed out (attempt %d/%d) — retrying",
                               attempt + 1, REBUTTAL_RETRIES + 1)
                time.sleep(1)
                continue
        except requests.RequestException as exc:
            last = exc
            break
        except (KeyError, IndexError, ValueError) as exc:
            raise RuntimeError("Invalid response from Ollama.") from exc

    raise RuntimeError(f"Ollama request failed after "
                       f"{REBUTTAL_RETRIES + 1} attempt(s): {last}") from last


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — LLM confirm + type + score one sentence's candidate targets
# ══════════════════════════════════════════════════════════════════════════════

class _CandVerdict(BaseModel):
    # All six fields are intentionally REQUIRED. Defaults here are dangerous:
    # if the model omits is_rebuttal or attack_type, Pydantic would silently
    # invent a value and a real rebuttal could be discarded.
    id: int = Field(description="candidate id from the input list")
    is_rebuttal: bool
    attack_type: Literal["undermining", "undercut", "rebuttal"]
    # A counterexample is a METHOD, not a target — it can rebut a conclusion or
    # undermine a premise — so it is a flag alongside the type, not one of them.
    is_counterexample: bool
    # Keep this short. The prompt asks for a compact rationale, and the field
    # comes before effectiveness so the model explains first and scores second.
    rationale: str
    # Nullable because non-rebuttals have no effectiveness score. The field is
    # still REQUIRED, so the model must explicitly output null for them.
    effectiveness: Optional[int] = Field(ge=1, le=3)


class _SentenceVerdicts(BaseModel):
    verdicts: list[_CandVerdict] = Field(default_factory=list)


def judge_sentence_llm(sentence: str, candidates: list,
                       model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST,
                       argument_id: Optional[str] = None) -> dict:
    """Confirm, type and score one sentence's candidate targets in ONE call.

    Each candidate includes the FULL opponent-turn argument context (claim +
    premises + original turn text), while ``target_text`` identifies the exact
    point shortlisted by NLI.  Returns both parsed verdicts and diagnostics so
    failures are visible instead of silently becoming an empty rebuttal list.
    """
    blocks = []
    for c in candidates:
        premises = c.get("argument_premises") or []
        premise_lines = "\n".join(f"  - {p}" for p in premises) or "  - (none provided)"
        blocks.append(
            f'CANDIDATE {c["id"]}\n'
            f'Exact NLI target: ({c["kind"]}, turn {c["turn"]}) "{c["target_text"]}"\n'
            f'Opponent full argument context:\n'
            f'  Claim: {c.get("argument_claim") or "(none provided)"}\n'
            f'  Premises:\n{premise_lines}\n'
            f'  Original turn text: {c.get("argument_text") or "(none provided)"}'
        )
    user = (f'REBUTTING SENTENCE:\n"{sentence}"\n\n'
            f'CANDIDATE OPPONENT ARGUMENTS:\n\n' + "\n\n".join(blocks))

    debug = {"user_prompt": user, "raw_response": None, "error": None}
    print("\n" + "=" * 90)
    print("[REBUTTAL][LLM INPUT]")
    print(user)

    try:
        with log_call("rebuttal_effectiveness", "ollama_sentence_judgment",
                      argument_id=argument_id, sentence=sentence,
                      n_candidates=len(candidates), model=model,
                      host=host) as rec:
            raw = _chat_json(load_prompt(_SENTENCE_PROMPT_FILE), user,
                             _SentenceVerdicts.model_json_schema(),
                             model, host,
                             num_predict=max(500, 260 * len(candidates)))
            debug["raw_response"] = raw
            rec["response"] = {"raw_content": raw}
            print("[REBUTTAL][LLM RAW RESPONSE]")
            print(raw)

            # Defensive guard: a runaway rationale should fail explicitly
            # rather than consuming the whole generation budget.
            if len(raw) > 12000:
                raise RuntimeError(
                    f"LLM response is unexpectedly long ({len(raw)} characters); "
                    "rationale generation appears to be runaway."
                )

            try:
                parsed = _SentenceVerdicts.model_validate_json(raw)
            except (json.JSONDecodeError, ValidationError) as exc:
                truncated = not raw.rstrip().endswith("}")
                hint = (" — output was cut off before the JSON closed; "
                        "the model likely exceeded the compact-output budget.") if truncated else ""
                raise RuntimeError(
                    f"Sentence judgment returned unparseable output: {exc}{hint}"
                ) from exc

            out = {}
            for v in parsed.verdicts:
                # Literal validation guarantees the exact three attack labels.
                # Do not silently replace an invalid label with "rebuttal".
                # Also never let a non-rebuttal receive an effectiveness score.
                if v.is_rebuttal and v.effectiveness is None:
                    raise ValueError(
                        f"Candidate {v.id} is_rebuttal=true but effectiveness is null"
                    )
                if not v.rationale.strip():
                    raise ValueError(f"Candidate {v.id} has an empty rationale")

                out[v.id] = {
                    "is_rebuttal": bool(v.is_rebuttal),
                    "attack_type": v.attack_type,
                    "is_counterexample": bool(v.is_counterexample),
                    "effectiveness": (_score_1to3(v.effectiveness)
                                      if v.is_rebuttal else None),
                    "rationale": v.rationale.strip(),
                }
            rec["response"]["verdicts"] = out
            print("[REBUTTAL][LLM PARSED VERDICTS]")
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return {"verdicts": out, "debug": debug}
    except Exception as e:
        debug["error"] = str(e)
        logger.warning("sentence judgment failed (%s) — this sentence is "
                       "recorded as unjudged, not as a non-rebuttal", e)
        print("[REBUTTAL][LLM ERROR]")
        print(str(e))
        return {"verdicts": {}, "debug": debug}


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def score_rebuttal_debate(
    turns: list,
    model: Optional[str] = None,
    screen_min: Optional[float] = None,
    top_k: Optional[int] = None,
    exclude_openers: Optional[bool] = None,
    use_llm: bool = True,
) -> dict:
    """
    Rebuttal across a whole debate, at two levels.

    Parameters
    ----------
    turns : ordered turn dicts, each
        {"speaker": str, "turn_no": int (optional), "text": str,
         "claim": str, "premises": [str]}
        Non-adjacent links are found automatically: turn 5 may rebut turn 1.
    screen_min, top_k : NLI contradiction screen and candidates per sentence.
        The screen only SHORTLISTS; it never scores.
    use_llm : False runs the NLI screen alone and reports candidates without
        confirming or scoring them — useful for inspecting the screen itself.

    Returns
    -------
    {
      "rebuttals":  [ per-sentence attacks with target, type, 1-3, rationale ],
      "per_turn":   [ {turn_no, speaker, rebuttals_made, mean_effectiveness} ],
      "by_speaker": { speaker: {rebuttals_made, mean_effectiveness,
                                points_faced, points_answered,
                                points_unaddressed, unaddressed} },
      "params": {...}
    }

    `mean_effectiveness` describes only the attacks a speaker MADE — it cannot
    see the points they ignored, since an unanswered point produces no rebuttal
    row at all. `unaddressed` names those points, so read the two together.
    """
    screen_min = REBUTTAL_SCREEN_MIN if screen_min is None else screen_min
    top_k = REBUTTAL_TOPK if top_k is None else top_k
    if exclude_openers is None:
        exclude_openers = REBUTTAL_EXCLUDE_OPENERS
    mdl = model or OLLAMA_MODEL

    for i, t in enumerate(turns):
        t.setdefault("turn_no", i)

    openers: dict = {}
    for t in turns:
        openers.setdefault(t.get("speaker", "UNKNOWN"), t["turn_no"])

    # Every point anyone put on the table, keyed for coverage bookkeeping.
    # (speaker, turn_no, kind, text) -> record
    all_points: dict = {}
    for t in turns:
        sp = t.get("speaker", "UNKNOWN")
        claim = (t.get("claim") or "").strip()
        if claim:
            all_points[(sp, t["turn_no"], "claim", claim)] = {
                "speaker": sp, "turn": t["turn_no"], "kind": "claim",
                "text": claim, "argument_claim": claim,
                "argument_premises": _as_list(t.get("premises")),
                "argument_text": (t.get("text") or "").strip(),
                "answered": False, "best_effectiveness": None}
        for p in _as_list(t.get("premises")):
            all_points[(sp, t["turn_no"], "premise", p)] = {
                "speaker": sp, "turn": t["turn_no"], "kind": "premise",
                "text": p, "argument_claim": claim,
                "argument_premises": _as_list(t.get("premises")),
                "argument_text": (t.get("text") or "").strip(),
                "answered": False, "best_effectiveness": None}

    rebuttals: list = []
    per_turn: list = []
    diagnostics: list = []

    for i, t in enumerate(turns):
        sp = t.get("speaker", "UNKNOWN")
        tno = t["turn_no"]
        if exclude_openers and tno == openers.get(sp):
            continue    # an opening turn introduces; it does not rebut

        # ── STEP 1: target pool — every earlier OPPONENT point ──────────────
        targets = [rec for key, rec in all_points.items()
                   if rec["speaker"] != sp and rec["turn"] < tno]
        if not targets:
            continue

        turn_scores = []
        for sent in _sentences(t.get("text", "")):
            # ── STEP 2: NLI screen. Contradiction is the right signal here —
            # a rebuttal IS a contradiction. Used only to shortlist.
            scored = []
            for tgt in targets:
                probs = _nli_probs(sent, tgt["text"])
                scored.append((probs["contradiction"], tgt))
            scored.sort(key=lambda x: x[0], reverse=True)
            candidates = [(c, tgt) for c, tgt in scored if c >= screen_min][:top_k]

            nli_debug = [
                {
                    "candidate_id": idx,
                    "nli_contradiction": round(contra, 4),
                    "target": {
                        "turn": tgt["turn"], "speaker": tgt["speaker"],
                        "kind": tgt["kind"], "text": tgt["text"]
                    },
                    "full_argument_context": {
                        "claim": tgt.get("argument_claim", ""),
                        "premises": tgt.get("argument_premises", []),
                        "text": tgt.get("argument_text", ""),
                    },
                }
                for idx, (contra, tgt) in enumerate(candidates)
            ]

            print("\n" + "-" * 90)
            print(f"[REBUTTAL][NLI] turn={tno} speaker={sp}")
            print(f"[REBUTTAL][SENTENCE] {sent}")
            print("[REBUTTAL][NLI FILTERED CANDIDATES]")
            print(json.dumps(nli_debug, indent=2, ensure_ascii=False))

            sentence_debug = {
                "turn": tno, "speaker": sp, "sentence": sent,
                "nli_filtered_candidates": nli_debug,
                "llm": None,
                "final_confirmed_rebuttals": [],
            }

            if not candidates:
                sentence_debug["status"] = "no_nli_candidate_above_threshold"
                diagnostics.append(sentence_debug)
                continue

            # ── STEP 3: LLM confirm + type + score ──────────────────────────
            verdicts = {}
            if use_llm:
                payload = [{
                    "id": idx,
                    "kind": tgt["kind"],
                    "turn": tgt["turn"],
                    "target_text": tgt["text"],
                    "argument_claim": tgt.get("argument_claim", ""),
                    "argument_premises": tgt.get("argument_premises", []),
                    "argument_text": tgt.get("argument_text", ""),
                } for idx, (_, tgt) in enumerate(candidates)]
                llm_result = judge_sentence_llm(
                    sent, payload, model=mdl, host=OLLAMA_HOST,
                    argument_id=arg_id(sent)
                )
                verdicts = llm_result["verdicts"]
                sentence_debug["llm"] = llm_result["debug"] | {
                    "parsed_verdicts": verdicts
                }
            else:
                sentence_debug["llm"] = {"skipped": True}

            for idx, (contra, tgt) in enumerate(candidates):
                if use_llm:
                    if not verdicts:
                        continue            # judgment failed — not a verdict
                    v = verdicts.get(idx)
                    if not v or not v["is_rebuttal"]:
                        continue            # LLM says this is not a rebuttal
                    attack_type = v["attack_type"]
                    effectiveness = v["effectiveness"]
                    rationale = v["rationale"]
                    source = "llm"
                else:
                    attack_type = None
                    effectiveness = None
                    rationale = ""
                    source = "nli_screen_only"

                result_row = {
                    "from": {"turn": tno, "speaker": sp, "sentence": sent},
                    "target": {"turn": tgt["turn"], "speaker": tgt["speaker"],
                               "kind": tgt["kind"], "text": tgt["text"]},
                    "attack_type": attack_type,
                    "effectiveness": effectiveness,
                    "label": _band_word(effectiveness),
                    "rationale": rationale,
                    "nli_contradiction": round(contra, 4),   # screen value only
                    "source": source,
                }
                rebuttals.append(result_row)
                sentence_debug["final_confirmed_rebuttals"].append(result_row)

                if effectiveness is not None:
                    turn_scores.append(effectiveness)
                    tgt["answered"] = True
                    prev = tgt["best_effectiveness"]
                    tgt["best_effectiveness"] = (effectiveness if prev is None
                                                 else max(prev, effectiveness))

            sentence_debug["status"] = (
                "confirmed_rebuttal" if sentence_debug["final_confirmed_rebuttals"]
                else "nli_candidates_rejected_or_llm_failed"
            )
            diagnostics.append(sentence_debug)
            print("[REBUTTAL][FINAL FOR SENTENCE]")
            print(json.dumps(sentence_debug["final_confirmed_rebuttals"],
                             indent=2, ensure_ascii=False))

        if turn_scores:
            per_turn.append({
                "turn_no": tno, "speaker": sp,
                "rebuttals_made": len(turn_scores),
                "mean_effectiveness": round(sum(turn_scores) / len(turn_scores), 2)})

        log_event("rebuttal_effectiveness", "debate_turn", argument_id=None,
                  turn=tno, speaker=sp, rebuttals_made=len(turn_scores))

    # ── STEP 4: per-speaker rollup + global sufficiency ─────────────────────
    speakers = sorted({t.get("speaker", "UNKNOWN") for t in turns})
    by_speaker: dict = {}

    for sp in speakers:
        made = [r["effectiveness"] for r in rebuttals
                if r["from"]["speaker"] == sp and r["effectiveness"] is not None]
        # Everything the OTHER side put to this speaker.
        faced = [rec for rec in all_points.values() if rec["speaker"] != sp]
        answered = [c for c in faced if c["answered"]]
        unaddressed = [c for c in faced if not c["answered"]]

        entry = {
            "rebuttals_made": len(made),
            "mean_effectiveness": (round(sum(made) / len(made), 2)
                                   if made else None),
            "points_faced": len(faced),
            "points_answered": len(answered),
            "points_unaddressed": len(unaddressed),
            # Free bookkeeping over links already computed — no LLM call.
            # Delete these three lines if the counts are not wanted.
            "unaddressed": [{"turn": c["turn"], "kind": c["kind"],
                             "text": c["text"]} for c in unaddressed],
        }

        by_speaker[sp] = entry
        log_event("rebuttal_effectiveness", "speaker_summary", argument_id=None,
                  speaker=sp, rebuttals_made=len(made),
                  mean_effectiveness=entry["mean_effectiveness"],
                  points_faced=len(faced), points_answered=len(answered))
        logger.info("Rebuttal %s | made=%d mean=%s | faced=%d answered=%d "
                    "unaddressed=%d",
                    sp, len(made), entry["mean_effectiveness"], len(faced),
                    len(answered), len(unaddressed))

    return {
        "rebuttals": rebuttals,
        "per_turn": per_turn,
        "by_speaker": by_speaker,
        "diagnostics": diagnostics,
        "params": {"screen_min": screen_min, "top_k": top_k,
                   "exclude_openers": exclude_openers, "use_llm": use_llm,
                   "screen_note": "nli_contradiction shortlists candidates "
                                  "only; it is never a score"},
    }
