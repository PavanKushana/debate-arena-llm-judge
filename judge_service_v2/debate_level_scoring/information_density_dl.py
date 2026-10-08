"""
information_density_dl.py — debate-level information density (content novelty).

Belongs in: judge_service_v2/debate_level_scoring/information_density_dl.py

WHAT THIS MEASURES
------------------
Across ONE speaker's turns, how much NEW content they introduce as opposed to
restating points they already made. It is a novelty / non-redundancy measure
over the speaker's own propositions, gated by relevance to the topic.

Deliberately distinct from the other metrics:
    cogency   — is the reasoning sound?
    rebuttal  — did the speaker answer the opponent?
    THIS      — does the speaker develop the case, or circle the same point?

ONE PATH, NOT TWO
-----------------
An earlier version ran an NLI baseline and an LLM extractor side by side. They
were not two measurements of one quantity — they computed DIFFERENT quantities
under the same field name:

    NLI:  information_density = mean(novelty x topic_relevance)   an average of
                                                                  probabilities
    LLM:  information_density = on-topic points / statements       a ratio of
                                                                  counts

Numbers from the two paths were never comparable, so any run mixing them was
uninterpretable. The LLM path is kept because the underlying task — deciding
that "money reduces financial stress" and "wealth lowers money-related anxiety"
are the SAME point reworded — is paraphrase clustering, which a language model
does directly and an entailment threshold only approximates.

The LLM does the clustering. The arithmetic is done here, in plain Python, from
the clusters it returns — so the score is reproducible from the logged points
and no number depends on the model doing sums.

new vs repeat is recomputed from TURN ORDER (earliest mention of a point is the
new one), never taken from the model. A mislabel cannot corrupt the counts.

WHAT THE HEADLINE MEANS
-----------------------
    information_density = distinct on-topic points / total statements

Read it as: of everything this speaker said, what share was a point not already
made. 1.0 means every statement carried a new on-topic point; 0.4 means most of
the speaking restated ground already covered. Note the mixed units — points in
the numerator, statements in the denominator — which is what makes it a density
rather than a proportion.

MEASUREMENT CAVEAT
------------------
This measures BREADTH of contribution, not importance or truth. A speaker who
introduces many novel-but-trivial points scores high. It is also not one of
Wachsmuth's dimensions — unlike cogency and global sufficiency, this metric is
the project's own and has no annotated corpus to validate against. Say so when
reporting it.

NOTE ON CALIBRATION
    HIGH/MEDIUM bands are hand-picked round numbers, never checked against
    human judgement. The counts underneath them are the trustworthy part.

Prompt file (MUST stay in step with the schema — `format` is a grammar
constraint, and a prompt asking for a field the schema forbids is a usual cause
of unparseable output):
    information_density_points_system.txt  ->  _InformationPoints
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional

import requests
from pydantic import BaseModel, Field, ValidationError

from ..utils.telemetry import log_call, arg_id
from ..utils.judge_config import OLLAMA_MODEL, OLLAMA_HOST
from ..utils.prompts import load_prompt

logger = logging.getLogger(__name__)

_POINTS_PROMPT_FILE = "information_density_points_system.txt"

# ── Ollama transport (env-overridable) ───────────────────────────────────────
# Connect fails fast; the read timeout must cover a COLD model load, which for
# a large model on a shared host is minutes, not seconds.
INFODENSITY_CONNECT_TIMEOUT = float(os.getenv("INFODENSITY_CONNECT_TIMEOUT", "10"))
INFODENSITY_READ_TIMEOUT = float(os.getenv("INFODENSITY_READ_TIMEOUT", "300"))
INFODENSITY_RETRIES = int(os.getenv("INFODENSITY_RETRIES", "1"))
INFODENSITY_KEEP_ALIVE = os.getenv("INFODENSITY_KEEP_ALIVE", "30m")
# Reasoning models emit a thinking block the grammar constraint forbids, and
# num_predict caps thinking AND answer together — the budget goes to reasoning
# and `content` comes back empty. Set to "" for plain instruct models.
INFODENSITY_THINK = os.getenv("INFODENSITY_THINK", "false")

# ── HIGH/MEDIUM bands on the headline. Uncalibrated. ─────────────────────────
INFODENSITY_HIGH = float(os.getenv("INFODENSITY_HIGH", "0.55"))
INFODENSITY_MEDIUM = float(os.getenv("INFODENSITY_MEDIUM", "0.30"))



# ══════════════════════════════════════════════════════════════════════════════
#  Pure helpers (no model / IO — unit-testable)
# ══════════════════════════════════════════════════════════════════════════════

def _band(x: float) -> str:
    if x >= INFODENSITY_HIGH:
        return "HIGH"
    if x >= INFODENSITY_MEDIUM:
        return "MEDIUM"
    return "LOW"


# ══════════════════════════════════════════════════════════════════════════════
#  Ollama transport
# ══════════════════════════════════════════════════════════════════════════════

def _chat_json(system_prompt: str, user: str, schema: dict,
               model: str, host: str, temperature: float = 0.0,
               num_predict: int = 1536) -> str:
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
        "keep_alive": INFODENSITY_KEEP_ALIVE,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if INFODENSITY_THINK.lower() in ("false", "0", "no"):
        payload["think"] = False

    last: Optional[Exception] = None
    for attempt in range(INFODENSITY_RETRIES + 1):
        try:
            r = requests.post(f"{host}/api/chat", json=payload,
                              timeout=(INFODENSITY_CONNECT_TIMEOUT,
                                       INFODENSITY_READ_TIMEOUT))
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
                    f"Set INFODENSITY_THINK=false, raise num_predict, or use a "
                    f"non-reasoning model.")
            raise RuntimeError(
                f"Model generated {used} tokens but returned empty content "
                f"(done_reason={reason!r}). Its output is not reaching "
                f"Ollama's `content` field — usually a reasoning model that "
                f"does not work with the `format` grammar constraint. Use a "
                f"plain instruct model such as qwen2.5:7b.")

        except requests.Timeout as exc:
            last = exc
            if attempt < INFODENSITY_RETRIES:
                logger.warning("Ollama timed out (attempt %d/%d) — retrying",
                               attempt + 1, INFODENSITY_RETRIES + 1)
                time.sleep(1)
                continue
        except requests.RequestException as exc:
            last = exc
            break
        except (KeyError, IndexError, ValueError) as exc:
            raise RuntimeError("Invalid response from Ollama.") from exc

    raise RuntimeError(f"Ollama request failed after "
                       f"{INFODENSITY_RETRIES + 1} attempt(s): {last}") from last


# ══════════════════════════════════════════════════════════════════════════════
#  The LLM call — extract points, cluster restatements
# ══════════════════════════════════════════════════════════════════════════════

class _Mention(BaseModel):
    turn: int = Field(default=0, description="0-based turn index")
    statement: str = Field(default="", description="the speaker's own wording")
    reason: str = Field(default="", description="why this is a new point, or "
                                                "how it restates the cluster")


class _InfoPoint(BaseModel):
    summary: str = Field(default="", description="the point, in one clause")
    topic_relevant: bool = True
    mentions: list[_Mention] = Field(default_factory=list)


class _InformationPoints(BaseModel):
    points: list[_InfoPoint] = Field(default_factory=list)


def detect_information_points_llm(
    turns: list,
    topic: Optional[str] = None,
    model: str = OLLAMA_MODEL,
    host: str = OLLAMA_HOST,
    temperature: float = 0.0,
    argument_id: Optional[str] = None,
) -> list:
    """Extract the speaker's information points, grouping restatements.

    The model's job is CLUSTERING: deciding that two differently worded
    statements make the same point. That is what it does better than an
    entailment threshold, and it is all it is asked for — every count and ratio
    downstream is computed here from the clusters, so no number depends on the
    model doing arithmetic.

    Returns the parsed list of point dicts.
    """
    if not turns:
        return []

    numbered = "\n".join(f"[turn {i}] {t}" for i, t in enumerate(turns))
    user = f"TOPIC: {topic or '(not given)'}\n\nSPEAKER TURNS:\n{numbered}"
    aid = argument_id or (arg_id(topic) if topic else None)

    with log_call("information_density", "ollama_information_points",
                  argument_id=aid, topic=topic, n_turns=len(turns),
                  model=model, host=host) as rec:
        raw = _chat_json(load_prompt(_POINTS_PROMPT_FILE), user,
                         _InformationPoints.model_json_schema(), model, host,
                         temperature,
                         # ~180 tokens per point cluster; scale with the number
                         # of turns so a long speech is not truncated.
                         num_predict=max(1024, 400 * len(turns)))
        rec["response"] = {"raw_content": raw}

        try:
            parsed = _InformationPoints.model_validate_json(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            # The raw text is the only way to tell the failure shapes apart:
            # truncation, wrong fields (prompt drifted from schema), or prose
            # wrapped around the JSON.
            logger.error("Information points did not match schema. Raw: %s",
                         raw[:600])
            raise RuntimeError(
                f"Information points returned unparseable output: {exc}"
            ) from exc

        points = [p.model_dump() for p in parsed.points if p.mentions]
        rec["response"]["n_points"] = len(points)
        return points


# ══════════════════════════════════════════════════════════════════════════════
#  Deterministic explanation — never an extra model call
# ══════════════════════════════════════════════════════════════════════════════

def _per_turn_reasoning(points: list, n_turns: int) -> list:
    """One plain-English entry per turn, built from the clusters.

    A single turn can touch several points — raising one new idea while
    restating another — so all of a turn's mentions are combined into one
    entry. Uses the model's own per-mention reason when it gave one.
    """
    by_turn: dict = {}
    for p in points:
        mentions = sorted(p.get("mentions", []), key=lambda m: m.get("turn", 0))
        if not mentions:
            continue
        first = mentions[0]
        for idx, m in enumerate(mentions):
            by_turn.setdefault(m.get("turn"), []).append({
                "is_new": idx == 0,
                "on_topic": bool(p.get("topic_relevant", True)),
                "point": p.get("summary", ""),
                "statement": m.get("statement", ""),
                "reason": (m.get("reason") or "").strip(),
                "first_seen_turn": first.get("turn"),
            })

    out = []
    for ti in range(n_turns):
        entries = by_turn.get(ti, [])
        if not entries:
            out.append({"turn": ti,
                        "reasoning": "No substantive point was recorded for "
                                     "this turn."})
            continue
        frags = []
        for e in entries:
            stmt = e["statement"]
            why = (e["reason"] or "").rstrip(".")
            if e["is_new"] and e["on_topic"]:
                frags.append(f'introduces a new point — "{stmt}"'
                             + (f" ({why})" if why else ""))
            elif e["is_new"]:
                frags.append(f'raises "{stmt}", which is new but off-topic, so '
                             f'it does not count toward density')
            else:
                frags.append(f'repeats "{e["point"]}" from turn '
                             f'{e["first_seen_turn"]} — restated as "{stmt}"'
                             + (f" ({why})" if why else ""))
        out.append({"turn": ti, "reasoning": f"Turn {ti}: " + "; ".join(frags) + "."})
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def score_information_density(
    turns: list,
    topic: Optional[str] = None,
    speaker: Optional[str] = None,
    model: Optional[str] = None,
    host: str = OLLAMA_HOST,
    argument_id: Optional[str] = None,
) -> dict:
    """
    Information density of ONE speaker across their ordered `turns`.

    Returns
    -------
    information_density : distinct on-topic points / total statements. Of
                          everything this speaker said, the share that carried a
                          point not already made. Mixed units on purpose —
                          points over statements — which is what makes it a
                          density rather than a proportion.
    label               : HIGH / MEDIUM / LOW (uncalibrated bands)
    distinct_points     : how many separate points were made
    on_topic_points     : of those, how many bore on the topic
    total_statements    : every statement, first mentions and restatements
    restatements        : how many were repeats
    new_statements / repeated_statements : the explicit lists, each carrying the
                          model's own reason
    per_turn_reasoning  : one plain-English entry per turn
    points              : the full clusters, so every count above can be
                          recomputed from the logged output

    Raises RuntimeError if the LLM call fails — there is no second path to fall
    back to, and a fabricated number would be worse than an error.
    """
    n_turns = len(turns)
    aid = argument_id or (arg_id(topic) if topic else None)

    with log_call("information_density", "score", argument_id=aid,
                  speaker=speaker, topic=topic, n_turns=n_turns) as rec:

        points = detect_information_points_llm(
            turns, topic=topic, model=model or OLLAMA_MODEL, host=host,
            argument_id=aid)

        new_statements: list = []
        repeated_statements: list = []
        total_statements = 0
        on_topic_points = 0

        for p in points:
            mentions = sorted(p.get("mentions", []),
                              key=lambda m: m.get("turn", 0))
            if not mentions:
                continue
            total_statements += len(mentions)
            on_topic = bool(p.get("topic_relevant", True))
            if on_topic:
                on_topic_points += 1

            # Earliest mention is the new one — recomputed from turn order, not
            # taken from the model, so a mislabel cannot shift the counts.
            first = mentions[0]
            new_statements.append({
                "turn": first.get("turn"),
                "statement": first.get("statement", ""),
                "point": p.get("summary", ""),
                "topic_relevant": on_topic,
                "reason": (first.get("reason") or "").strip(),
            })
            for m in mentions[1:]:
                repeated_statements.append({
                    "turn": m.get("turn"),
                    "statement": m.get("statement", ""),
                    "repeats_point": p.get("summary", ""),
                    "first_seen_turn": first.get("turn"),
                    "reason": (m.get("reason") or "").strip(),
                })

        distinct_points = len(points)
        if total_statements == 0:
            result = {
                "information_density": None,
                "label": None,
                "distinct_points": 0, "on_topic_points": 0,
                "total_statements": 0, "restatements": 0,
                "new_statements": [], "repeated_statements": [],
                "reasoning": "No substantive statements were found for this "
                             "speaker, so density was not measured.",
                "per_turn_reasoning": [
                    {"turn": i, "reasoning": "No substantive point was "
                                             "recorded for this turn."}
                    for i in range(n_turns)],
                "points": [], "speaker": speaker,
            }
            rec["response"] = {k: v for k, v in result.items() if k != "points"}
            return result

        density = round(on_topic_points / total_statements, 4)
        label = _band(density)

        result = {
            "information_density": density,
            "label": label,
            "distinct_points": distinct_points,
            "on_topic_points": on_topic_points,
            "total_statements": total_statements,
            "restatements": len(repeated_statements),
            "new_statements": new_statements,
            "repeated_statements": repeated_statements,
            "reasoning": (
                f"{label} information density ({density:.2f}) — "
                f"{distinct_points} distinct point"
                f"{'' if distinct_points == 1 else 's'} across "
                f"{total_statements} statement"
                f"{'' if total_statements == 1 else 's'}; "
                f"{on_topic_points} on-topic, "
                f"{len(repeated_statements)} restated."),
            "per_turn_reasoning": _per_turn_reasoning(points, n_turns),
            "points": points,
            "speaker": speaker,
        }
        # Telemetry: same record, minus the bulky cluster list.
        rec["response"] = {k: v for k, v in result.items() if k != "points"}

    logger.info("Information density | speaker=%s | %.3f (%s) | %d points "
                "(%d on-topic) across %d statements, %d restated",
                speaker, density, label, distinct_points, on_topic_points,
                total_statements, len(repeated_statements))
    return result


def score_information_density_by_speaker(
    turn_records: list,
    topic: Optional[str] = None,
    model: Optional[str] = None,
    host: str = OLLAMA_HOST,
) -> dict:
    """
    Group turns by speaker and score each speaker's density.

    turn_records : ordered dicts, each with "speaker" and "turn" (the text).
        Order is preserved per speaker.

    One LLM call per speaker. A failure is recorded against that speaker rather
    than sinking the others — the metric is per-speaker anyway, so a partial
    result is still usable.
    """
    order: list = []
    grouped: dict = {}
    for rec in turn_records:
        sp = rec.get("speaker", "UNKNOWN")
        if sp not in grouped:
            grouped[sp] = []
            order.append(sp)
        grouped[sp].append(rec.get("turn", ""))

    out: dict = {}
    for sp in order:
        try:
            out[sp] = score_information_density(
                turns=grouped[sp], topic=topic, speaker=sp,
                model=model, host=host)
        except Exception as e:
            logger.warning("information density failed for %s (%s)", sp, e)
            out[sp] = {"information_density": None, "label": None,
                       "speaker": sp, "error": str(e)}
    return out
