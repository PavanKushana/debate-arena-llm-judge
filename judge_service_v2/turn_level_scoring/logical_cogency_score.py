"""
logical_cogency_score.py — cogency as local acceptability, relevance, and
sufficiency (Wachsmuth et al. 2017).

Belongs in: judge_service_v2/turn_level_scoring/logical_cogency_score.py

THE FRAMEWORK
-------------
Cogency is the logical dimension of argument quality, defined by three
sub-dimensions (Wachsmuth et al., "Computational Argumentation Quality
Assessment in Natural Language", EACL 2017):

    local acceptability : premises worthy of being believed
    local relevance     : premises support or attack the conclusion
    local sufficiency   : premises enough to draw the conclusion

Expert annotators rate each on a 1 (low) to 3 (high) scale, judging the premise
SET against the conclusion. This module returns the same three scores on the
same scale, so results can be compared against the Dagstuhl ArgQuality corpus
without rescaling.

ONE LLM CALL, RATIONALE BEFORE SCORE
------------------------------------
All three dimensions come from a single call: the model reads the argument
once, and the three judgments stay mutually consistent. In the schema each
rationale is declared BEFORE its score, because field order controls generation
order — the model reasons first and then commits to a number, rather than
picking a number and justifying it afterwards. Each rationale must QUOTE the
premise text it turns on, so a reader can see what drove the score.

KEEPING THE DIMENSIONS INDEPENDENT
----------------------------------
Sufficiency is judged GRANTING the premises as true. Without that instruction a
model that has just scored acceptability 1 drags sufficiency down with it and
the dimensions collapse into one. An argument can be well formed and rest on a
falsehood; that combination is informative and must stay visible.

Relevance and sufficiency are likewise distinct: relevance asks whether the
premises BEAR ON the conclusion, sufficiency whether they ESTABLISH it. A
premise giving weak but genuine support is fully relevant and only partly
sufficient.

WHY RELEVANCE IS NOT SCORED BY NLI
----------------------------------
An earlier version computed relevance as 1 - P(neutral) from DeBERTa, on the
theory that NLI's neutral class is exactly "neither supports nor attacks". The
theory is sound; the model does not behave that way on non-deductive arguments.
Measured on a real turn:

    claim "Money contributes to happiness."
      "Money lets people travel."          -> neutral 0.973, relevance 0.027
      "Money reduces financial stress."    -> neutral 0.943, relevance 0.057

Both premises are plainly on topic. NLI reads them as neutral because neither
ENTAILS the claim, which is a different question. The values are not near any
threshold, so no cut point rescues it — the measure has almost no
discriminative range on ordinary argumentative text.

Relevance therefore moved into the LLM call with the other two dimensions. The
NLI computation is KEPT and still reported as `local_relevance_nli_diagnostic`
and `relevance_per_premise`: DeBERTa is already loaded for stance scoring so it
costs nothing, and "we tried NLI for relevance and it did not discriminate" is
a result worth reporting rather than hiding. It is a diagnostic, NOT a score.

Note that the NLI diagnostic is computed per premise and reported as the
maximum. Concatenating premises into one NLI input drifts far outside the
model's single-sentence training distribution and collapses everything toward
neutral (an observed case read 0.96 neutral for two on-topic premises).

NO COMPOSITE SCORE
------------------
The three dimensions are reported side by side, never averaged. Wachsmuth's
cogency is annotated by humans as its own dimension, not computed from the
three, so a mean here would be an invention rather than the framework.

NOTE ON CALIBRATION
    Not validated against human annotation. The 1-3 scale matches the corpus by
    construction, but nothing here has been checked against expert ratings.

Prompt file (MUST stay in step with the schema — `format` is a grammar
constraint, and a prompt asking for a field the schema forbids is one of the
usual causes of unparseable output):
    cogency_judgment_system.txt  ->  _CogencyJudgment

Prerequisites:
    pip install torch transformers        # NLI diagnostic (shared with stance)
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import requests
from pydantic import BaseModel, Field, ValidationError

from ..utils.telemetry import log_call, arg_id
from ..utils.judge_config import OLLAMA_MODEL, OLLAMA_HOST
from ..utils.prompts import load_prompt
# Reuse the stance module's DeBERTa NLI so only ONE copy is ever loaded.
from .stance_support_score import _nli_probs

logger = logging.getLogger(__name__)

_JUDGMENT_PROMPT_FILE = "cogency_judgment_system.txt"

# ── Ollama transport (env-overridable) ───────────────────────────────────────
# Connect fails fast; the read timeout must cover a COLD model load, which for
# a large model on a shared host is minutes, not seconds.
COGENCY_CONNECT_TIMEOUT = float(os.getenv("COGENCY_CONNECT_TIMEOUT", "10"))
COGENCY_READ_TIMEOUT = float(os.getenv("COGENCY_READ_TIMEOUT", "300"))
COGENCY_RETRIES = int(os.getenv("COGENCY_RETRIES", "1"))
COGENCY_KEEP_ALIVE = os.getenv("COGENCY_KEEP_ALIVE", "30m")
# Reasoning models (gpt-oss, deepseek-r1, qwen3) emit a thinking block that the
# grammar constraint forbids, and num_predict caps thinking AND answer
# together, so the budget goes to reasoning and `content` comes back empty.
# Set to "" for plain instruct models, which reject the field on some builds.
COGENCY_THINK = os.getenv("COGENCY_THINK", "false")
# Whether to run the NLI relevance diagnostic. Free in wall-clock terms
# (DeBERTa is already resident for stance), but skippable for batch runs.
COGENCY_NLI_DIAGNOSTIC = os.getenv("COGENCY_NLI_DIAGNOSTIC", "true")


# ══════════════════════════════════════════════════════════════════════════════
#  Pure helpers (no model / IO — unit-testable)
# ══════════════════════════════════════════════════════════════════════════════

def _as_list(premises) -> list:
    if premises is None:
        return []
    if isinstance(premises, str):
        return [premises] if premises.strip() else []
    return [str(p).strip() for p in premises if str(p).strip()]


def _score_1to3(x) -> Optional[int]:
    """Clamp an LLM-returned score to the integer range 1-3."""
    try:
        return max(1, min(3, int(round(float(x)))))
    except (TypeError, ValueError):
        return None


def _band_word(score: Optional[int]) -> str:
    return {1: "low", 2: "moderate", 3: "high"}.get(score, "not measured")


def _enabled(flag: str) -> bool:
    return flag.lower() not in ("false", "0", "no", "")


# ══════════════════════════════════════════════════════════════════════════════
#  NLI relevance — DIAGNOSTIC ONLY, not a score
# ══════════════════════════════════════════════════════════════════════════════

def relevance_nli_diagnostic(claim: str, premises) -> dict:
    """1 - P(neutral) per premise, maximum taken. NOT the relevance score.

    Kept because DeBERTa is already loaded and the numbers document a real
    negative result: on non-deductive arguments this sits near zero even for
    plainly on-topic premises, because NLI's neutral class means "does not
    entail", not "is not about this". See the module docstring for the measured
    example.

    Per premise, not joined: concatenation drifts outside the model's
    single-sentence training distribution and collapses toward neutral.
    """
    prem_list = _as_list(premises)
    if not prem_list:
        return {"max": None, "per_premise": []}

    per = []
    for p in prem_list:
        probs = _nli_probs(p, claim)
        per.append({"premise": p,
                    "relevance_raw": round(1.0 - probs["neutral"], 4),
                    "probs": {k: round(v, 4) for k, v in probs.items()}})

    return {"max": round(max(x["relevance_raw"] for x in per), 4),
            "per_premise": per}


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
        "keep_alive": COGENCY_KEEP_ALIVE,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if COGENCY_THINK.lower() in ("false", "0", "no"):
        payload["think"] = False

    last: Optional[Exception] = None
    for attempt in range(COGENCY_RETRIES + 1):
        try:
            r = requests.post(f"{host}/api/chat", json=payload,
                              timeout=(COGENCY_CONNECT_TIMEOUT,
                                       COGENCY_READ_TIMEOUT))
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
                    f"tokens, done_reason={reason!r}). num_predict caps "
                    f"thinking AND answer together. Set COGENCY_THINK=false, "
                    f"raise num_predict, or use a non-reasoning model.")
            raise RuntimeError(
                f"Model generated {used} tokens but returned empty content "
                f"(done_reason={reason!r}, message keys={list(msg.keys())}). "
                f"The model's output is not reaching Ollama's `content` field "
                f"— usually a reasoning model that does not work with the "
                f"`format` grammar constraint. Use a plain instruct model such "
                f"as qwen2.5:7b or llama3.1:8b.")

        except requests.Timeout as exc:
            last = exc
            if attempt < COGENCY_RETRIES:
                logger.warning("Ollama timed out (attempt %d/%d) — the model "
                               "may have been cold; retrying",
                               attempt + 1, COGENCY_RETRIES + 1)
                time.sleep(1)
                continue
        except requests.RequestException as exc:
            last = exc
            break
        except (KeyError, IndexError, ValueError) as exc:
            raise RuntimeError("Invalid response from Ollama.") from exc

    raise RuntimeError(f"Ollama request failed after "
                       f"{COGENCY_RETRIES + 1} attempt(s): {last}") from last


# ══════════════════════════════════════════════════════════════════════════════
#  The three dimensions — ONE LLM call
# ══════════════════════════════════════════════════════════════════════════════

class _CogencyJudgment(BaseModel):
    # Each rationale is declared BEFORE its score: field order controls
    # generation order, so the model reasons first and then commits to a
    # number. Dimension order follows the paper: acceptability, relevance,
    # sufficiency.
    acceptability_rationale: str = Field(
        default="", description="Why this acceptability score. Must quote the "
                                "premise text being judged.")
    # Score fields are REQUIRED (no default): a missing/partial LLM response
    # must fail validation and be reported as null, never silently become a
    # constant middle score of 2.
    acceptability: int = Field(ge=1, le=3)
    relevance_rationale: str = Field(
        default="", description="Why this relevance score. Must quote the "
                                "premise text being judged.")
    relevance: int = Field(ge=1, le=3)
    sufficiency_rationale: str = Field(
        default="", description="Why this sufficiency score. Must quote the "
                                "premise text, and name the gap if any.")
    sufficiency: int = Field(ge=1, le=3)


def judge_cogency_llm(claim: str, premises,
                      model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST,
                      temperature: float = 0.0,
                      argument_id: Optional[str] = None) -> dict:
    """All three cogency sub-dimensions, judged in ONE call.

    Each is a judgment call with no deterministic route: "worthy of being
    believed", "bears on the conclusion", and "enough to draw the conclusion"
    cannot be computed. One call means the model reads the argument once and
    the three judgments stay mutually consistent, at a third of the round trips
    of one call per dimension.

    The prompt requires sufficiency to be judged GRANTING the premises as true,
    which is what keeps it independent of acceptability.
    """
    prem_list = _as_list(premises)
    if not prem_list:
        raise ValueError("premises must be a non-empty list")

    numbered = "\n".join(f"- {p}" for p in prem_list)
    user = f"CONCLUSION:\n{claim}\n\nPREMISES:\n{numbered}"
    aid = argument_id or arg_id(claim)

    with log_call("logical_cogency", "ollama_cogency_judgment",
                  argument_id=aid, claim=claim, premises=prem_list,
                  model=model, host=host) as rec:
        raw = _chat_json(load_prompt(_JUDGMENT_PROMPT_FILE), user,
                         _CogencyJudgment.model_json_schema(), model, host,
                         temperature, num_predict=1024)
        rec["response"] = {"raw_content": raw}

        try:
            j = _CogencyJudgment.model_validate_json(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            # The raw text is the only way to tell the failure shapes apart:
            # truncation, wrong fields (prompt drifted from schema), or prose
            # wrapped around the JSON.
            logger.error("Cogency judgment did not match schema. Raw: %s",
                         raw[:600])
            raise RuntimeError(
                f"Cogency judgment returned unparseable output: {exc}") from exc

        out = {
            "acceptability": _score_1to3(j.acceptability),
            "acceptability_rationale": j.acceptability_rationale.strip(),
            "relevance": _score_1to3(j.relevance),
            "relevance_rationale": j.relevance_rationale.strip(),
            "sufficiency": _score_1to3(j.sufficiency),
            "sufficiency_rationale": j.sufficiency_rationale.strip(),
        }
        rec["response"]["judgment"] = out
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  Deterministic explanation — never an LLM call
# ══════════════════════════════════════════════════════════════════════════════

def _build_reasoning(prem_list, j, methods) -> str:
    """Plain-English readout, built from values already computed. Cannot itself
    fail, which matters most when the LLM call HAS failed — that is exactly
    when an explanation is needed.
    """
    if not prem_list:
        return ("No premises were given, so this is a bare assertion — there "
                "are no premises to judge for acceptability, relevance, or "
                "sufficiency.")

    if methods["judgment"] != "llm":
        return ("The cogency judgment failed for this turn, so none of the "
                "three dimensions was measured. The scores are absent, not "
                "low.")

    parts = []
    for label, score_key, rat_key in (
            ("Local acceptability", "acceptability", "acceptability_rationale"),
            ("Local relevance", "relevance", "relevance_rationale"),
            ("Local sufficiency", "sufficiency", "sufficiency_rationale")):
        score = j.get(score_key)
        rationale = (j.get(rat_key) or "no rationale given").rstrip(".")
        parts.append(f"{label} is {_band_word(score)} ({score}/3): "
                     f"{rationale}.")
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def score_logical_cogency(
    claim: str,
    premises,
    argument_id: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """
    Cogency of ONE turn from its mined claim + premises, as the three
    sub-dimensions of Wachsmuth et al. (2017), each on the paper's 1-3 scale.

      local_acceptability : premises worthy of being believed
      local_relevance     : premises support or attack the conclusion
      local_sufficiency   : premises enough to draw the conclusion

    All three come from ONE LLM call, each with a rationale that quotes the
    premise text driving the score. They are reported side by side and never
    averaged — Wachsmuth's cogency is a dimension humans annotate in its own
    right, not one computed from the three.

    `local_relevance_nli_diagnostic` and `relevance_per_premise` carry the
    DeBERTa 1 - P(neutral) values. These are NOT scores; see the module
    docstring on why NLI does not discriminate relevance on non-deductive
    arguments.

    A score of None means that dimension was not measured, never that it scored
    low. `methods` records which path each part took.
    """
    if not claim or not claim.strip():
        raise ValueError("claim must be a non-empty string")

    claim = claim.strip()
    prem_list = _as_list(premises)
    aid = argument_id or arg_id(claim)

    with log_call("logical_cogency", "score", argument_id=aid,
                  claim=claim, premises=prem_list) as rec:

        methods = {"judgment": None, "nli_diagnostic": None}

        # ── NLI relevance diagnostic (not a score) ──────────────────────────
        if prem_list and _enabled(COGENCY_NLI_DIAGNOSTIC):
            try:
                nli = relevance_nli_diagnostic(claim, prem_list)
                methods["nli_diagnostic"] = "nli"
            except Exception as e:
                logger.warning("NLI relevance diagnostic failed (%s) — it is "
                               "a diagnostic only, so scoring continues", e)
                nli = {"max": None, "per_premise": []}
                methods["nli_diagnostic"] = "failed"
        else:
            nli = {"max": None, "per_premise": []}
            methods["nli_diagnostic"] = ("no_premises" if not prem_list
                                         else "disabled")

        # ── the three dimensions: one LLM call ──────────────────────────────
        j = {"acceptability": None, "acceptability_rationale": "",
             "relevance": None, "relevance_rationale": "",
             "sufficiency": None, "sufficiency_rationale": ""}
        if not prem_list:
            methods["judgment"] = "no_premises"
        else:
            try:
                j = judge_cogency_llm(claim, prem_list,
                                      model=model or OLLAMA_MODEL,
                                      argument_id=aid)
                methods["judgment"] = "llm"
            except Exception as e:
                logger.warning("cogency judgment failed (%s) — none of the "
                               "three dimensions is available for this turn", e)
                methods["judgment"] = "failed"

        result = {
            # ── the three sub-dimensions, 1-3 (Wachsmuth et al. 2017) ──────
            "local_acceptability": j["acceptability"],
            "local_relevance": j["relevance"],
            "local_sufficiency": j["sufficiency"],
            # ── rationales, each quoting the premise it turns on ───────────
            "acceptability_rationale": j["acceptability_rationale"],
            "relevance_rationale": j["relevance_rationale"],
            "sufficiency_rationale": j["sufficiency_rationale"],
            "reasoning": _build_reasoning(prem_list, j, methods),
            # ── diagnostics — NOT scores ──────────────────────────────────
            "local_relevance_nli_diagnostic": nli["max"],
            "relevance_per_premise": nli["per_premise"],
            "n_premises": len(prem_list),
            "methods": methods,
        }

        # Telemetry: same record, minus the bulky per-premise NLI breakdown.
        rec["response"] = {k: v for k, v in result.items()
                           if k != "relevance_per_premise"}

    logger.info("Cogency %r | acceptability=%s relevance=%s sufficiency=%s "
                "| %d premises | nli_relevance_diag=%s",
                (claim[:50] + "…") if len(claim) > 50 else claim,
                j["acceptability"], j["relevance"], j["sufficiency"],
                len(prem_list), nli["max"])
    return result
