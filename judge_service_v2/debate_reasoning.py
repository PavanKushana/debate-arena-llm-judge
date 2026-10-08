"""
debate_reasoning.py — final reasoning pass over a scored debate.

After every metric has been COMPUTED (by the small instruct model + NLI), this
runs ONE debate-level call on a reasoning model (default gpt-oss:20b) to:

  1. write a plain-English summary of how the debate went, per speaker, and
  2. flag any metric whose score clearly CONTRADICTS what the turns show.

Why a second, different model here:
  * The scoring calls use schema-constrained JSON output, which reasoning models
    like gpt-oss cannot satisfy (they return empty `content` under the `format`
    grammar). This call uses PLAIN chat — no schema — so gpt-oss works and its
    reasoning strength is used where it belongs: the human-readable synthesis.

The one hard rule: this model NEVER produces or changes a score. It explains the
existing numbers and, at most, says one looks wrong. The numbers stay owned by
the scoring model; the discrepancy flags are the reasoning model's OPINION, and
the flagging in code is deterministic on top of that opinion — a triage pointer
for human inspection, not a verdict.
"""
from __future__ import annotations

import json
import logging
import os
import re

import requests

from .utils.judge_config import OLLAMA_MODEL, OLLAMA_HOST
from .utils.prompts import load_prompt
from .utils.telemetry import log_call, arg_id

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_FILE = "debate_reasoning_system.txt"

# A reasoning model, deliberately different from the scoring model. Plain chat,
# so gpt-oss's empty-content-under-`format` problem does not apply here.
REASONING_MODEL = os.getenv("REASONING_MODEL", "gpt-oss:20b")
REASONING_HOST = os.getenv("REASONING_HOST", OLLAMA_HOST)
REASONING_CONNECT_TIMEOUT = float(os.getenv("REASONING_CONNECT_TIMEOUT", "10"))
REASONING_READ_TIMEOUT = float(os.getenv("REASONING_READ_TIMEOUT", "300"))
REASONING_TEMPERATURE = float(os.getenv("REASONING_TEMPERATURE", "0.2"))


def _fmt(v):
    if v is None:
        return "null (not computed)"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _scores_digest(result: dict) -> str:
    """Compact text of the computed per-speaker scores the reasoning model must
    concur with (or flag). Null means the metric could not be computed."""
    lines = []
    for sp, d in (result.get("by_speaker") or {}).items():
        lines.append(f"Speaker {sp} (stance {d.get('stance')}):")
        st = d.get("stance_support") or {}
        lines.append(f"  stance_support: mean {_fmt(st.get('mean_stance_support'))}, "
                     f"position_held={st.get('position_held')}")
        cg = d.get("logical_cogency") or {}
        lines.append(f"  logical_cogency: acceptability {_fmt(cg.get('mean_local_acceptability'))}, "
                     f"relevance {_fmt(cg.get('mean_local_relevance'))}, "
                     f"sufficiency {_fmt(cg.get('mean_local_sufficiency'))} (each 1-3)")
        cs = d.get("consistency") or {}
        lines.append(f"  consistency: {_fmt(cs.get('mean_consistency') or cs.get('consistency'))}")
        rb = d.get("rebuttal") or {}
        lines.append(f"  rebuttal_effectiveness: {_fmt(rb.get('mean_effectiveness'))}, "
                     f"made {rb.get('rebuttals_made', 0)}")
        idn = d.get("information_density") or {}
        lines.append(f"  information_density: {_fmt(idn.get('information_density'))}")
        asv = d.get("argument_survival") or {}
        lines.append(f"  argument_survival: mean {_fmt(asv.get('argument_survival_score'))}, "
                     f"still_standing {asv.get('still_standing')}/{asv.get('arguments_introduced')}")
    return "\n".join(lines)


def _turns_digest(result: dict) -> str:
    pt = result.get("per_turn")
    if pt:
        return "\n".join(
            f"[turn {t.get('turn_no')}] {t.get('speaker')} ({t.get('stance','?')}): "
            f"{t.get('claim') or t.get('text','')}" for t in pt)
    return "(turn text not included in result)"


# ── parse the two headed sections from the model's plain-text reply ───────────
_DISCREP_LINE = re.compile(r"^\s*[-*]\s*(.+?)\s*:\s*(.+?)\s*$")


def _parse(raw: str) -> dict:
    """Split the reply into narrative + discrepancy list. Robust to the model
    adding stray text: we key off the two headings and read bullets."""
    narrative, disc_block = raw, ""
    m = re.search(r"###\s*Discrepan", raw, re.IGNORECASE)
    if m:
        narrative = raw[:m.start()]
        disc_block = raw[m.end():]
    narrative = re.sub(r"###\s*Summary\s*", "", narrative, flags=re.IGNORECASE).strip()

    discrepancies = []
    if disc_block and "none" not in disc_block.strip().lower()[:12]:
        for line in disc_block.splitlines():
            if not line.strip().startswith(("-", "*")):
                continue
            mm = _DISCREP_LINE.match(line)
            if mm:
                discrepancies.append({"metric": mm.group(1).strip(),
                                      "reason": mm.group(2).strip()})
    return {"narrative": narrative, "discrepancies": discrepancies}


def synthesize_reasoning(result: dict,
                         model: str | None = None,
                         host: str | None = None) -> dict:
    """
    One reasoning-model call over the fully-scored `result`. Returns
    {model, narrative, discrepancies:[{metric, reason}], has_mismatch, raw}.
    Never raises for a model/parse problem — reasoning is presentation, so on
    failure it returns an honest {error:...} rather than a fabricated summary.
    """
    model = model or REASONING_MODEL
    host = host or REASONING_HOST
    topic = result.get("topic", "")
    user = (f"TOPIC:\n{topic}\n\n"
            f"COMPUTED SCORES:\n{_scores_digest(result)}\n\n"
            f"DEBATE TURNS:\n{_turns_digest(result)}")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": load_prompt(_SYSTEM_PROMPT_FILE)},
            {"role": "user", "content": user},
        ],
        "stream": False,
        # NO "format" here — plain chat, so a reasoning model returns content.
        "options": {"temperature": REASONING_TEMPERATURE},
    }
    aid = arg_id(topic, prefix="debate")
    try:
        with log_call("debate_reasoning", "ollama_reasoning_synthesis",
                      argument_id=aid, topic=topic, model=model, host=host) as rec:
            r = requests.post(f"{host}/api/chat", json=payload,
                              timeout=(REASONING_CONNECT_TIMEOUT,
                                       REASONING_READ_TIMEOUT))
            r.raise_for_status()
            raw = (r.json().get("message") or {}).get("content", "") or ""
            rec["response"] = {"raw_content": raw}
            if not raw.strip():
                # e.g. a reasoning model still misbehaving on this host
                out = {"model": model, "error": "reasoning model returned empty "
                       "content (is it a plain instruct/reasoning model that "
                       "returns content? try a different REASONING_MODEL)"}
                rec["response"]["parsed"] = out
                return out
            parsed = _parse(raw)
            out = {"model": model,
                   "narrative": parsed["narrative"],
                   "discrepancies": parsed["discrepancies"],
                   "has_mismatch": bool(parsed["discrepancies"]),
                   "raw": raw}
            rec["response"]["parsed"] = {k: v for k, v in out.items() if k != "raw"}
            logger.info("Debate reasoning done | model=%s | discrepancies=%d",
                        model, len(parsed["discrepancies"]))
            return out
    except requests.RequestException as exc:
        logger.warning("reasoning synthesis failed: %s", exc)
        return {"model": model, "error": f"reasoning model request failed: {exc}"}
    except Exception as exc:  # presentation step — never sink the debate
        logger.warning("reasoning synthesis error: %s", exc)
        return {"model": model, "error": str(exc)}
