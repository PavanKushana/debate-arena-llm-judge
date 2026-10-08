"""
topic_hypotheses.py — generate PRO and CON stance hypotheses for a topic.

Belongs in: judge_api/utils/topic_hypotheses.py  (next to argument_miner.py)

Same shape as argument_miner.py, but the input is the debate TOPIC (not one
argument) and the output is the two stance hypotheses a debate is fought over:

    pro_hypothesis : the claim the PRO/affirmative side defends
    con_hypothesis : the claim the CON/negative side defends (its direct opposite)

These feed stance_support_score.py: score each turn against ITS speaker's own
hypothesis and read entailment as support. A real, fluent CON hypothesis
("Money cannot buy happiness") is what lets you read entailment for CON turns
instead of relying on the noisier contradiction signal against the PRO thesis.

Uses a local Ollama model with structured (JSON-schema-constrained) output, so
the response parses straight into TopicHypotheses — no manual JSON cleanup.

A topic is stable across a whole debate, so results are cached per
(topic, model): you pay for one LLM call, then reuse it for every turn.

Logging uses the project's two existing log files (see utils/logging_config.py):
  * judge_api.log            — a concise human-readable line per generation /
                               cache hit (via the module logger).
  * judge_api_telemetry.jsonl — the structured record of the LLM call (topic +
                               model in, raw content + parsed pro/con out) via
                               utils/telemetry.log_call. The system prompt is
                               never logged (it lives in prompts/).

Prerequisites:
    pip install ollama pydantic requests
    ollama pull qwen2.5:7b        # or whatever OLLAMA_MODEL points at
    # Ollama >= 0.5 is required for schema-constrained `format`.
"""
from __future__ import annotations

import json
import logging
import os

import requests
from pydantic import BaseModel, Field, ValidationError

from .judge_config import OLLAMA_MODEL, OLLAMA_HOST
from .prompts import load_prompt
from .telemetry import log_call, log_event, arg_id

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_FILE = "topic_hypotheses_system.txt"

# Connect timeout stays short — an unreachable host should fail fast. The read
# timeout must cover a COLD model load, which for a large model on a shared
# server is minutes, not seconds. A hardcoded 60s was the cause of the read
# timeouts on stance-support (this call feeds it), so it matches the other
# LLM callers now: (connect, read).
HYP_CONNECT_TIMEOUT = float(os.getenv("HYP_CONNECT_TIMEOUT", "10"))
HYP_READ_TIMEOUT = float(os.getenv("HYP_READ_TIMEOUT", "300"))


def _clean(s: str) -> str:
    """Normalise one hypothesis into a clean declarative sentence."""
    s = (s or "").strip().strip('"').strip("'").strip()
    if s and s[-1] not in ".!?":
        s += "."
    return s


class TopicHypotheses(BaseModel):
    """The two stance hypotheses a debate topic is argued over."""

    pro_hypothesis: str = Field(
        min_length=3,
        description="The core claim the PRO/affirmative side defends — one "
                    "concise declarative sentence that agrees with the topic.")
    con_hypothesis: str = Field(
        min_length=3,
        description="The core claim the CON/negative side defends — one concise "
                    "declarative sentence stating the opposite position.")

    def for_stance(self, stance: str) -> str:
        """The hypothesis this side defends."""
        s = stance.upper()
        if s == "PRO":
            return self.pro_hypothesis
        if s == "CON":
            return self.con_hypothesis
        raise ValueError(f"stance must be PRO or CON, got {stance!r}")

    def opposing(self, stance: str) -> str:
        """The hypothesis the OTHER side defends — used for the relevance
        check in stance_support_score, not for scoring."""
        return self.for_stance("CON" if stance.upper() == "PRO" else "PRO")

# Topic → hypotheses is stable, so cache it for the life of the process.
_HYP_CACHE: dict[tuple, TopicHypotheses] = {}


def generate_topic_hypotheses(
    topic: str,
    model: str = OLLAMA_MODEL,
    host: str = OLLAMA_HOST,
    temperature: float = 0.0,
    use_cache: bool = True,
) -> TopicHypotheses:
    """
    Ask the LLM for the PRO and CON stance hypotheses of `topic`.

    Parameters
    ----------
    topic       : the debate topic (e.g. "Money can buy happiness").
    model, host : Ollama model + host (default from judge_config).
    temperature : 0.0 for deterministic, reusable hypotheses.
    use_cache   : reuse a previously generated result for the same
                  (topic, model). A debate reuses one topic across every turn,
                  so this avoids one LLM call per turn.

    Returns
    -------
    TopicHypotheses(pro_hypothesis=..., con_hypothesis=...)
    """
    if not topic or not topic.strip():
        raise ValueError("topic must be a non-empty string")

    topic_norm = topic.strip()
    tid = arg_id(topic_norm, prefix="topic")
    cache_key = (topic_norm.lower(), model, host)

    if use_cache and cache_key in _HYP_CACHE:
        # Narrative (judge_api.log): quiet — a cache hit is routine, so debug.
        logger.debug("Stance hypotheses cache hit for topic %r", topic_norm)
        # Telemetry (judge_api_telemetry.jsonl): a lean fact, no LLM call.
        log_event("stance_hypothesis", "cache_hit",
                  argument_id=tid, topic=topic_norm, model=model)
        return _HYP_CACHE[cache_key]

    system_prompt = load_prompt(_SYSTEM_PROMPT_FILE)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": topic_norm},
        ],
        "stream": False,
        "format": TopicHypotheses.model_json_schema(),
        "options": {"temperature": temperature},
    }

    # Two log files, one generation:
    #  - telemetry (log_call): structured topic-in / raw+parsed-out record.
    #  - narrative (logger.info): one concise human-readable line, below.
    # The system prompt is intentionally NOT logged (it lives in prompts/).
    try:
        with log_call("stance_hypothesis", "ollama_generate_hypotheses",
                      argument_id=tid, topic=topic_norm,
                      model=model, host=host) as rec:
            response = requests.post(f"{host}/api/chat", json=payload,
                                     timeout=(HYP_CONNECT_TIMEOUT,
                                              HYP_READ_TIMEOUT))
            response.raise_for_status()

            raw = response.json()["message"]["content"]
            # Set BEFORE parsing so a parse failure still logs the raw text.
            rec["response"] = {"raw_content": raw}

            parsed = TopicHypotheses.model_validate_json(raw)
            result = TopicHypotheses(
                pro_hypothesis=_clean(parsed.pro_hypothesis),
                con_hypothesis=_clean(parsed.con_hypothesis),
            )
            if result.pro_hypothesis.lower() == result.con_hypothesis.lower():
                raise ValueError(
                    f"model returned identical PRO and CON hypotheses for "
                    f"topic {topic_norm!r}: {result.pro_hypothesis!r}")

            rec["response"]["pro_hypothesis"] = result.pro_hypothesis
            rec["response"]["con_hypothesis"] = result.con_hypothesis

            if use_cache:
                _HYP_CACHE[cache_key] = result

            logger.info("Generated stance hypotheses for topic %r | "
                        "PRO: %r | CON: %r",
                        topic_norm, result.pro_hypothesis, result.con_hypothesis)
            return result

    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    except (KeyError, IndexError) as exc:
        raise RuntimeError("Invalid response received from Ollama.") from exc
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        logger.error("Model output did not match schema: %s", raw)
        raise RuntimeError(
            "Model returned output that did not match the schema"
        ) from exc


def stance_hypothesis_map(topic: str, **kwargs) -> dict:
    """Convenience: return {"PRO": pro_hypothesis, "CON": con_hypothesis} —
    exactly the shape stance_support_score.score_stance_support_batch expects
    as its `hypotheses=` argument. Accepts the same kwargs as
    generate_topic_hypotheses."""
    h = generate_topic_hypotheses(topic, **kwargs)
    return {"PRO": h.pro_hypothesis, "CON": h.con_hypothesis}
