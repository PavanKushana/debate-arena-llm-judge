"""
argument_miner.py — claim & premise extraction for one debate turn.

Belongs in: judge_service_v2/utils/argument_miner.py

Uses a local Ollama model with structured (JSON-schema-constrained) output, so
the response parses straight into ArgumentStructure — no manual JSON cleanup.

IMPORTANT — the system prompt and this schema must stay in step.
`format` is a GRAMMAR constraint: the decoder physically cannot emit tokens
outside the schema. If the prompt asks for fields the schema forbids (or vice
versa) the model fights the grammar, burns output tokens, and requests time
out. Any change to prompts/argument_miner_system.txt must be mirrored here.

MODEL CHOICE
Reasoning models (gpt-oss, deepseek-r1, qwen3 in thinking mode) emit a
chain-of-thought block before their answer, which the grammar constraint
forbids — the two fight and generation crawls. Prefer a plain instruct model
for mining (qwen2.5:7b, llama3.1:8b). If you must use a reasoning model, pass
think=False to turn its reasoning off (needs Ollama >= 0.9).

Prerequisites:
    pip install requests pydantic
    ollama pull qwen2.5:7b
    # Ollama >= 0.5 is required for schema-constrained `format`.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import requests
from pydantic import BaseModel, Field, ValidationError

from .judge_config import OLLAMA_MODEL, OLLAMA_HOST
from .prompts import load_prompt
from .telemetry import log_call

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_FILE = "argument_miner_system.txt"

# ── Tunables (env-overridable) ───────────────────────────────────────────────
# Connect timeout stays short — an unreachable host should fail fast. The read
# timeout must cover a COLD model load, which for a large model on a shared
# server is minutes, not seconds. 60s was the cause of the read timeouts.
MINER_CONNECT_TIMEOUT = float(os.getenv("MINER_CONNECT_TIMEOUT", "10"))
MINER_READ_TIMEOUT = float(os.getenv("MINER_READ_TIMEOUT", "300"))
# One retry covers the common case: the first call pays the model load, the
# retry runs against a warm model.
MINER_RETRIES = int(os.getenv("MINER_RETRIES", "1"))
# Hard cap on generated tokens. A claim plus a handful of premises is a few
# hundred tokens; this stops a confused model generating until the timeout.
MINER_NUM_PREDICT = int(os.getenv("MINER_NUM_PREDICT", "512"))
# How long Ollama keeps the model resident after a call. Long enough that the
# turns of one debate all hit a warm model.
MINER_KEEP_ALIVE = os.getenv("MINER_KEEP_ALIVE", "30m")


class ArgumentStructure(BaseModel):
    """The mined structure of a single argument.

    Mirrors prompts/argument_miner_system.txt exactly — see the module
    docstring on why the two must not drift apart.
    """

    claim: str = Field(
        default="",
        description="The single main conclusion the argument is trying to "
                    "establish. Empty when the turn asserts no arguable "
                    "conclusion.",
    )
    premises: list[str] = Field(
        default_factory=list,
        description="The reasons or evidence offered in support of the claim.",
    )


def extract_claim_and_premises(
    argument: str,
    model: str = OLLAMA_MODEL,
    host: str = OLLAMA_HOST,
    temperature: float = 0.0,
    argument_id: Optional[str] = None,
    think: Optional[bool] = None,
) -> ArgumentStructure:
    """
    Mine one turn into {claim, premises}.

    Parameters
    ----------
    argument    : the turn text.
    model, host : Ollama model + host (defaults from judge_config).
    temperature : 0.0 for deterministic, reproducible mining.
    argument_id : optional stable id for telemetry correlation.
    think       : pass False to disable reasoning on a thinking model
                  (gpt-oss, deepseek-r1, qwen3). Omit for plain instruct
                  models — older Ollama builds reject the field.

    Raises
    ------
    RuntimeError on transport failure, a non-2xx response, or output that does
    not match the schema.
    """
    if not argument or not argument.strip():
        raise ValueError("argument must be a non-empty string")

    system_prompt = load_prompt(_SYSTEM_PROMPT_FILE)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": argument.strip()},
        ],
        "stream": False,
        "format": ArgumentStructure.model_json_schema(),
        "keep_alive": MINER_KEEP_ALIVE,
        "options": {
            "temperature": temperature,
            "num_predict": MINER_NUM_PREDICT,
        },
    }
    if think is not None:
        payload["think"] = think

    timeout = (MINER_CONNECT_TIMEOUT, MINER_READ_TIMEOUT)

    # One telemetry record per call: argument in, raw LLM content out, PLUS the
    # parsed claim/premises. That artifact is what feeds NLI downstream, so
    # having input → raw json → parsed structure on one correlated line is what
    # makes a bad mining result diagnosable after the fact rather than only
    # inferable from odd downstream scores. The system prompt is deliberately
    # NOT logged — it lives in prompts/.
    with log_call("argument_miner", "ollama_mine_claim_premises",
                  argument_id=argument_id, argument=argument,
                  model=model, host=host) as rec:

        raw = None
        last_exc: Optional[Exception] = None

        for attempt in range(MINER_RETRIES + 1):
            try:
                response = requests.post(f"{host}/api/chat", json=payload,
                                         timeout=timeout)
                response.raise_for_status()
                raw = response.json()["message"]["content"]
                break
            except requests.Timeout as exc:
                last_exc = exc
                if attempt < MINER_RETRIES:
                    logger.warning(
                        "Mining timed out after %ss (attempt %d/%d) — the "
                        "model may have been cold; retrying against a warm "
                        "model", MINER_READ_TIMEOUT, attempt + 1,
                        MINER_RETRIES + 1)
                    time.sleep(1)
                    continue
            except requests.RequestException as exc:
                last_exc = exc
                break
            except (KeyError, IndexError, ValueError) as exc:
                raise RuntimeError(
                    "Invalid response received from Ollama.") from exc

        if raw is None:
            raise RuntimeError(
                f"Ollama request failed after {MINER_RETRIES + 1} attempt(s): "
                f"{last_exc}") from last_exc

        # Set BEFORE parsing so a parse failure still logs the raw text.
        rec["response"] = {"raw_content": raw}

        try:
            result = ArgumentStructure.model_validate_json(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error("Model output did not match schema: %s", raw)
            raise RuntimeError(
                "Model returned output that did not match the schema. Check "
                "that prompts/argument_miner_system.txt still matches "
                "ArgumentStructure — a prompt asking for fields the schema "
                "forbids is the usual cause."
            ) from exc

        result.claim = result.claim.strip()
        result.premises = [p.strip() for p in result.premises if p and p.strip()]

        if not result.claim:
            logger.info("No claim mined from argument %r — turn asserts no "
                        "arguable conclusion, or mining failed to find one",
                        argument[:80])

        rec["response"]["claim"] = result.claim
        rec["response"]["premises"] = result.premises
        return result


def warm_up(model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST) -> bool:
    """Load the model into memory ahead of real traffic.

    Call once at service start (next to the NLI warmup in main.py). Without it,
    the first mining request of the day pays the full model load inside its own
    timeout, which is the single most common cause of a mining timeout.

    Returns True if the model is resident, False otherwise — never raises, so
    a failed warmup does not stop the service from starting.
    """
    try:
        r = requests.post(f"{host}/api/chat",
                          json={"model": model,
                                "messages": [{"role": "user", "content": "ok"}],
                                "stream": False,
                                "keep_alive": MINER_KEEP_ALIVE,
                                "options": {"num_predict": 1}},
                          timeout=(MINER_CONNECT_TIMEOUT, MINER_READ_TIMEOUT))
        r.raise_for_status()
        logger.info("Ollama model %s warmed up on %s", model, host)
        return True
    except Exception as e:
        logger.warning("Ollama warmup failed for %s on %s: %s", model, host, e)
        return False