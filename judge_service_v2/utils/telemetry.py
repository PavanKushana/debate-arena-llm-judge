"""
utils/telemetry.py — structured, replayable logging of every external or
model call made while scoring a metric: LLM calls (argument mining, ArgQ
fallback, rebuttal quality), Elasticsearch requests, IBM ArgQ-30k corpus
neighbour lookups, and NLI CrossEncoder calls.

Writes ONE JSON object per line to the dedicated telemetry file configured
in utils/logging_config.py — never into the human-readable judge_api.log.
That separation is what makes this useful for more than debugging: the
file is a flat, append-only record of every metric's exact inputs and
outputs, so it can be replayed or joined against annotated ground truth
without re-running the pipeline.

Usage
-----
    with log_call("argquality", "ollama_fallback", argument_id=arg_id,
                  request={"prompt": prompt, "model": model}) as rec:
        r = requests.post(...)
        r.raise_for_status()
        rec["response"] = r.json()

On success, ONE record is written with status="ok" and whatever you set on
`rec["response"]`. On any exception inside the block, ONE record is written
with status="error" (exception type + message) and the exception is
RE-RAISED unchanged — every caller's existing try/except and fallback
behaviour keeps working exactly as before; this only observes, it never
changes control flow.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

_telemetry = logging.getLogger("judge_api.telemetry")
_fallback_logger = logging.getLogger(__name__)

# A single string longer than this is truncated (with a marker showing how
# much was cut) — protects against runaway payloads (e.g. a long ES hit
# list, or an LLM echoing back its whole prompt) without ever silently
# dropping a field.
_MAX_STR = 4000
# A numeric list longer than this is recorded as {"_type": "vector", "len":
# N} instead of dumped verbatim. A 384-dim embedding has zero diagnostic
# value as literal floats in a log line and would dominate the file size on
# every single call — this is the one piece of "everything" that's
# deliberately NOT stored verbatim, and it's the only one.
_MAX_NUMERIC_LIST = 32
_MAX_LIST_ITEMS = 200


def _safe(value: Any) -> Any:
    """Recursively make `value` JSON-serialisable and bounded in size."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > _MAX_STR:
            return value[:_MAX_STR] + f"...[truncated, {len(value) - _MAX_STR} more chars]"
        return value
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(x, (int, float)) for x in value) \
                and len(value) > _MAX_NUMERIC_LIST:
            return {"_type": "vector", "len": len(value)}
        items = [_safe(x) for x in value[:_MAX_LIST_ITEMS]]
        if len(value) > _MAX_LIST_ITEMS:
            items.append(f"...[{len(value) - _MAX_LIST_ITEMS} more items truncated]")
        return items
    # numpy scalars, custom objects, etc.
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def arg_id(text: str, prefix: str = "arg") -> str | None:
    """
    Derive a stable, content-based id for an argument so its records join
    across metrics and endpoints WITHOUT any change to the API request
    schema: the same argument text always yields the same id. Returns None
    for empty input (telemetry then simply logs with argument_id=None).

        arg_id("Compulsory voting raises turnout")  →  "arg:6f1c0a3e9b2d"

    Use a richer id (e.g. "{debate_id}:{speaker}:{turn}") when the caller
    genuinely has that context; this is the zero-friction fallback that still
    makes every record correlatable.
    """
    if not text or not text.strip():
        return None
    return f"{prefix}:{hashlib.md5(text.strip().encode()).hexdigest()[:12]}"


@contextmanager
def log_call(metric: str, event: str, argument_id: str | None = None,
             **request_fields) -> Iterator[dict]:
    """
    Wrap one external/model call and guarantee exactly one structured record
    is written to the telemetry file — on success OR failure.

    Parameters
    ----------
    metric      : which scorer this belongs to, e.g. "argquality", "nli",
                  "delta_votes", "rebuttal", "argument_miner".
    event       : which specific call within that metric, e.g.
                  "corpus_neighbours", "ollama_fallback", "es_granular_similarity".
    argument_id : optional stable id (e.g. "{debate_id}:{speaker}:{turn}:{i}")
                  so records for the same argument can be joined across
                  metrics later. None is fine — every call still gets logged,
                  it just isn't correlated to a specific argument yet.
    **request_fields : whatever describes the call being made (prompt,
                  payload, query text, k, model name, ...). Captured as-is.

    Yields
    ------
    A mutable `rec` dict. Set rec["response"] = ... inside the block before
    it exits normally — that becomes the logged response payload.
    """
    rec: dict = {
        "ts":          round(time.time(), 3),
        "metric":      metric,
        "event":       event,
        "argument_id": argument_id,
        "request":     _safe(request_fields),
    }
    t0 = time.perf_counter()
    try:
        yield rec
        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "error"
        rec["error"]  = {"type": type(e).__name__, "message": str(e)}
        raise
    finally:
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if "response" in rec:
            rec["response"] = _safe(rec["response"])
        try:
            _telemetry.info(json.dumps(rec, default=str))
        except Exception:
            # Telemetry must never be able to break scoring. If a record
            # somehow can't be serialised, drop it and say so in the
            # narrative log rather than raising from inside a `finally`.
            _fallback_logger.warning(
                "Telemetry write failed for %s/%s (argument_id=%s)",
                metric, event, argument_id,
            )


def log_event(metric: str, event: str, argument_id: str | None = None,
              **fields) -> None:
    """
    Log a single fact with no request/response shape and no timing — e.g. a
    cache hit, a gate pass/fail decision, or "skipped: X unavailable". Use
    log_call() instead whenever there's an actual external/model call to
    time and capture.
    """
    rec = {
        "ts":          round(time.time(), 3),
        "metric":      metric,
        "event":       event,
        "argument_id": argument_id,
        "status":      "ok",
        **{k: _safe(v) for k, v in fields.items()},
    }
    try:
        _telemetry.info(json.dumps(rec, default=str))
    except Exception:
        _fallback_logger.warning(
            "Telemetry write failed for %s/%s (argument_id=%s)",
            metric, event, argument_id,
        )
