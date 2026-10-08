"""
es_client.py — thin client for Rosen's Debate Arguments API (Elasticsearch).

Implements exactly the contract in TEAM_LOGIC_HANDOFF: /semantic-search,
/granular-similarity, /topic-cluster, /health. Every call degrades gracefully
— if the Webis VPN is down or the server is unreachable, the function returns
None (never raises), so the scoring layer can cleanly fall back to the Part 1
LLM/extrapolation path instead of crashing the judge.

Scores from the server are raw (1.0–2.0); callers normalise with
judge_config.es_norm().

Every request, response, and failure is recorded via utils.telemetry.log_call
into logs/judge_api_telemetry.jsonl — this was previously invisible (the only
trace was a single stray `print`, and failures weren't logged at all).
"""

import requests

from .judge_config import ES_API, ES_MIN_MATCH
from .telemetry import log_call

_TIMEOUT = 8


def es_available() -> bool:
    """True if Rosen's API answers /health (i.e. VPN connected, server up)."""
    try:
        with log_call("es_client", "health", request={}) as rec:
            r = requests.get(f"{ES_API}/health", timeout=4)
            ok = bool(r.ok and r.json().get("status") == "ok")
            rec["response"] = {"status_code": r.status_code, "ok": ok}
            return ok
    except Exception:
        return False


def semantic_search(argument: str, top_n: int = 5,
                    quality_only: bool = False,
                    min_score: float = ES_MIN_MATCH,
                    argument_id: str = None):
    """
    POST /semantic-search — vector search over the CMV index.
    Returns the list of similar arguments (each with similarity_score,
    1.0–2.0) or None if the server is unreachable.
    """
    payload = {"argument": argument, "top_n": top_n,
               "quality_only": quality_only, "min_score": min_score}
    try:
        with log_call("delta_votes", "es_semantic_search",
                      argument_id=argument_id, **payload) as rec:
            r = requests.post(f"{ES_API}/semantic-search", json=payload,
                              timeout=_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            rec["response"] = data
            return data.get("similar_arguments", [])
    except Exception:
        return None


def granular_similarity(argument: str, claim: str = "", premises=None,
                        top_n: int = 5, min_score: float = ES_MIN_MATCH,
                        argument_id: str = None):
    """
    POST /granular-similarity — claim-level and premise-level match scores.
    Returns the `results` list (each with overall_score, claim_score,
    best_premise_score, cluster_member_count, ...) or None if unreachable.
    """
    payload = {"argument": argument, "top_n": top_n, "min_score": min_score}
    if claim:
        payload["claim"] = claim
    if premises:
        payload["premises"] = premises
    try:
        with log_call("delta_votes", "es_granular_similarity",
                      argument_id=argument_id, **payload) as rec:
            r = requests.post(f"{ES_API}/granular-similarity", json=payload,
                              timeout=_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            rec["response"] = data
            return data.get("results", [])
    except Exception:
        return None


def topic_cluster(topic: str, quality_aggregation: str = "max",
                  argument_id: str = None):
    """
    GET /topic-cluster — the broader reference set for a topic. Returns the
    full response dict (with argument_count, arguments[...]) or None.
    """
    params = {"topic": topic, "quality_aggregation": quality_aggregation}
    try:
        with log_call("delta_votes", "es_topic_cluster",
                      argument_id=argument_id, **params) as rec:
            r = requests.get(f"{ES_API}/topic-cluster", params=params,
                             timeout=_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            rec["response"] = data
            return data
    except Exception:
        return None
