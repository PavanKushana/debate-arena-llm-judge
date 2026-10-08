"""
judge_config.py — central configuration for judge_service_v2.

Only the settings the CURRENT metrics actually use live here. (Earlier v1
settings — ArgQuality corpus weights, emotion/composure, engagement, the
final-speaker composite weights, delivery/tie-break, token budget — were
removed; nothing in the current code imported them.)

All values are environment-overridable; a .env in the package parent is loaded
if python-dotenv is installed (shell env still wins).
"""

import os
from pathlib import Path

# Load .env from the package parent (override=False → shell env wins).
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass   # python-dotenv not installed — silently fall back to system env vars


# ── LLM (Ollama) ──────────────────────────────────────────────────────────────
# Used by: argument mining, topic hypotheses, cogency (premise/warrant/defects),
# rebuttal attack-type + sentence confirm, information-density LLM path.
OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")   # local Ollama default
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


# ── Elasticsearch reference retrieval (Delta / Votes) ─────────────────────────
# Rosen's scores are cosine + 1.0, i.e. in [1.0, 2.0]. es_norm() maps them back
# to an ABSOLUTE 0-1 cosine scale. Used by es_client + delta_votes_score.
ES_API       = os.getenv("ES_API", "")   # optional; empty = Delta/Votes report "not applicable"
ES_MIN_MATCH = float(os.getenv("ES_MIN_MATCH", "1.5"))   # raw scale (1.0–2.0)


def es_norm(raw_score: float) -> float:
    """Map an ES raw score (1.0–2.0) to an ABSOLUTE 0.0–1.0 cosine similarity."""
    return round(min(max(raw_score - 1.0, 0.0), 1.0), 3)


# ── Delta (rare-event bonus signal) ───────────────────────────────────────────
# An argument earns a delta iff it matches a delta-flagged reference argument
# with ABSOLUTE similarity >= DELTA_SIM_THRESHOLD (cosine scale; raw ES >= 1.8).
DELTA_SIM_THRESHOLD = float(os.getenv("DELTA_SIM_THRESHOLD", "0.8"))


# ── Rebuttal NLI-contradiction thresholds ─────────────────────────────────────
# Used by the rebuttal metric to decide claim- vs premise-level attacks.
REBUTTAL_CONTRA_THRESHOLD = float(os.getenv("REBUTTAL_CONTRA_THRESHOLD", "0.55"))  # rebuttal vs claim
UNDERCUT_CONTRA_THRESHOLD = float(os.getenv("UNDERCUT_CONTRA_THRESHOLD", "0.45"))  # rebuttal vs premise
