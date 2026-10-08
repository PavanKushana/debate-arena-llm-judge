"""
api.py — FastAPI service exposing every metric in judge_service_v2.

Run from the directory that CONTAINS judge_service_v2/:

    uvicorn judge_service_v2.api:app --host 0.0.0.0 --port 8500 --reload

Then open http://localhost:8500/ for the tester UI.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .utils.logging_config import setup_logging
from .turn_level_scoring.stance_support_score import (
    score_stance_support, score_stance_support_batch,
    aggregate_stance_by_speaker)
from .turn_level_scoring.logical_cogency_score import score_logical_cogency
from .turn_level_scoring.consistency_score import score_consistency
from .turn_level_scoring.delta_votes_score import score_delta_votes
from .debate_level_scoring.rebuttal_effectiveness_dl import score_rebuttal_debate
from .debate_level_scoring.information_density_dl import (
    score_information_density)
from .debate_level_scoring.argument_survival_dl import score_argument_survival
from .debate_level_scoring.llm_debate_winner import determine_llm_debate_winner
from .utils.topic_hypotheses import generate_topic_hypotheses
from .utils.argument_miner import extract_claim_and_premises
from .orchestrator import score_debate

setup_logging()
logger = logging.getLogger("judge_api")


def _json_safe(obj: Any) -> Any:
    """numpy / NaN sanitiser so FastAPI can always encode a response."""
    try:
        import numpy as np
    except ImportError:
        np = None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if np is not None:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            f = float(obj)
            return None if (math.isnan(f) or math.isinf(f)) else f
        if isinstance(obj, np.ndarray):
            return _json_safe(obj.tolist())
    if hasattr(obj, "model_dump"):
        return _json_safe(obj.model_dump())
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj


app = FastAPI(title="Debate Judge — metric tester API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_TESTER = Path(__file__).resolve().parent / "tester.html"


def _ok(payload: Any) -> JSONResponse:
    return JSONResponse(content=_json_safe(payload))


def _fail(e: Exception, where: str) -> HTTPException:
    logger.exception("%s failed", where)
    return HTTPException(status_code=400, detail=f"{where} failed: {e}")


# ── request models ───────────────────────────────────────────────────────────

class StanceRequest(BaseModel):
    topic: str = Field(..., description="Debate topic, in any form")
    turn: str = Field(..., description="The speaker's turn text")
    stance: str = Field("PRO", description="PRO or CON")
    claim: Optional[str] = Field(
        None, description="Mined claim; scored instead of the whole turn")
    hypothesis: Optional[str] = Field(
        None, description="The claim THIS speaker's side defends, one sentence. "
                          "Generated from the topic when omitted. Never a pair — "
                          "stance already says which side.")
    model: Optional[str] = Field(None, description="Ollama model")


class StanceBatchRequest(BaseModel):
    topic: str
    turns: List[dict] = Field(
        ..., description='[{"turn","stance","speaker"?,"turn_no"?,"claim"?}]')
    pro_hypothesis: Optional[str] = None
    con_hypothesis: Optional[str] = None
    model: Optional[str] = None


class CogencyRequest(BaseModel):
    claim: str = Field(..., description="The turn's mined claim")
    premises: List[str] = Field(default_factory=list)
    model: Optional[str] = Field(None, description="Ollama model")


class ConsistencyRequest(BaseModel):
    claim: str
    prior_claims: List[Any] = Field(
        default_factory=list,
        description='Strings, or {"claim","turn_no"} objects')


class DeltaVotesRequest(BaseModel):
    es_results: List[dict] = Field(default_factory=list)


class InfoDensityRequest(BaseModel):
    turns: List[str] = Field(..., description="One speaker's turns, in order")
    topic: Optional[str] = None
    speaker: Optional[str] = None
    model: Optional[str] = None


class RebuttalTurn(BaseModel):
    speaker: str
    turn_no: Optional[int] = None
    text: Optional[str] = ""
    claim: Optional[str] = ""
    premises: Optional[List[str]] = None


class RebuttalRequest(BaseModel):
    turns: List[RebuttalTurn]
    use_llm: bool = Field(True, description="False runs the NLI screen alone")
    screen_min: Optional[float] = Field(None, description="NLI screen (default 0.40)")
    top_k: Optional[int] = Field(None, description="Candidates per sentence (default 2)")
    exclude_openers: Optional[bool] = None
    model: Optional[str] = None


class SurvivalRequest(BaseModel):
    turns: List[RebuttalTurn]
    rebuttal_result: Optional[dict] = Field(
        None, description="Output of /score/rebuttal-debate over the SAME turns. "
                          "Omit and it is computed here first.")
    use_llm: bool = True
    model: Optional[str] = None


class TopicRequest(BaseModel):
    topic: str
    use_cache: bool = True
    model: Optional[str] = None


class MineRequest(BaseModel):
    argument: str
    model: Optional[str] = None


class DebateTurnIn(BaseModel):
    speaker: str
    turn_no: Optional[int] = None
    text: str = ""
    stance: Optional[str] = None
    claim: Optional[str] = None
    premises: Optional[List[str]] = None


class LLMWinnerRequest(BaseModel):
    topic: str
    turns: List[DebateTurnIn]
    model: Optional[str] = Field(
        None, description="Ollama model for the one-shot LLM judge call; e.g. gpt-oss:20b")
    temperature: float = Field(0.0, ge=0.0, le=1.0)


class DebateRequest(BaseModel):
    topic: str
    turns: List[DebateTurnIn]
    stance_by_speaker: Optional[Dict[str, str]] = Field(
        None, description='e.g. {"Alice":"PRO","Bob":"CON"}')
    model: Optional[str] = Field(None, description="Ollama model for all LLM calls")
    use_llm_rebuttal: bool = True
    exclude_openers: bool = True
    include_detail: bool = True
    write_report: bool = True
    es_results_by_turn: Optional[Dict[str, List[dict]]] = Field(
        None, description='{"1":[es_result,...]} for Delta/Votes')


# ── health + tester ──────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "judge-api"}


@app.get("/health/ollama")
def health_ollama():
    """Ollama reachability and the models it currently has loaded."""
    from .utils.judge_config import OLLAMA_HOST
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=4)
        r.raise_for_status()
        return {"status": "ok", "reachable": True, "host": OLLAMA_HOST,
                "models": [m.get("name") for m in r.json().get("models", [])]}
    except Exception as e:
        return {"status": "down", "reachable": False,
                "host": OLLAMA_HOST, "error": str(e)}


@app.get("/health/all")
def health_all():
    return {"judge_api": {"status": "ok"}, "ollama": health_ollama()}


@app.get("/")
def tester():
    if _TESTER.exists():
        return FileResponse(str(_TESTER))
    return JSONResponse({"detail": "tester.html not found next to api.py"},
                        status_code=404)


# ── turn level ───────────────────────────────────────────────────────────────

@app.post("/score/stance-support")
def api_stance(req: StanceRequest):
    """Did this turn support the side its speaker was assigned?"""
    try:
        return _ok(score_stance_support(
            topic=req.topic, turn=req.turn, stance=req.stance,
            claim=req.claim, hypothesis=req.hypothesis, model=req.model))
    except Exception as e:
        raise _fail(e, "stance-support")


@app.post("/score/stance-support-batch")
def api_stance_batch(req: StanceBatchRequest):
    """Score every turn, then roll up per speaker: did each hold their position?
    weak_turns names the turns where they did not back their own side."""
    try:
        hyps = {k: v for k, v in (("PRO", req.pro_hypothesis),
                                  ("CON", req.con_hypothesis)) if v}
        rows = score_stance_support_batch(
            topic=req.topic, turns=req.turns,
            hypotheses=hyps or None, model=req.model)
        return _ok({"turns": rows,
                    "by_speaker": aggregate_stance_by_speaker(rows)})
    except Exception as e:
        raise _fail(e, "stance-support-batch")


@app.post("/score/logical-cogency")
def api_cogency(req: CogencyRequest):
    """Local acceptability, relevance and sufficiency, each 1-3."""
    try:
        return _ok(score_logical_cogency(
            claim=req.claim, premises=req.premises, model=req.model))
    except Exception as e:
        raise _fail(e, "logical-cogency")


@app.post("/score/consistency")
def api_consistency(req: ConsistencyRequest):
    """Does this claim contradict the speaker's earlier claims?"""
    try:
        return _ok(score_consistency(claim=req.claim,
                                     prior_claims=req.prior_claims))
    except Exception as e:
        raise _fail(e, "consistency")


@app.post("/score/delta-votes")
def api_delta_votes(req: DeltaVotesRequest):
    try:
        return _ok(score_delta_votes(req.es_results))
    except Exception as e:
        raise _fail(e, "delta-votes")


# ── debate level ─────────────────────────────────────────────────────────────

@app.post("/score/information-density")
def api_infodensity(req: InfoDensityRequest):
    """How much of one speaker's talk carried a point not already made."""
    try:
        return _ok(score_information_density(
            turns=req.turns, topic=req.topic, speaker=req.speaker,
            model=req.model))
    except Exception as e:
        raise _fail(e, "information-density")


@app.post("/score/rebuttal-debate")
def api_rebuttal(req: RebuttalRequest):
    """Sentence-level rebuttal links plus per-speaker global sufficiency."""
    try:
        return _ok(score_rebuttal_debate(
            [t.model_dump() for t in req.turns],
            model=req.model, screen_min=req.screen_min, top_k=req.top_k,
            exclude_openers=req.exclude_openers, use_llm=req.use_llm))
    except Exception as e:
        raise _fail(e, "rebuttal-debate")


@app.post("/score/argument-survival")
def api_survival(req: SurvivalRequest):
    """Which arguments survived. Bookkeeping over the rebuttal links — pass
    rebuttal_result to reuse a pass you already ran."""
    try:
        turns = [t.model_dump() for t in req.turns]
        reb = req.rebuttal_result
        if reb is None:
            reb = score_rebuttal_debate(turns, model=req.model,
                                        use_llm=req.use_llm)
        return _ok(score_argument_survival(turns, rebuttal_result=reb))
    except Exception as e:
        raise _fail(e, "argument-survival")


@app.post("/score/debate/llm-winner")
def api_llm_debate_winner(req: LLMWinnerRequest):
    """One-shot LLM-as-a-judge over the raw two-speaker debate.

    This is intentionally independent from the deterministic winner engine in
    debate_winner.py. Exactly one Ollama /api/chat call is made.
    """
    try:
        return _ok(determine_llm_debate_winner(
            topic=req.topic,
            turns=[t.model_dump(exclude_none=True) for t in req.turns],
            model=req.model,
            temperature=req.temperature,
        ))
    except Exception as e:
        raise _fail(e, "llm-debate-winner")


@app.post("/score/debate")
def api_debate(req: DebateRequest):
    """Every metric over a whole debate in one call, per speaker."""
    try:
        es = None
        if req.es_results_by_turn:
            es = {}
            for k, v in req.es_results_by_turn.items():
                try:
                    es[int(k)] = v
                except (TypeError, ValueError):
                    es[k] = v
        return _ok(score_debate(
            topic=req.topic,
            turns=[t.model_dump(exclude_none=True) for t in req.turns],
            stance_by_speaker=req.stance_by_speaker, model=req.model,
            use_llm_rebuttal=req.use_llm_rebuttal,
            exclude_openers=req.exclude_openers,
            es_results_by_turn=es, include_detail=req.include_detail,
            write_report=req.write_report))
    except Exception as e:
        raise _fail(e, "score-debate")


# ── LLM utilities ────────────────────────────────────────────────────────────

@app.post("/util/topic-hypotheses")
def api_topic(req: TopicRequest):
    try:
        kw = {"model": req.model} if req.model else {}
        h = generate_topic_hypotheses(topic=req.topic,
                                      use_cache=req.use_cache, **kw)
        return _ok({"pro_hypothesis": h.pro_hypothesis,
                    "con_hypothesis": h.con_hypothesis})
    except Exception as e:
        raise _fail(e, "topic-hypotheses")


@app.post("/util/mine-argument")
def api_mine(req: MineRequest):
    try:
        kw = {"model": req.model} if req.model else {}
        s = extract_claim_and_premises(req.argument, **kw)
        return _ok({"claim": s.claim, "premises": s.premises})
    except Exception as e:
        raise _fail(e, "mine-argument")
