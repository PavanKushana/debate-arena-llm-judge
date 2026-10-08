"""
llm_debate_winner.py — one-shot LLM-as-a-judge for a complete two-speaker debate.

This module is intentionally separate from debate_winner.py.
- debate_winner.py: deterministic, metric-based winner decision.
- this module: one LLM call that reads the raw debate and produces an
  independent winner + concise decision rationale.

The LLM is asked for decision rationale, not hidden chain-of-thought.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests

try:
    from ..utils.judge_config import OLLAMA_HOST, OLLAMA_MODEL
except ImportError:  # standalone test/import fallback
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

logger = logging.getLogger(__name__)

LLM_WINNER_CONNECT_TIMEOUT = float(os.getenv("LLM_WINNER_CONNECT_TIMEOUT", "10"))
LLM_WINNER_READ_TIMEOUT = float(os.getenv("LLM_WINNER_READ_TIMEOUT", "300"))
LLM_WINNER_KEEP_ALIVE = os.getenv("LLM_WINNER_KEEP_ALIVE", "30m")
LLM_WINNER_NUM_PREDICT = int(os.getenv("LLM_WINNER_NUM_PREDICT", "2048"))
LLM_WINNER_THINK = os.getenv("LLM_WINNER_THINK", "false")


_SYSTEM_PROMPT = """
You are an expert debate judge.

Evaluate the complete debate between exactly two speakers. Decide which speaker
performed better overall on the debate itself, or return TIE when the debate
cannot reasonably distinguish them.

Consider, in combination:
- whether the arguments are logically coherent;
- whether claims are supported by reasons or evidence;
- whether the speaker directly addresses and responds to the opponent;
- whether rebuttals actually weaken the opponent's important arguments;
- whether the speaker remains consistent;
- whether the speaker introduces relevant and substantive points;
- whether important objections are left unanswered.

Do not judge the person. Judge only the arguments presented in the supplied
transcript and the stated topic.

Give a concise decision rationale based on observable parts of the debate.
Do not provide hidden chain-of-thought or private internal reasoning.
Do not use a numerical overall score, weighted formula, or popularity argument.

Return ONLY valid JSON with exactly these fields:
{
  "winner": "<speaker name>" or "TIE",
  "reasoning": "clear explanation of why the winner was selected",
  "speaker_analysis": {
    "<speaker A>": "main strengths and weaknesses",
    "<speaker B>": "main strengths and weaknesses"
  },
  "decisive_points": [
    "specific debate point that influenced the decision",
    "another specific debate point if relevant"
  ]
}
""".strip()


def _normalise_turns(turns: List[dict]) -> tuple[List[str], str]:
    speakers: List[str] = []
    lines: List[str] = []

    for i, turn in enumerate(turns, start=1):
        speaker = str(turn.get("speaker") or "UNKNOWN").strip()
        text = str(turn.get("text") or "").strip()
        if speaker not in speakers:
            speakers.append(speaker)
        if text:
            lines.append(f"Turn {turn.get('turn_no') or i} — {speaker}: {text}")

    return speakers, "\n".join(lines)


def _validate_result(result: Dict[str, Any], speakers: List[str]) -> Dict[str, Any]:
    winner = str(result.get("winner") or "").strip()
    allowed = set(speakers) | {"TIE"}
    if winner not in allowed:
        raise ValueError(
            f"LLM returned invalid winner {winner!r}; expected one of {sorted(allowed)}"
        )

    reasoning = str(result.get("reasoning") or "").strip()
    if not reasoning:
        raise ValueError("LLM returned an empty reasoning field")

    analysis = result.get("speaker_analysis")
    if not isinstance(analysis, dict):
        analysis = {}

    decisive = result.get("decisive_points")
    if not isinstance(decisive, list):
        decisive = []

    return {
        "winner": winner,
        "reasoning": reasoning,
        "speaker_analysis": {
            speaker: str(analysis.get(speaker) or "").strip()
            for speaker in speakers
        },
        "decisive_points": [str(x).strip() for x in decisive if str(x).strip()],
    }


def determine_llm_debate_winner(
    topic: str,
    turns: List[dict],
    model: Optional[str] = None,
    host: str = OLLAMA_HOST,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Make exactly one Ollama chat call and return the LLM judge result."""
    if not topic or not topic.strip():
        raise ValueError("topic must be a non-empty string")
    if not turns:
        raise ValueError("turns must not be empty")

    speakers, transcript = _normalise_turns(turns)
    if len(speakers) != 2:
        raise ValueError(
            f"LLM debate winner requires exactly two speakers; received {len(speakers)}"
        )
    if not transcript:
        raise ValueError("debate contains no non-empty turn text")

    selected_model = model or OLLAMA_MODEL
    user_prompt = (
        f"TOPIC:\n{topic.strip()}\n\n"
        f"SPEAKERS: {speakers[0]} and {speakers[1]}\n\n"
        f"DEBATE TRANSCRIPT:\n{transcript}"
    )

    payload: Dict[str, Any] = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "keep_alive": LLM_WINNER_KEEP_ALIVE,
        "options": {
            "temperature": temperature,
            "num_predict": LLM_WINNER_NUM_PREDICT,
        },
    }

    # GPT-OSS / other reasoning models can put their reasoning in a separate
    # thinking field. We only need the final decision JSON in content.
    if LLM_WINNER_THINK.lower() in {"false", "0", "no"}:
        payload["think"] = False

    response = requests.post(
        f"{host}/api/chat",
        json=payload,
        timeout=(LLM_WINNER_CONNECT_TIMEOUT, LLM_WINNER_READ_TIMEOUT),
    )
    response.raise_for_status()

    body = response.json()
    content = ((body.get("message") or {}).get("content") or "").strip()
    if not content:
        raise RuntimeError(
            f"Ollama returned no final content for model {selected_model!r}. "
            "For reasoning models, set LLM_WINNER_THINK=false or use a plain "
            "instruct model."
        )

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM winner returned invalid JSON: {content[:500]}"
        ) from exc

    result = _validate_result(parsed, speakers)
    result["model"] = selected_model
    result["speakers"] = speakers
    return result
