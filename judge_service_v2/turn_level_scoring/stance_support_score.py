"""
stance_support_score.py — did the speaker hold the side he was assigned?

The topic is turned into the two claims the debate is fought over
(utils/topic_hypotheses), and the turn's mined claim is run through DeBERTa NLI
against THIS speaker's own-side claim:

    entailment    -> supported his side
    contradiction -> argued against it
    neutral       -> took no clear position

Hypotheses come from the LLM only. There is no "tidy the raw topic into a
thesis" fallback: that works only for already-declarative topics, and produces
confident-looking nonsense for question- and motion-form ones. If the LLM is
unreachable this raises.

Thresholds are hand-picked and unvalidated — treat the raw score as usable for
ranking and HIGH/MEDIUM/LOW as provisional.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

from ..utils.telemetry import log_call, arg_id
from ..utils.topic_hypotheses import generate_topic_hypotheses

logger = logging.getLogger(__name__)

STANCE_NLI_MODEL = os.getenv(
    "STANCE_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
STANCE_HIGH = float(os.getenv("STANCE_HIGH", "0.60"))
STANCE_MEDIUM = float(os.getenv("STANCE_MEDIUM", "0.35"))
STANCE_MAX_LENGTH = int(os.getenv("STANCE_MAX_LENGTH", "256"))
# For checkpoints with generic LABEL_0/1/2 names, e.g.
# "entailment,neutral,contradiction" in that model's index order.
STANCE_NLI_LABEL_ORDER = os.getenv("STANCE_NLI_LABEL_ORDER", "").strip()

_VALID_STANCES = {"PRO", "CON"}
_CANON = ("entailment", "neutral", "contradiction")

_model = None
_tokenizer = None
_idx_to_canon: Optional[dict] = None
_device = None


def _canonical_label(raw: str) -> Optional[str]:
    s = str(raw).lower()
    if "contrad" in s:
        return "contradiction"
    if "entail" in s:
        return "entailment"
    if "neutral" in s:
        return "neutral"
    return None


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


def _probs_from_logits(logits, idx_to_canon: dict) -> dict:
    """Map logits to canonical labels using the model's OWN label order.
    Hardcoding indices breaks silently on contradiction-first checkpoints."""
    probs = _softmax(np.asarray(logits, dtype=np.float64))
    out = {c: 0.0 for c in _CANON}
    for idx, canon in idx_to_canon.items():
        out[canon] = float(probs[idx])
    return out


def _label(support: float) -> str:
    if support >= STANCE_HIGH:
        return "HIGH"
    if support >= STANCE_MEDIUM:
        return "MEDIUM"
    return "LOW"


def _side(h, stance: str) -> str:
    return h.pro_hypothesis if stance == "PRO" else h.con_hypothesis


def _get_nli_model():
    global _model, _tokenizer, _idx_to_canon, _device
    if _model is not None:
        return _model, _tokenizer, _idx_to_canon

    try:
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
    except ImportError as exc:
        raise RuntimeError("needs torch + transformers") from exc

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(STANCE_NLI_MODEL)
    _model = AutoModelForSequenceClassification.from_pretrained(STANCE_NLI_MODEL)
    _model.to(_device).eval()
    _idx_to_canon = _resolve_label_map(_model.config)
    logger.info("Stance NLI loaded: %s on %s | labels=%s",
                STANCE_NLI_MODEL, _device, _idx_to_canon)
    return _model, _tokenizer, _idx_to_canon


def _resolve_label_map(config) -> dict:
    if STANCE_NLI_LABEL_ORDER:
        order = [x.strip().lower() for x in STANCE_NLI_LABEL_ORDER.split(",")]
        mapping = {i: c for i, n in enumerate(order)
                   if (c := _canonical_label(n))}
    else:
        mapping = {int(i): c for i, n in (config.id2label or {}).items()
                   if (c := _canonical_label(n))}
    if set(mapping.values()) != set(_CANON):
        raise RuntimeError(
            f"Could not map model labels to entailment/neutral/contradiction "
            f"(id2label={getattr(config, 'id2label', None)}). Set "
            f"STANCE_NLI_LABEL_ORDER in this model's index order.")
    return mapping


def _nli_probs(text: str, hypothesis: str) -> dict:
    """One NLI forward pass. Shared by cogency and rebuttal — only one DeBERTa
    is ever loaded in the process."""
    import torch

    if not hypothesis or not hypothesis.strip():
        raise ValueError("hypothesis must be a non-empty string")
    model, tokenizer, idx_to_canon = _get_nli_model()
    enc = tokenizer(text, hypothesis, truncation=True,
                    max_length=STANCE_MAX_LENGTH,
                    return_tensors="pt").to(_device)
    with torch.no_grad():
        logits = model(**enc).logits[0].detach().cpu().numpy()
    return _probs_from_logits(logits, idx_to_canon)


def score_stance_support(
    topic: str,
    turn: str,
    stance: str,
    claim: Optional[str] = None,
    hypothesis: Optional[str] = None,
    model: Optional[str] = None,
    argument_id: Optional[str] = None,
) -> dict:
    """
    How strongly one turn supports its speaker's own side.

    claim      : the mined claim. Scored instead of the whole turn when present
                 — stance is about the conclusion; premises are neutral noise.
    hypothesis : the claim THIS speaker's side defends, one sentence. Generated
                 from the topic when omitted. `stance` says which side, so this
                 is never a PRO/CON pair.

    Returns stance_support_score (P(entails own side)), label, stance,
    scored_on, hypothesis, probs.
    """
    if not turn or not turn.strip():
        raise ValueError("turn must be a non-empty string")
    stance_up = stance.upper()
    if stance_up not in _VALID_STANCES:
        raise ValueError(f"stance must be PRO or CON, got {stance!r}")

    text, scored_on = ((claim.strip(), "claim") if claim and claim.strip()
                       else (turn.strip(), "turn"))

    own = (hypothesis or "").strip()
    if not own:
        kw = {"model": model} if model else {}
        own = _side(generate_topic_hypotheses(topic=topic, use_cache=True, **kw),
                    stance_up).strip()

    aid = argument_id or arg_id(turn)
    with log_call("stance_support", "nli_stance", argument_id=aid,
                  topic=topic, stance=stance_up, turn=turn, claim=claim,
                  scored_on=scored_on, hypothesis=own,
                  model=STANCE_NLI_MODEL) as rec:
        probs = _nli_probs(text, own)
        support = probs["entailment"]
        result = {
            "stance_support_score": round(support, 4),
            "label": _label(support),
            "stance": stance_up,
            "scored_on": scored_on,
            "hypothesis": own,
            "probs": {k: round(v, 4) for k, v in probs.items()},
        }
        rec["response"] = result
        return result


def score_stance_support_batch(
    topic: str,
    turns: List[dict],
    hypotheses: Optional[dict] = None,
    model: Optional[str] = None,
) -> List[dict]:
    """
    Score many turns for one topic. Each item: {"turn", "stance", and optionally
    "claim", "speaker", "turn_no", "argument_id"}.

    Hypotheses are resolved once for the batch — both sides, since a debate has
    speakers on both. One failure does not sink the rest.
    """
    if not (hypotheses and hypotheses.get("PRO") and hypotheses.get("CON")):
        kw = {"model": model} if model else {}
        h = generate_topic_hypotheses(topic=topic, use_cache=True, **kw)
        hypotheses = {"PRO": h.pro_hypothesis, "CON": h.con_hypothesis}

    results: List[dict] = []
    for item in turns:
        passthrough = {k: item[k] for k in ("speaker", "turn_no") if k in item}
        try:
            res = score_stance_support(
                topic=topic, turn=item["turn"], stance=item["stance"],
                claim=item.get("claim"),
                hypothesis=hypotheses[item["stance"].upper()],
                model=model, argument_id=item.get("argument_id"))
            results.append({**passthrough, **res})
        except Exception as e:
            logger.warning("stance support failed for a turn: %s", e)
            results.append({**passthrough, "error": str(e),
                            "stance_support_score": None})
    return results


def aggregate_stance_by_speaker(turn_results: List[dict]) -> dict:
    """
    Did each speaker hold their assigned position across the debate?

    weak_turns lists the turn_nos that scored below STANCE_MEDIUM — the turns
    where the speaker did not back their own side.
    """
    by_speaker: dict = {}
    for r in turn_results:
        sp = r.get("speaker")
        if sp is None or r.get("stance_support_score") is None:
            continue
        by_speaker.setdefault(sp, []).append(r)

    out: dict = {}
    for sp, rows in by_speaker.items():
        supports = [r["stance_support_score"] for r in rows]
        mean_support = sum(supports) / len(supports)
        weak = [r.get("turn_no") for r in rows
                if r["stance_support_score"] < STANCE_MEDIUM]
        out[sp] = {
            "stance": rows[0].get("stance"),
            "turns_scored": len(rows),
            "mean_stance_support": round(mean_support, 4),
            "min_stance_support": round(min(supports), 4),
            "weak_turns": weak,
            "position_held": bool(not weak and mean_support >= STANCE_MEDIUM),
        }
    return out
