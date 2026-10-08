from fastapi import APIRouter
import json
import re
from models import DebateRequest
from services.ollama import client, MODEL
from services.scoring import score_argument
from services.ollama import generate_references
from services.elasticsearch import get_reference_arguments, check_health
from services.huggingface import score_arguments_hf

router = APIRouter()

@router.post("/score")
def score_debate(request: DebateRequest):
    # Step 1: Extract arguments using LLM
    prompt = f"""You are a debate analyst. Extract specific arguments from the transcript below.

IMPORTANT RULES:
- Extract ONLY arguments that appear in the transcript
- Do NOT repeat the topic as an argument
- Each argument must be a specific claim made by the speaker
- Include the reason/evidence they gave

Return ONLY a JSON array like this:
[
  {{"claim": "specific claim made", "premise": "reason or evidence given"}},
  {{"claim": "another specific claim", "premise": "reason or evidence given"}}
]

Topic: {request.topic}
Speaker: {request.speaker}
Transcript: {request.transcript}

Return ONLY the JSON array, nothing else."""

    response = client.generate(model=MODEL, prompt=prompt)
    raw = response["response"].strip()

    raw = re.sub(r"```json|```", "", raw).strip()
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start != -1 and end > start:
        raw = raw[start:end]

    try:
        extracted_args = json.loads(raw)
        if not isinstance(extracted_args, list):
            extracted_args = []
    except:
        extracted_args = []

    # Step 2: Get references — Elasticsearch first, then fallback
    es_available = check_health()
    if request.reference_arguments:
        references = request.reference_arguments
        ref_source = "provided"
    elif es_available:
        es_refs = get_reference_arguments(request.topic)
        if es_refs:
            references = es_refs
            ref_source = "elasticsearch"
        else:
            references = generate_references(request.topic)
            ref_source = "llm_generated"
    else:
        references = generate_references(request.topic)
        ref_source = "llm_generated"

    # Step 3: Score using Ollama (LLM-as-judge)
    matched = 0
    total_quality = 0.0
    ollama_results = []

    # Normalize extracted args format
    normalized_args = []
    for arg in extracted_args:
        if isinstance(arg, dict):
            normalized_args.append({
                "argument": arg.get("claim", arg.get("argument", "")),
                "premise": arg.get("premise", "")
            })

    for arg in normalized_args:
        best_match, best_score = score_argument(arg.get("argument", ""), references)

        if best_score >= 0.5:
            matched += 1

        total_quality += best_score
        ollama_results.append({
            "argument": arg.get("argument", ""),
            "best_match": best_match,
            "quality_score": round(best_score, 2)
        })

    total_args = len(normalized_args)
    ollama_coverage = round(matched / total_args, 2) if total_args > 0 else 0.0
    ollama_quality = round(total_quality / total_args, 2) if total_args > 0 else 0.0
    ollama_final = round(ollama_coverage * ollama_quality, 2)

    # Step 4: Score using HuggingFace zero-shot
    hf_scores = score_arguments_hf(normalized_args, references)

    return {
        "speaker": request.speaker,
        "topic": request.topic,
        "reference_source": ref_source,
        "references_used": references[:3],
        "ollama_scoring": {
            "extracted_arguments": ollama_results,
            "coverage": ollama_coverage,
            "quality": ollama_quality,
            "final_score": ollama_final
        },
        "huggingface_scoring": hf_scores,
        "extracted_arguments": ollama_results,
        "coverage": ollama_coverage,
        "quality": ollama_quality,
        "final_score": ollama_final
    }