from fastapi import APIRouter
from models import DebateRequest, DebateRequest2, WinnerRequest
from routes.score import score_debate
from routes.winner import determine_winner
from services.ollama import generate_references, client, MODEL
from services.scoring import detect_rebuttals

router = APIRouter()

@router.post("/debate")
def full_debate(request: DebateRequest2):
    if not request.teams:
        return {"error": "No teams provided"}

    # Generate references ONCE for all teams
    references = request.reference_arguments if request.reference_arguments else generate_references(request.topic)

    # Score each team using SAME references
    all_scorecards = []
    for team in request.teams:
        scorecard = score_debate(DebateRequest(
            topic=request.topic,
            speaker=team.get("speaker", "Unknown"),
            transcript=team.get("transcript", ""),
            reference_arguments=references
        ))
        all_scorecards.append(scorecard)

    # Detect rebuttals between teams
    rebuttals = detect_rebuttals(all_scorecards)

    # Determine winner
    winner_result = determine_winner(WinnerRequest(
        teams=all_scorecards
    ))

    # LLM independent assessment
    scorecards_summary = "\n".join([
        f"- {s['speaker']}: coverage={s['coverage']}, quality={s['quality']}, score={s['final_score']}"
        for s in all_scorecards
    ])

    assessment_prompt = f"""You are an expert debate judge. Based on the following debate scores, provide a brief holistic assessment of the debate.

Topic: {request.topic}

Scores:
{scorecards_summary}

Winner: {winner_result['overall_winner']}

Provide a 2-3 sentence assessment of how the debate went, who argued most effectively and why.
Reply with plain text only, no JSON."""

    assessment_response = client.generate(model=MODEL, prompt=assessment_prompt)
    llm_assessment = assessment_response["response"].strip()

    return {
        "topic": request.topic,
        "references_used": references,
        "scorecards": all_scorecards,
        "rebuttals": rebuttals,
        "winner": winner_result,
        "llm_assessment": llm_assessment
    }