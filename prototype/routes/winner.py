from fastapi import APIRouter
from models import WinnerRequest

router = APIRouter()

def coverage_based(teams: list) -> dict:
    """Winner = team with highest coverage"""
    best = max(teams, key=lambda t: t.get("coverage", 0))
    top = [t for t in teams if t.get("coverage", 0) == best.get("coverage", 0)]
    winner = "Draw" if len(top) > 1 else best.get("speaker", "Unknown")
    return {
        "strategy": "coverage_based",
        "winner": winner,
        "reasoning": f"{winner} addressed the most reference arguments.",
        "all_scores": sorted(
            [{"speaker": t.get("speaker"), "coverage": t.get("coverage", 0)} for t in teams],
            key=lambda x: x["coverage"], reverse=True
        )
    }

def quality_weighted(teams: list) -> dict:
    """Winner = team with highest quality score"""
    best = max(teams, key=lambda t: t.get("quality", 0))
    top = [t for t in teams if t.get("quality", 0) == best.get("quality", 0)]
    winner = "Draw" if len(top) > 1 else best.get("speaker", "Unknown")
    return {
        "strategy": "quality_weighted",
        "winner": winner,
        "reasoning": f"{winner} had the highest average argument quality.",
        "all_scores": sorted(
            [{"speaker": t.get("speaker"), "quality": t.get("quality", 0)} for t in teams],
            key=lambda x: x["quality"], reverse=True
        )
    }

def composite_score(teams: list) -> dict:
    """Winner = 0.5 x coverage + 0.5 x quality"""
    for t in teams:
        t["composite"] = round(0.5 * t.get("coverage", 0) + 0.5 * t.get("quality", 0), 2)
    best = max(teams, key=lambda t: t.get("composite", 0))
    top = [t for t in teams if t.get("composite", 0) == best.get("composite", 0)]
    winner = "Draw" if len(top) > 1 else best.get("speaker", "Unknown")
    return {
        "strategy": "composite",
        "winner": winner,
        "reasoning": f"{winner} won using composite formula: 0.5 x coverage + 0.5 x quality.",
        "all_scores": sorted(
            [{"speaker": t.get("speaker"), "composite_score": t.get("composite", 0)} for t in teams],
            key=lambda x: x["composite_score"], reverse=True
        )
    }

def final_score_based(teams: list) -> dict:
    """Winner = team with highest final score (coverage x quality)"""
    best = max(teams, key=lambda t: t.get("final_score", 0))
    top = [t for t in teams if t.get("final_score", 0) == best.get("final_score", 0)]
    winner = "Draw" if len(top) > 1 else best.get("speaker", "Unknown")
    return {
        "strategy": "final_score",
        "winner": winner,
        "reasoning": f"{winner} won with highest score of {best.get('final_score', 0)}!",
        "all_scores": sorted(
            [{"speaker": t.get("speaker"), "final_score": t.get("final_score", 0)} for t in teams],
            key=lambda x: x["final_score"], reverse=True
        )
    }

@router.post("/winner")
def determine_winner(request: WinnerRequest):
    if not request.teams:
        return {"error": "No teams provided"}

    teams = request.teams

    # Run all strategies
    results = {
        "coverage_based": coverage_based(teams),
        "quality_weighted": quality_weighted(teams),
        "composite": composite_score(teams),
        "final_score": final_score_based(teams)
    }

    # Overall winner — majority vote across strategies
    votes = {}
    for strategy_result in results.values():
        w = strategy_result["winner"]
        votes[w] = votes.get(w, 0) + 1

    overall_winner = max(votes, key=votes.get)

    return {
        "overall_winner": overall_winner,
        "vote_count": votes,
        "strategies": results
    }