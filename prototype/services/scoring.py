import re
from services.ollama import client, MODEL
from services.logic_modules import LogicDetector

detector = LogicDetector()

def score_argument(claim: str, references: list) -> tuple:
    """Score a single argument against all references, return best match and score"""
    best_match = None
    best_score = 0.0

    for ref in references:
        score_prompt = f"""You are a debate judge. Rate how similar these two arguments are.

Rules:
- 0.0 = completely different topics
- 0.5 = same topic but different viewpoint
- 1.0 = same argument, same viewpoint

Argument 1: {claim}
Argument 2: {ref["argument"]}

Reply with ONE decimal number only between 0.0 and 1.0:"""

        score_response = client.generate(model=MODEL, prompt=score_prompt)
        score_text = score_response["response"].strip()

        numbers = re.findall(r"0?\.\d+|\d+\.\d+", score_text)
        if numbers:
            score = float(numbers[0])
            score = max(0.0, min(1.0, score))
        else:
            score = 0.0

        if score > best_score:
            best_score = score
            best_match = ref["argument"]

    return best_match, best_score


def detect_rebuttals(all_scorecards: list) -> list:
    """Detect rebuttals and undercuts between teams"""
    rebuttal_results = []

    # Compare each pair of teams
    for i in range(len(all_scorecards)):
        for j in range(len(all_scorecards)):
            if i == j:
                continue

            team_a = all_scorecards[i]
            team_b = all_scorecards[j]

            for arg_a in team_a.get("extracted_arguments", []):
                for arg_b in team_b.get("extracted_arguments", []):
                    try:
                        relation = detector.classify_relation(
                            {"claim": arg_a.get("argument", ""), "premises": []},
                            {"claim": arg_b.get("argument", ""), "premises": []}
                        )

                        if relation["confidence"] >= 0.6:
                            rebuttal_results.append({
                                "from_speaker": team_b["speaker"],
                                "to_speaker": team_a["speaker"],
                                "argument": arg_b.get("argument", ""),
                                "targets": arg_a.get("argument", ""),
                                "relation": relation["relation"],
                                "confidence": relation["confidence"],
                                "reasoning": relation["reasoning"]
                            })
                    except:
                        continue

    return rebuttal_results