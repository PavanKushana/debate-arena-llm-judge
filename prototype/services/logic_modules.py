import os
import json
import re
import logging
import ollama

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    # Strip thinking tags (qwen3 outputs these)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    # Find first JSON object
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse JSON from model response: {text[:300]}")


class LogicDetector:
    def __init__(self, model: str = "llama3.2", ollama_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")):
        self.model = model
        self.client = ollama.Client(host=ollama_url)

    def detect_support(self, premises: list[str], claim: str) -> dict:
        """Module 1 — determine whether premises logically support a claim."""
        premises_text = "\n".join(f"- {p}" for p in premises)
        prompt = f"""You are a logic expert. Decide whether the premises logically support the claim.

Premises:
{premises_text}

Claim: {claim}

Rules:
- "supported": true if the premises give sufficient logical reason to accept the claim.
- "confidence": your certainty as a float between 0.0 and 1.0.
- "reasoning": one sentence explaining your decision.

Reply with ONLY this JSON object, nothing else:
{{
  "supported": true or false,
  "confidence": 0.0,
  "reasoning": "..."
}}"""
        response = self.client.generate(model=self.model, prompt=prompt)
        raw = response.get("response", "")
        result = _extract_json(raw)
        return {
            "supported": bool(result.get("supported", False)),
            "confidence": max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
            "reasoning": str(result.get("reasoning", "No reasoning provided.")),
        }

    def classify_relation(self, argument_1: dict, argument_2: dict) -> dict:
        """Module 2 — classify the logical relation between two arguments."""
        def fmt(arg):
            prems = "\n".join(f"  - {p}" for p in arg.get("premises", []))
            return f"Premises:\n{prems}\nClaim: {arg.get('claim', '')}"

        prompt = f"""You are an argumentation theory expert.

Argument 1:
{fmt(argument_1)}

Argument 2:
{fmt(argument_2)}

Classify how Argument 2 relates to Argument 1:
- "rebuttal": Argument 2 directly contradicts the CONCLUSION (claim) of Argument 1.
- "undercut": Argument 2 attacks the EVIDENCE or REASONING of Argument 1 without denying its conclusion.

Rules:
- "relation": exactly "rebuttal" or "undercut".
- "confidence": float between 0.0 and 1.0.
- "reasoning": one sentence explaining your decision.

Reply with ONLY this JSON object, nothing else:
{{
  "relation": "rebuttal",
  "confidence": 0.0,
  "reasoning": "..."
}}"""
        response = self.client.generate(model=self.model, prompt=prompt)
        raw = response.get("response", "")
        result = _extract_json(raw)
        relation = str(result.get("relation", "rebuttal")).lower().strip()
        if relation not in ("rebuttal", "undercut"):
            relation = "rebuttal"
        return {
            "relation": relation,
            "confidence": max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
            "reasoning": str(result.get("reasoning", "No reasoning provided.")),
        }
