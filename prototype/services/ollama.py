import os
import ollama
import json
import re

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

client = ollama.Client(host=OLLAMA_URL)

def generate_references(topic: str) -> list:
    """Generate topic-specific reference arguments using LLM"""
    ref_prompt = f"""Generate exactly 6 common debate arguments about this topic.
Return ONLY a valid JSON array. No explanation, no markdown, no extra text.

Example output format:
[
  {{"id": 1, "argument": "first argument here"}},
  {{"id": 2, "argument": "second argument here"}},
  {{"id": 3, "argument": "third argument here"}},
  {{"id": 4, "argument": "fourth argument here"}},
  {{"id": 5, "argument": "fifth argument here"}},
  {{"id": 6, "argument": "sixth argument here"}}
]

Topic: {topic}

Output ONLY the JSON array:"""

    ref_response = client.generate(model=MODEL, prompt=ref_prompt)
    ref_raw = ref_response["response"].strip()
    ref_raw = re.sub(r"```json|```", "", ref_raw).strip()

    start = ref_raw.find("[")
    end = ref_raw.rfind("]") + 1
    if start != -1 and end > start:
        ref_raw = ref_raw[start:end]

    try:
        references = json.loads(ref_raw)
        if isinstance(references, list) and len(references) > 0:
            return references
    except:
        pass

    return [
        {"id": 1, "argument": f"There are strong reasons to support: {topic}"},
        {"id": 2, "argument": f"There are strong reasons to oppose: {topic}"},
        {"id": 3, "argument": f"Economic impact of: {topic}"},
        {"id": 4, "argument": f"Social impact of: {topic}"},
        {"id": 5, "argument": f"Long term consequences of: {topic}"},
        {"id": 6, "argument": f"Ethical considerations of: {topic}"},
    ]