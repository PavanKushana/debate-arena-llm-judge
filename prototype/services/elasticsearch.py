import os
import requests

# Optional reference-argument search API; without it the prototype falls back to
# LLM-generated reference arguments.
ES_BASE_URL = os.getenv("ES_API", "http://localhost:8000")

def get_reference_arguments(topic: str) -> list:
    """Fetch reference arguments using similarity search"""
    try:
        # First try exact match
        response = requests.get(
            f"{ES_BASE_URL}/arguments",
            params={"topic": topic},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            arguments = data.get("arguments", [])
            if arguments:
                return [
                    {
                        "id": arg.get("id", i),
                        "argument": arg.get("claim", ""),
                        "premises": arg.get("premises", []),
                        "stance": arg.get("stance", ""),
                        "quality_signal": arg.get("quality_signal", False)
                    }
                    for i, arg in enumerate(arguments)
                    if arg.get("claim", "").strip()
                ]

        # Fallback — use similarity search with topic as query
        sim_response = requests.post(
            f"{ES_BASE_URL}/similarity-search",
            json={"argument": topic, "top_n": 10},
            timeout=10
        )
        if sim_response.status_code == 200:
            data = sim_response.json()
            similar = data.get("similar_arguments", [])
            if similar:
                return [
                    {
                        "id": arg.get("id", i),
                        "argument": arg.get("claim", ""),
                        "premises": arg.get("premises", []),
                        "stance": arg.get("stance", ""),
                        "quality_signal": arg.get("quality_signal", False)
                    }
                    for i, arg in enumerate(similar)
                    if arg.get("claim", "").strip()
                ]
        return []
    except Exception as e:
        print(f"Elasticsearch error: {e}")
        return []

def similarity_search(argument: str, top_n: int = 5) -> list:
    """Find similar arguments from Elasticsearch"""
    try:
        response = requests.post(
            f"{ES_BASE_URL}/similarity-search",
            json={"argument": argument, "top_n": top_n},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("similar_arguments", [])
        return []
    except Exception as e:
        print(f"Elasticsearch similarity search error: {e}")
        return []

def check_health() -> bool:
    """Check if Elasticsearch API is available"""
    try:
        response = requests.get(f"{ES_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False