import json
import os
import requests


SERPER_API_KEY = os.environ["SERPER_API_KEY"]


declaration = {
    "type": "function",
    "function": {
        "name": "google_web_search",
        "description": "Searches for information on the web using Serper's search endpoint.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for information on the web."
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def google_web_search(query: str) -> str:
    """
    Searches for information on the web using Serper's search endpoint.
    Requires SERPER_API_KEY in the environment.
    """
    url = "https://google.serper.dev/search"
    payload = {"q": query, "gl": "de"}
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        return f"Error calling search API: {e}"
    try:
        response_json = response.json()
    except json.JSONDecodeError:
        return f"Error: invalid JSON from search API: {response.text[:500]}"
    organic = response_json.get("organic", [])
    if not organic:
        return json.dumps({"organic": [], "message": "No organic results returned."})
    return json.dumps(organic, ensure_ascii=False)
