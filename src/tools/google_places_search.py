import json
import os
import requests


SERPER_API_KEY = os.environ["SERPER_API_KEY"]


declaration = {
    "type": "function",
    "function": {
        "name": "google_places_search",
        "description": "Finds places, venues, or restaurants using Serper's places endpoint.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for places, venues, or restaurants."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def google_places_search(query: str) -> str:
    """
    Finds places, venues, or restaurants using Serper's places endpoint.
    If openai_client and model are provided, raw results are parsed by LLM into an answer.
    """
    url = "https://google.serper.dev/places"
    payload = {"q": query, "gl": "de"}
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        return f"Error calling places API: {e}"
    try:
        response_json = response.json()
    except json.JSONDecodeError:
        return f"Error: invalid JSON from places API: {response.text[:500]}"
    places = response_json.get("places", [])
    if not places:
        return json.dumps({"places": [], "message": "No places returned."})
    return json.dumps(places, ensure_ascii=False)
