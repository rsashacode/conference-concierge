"""RAG search over the session's indexed schedule and schedule overview retrieval."""

from src.rag.schedule_rag import rag_query, get_schedule_overview_text

declaration = {
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": "Semantic search over the user's uploaded conference schedule. \
            Use this to find talks/sessions by topic, track, or keyword. \
                Returns matching sessions with title, room, time, and excerpt.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string", 
                    "description": "Search query (e.g. 'RAG', 'machine learning', 'keynote') to find relevant sessions in the schedule."
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

get_schedule_overview_declaration = {
    "type": "function",
    "function": {
        "name": "get_schedule_overview",
        "description": "Retrieve the full schedule overview for the user's uploaded conference. \
            Returns a compact list of all sessions (title, time, room, track) so you can see the whole program at a glance.\
                Call this when you need the full schedule structure; use rag_search for topic-specific sessions.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}


def rag_search(query: str, session_id: str | None = None) -> str:
    """
    Semantic search over the session's indexed schedule.
    session_id is injected by the executor (conversation_id).
    """
    if not session_id:
        return "No session context. Use this tool from the conference concierge with a session that has an uploaded schedule."
    return rag_query(session_id, query)


def get_schedule_overview(session_id: str | None = None) -> str:
    """
    Return the compact schedule overview (all sessions: title, time, room, track) for the session.
    session_id is injected by the executor.
    """
    if not session_id:
        return "No session context."
    return get_schedule_overview_text(session_id)
