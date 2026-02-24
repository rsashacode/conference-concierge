import json
from pathlib import Path
from typing import Iterator, Any

import chromadb
from openai import OpenAI
from pydantic import BaseModel, Field

from src.prompts import RERANK_SYSTEM


DB_DIR = Path("db")
SCHEDULE_OVERVIEW_FILENAME = "schedule_overview.txt"
CHROMA_SUBDIR = "chroma"
CHROMA_COLLECTION_NAME = "schedule"
EMBEDDING_MODEL = "text-embedding-3-small"
RERANK_MODEL = "gpt-4o-mini"
RAG_RETRIEVE_K = 20
RAG_TOP_K = 5


class RerankResult(BaseModel):
    index: int = Field(description="The original index of the entry")
    score: int = Field(description="Number from 1 to 10 (relevance)")
    reason: str = Field(description="One short phrase why it's relevant (optional)", default="")


class RerankResponse(BaseModel):
    results: list[RerankResult] = Field(description="The reranked results")


def _talk_to_text(talk: dict) -> str:
    """One talk dict -> one searchable text block."""
    parts = [
        f"Title: {talk.get('title') or ''}",
        f"Track: {talk.get('track') or ''}",
        f"Type: {talk.get('type') or ''}",
        f"Room: {talk.get('room') or ''}",
        f"Date: {talk.get('date') or ''}",
        f"Start: {talk.get('start') or ''}",
        f"Duration: {talk.get('duration') or ''}",
        f"Abstract: {talk.get('abstract') or ''}",
        f"Description: {(talk.get('description') or '')[:4000]}",
    ]
    for i, p in enumerate(talk.get("persons") or []):
        parts.append(f"Speaker {i + 1}: {p.get('public_name') or p.get('name') or ''}")
        if p.get("biography"):
            parts.append(f"  Biography: {p['biography'][:1500]}")
    return "\n".join(parts)


def schedule_docs_from_json(data: dict) -> Iterator[tuple[str, str, dict]]:
    """Yield (id, text, metadata) for each talk in pretalx-style schedule JSON."""
    schedule = data.get("schedule") or data
    days = (schedule.get("conference") or {}).get("days") or schedule.get("days") or []
    for day in days:
        day_date = day.get("date") or ""
        for room_name, talks in (day.get("rooms") or {}).items():
            for talk in talks:
                guid = talk.get("guid") or talk.get("code") or talk.get("id")
                sid = str(guid) if guid is not None else f"{day_date}_{room_name}_{talk.get('start', '')}"
                meta = {
                    "room": room_name,
                    "date": day_date,
                    "start": talk.get("start") or "",
                    "track": talk.get("track") or "",
                    "title": (talk.get("title") or "")[:200],
                }
                yield sid, _talk_to_text(talk), meta


def schedule_overview_from_json(data: dict) -> str:
    """Compact text overview of the full schedule for the LLM."""
    schedule = data.get("schedule") or data
    conference = schedule.get("conference") or {}
    title = conference.get("title") or "Conference"
    lines = [f"# {title}", ""]
    days = conference.get("days") or schedule.get("days") or []
    for day in days:
        lines.append(f"## {day.get('date') or ''}\n")
        rooms = day.get("rooms") or {}
        for room_name in sorted(rooms.keys()):
            for talk in rooms[room_name]:
                start = talk.get("start") or ""
                lines.append(f"- {start} | {room_name} | {talk.get('track') or ''}")
                lines.append(f"  {talk.get('title') or ''}")
        lines.append("")
    return "\n".join(lines)


def _chroma_client(session_id: str):
    path = DB_DIR / session_id / CHROMA_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _embed(texts: list[str], client: OpenAI) -> list[list[float]]:
    if not texts:
        return []
    out = []
    for i in range(0, len(texts), 100):
        batch = [t.strip() or " " for t in texts[i : i + 100]]
        r = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        out.extend(e.embedding for e in r.data)
    return out


def index_schedule_for_session(session_id: str, schedule_path: str | Path) -> str:
    """Load schedule JSON, embed, store in ChromaDB and write overview. Returns status message."""
    path = Path(schedule_path)
    if not path.exists():
        return f"File not found: {path}"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return f"Invalid or unreadable JSON: {e}"
    schedule = raw.get("schedule") or raw
    days = (schedule.get("conference") or {}).get("days") or schedule.get("days") or []
    if not days:
        return "Not a recognized schedule format (missing days)."

    ids, texts, metadatas = [], [], []
    for sid, text, meta in schedule_docs_from_json(raw):
        ids.append(sid)
        texts.append(text)
        metadatas.append(meta)
    if not ids:
        return "No talks found in schedule."

    session_dir = DB_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / SCHEDULE_OVERVIEW_FILENAME).write_text(schedule_overview_from_json(raw), encoding="utf-8")

    client = OpenAI()
    embeddings = _embed(texts, client)
    chroma = _chroma_client(session_id)
    
    try:
        chroma.delete_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        pass
    
    coll = chroma.create_collection(name=CHROMA_COLLECTION_NAME)
    for m in metadatas:
        for k, v in m.items():
            if v is None or not isinstance(v, (str, int, float, bool)):
                m[k] = "" if v is None else str(v)
    coll.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)  # pyright: ignore[reportArgumentType]
    return f"Indexed {len(ids)} sessions for RAG and saved schedule overview."


def get_schedule_overview_text(session_id: str) -> str:
    """Return schedule overview for this session or a short message if none."""
    path = DB_DIR / session_id / SCHEDULE_OVERVIEW_FILENAME
    if not path.exists():
        return "No schedule overview for this session. Upload a schedule file first."
    return path.read_text(encoding="utf-8")


def _rerank(
        query: str, 
        docs: list[str], 
        metadatas: list[dict | Any], 
        distances: list[float], 
        client: OpenAI, 
        top_k: int = RAG_TOP_K
    ):
    """Rerank with OpenAI; return up to top_k (doc, metadata, distance)."""
    if not docs:
        return []
    
    blocks = []
    for i, (doc, meta) in enumerate(zip(docs, metadatas)):
        meta = meta or {}
        excerpt = doc[:600] + "..." if len(doc) > 600 else doc
        blocks.append(
            f"[{i}] \
            Title: {meta.get('title') or '(no title)'}\n \
            Room: {meta.get('room') or ''} | \
            Track: {meta.get('track') or ''}\n \
            Excerpt: {excerpt}"
        )
    resp = client.chat.completions.parse(
        model=RERANK_MODEL,
        messages=[
            {"role": "system", "content": RERANK_SYSTEM}, 
            {"role": "user", "content": f"Query: {query}\n\nRetrieved entries:\n" + "\n\n".join(blocks)}],
        response_format=RerankResponse,
    )
    result = resp.choices[0].message.parsed.results if resp.choices[0].message.parsed else []
    return result


def rag_query(session_id: str, query: str, n_results: int = RAG_TOP_K) -> str:
    """Semantic search + rerank; return formatted top results."""
    try:
        coll = _chroma_client(session_id).get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        return "No schedule has been indexed for this session. Upload a schedule file first."

    client = OpenAI()
    q_emb = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    results = coll.query(query_embeddings=[q_emb.data[0].embedding], n_results=RAG_RETRIEVE_K)
    docs = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    if not docs:
        return "No matching sessions found."

    reranked = _rerank(query, docs, metadatas, distances, client, top_k=n_results)
    if not reranked:
        return "No relevant sessions found after re-ranking."
    reranked = sorted(reranked, key=lambda r: r.score, reverse=True)

    out = []
    for i, rr in enumerate(reranked):
        idx = rr.index
        if idx < 0 or idx >= len(docs):
            continue
        doc = docs[idx]
        meta = metadatas[idx] or {}
        excerpt = doc[:800] + "..." if len(doc) > 800 else doc
        
        out.append(f"--- Result {i + 1} ---")
        out.append(f"Title: {meta.get('title') or '(no title)'}")
        out.append(f"Room: {meta.get('room') or ''}")
        out.append(f"Date: {meta.get('date') or ''}")
        out.append("Start: {meta.get('start') or ''}")
        out.append(f"Track: {meta.get('track') or ''}")
        out.append(f"Excerpt: {excerpt}\n")
    return "\n".join(out).strip()
