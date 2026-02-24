import json
import shutil
import uuid
from pathlib import Path
from datetime import datetime, timezone

DB_DIR = Path(__file__).resolve().parent / "db"
META_FILENAME = "meta.json"
UPLOADED_SUBDIR = "uploaded"


def _ensure_db_dir() -> Path:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return DB_DIR


def get_session_dir(session_id: str) -> Path:
    _ensure_db_dir()
    d = DB_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_uploaded_dir(session_id: str) -> Path:
    """Return the session's uploaded-files subfolder (db/<session_id>/uploaded/). Creates it if needed."""
    d = get_session_dir(session_id) / UPLOADED_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_uploaded_files(session_id: str) -> list[str]:
    """
    Return absolute file paths of all files in the session's uploaded subfolder
    (db/<session_id>/uploaded/). Used to pre-populate the File component after refresh or session change.
    """
    if not session_id:
        return []
    d = get_session_dir(session_id) / UPLOADED_SUBDIR
    if not d.exists() or not d.is_dir():
        return []
    out = [str(p.resolve()) for p in d.iterdir() if p.is_file()]
    out.sort(key=lambda x: x.lower())
    return out


def _meta_path(session_id: str) -> Path:
    return DB_DIR / session_id / META_FILENAME


def _read_session_meta(session_id: str) -> dict | None:
    """
    Read meta.json for a session. Contains the session title and creation timestamp.
    Args:
        session_id: ID of the session
    Returns:
        A dictionary containing the session metadata, or None if the file is missing or invalid
    """
    path = _meta_path(session_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_session_meta(session_id: str, title: str, created_at: str) -> None:
    """
    Write meta.json for a session.
    Args:
        session_id: ID of the session
        title: Title of the session
        created_at: Creation timestamp of the session
    """
    get_session_dir(session_id)
    path = _meta_path(session_id)
    path.write_text(json.dumps({"title": title, "created_at": created_at}, indent=2))


def list_sessions() -> list[dict]:
    """
    Return list of {id, title, created_at} for all sessions from db/<session_id>/meta.json
    Args:
        None
    Returns:
        A list of dictionaries containing the session metadata for all sessions
    """
    _ensure_db_dir()
    out = []
    dt = datetime.now(timezone.utc).isoformat()
    for path in DB_DIR.iterdir():
        if not path.is_dir():
            continue
        session_id = path.name
        meta = _read_session_meta(session_id)
        if meta:
            title = meta.get("title") or f"Session {session_id[:8]}"
            created_at = meta.get("created_at") or dt
        else:
            title = f"Session {session_id[:8]}"
            created_at = dt
        out.append({"id": session_id, "title": title, "created_at": created_at})
    out.sort(key=lambda s: s["created_at"], reverse=True)
    return out


def create_session(title: str | None = None) -> dict:
    """
    Create a new session (folder + meta.json); returns {id, title, created_at}.
    Args:
        title: Title of the session
    Returns:
        A dictionary containing the session metadata
    """
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    title = title or f"Session {session_id[:8]}"
    _write_session_meta(session_id, title, now)
    save_plan(session_id, [])  # ensure plan.json exists from the start
    return {"id": session_id, "title": title, "created_at": now}


def delete_session(session_id: str) -> bool:
    """
    Remove session folder.
    Args:
        session_id: ID of the session
    Returns:
        True if the folder existed and was deleted, False otherwise
    """
    path = DB_DIR / session_id
    if not path.exists() or not path.is_dir():
        return False
    shutil.rmtree(path)
    return True


def load_history(session_id: str) -> list[dict]:
    """
    Load chat history for a session (list of {role, content}).
    Args:
        session_id: ID of the session
    Returns:
        A list of dictionaries containing the chat history for the session
    """
    path = DB_DIR / session_id / "history.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def save_history(session_id: str, history: list[dict]) -> None:
    """
    Save chat history for a session.
    Args:
        session_id: ID of the session
        history: A list of dictionaries containing the chat history for the session
    """
    path = get_session_dir(session_id) / "history.json"
    path.write_text(json.dumps(history, indent=2))


def load_plan(session_id: str) -> list[dict]:
    """
    Load last saved plan (list of {id, task_description, status, result}).
    Args:
        session_id: ID of the session
    Returns:
        A list of dictionaries containing the plan for the session
    """
    path = DB_DIR / session_id / "plan.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def save_plan(session_id: str, plan: list[dict]) -> None:
    """
    Save plan for a session.
    Args:
        session_id: ID of the session
        plan: A list of dictionaries containing the plan for the session
    """
    path = get_session_dir(session_id) / "plan.json"
    path.write_text(json.dumps(plan, indent=2))


SCHEDULE_COMPLETE_FILENAME = "schedule_complete"


def get_schedule_complete(session_id: str) -> bool:
    """
    Return True if the session has a completed schedule (chat should be locked).
    Args:
        session_id: ID of the session
    Returns:
        True if the session has a completed schedule (chat should be locked), False otherwise
    """
    if not session_id:
        return False
    path = DB_DIR / session_id / SCHEDULE_COMPLETE_FILENAME
    return path.exists()


def set_schedule_complete(session_id: str) -> None:
    """
    Mark the session as having a completed schedule (locks chat).
    Args:
        session_id: ID of the session
    """
    path = get_session_dir(session_id) / SCHEDULE_COMPLETE_FILENAME
    path.write_text("")


UPLOAD_STATUS_FILENAME = "upload_status.txt"
LOGS_FILENAME = "logs.txt"


def get_upload_status(session_id: str) -> str:
    """Return the last upload status message for the session (for display on reload)."""
    if not session_id:
        return ""
    path = DB_DIR / session_id / UPLOAD_STATUS_FILENAME
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def set_upload_status(session_id: str, message: str) -> None:
    """Persist the upload status message so it can be shown on reload."""
    if not session_id:
        return
    path = get_session_dir(session_id) / UPLOAD_STATUS_FILENAME
    path.write_text(message or "", encoding="utf-8")


def get_logs(session_id: str) -> str:
    """Return the last run's logs for the session (for display on reload)."""
    if not session_id:
        return ""
    path = DB_DIR / session_id / LOGS_FILENAME
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def save_logs(session_id: str, content: str) -> None:
    """Persist run logs so they can be shown on reload."""
    if not session_id:
        return
    path = get_session_dir(session_id) / LOGS_FILENAME
    path.write_text(content or "", encoding="utf-8")


def save_uploaded_file(session_id: str, source_path: str | Path, filename: str | None = None) -> Path:
    """
    Copy an uploaded file to db/<session_id>/uploaded/<filename>.
    Args:
        session_id: ID of the session
        source_path: Path to the source file
        filename: Name of the file
    Returns:
        Path to the saved file
    """
    source = Path(source_path)
    name = filename or source.name
    dest_dir = get_uploaded_dir(session_id)
    dest = dest_dir / name
    shutil.copy2(source, dest)
    return dest
