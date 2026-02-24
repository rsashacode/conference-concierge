import dotenv
dotenv.load_dotenv(override=True)


import json
import logging
import queue
import threading
from pathlib import Path

import gradio as gr

from src.app import ConferenceConcierge
from session_store import (
    list_sessions,
    create_session,
    delete_session,
    load_history,
    save_history,
    load_plan,
    save_plan,
    save_uploaded_file,
    list_uploaded_files,
    get_schedule_complete,
    set_schedule_complete,
    get_upload_status,
    set_upload_status,
    get_logs,
    save_logs,
)
from src.rag.schedule_rag import index_schedule_for_session
from src.agents import init_logging


AGENTS_LOGGER_NAME = "ConferenceConcierge"
AGENT_LOGGER_NAMES = [
    AGENTS_LOGGER_NAME,
    "IntakeAgent",
    "PlanningAgent",
    "ExecutorAgent",
]
init_logging(name=AGENTS_LOGGER_NAME)

class QueueLogHandler(logging.Handler):
    """Forwards log records to a queue as formatted strings for the UI."""

    def __init__(self, q: queue.Queue):
        super().__init__()
        self._queue = q
        self.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s | %(message)s", datefmt="%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._queue.put(self.format(record))
        except Exception:
            pass


def _plan_to_data(plan: list) -> list[list]:
    """Convert state.plan (Task models or dicts) to rows for gr.DataFrame [id, description, status]."""
    if not plan:
        return []
    rows = []
    for t in plan:
        if isinstance(t, dict):
            id_, desc = str(t.get("id", "")), (t.get("task_description") or "")
            status = t.get("status", "")
        else:
            id_ = str(t.id)
            desc = t.task_description or ""
            status = t.status
        rows.append([
            id_,
            desc[:120] + ("..." if len(desc) > 120 else ""),
            status,
        ])
    return rows


def _history_to_chatbot(history: list[dict]) -> list[dict]:
    """Convert interaction_history to Gradio Chatbot format: list of {role, content}."""
    return [{"role": m["role"], "content": m.get("content") or ""} for m in history]


def _chatbot_to_history(chatbot_value) -> list[dict]:
    """Convert Chatbot value (list of dicts or Message-like) to interaction_history."""
    if not chatbot_value:
        return []
    # Handle Gradio ChatbotDataMessages (root list) or plain list
    if hasattr(chatbot_value, "root"):
        messages = chatbot_value.root
    else:
        messages = list(chatbot_value)
    out = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "assistant")
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        if isinstance(content, list):
            # Normalized content: list of {type, text} or similar
            parts = [x.get("text", x) if isinstance(x, dict) else str(x) for x in content]
            content = " ".join(str(p) for p in parts)
        out.append({"role": role, "content": str(content)})
    return out


PLAN_HEADERS = ["ID", "Task", "Status"]


def get_session_choices() -> list[tuple[str, str]]:
    """Return [(display_label, session_id), ...] for dropdown. Label is the session UUID."""
    sessions = list_sessions()
    return [(s["id"], s["id"]) for s in sessions]


def on_load():
    """Initial load: ensure at least one session and return dropdown choices, selected id, chat/plan/logs, upload status, and uploaded files."""
    sessions = list_sessions()
    if not sessions:
        entry = create_session("New session")
        sessions = list_sessions()
    choices = get_session_choices()
    selected = sessions[0]["id"] if sessions else ""
    history = load_history(selected) if selected else []
    plan = load_plan(selected) if selected else []
    chat_locked = get_schedule_complete(selected)
    uploaded_paths = list_uploaded_files(selected) if selected else []
    logs_text = get_logs(selected) if selected else ""
    upload_status_text = get_upload_status(selected) if selected else ""
    return (
        gr.update(choices=choices, value=selected),
        selected,
        _history_to_chatbot(history),
        _plan_to_data(plan),
        logs_text,
        upload_status_text,
        gr.update(interactive=not chat_locked),
        gr.update(value=uploaded_paths),
    )


def on_session_change(session_id: str):
    """When user picks another session: load its history, plan, logs, upload status, and uploaded files list."""
    if not session_id:
        return gr.update(), gr.update(value=[], headers=PLAN_HEADERS), "", "", gr.update(), []
    history = load_history(session_id)
    plan = load_plan(session_id)
    rows = _plan_to_data(plan)
    chat_locked = get_schedule_complete(session_id)
    uploaded_paths = list_uploaded_files(session_id)
    logs_text = get_logs(session_id)
    upload_status_text = get_upload_status(session_id)
    return (
        _history_to_chatbot(history),
        gr.update(value=rows, headers=PLAN_HEADERS) if rows else gr.update(value=[], headers=PLAN_HEADERS),
        logs_text,
        upload_status_text,
        gr.update(interactive=not chat_locked),
        uploaded_paths,
    )


def on_new_session():
    """Create a new session and switch to it."""
    entry = create_session("New session")
    choices = get_session_choices()
    return (
        gr.update(choices=choices, value=entry["id"]),
        entry["id"],
        [],
        gr.update(value=[], headers=PLAN_HEADERS),
        "",
        "",
        gr.update(visible=False),  # session_msg
        gr.update(interactive=True),  # msg_in
        gr.update(value=[]),  # file_upload
    )


def on_delete_session(session_id: str):
    """Delete current session and switch to another."""
    if not session_id:
        return gr.update(), "", [], gr.update(), "", "", "", gr.update(), gr.update(value=[])
    delete_session(session_id)
    sessions = list_sessions()
    choices = get_session_choices()
    new_id = sessions[0]["id"] if sessions else ""
    if not new_id:
        entry = create_session("New session")
        new_id = entry["id"]
        choices = get_session_choices()
    history = load_history(new_id) if new_id else []
    chat_locked = get_schedule_complete(new_id)
    uploaded_paths = list_uploaded_files(new_id) if new_id else []
    logs_text = get_logs(new_id) if new_id else ""
    upload_status_text = get_upload_status(new_id) if new_id else ""
    return (
        gr.update(choices=choices, value=new_id),
        new_id,
        _history_to_chatbot(history),
        gr.update(value=[], headers=PLAN_HEADERS),
        "Session deleted.",
        logs_text,
        upload_status_text,
        gr.update(interactive=not chat_locked),
        gr.update(value=uploaded_paths),
    )


def _is_schedule_file(path: Path) -> bool:
    """Heuristic: JSON file named schedule.json or containing schedule.conference / schedule.days."""
    if path.suffix.lower() != ".json":
        return False
    if path.name.lower() == "schedule.json":
        return True
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        s = data.get("schedule") or data
        return bool(s.get("conference") or s.get("days"))
    except Exception:
        return False


def on_file_upload(files: list | None, session_id: str):
    """Save uploaded file(s) to db/<session_id>/uploaded/. If a schedule JSON is uploaded, index it for RAG.
    Yields immediately to block chat (upload status not ready), then yields again when done.
    Updates the File component with the current list of uploaded files so they persist after refresh."""
    # Block chat immediately while upload is in progress
    yield "Uploading…", gr.update(interactive=False), gr.update()

    if not session_id:
        yield "Select or create a session first.", gr.update(interactive=True), gr.update()
        return
    if not files:
        chat_locked = get_schedule_complete(session_id)
        paths = list_uploaded_files(session_id)
        msg = "No file selected."
        set_upload_status(session_id, msg)
        yield msg, gr.update(interactive=not chat_locked), gr.update(value=paths)
        return
    saved = []
    index_msgs = []
    for f in files:
        path = None
        if isinstance(f, (str, Path)):
            path = Path(f)
        elif hasattr(f, "name") and f.name:
            path = Path(f.name)
        elif getattr(f, "file_path", None):
            path = Path(f.file_path)
        elif isinstance(f, dict) and f.get("name"):
            path = Path(f["name"])
        if path is None or not path.exists():
            continue
        dest = save_uploaded_file(session_id, path, path.name)
        saved.append(path.name)
        if _is_schedule_file(dest):
            msg = index_schedule_for_session(session_id, dest)
            index_msgs.append(msg)
    out = f"Saved: {', '.join(saved)}" if saved else "No files saved."
    if index_msgs:
        out += " " + " ".join(index_msgs)
    set_upload_status(session_id, out)
    chat_locked = get_schedule_complete(session_id)
    paths = list_uploaded_files(session_id)
    yield out, gr.update(interactive=not chat_locked), gr.update(value=paths)


def add_user_message_and_clear(msg_in, chat_history: list, session_id: str):
    """
    Add the user message to the chat and clear the input immediately.
    Returns (updated chatbot value, gr.update to clear msg_in) so the message appears right away.
    """
    if not session_id:
        return chat_history, gr.update(value="")
    message_text = str(msg_in or "").strip()
    if not message_text:
        return chat_history, gr.update(value="")
    history_list = _chatbot_to_history(chat_history)
    new_chat = history_list + [{"role": "user", "content": message_text}]
    return _history_to_chatbot(new_chat), gr.update(value="")


def _no_change():
    """No UI state change for msg_in."""
    return gr.update()


def run_chat(
    chat_history: list,
    session_id: str,
):
    """
    Synchronous generator: run one turn (Concierge run_step), yield (chat_history, plan_rows, logs, msg_in_update)
    for progress updates, then final result. Runs run_step in a thread so we can yield progress.
    Expects chat_history to already include the user message (added by add_user_message_and_clear).
    When schedule is successfully built, locks chat (msg_in becomes non-interactive).
    """
    if not session_id:
        yield chat_history, [], "No session selected.", _no_change()
        return

    history_list = _chatbot_to_history(chat_history)
    message_text = ""
    if history_list and history_list[-1].get("role") == "user":
        message_text = history_list[-1].get("content") or ""
    if not message_text.strip():
        yield chat_history, _plan_to_data(load_plan(session_id) or []), "No message to process.", _no_change()
        return

    existing_logs = get_logs(session_id) or ""
    log_lines = existing_logs.split("\n") if existing_logs else []
    progress_queue: queue.Queue = queue.Queue()
    result_holder: list = []
    exc_holder: list = []
    plan_holder: list = []  # mutable; thread updates this so we can show live plan

    # Stream agent logs (IntakeAgent, PlanningAgent, ExecutorAgent, ConferenceConcierge) to the queue
    queue_handler = QueueLogHandler(progress_queue)
    for logger_name in AGENT_LOGGER_NAMES:
        logging.getLogger(logger_name).addHandler(queue_handler)

    def on_progress(msg: str):
        progress_queue.put(msg)  # appended to log_lines in the loop below

    def _plan_to_disk(plan: list) -> list[dict]:
        """Convert plan (Task objects or dicts) to list of dicts for plan.json."""
        out = []
        for t in plan:
            if isinstance(t, dict):
                out.append({
                    "id": t.get("id", ""),
                    "task_description": t.get("task_description") or "",
                    "status": t.get("status", ""),
                    "result": t.get("result", ""),
                })
            else:
                out.append({
                    "id": t.id,
                    "task_description": t.task_description or "",
                    "status": t.status,
                    "result": getattr(t, "result", "") or "",
                })
        return out

    def on_plan(plan: list):
        plan_holder.clear()
        plan_holder.extend(plan)
        if plan:
            save_plan(session_id, _plan_to_disk(plan))

    def run_step_in_thread():
        try:
            concierge = ConferenceConcierge(conversation_id=session_id)
            saved_history = load_history(session_id)
            concierge.state.interaction_history = saved_history.copy()
            state = concierge.run_step(
                message_text,
                progress_callback=on_progress,
                plan_callback=on_plan,
            )
            result_holder.append(state)
        except Exception as e:
            exc_holder.append(e)
        finally:
            progress_queue.put(None)  # signal done

    try:
        thread = threading.Thread(target=run_step_in_thread, daemon=True)
        thread.start()

        # Yield progress updates while thread runs (on each log line or every ~0.5s)
        while thread.is_alive() or not progress_queue.empty():
            try:
                got = progress_queue.get(timeout=0.5)
                if got is None:
                    break
                log_lines.append(got)
            except queue.Empty:
                pass
            history_list = _chatbot_to_history(chat_history)
            streaming_display = history_list + [
                {"role": "assistant", "content": "..."},
            ]
            # Use live plan from thread if available, else last saved plan
            current_plan = list(plan_holder) if plan_holder else (load_plan(session_id) or [])
            yield streaming_display, _plan_to_data(current_plan), "\n".join(log_lines), _no_change()

        thread.join(timeout=1.0)
    finally:
        for logger_name in AGENT_LOGGER_NAMES:
            logging.getLogger(logger_name).removeHandler(queue_handler)

    if exc_holder:
        e = exc_holder[0]
        history_list = _chatbot_to_history(chat_history)
        err_msg = f"Error: {type(e).__name__}: {e}"
        save_logs(session_id, "\n".join(log_lines))
        yield (
            history_list + [{"role": "assistant", "content": err_msg}],
            _plan_to_data(load_plan(session_id) or []),
            err_msg,
            _no_change(),
        )
        return

    state = result_holder[0]
    final_content = ""
    if state and state.interaction_history and state.interaction_history[-1].get("role") == "assistant":
        final_content = state.interaction_history[-1].get("content", "")

    history_list = _chatbot_to_history(chat_history)
    new_chat = history_list + [
        {"role": "assistant", "content": final_content},
    ]
    save_history(session_id, new_chat)
    save_plan(session_id, _plan_to_disk(state.plan))

    plan_rows = _plan_to_data(state.plan)
    logs_text = "\n".join(log_lines)
    save_logs(session_id, logs_text)
    # Schedule successfully built: lock chat so user cannot send more messages
    if state.synthesized_schedule:
        set_schedule_complete(session_id)
        yield new_chat, plan_rows, logs_text, gr.update(interactive=False)
    else:
        yield new_chat, plan_rows, logs_text, _no_change()


def build_ui():
    layout_css = """
    .gr-form { gap: 0.75rem; }
    /* Full width with minor padding */
    .gradio-container { max-width: 100% !important; width: 100% !important; padding: 1rem 1.25rem !important; box-sizing: border-box !important; }
    /* Main layout fills viewport to avoid page scroll */
    .main-layout { min-height: calc(100vh - 2rem) !important; align-items: stretch !important; }
    .main-layout > div { min-height: inherit !important; }
    .chatbot-wrap { height: min(42vh, 420px) !important; min-height: 200px !important; }
    .chatbot-wrap .chatbot { height: 100% !important; min-height: 180px !important; }
    .logs-wrap textarea { min-height: 20vh !important; max-height: 26vh !important; font-family: monospace; font-size: 0.85em; }
    .todo-table { font-size: 0.9em; }
    """
    with gr.Blocks(title="Conference Concierge", css=layout_css, fill_width=True, fill_height=True) as demo:
        session_id_state = gr.State("")

        with gr.Row(equal_height=False, elem_classes=["main-layout"]):
            # Left sidebar: sessions, file upload, to-do
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Sessions")
                session_dropdown = gr.Dropdown(
                    label="Session",
                    choices=[],
                    value=None,
                    allow_custom_value=False,
                    interactive=True,
                )
                with gr.Row():
                    new_btn = gr.Button("New session")
                    delete_btn = gr.Button("Delete", variant="stop")
                session_msg = gr.Textbox(label="Session message", interactive=False, visible=False)
                gr.Markdown("### Upload file")
                file_upload = gr.File(label="File", file_count="multiple", type="filepath")
                upload_status = gr.Textbox(label="Upload status", interactive=False, visible=True)
                gr.Markdown("### To-do")
                plan_df = gr.Dataframe(
                    value=[],
                    headers=PLAN_HEADERS,
                    datatype=["str", "str", "str"],
                    label="Tasks",
                    interactive=False,
                    wrap=True,
                )

            # Right: chat and logs fill remaining height
            with gr.Column(scale=2, elem_classes=["right-column"]):
                with gr.Group(elem_classes=["chatbot-wrap"]):
                    chatbot = gr.Chatbot(
                        label="Chat",
                        height=400,
                    )
                msg_in = gr.Textbox(
                    placeholder="Enter your conference schedule preferences…",
                    show_label=False,
                    lines=1,
                    max_lines=4,
                )

                gr.Markdown("### Running logs")
                with gr.Group(elem_classes=["logs-wrap"]):
                    logs_box = gr.Textbox(
                        label="Logs",
                        lines=10,
                        max_lines=20,
                        interactive=False,
                    )

        # Load on startup
        demo.load(
            fn=lambda: on_load(),
            outputs=[
                session_dropdown,
                session_id_state,
                chatbot,
                plan_df,
                logs_box,
                upload_status,
                msg_in,
                file_upload,
            ],
        )

        # Session dropdown change: load history, plan, logs, upload status, and uploaded files for selected session
        def _on_session_change(sid):
            chat_hist, plan_rows, logs, upload_status_val, msg_in_up, uploaded_paths = on_session_change(sid)
            return chat_hist, sid, plan_rows, logs, upload_status_val, gr.update(visible=False), msg_in_up, gr.update(value=uploaded_paths)

        session_dropdown.change(
            fn=_on_session_change,
            inputs=[session_dropdown],
            outputs=[chatbot, session_id_state, plan_df, logs_box, upload_status, session_msg, msg_in, file_upload],
        )

        # New session
        new_btn.click(
            fn=on_new_session,
            outputs=[session_dropdown, session_id_state, chatbot, plan_df, logs_box, upload_status, session_msg, msg_in, file_upload],
        ).then(fn=lambda: gr.update(visible=False), outputs=[session_msg])

        # Delete session
        delete_btn.click(
            fn=on_delete_session,
            inputs=[session_id_state],
            outputs=[session_dropdown, session_id_state, chatbot, plan_df, session_msg, logs_box, upload_status, msg_in, file_upload],
        )

        # File upload (saved to session; schedule JSON is indexed for RAG). Blocks chat while uploading.
        file_upload.upload(
            fn=on_file_upload,
            inputs=[file_upload, session_id_state],
            outputs=[upload_status, msg_in, file_upload],
        )

        # Chat submit (Enter or arrow in text field): add user message and clear input, then run backend
        msg_in.submit(
            fn=add_user_message_and_clear,
            inputs=[msg_in, chatbot, session_id_state],
            outputs=[chatbot, msg_in],
        ).then(
            fn=run_chat,
            inputs=[chatbot, session_id_state],
            outputs=[chatbot, plan_df, logs_box, msg_in],
        )

    return demo


def main():
    from dotenv import load_dotenv
    load_dotenv(override=True)

    demo = build_ui()
    demo.queue()
    demo.launch(inbrowser=False)


if __name__ == "__main__":
    main()
