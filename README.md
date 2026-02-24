# Conference Concierge

An AI agent that helps users build a personal conference schedule.

---

## Run instructions

- **Python:** 3.12+
- **Setup:** From the project root, create a venv and install with [uv](https://docs.astral.sh/uv/) (or pip):
  ```bash
  uv sync
  ```
- **Environment:** Create a `.env` file in the project root and set:
  - `OPENAI_API_KEY` – required for agents and RAG (embeddings, rerank).
  - `SERPER_API_KEY` – required for `google_web_search` and `google_places_search`.
- **Run the Gradio UI:**
  ```bash
  uv run gradio_app.py
  ```
  Then open the URL shown in the terminal (e.g. http://127.0.0.1:7860). Create a session, optionally upload a `schedule.json` (pretalx format), and describe your goal (conference + interests); the agent will clarify if needed, then plan and execute.

- **CLI-style single run (no UI):**
  ```bash
  uv run python -m src.app
  ```
  (Uses a one-off conversation and the example flow in `__main__`.)

---

## Repository structure

```
libra_ai_engineering_take_home/
├── gradio_app.py                   # Gradio UI: sessions, chat, file upload, plan view, logs
├── session_store.py                # Session CRUD, file save to db/<session_id>/uploaded/, history/plan load/save
├── src/
│   ├── app.py                      # ConferenceConcierge: run_step loop, checkpoints, agent orchestration
│   ├── state.py                    # AgentState, StateCheckpoint (conversation + plan + generated schedule)
│   ├── prompts.py                  # System prompts for Intake, Planning, Executor agents
│   ├── responses.py                # Pydantic models: IntakeDecision, Task, Plan, PlanDescription
│   ├── guardrails.py               # check_input / check_output for user and assistant messages
│   ├── agents/
│   │   ├── agent.py                # Base Agent (OpenAI client, run), init_logging
│   │   ├── intake_agent.py         # IntakeAgent: clarify vs plan from full interaction_history
│   │   ├── planning_agent.py       # PlanningAgent: query_to_plan → list of task descriptions
│   │   └── executor_agent.py       # ExecutorAgent: tools, submit_task_result, generate_schedule loop
│   ├── tools/                      # Tools invoked by ExecutorAgent
│   │   ├── rag_search.py           # rag_search, get_schedule_overview (session-scoped)
│   │   ├── google_web_search.py    # Serper web search
│   │   └── google_places_search.py # Serper places (venues, restaurants)
│   └── rag/
│       └── schedule_rag.py         # Index pretalx JSON → ChromaDB; rag_query; schedule_overview_from_json
├── db/                             # Runtime data (one dir per session/conversation_id)
│   └── <session_id>/
│       ├── meta.json               # Session title, created_at
│       ├── history.json            # Chat messages (interaction_history)
│       ├── plan.json               # Current plan (tasks) for UI
│       ├── schedule.json           # Uploaded schedule file (if any)
│       ├── schedule_overview.txt   # Generated overview for get_schedule_overview
│       ├── chroma/                 # ChromaDB persistence for RAG (per session)
│       └── checkpoints.pkl         # State checkpoints for transparency / resume
└── examples/
    └── schedule.json               # Example pretalx-style schedule for testing
```

**What’s what**

| Path | Role |
|------|------|
| **gradio_app.py** | Entry point for the UI: create/select sessions, upload files, chat with the agent, see plan and logs. File uploads are saved to `db/<session_id>/uploaded/`; JSON files detected as schedule are indexed via `schedule_rag`. |
| **session_store.py** | Session list/create/delete, load/save `history.json` and `plan.json`, `save_uploaded_file` into session’s `uploaded/` subfolder. No agent logic. |
| **src/app.py** | `ConferenceConcierge`: one `run_step()` per user message—guardrails → append message → intake → planning → execution (per task) → response. Manages `AgentState`, checkpoints, and delegates to agents. |
| **src/state.py** | `AgentState` (conversation_id, query_to_plan, plan, synthesized_schedule, interaction_history, intake fields) and `StateCheckpoint` (step_index, state snapshot, agent name). |
| **src/agents/** | Intake (clarify or plan), Planning (task list), Executor (tools + submit_task_result). Base `Agent` in `agent.py`; no framework, custom prompts and tool wiring. |
| **src/tools/** | RAG and schedule overview are session-scoped; web and places use Serper. Executor gets these as callables with `session_id` injected. |
| **src/rag/schedule_rag.py** | Parses **pretalx-style** schedule JSON only (`schedule.conference.days` / `schedule.days`), builds docs for ChromaDB, writes `schedule_overview.txt`, and provides `rag_query` + `get_schedule_overview_text`. |
| **db/** | All persistent state: session metadata, chat history, plan, uploaded schedule, RAG ChromaDB, and checkpoints. Created at runtime. |

---

## Execution flow

Each user message triggers one **run_step**:

1. **Input guardrail** – User message is checked; if rejected, a safe reply is appended and the step ends.
2. **Append user message** – Message is added to `interaction_history` and state is checkpointed.
3. **Intake** – If `query_to_plan` is empty:
   - **IntakeAgent** runs on full `interaction_history`.
   - Either **clarify**: sets `necessary_details_required` / `optional_details`, appends one assistant clarification message, applies output guardrail, and **returns** (no plan yet).
   - Or **plan**: sets `query_to_plan` (summary for the planner) and continues.
4. **Planning** – **PlanningAgent** runs on `query_to_plan` only; writes `plan_description` (list of task strings). Tasks are instantiated as `Task(id, task_description, status="pending", result="")`.
5. **Execution** – For each **pending** task, in order:
   - Task status → `in_progress`.
   - **ExecutorAgent** loop until task is `completed` or `failed`: receives state + current task, may call tools (RAG, web search, places, `get_schedule_overview`, `generate_schedule`), must call `submit_task_result` to complete. Each run is checkpointed.
6. **Response** – Output guardrail is applied to `synthesized_schedule`; that content is appended as the assistant reply and state is returned.

State is checkpointed after: user input, intake, planning, and each executor run (for transparency and optional persistence/resume).

---

## Agent architecture

- **Base:** `Agent` (in `src/agents/agent.py`) – name, model, system prompt, OpenAI client; subclasses implement `run(state) -> state`.
- **IntakeAgent** – No tools. Reads full `interaction_history`; returns either a clarification (with `necessary_details_required` / `optional_details` and one user-facing message) or a `query_to_plan` summary. Structured output: `IntakeDecision` (action, summary, user_message, necessary_details_required, optional_details).
- **PlanningAgent** – No tools. Input: only `query_to_plan`. Output: `PlanDescription` (list of task description strings). No conversation history.
- **ExecutorAgent** – Has tools (see below). Input per turn: previous tasks’ descriptions + results, current `synthesized_schedule`, current task, and this task’s `execution_history`. Calls tools and must call `submit_task_result` to mark a task complete; may call `generate_schedule` to build/refine the final schedule. Bounded by `MAX_TASK_TURNS` per task.

**State** (`AgentState`): `conversation_id`, `query_to_plan`, `plan_description`, `plan` (list of `Task`), `synthesized_schedule`, `interaction_history`, and intake fields `necessary_details_required` / `optional_details`. Checkpoints (`StateCheckpoint`) store step index, state snapshot, agent name, and metadata after each transition.

---

## How the agent loop works

1. **Intake** – Decides whether the conversation has enough information to plan (conference identity + at least one user interest). If not, returns a single friendly clarification message; if yes, produces a `query_to_plan` summary for the planner.
2. **Planning** – Turns `query_to_plan` into a structured list of task descriptions. Each task is single-purpose and actionable.
3. **Execution** – For each pending task, the executor runs a loop: call tools (schedule RAG, web search, places search, schedule generation), then **submit_task_result** with the full result. Task results and a **synthesized_schedule** are passed to later tasks. The executor can call **generate_schedule** to build or refine the final schedule from gathered information.

**Flow:** User message → Intake → (clarify and return **or** plan → execute all tasks → return state with synthesized schedule).

---

## Tools integrated

| Tool | Purpose |
|------|--------|
| **get_schedule_overview** | Returns the full schedule overview (all sessions: title, time, room, track) for the user’s uploaded conference. Session-scoped via `session_id`. |
| **rag_search** | Semantic search over the user’s uploaded schedule (Chromadb + OpenAI embeddings, optional LLM rerank). Use for topic-specific sessions (e.g. "RAG", "keynotes"). |
| **google_web_search** | Web search via [Serper](https://serper.dev) (e.g. conference info, program details). |
| **google_places_search** | Places/venues/restaurants via Serper (e.g. lunch near venue). |
| **generate_schedule** | LLM generates a personal schedule from completed task results and current generated schedule. Can be called multiple times to refine. |
| **submit_task_result** | Required to mark a task complete; passes the full result string to the next steps. |

RAG runs per session: upload a pretalx-style `schedule.json` in the UI (stored under `db/<session_id>/uploaded/`); it is indexed under `db/<session_id>/` (Chromadb + overview text). The executor injects `session_id` into RAG tools.  
**Env:** `OPENAI_API_KEY` (OpenAI client + embeddings/rerank), `SERPER_API_KEY` (web + places).

---

## Context strategy

- **Intake:** Full `interaction_history` (all user/assistant messages). No truncation in the current implementation; ToDo: for very long chats we would keep the last N turns or summarize older ones.
- **Planning:** Only the current `query_to_plan` string (one summary). No full history for clarity.
- **Executor (per task):** System prompt + one user message containing: previous tasks’ descriptions and results, current `synthesized_schedule`, and current task description. Then that task’s `execution_history` (assistant/tool messages). We do not feed other tasks’ raw tool traffic; only their final results. This caps context per task and avoids prompt bloat.
- **Overflow mitigation:** Executor has a max turns per task (~20). For longer runs we could truncate or summarize old tool results and keep the last K messages per task.

---

## Example walkthrough

You can reproduce this flow in the Gradio UI (or adapt it for the CLI).

1. **Upload a schedule for RAG**  
   In the UI, create a session and upload a pretalx-style `schedule.json` (e.g. from `examples/schedule.json`). This indexes the schedule for the session so the agent can use `get_schedule_overview` and `rag_search`.

2. **Say "Hi there"**  
   Send the message. The agent has no conference or interests yet, so **Intake** will choose to clarify.

3. **Wait for the chat to clarify**  
   The assistant replies with a friendly message asking for the conference (name, year, location) and at least one interest or preference.

4. **Send a clarification**  
   Reply with your full request, for example:

   > I would like to visit PyConDE in Darmstadt 2025. I am flexible and available all day long and do not have food preferences, but I would like to have lunch as close to the venue as possible at a place that has at least 4.5 stars on google. I am most interested in RAG topics at the conference.

   **Intake** will set `query_to_plan`, **Planning** will produce tasks (e.g. get schedule overview, find RAG-related sessions, find lunch near venue, generate personal schedule), and **Execution** will run each task (RAG/search/places tools, then `generate_schedule`). The final assistant message will be your generated schedule.

---

## Evaluation scenarios and what “success” means

1. **Clarification then plan** – User says only “I want a schedule.” **Success:** Intake responds with a friendly message asking for conference (year, name, location) and at least one interest; after the user provides them, the next turn produces a plan and runs execution.
2. **Single-shot full request** – User gives conference + interests + preferences in one message. **Success:** Intake sets `query_to_plan`, planner outputs a list of tasks, executor runs each, uses tools where needed, and returns a generated schedule.
3. **Tool use (schedule)** – User uploads a schedule and asks for sessions on a topic (e.g. “RAG”). **Success:** Executor calls `get_schedule_overview` and/or `rag_search`, uses the results in the task, and submits a result that reflects the schedule content.
4. **Tool use (web + places)** – User asks for lunch near the venue (e.g. “4.5+ stars near conference”). **Success:** Executor calls `google_places_search` (and optionally `google_web_search`), and the task result contains concrete place suggestions.
5. **Multi-task and logging** – Request implies several steps (e.g. get schedule → find talks by topic → find lunch → generate schedule). **Success:** Plan has multiple tasks; all run in order; each task’s result is passed to the next; logs show which agent ran, which task, which tools were called, and task completion (e.g. result length).

---

## ToDo — improvements

- **RAG schedule format** – The pipeline only accepts **pretalx-style** schedule JSON (as shown in `examples`). Other formats (e.g. CSV, pdf, custom JSON, or other) are not parsed and will yield errors **Improvement:** Add adapters or a small format-detection layer (e.g. pretalx vs. CSV vs. iCal) and normalize to a common internal representation before indexing.
- **Long conversation context (Intake)** – Intake uses the full `interaction_history` with no limitation. For very long chats this can bloat prompts. **Improvement:** Keep the last N turns and/or summarize older turns before sending to the intake agent.
- **Executor context overflow** – Executor has a max turns per task (~20); within a task, full tool call history is kept. **Improvement:** Truncate or summarize older tool results and keep the last K messages per task.
- **Tests** – No automated tests in the repo. **Improvement:** Add unit tests and integration tests for the agent loop.
- **Error handling and retries** – External calls (OpenAI, Serper, ChromaDB) could be wrapped with retries and clearer error messages.
- **Persistence / resume** – Checkpoints are saved but there is no "resume from step N" in the UI. **Improvement:** Expose checkpoint replay or resume in the app.

