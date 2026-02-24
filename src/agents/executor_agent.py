import json
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam

from src.agents.agent import Agent
from src.state import AgentState
from src.tools import google_web_search, google_places_search, rag_search
from src.tools.rag_search import get_schedule_overview
from src.tools.google_web_search import declaration as google_web_search_declaration
from src.tools.google_places_search import declaration as google_places_search_declaration
from src.tools.rag_search import declaration as rag_search_declaration
from src.tools.rag_search import get_schedule_overview_declaration
from src.responses import Task
from src.prompts import SYNTHESIZER_PROMPT

TOOL_REGISTRY = {
    "google_web_search": {
        "function": google_web_search,
        "declaration": google_web_search_declaration,
    },
    "google_places_search": {
        "function": google_places_search,
        "declaration": google_places_search_declaration,
    },
    "rag_search": {
        "function": rag_search,
        "declaration": rag_search_declaration,
    },
    "get_schedule_overview": {
        "function": get_schedule_overview,
        "declaration": get_schedule_overview_declaration,
    },
}

SUBMIT_TASK_RESULT_DECLARATION = {
    "type": "function",
    "function": {
        "name": "submit_task_result",
        "description": "Call this when the task is fully complete. \
            Pass the complete result so it can be used by later steps. Do not use plain text to finish; you must call this tool.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "result": {"type": "string", "description": "The full task result (all gathered information)."},
            },
            "required": ["result"],
            "additionalProperties": False,
        },
    },
}

GENERATE_SCHEDULE_DECLARATION = {
    "type": "function",
    "function": {
        "name": "generate_schedule",
        "description": "Generate schedule from the gathered information so far into a personal schedule for the user. \
            Can be called multiple times to refine the schedule.",
    },
}

EXECUTOR_TOOLS = [
    TOOL_REGISTRY["google_web_search"]["declaration"],
    TOOL_REGISTRY["google_places_search"]["declaration"],
    TOOL_REGISTRY["rag_search"]["declaration"],
    TOOL_REGISTRY["get_schedule_overview"]["declaration"],
    SUBMIT_TASK_RESULT_DECLARATION,
    GENERATE_SCHEDULE_DECLARATION,
]

MAX_TASK_TURNS = 20


def _format_message(msg) -> dict:
    out = {"role": msg.role, "content": msg.content if msg.content else ""}
    if not msg.tool_calls:
        return out
    tool_calls_param = []
    for tc in msg.tool_calls:
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        tool_calls_param.append(
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": getattr(fn, "name", ""), 
                    "arguments": getattr(fn, "arguments", "{}")
                },
            }
        )
    out["tool_calls"] = tool_calls_param
    return out


def _build_user_content(state: AgentState, task: Task) -> str:
    completed = state.get_completed_tasks()
    lines = [f"Task {t.id}: {t.task_description}: {t.result}" for t in completed]
    return (
        f"Previous tasks descriptions and their results:\n"
        f"{'\n'.join(lines)}\n"
        f"Synthesized schedule so far: {state.synthesized_schedule}\n"
        f"Current task: {task.task_description}\n"
    )


class ExecutorAgent(Agent):
    """
    Agent that executes the task using the available tools.
    """
    def __init__(self, name, model, role, system_prompt, tools):
        super().__init__(name, model, role, system_prompt, tools)

    def _executor_messages(self, state: AgentState, task: Task) -> list[ChatCompletionMessageParam]:
        user_content = _build_user_content(state, task)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ] + task.execution_history

    def _execute_tool_check_completed(
        self,
        state: AgentState,
        task: Task,
        tool_name: str,
        tool_args: dict,
        tool_call_id: str,
    ) -> bool:
        if tool_name == "submit_task_result":
            task.result = (tool_args.get("result") or "").strip()
            task.status = "completed"
            self.log(f"{self.name}: task {task.id} completed; result length={len(task.result)}")
            return True

        if tool_name == "generate_schedule":
            synth_content = (
                "\n".join([f"Task {t.id}: {t.task_description}: {t.result}" for t in state.get_completed_tasks()])
                + "\n"
                + f"Synthesized schedule so far: {state.synthesized_schedule}\n"
                + f"Current task execution history: {task.execution_history}"
            )
            synth_messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": SYNTHESIZER_PROMPT},
                {"role": "user", "content": synth_content},
            ]
            synth_response = self.client.chat.completions.create(
                model=self.model,
                messages=synth_messages,
            )
            state.synthesized_schedule = synth_response.choices[0].message.content or ""
            result = state.synthesized_schedule
        else:
            if tool_name in ("rag_search", "get_schedule_overview"):
                tool_args = {**tool_args, "session_id": state.conversation_id}
            self.log(f"{self.name}: calling tool {tool_name} with args {str(tool_args)[:100]}...")
            result = TOOL_REGISTRY[tool_name]["function"](**tool_args)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)

        task.execution_history.append({"role": "tool", "content": result, "tool_call_id": tool_call_id})
        return False

    def _handle_no_tool_response(self, response, task: Task) -> None:
        content = response.choices[0].message.content
        if content:
            task.execution_history.append({"role": "assistant", "content": content})

    def run(self, state: AgentState, task: Task) -> AgentState:
        self.log(f"{self.name}: executing task {task.id}: {task.task_description[:80]}...")

        errors_count = 0
        task_completed = False
        turn = 0

        while True:
            turn += 1
            if turn > MAX_TASK_TURNS:
                self.log(f"{self.name}: task {task.id} hit max turns ({MAX_TASK_TURNS}), marking failed")
                task.status = "failed"
                break

            self.log(f"{self.name}: thinking about task {task.id} (turn {turn})")
            exec_messages = self._executor_messages(state, task)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=exec_messages,
                tools=EXECUTOR_TOOLS,
            )
            msg = response.choices[0].message
            task.execution_history.append(_format_message(msg))

            if msg.tool_calls:
                for tool in msg.tool_calls:
                    if task_completed:
                        break
                    fn = getattr(tool, "function", None)
                    if fn is None:
                        continue
                    tool_name = getattr(fn, "name", None) or ""
                    tool_args_str = getattr(fn, "arguments", None) or "{}"
                    try:
                        tool_args = json.loads(tool_args_str)
                        task_completed = self._execute_tool_check_completed(state, task, tool_name, tool_args, tool.id)
                    except Exception as e:
                        errors_count += 1
                        if errors_count >= 5:
                            raise RuntimeError(f"Error calling tools: {e}")
                        self.log(f"{self.name}: error calling tool {tool_name}: {e}")
                        task.execution_history.append(
                            {"role": "tool", "content": f"Error: {e}", "tool_call_id": tool.id}
                        )
            else:
                self.log(f"{self.name}: no tool in response")
                self._handle_no_tool_response(response, task)

            if task_completed:
                break

        return state
