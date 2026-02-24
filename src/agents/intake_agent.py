from typing import cast

from openai.types.chat import ChatCompletionMessageParam

from src.agents.agent import Agent
from src.state import AgentState
from src.responses import IntakeDecision


class IntakeAgent(Agent):
    """
    Agent that either asks for clarification or produces a summary for the planner.
    """

    def __init__(self, name, model, role, system_prompt, tools):
        super().__init__(name, model, role, system_prompt, tools)

    def run(self, state: AgentState) -> AgentState:
        self.log(f"{self.name}: intake (history length={len(state.interaction_history)})")
        messages = cast(
            list[ChatCompletionMessageParam],
            [{"role": "system", "content": self.system_prompt}] + state.interaction_history,
        )
        response = self.client.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=IntakeDecision,
        )
        result = response.choices[0].message
        if result.refusal:
            raise RuntimeError(result.refusal)
        parsed = result.parsed
        if not parsed:
            raise RuntimeError("Failed to parse intake response")

        if parsed.action == "clarify":
            state.necessary_details_required = parsed.necessary_details_required or ["need_more"]
            state.optional_details = parsed.optional_details or []
            if parsed.user_message:
                state.interaction_history.append({"role": "assistant", "content": parsed.user_message})
            self.log(f"{self.name}: action=clarify, necessary_details_required={state.necessary_details_required}, optional_details={state.optional_details}")
        else:
            state.necessary_details_required = []
            state.query_to_plan = parsed.summary or ""
            self.log(f"{self.name}: action=plan, query_to_plan length={len(state.query_to_plan)}")
        return state
