from src.agents.agent import Agent
from src.state import AgentState
from src.responses import PlanDescription


class PlanningAgent(Agent):
    """
    Agent that plans the execution of the task.
    """
    def __init__(self, name, model, role, system_prompt, tools):
        super().__init__(name, model, role, system_prompt, tools)
        
    def run(self, state: AgentState) -> AgentState:
        self.log(f"{self.name}: generating plan from query_to_plan (length={len(state.query_to_plan)} chars)")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": state.query_to_plan},
        ]
        response = self.client.chat.completions.parse(
            model=self.model, 
            messages=messages, 
            response_format=PlanDescription,
        )
        
        result = response.choices[0].message
        if result.refusal:
            raise RuntimeError(result.refusal)
        state.plan_description = result.parsed.plan_description if result.parsed else []
        self.log(f"{self.name}: plan generated ({len(state.plan_description)} tasks)")
        return state
