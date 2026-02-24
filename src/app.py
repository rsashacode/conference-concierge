import json
import logging
import uuid
import pickle
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from src.state import AgentState, StateCheckpoint
from src.agents import PlanningAgent, IntakeAgent, ExecutorAgent, init_logging
from src.guardrails import check_input, check_output
from src.prompts import INTAKE_AGENT_PROMPT, PLANNING_AGENT_PROMPT, EXECUTOR_AGENT_PROMPT
from src.responses import Task


AGENTS_LOGGER_NAME = "ConferenceConcierge"
init_logging(name=AGENTS_LOGGER_NAME)
load_dotenv(override=True)

class ConferenceConcierge:
    """
    Conference Concierge that helps users plan their conferences.
    """
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.reload(conversation_id)
    
    def log(self, msg):
        """
        Simple logger for the agents.
        """
        log = logging.getLogger(AGENTS_LOGGER_NAME)
        log.info(msg)
    
    def reload(self, conversation_id: str):
        """
        Reload the conversation.
        """
        self.conversation_id = conversation_id
        self.state = AgentState(conversation_id=self.conversation_id)
        self._step_index = 0
        self.checkpoints: list[StateCheckpoint] = []
        self._init_agents()
        
    def _save_checkpoint(
            self,
            agent_name: str | None = None,
            metadata: dict | None = None,
        ) -> None:
        """
        Record current state as a checkpoint (used after an agent run).
        """
        self.checkpoints.append(
            StateCheckpoint(
                step_index=self._step_index,
                state=self.state.model_copy(deep=True),
                agent_name=agent_name,
                metadata=metadata or {},
            )
        )
        Path(f"db/{self.conversation_id}").mkdir(parents=True, exist_ok=True)
        path = f"db/{self.conversation_id}/checkpoints.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.checkpoints, f)
        self._step_index += 1
            
    def _update_state(
        self,
        fn: Callable[..., Any],
        fn_args: tuple = (),
        fn_kwargs: dict = {},
        agent_name: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Update the state of the conversation.
        """
        self.state = fn(*fn_args, **fn_kwargs)
        self._save_checkpoint(agent_name=agent_name, metadata=metadata)


    def _construct_plan(self):
        """
        Construct the plan for the conversation.
        """
        for task_id, task_description in enumerate(self.state.plan_description):
            self.state.plan.append(
                Task(
                    id=task_id,
                    task_description=task_description,
                    status="pending",
                    result="",
                )
            )

    def _init_agents(self):
        """
        Initialize the agents.
        """
        self.intake_agent = IntakeAgent(
            name="IntakeAgent",
            model="gpt-4o-mini",
            role="IntakeAgent",
            system_prompt=INTAKE_AGENT_PROMPT,
            tools=[],
        )
        self.planning_agent = PlanningAgent(
            name="PlanningAgent",
            model="gpt-4o-mini",
            role="PlanningAgent",
            system_prompt=PLANNING_AGENT_PROMPT,
            tools=[]
        )
        self.executor_agent = ExecutorAgent(
            name="ExecutorAgent",
            model="gpt-4o-mini",
            role="ExecutorAgent",
            system_prompt=EXECUTOR_AGENT_PROMPT,
            tools=[],
        )
    
    def run_step(
        self,
        user_query: str,
        progress_callback: Callable[[str], None] | None = None,
        plan_callback: Callable[[list[Task]], None] | None = None,
    ):
        """
        Run one turn. 
        Optionally call progress_callback(message) and plan_callback(plan) to stream status for Gradio.
        """
        def report(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)

        def report_plan() -> None:
            if plan_callback:
                plan_callback(list(self.state.plan))

        self.log(f"run_step: user_query={user_query[:80] + '...' if len(user_query) > 80 else user_query}")
        
        self.state.interaction_history.append({"role": "user", "content": user_query})
        allowed, reject_msg = check_input(user_query)
        if not allowed:
            self.state.interaction_history.append({"role": "assistant", "content": reject_msg})
            self._save_checkpoint(agent_name=None, metadata={})
            return self.state
        
        self._save_checkpoint(agent_name=None, metadata={})

        while True:
            if not self.state.query_to_plan:
                report("Understanding your request…")
                self.log(f"Invoking {self.intake_agent.name}")
                self._update_state(
                    fn=self.intake_agent.run,
                    fn_args=(self.state,),
                    agent_name=self.intake_agent.name,
                    metadata={"event": "after_intake"},
                )
                last_content = self.state.interaction_history[-1]["content"]
                _safe, check_message = check_output(last_content)
                if not _safe:
                    self.state.interaction_history[-1]["content"] = check_message
                    self._save_checkpoint(agent_name=None, metadata={})
                if not self.state.query_to_plan:
                    return self.state
            else:
                report("Planning your schedule…")
                self.log(f"Invoking {self.planning_agent.name}")
                self._update_state(
                    fn=self.planning_agent.run,
                    fn_args=(self.state,),
                    agent_name=self.planning_agent.name,
                    metadata={"event": "after_planning", "has_plan": bool(self.state.plan)},
                )

                self._construct_plan()
                report_plan()
                report("Searching for sessions and building your schedule…")

                for task in self.state.get_pending_tasks():
                    report(f"Executing task {task.id}: {task.task_description[:80]}...")
                    task.status = "in_progress"
                    report_plan()
                    self.log(f"Invoking {self.executor_agent.name}")
                    
                    while task.status not in ["completed", "failed"]:
                        self._update_state(
                            fn=self.executor_agent.run,
                            fn_args=(self.state, task,),
                            agent_name=self.executor_agent.name,
                            metadata={"event": "after_execution", "task_id": task.id},
                        )
                        report_plan()

                _safe, check_message = check_output(self.state.synthesized_schedule or "")
                content = self.state.synthesized_schedule if _safe else check_message
                self.state.interaction_history.append({"role": "assistant", "content": content})
                return self.state
            

    def get_checkpoints(self) -> list[StateCheckpoint]:
        """
        Return all checkpoints for this conversation (for review, debugging, or persistence)
        """
        return self.checkpoints

    def get_state_at_step(self, step_index: int) -> AgentState | None:
        """
        Return a copy of state at the given step index, or None if no such checkpoint
        """
        for cp in self.checkpoints:
            if cp.step_index == step_index:
                return cp.state.model_copy(deep=True)
        return None

if __name__ == "__main__":
    conversation_id = str(uuid.uuid4())
    app = ConferenceConcierge(conversation_id)
    state = app.run_step("Hi there!")
    state = app.run_step(
        "I would like to visit PyConDE in Darmstadt 2025. \
            I am flexible and available all day long and do not have food preferences, \
                but I would like to have lunch as close to the venue as possible at a place that has at least 4.5 stars on google. \
                    I am most interested in RAG topics at the conference."
        )
    print(state)