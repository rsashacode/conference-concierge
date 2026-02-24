from datetime import datetime, timezone
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from src.responses import Task


class AgentState(BaseModel):
    conversation_id: str
    
    necessary_details_required: list[str] = []
    optional_details: list[str] = []
    
    query_to_plan: str = ""
    plan_description: list[str] = Field(default_factory=list)
    plan: list[Task] = Field(default_factory=list)
    
    synthesized_schedule: str = ""
    final_result: str = ""

    interaction_history: list[dict[Literal["role", "content"], str]] = Field(default_factory=list)
    
    
    def get_pending_tasks(self) -> list[Task]:
        return [task for task in self.plan if task.status == "pending"]
    
    def get_in_progress_tasks(self) -> list[Task]:
        return [task for task in self.plan if task.status == "in_progress"]
    
    def get_completed_tasks(self) -> list[Task]:
        return [task for task in self.plan if task.status == "completed"]
    
    def get_failed_tasks(self) -> list[Task]:
        return [task for task in self.plan if task.status == "failed"]


class StateCheckpoint(BaseModel):
    step_index: int = Field(description="Monotonically increasing step number")
    state: AgentState = Field(description="Deep copy of state at this step")
    agent_name: Optional[str] = Field(default=None, description="Name of agent that produced this state, or None for user input")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional extra context (e.g. reason for transition)")