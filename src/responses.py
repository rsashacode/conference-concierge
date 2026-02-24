from pydantic import BaseModel, Field
from typing import Any, Literal, Optional


class IntakeDecision(BaseModel):
    action: Literal["clarify", "plan"] = Field(
        description="Either 'clarify' (need more from user) or 'plan' (ready to plan)."
    )
    necessary_details_required: list[str] = Field(
        default_factory=list,
        description="When action is 'clarify', list of necessary details still missing."
    )
    optional_details: list[str] = Field(
        default_factory=list,
        description="When action is 'clarify', list of optional details that are not necessary but can be helpful for the planning agent to build a personal schedule."
    )
    user_message: Optional[str] = Field(
        default=None,
        description="When action is 'clarify', the friendly message to show the user."
    )
    summary: Optional[str] = Field(
        default=None,
        description="When action is 'plan', the concise summary for the planning agent."
    )


class Task(BaseModel):
    id: int
    task_description: str = Field(
        description="A description of the task to be executed."
    )
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(
        default="pending", 
        description="The status of the task."
    )
    execution_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="The history of the task execution (API-format messages: assistant with optional tool_calls, tool)."
    )
    result: str = Field(
        default="",
        description="Exact answer to the task: the concrete output (e.g. schedule, list) and where it came from (sources/URLs).",
    )


class Plan(BaseModel):
    plan: list[Task] = Field(
        default_factory=list,
        description="A list of tasks to be executed to build a personal conference schedule."
    )


class PlanDescription(BaseModel):
    plan_description: list[str] = Field(
        default_factory=list,
        description="A list tasks to be executed to build a personal conference schedule."
    )