from pydantic import BaseModel, Field, field_validator

MAX_SUBTASKS = 3


class Plan(BaseModel):
    """Structured output of the planner: the request split into small subtasks."""

    subtasks: list[str] = Field(description="Between 2 and 3 short, independent subtasks.")

    @field_validator("subtasks")
    @classmethod
    def limit_subtasks(cls, v: list[str]) -> list[str]:
        # Validators are not part of the JSON schema sent to the LLM, so they are a safe place
        # for rules the provider's structured-output mode may not support (min/max items).
        return v[:MAX_SUBTASKS]
