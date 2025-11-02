from pydantic import BaseModel, Field, field_validator
from typing import List


class SysMLResponse(BaseModel):
    """Structured output for SysML code generation."""

    code: str = Field(description="Complete validated SysML v2 code")
    validated: bool = Field(description="Whether the code passed validation")
    syntax_checks_performed: List[str] = Field(default_factory=list)

    @field_validator("code")
    @classmethod
    def code_not_empty(cls, v: str):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Code cannot be empty")
        return v.strip()
