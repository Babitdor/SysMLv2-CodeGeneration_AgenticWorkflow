"""Named agent tools for the SysML pipeline.

The frontend trace speaks in tool names. ``generate_sysml`` wraps the deep
agent (which owns planning, the syntax reference and its own syntax repair);
``validate_sysml`` is the deterministic validator from ``webapp.validation``.
Both are exposed as LangChain tools so they can be bound to a model later, and
the pipeline can call the same objects directly.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from .validation import ValidationReport, report_detail, validate_sysml


class GenerateSysMLArgs(BaseModel):
    requirements: str = Field(description="What the SysML v2 model must contain.")
    previous_code: str = Field(
        default="", description="Existing model to refine, if any."
    )
    validation_errors: str = Field(
        default="", description="Parser findings the new model must fix."
    )


class GenerateSysMLTool(BaseTool):
    """Draft or repair SysML v2 text with the project's deep agent."""

    name: str = "generate_sysml"
    description: str = (
        "Generate or repair a complete SysML v2 textual model. Use the user "
        "requirements, the current model and any validation errors. Returns "
        "the complete model."
    )
    args_schema: type[BaseModel] = GenerateSysMLArgs
    agent: Any = Field(exclude=True)

    def stream(self, prompt: str, thread_id: Optional[str] = None) -> str:
        """Run one generation pass and return the final extracted code."""
        code = ""
        errors = []
        for event in self.agent.stream_code(prompt, thread_id):
            kind = event.get("type")
            if kind == "code":
                candidate = (event.get("code") or "").strip()
                if candidate:
                    code = candidate
            elif kind == "error":
                errors.append(str(event.get("detail") or "unknown error"))
        if not code:
            raise RuntimeError(
                errors[-1] if errors else "The agent returned no SysML code."
            )
        return code

    def _run(
        self,
        requirements: str,
        previous_code: str = "",
        validation_errors: str = "",
    ) -> str:
        parts = [
            "Create or repair a complete SysML v2 model.",
            "USER REQUIREMENTS:\n" + requirements.strip(),
        ]
        if previous_code.strip():
            parts.append("CURRENT MODEL:\n```sysml\n" + previous_code.strip() + "\n```")
        if validation_errors.strip():
            parts.append("VALIDATION ERRORS:\n" + validation_errors.strip())
        return self.stream("\n\n".join(parts))


class ValidateSysMLArgs(BaseModel):
    code: str = Field(description="SysML v2 text to validate.")


class ValidateSysMLTool(BaseTool):
    """Run the modular validator and return a structured report."""

    name: str = "validate_sysml"
    description: str = (
        "Validate SysML v2 syntax and structure. Returns line/column findings "
        "and whether the model parsed."
    )
    args_schema: type[BaseModel] = ValidateSysMLArgs

    def report(self, code: str) -> ValidationReport:
        return validate_sysml(code)

    def _run(self, code: str) -> str:
        return report_detail(self.report(code))


__all__ = [
    "GenerateSysMLArgs",
    "GenerateSysMLTool",
    "ValidateSysMLArgs",
    "ValidateSysMLTool",
]
