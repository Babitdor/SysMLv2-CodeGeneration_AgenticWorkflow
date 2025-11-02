from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator
from .ValidationState import ValidationStatus, ValidationResult
# from .SysMLResponse import SysMLResponse


class WorkflowState(BaseModel):
    """State management for the workflow"""

    original_query: str = Field(
        default="", description="Description of the original query"
    )
    processed_query: str = Field(
        default="", description="Description of the processed query"
    )
    code: str = Field(default="", description="The SysML code content")
    error: str = Field(default="", description="Latest error message")
    is_valid: ValidationStatus = Field(
        default=ValidationStatus.PENDING,
        description="Whether the code passed validation",
    )
    iteration: int = Field(default=1, ge=1, description="Current iteration number")
    validation_history: List[ValidationResult] = Field(
        default_factory=list, description="History of validation attempts"
    )
    human_feedback: str = Field(
        default="", description="Human feedback for improvements"
    )
    approval_status: str = Field(
        default="", description="Approval status of code (pending, approved, rejected)"
    )
    max_iterations: int = 5
    # metadata: Optional[SysMLResponse] = Field(
    #     default=None, description="Structured metadata from code generation"
    # )

    @field_validator("code")
    @classmethod
    def code_validation(cls, v):
        """Validate code - allow empty for initial states, but warn"""
        # Allow empty code for initial states (iteration 1) or when explicitly set to empty
        if not v.strip():
            # Only require non-empty code after the first iteration
            # This allows initial states to have empty code
            return v
        return v

    @field_validator("original_query")
    @classmethod
    def original_query_must_not_be_empty(cls, v):
        """Ensure original query is not empty"""
        if not v.strip():
            raise ValueError("Original query cannot be empty")
        return v

    @field_validator("processed_query")
    @classmethod
    def processed_query_must_not_be_empty(cls, v):
        """Ensure processed query is not empty"""
        if not v.strip():
            raise ValueError("Processed query cannot be empty")
        return v

    def add_validation_result(self, result: ValidationResult):
        """Add a validation result to history"""
        self.validation_history.append(result)
        self.is_valid = (
            ValidationStatus.VALID if result.success else ValidationStatus.INVALID
        )
        if not result.success and result.errors:
            self.error = result.errors[0].message
        else:
            self.error = ""
        self.iteration += 1

    def get_latest_validation(self) -> Optional[ValidationResult]:
        """Get the most recent validation result"""
        return self.validation_history[-1] if self.validation_history else None

    def get_success_rate(self) -> float:
        """Calculate success rate of validations"""
        if not self.validation_history:
            return 0.0
        successful = sum(1 for result in self.validation_history if result.success)
        return successful / len(self.validation_history)

    def is_ready_for_validation(self) -> bool:
        """Check if the state has code ready for validation"""
        return (
            bool(self.code.strip())
            and self.code.strip() != "// Initial placeholder - will be generated"
        )

    def needs_code_generation(self) -> bool:
        """Check if the state needs code generation"""
        return (
            not self.code.strip()
            or self.code.strip() == "// Initial placeholder - will be generated"
        )

    @classmethod
    def from_state(
        cls, state: Union["WorkflowState", Dict[str, Any]]
    ) -> "WorkflowState":
        """Safely create a WorkflowState from either a dict or existing WorkflowState"""
        if isinstance(state, WorkflowState):
            return state
        if isinstance(state, dict):
            return cls(**state)
        raise TypeError(f"Cannot create WorkflowState from type {type(state)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert WorkflowState to a dictionary"""
        return self.model_dump()
