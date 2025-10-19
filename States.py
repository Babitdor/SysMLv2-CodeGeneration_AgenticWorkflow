from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict


class ValidationStatus(str, Enum):
    """Enumeration for validation status"""

    VALID = "valid"
    INVALID = "invalid"
    PENDING = "pending"
    ERROR = "error"


class ErrorInfo(BaseModel):
    """Model for error information"""

    name: str = Field(..., description="Error name/type")
    message: str = Field(..., description="Error message")
    traceback: List[str] = Field(default_factory=list, description="Error traceback")
    line_number: Optional[int] = Field(
        None, description="Line number where error occurred"
    )
    column_number: Optional[int] = Field(
        None, description="Column number where error occurred"
    )


class ValidationResult(BaseModel):
    """Model for validation results"""

    success: bool = Field(..., description="Whether validation was successful")
    errors: List[ErrorInfo] = Field(
        default_factory=list, description="List of errors found"
    )
    warnings: List[str] = Field(default_factory=list, description="List of warnings")
    output: str = Field(default="", description="Cleaned output from validation")
    raw_output: str = Field(default="", description="Raw output from validation")
    execution_time: Optional[float] = Field(
        None, description="Execution time in seconds"
    )

    @model_validator(mode="after")
    def success_must_match_errors(self):
        """Ensure success status matches error count"""
        # Only enforce that success=True cannot have errors
        # Allow success=False with no errors (e.g., for kernel failures)
        if self.success and self.errors:
            raise ValueError("Cannot have success=True when errors are present")
        return self


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
        default=ValidationStatus.PENDING, description="Validation status"
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


class SysMLConfig(BaseModel):
    """Configuration for SysML validator"""

    kernel_name: str = Field(default="sysml", description="Jupyter kernel name")
    timeout: int = Field(default=30, gt=0, description="Execution timeout in seconds")
    silent_startup: bool = Field(default=True, description="Suppress startup warnings")
    max_output_length: int = Field(
        default=5000, gt=0, description="Maximum output length to keep"
    )
    suppress_warnings: List[str] = Field(
        default_factory=lambda: [
            "sun.misc.Unsafe",
            "log4j:WARN",
            "Reading D:\\",
            "Reading",
            "Reading /",
            ".kerml...",
            ".sysml...",
        ],
        description="Warning patterns to suppress",
    )


@dataclass
class ApprovedCodeEntry:
    """Data structure for storing approved SysML code entries"""

    id: str
    task: str
    generated_code: str
    human_feedback: str
    validation_info: Dict[str, Any]
    workflow_metadata: Dict[str, Any]
    created_at: str
    embedding_text: str  # Combined text for embedding generation

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_workflow_state(cls, state: WorkflowState) -> "ApprovedCodeEntry":
        """Create an ApprovedCodeEntry from a successful workflow state"""

        # Generate unique ID based on task and code hash
        task_hash = hashlib.md5(state.original_query.encode()).hexdigest()[:8]
        code_hash = hashlib.md5(state.code.encode()).hexdigest()[:8]
        entry_id = f"sysml_{task_hash}_{code_hash}_{int(datetime.now().timestamp())}"

        # Extract validation information
        latest_validation = state.get_latest_validation()
        validation_info = {
            "success": latest_validation.success if latest_validation else False,
            "validation_count": len(state.validation_history),
            "success_rate": state.get_success_rate(),
            "final_errors": [
                {"name": err.name, "message": err.message}
                for err in (latest_validation.errors if latest_validation else [])
            ],
        }

        # Workflow metadata
        workflow_metadata = {
            "iterations_used": state.iteration,
            "max_iterations": state.max_iterations,
            "approval_status": state.approval_status,
            "validation_status": str(state.is_valid),
            "workflow_completed": True,
        }

        # Create embedding text (combination of task, code comments, and key parts)
        embedding_text = cls._create_embedding_text(
            state.original_query, state.code, state.human_feedback
        )

        return cls(
            id=entry_id,
            task=state.original_query,
            generated_code=state.code,
            human_feedback=state.human_feedback,
            validation_info=validation_info,
            workflow_metadata=workflow_metadata,
            created_at=datetime.now().isoformat(),
            embedding_text=embedding_text,
        )

    @staticmethod
    def _create_embedding_text(task: str, code: str, feedback: str) -> str:
        """Create text suitable for embedding generation"""
        # Extract comments and key elements from code
        code_lines = code.split("\n")

        # Get package and main element declarations
        key_elements = []
        for line in code_lines:
            line = line.strip()
            if (
                line.startswith("package ")
                or line.startswith("part ")
                or line.startswith("action ")
                or line.startswith("attribute ")
                or line.startswith("// ")
            ):
                key_elements.append(line)

        # Combine for embedding
        embedding_parts = [
            f"Task: {task}",
            f"Key Elements: {' '.join(key_elements[:10])}",  # Limit to prevent too long text
        ]

        if feedback.strip():
            embedding_parts.append(f"Feedback: {feedback}")

        return " | ".join(embedding_parts)
