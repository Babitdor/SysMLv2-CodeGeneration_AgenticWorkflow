from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field, model_validator


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
