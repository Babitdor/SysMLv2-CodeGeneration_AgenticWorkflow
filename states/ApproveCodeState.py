from typing import Dict, Any
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict
from .WorkflowState import WorkflowState


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
