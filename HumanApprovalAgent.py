import logging
from States import WorkflowState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HumanApprovalAgent:
    """Agent for handling human feedback and approval"""

    def __init__(self):
        pass

    def request_approval(self, state: WorkflowState) -> WorkflowState:
        """Request human approval for the generated code"""
        logger.info("Human Approval Agent: Requesting human approval")

        print("\n" + "=" * 60)
        print("HUMAN APPROVAL REQUIRED")
        print("=" * 60)
        print(f"Original Query: {state.original_query}")
        print(f"\nGenerated SysML Code:\n{state.code}")
        print(f"\nValidation Result: {state.validation_history}")

        # In a real implementation, this would integrate with a UI/API
        approval = (
            input("\nApprove this code? (approve/reject/feedback): ").lower().strip()
        )

        if approval == "approve":
            state.approval_status = "approved"
            print("Code approved! Adding to knowledge base...")
        elif approval == "reject":
            state.approval_status = "rejected"
            feedback = input("Please provide feedback for improvement: ")
            state.human_feedback = feedback
        else:
            state.approval_status = "feedback"
            feedback = input("Please provide your feedback: ")
            state.human_feedback = feedback

        return state
