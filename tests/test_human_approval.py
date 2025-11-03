import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.HumanApprovalAgent import HumanApprovalAgent
from states.WorkflowState import WorkflowState
from states.ValidationState import ValidationResult


def test_human_approval():
    """Test the HumanApprovalAgent with sample data"""

    # Create a sample WorkflowState
    sample_state = WorkflowState(
        original_query="Create a SysML package for a drone system",
        code="""package DroneSystem {
    part def Drone {
        part battery : Battery;
        part motor : Motor[4];
        part controller : FlightController;
    }
    
    part def Battery {
        attribute capacity : Real;
        attribute voltage : Real;
    }
}""",
    )

    # Initialize the agent
    agent = HumanApprovalAgent()

    # Request approval
    result_state = agent.request_approval(sample_state)

    # Display results
    print("\n" + "=" * 60)
    print("APPROVAL RESULT")
    print("=" * 60)
    print(f"Approval Status: {result_state.approval_status}")
    if result_state.human_feedback:
        print(f"Feedback: {result_state.human_feedback}")
    print("=" * 60)

    return result_state


if __name__ == "__main__":
    # Run the test
    test_human_approval()
