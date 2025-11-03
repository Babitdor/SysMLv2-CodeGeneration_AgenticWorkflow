from langchain_ollama import ChatOllama
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from states.WorkflowState import WorkflowState
from agents.QueryAgent import QueryAgent

def test_query_agent():
    """Test main function for QueryAgent"""

    # Test scenarios
    test_queries = [
        "Create a SysML model for a simple car system",
        "Design a battery management system with temperature monitoring",
        "Model a robotic arm with 3 joints and servo motors",
        "Create a requirements model for a smart home system",
    ]

    print("=== QueryAgent Test Suite ===\n")

    # Initialize QueryAgent with mock LLM
    query_agent = QueryAgent(
        llm=ChatOllama(
            model="qwen3:1.7b",
            validate_model_on_init=True,
        )
    )
    # Test each query
    for i, query in enumerate(test_queries, 1):
        print(f"--- Test Case {i} ---")
        print(f"Original Query: {query}")

        # Create state and process
        state = WorkflowState(original_query=query)
        response_state = query_agent.process(state)

        print(
            f"Processed Query Length: {len(response_state.processed_query)} characters"
        )
        print(f"Processed Query Preview:")
        print(f"{response_state.processed_query}")
        print(f"Success: {'Yes' if response_state.processed_query else 'No'}")
        print()

    print("\n=== Test Suite Complete ===")


if __name__ == "__main__":
    test_query_agent()
