from States import WorkflowState
import logging
from langchain_ollama import ChatOllama
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryAgent:
    """Agent responsible for processing and refining user queries"""

    def __init__(self, llm, config_path: str = "./prompt.yaml"):
        self.llm = llm
        self.system_prompt = self._load_system_prompt(config_path)

    def _load_system_prompt(self, config_path: str) -> str:
        """Load system prompt from a YAML config file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"System prompt loaded from {config_path}")
            return config.get("Query-Agent", "")
        except FileNotFoundError:
            logger.warning(
                f"Config file {config_path} not found. Using default prompt."
            )
            return self._get_default_system_prompt()
        except Exception as e:
            logger.warning(f"Error loading config: {e}. Using default prompt.")
            return self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """Provide a default system prompt for SysML code generation."""
        logger.info("Loading default system prompt...")
        return """You are a SysML v2 expert assistant specialized in creating high-quality SysML diagrams and code.
        
        Guidelines:
        - Generate complete, syntactically correct SysML v2 code
        - Use proper SysML v2 syntax and keywords
        - Include appropriate package declarations
        - Use clear, descriptive names for elements
        - Follow SysML v2 best practices and standards
        - Provide complete, compilable code
        - Use comments to explain complex relationships
        - Address all requirements in the query
        - Use appropriate diagram types and elements
        
        Always wrap your SysML code in ```sysml code blocks for easy extraction.
        If this is a correction iteration, focus on fixing the specific errors mentioned."""

    def process(self, state: WorkflowState) -> WorkflowState:
        """Process the original query into a structured prompt"""
        try:
            logger.info("Query Agent: Processing user query")

            messages = [
                ("system", self.system_prompt),
                ("human", state.original_query),
            ]
            response = self.llm.invoke(messages)

            state.processed_query = response.content
            logger.info("Query Agent: Successfully processed query")

        except Exception as e:
            logger.error(f"Query Agent error: {str(e)}")
            state.processed_query = state.original_query

        return state


def main():
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
            model="mistral",
            validate_model_on_init=True,
        )
    )

    print(f"System Prompt Preview (first 100 chars):")
    print(f"{query_agent.system_prompt}...\n")

    # Test each query
    for i, query in enumerate(test_queries, 1):
        print(f"--- Test Case {i} ---")
        print(f"Original Query: {query}")

        # Create state and process
        state = WorkflowState(original_query=query)
        result_state = query_agent.process(state)

        print(f"Processed Query Length: {len(result_state.processed_query)} characters")
        print(f"Processed Query Preview:")
        print(f"{result_state.processed_query}")
        print(f"Success: {'Yes' if result_state.processed_query else 'No'}")
        print()

    print("\n=== Test Suite Complete ===")


if __name__ == "__main__":
    main()
