from typing import List
from langchain_ollama import ChatOllama
from dataclasses import dataclass
from langchain.prompts import ChatPromptTemplate
import logging
import yaml
from States import WorkflowState


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeCorrectionAgent:
    """Agent responsible for analyzing errors and providing correction guidance"""

    def __init__(
        self,
        llm,
        error_solutions_db=None,
        config_path: str = "./prompt.yaml",
    ):
        self.llm = llm
        self.error_solutions_db = error_solutions_db
        self.system_prompt = self._load_system_prompt(config_path)
        self.prompt = """
        These are the following: 
        
        Current Errors: {errors}
        
        SysML Code with Issues: {sysml_code}
        
        Error Solutions Database Context: {solution_context}
        
        Analyze each error and provide:
        1. Root cause analysis
        2. Specific correction steps
        3. Code examples for fixes
        4. Prevention strategies
        
        Focus on actionable, specific guidance that the SysML Agent can implement directly.
        
        Correction Guidance:
        """

    def _load_system_prompt(self, config_path: str) -> str:
        """Load system prompt from a YAML config file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"System prompt loaded from {config_path}")
            return config.get("Code-Correction-Agent", "")
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

    def analyze_errors(self, state: WorkflowState) -> WorkflowState:
        """Analyze errors and provide correction guidance"""
        try:
            logger.info("Code Correction Agent: Analyzing errors")

            # Get relevant solutions from error database
            # solution_context = self._get_error_solutions(state.error)  # type: ignore

            messages = [
                ("system", self.system_prompt),
                (
                    "human",
                    self.prompt.format(
                        errors="; ".join(state.error),
                        sysml_code=state.code,
                        solution_context="No Error Solution Context",
                    ),
                ),
            ]
            # solution_context=solution_context,

            response = self.llm.invoke(messages)

            # Store correction guidance for the SysML agent
            state.processed_query += f"\n\nCORRECTION GUIDANCE:\n{response.content}"

            logger.info("Code Correction Agent: Error analysis completed")

        except Exception as e:
            logger.error(f"Code Correction Agent error: {str(e)}")

        return state

    def _get_error_solutions(self, errors: List[str]) -> str:
        """Retrieve relevant error solutions from database"""
        try:
            # Create a search query from errors
            error_query = " ".join(errors)
            docs = self.error_solutions_db.similarity_search(error_query, k=3)  # type: ignore
            return "\n\n".join([doc.page_content for doc in docs])
        except:
            return "No specific error solutions found in database."


def test_analyze_errors():
    """Test the analyze_errors method with sample data"""
    print("=== Testing CodeCorrectionAgent.analyze_errors ===\n")

    # Create agent instance
    agent = CodeCorrectionAgent(
        llm=ChatOllama(
            model="SysML-V2-llama3.1:latest",
            temperature=0.15,
            validate_model_on_init=True,
        )
    )

    # Create test state with sample errors and code
    test_state = WorkflowState(
        error="Missing semicolon after part definition",
        code="""package TestSystem {
    part engine : Engine
    part transmission : Transmission;
}""",
        processed_query="Original query: Create a simple automotive system",
    )

    print("Initial State:")
    print(f"Errors: {test_state.error}")
    print(f"Code:\n{test_state.code}")
    print(f"Processed Query: {test_state.processed_query}")
    print("\n" + "=" * 50 + "\n")

    # Test the analyze_errors method
    result_state = agent.analyze_errors(test_state)

    print("Result State:")
    print(f"Errors: {result_state.error}")
    print(f"Code:\n{result_state.code}")
    print(f"Updated Processed Query:\n{result_state.processed_query}")
    print("\n" + "=" * 50 + "\n")

    # Verify that correction guidance was added
    assert (
        "CORRECTION GUIDANCE:" in result_state.processed_query
    ), "Correction guidance should be added"
    print("✓ Correction guidance was added to processed_query")

    # Verify the guidance content
    assert (
        "Root Cause:" in result_state.processed_query
    ), "Should contain root cause analysis"
    assert "Fix Steps:" in result_state.processed_query, "Should contain fix steps"
    print("✓ Correction guidance contains expected elements")

    print("\n=== Test Completed Successfully ===")



if __name__ == "__main__":
    # Configure logging for test output
    logging.basicConfig(level=logging.INFO)

    # Run main test
    test_analyze_errors()

    print("\n🎉 All tests completed successfully!")
