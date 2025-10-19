from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import re
import yaml
import logging
from datetime import datetime
from States import WorkflowState, ValidationStatus, ValidationResult, ErrorInfo

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SysMLAgent:
    """Agent responsible for generating SysML code using Ollama LLM"""

    def __init__(
        self,
        llm,
        config_path: str = "./prompt.yaml",
        knowledge_base=None,
    ):
        """Initialize the SysML Agent with an LLM and optional knowledge base."""
        self.llm = llm
        self.knowledge_base = knowledge_base
        self.system_prompt = self._load_system_prompt(config_path)

    def _load_system_prompt(self, config_path: str) -> str:
        """Load system prompt from a YAML config file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"System prompt loaded from {config_path}")
            return config.get("SysML-Assistant", "")
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

    def extract_sysml_code(self, text: str) -> str:
        """Extract SysMLv2 code from text, handling markdown formatting."""
        # Try to find code in markdown blocks first
        sysml_pattern = r"```sysml\n?(.*?)\n?```"
        match = re.search(sysml_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try generic code blocks
        code_pattern = r"```\n?(.*?)\n?```"
        match = re.search(code_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Return as-is if no code blocks found
        return text.strip()

    def _get_knowledge_context(self, query: str) -> str:
        """Retrieve relevant context from knowledge base"""
        if not self.knowledge_base:
            return "No knowledge base available."

        try:
            docs = self.knowledge_base.search_similar_entries(query, n_results=3)
            context_entries = []
            for doc in docs:
                if hasattr(doc, "page_content"):
                    # Handle Document objects
                    context_entries.append(doc.page_content)
                elif isinstance(doc, dict):
                    # Handle dictionary results
                    # Assuming the content is in a 'content' or 'text' field
                    content = doc.get("content") or doc.get("text") or str(doc)
                    context_entries.append(content)
                else:
                    # Handle any other format by converting to string
                    context_entries.append(str(doc))

            context = "\n\n".join(context_entries)
            logger.info("Retrieved knowledge base context")
            return context
        except Exception as e:
            logger.warning(f"Error retrieving knowledge context: {e}")
            return "No relevant examples found in knowledge base."

    def _create_prompt_with_context(self, state: WorkflowState) -> str:
        """Create a comprehensive prompt including context from previous attempts."""
        prompt_parts = [f"Requirements: {state.processed_query}"]

        # Add knowledge base context
        if self.knowledge_base:
            knowledge_context = self._get_knowledge_context(state.processed_query)
            if (
                knowledge_context
                and knowledge_context != "No knowledge base available."
            ):
                prompt_parts.append(f"\nKnowledge Base Context: {knowledge_context}")

        # Add validation history context if available
        if state.validation_history:
            latest_validation = state.get_latest_validation()
            if latest_validation and not latest_validation.success:
                prompt_parts.append("\nPrevious attempt had errors:")
                for error in latest_validation.errors:
                    error_msg = f"- {error.message}"
                    if error.line_number:
                        error_msg += f" (line {error.line_number})"
                    prompt_parts.append(error_msg)

                if latest_validation.output:
                    prompt_parts.append(
                        f"\nPrevious validation output:\n{latest_validation.output}"
                    )

        # Add human feedback if provided
        if state.human_feedback:
            prompt_parts.append(f"\nAdditional feedback: {state.human_feedback}")

        # Add error context from current state
        if state.error:
            prompt_parts.append(f"\nCurrent errors to fix: {state.error}")

        # Add iteration context
        if state.iteration > 1:
            prompt_parts.append(
                f"\nThis is iteration {state.iteration}. Please fix the issues from previous attempts."
            )

        prompt_text = "\n".join(prompt_parts)
        logger.debug(f"Generated prompt: {prompt_text}")
        return prompt_text

    def generate_code(self, state: WorkflowState) -> WorkflowState:
        """Generate SysML code based on the current state and return updated state."""
        try:
            logger.info(
                f"SysML Agent: Generating SysML code (iteration {state.iteration})"
            )

            # Create contextual prompt
            contextual_prompt = self._create_prompt_with_context(state)

            # Create the chat prompt template
            messages = [
                ("system", self.system_prompt),
                ("human", contextual_prompt),
            ]
            # Get LLM response
            ai_msg = self.llm.invoke(messages)  # type: ignore
            sysml_code = self.extract_sysml_code(ai_msg.content)  # type: ignore

            # Validate that we got some code
            if not sysml_code.strip():
                raise ValueError("LLM did not generate any code")

            # Update state with new code
            state.code = sysml_code
            state.error = ""  # Reset error since we have new code
            state.is_valid = ValidationStatus.PENDING  # Reset validation status

            logger.info("SysML Agent: Successfully generated SysML code")
            return state

        except Exception as e:
            error_msg = f"LLM Generation Error: {str(e)}"
            logger.error(f"SysML Agent error: {error_msg}")

            # Update state with error information
            state.error = error_msg
            state.is_valid = ValidationStatus.ERROR
            return state

    def generate_with_feedback(
        self, state: WorkflowState, feedback: str
    ) -> WorkflowState:
        """Generate code with additional human feedback."""
        logger.info("SysML Agent: Generating code with human feedback")

        # Update the state with the feedback
        state.human_feedback = feedback

        return self.generate_code(state)

    def refine_code(
        self, state: WorkflowState, specific_instructions: str
    ) -> WorkflowState:
        """Refine existing code with specific instructions."""
        if not state.code:
            raise ValueError("Cannot refine code: no existing code in state")

        logger.info("SysML Agent: Refining existing code")

        # Create a refinement query that combines original and refinement instructions
        refinement_query = f"""
        Please refine the following SysML code based on these instructions: {specific_instructions}

        Current SysML code:
        ```sysml
        {state.code}
        ```

        Original requirements: {state.original_query}
        """

        # Create new state for refinement with updated processed query
        refined_state = WorkflowState(
            original_query=state.original_query,
            processed_query=refinement_query,
            code=state.code,
            error=state.error,
            is_valid=state.is_valid,
            iteration=state.iteration,
            validation_history=state.validation_history,
            human_feedback=state.human_feedback,
            approval_status=state.approval_status,
            max_iterations=state.max_iterations,
        )

        return self.generate_code(refined_state)

    def fix_errors(self, state: WorkflowState) -> WorkflowState:
        """Fix errors based on validation results."""
        if not state.validation_history:
            logger.warning("No validation history available for error fixing")
            return state

        latest_validation = state.get_latest_validation()
        if not latest_validation or latest_validation.success:
            logger.info("No errors to fix - latest validation was successful")
            return state

        logger.info(
            f"SysML Agent: Fixing {len(latest_validation.errors)} validation errors"
        )

        # The error context will be automatically included in the prompt generation
        return self.generate_code(state)


def main():
    """Test main function to demonstrate SysML Agent functionality"""
    import time

    print("=" * 60)
    print("SysML Agent Test Suite")
    print("=" * 60)

    try:
        # Initialize the agent
        print("\n1. Initializing SysML Agent...")
        agent = SysMLAgent(
            llm=ChatOllama(
                model="SysML-V2-llama3.1:latest",
                temperature=0.15,
                validate_model_on_init=True,
            )
        )
        print("✅ Agent initialized successfully")

        # Test 1: Basic code generation
        print("\n2. Testing basic code generation...")
        test_state = WorkflowState(
            original_query="Create a simple vehicle system model",
            processed_query="Create a simple vehicle system model with basic components like engine, wheels, and body. Use proper SysML v2 syntax.",
        )

        print(f"   Query: {test_state.processed_query}")
        print("   Generating code...")

        start_time = time.time()
        result_state = agent.generate_code(test_state)
        generation_time = time.time() - start_time

        if result_state.code:
            print("✅ Code generated successfully")
            print(f"   Generation time: {generation_time:.2f}s")
            print(f"   Code length: {len(result_state.code)} characters")
            print(f"   Status: {result_state.is_valid}")
            print("\n   Generated Code Preview:")
            print("   " + "-" * 50)
            # Show first few lines of generated code
            code_lines = result_state.code.split("\n")[:10]
            for line in code_lines:
                print(f"   {line}")
            if len(result_state.code.split("\n")) > 10:
                print("   ... (truncated)")
            print("   " + "-" * 50)
        else:
            print("❌ Code generation failed")
            if result_state.error:
                print(f"   Error: {result_state.error}")

        # Test 2: Code generation with human feedback
        print("\n3. Testing code generation with feedback...")
        feedback = "Change attribute mass to 600 kg"
        print(f"   Feedback: {feedback}")

        start_time = time.time()
        feedback_state = agent.generate_with_feedback(result_state, feedback)
        feedback_time = time.time() - start_time

        if feedback_state.code != result_state.code:
            print("✅ Code updated with feedback")
            print(f"   Generation time: {feedback_time:.2f}s")
            print(f"   New code length: {len(feedback_state.code)} characters")
        else:
            print("⚠️  Code unchanged (might be identical or error occurred)")

        # Test 3: Code refinement
        print("\n4. Testing code refinement...")
        refinement_instructions = "remove the attribute mass from Vehicle"
        print(f"   Instructions: {refinement_instructions}")

        try:
            start_time = time.time()
            refined_state = agent.refine_code(feedback_state, refinement_instructions)
            refinement_time = time.time() - start_time

            print("✅ Code refinement completed")
            print(f"   Refinement time: {refinement_time:.2f}s")
            print(f"   Final code length: {len(refined_state.code)} characters")
        except Exception as e:
            print(f"❌ Refinement failed: {e}")

        # Test 4: Error simulation and handling
        print("\n5. Testing error handling...")

        # Create a state with simulated validation errors
        error_state = WorkflowState(
            original_query="Create a faulty system model",
            processed_query="Create a system model with intentional syntax errors for testing",
            code="attribute mass :> ISQ::mass = 1200.0 [kg]",
            error="Syntax error: missing semicolon",
            is_valid=ValidationStatus.INVALID,
            iteration=2,
        )

        # Add a mock validation result with errors
        mock_validation = ValidationResult(
            success=False,
            errors=[
                ErrorInfo(
                    name="SyntaxError",
                    message="Missing semicolon at end of statement",
                    line_number=5,
                ),  # type: ignore
                ErrorInfo(
                    name="TypeError", message="Invalid type assignment", line_number=8
                ),  # type: ignore
            ],
            output="Compilation failed with 2 errors",
        )  # type: ignore
        error_state.add_validation_result(mock_validation)

        print(f"   Simulated errors: {len(mock_validation.errors)}")
        print("   Attempting error fixes...")

        try:
            start_time = time.time()
            fixed_state = agent.fix_errors(error_state)
            fix_time = time.time() - start_time

            print("✅ Error fixing attempted")
            print(f"   Fix time: {fix_time:.2f}s")
            print(f"   Status: {fixed_state.is_valid}")
        except Exception as e:
            print(f"❌ Error fixing failed: {e}")

        # Test 5: Extract code functionality
        print("\n6. Testing code extraction...")
        test_responses = [
            "Here's your SysML code:\n```sysml\npackage TestPackage;\npart Vehicle;\n```\nThat should work!",
            "```\npackage SimpleTest;\npart Engine;\n```",
            "package DirectCode;\npart Wheel;",
        ]

        for i, response in enumerate(test_responses, 1):
            extracted = agent.extract_sysml_code(response)
            print(
                f"   Test {i}: {'✅ Extracted' if extracted else '❌ Failed'} - Length: {len(extracted)}"
            )

        print("\n" + "=" * 60)
        print("Test Summary:")
        print("- Basic code generation: Completed")
        print("- Feedback integration: Completed")
        print("- Code refinement: Completed")
        print("- Error handling: Completed")
        print("- Code extraction: Completed")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        print("Please check your Ollama installation and model availability.")
        print("Make sure the SysML-V2-llama3.1:latest model is installed.")


if __name__ == "__main__":
    main()
