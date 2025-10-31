from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import re
import yaml
import logging
import os
import sys
from dotenv import load_dotenv
from typing import List, Optional
from langchain.tools import BaseTool


load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from states.WorkflowState import WorkflowState
from states.ValidationState import ValidationStatus, ValidationResult, ErrorInfo
from tools.SysMLValidatorTool import sysml_validator_tool
from tools.SysMLSyntaxTool import syntax_tool

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SysMLAgent:
    """Agent responsible for generating SysML code"""

    def __init__(
        self,
        llm,
        config_path: str = "agents/prompt/prompt.yaml",
        knowledge_base=None,
        tools: Optional[List[BaseTool]] = None,
    ) -> None:
        self.name = "SysMLAgent"
        self.llm = llm
        self.tools = tools
        self.knowledge_base = knowledge_base
        self.system_prompt = self._load_system_prompt(config_path)

    def _load_system_prompt(self, config_path: str) -> str:
        """Load system prompt from a YAML config file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"System prompt loaded from {config_path}")
            prompt = config.get("SysML-Agent", "")
            if not prompt:
                raise ValueError(
                    f"'SysML-Agent' key not found or empty in {config_path}."
                )
            return prompt
        except FileNotFoundError:
            raise FileNotFoundError(
                f"❌ Config file {config_path} not found. Please provide a valid prompt.yaml."
            )
        except yaml.YAMLError as e:
            raise ValueError(f"❌ YAML parsing error in {config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"❌ Unexpected error loading system prompt: {e}")

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
            docs = self.knowledge_base.similarity_search(query, k=3)
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
            formatted_context = f"These are relevant approved solutions:\n\n{context}"
            logger.info("Retrieved knowledge base context")
            return formatted_context
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

            ##### Chaining Code
            # prompt = ChatPromptTemplate.from_messages(
            #     [("system", self.system_prompt), ("human", "{context}")]
            # )
            # chain = prompt | self.llm | StrOutputParser()
            # Create the chat prompt template
            # response = chain.invoke({"context": context})  # type: ignore

            # Create contextual prompt
            context = self._create_prompt_with_context(state)

            agent = create_agent(
                name="SysML-Agent",
                model=self.llm,
                tools=[sysml_validator_tool, syntax_tool],
                # tools=self.tools,
                system_prompt=self.system_prompt,
            )

            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": context,
                        }
                    ]
                },
                {"recursion_limit": 100},
            )
            if isinstance(response, dict):
                if "messages" in response:
                    response = response["messages"][-1].content
                elif "output" in response:
                    response = response["output"]
                else:
                    # Fallback: convert to string
                    response = str(response)
            else:
                response = str(response)

            sysml_code = self.extract_sysml_code(response)  # type: ignore

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
                model="qwen3-coder:480b-cloud",
                temperature=0.15,
                num_ctx=16000,
                # validate_model_on_init=True,
            )
        )
        print("✅ Agent initialized successfully")

        # Test 1: Basic code generation
        print("\n2. Testing basic code generation...")
        test_state = WorkflowState(
            original_query="Create a simple vehicle system model",
            processed_query="""**## System Analysis**  
- **System Type:** Cyber‑physical (embedded control + human interface)  
- **Domain:** Building Automation (smart‑home)  
- **Complexity Level:** Moderate (≈ 8 – 12 part defs, 5–7 requirements, a few constraints)  
- **Primary Focus:** Requirements & Constraints (with minimal behavioral context)

---

#### **## Elements to Include**

**### Structural Elements**  
- **SmartHome:** top‑level part holding all subsystems  
  - *SysML v2 Construct:* `part def`  
  - *Rationale:* Provides a scoped owner for all requirements.  
- **Device:** generic sensor/actuator container  
  - *Construct:* `part def` – reusable for IoT devices.  
- **Gateway:** bridge between local network and cloud  
  - *Construct:* `part def` – captures communication links.  
- **ControlCenter:** logical engine (controller/logic)  
  - *Construct:* `part def` – central processing point.  

**### Behavioral Elements**  
- **UserInteraction:** high‑level user command action  
  - *Construct:* `action def` – describes user‑initiated events.  
- **SystemResponse:** automatic controller reaction  
  - *Construct:* `action def` – required to tie behavior to requirements.  

**### Requirements & Constraints**  
- **ReqSecurity** – “All wireless links must use end‑to‑end TLS encryption.”  
  - *Construct:* `requirement def` – verifiable security property.  
- **ReqEnergyEfficiency** – “The system shall consume ≤ 15 W in idle.”  
  - *Construct:* `requirement def`.  
- **ReqReliability** – “Device uptime ≥ 99.9 % over 12 h.”  
  - *Construct:* `requirement def`.  
- **ReqInteroperability** – “All devices must support MQTT v5.”  
  - *Construct:* `requirement def`.  
- **ReqEaseOfUse** – “Setup wizard shall complete within 5 min.”  
  - *Construct:* `requirement def`.  
- **ReqPrivacy** – “Personal data must not leave the local network.”  
  - *Construct:* `constraint def`.  
- **ReqAvailability** – “Continuous operation for 24 h periods.”  
  - *Construct:* `requirement def`.  
- **ReqMaintainability** – “Hot‑swappable device modules.”  
  - *Construct:* `requirement def`.  

---

#### **## SysML v2 Constructs Map**  
**Definitions**  
- `part def`: SmartHome, Device, Gateway, ControlCenter  
- `action def`: UserInteraction, SystemResponse  
- `requirement def`: ReqSecurity, ReqEnergyEfficiency, … , ReqMaintainability  
- `constraint def`: ReqPrivacy  

**Usages**  
- *Part usage:* smartHome : SmartHome, gateway : Gateway, device[4] : Device, controlCenter : ControlCenter  
- *Action usage:* perform userInteraction, perform systemResponse  

**Relationships**  
- *satisfy:*  
  - `smartHome` satisfies `ReqSecurity`, `ReqEnergyEfficiency`  
  - `gateway` satisfies `ReqInteroperability`, `ReqPrivacy`  
  - `controlCenter` satisfies `ReqReliability`, `ReqAvailability`  
  - `device` satisfies `ReqMaintainability`, `ReqEaseOfUse`  

---

#### **## Package Structure**  
- **Root Package:** SmartHomeRequirements  
- **Sub‑packages:** `StructuralElements`, `BehavioralElements`, `Requirements`  
- **Imports:** `import ISU::*` (units for power, time)  
- **Aliases:** none required  

---

#### **## Recommended Views**  
- `[X] Requirement View:` traceability of each requirement to owning part.  
- `[X] Interconnection View:` illustrates how components satisfy security/interop.  
- `[X] Definition Tree:` reveals part hierarchies driving requirement scopes.  
- `[X] Usage View:` shows a concrete smart‑home instance with assigned requirements.  

*These views collectively provide a clear, traceable requirements model for the smart home system while aligning with SysML v2 best practices.*""",
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
