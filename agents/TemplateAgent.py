from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
import re
import yaml
import logging
import os
import sys
import asyncio
from dotenv import load_dotenv
from typing import List, Optional
from langchain.tools import BaseTool


load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server import get_mcp_client
from states.WorkflowState import WorkflowState
from states.ValidationState import ValidationStatus


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TemplateAgent:
    """Agent responsible for generating SysML Template code"""

    def __init__(
        self,
        llm,
        config_path: str = "agents/prompt/prompt.yaml",
    ) -> None:
        self.name = "TemplateAgent"
        self.llm = llm
        self.system_prompt = self._load_system_prompt(config_path)

    def _load_system_prompt(self, config_path: str) -> str:
        """Load system prompt from a YAML config file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"System prompt loaded from {config_path}")
            prompt = config.get("Template-Agent", "")
            if not prompt:
                raise ValueError(
                    f"'Template-Agent' key not found or empty in {config_path}."
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

    def generate_template_code(self, state: WorkflowState) -> WorkflowState:
        """Generate SysML template code based on the current state and return updated state."""

        try:
            logger.info(
                f"Template Agent: Generating SysML code (iteration {state.iteration})"
            )

            agent = create_agent(
                name=self.name,
                model=self.llm,
                system_prompt=self.system_prompt,
            )

            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Create a SysML-v2 model with the following: \n\n Requirements: {state.processed_query}",
                        }
                    ]
                },
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

                # Fallback to legacy extraction
            sysml_code = self.extract_sysml_code(response)

            if not sysml_code.strip():
                raise ValueError("LLM did not generate any code")

            state.code = sysml_code

            logger.info("Template Agent: Successfully generated SysML template code")
            return state

        except Exception as e:
            error_msg = f"LLM Generation Error: {str(e)}"
            logger.error(f"Template Agent error: {error_msg}")

            # Update state with error information
            state.error = error_msg
            state.is_valid = ValidationStatus.ERROR
            return state


def main():
    """Test main function to demonstrate Template Agent functionality"""
    import time

    print("=" * 60)
    print("Template Agent Test Suite")
    print("=" * 60)

    try:
        # Initialize the agent
        print("\n1. Initializing Template Agent...")
        agent = TemplateAgent(
            llm=ChatOllama(
                # reasoning=True,
                model="QwenCoder2.5-7B-SysML",
                temperature=0,
                num_ctx=16000,
                # validate_model_on_init=True,
            )
        )
        print("✅ Agent initialized successfully")

        # Test 1: Basic code generation
        print("\n2. Testing basic code generation...")
        test_state = WorkflowState(
            original_query="Create a simple vehicle system model",
            processed_query="""SYSTEM_ANALYSIS:
  system_type: Cyber-physical
  domain: Automotive
  complexity_level: Simple
  primary_focus: All

STRUCTURAL_ELEMENTS:
  - name: Car
    description: Top-level container for the vehicle
    sysml_construct: part def
    rationale: Core system that aggregates mechanical subsystems
  - name: Engine
    description: Internal combustion unit that provides propulsion
    sysml_construct: part def
    rationale: Mechanical component with its own behavior
  - name: Transmission
    description: Power transfer assembly from engine to wheels
    sysml_construct: part def
    rationale: Enables gear selection
  - name: Wheels
    description: Rolling contact elements that move the car
    sysml_construct: part def
    rationale: The final mechanical interface with the road
  - name: FuelPort
    description: External point for refueling
    sysml_construct: port def
    rationale: Interaction point with the environment
  - name: Fuel
    description: Consumable resource for the Engine
    sysml_construct: item def
    rationale: Flows from FuelPort to Engine

BEHAVIORAL_ELEMENTS:
  - name: StartEngine
    description: Ignition sequence that brings Engine to Running state
    sysml_construct: action def
    rationale: Discrete operation triggered by driver
  - name: EngineStates
    description: Activity lifecycle of the Engine (Off -> Running)
    sysml_construct: state def
    rationale: Reflects the Engine’s mode

REQUIREMENTS_CONSTRAINTS:
  - name: EngineStartTime
    description: Engine must reach Running state within 5s of key turn
    sysml_construct: requirement def
    rationale: Safety and usability requirement
  - name: FuelFlowRate
    description: Fuel must flow at ≤ 10 L/min during idle
    sysml_construct: constraint def
    rationale: Defines a flow limit for idle operation

DEFINITIONS_MAP:
  part_def:
    - Car
    - Engine
    - Transmission
    - Wheels
  item_def:
    - Fuel
  port_def:
    - FuelPort
  connection_def:
    - FuelConnection
  action_def:
    - StartEngine
  state_def:
    - EngineStates
  requirement_def:
    - EngineStartTimeRequirement
  constraint_def:
    - FuelFlowRateConstraint

USAGES_MAP:
  part_usage:
    - "car : Car"
    - "engine : Engine"
    - "transmission : Transmission"
    - "wheels[4] : Wheels"
  connection_usage:
    - "engineToTransmission : TransmissionConnection connects Engine to Transmission"
    - "fuelToEngine : FuelConnection connects FuelPort to Engine"
  port_usage:
    - "vehicleFuelPort : FuelPort"
  action_usage:
    - "car.perform startEngine"

RELATIONSHIPS_MAP:
  specializations:
    - child: ProductionCar
      parent: Car
      notation: ":>"
  redefinitions:
    - element: engine
      property: displacement
      value: "2.0 [L]"
      notation: ":>>"
  references:
    - source: engine.startEngine
      target: StartEngine
      notation: "::>"
  satisfactions:
    - requirement: EngineStartTime
      satisfied_by: engine

PACKAGE_STRUCTURE:
  root_package: SimpleCarModel
  sub_packages:
    - Mechanical
    - Powertrain
  imports:
    - "ISQ::*"
    - "SI::*"
  aliases:
    - old_name: ISQ::MassValue
      new_name: MassValue
    - old_name: ISQ::TimeValue
      new_name: TimeValue

RECOMMENDED_VIEWS:
  - view_type: Interconnection
    needed: yes
    purpose: Show how Engine, Transmission, and Wheels are connected
  - view_type: Definition
    needed: yes
    purpose: Display part, port, and item definitions
  - view_type: Usage
    needed: yes
    purpose: Illustrate a concrete Car instance with its subsystems
  - view_type: State_Machine
    needed: yes
    purpose: EngineStates (Off → Running)
  - view_type: Requirement
    needed: yes
    purpose: Map requirements to their satisfying elements
  - view_type: Parametric
    needed: no
    purpose: N/A""",
        )

        print(f"   Query: {test_state.processed_query}")
        print("   Generating code...")

        start_time = time.time()
        result_state = agent.generate_template_code(test_state)
        generation_time = time.time() - start_time

        if result_state.code:  # type: ignore
            print("✅ Code generated successfully")
            print(f"   Generation time: {generation_time:.2f}s")
            print(f"   Code length: {len(result_state.code)} characters")  # type: ignore
            print(f"   Status: {result_state.is_valid}")  # type: ignore
            print("\n   Generated Code Preview:")
            print("   " + "-" * 50)
            # Show first few lines of generated code
            code_lines = result_state.code.split("\n")[:10]  # type: ignore
            for line in code_lines:
                print(f"   {line}")
            if len(result_state.code.split("\n")) > 10:  # type: ignore
                print("   ... (truncated)")
            print("   " + "-" * 50)
        else:
            print("❌ Code generation failed")
            if result_state.error:  # type: ignore
                print(f"   Error: {result_state.error}")  # type: ignore

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        print("Please check your Ollama installation and model availability.")
        print(f"Make sure the model is installed.")


if __name__ == "__main__":
    main()
