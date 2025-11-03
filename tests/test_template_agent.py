from langchain_ollama import ChatOllama
import logging
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from states.WorkflowState import WorkflowState
from agents.TemplateAgent import TemplateAgent


def test_template_agent():
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
                model="Qwen3-4B-SysMLv2:latest",
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
    test_template_agent()
