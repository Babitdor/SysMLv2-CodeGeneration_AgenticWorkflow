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
from states.SysMLResponse import SysMLResponse
from states.ValidationState import ValidationStatus, ValidationResult, ErrorInfo
from tools.SysMLValidatorTool import sysml_validator_tool
from tools.SysMLSyntaxTool import syntax_tool
from tools.SysMLExampleSearchTool import SysMLExampleSearchTool


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SysMLAgent:
    """Agent responsible for generating SysML code"""

    def __init__(
        self,
        llm,
        config_path: str = "agents/prompt/prompt.yaml",
        rag_knowledge_base=None,
        tools: Optional[List[BaseTool]] = None,
    ) -> None:
        self.name = "SysMLAgent"
        self.llm = llm
        self.rag_knowledge_base = rag_knowledge_base
        self.parser = PydanticOutputParser(pydantic_object=SysMLResponse)

        # Load and enhance system prompt with format instructions
        base_prompt = self._load_system_prompt(config_path)
        format_instructions = self.parser.get_format_instructions()

        self.system_prompt = f"""{base_prompt}

        OUTPUT FORMAT REQUIREMENTS:
        {format_instructions}

        CRITICAL: After using tools and generating code, your final response MUST be valid JSON matching the schema above.
        Include the complete SysML code in the "code" field, validation status in "validated", and any tool usage in the respective fields.
        """
        self.tools = self._initialize_tools(tools)

    def _initialize_tools(
        self, custom_tools: Optional[List[BaseTool]] = None
    ) -> List[BaseTool]:
        """Initialize all tools for the agent"""
        tools_list = []

        # Add standard tools
        tools_list.append(syntax_tool)
        tools_list.append(sysml_validator_tool)

        # Add example search tool if RAG knowledge base is available
        if self.rag_knowledge_base:
            try:
                sysml_kb_search_tool = SysMLExampleSearchTool(
                    rag_knowledge_base=self.rag_knowledge_base
                )
                tools_list.append(sysml_kb_search_tool)
                logger.info("✅ SysMLExampleSearchTool initialized and added to agent")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize SysMLExampleSearchTool: {e}")
        else:
            logger.info(
                "ℹ️ RAG knowledge base not available, skipping example search tool"
            )

        # Add custom tools if provided
        if custom_tools:
            tools_list.extend(custom_tools)
            logger.info(f"Added {len(custom_tools)} custom tools")

        logger.info(f"Total tools initialized: {len(tools_list)}")
        for tool in tools_list:
            logger.info(f"  - {tool.name}: {tool.description[:60]}...")

        return tools_list

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

    # def _get_knowledge_context(self, query: str) -> str:
    #     """Retrieve relevant context from knowledge base"""
    #     if not self.knowledge_base:
    #         return "No knowledge base available."

    #     try:
    #         docs = self.knowledge_base.similarity_search(query, k=3)
    #         context_entries = []
    #         for doc in docs:
    #             if hasattr(doc, "page_content"):
    #                 # Handle Document objects
    #                 context_entries.append(doc.page_content)
    #             elif isinstance(doc, dict):
    #                 # Handle dictionary results
    #                 # Assuming the content is in a 'content' or 'text' field
    #                 content = doc.get("content") or doc.get("text") or str(doc)
    #                 context_entries.append(content)
    #             else:
    #                 # Handle any other format by converting to string
    #                 context_entries.append(str(doc))

    #         context = "\n\n".join(context_entries)
    #         formatted_context = f"These are relevant approved solutions:\n\n{context}"
    #         logger.info("Retrieved knowledge base context")
    #         return formatted_context
    #     except Exception as e:
    #         logger.warning(f"Error retrieving knowledge context: {e}")
    #         return "No relevant examples found in knowledge base."

    def _create_prompt_with_context(self, state: WorkflowState) -> str:
        """Create a comprehensive prompt including context from previous attempts."""
        prompt_parts = [
            f"This is the current SysML v2 code that is validated : {state.code}"
        ]

        # Add human feedback if provided
        if state.human_feedback:
            prompt_parts.append(
                f"\n Now the user wants some changes to the code \n Here is the user feedback: {state.human_feedback}"
            )
        else:
            prompt_parts.append(
                f"\n No user feedback was given so just optimize the code."
            )

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

            agent = create_agent(
                name=self.name,
                model=self.llm,
                tools=self.tools,
                system_prompt=self.system_prompt,
            )

            # Create contextual prompt for refinement on (reject and feedback)
            if state.approval_status and state.approval_status in [
                "rejected",
                "feedback",
            ]:
                query = self._create_prompt_with_context(state)

            else:
                query = [
                    f"Here is a SysML-v2 template code that needs to be validated: \n\n {state.code} "
                ]

            response = agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
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

            # Try to parse structured response
            try:
                parsed_response = self.parser.parse(response)

                # Update state with parsed structured data
                state.code = parsed_response.code

                if parsed_response.validated:
                    state.is_valid = ValidationStatus.VALID

                    # Add successful validation to history
                    validation_result = ValidationResult(
                        success=True,
                        errors=[],
                        warnings=[],
                        output="Code validated successfully by SysMLAgent",
                        raw_output=response[:500],  # Truncate for storage
                    )
                    state.validation_history.append(validation_result)
                    state.error = ""  # Clear any previous errors

                    logger.info("✅ Successfully validated SysML code")

                else:
                    state.is_valid = ValidationStatus.INVALID

                    # Add failed validation to history if there are validation issues
                    validation_result = ValidationResult(
                        success=False,
                        errors=[
                            ErrorInfo(
                                name="ValidationError",
                                message="Code validation failed",
                                traceback=[],
                            )
                        ],
                        warnings=[],
                        output="Code validation failed by SysMLAgent",
                        raw_output=response[:500],
                    )
                    state.validation_history.append(validation_result)

                    logger.warning("⚠️ Code validation marked as failed")

            except Exception as parse_error:
                logger.warning(f"⚠️ Structured parsing failed: {parse_error}")
                logger.info("📋 Falling back to legacy code extraction")

                # Fallback to legacy extraction
                sysml_code = self.extract_sysml_code(response)

                if not sysml_code.strip():
                    raise ValueError("LLM did not generate any code")

                state.code = sysml_code
                state.error = f"Note: Used fallback extraction. Parsing errors: {str(parse_error)}"
                state.is_valid = ValidationStatus.PENDING

                # Create minimal metadata
                # state.metadata = SysMLResponse(
                #     code=sysml_code,
                #     validated=False,
                #     syntax_checks_performed=[],
                # )

            logger.info("SysML Agent: Successfully generated SysML code")
            return state

        except Exception as e:
            error_msg = f"LLM Generation Error: {str(e)}"
            logger.error(f"SysML Agent error: {error_msg}")

            # Add error validation to history
            validation_result = ValidationResult(
                success=False,
                errors=[
                    ErrorInfo(name="GenerationError", message=error_msg, traceback=[])
                ],
                warnings=[],
                output="Code generation failed",
                raw_output=str(e)[:500],
            )
            state.validation_history.append(validation_result)
            return state

    def generate_with_feedback(
        self, state: WorkflowState, feedback: str
    ) -> WorkflowState:
        """Generate code with additional human feedback."""
        logger.info("SysML Agent: Generating code with human feedback")

        # Update the state with the feedback
        state.human_feedback = feedback

        return self.generate_code(state)  # type: ignore

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

        return self.generate_code(refined_state)  # type: ignore


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
                # model="SysML-V2-llama3.1:latest",
                # reasoning=True,
                model="qwen3-coder:480b-cloud",
                # temperature=0.15,
                num_ctx=16000,
                validate_model_on_init=True,
            )
        )
        print("✅ Agent initialized successfully")

        # Test 1: Basic code generation
        print("\n2. Testing basic code generation...")
        test_state = WorkflowState(
            original_query="Create a simple vehicle system model",
            code="""package SimpleCarModel {
                        private import ISQ::*;
                        private import SI::*;
                        alias ISQ::MassValue as MassValue;
                        alias ISQ::TimeValue as TimeValue;
                        public import ScalarValues::*;

                        // STRUCTURAL ELEMENTS
                        part def Car {
                            :>> mass: MassValue = 1200.0 [kg];
                            port fuelPort: FuelPort;
                            part engine: Engine;
                            part transmission: Transmission;
                            part wheels[4]: Wheel;
                        }

                        part def Engine {
                            :>> mass: MassValue = 60.0 [kg];
                            attribute displacement: Real = 1998.0 ["L"];
                            attribute maxPower: Real = 150.0 [kW];
                            attribute maxTorque: Real = 1400.0 [Nm];
                            attribute fuelConsumptionIdle: Real = 10.0 [L/h];
                            state fuelStates { 
                                entry fuelOff; then off; 
                                accept FuelCmd send fuelCmd; 
                                accept StartCmd send startCmd; 
                                state off; 
                                state on;
                            }
                            action startEngine;
                        }

                        part def Transmission {
                            :>> mass: MassValue = 30.0 [kg];
                            attribute gearRatio: Real = 3.5;
                            attribute maxTorqueOut: Real = 1200.0 [Nm];
                            state clutchStates { entry clutchOff; then engaged; 
                                state engaged; 
                                state disengaged; 
                            }
                        }

                        part def Wheel {
                            :>> mass: MassValue = 15.0 [kg];
                            attribute diameter: Real = 0.5687 [m];
                            attribute width: Real = 0.2244 [m];
                            attribute maxPressure: Real = 3.0 [bar];
                        }

                        port def FuelPort {
                            out attribute fuelCmd: FuelCmd;
                            in attribute startCmd: StartCmd;
                            in item fuelIn: Fuel;
                        }

                        item def Fuel {
                            :>> mass: MassValue;
                            attribute composition: String;
                            attribute boilingPoint: Real = 150.0 [C];
                        }

                        // BEHAVIORAL ELEMENTS
                        action def StartEngine {
                            input fuelCmd: FuelCmd;
                            output startCmd: StartCmd;
                            then assign variable fuelRate := 0.0 [L/min];
                        }

                        state def EngineStates {
                            entry fuelOff; 
                            accept StartCmd send startCmd; 
                            state off; 
                            state on;
                        }

                        // REQUIREMENTS CONSTRAINTS
                        requirement def EngineStartTime {
                            subject engine: Engine;
                            attribute timeToStart: Real = 5.0 [s];
                            constraint { timeToStart <= 5.0 [s] }
                        }

                        constraint def FuelFlowRate {
                            subject fuelTank: FuelTank;
                            attribute flowRate: Real = 10.0 [L/min];
                            constraint { flowRate <= 10.0 [L/min] }
                        }

                        analysis def FuelConsumptionAnalysis {
                            subject car: Car;
                            objective evaluateFuelConsumption;
                        }

                        // DEFINITIONS MAP
                        part def :> Engine;
                        part def :> Transmission;
                        part def :> Wheel;
                        item def :> Fuel;
                        action def :> StartEngine;
                        state def :> EngineStates;

                        // USAGES MAP
                        part car: Car {
                            part engine: Engine {
                                perform action startEngine;
                                state fuelStates;
                            }
                            part transmission: Transmission;
                            part wheels[4]: Wheel;
                            port fuelPort: FuelPort {
                                out attribute fuelCmd: FuelCmd;
                                in attribute startCmd: StartCmd;
                                in item fuelIn: Fuel;
                            }
                        }

                        connection def FuelConnection;
                        connection fuelToEngine: FuelConnection connects car.fuelPort.fuelIn to car.engine.fuelIn;

                        action startEngine: StartEngine {
                            input fuelCmd: FuelPort::fuelCmd;
                            output startCmd: FuelPort::startCmd;
                            then assign variable fuelRate := 0.0 [L/min];
                        }

                        state engineStates: EngineStates {
                            entry fuelOff; 
                            accept StartCmd send fuelPort.startCmd; 
                            state off; 
                            state on;
                        }

                        // RELATIONSHIPS MAP
                        part def ProductionCar :> Car;
                        part def SmallEngine :> Engine { :>> displacement = 2.0 [L]; }
                        action def TriggerStartEngine redefines startEngine {
                            :>> displacement = 2.0 [L];
                            :>> mass = 60.0 [kg];
                            :>> maxPower = 150.0 [kW];
                            :>> maxTorque = 1400.0 [Nm];
                            :>> fuelConsumptionIdle = 10.0 [L/h];
                        }
                        state def SmallEngineStates redefines engineStates {
                            entry fuelOff; 
                            accept StartCmd send smallCar.startCmd; 
                            state off; 
                            state on;
                        }
                        part def SmallTransmission :> Transmission { :>> gearRatio = 3.5; }
                        part def SmallWheel :> Wheel { :>> diameter = 0.5687 [m]; :>> width = 0.2244 [m]; }
                        item def SmallFuel redefines Fuel { :>> boilingPoint = 150.0 [C]; }
                        port def SmallFuelPort redefines FuelPort {
                            out attribute redefines fuelCmd;
                            in attribute redefines startCmd;
                            in item redefines fuelIn;
                        }
                        connection def SmallFuelConnection redefines FuelConnection;
                        connection smallFuelToEngine: SmallFuelConnection connects 
                            smallFuelPort.fuelIn to engine.fuelIn;

                        // PACKAGE STRUCTURE
                        package Mechanical {
                            part def Engine;
                            part def Transmission;
                            part def Wheel;
                            port def FuelPort;
                            item def Fuel;
                        }

                        package Powertrain {
                            part def Engine {
                                :>> displacement = 2.0 [L];
                                :>> mass = 60.0 [kg];
                                action startEngine;
                                state fuelStates;
                            }
                            part def Transmission {
                                :>> gearRatio = 3.5;
                                state clutchStates;
                            }
                            part def Wheel {
                                :>> diameter = 0.5687 [m];
                                :>> width = 0.2244 [m];
                                attribute maxPressure = 3.0 [bar];
                            }
                            port def FuelPort {
                                out attribute fuelCmd;
                                in attribute startCmd;
                                in item fuelIn;
                            }
                        }

                        // RECOMMENDED VIEWS
                        view InterconnectionView {
                            view car : Car connects engine to transmission to wheels
                            view fuelConnection : FuelConnection connects fuelPort to engine
                            view startProcess : StartEngine connects fuelPort to engine
                        }

                        view DefinitionView {
                            part def Engine;
                            part def Transmission;
                            part def Wheel;
                            item def Fuel;
                            port def FuelPort;
                            action def StartEngine;
                            state def EngineStates;
                        }

                        view UsageView {
                            part car: Car
                            part engine: Engine
                            part transmission: Transmission
                            part wheels[4]: Wheel
                            port fuelPort: FuelPort
                            action startEngine: StartEngine
                            state engineStates: EngineStates
                        }

                        view StateMachineView {
                            state def EngineStates {
                                entry fuelOff; 
                                accept StartCmd send startCmd; 
                                state off; 
                                state on;
                            }
                        }

                        view RequirementView {
                            requirement def EngineStartTime {
                                subject engine: Engine;
                                attribute timeToStart: Real = 5.0 [s];
                                constraint { timeToStart <= 5.0 [s] }
                            }
                            satisfaction of EngineStartTime by car.engine;
                        }

                        view ParametricView {
                            // N/A
                        }
                    }
        """,
        )

        print(f"   Query: {test_state.processed_query}")
        print("   Generating code...")

        start_time = time.time()
        result_state = agent.generate_code(test_state)
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

        # # Test 2: Code generation with human feedback
        # print("\n3. Testing code generation with feedback...")
        # feedback = "Change attribute mass to 600 kg"
        # print(f"   Feedback: {feedback}")

        # start_time = time.time()
        # feedback_state = agent.generate_with_feedback(result_state, feedback)  # type: ignore
        # feedback_time = time.time() - start_time

        # if feedback_state.code != result_state.code:  # type: ignore
        #     print("✅ Code updated with feedback")
        #     print(f"   Generation time: {feedback_time:.2f}s")
        #     print(f"   New code length: {len(feedback_state.code)} characters")
        # else:
        #     print("⚠️  Code unchanged (might be identical or error occurred)")

        # # Test 3: Code refinement
        # print("\n4. Testing code refinement...")
        # refinement_instructions = "remove the attribute mass from Vehicle"
        # print(f"   Instructions: {refinement_instructions}")

        # try:
        #     start_time = time.time()
        #     refined_state = agent.refine_code(feedback_state, refinement_instructions)
        #     refinement_time = time.time() - start_time

        #     print("✅ Code refinement completed")
        #     print(f"   Refinement time: {refinement_time:.2f}s")
        #     print(f"   Final code length: {len(refined_state.code)} characters")
        # except Exception as e:
        #     print(f"❌ Refinement failed: {e}")

        # # Test 4: Error simulation and handling
        # print("\n5. Testing error handling...")

        # # Create a state with simulated validation errors
        # error_state = WorkflowState(
        #     original_query="Create a faulty system model",
        #     processed_query="Create a system model with intentional syntax errors for testing",
        #     code="attribute mass :> ISQ::mass = 1200.0 [kg]",
        #     error="Syntax error: missing semicolon",
        #     is_valid=ValidationStatus.INVALID,
        #     iteration=2,
        # )

        # Add a mock validation result with errors
    #     mock_validation = ValidationResult(
    #         success=False,
    #         errors=[
    #             ErrorInfo(
    #                 name="SyntaxError",
    #                 message="Missing semicolon at end of statement",
    #                 line_number=5,
    #             ),  # type: ignore
    #             ErrorInfo(
    #                 name="TypeError", message="Invalid type assignment", line_number=8
    #             ),  # type: ignore
    #         ],
    #         output="Compilation failed with 2 errors",
    #     )  # type: ignore
    #     error_state.add_validation_result(mock_validation)

    #     print(f"   Simulated errors: {len(mock_validation.errors)}")
    #     print("   Attempting error fixes...")

    #     try:
    #         start_time = time.time()
    #         fixed_state = agent.fix_errors(error_state)
    #         fix_time = time.time() - start_time

    #         print("✅ Error fixing attempted")
    #         print(f"   Fix time: {fix_time:.2f}s")
    #         print(f"   Status: {fixed_state.is_valid}")
    #     except Exception as e:
    #         print(f"❌ Error fixing failed: {e}")

    #     # Test 5: Extract code functionality
    #     print("\n6. Testing code extraction...")
    #     test_responses = [
    #         "Here's your SysML code:\n```sysml\npackage TestPackage;\npart Vehicle;\n```\nThat should work!",
    #         "```\npackage SimpleTest;\npart Engine;\n```",
    #         "package DirectCode;\npart Wheel;",
    #     ]

    #     for i, response in enumerate(test_responses, 1):
    #         extracted = agent.extract_sysml_code(response)
    #         print(
    #             f"   Test {i}: {'✅ Extracted' if extracted else '❌ Failed'} - Length: {len(extracted)}"
    #         )

    #     print("\n" + "=" * 60)
    #     print("Test Summary:")
    #     print("- Basic code generation: Completed")
    #     print("- Feedback integration: Completed")
    #     print("- Code refinement: Completed")
    #     print("- Error handling: Completed")
    #     print("- Code extraction: Completed")
    #     print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        print("Please check your Ollama installation and model availability.")
        print("Make sure the model is installed.")


if __name__ == "__main__":
    main()
