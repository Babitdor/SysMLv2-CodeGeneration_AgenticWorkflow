from langchain_ollama import ChatOllama
import logging
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from states.WorkflowState import WorkflowState
from agents.SysMLAgent import SysMLAgent

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_sysml_agent():
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

        # Test 2: Code generation with human feedback
        print("\n3. Testing code generation with feedback...")
        feedback = "Change attribute mass to 600 kg"
        print(f"   Feedback: {feedback}")

        start_time = time.time()
        feedback_state = agent.generate_with_feedback(result_state, feedback)  # type: ignore
        feedback_time = time.time() - start_time

        if feedback_state.code != result_state.code:  # type: ignore
            print("✅ Code updated with feedback")
            print(f"   Generation time: {feedback_time:.2f}s")
            print(f"   New code length: {len(feedback_state.code)} characters")
        else:
            print("⚠️  Code unchanged (might be identical or error occurred)")

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        print("Please check your Ollama installation and model availability.")
        print("Make sure the model is installed.")


if __name__ == "__main__":
    test_sysml_agent()
