from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from deepagents import FilesystemPermission, create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    LocalShellBackend,
)
from langchain_core.messages import AIMessageChunk, ToolMessage
import json
import re
import time
import yaml
import logging
import os
import sys
import asyncio
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from langchain.tools import BaseTool


load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from states.WorkflowState import WorkflowState
from states.SysMLResponse import SysMLResponse
from states.ValidationState import ValidationStatus, ValidationResult, ErrorInfo
from tools.SysMLValidatorTool import sysml_validator_tool

try:
    from tools.SysMLSyntaxTool import syntax_tool
except ModuleNotFoundError as exc:
    if exc.name != "tools.SysMLSyntaxTool":
        raise
    syntax_tool = None

try:
    from tools.SysMLExampleSearchTool import SysMLExampleSearchTool
except ModuleNotFoundError as exc:
    if exc.name != "tools.SysMLExampleSearchTool":
        raise
    SysMLExampleSearchTool = None


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Tool-call tracing: a `tool` event names the call, what it asked for, and a
# capped head of what came back. The cap keeps a chatty tool (a 50 KB read) from
# flooding the transcript while `chars` still reports the full length, so a
# client can say "showing 1200 of 50213".
TRACE_CHARS = 1200
# A summary is one line in a dense list, so it is its own, tighter bound.
SUMMARY_CHARS = 120

# The argument that says what a call is about, per tool; anything else is
# summarised by its argument names.
_SUMMARY_ARGS = {
    "read_file": "file_path",
    "write_file": "file_path",
    "grep": "pattern",
    "glob": "pattern",
    "ls": "path",
    "shell": "command",
}


def _preview(text: str) -> Tuple[str, bool, int]:
    """(capped head, was it cut, full length) under SYSML_TRACE_CHARS."""
    try:
        limit = int(os.environ.get("SYSML_TRACE_CHARS", TRACE_CHARS))
    except ValueError:
        limit = TRACE_CHARS
    limit = max(0, limit)
    full = text or ""
    return full[:limit], len(full) > limit, len(full)


def _text_of(content: Any) -> str:
    """A ToolMessage's content as text: it is a str, or a list of blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        return "" if content is None else str(content)
    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if not isinstance(text, str):
                text = block.get("content")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _tool_summary(name: Optional[str], args: str) -> str:
    """One line naming what a call is about; malformed or partial JSON is fine.

    Arguments arrive as fragments, so the first summary is usually computed from
    text that does not parse yet: the wanted key is read straight out of the
    fragment and the unknown case degrades to the argument names seen so far.
    """
    wanted = _SUMMARY_ARGS.get((name or "").lower())
    text = args or ""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        value = parsed.get(wanted) if wanted else None
        if isinstance(value, str) and value.strip():
            return value.strip()[:SUMMARY_CHARS]
        if not wanted:
            # A tool outside the map: the first argument carrying text is what
            # the call is about; its keys alone read as noise in the row.
            for candidate in parsed.values():
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()[:SUMMARY_CHARS]
        return ", ".join(sorted(str(key) for key in parsed))[:SUMMARY_CHARS]
    if wanted:
        match = re.search(r'"%s"\s*:\s*"([^"]*)' % re.escape(wanted), text)
        if match:
            return match.group(1).strip()
    return ", ".join(dict.fromkeys(re.findall(r'"([A-Za-z0-9_]+)"\s*:', text)))


class SysMLAgent:
    """Agent responsible for generating SysML code"""

    def __init__(
        self,
        llm,
        config_path: str = "agents/prompt/SysML.j2",
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

        WORKSPACE:
        - Your shell and filesystem are the Docker container you run in. Every deliverable goes
          under /workspace: models in /workspace/current/, other artifacts in /workspace/artifacts/.
        - Files the user uploads arrive in /workspace/uploads/. List them with ls("/workspace/uploads"),
          read them with read_file before modeling, and never assume their content.
        - Save the model with write_file(file_path="/workspace/current/<name>.sysml"), then validate
          that saved file with SysML-validator-tool(file_path="..."). Inline code validation is a
          fallback for short probes only.
        - Files under /workspace persist across container restarts. Do not write outside /workspace
          except read-only /skills/.

        VALIDATION RULES:
        - Validate the saved file via SysML-validator-tool(file_path=...); use code=... only when no file exists.
        - Diagram playbooks live under /skills/<diagram>/SKILL.md; read them with read_file(limit=1000).
        - /skills/ is read-only. Do not edit SKILL.md or LEARNED.md; report new fix patterns in the JSON.
        - There is no example-search or syntax-reference tool; skills and /skills/sysml-v2-reference replace them.
        - Set "validated" to true only when the final tool report says "✅ Code is valid!".
        - Local fallback checks and the model's own claims do not establish compiler acceptance.
        """
        self.tools = self._initialize_tools(tools)

        # agents/skills/*/SKILL.md are the SysML v2 diagram playbooks. They are
        # routed to a read-only backend so the skills middleware can list and
        # load them, while every other path stays on the Docker sandbox.
        skills_root = Path(__file__).resolve().parent / "skills"
        self.skills_root = skills_root if any(skills_root.glob("*/SKILL.md")) else None
        if self.skills_root is None:
            logger.warning("⚠️ No SysML skills found under %s", skills_root)

        # The deep agent executes inside Docker: this process already runs in the
        # sysml-agents container, so LocalShellBackend gives the agent a real
        # container filesystem and shell rooted at /workspace instead of the
        # in-memory StateBackend. root_dir="/" keeps paths absolute so the
        # validator can read /workspace/current/*.sysml, and virtual_mode blocks
        # ".." escapes; it does not restrict shell commands, which stay inside
        # the container.
        # The agent's deliverables live in the workspace volume: current/ holds the
        # SysML model files the validator reads back, artifacts/ holds everything
        # else it produces.
        workspace_root = Path(os.environ.get("SYSML_WORKSPACE_ROOT", "/workspace"))
        for subdir in ("current", "artifacts", "uploads"):
            try:
                (workspace_root / subdir).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("⚠️ Cannot prepare %s: %s", workspace_root / subdir, exc)

        sandbox = LocalShellBackend(
            root_dir="/",
            virtual_mode=True,
            timeout=120,
            inherit_env=True,
        )
        self.sandbox = sandbox

        if self.skills_root is not None:
            backend = CompositeBackend(
                default=sandbox,
                routes={
                    "/skills/": FilesystemBackend(
                        root_dir=str(skills_root), virtual_mode=True
                    )
                },
            )
            permissions = [
                FilesystemPermission(
                    operations=["write"], paths=["/skills/**"], mode="deny"
                )
            ]
            skill_sources = ["/skills/"]
        else:
            backend = sandbox
            permissions = None
            skill_sources = None

        # The validator reads saved models through the same backend, so attach it
        # before the agent is built; without a sandbox only inline code validates.
        for tool in self.tools:
            if getattr(tool, "sandbox", None) is None and hasattr(tool, "sandbox"):
                tool.sandbox = sandbox

        # Build the deep agent once: planning/todo, delegation and the diagram
        # skills ride along with the project tools. The checkpointer keeps each
        # thread's messages (and the agent's own todos) so a follow-up request on
        # the same thread_id continues the conversation instead of starting over.
        self.checkpointer = InMemorySaver()
        self.agent = create_deep_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            backend=backend,
            skills=skill_sources,
            permissions=permissions,
            checkpointer=self.checkpointer,
            name=self.name,
        )

    def _initialize_tools(
        self, custom_tools: Optional[List[BaseTool]] = None
    ) -> List[BaseTool]:
        """Initialize all tools for the agent"""
        tools_list = []

        # Add available standard tools.
        if syntax_tool is not None:
            tools_list.append(syntax_tool)
        tools_list.append(sysml_validator_tool)

        # Add example search tool if RAG knowledge base is available
        if self.rag_knowledge_base and SysMLExampleSearchTool is not None:
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

    def _create_prompt_with_context(self, state: WorkflowState) -> str:
        """Create a comprehensive prompt including context from previous attempts."""
        prompt_parts = [f"Current SysML v2 code:\n{state.code}"]
        latest_validation = state.get_latest_validation()
        if latest_validation and not latest_validation.success:
            prompt_parts.append(
                f"Previous SysML-validator-tool report:\n{latest_validation.output}\n"
                "Repair the reported errors before returning the complete model."
            )

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

            # Create contextual prompt for refinement on (reject and feedback)
            if state.approval_status and state.approval_status in [
                "rejected",
                "feedback",
            ]:
                query = self._create_prompt_with_context(state)

            elif state.code.strip():
                query = self._create_prompt_with_context(state)

            else:
                query = [
                    f"Create a SysML-v2 model with the following:\n\n Requirements: {state.processed_query}"
                ]

            response = self.agent.invoke(
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

            state.code = self._final_code(response)
            if not state.code.strip():
                raise ValueError("LLM did not generate any code")

            # Validate the exact returned code; the model's "validated" field is
            # informational only and cannot approve its own output.
            verdict = sysml_validator_tool._run(code=state.code)
            validation_succeeded = f"\n✅ Code is valid!" in f"\n{verdict}"
            state.is_valid = (
                ValidationStatus.VALID if validation_succeeded else ValidationStatus.INVALID
            )
            state.error = "" if validation_succeeded else verdict
            state.validation_history.append(
                ValidationResult(
                    success=validation_succeeded,
                    errors=(
                        []
                        if validation_succeeded
                        else [ErrorInfo(name="ValidationError", message=verdict)]
                    ),
                    warnings=[],
                    output=verdict,
                    raw_output=verdict[:500],
                )
            )
            if validation_succeeded:
                logger.info("✅ SysML validator accepted the generated code")
            else:
                logger.warning("⚠️ SysML validation did not establish compiler acceptance")

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
            state.is_valid = ValidationStatus.ERROR
            state.error = error_msg
            return state

    def stream_generation(self, query: str, thread_id: Optional[str] = None):
        """Stream one generation run: token/tool deltas, then {"type":"generated"}.

        No validation here — `stream_code` and the web pipeline own that, so a
        repair loop can re-run generation without duplicate verdicts.
        """
        config: dict = {"recursion_limit": 100}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}

        buffer: List[str] = []
        announced = set()
        # A tool call arrives as fragments: id and name on one chunk, the JSON
        # arguments split across several. Records are keyed by the chunk index,
        # because a later fragment usually carries no name, and are looked up by
        # call id once the tool answers — so the same tool used twice stays two
        # independent rows.
        partials: Dict[Any, Dict[str, Any]] = {}
        by_id: Dict[str, Dict[str, Any]] = {}

        def merge(index: Any, fragment: Dict[str, Any]) -> Dict[str, Any]:
            record = partials.get(index)
            if record is None:
                record = {
                    "index": index,
                    "id": "",
                    "name": None,
                    "args": "",
                    "started": 0.0,
                }
                partials[index] = record
            if fragment.get("id"):
                record["id"] = fragment["id"]
                by_id.setdefault(fragment["id"], record)
            if fragment.get("name"):
                record["name"] = fragment["name"]
            record["args"] += fragment.get("args") or ""
            return record

        try:
            for item in self.agent.stream(
                {"messages": [{"role": "user", "content": query}]},
                config=config,
                stream_mode="messages",
            ):
                chunk = item[0] if isinstance(item, tuple) else item
                if isinstance(chunk, AIMessageChunk):
                    for call in chunk.tool_call_chunks or []:
                        record = merge(call.get("index"), call)
                        if record["id"] and record["id"] not in announced:
                            announced.add(record["id"])
                            record["started"] = time.monotonic()
                            yield {
                                "type": "tool",
                                "name": record["name"] or "tool",
                                "status": "start",
                                "call_id": record["id"],
                                "summary": _tool_summary(
                                    record["name"], record["args"]
                                ),
                            }
                    text = self._chunk_text(chunk)
                    if text:
                        buffer.append(text)
                        yield {"type": "token", "text": text}
                elif isinstance(chunk, ToolMessage):
                    call_id = chunk.tool_call_id or ""
                    record = by_id.pop(call_id, None)
                    if record is not None:
                        partials.pop(record["index"], None)
                    name = (record["name"] if record else None) or chunk.name or "tool"
                    args = record["args"] if record else ""
                    output, truncated, chars = _preview(_text_of(chunk.content))
                    yield {
                        "type": "tool",
                        "name": name,
                        # Repeated from the start event: only now are the
                        # arguments complete, so this is the summary worth
                        # keeping on the finished row.
                        "summary": _tool_summary(name, args),
                        "status": (
                            "failed"
                            if getattr(chunk, "status", None) == "error"
                            else "done"
                        ),
                        "call_id": call_id,
                        "ms": (
                            int((time.monotonic() - record["started"]) * 1000)
                            if record and record["started"]
                            else 0
                        ),
                        "chars": chars,
                        "truncated": truncated,
                        "args": args,
                        "output": output,
                    }
        except Exception as e:
            logger.error(f"SysML Agent streaming error: {e}")
            yield {"type": "error", "detail": str(e)}
            return

        code = self._final_code("".join(buffer))
        yield {"type": "generated", "code": code}

    def stream_code(self, query: str, thread_id: Optional[str] = None):
        """Stream generation followed by compiler validation with local fallback evidence."""
        for event in self.stream_generation(query, thread_id):
            if event["type"] == "error":
                yield event
                return
            if event["type"] == "generated":
                code = event["code"]
                continue
            yield event

        yield {"type": "code", "code": code}

        try:
            verdict = sysml_validator_tool._run(code)
        except Exception as e:
            verdict = f"❌ Validator error: {e}"
        yield {
            "type": "validation",
            "ok": bool(code.strip()) and f"\n✅ Code is valid!" in f"\n{verdict}",
            "detail": verdict,
        }

    @staticmethod
    def _chunk_text(chunk: AIMessageChunk) -> str:
        """Token text from a chunk, tolerating provider-specific shapes."""
        text = getattr(chunk, "text", "")
        if isinstance(text, str):
            return text
        if isinstance(chunk.content, str):
            return chunk.content
        return ""

    def _final_code(self, answer: str) -> str:
        """Structured JSON output wins; markdown fences are the fallback."""
        try:
            return self.parser.parse(answer).code
        except Exception:
            return self.extract_sysml_code(answer)

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
