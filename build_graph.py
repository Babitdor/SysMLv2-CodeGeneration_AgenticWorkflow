from typing import Dict, Any, Optional, List, Union
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.schema import Document
from agents.QueryAgent import QueryAgent
from agents.TemplateAgent import TemplateAgent
from agents.HumanApprovalAgent import HumanApprovalAgent
from agents.SysMLAgent import SysMLAgent
from langgraph.graph import StateGraph, END
from states.WorkflowState import WorkflowState
from states.ValidationState import ValidationStatus
from rag.Injestion import SysMLRAGPipeline
import logging
import os
from datetime import datetime
import json
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SysMLWorkflow:
    """
    Enhanced SysML workflow with RAG and vector database integration.
    Simplified workflow with SysMLAgent handling validation and corrections via tools.
    """

    def __init__(
        self,
        agents_prompt: str = "./prompt/prompt.yaml",
        model_name: str = "qwen3:1.7b",
        common_llm: Optional[Union[str, Dict, Any]] = None,
        sysml_agent_llm: Optional[Union[str, Any]] = None,
        query_agent_llm: Optional[Union[str, Any]] = None,
        template_agent_llm: Optional[Union[str, Any]] = None,
        temperature: float = 0.15,
        enable_rag: bool = True,
        context_length: int = 16000,
        rag_persist_directory: str = "./rag/chroma_db",
        rag_collection_name: str = "sysml_v2_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2",
        github_token: Optional[str] = None,
        enable_approved_solutions_db: bool = True,
        approved_solutions_persist_dir: str = "./rag/approved_solutions",
    ):
        """Initialize the SysML Workflow with simplified architecture"""
        # Initialize LLM
        self.temperature = temperature
        self.context_length = context_length

        self.common_llm_instance = (
            self._resolve_llm(common_llm, model_name) if common_llm else None
        )

        # RAG Configuration
        self.enable_rag = enable_rag
        self.rag_pipeline = None

        if enable_rag:
            try:
                logger.info("Initializing SysMLRAGPipeline...")
                self.rag_pipeline = SysMLRAGPipeline(
                    collection_name=rag_collection_name,
                    embedding_model=embedding_model,
                    persist_directory=rag_persist_directory,
                    github_token=github_token,
                )

                stats = self.rag_pipeline.get_collection_stats()
                if stats.get("total_chunks", 0) > 0:
                    logger.info(
                        f"✅ RAG pipeline initialized with {stats['total_chunks']} chunks"
                    )
                else:
                    logger.warning(
                        "⚠️ RAG database is empty. Run Injestion.py first to populate it."
                    )

            except Exception as e:
                logger.warning(f"⚠️ RAG initialization failed: {e}")
                self.enable_rag = False
                self.rag_pipeline = None
        else:
            logger.info("RAG system disabled")

        # Approved Solutions Database
        self.enable_approved_solutions = enable_approved_solutions_db
        self.approved_solutions_db = None
        self.approved_solutions_persist_dir = os.path.abspath(
            approved_solutions_persist_dir
        )
        self.embeddings = None

        if enable_approved_solutions_db:
            try:
                logger.info("Initializing approved solutions database...")

                # Initialize embeddings
                self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
                os.makedirs(approved_solutions_persist_dir, exist_ok=True)

                self.approved_solutions_db = Chroma(
                    persist_directory=approved_solutions_persist_dir,
                    collection_name="approved_solutions",
                    embedding_function=self.embeddings,
                )

                stats = self.get_approved_db_statistics()
                total = stats.get("total_entries", 0)

                if total > 0:
                    logger.info(f"✅ Loaded {total} approved solutions from disk")
                else:
                    logger.info("ℹ️ No existing approved solutions found in database")

                logger.info("✅ Approved solutions database initialized")

            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to initialize approved solutions database: {e}"
                )
                import traceback

                traceback.print_exc()
                self.enable_approved_solutions = False

        # Initialize agents
        logger.info("Initializing agents...")
        self.query_agent = QueryAgent(
            llm=self._get_agent_llm(query_agent_llm, model_name),
            config_path=agents_prompt,
        )
        self.template_agent = TemplateAgent(
            llm=self._get_agent_llm(template_agent_llm, model_name),
            config_path=agents_prompt,
        )
        self.sysml_agent = SysMLAgent(
            llm=self._get_agent_llm(sysml_agent_llm, model_name),
            config_path=agents_prompt,
            rag_knowledge_base=self.rag_pipeline if self.rag_pipeline else None,
        )
        self.approval_agent = HumanApprovalAgent()
        logger.info("✅ All agents initialized successfully")

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _get_agent_llm(
        self, agent_llm: Optional[Union[str, Dict, Any]], default_model: str
    ):
        """Get LLM for an agent with fallback logic"""
        if agent_llm is not None:
            # Agent has specific LLM
            return self._resolve_llm(agent_llm, default_model)
        elif self.common_llm_instance is not None:
            # Use common LLM
            return self.common_llm_instance
        else:
            # Use default
            return self._resolve_llm(None, default_model)

    def _resolve_llm(self, provider: Optional[Union[str, Any]], default: str):
        """
        provider can be:
        - None  -> use default Ollama model
        - string model name ONLY -> assume local Ollama model
        - dict describing provider ex:
            {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.3}
            {"provider": "openrouter", "model": "qwen/qwen2-72b", "temperature": 0.2}
            {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929", "temperature": 0.1, "max_tokens": 8192}
            {"provider": "ollama", "model": "llama3.1:8b", "base_url": "http://localhost:11434", "temperature": 0.2, "num_ctx": 8192}
        - OR already initialized LLM object -> return as is
        """
        # already instantiated
        if hasattr(provider, "invoke"):
            return provider

        # default means ollama local
        if provider is None:
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=default, temperature=self.temperature, num_ctx=self.context_length
            )

        # provider is a string → ollama local
        if isinstance(provider, str):
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=provider,
                temperature=self.temperature,
                num_ctx=self.context_length,
            )

        # dict mode
        p = provider.get("provider")
        m = provider.get("model")

        if p == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),  # type: ignore
                model=m,
                temperature=provider.get("temperature", self.temperature),
            )

        if p == "openrouter":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),  # type: ignore
                base_url="https://openrouter.ai/api/v1",
                model=m,
                temperature=provider.get("temperature", self.temperature),
            )

        if p == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model_name=m, # type: ignore
                api_key=os.getenv("ANTHROPIC_API_KEY"),  # type: ignore
                timeout=provider.get("timeout", 30),
                temperature=provider.get("temperature", self.temperature),
                max_tokens_to_sample=provider.get("max_tokens", 4096),
                stop=None,
            )

        if p == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=m,
                temperature=provider.get("temperature", self.temperature),
                num_ctx=provider.get("num_ctx", self.context_length),
            )

        raise ValueError(f"Unknown provider type: {provider}")

    def _build_workflow(self) -> StateGraph:
        """Build the simplified LangGraph workflow"""
        workflow = StateGraph(WorkflowState)

        # Add nodes - simplified architecture
        workflow.add_node("query_processing", self._query_processing_node)
        workflow.add_node("template_generation", self._template_generation_node)
        workflow.add_node("code_generation", self._code_generation_node)
        workflow.add_node("human_approval", self._human_approval_node)
        workflow.add_node("update_knowledge_base", self._update_knowledge_base)

        # Define workflow edges
        workflow.set_entry_point("query_processing")

        workflow.add_edge("query_processing", "template_generation")

        workflow.add_edge("template_generation", "code_generation")

        # SysMLAgent now handles validation and corrections internally
        workflow.add_conditional_edges(
            "code_generation",
            self._route_after_code_generation,
            {"approve": "human_approval", "retry": "code_generation", "end": END},
        )

        workflow.add_conditional_edges(
            "human_approval",
            self._route_after_approval,
            {
                "approved": "update_knowledge_base",
                "retry": "code_generation",
                "end": END,
            },
        )

        workflow.add_edge("update_knowledge_base", END)

        return workflow.compile()  # type: ignore

    def _ensure_workflow_state(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> WorkflowState:
        """Safely convert input to WorkflowState instance"""
        if isinstance(state, WorkflowState):
            return state
        elif isinstance(state, dict):
            return WorkflowState.from_state(state)
        else:
            if hasattr(state, "model_dump"):
                return WorkflowState.from_state(state.model_dump())
            elif hasattr(state, "dict"):
                return WorkflowState.from_state(state.dict())
            else:
                raise TypeError(f"Cannot convert {type(state)} to WorkflowState")

    def _query_processing_node(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Query processing node - analyzes and structures the user query"""
        workflow_state = self._ensure_workflow_state(state)
        logger.info("🔍 Processing user query...")

        updated_state = self.query_agent.process(workflow_state)
        logger.info("✅ Query processed and structured")

        return updated_state.to_dict()

    def _template_generation_node(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Template generation node - creates initial SysML template"""
        workflow_state = self._ensure_workflow_state(state)

        logger.info(
            f"📝 Generating SysML template (iteration {workflow_state.iteration})"
        )

        updated_state = self.template_agent.generate_template_code(workflow_state)

        if updated_state.code:
            logger.info("✅ Template generated successfully")
        else:
            logger.warning("⚠️ Template generation produced no code")

        return updated_state.to_dict()

    def _code_generation_node(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Code generation node - SysML Agent refines template
        SysMLAgent has access to RAG tool for enhancement and validation tools
        """
        workflow_state = self._ensure_workflow_state(state)

        logger.info(
            f"🔧 Refining SysML code with SysMLAgent (iteration {workflow_state.iteration})"
        )

        updated_state = self.sysml_agent.generate_code(workflow_state)

        return updated_state.to_dict()

    def _human_approval_node(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Human approval node"""
        workflow_state = self._ensure_workflow_state(state)
        updated_state = self.approval_agent.request_approval(workflow_state)
        return updated_state.to_dict()

    def _route_after_code_generation(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> str:
        """
        Route after code generation.
        Since SysMLAgent handles validation internally, we check if code is valid
        and route accordingly.
        """
        workflow_state = self._ensure_workflow_state(state)

        is_code_valid = False

        if workflow_state.is_valid == ValidationStatus.VALID:
            is_code_valid = True
            logger.info("✅ Code validation status: VALID")

        else:
            latest_validation = workflow_state.get_latest_validation()

            if latest_validation and latest_validation.success:
                logger.info("✅ Code validated successfully via validation history")
            else:
                logger.warning(
                    f"⚠️ Code validation status: {workflow_state.is_valid}, "
                    f"Validation history: {len(workflow_state.validation_history)} entries"
                )

        if is_code_valid:
            logger.info("✅ Code validated successfully, routing to human approval")
            return "approve"
        elif workflow_state.iteration < workflow_state.max_iterations:
            logger.info(
                f"🔄 Validation failed, retrying from template generation "
                f"(iteration {workflow_state.iteration}/{workflow_state.max_iterations})"
            )
            workflow_state.iteration += 1
            return "retry"
        else:
            logger.warning(
                f"⚠️ Max iterations reached ({workflow_state.max_iterations}), ending workflow"
            )
            return "end"

    def _route_after_approval(self, state: Union[WorkflowState, Dict[str, Any]]) -> str:
        """Route after human approval"""
        workflow_state = self._ensure_workflow_state(state)

        if workflow_state.approval_status == "approved":
            logger.info("✅ Code approved, updating knowledge base")
            return "approved"
        elif (
            workflow_state.approval_status in ["rejected", "feedback"]
            and workflow_state.iteration < workflow_state.max_iterations
        ):
            logger.info("🔄 Code rejected/feedback provided, retrying generation")
            workflow_state.iteration += 1
            return "retry"
        else:
            logger.info("ℹ️ Ending workflow after approval phase")
            return "end"

    def _update_knowledge_base(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update approved solutions knowledge base"""
        workflow_state = self._ensure_workflow_state(state)

        try:
            if (
                self.enable_approved_solutions
                and self.approved_solutions_db
                and workflow_state.approval_status == "approved"
            ):
                logger.info("💾 Storing approved solution in database...")

                # Create unique ID
                solution_id = f"sysml_{abs(hash(workflow_state.code))}_{int(datetime.now().timestamp())}"

                # Prepare metadata
                metadata = {
                    "solution_id": solution_id,
                    "task": workflow_state.original_query,
                    "human_feedback": getattr(workflow_state, "human_feedback", ""),
                    "iterations_used": workflow_state.iteration,
                    "max_iterations": workflow_state.max_iterations,
                    "created_at": datetime.now().isoformat(),
                    "validation_success": True,
                    "doc_type": "approved_solution",
                }

                # Create embedding text
                embedding_text = (
                    f"Task: {workflow_state.original_query}\n\n"
                    f"SysML Code:\n{workflow_state.code}"
                )

                # Create document
                doc = Document(page_content=embedding_text, metadata=metadata)

                # Add to vector store
                logger.info(f"Adding document with ID: {solution_id}")
                self.approved_solutions_db.add_documents(
                    documents=[doc], ids=[solution_id]
                )

                # Save JSON backup
                full_entry = {
                    "id": solution_id,
                    "task": workflow_state.original_query,
                    "generated_code": workflow_state.code,
                    "human_feedback": metadata["human_feedback"],
                    "workflow_metadata": {
                        "iterations_used": workflow_state.iteration,
                        "max_iterations": workflow_state.max_iterations,
                        "workflow_status": "human_approved",
                    },
                    "created_at": metadata["created_at"],
                }

                full_entry_dir = os.path.join(
                    os.path.abspath(self.approved_solutions_persist_dir), "full_entries"
                )
                os.makedirs(full_entry_dir, exist_ok=True)

                json_path = os.path.join(full_entry_dir, f"{solution_id}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(full_entry, f, indent=2, ensure_ascii=False)

                # Verify the entry was added
                stats = self.get_approved_db_statistics()
                logger.info(
                    f"✅ Successfully stored approved solution! (Total: {stats.get('total_entries', 0)})"
                )

        except Exception as e:
            logger.error(f"❌ Error in knowledge base update: {str(e)}")
            import traceback

            traceback.print_exc()

        return workflow_state.to_dict()

    def run(self, query: str, max_iterations: int = 5) -> Dict[str, Any]:
        """Run the enhanced workflow"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 Starting SysML Workflow")
        logger.info(f"Query: {query}")
        logger.info(f"Max iterations: {max_iterations}")
        logger.info(f"RAG enabled: {self.enable_rag}")
        logger.info(f"{'='*80}\n")

        initial_state = WorkflowState(
            original_query=query, processed_query=query, max_iterations=max_iterations
        )

        try:
            result_dict = self.workflow.invoke(initial_state.to_dict(), {"recursion_limit": 100})  # type: ignore
            result_state = WorkflowState.from_state(result_dict)

            final_result = result_state.to_dict()
            final_result["success"] = True

            logger.info(f"\n{'='*80}")
            logger.info(f"✅ Workflow completed successfully")
            logger.info(f"Total iterations: {result_state.iteration}")
            logger.info(f"Approval status: {result_state.approval_status}")
            logger.info(f"{'='*80}\n")

            return final_result

        except Exception as e:
            logger.error(f"\n{'='*80}")
            logger.error(f"❌ Workflow execution error: {str(e)}")
            logger.error(f"{'='*80}\n")
            return {"success": False, "error": str(e)}

    def search_similar_solutions(
        self, query: str, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar approved solutions"""
        if not self.enable_approved_solutions or not self.approved_solutions_db:
            return []

        try:
            docs = self.approved_solutions_db.similarity_search(query, k=n_results)
            results = []

            for doc in docs:
                results.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                )

            return results
        except Exception as e:
            logger.error(f"Error searching similar solutions: {e}")
            return []

    def get_rag_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        if not self.enable_rag or not self.rag_pipeline:
            return {"enabled": False}

        try:
            stats = self.rag_pipeline.get_collection_stats()
            stats["enabled"] = True
            return stats
        except Exception as e:
            logger.error(f"Error getting RAG statistics: {e}")
            return {"enabled": True, "error": str(e)}

    def get_approved_db_statistics(self) -> Dict[str, Any]:
        """Get approved solutions database statistics"""
        if not self.enable_approved_solutions or not self.approved_solutions_db:
            return {"enabled": False}

        try:
            collection = self.approved_solutions_db._collection
            count = collection.count()
            sample_data = collection.get(limit=5)
            sample_ids = sample_data.get("ids", [])

            return {
                "enabled": True,
                "total_entries": count,
                "collection_name": "approved_solutions",
                "sample_ids": sample_ids[:3],
                "persist_directory": self.approved_solutions_persist_dir,
            }

        except Exception as e:
            logger.error(f"Error getting approved DB statistics: {e}")
            import traceback

            traceback.print_exc()
            return {"enabled": True, "error": str(e)}

    def cleanup(self):
        """Clean up all resources"""
        try:
            logger.info("🧹 Starting cleanup...")

            if self.enable_approved_solutions and self.approved_solutions_db:
                try:
                    stats = self.get_approved_db_statistics()
                    logger.info(
                        f"✅ Approved solutions: {stats.get('total_entries', 0)} entries saved"
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Error checking approved solutions: {e}")

            if self.rag_pipeline:
                logger.info("✅ RAG pipeline cleaned up (auto-persisted)")

            logger.info("✅ Cleanup completed successfully")

        except Exception as e:
            logger.warning(f"⚠️ Error during cleanup: {e}")
