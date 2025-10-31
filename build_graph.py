from typing import Dict, Any, Optional, List, Union

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.schema import Document

from agents.ValidatorAgent import ValidatorAgent
from agents.QueryAgent import QueryAgent
from agents.HumanApprovalAgent import HumanApprovalAgent
from agents.SysMLAgent import SysMLAgent
from agents.CodeCorrectionAgent import CodeCorrectionAgent
from langgraph.graph import StateGraph, END
from states.WorkflowState import WorkflowState

from rag.Injestion import SysMLRAGPipeline

import logging
import os
from datetime import datetime
import json
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SysMLWorkflow:
    """
    Enhanced SysML workflow with RAG and vector database integration.
    """

    def __init__(
        self,
        config_path: str = "./prompt.yaml",
        model_name: str = "SysML-V2-llama3.1:latest",
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
        """Initialize the SysML Workflow with fixed approved solutions DB"""
        # Initialize LLM
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_ctx=context_length
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

        # FIXED: Approved Solutions Database with proper ChromaDB setup
        self.enable_approved_solutions = enable_approved_solutions_db
        self.approved_solutions_db = None
        self.approved_solutions_persist_dir = os.path.abspath(
            approved_solutions_persist_dir
        )
        self.embeddings = None
        self.chroma_client = None  # Store client reference

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
        self.query_agent = QueryAgent(self.llm, config_path=config_path)
        self.sysml_agent = SysMLAgent(
            llm=self.llm,
            config_path=config_path,
            knowledge_base=self.approved_solutions_db,
        )
        self.validator_agent = ValidatorAgent()
        self.correction_agent = CodeCorrectionAgent(
            self.llm,
            error_solutions_db=self.rag_pipeline,
        )
        self.approval_agent = HumanApprovalAgent()
        logger.info("✅ All agents initialized successfully")

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the enhanced LangGraph workflow"""
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("query_processing", self._query_processing_node)
        workflow.add_node("rag_enhancement", self._rag_enhancement_node)
        workflow.add_node("code_generation", self._code_generation_node)
        workflow.add_node("validation", self._validation_node)
        workflow.add_node("error_correction", self._error_correction_node)
        workflow.add_node("human_approval", self._human_approval_node)
        workflow.add_node("update_knowledge_base", self._update_knowledge_base)

        # Define workflow edges
        workflow.set_entry_point("query_processing")

        workflow.add_conditional_edges(
            "query_processing",
            self._route_after_query_processing,
            {"rag": "rag_enhancement", "generate": "code_generation"},
        )

        workflow.add_edge("rag_enhancement", "code_generation")
        workflow.add_edge("code_generation", "validation")

        workflow.add_conditional_edges(
            "validation",
            self._route_after_validation,
            {"approve": "human_approval", "correct": "error_correction", "end": END},
        )

        workflow.add_edge("error_correction", "code_generation")

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
        """Enhanced query processing node with approved solutions search"""
        workflow_state = self._ensure_workflow_state(state)

        if self.enable_approved_solutions and self.approved_solutions_db:
            try:
                similar_docs = self.approved_solutions_db.similarity_search(
                    workflow_state.original_query, k=3
                )

                if similar_docs:
                    logger.info(f"Found {len(similar_docs)} similar approved solutions")
                    context = "\n\nSimilar approved solutions:\n"
                    for i, doc in enumerate(similar_docs, 1):
                        preview = doc.page_content[:150].replace("\n", " ")
                        context += f"{i}. {preview}...\n"
                    workflow_state.processed_query += context

            except Exception as e:
                logger.warning(f"Error searching approved solutions: {e}")

        updated_state = self.query_agent.process(workflow_state)
        return updated_state.to_dict()

    def _rag_enhancement_node(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """RAG enhancement node using SysMLRAGPipeline"""
        workflow_state = self._ensure_workflow_state(state)

        if self.enable_rag and self.rag_pipeline:
            try:
                logger.info("🔍 Enhancing query with RAG context...")

                code_results = self.rag_pipeline.query(
                    workflow_state.original_query, n_results=3, filter_doc_type="code"
                )

                doc_results = self.rag_pipeline.query(
                    workflow_state.original_query,
                    n_results=2,
                    filter_doc_type="documentation",
                )

                if code_results or doc_results:
                    rag_context = "\n\n=== RELEVANT SYSML PATTERNS AND EXAMPLES ===\n"

                    if code_results:
                        rag_context += "\nCode Examples:\n"
                        for i, result in enumerate(code_results, 1):
                            content = result["content"][:300]
                            source = result["metadata"].get("source", "unknown")
                            rag_context += f"\n{i}. From {source}\n{content}...\n"

                    if doc_results:
                        rag_context += "\nRelevant Documentation:\n"
                        for i, result in enumerate(doc_results, 1):
                            content = result["content"][:250]
                            rag_context += f"\n{i}. {content}...\n"

                    workflow_state.processed_query += rag_context
                    logger.info(
                        f"✅ Added context from {len(code_results) + len(doc_results)} documents"
                    )

            except Exception as e:
                logger.error(f"❌ RAG enhancement error: {e}")

        return workflow_state.to_dict()

    def _code_generation_node(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enhanced code generation with RAG error context"""
        workflow_state = self._ensure_workflow_state(state)

        if workflow_state.iteration > 1 and self.enable_rag and self.rag_pipeline:
            try:
                logger.info("🔍 Searching for error-specific solutions...")

                error_docs = self.rag_pipeline.query(
                    f"error fix solution {workflow_state.error}",
                    n_results=3,
                    filter_doc_type="error_doc",
                )

                if error_docs:
                    error_context = "\n\n=== ERROR-SPECIFIC SOLUTIONS ===\n"
                    for i, result in enumerate(error_docs, 1):
                        content = result["content"][:300]
                        error_code = result["metadata"].get("error_code", "N/A")
                        error_context += f"\n{i}. [Error: {error_code}]\n{content}...\n"

                    workflow_state.processed_query += error_context
                    logger.info(f"✅ Added {len(error_docs)} error solutions from RAG")

            except Exception as e:
                logger.warning(f"⚠️ Error context enhancement failed: {e}")

        updated_state = self.sysml_agent.generate_code(workflow_state)
        return updated_state.to_dict()

    def _validation_node(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validation node using SysMLValidatorTool"""
        workflow_state = self._ensure_workflow_state(state)

        logger.info(f"🔍 Validating SysML code (iteration {workflow_state.iteration})")
        updated_state = self.validator_agent.validate(workflow_state)

        if updated_state.is_valid.value == "valid":
            logger.info("✅ Validation passed!")
        else:
            error_count = (
                len(updated_state.get_latest_validation().errors)  # type: ignore
                if updated_state.get_latest_validation()
                else 0
            )
            logger.warning(f"❌ Validation failed with {error_count} error(s)")

        return updated_state.to_dict()

    def _error_correction_node(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enhanced error correction with RAG"""
        workflow_state = self._ensure_workflow_state(state)

        logger.info(
            f"🔧 Analyzing errors for correction (iteration {workflow_state.iteration})"
        )

        updated_state = self.correction_agent.analyze_errors(workflow_state)

        if self.enable_rag and self.rag_pipeline and updated_state.error:
            try:
                error_solutions = self.rag_pipeline.query(
                    f"solution fix {updated_state.error}",
                    n_results=3,
                    filter_doc_type="error_doc",
                )

                if error_solutions:
                    additional_guidance = (
                        "\n\n=== ADDITIONAL ERROR SOLUTIONS FROM KNOWLEDGE BASE ===\n"
                    )
                    for i, result in enumerate(error_solutions, 1):
                        content = result["content"][:250]
                        error_code = result["metadata"].get("error_code", "Unknown")
                        additional_guidance += (
                            f"\n{i}. [Error {error_code}] {content}...\n"
                        )

                    updated_state.processed_query += additional_guidance

            except Exception as e:
                logger.warning(f"⚠️ RAG error correction enhancement failed: {e}")

        return updated_state.to_dict()

    def _human_approval_node(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Human approval node"""
        workflow_state = self._ensure_workflow_state(state)
        updated_state = self.approval_agent.request_approval(workflow_state)
        return updated_state.to_dict()

    def _route_after_query_processing(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> str:
        """Route after query processing"""
        return "rag" if self.enable_rag else "generate"

    def _route_after_validation(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> str:
        """Route after validation based on results"""
        workflow_state = self._ensure_workflow_state(state)

        latest_validation = workflow_state.get_latest_validation()
        if latest_validation and latest_validation.success:
            logger.info("✅ Routing to human approval")
            return "approve"
        elif workflow_state.iteration < workflow_state.max_iterations:
            logger.info(
                f"🔄 Routing to error correction (iteration {workflow_state.iteration}/{workflow_state.max_iterations})"
            )
            return "correct"
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
            return "retry"
        else:
            logger.info("ℹ️ Ending workflow after approval phase")
            return "end"

    def _update_knowledge_base(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """FIXED: Update approved solutions knowledge base"""
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

                # Create embedding text (combine task and code for better search)
                embedding_text = (
                    f"Task: {workflow_state.original_query}\n\n"
                    f"SysML Code:\n{workflow_state.code}"
                )

                # FIXED: Create document with proper structure
                doc = Document(page_content=embedding_text, metadata=metadata)

                # FIXED: Add to vector store with explicit ID
                logger.info(f"Adding document with ID: {solution_id}")
                self.approved_solutions_db.add_documents(
                    documents=[doc], ids=[solution_id]
                )

                # FIXED: Force persistence (ChromaDB should auto-persist, but ensure it)
                # Note: Newer ChromaDB versions don't have persist(), it auto-persists
                try:
                    if hasattr(self.approved_solutions_db, "persist"):
                        self.approved_solutions_db.persist()  # type: ignore
                except AttributeError:
                    # Auto-persisting version, no action needed
                    pass

                # Also save as JSON backup
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

                # Save JSON backup
                full_entry_dir = os.path.join(
                    os.path.abspath(self.approved_solutions_persist_dir), "full_entries"
                )
                os.makedirs(full_entry_dir, exist_ok=True)

                json_path = os.path.join(full_entry_dir, f"{solution_id}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(full_entry, f, indent=2, ensure_ascii=False)

                # FIXED: Verify the entry was added
                stats = self.get_approved_db_statistics()
                logger.info(
                    f"✅ Successfully stored approved solution! (Total: {stats.get('total_entries', 0)})"
                )

                # Test retrieval
                try:
                    search_results = self.approved_solutions_db.similarity_search(
                        workflow_state.original_query, k=1
                    )
                    if search_results:
                        logger.info(f"✅ Verified: Solution is retrievable")
                    else:
                        logger.warning(
                            "⚠️ Warning: Could not retrieve the just-added solution"
                        )
                except Exception as e:
                    logger.warning(f"⚠️ Verification search failed: {e}")

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
            result_dict = self.workflow.invoke(initial_state.to_dict())  # type: ignore
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
        """FIXED: Get approved solutions database statistics"""
        if not self.enable_approved_solutions or not self.approved_solutions_db:
            return {"enabled": False}

        try:
            # FIXED: Get count directly from the ChromaDB collection
            collection = self.approved_solutions_db._collection

            # Get the actual count
            count = collection.count()

            # Get sample data for verification
            sample_data = collection.get(limit=5)
            sample_ids = sample_data.get("ids", [])

            return {
                "enabled": True,
                "total_entries": count,
                "collection_name": "approved_solutions",
                "sample_ids": sample_ids[:3],  # Show first 3 IDs
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

            # FIXED: No need to explicitly persist with newer ChromaDB
            # It auto-persists, but we can verify the data exists
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

    def reset_approved_solutions_db(self):
        """Reset the approved solutions database (for testing)"""
        if not self.chroma_client:
            logger.error("❌ ChromaDB client not initialized")
            return

        try:
            # Delete the collection
            try:
                self.chroma_client.delete_collection("approved_solutions")
                logger.info("✅ Deleted old collection")
            except Exception:
                logger.info("ℹ️ No existing collection to delete")

            # Recreate the collection
            self.approved_solutions_db = Chroma(
                client=self.chroma_client,
                collection_name="approved_solutions",
                embedding_function=self.embeddings,
            )

            logger.info("✅ Approved solutions database reset successfully")

        except Exception as e:
            logger.error(f"❌ Error resetting approved solutions database: {e}")
            import traceback

            traceback.print_exc()


def test_enhanced_workflow():
    """Test the enhanced workflow"""
    print("=" * 80)
    print("ENHANCED SYSML WORKFLOW TEST")
    print("=" * 80)

    workflow = None
    try:
        print("\n1. Initializing Enhanced SysML Workflow...")

        workflow = SysMLWorkflow(
            enable_rag=True,
            model_name="gpt-oss:20b-cloud",
            temperature=0.15,
            config_path="agents/prompt/prompt.yaml",
            rag_persist_directory="./rag/chroma_db",
            rag_collection_name="sysml_v2_knowledge",
            enable_approved_solutions_db=False,
            approved_solutions_persist_dir="./rag/approved_solutions",
            context_length=16000
        )
        print("✅ Enhanced workflow initialized successfully")

        # Display system status
        print("\n2. System Status:")
        print(
            f"   RAG System: {'✅ Enabled' if workflow.enable_rag else '❌ Disabled'}"
        )
        print(
            f"   Approved Solutions DB: {'✅ Enabled' if workflow.enable_approved_solutions else '❌ Disabled'}"
        )

        if workflow.enable_rag:
            rag_stats = workflow.get_rag_statistics()
            print(f"\n   RAG Pipeline Statistics:")
            print(f"   - Total chunks: {rag_stats.get('total_chunks', 0)}")

        if workflow.enable_approved_solutions:
            approved_stats = workflow.get_approved_db_statistics()
            print(f"\n   Approved Solutions Database:")
            print(f"   - Total entries: {approved_stats.get('total_entries', 0)}")
            if approved_stats.get("sample_ids"):
                print(f"   - Sample IDs: {approved_stats['sample_ids']}")

        # Test query
        test_query = "Create a SysML package for a simple drone"
        print(f"\n3. Running workflow with test query:")
        print(f"   '{test_query}'")

        result = workflow.run(test_query, max_iterations=5)

        if result.get("success"):
            print("\n4. Workflow Results:")
            print(f"   Status: ✅ Success")
            print(f"   Iterations: {result.get('iteration', 0)}")
            print(f"   Approval Status: {result.get('approval_status', 'unknown')}")

            if result.get("code"):
                print(f"   Final Code Length: {len(result['code'])} characters")
                print("\n5. Generated Code Preview:")
                print("-" * 50)
                code_lines = result["code"].split("\n")[:15]
                for line in code_lines:
                    print(f"   {line}")
                if len(result["code"].split("\n")) > 15:
                    print("   ... (truncated)")
                print("-" * 50)
        else:
            print(f"\n4. ❌ Workflow failed: {result.get('error', 'Unknown error')}")

        # Check updated statistics
        if workflow.enable_approved_solutions:
            print("\n6. Updated Approved Solutions Database:")
            updated_stats = workflow.get_approved_db_statistics()
            print(f"   - Total entries: {updated_stats.get('total_entries', 0)}")
            if updated_stats.get("sample_ids"):
                print(f"   - Sample IDs: {updated_stats['sample_ids']}")

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()

    finally:
        print("\n7. Cleanup:")
        if workflow:
            workflow.cleanup()
        print("   ✅ Resources cleaned up successfully")


if __name__ == "__main__":
    test_enhanced_workflow()
