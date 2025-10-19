from typing import Dict, Any, Optional, List, Union

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain.schema import Document

from ValidatorAgent import ValidatorAgent
from QueryAgent import QueryAgent
from HumanApprovalAgent import HumanApprovalAgent
from SysMLAgent import SysMLAgent
from CodeCorrectionAgent import CodeCorrectionAgent
from langgraph.graph import StateGraph, END
from States import WorkflowState, ApprovedCodeEntry

# Import the new components
from KnowledgeBase import KnowledgeBase
from RAG import SysMLRetriever

import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedSysMLWorkflow:
    """Enhanced SysML workflow with RAG and vector database integration"""

    def __init__(
        self,
        config_path: str = "./prompt.yaml",
        model_name: str = "SysML-V2-llama3.1:latest",
        temperature: float = 0.15,
        rag_data_path: str = "./rag_data",
        knowledge_base_config: Optional[Dict[str, Any]] = None,
        enable_vector_storage: bool = True,
        enable_rag: bool = True,
        force_rebuild_rag: bool = False,
    ):
        # Initialize LLM
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            validate_model_on_init=True,
        )

        # RAG Configuration
        self.enable_rag = enable_rag
        self.rag_retriever = None

        if enable_rag and os.path.exists(rag_data_path):
            try:
                logger.info("Initializing RAG system...")
                self.rag_retriever = SysMLRetriever(
                    file_paths=rag_data_path,
                    chunk_size=512,
                    chunk_overlap=128,
                    force_rebuild=force_rebuild_rag,
                    cache_dir="./agents/cache",
                )
                logger.info("RAG system initialized successfully")
            except Exception as e:
                logger.warning(f"RAG initialization failed: {e}")
                self.enable_rag = False
        else:
            logger.info("RAG disabled or data path not found")
            self.enable_rag = False

        # Vector Database Configuration
        self.enable_vector_storage = enable_vector_storage
        self.knowledge_base_db = None

        if enable_vector_storage:
            try:
                db_config = knowledge_base_config or {}
                self.knowledge_base_db = KnowledgeBase(**db_config)
                logger.info("Vector database initialized successfully")
            except Exception as e:
                logger.warning(f"Vector database initialization failed: {e}")
                self.enable_vector_storage = False

        # Initialize traditional vector stores (legacy support)
        self.embeddings = HuggingFaceEmbeddings()
        self.error_solutions_db = self.rag_retriever

        # Initialize fallback knowledge base for approved solutions
        self.knowledge_base = None
        try:
            self.knowledge_base = Chroma(
                collection_name="approved_solutions",
                embedding_function=self.embeddings,
                persist_directory="./agents/chroma_db",
            )
        except Exception as e:
            logger.warning(f"Failed to initialize fallback knowledge base: {e}")

        # Initialize agents with enhanced capabilities
        self.query_agent = QueryAgent(self.llm, config_path=config_path)
        self.sysml_agent = SysMLAgent(
            llm=self.llm, config_path=config_path, knowledge_base=self.knowledge_base_db
        )
        self.validator_agent = ValidatorAgent()
        self.correction_agent = CodeCorrectionAgent(self.llm, self.error_solutions_db)
        self.approval_agent = HumanApprovalAgent()

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

        # Route to RAG enhancement if enabled
        workflow.add_conditional_edges(
            "query_processing",
            self._route_after_query_processing,
            {"rag": "rag_enhancement", "generate": "code_generation"},
        )

        workflow.add_edge("rag_enhancement", "code_generation")
        workflow.add_edge("code_generation", "validation")

        # Conditional logic after validation
        workflow.add_conditional_edges(
            "validation",
            self._route_after_validation,
            {"approve": "human_approval", "correct": "error_correction", "end": END},
        )

        workflow.add_edge("error_correction", "code_generation")

        # Conditional logic after human approval
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
        """
        Safely convert input to WorkflowState instance.
        Handles both dict and WorkflowState inputs without double conversion.
        """
        if isinstance(state, WorkflowState):
            return state
        elif isinstance(state, dict):
            return WorkflowState.from_state(state)
        else:
            # Handle pydantic models
            if hasattr(state, "model_dump"):
                return WorkflowState.from_state(state.model_dump())
            elif hasattr(state, "dict"):
                return WorkflowState.from_state(state.dict())
            else:
                raise TypeError(f"Cannot convert {type(state)} to WorkflowState")

    def _query_processing_node(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enhanced query processing node"""
        # Convert to WorkflowState safely
        workflow_state = self._ensure_workflow_state(state)

        # Check for similar solutions first
        if self.enable_vector_storage and self.knowledge_base_db:
            try:
                similar_solutions = self.knowledge_base_db.search_similar_entries(
                    workflow_state.original_query, n_results=3
                )
                if similar_solutions:
                    logger.info(f"Found {len(similar_solutions)} similar solutions")
                    # Add similar solutions context to the processed query
                    context = "\n\nSimilar approved solutions found:\n"
                    for i, sol in enumerate(similar_solutions[:2], 1):
                        context += f"{i}. {sol['task'][:100]}... (similarity: {sol['similarity_score']:.3f})\n"
                    workflow_state.processed_query += context
            except Exception as e:
                logger.warning(f"Error searching similar solutions: {e}")

        updated_state = self.query_agent.process(workflow_state)
        return updated_state.to_dict()

    def _rag_enhancement_node(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """RAG enhancement node to add relevant context"""
        workflow_state = self._ensure_workflow_state(state)

        if self.enable_rag and self.rag_retriever:
            try:
                logger.info("Enhancing query with RAG context...")

                # Search for relevant examples and error patterns
                relevant_docs = self.rag_retriever.query(
                    workflow_state.original_query, k=5, mode="hybrid"
                )

                if relevant_docs:
                    # Add RAG context to processed query
                    rag_context = "\n\nRelevant SysML patterns and examples:\n"
                    for i, doc in enumerate(relevant_docs, 1):
                        content = doc.page_content
                        # Clean ranking prefix if present
                        if content.startswith("[Rank"):
                            content = (
                                content.split("\n", 1)[1]
                                if "\n" in content
                                else content
                            )

                        rag_context += f"\n{i}. {content[:300]}...\n"

                        # Add metadata context if available
                        if hasattr(doc, "metadata") and doc.metadata.get("error_id"):
                            rag_context += (
                                f"   (Error ID: {doc.metadata['error_id']})\n"
                            )

                    workflow_state.processed_query += rag_context
                    logger.info(
                        f"Added context from {len(relevant_docs)} relevant documents"
                    )
                else:
                    logger.info("No relevant RAG documents found")

            except Exception as e:
                logger.error(f"RAG enhancement error: {e}")

        return workflow_state.to_dict()

    def _code_generation_node(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enhanced code generation with RAG context"""
        workflow_state = self._ensure_workflow_state(state)

        # Add error-specific context if this is a correction iteration
        if workflow_state.iteration > 1 and self.enable_rag and self.rag_retriever:
            try:
                # Search for specific error solutions
                error_docs = self.rag_retriever.query(
                    f"error solution {workflow_state.error}", k=3, mode="rerank"
                )

                if error_docs:
                    error_context = "\n\nError-specific solutions:\n"
                    for doc in error_docs:
                        content = doc.page_content
                        if content.startswith("[Rank"):
                            content = (
                                content.split("\n", 1)[1]
                                if "\n" in content
                                else content
                            )
                        error_context += f"- {content[:200]}...\n"

                    workflow_state.processed_query += error_context

            except Exception as e:
                logger.warning(f"Error context enhancement failed: {e}")

        updated_state = self.sysml_agent.generate_code(workflow_state)
        return updated_state.to_dict()

    def _validation_node(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Wrapper node for validation"""
        workflow_state = self._ensure_workflow_state(state)
        updated_state = self.validator_agent.validate(workflow_state)
        return updated_state.to_dict()

    def _error_correction_node(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enhanced error correction with RAG"""
        workflow_state = self._ensure_workflow_state(state)

        # First, use traditional correction agent
        updated_state = self.correction_agent.analyze_errors(workflow_state)

        # Then enhance with RAG error solutions if available
        if self.enable_rag and self.rag_retriever and updated_state.error:
            try:
                # Search for specific error solutions
                error_solutions = self.rag_retriever.query(
                    f"solution fix {updated_state.error}", k=3, mode="rerank"
                )

                if error_solutions:
                    additional_guidance = (
                        "\n\nAdditional error solutions from knowledge base:\n"
                    )
                    for sol in error_solutions:
                        content = sol.page_content
                        if content.startswith("[Rank"):
                            content = (
                                content.split("\n", 1)[1]
                                if "\n" in content
                                else content
                            )
                        additional_guidance += f"- {content[:250]}...\n"

                    updated_state.processed_query += additional_guidance

            except Exception as e:
                logger.warning(f"RAG error correction enhancement failed: {e}")

        return updated_state.to_dict()

    def _human_approval_node(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enhanced human approval node"""
        workflow_state = self._ensure_workflow_state(state)
        updated_state = self.approval_agent.request_approval(workflow_state)

        # Store approved solutions in vector database
        if (
            self.enable_vector_storage
            and self.knowledge_base_db
            and updated_state.approval_status == "approved"
        ):
            try:
                logger.info("Storing approved solution in vector database...")

                # Create approved entry
                approved_entry = ApprovedCodeEntry.from_workflow_state(updated_state)

                # Store in vector database
                success = self.knowledge_base_db.store_approved_entry(approved_entry)

                if success:
                    logger.info("Successfully stored approved solution!")
                else:
                    logger.warning("Failed to store approved solution")

            except Exception as e:
                logger.error(f"Error storing approved solution: {e}")

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
            return "approve"
        elif workflow_state.iteration < workflow_state.max_iterations:
            return "correct"
        else:
            return "end"

    def _route_after_approval(self, state: Union[WorkflowState, Dict[str, Any]]) -> str:
        """Route after human approval"""
        workflow_state = self._ensure_workflow_state(state)

        if workflow_state.approval_status == "approved":
            return "approved"
        elif (
            workflow_state.approval_status in ["rejected", "feedback"]
            and workflow_state.iteration < workflow_state.max_iterations
        ):
            return "retry"
        else:
            return "end"

    def _update_knowledge_base(
        self, state: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update both traditional and vector knowledge bases"""
        workflow_state = self._ensure_workflow_state(state)

        try:
            logger.info("Updating knowledge bases with approved solution")

            # Update traditional knowledge base if available
            if self.knowledge_base is not None:
                doc = Document(
                    page_content=f"Query: {workflow_state.original_query}\nSolution: {workflow_state.code}",
                    metadata={
                        "approved": True,
                        "timestamp": datetime.now().isoformat(),
                        "query_type": "sysml_generation",
                        "iterations": workflow_state.iteration,
                    },
                )
                self.knowledge_base.add_documents([doc])
                logger.info("Traditional knowledge base updated successfully")
            else:
                logger.warning("Traditional knowledge base not available for update")

            # Vector database should already be updated in human_approval_node
            if self.enable_vector_storage and self.knowledge_base_db:
                logger.info("Vector database was already updated in approval phase")

            print("Knowledge bases updated successfully!")

        except Exception as e:
            logger.error(f"Error updating knowledge bases: {str(e)}")

        return workflow_state.to_dict()

    def run(self, query: str, max_iterations: int = 5) -> Dict[str, Any]:
        """Run the enhanced workflow"""
        initial_state = WorkflowState(
            original_query=query, processed_query=query, max_iterations=max_iterations
        )

        try:
            # Pass the initial state as a dict to the workflow
            result_dict = self.workflow.invoke(initial_state.to_dict()) # type: ignore

            # Convert result back to WorkflowState for processing
            result_state = WorkflowState.from_state(result_dict)

            # Return the final result as dict
            final_result = result_state.to_dict()
            final_result["success"] = True
            return final_result

        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}")
            return {"success": False, "error": str(e)}

    def search_similar_solutions(
        self, query: str, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar approved solutions"""
        if not self.enable_vector_storage or not self.knowledge_base_db:
            return []

        try:
            return self.knowledge_base_db.search_similar_entries(
                query, n_results=n_results
            )
        except Exception as e:
            logger.error(f"Error searching similar solutions: {e}")
            return []

    def get_rag_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        if not self.enable_rag or not self.rag_retriever:
            return {"enabled": False}

        try:
            stats = self.rag_retriever.get_statistics()
            stats["enabled"] = True
            return stats
        except Exception as e:
            logger.error(f"Error getting RAG statistics: {e}")
            return {"enabled": True, "error": str(e)}

    def get_vector_db_statistics(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        if not self.enable_vector_storage or not self.knowledge_base_db:
            return {"enabled": False}

        try:
            stats = self.knowledge_base_db.get_collection_stats()
            stats["enabled"] = True
            return stats
        except Exception as e:
            logger.error(f"Error getting vector DB statistics: {e}")
            return {"enabled": True, "error": str(e)}

    def cleanup(self):
        """Clean up all resources"""
        try:
            if hasattr(self, "validator_agent") and self.validator_agent:
                self.validator_agent.cleanup()
            if hasattr(self, "knowledge_base_db") and self.knowledge_base_db:
                self.knowledge_base_db.cleanup()
            if hasattr(self, "knowledge_base") and self.knowledge_base:
                # Persist Chroma database
                try:
                    self.knowledge_base.persist()
                except Exception as e:
                    logger.warning(f"Error persisting knowledge base: {e}")
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


def test_enhanced_workflow():
    """Test the enhanced workflow with all features"""
    print("=" * 80)
    print("ENHANCED SYSML WORKFLOW TEST")
    print("=" * 80)

    workflow = None
    try:
        # Initialize enhanced workflow
        print("\n1. Initializing Enhanced SysML Workflow...")
        workflow = EnhancedSysMLWorkflow(
            enable_rag=True,
            enable_vector_storage=True,
            force_rebuild_rag=False,  # Set to True to rebuild RAG cache
        )
        print("Enhanced workflow initialized successfully")

        # Display system status
        print("\n2. System Status:")
        print(f"   RAG System: {'Enabled' if workflow.enable_rag else 'Disabled'}")
        print(
            f"   Vector DB: {'Enabled' if workflow.enable_vector_storage else 'Disabled'}"
        )

        if workflow.enable_rag:
            rag_stats = workflow.get_rag_statistics()
            print(f"   RAG Documents: {rag_stats.get('total_documents', 0)}")
            print(f"   Error Documents: {rag_stats.get('error_documents', 0)}")

        if workflow.enable_vector_storage:
            vector_stats = workflow.get_vector_db_statistics()
            print(f"   Vector DB Entries: {vector_stats.get('total_entries', 0)}")

        # Test query
        test_query = "Create a SysML block definition for an automotive brake system with ABS functionality"
        print(f"\n3. Running enhanced workflow with query:")
        print(f"   '{test_query}'")

        # Run workflow
        result = workflow.run(test_query, max_iterations=3)

        if result["success"]:
            print("\n4. Workflow Results:")
            print(f"   Status: Success")
            print(f"   Iterations: {result['iterations']}")
            print(f"   Approval Status: {result['approval_status']}")
            print(f"   Success Rate: {result['success_rate']:.2%}")
            print(f"   Final Code Length: {len(result['final_code'])} characters")
            print(
                f"   Latest Validation: {'Success' if result['latest_validation_success'] else 'Failed'}"
            )

            if result["final_code"]:
                print("\n5. Generated Code Preview:")
                print("-" * 50)
                code_lines = result["final_code"].split("\n")[:15]
                for line in code_lines:
                    print(f"   {line}")
                if len(result["final_code"].split("\n")) > 15:
                    print("   ... (truncated)")
                print("-" * 50)
        else:
            print(f"\n4. Workflow failed: {result['error']}")

        # Test similarity search
        print("\n6. Testing Similarity Search:")
        similar = workflow.search_similar_solutions(
            "automotive system brake", n_results=3
        )
        print(f"   Found {len(similar)} similar solutions")

        for i, sol in enumerate(similar, 1):
            print(
                f"   {i}. {sol['task'][:80]}... (similarity: {sol['similarity_score']:.3f})"
            )

    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print("\n7. Cleanup:")
        if workflow:
            workflow.cleanup()
        print("   Resources cleaned up successfully")


if __name__ == "__main__":
    test_enhanced_workflow()
