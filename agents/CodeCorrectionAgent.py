from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

import logging
import yaml

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from states.WorkflowState import WorkflowState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeCorrectionAgent:
    """
    Agent responsible for analyzing errors and providing correction guidance.

    Updated to work with new SysMLRAGPipeline instead of old vector store.
    """

    def __init__(
        self,
        llm,
        error_solutions_db=None,  # Can be SysMLRAGPipeline instance
        config_path: str = "agents/prompt/prompt.yaml",
    ):
        self.llm = llm
        self.error_solutions_db = error_solutions_db  # SysMLRAGPipeline or None
        self.system_prompt = self._load_system_prompt(config_path)

    def _load_system_prompt(self, config_path: str) -> str:
        """Load system prompt from a YAML config file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"System prompt loaded from {config_path}")
            prompt = config.get("Code-Correction-Agent", "")
            if not prompt:
                raise ValueError(
                    f"'Code-Correction-Agent' key not found or empty in {config_path}."
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

    def analyze_errors(self, state: WorkflowState) -> WorkflowState:
        """Analyze errors and provide correction guidance"""
        try:
            logger.info("Code Correction Agent: Analyzing errors")

            # Get relevant solutions from error database (new RAG pipeline)
            solution_context = self._get_error_solutions(state.error)

            agent = create_agent(
                name="CodeCorrectionAgent",
                model=self.llm,
                tools=[],
                system_prompt=self.system_prompt,
            )
            response = agent.invoke(
                (
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"""These are the following: 
                                
                                Current Errors: {state.error}
                                
                                SysML Code with Issues: ```{state.code}```
                                
                                Error Solutions Database Context: {solution_context}
                                
                                Analyze each error and provide:
                                1. Root cause analysis
                                2. Specific correction steps
                                3. Code examples for fixes
                                4. Prevention strategies
                                                
                            Focus on actionable, specific guidance that the SysML Agent can implement directly.
                                                
                            Correction Guidance:
                        """,
                            }
                        ]
                    }
                )
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

            ### Old Chaining Code
            # prompt = ChatPromptTemplate.from_messages(
            #     [
            #         ("system", self.system_prompt),
            #         (
            #             "human",
            #             """These are the following:

            #                 Current Errors: {errors}

            #                 SysML Code with Issues: {sysml_code}

            #                 Error Solutions Database Context: {solution_context}

            #                 Analyze each error and provide:
            #                     1. Root cause analysis
            #                     2. Specific correction steps
            #                     3. Code examples for fixes
            #                     4. Prevention strategies

            #                 Focus on actionable, specific guidance that the SysML Agent can implement directly.

            #                 Correction Guidance:
            #             """,
            #         ),
            #     ]
            # )

            # chain = prompt | self.llm | StrOutputParser()

            # response = chain.invoke(
            #     {
            #         "errors": state.error,
            #         "sysml_code": state.code,
            #         "solution_context": solution_context,
            #     }
            # )

            # Clean up any model-specific tokens
            cleaned_response = (
                response.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
            )

            # Store correction guidance for the SysML agent
            state.processed_query += f"\n\nCORRECTION GUIDANCE:\n{cleaned_response}"

            logger.info("Code Correction Agent: Error analysis completed")

        except Exception as e:
            logger.error(f"Code Correction Agent error: {str(e)}")

        return state

    def _get_error_solutions(self, errors: str) -> str:
        """
        Retrieve relevant error solutions from database.

        Updated to work with new SysMLRAGPipeline which has a different API.
        """
        if not errors or not errors.strip():
            return "No errors provided for solution lookup."

        try:
            # Check if error_solutions_db is the new SysMLRAGPipeline
            if hasattr(self.error_solutions_db, "query"):
                # New SysMLRAGPipeline interface
                logger.info("Using SysMLRAGPipeline for error solutions")

                # Query error documentation specifically
                results = self.error_solutions_db.query(  # type: ignore
                    query_text=f"error solution fix {errors}",
                    n_results=3,
                    filter_doc_type="error_doc",
                )

                if results:
                    solution_texts = []
                    for result in results:
                        content = result.get("content", "")
                        metadata = result.get("metadata", {})
                        error_code = metadata.get("error_code", "N/A")

                        solution_texts.append(
                            f"[Error Code: {error_code}]\n{content}\n"
                        )

                    return "\n\n".join(solution_texts)
                else:
                    logger.info("No specific error solutions found in RAG database")
                    return "No specific error solutions found in database."

            # Fallback for old similarity_search interface (if still using legacy DB)
            elif hasattr(self.error_solutions_db, "similarity_search"):
                logger.info("Using legacy vector store for error solutions")
                docs = self.error_solutions_db.similarity_search(errors, k=3)  # type: ignore
                return "\n\n".join([doc.page_content for doc in docs])

            else:
                logger.warning("Error solutions database not properly configured")
                return "Error solutions database not available."

        except Exception as e:
            logger.warning(f"Error retrieving solutions from database: {e}")
            return f"Error accessing solutions database: {str(e)}"


def test_analyze_errors():
    """Test the analyze_errors method with sample data"""
    print("=" * 80)
    print("TESTING CodeCorrectionAgent.analyze_errors")
    print("=" * 80)

    # Create agent instance without RAG (will work without it)
    agent = CodeCorrectionAgent(
        llm=ChatOllama(
            model="qwen3-coder:480b-cloud",
            temperature=0.15,
        ),
        error_solutions_db=None,  # Test without RAG first
    )

    # Create test state with sample errors and code
    test_state = WorkflowState(
        error="Missing semicolon after part definition on line 3",
        code="""package TestSystem {
    part def Engine;
    part def Transmission
}""",
        processed_query="Original query: Create a simple automotive system",
    )

    print("\n1. Initial State:")
    print(f"   Errors: {test_state.error}")
    print(f"   Code:\n{test_state.code}")
    print(f"   Processed Query: {test_state.processed_query}")
    print("\n" + "=" * 80 + "\n")

    # Test the analyze_errors method
    print("2. Running Error Analysis...")
    result_state = agent.analyze_errors(test_state)

    print("\n3. Result State:")
    print(f"   Errors: {result_state.error}")
    print(f"   Code:\n{result_state.code}")
    print(f"\n   Updated Processed Query:")
    print("   " + "-" * 70)
    # Show correction guidance
    if "CORRECTION GUIDANCE:" in result_state.processed_query:
        guidance_start = result_state.processed_query.index("CORRECTION GUIDANCE:")
        guidance = result_state.processed_query[guidance_start:]
        for line in guidance.split("\n")[:20]:  # Show first 20 lines
            print(f"   {line}")
    print("   " + "-" * 70)

    # Verify that correction guidance was added
    assert (
        "CORRECTION GUIDANCE:" in result_state.processed_query
    ), "Correction guidance should be added"
    print("\n✅ Correction guidance was added to processed_query")

    print("\n" + "=" * 80)
    print("✅ Test Completed Successfully")
    print("=" * 80)


def test_with_rag_pipeline():
    """Test CodeCorrectionAgent with SysMLRAGPipeline"""
    print("\n" + "=" * 80)
    print("TESTING CodeCorrectionAgent with SysMLRAGPipeline")
    print("=" * 80)

    try:
        # Import RAG pipeline
        from rag.Injestion import SysMLRAGPipeline

        # Initialize RAG pipeline (should have existing data)
        print("\n1. Initializing SysMLRAGPipeline...")
        rag_pipeline = SysMLRAGPipeline(
            collection_name="sysml_v2_knowledge", persist_directory="./chroma_db"
        )

        stats = rag_pipeline.get_collection_stats()
        print(f"   Total chunks in RAG: {stats.get('total_chunks', 0)}")

        if stats.get("total_chunks", 0) == 0:
            print("   ⚠️  RAG database is empty. Run Injestion.py first.")
            print("   Skipping RAG-based test...")
            return

        # Create agent with RAG
        print("\n2. Creating CodeCorrectionAgent with RAG...")
        agent = CodeCorrectionAgent(
            llm=ChatOllama(
                model="qwen2.5-coder:7b",
                temperature=0.25,
            ),
            error_solutions_db=rag_pipeline,
        )

        # Test with a SysML error
        test_state = WorkflowState(
            error="Syntax error: unexpected token 'part', expecting ';' or '}'",
            code="""package VehicleSystem {
                    part def Engine
                    part def Transmission;
                }""",
            processed_query="Create a vehicle system model",
        )

        print("\n3. Running Error Analysis with RAG Context...")
        print(f"   Error: {test_state.error}")

        result_state = agent.analyze_errors(test_state)

        print("\n4. Results:")
        if "CORRECTION GUIDANCE:" in result_state.processed_query:
            guidance_start = result_state.processed_query.index("CORRECTION GUIDANCE:")
            guidance = result_state.processed_query[guidance_start:]
            print("   Correction guidance generated:")
            for line in guidance.split("\n")[:15]:
                print(f"   {line}")
            print("\n✅ Successfully integrated RAG context into correction guidance")

    except ImportError:
        print("\n⚠️  Could not import SysMLRAGPipeline (Injestion.py)")
        print("   Make sure Injestion.py is in the correct location")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Configure logging for test output
    logging.basicConfig(level=logging.INFO)

    # Run basic test
    test_analyze_errors()

    # Run RAG-integrated test
    test_with_rag_pipeline()

    print("\n🎉 All tests completed!")
