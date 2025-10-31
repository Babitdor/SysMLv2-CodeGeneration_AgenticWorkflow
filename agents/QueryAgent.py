import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from states.WorkflowState import WorkflowState
import logging
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
import yaml
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryAgent:
    """Agent responsible for processing and refining user queries"""

    def __init__(self, llm, config_path: str = "agents/prompt/prompt.yaml") -> None:
        self.name = "QueryAgent"
        self.llm = llm
        self.system_prompt = self._load_system_prompt(config_path)

    def _load_system_prompt(self, config_path: str) -> str:
        """Load system prompt from a YAML config file. Raise if not found."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"System prompt loaded from {config_path}")
            prompt = config.get("Query-Agent", "")
            if not prompt:
                raise ValueError(
                    f"'Query-Agent' key not found or empty in {config_path}."
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

    def process(self, state: WorkflowState) -> WorkflowState:
        """Process the original query into a structured prompt"""
        try:
            # logger.info("Query Agent: Processing user query")

            # self.prompt = ChatPromptTemplate.from_messages(
            #     [("system", self.system_prompt), ("human", "Input: {query}")]
            # )
            # chain = self.prompt | self.llm | StrOutputParser()
            # response = chain.invoke({"query": state.original_query})
            # state.processed_query = response
            # logger.info("Query Agent: Successfully processed query")
            agent = create_agent(
                model=self.llm,
                tools=[],
                system_prompt=self.system_prompt,
                name="Query-Agent",
            )

            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Input: {state.original_query}",
                        }
                    ]
                }
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

            state.processed_query = response
            logger.info("Query Agent: Successfully processed query")

        except Exception as e:
            logger.error(f"Query Agent error: {str(e)}")
            state.processed_query = state.original_query

        return state


def main():
    """Test main function for QueryAgent"""

    # Test scenarios
    test_queries = [
        "Create a SysML model for a simple car system",
        "Design a battery management system with temperature monitoring",
        "Model a robotic arm with 3 joints and servo motors",
        "Create a requirements model for a smart home system",
    ]

    print("=== QueryAgent Test Suite ===\n")

    # Initialize QueryAgent with mock LLM
    query_agent = QueryAgent(
        llm=ChatOllama(
            model="gpt-oss:20b-cloud",
            validate_model_on_init=True,
        )
    )
    # Test each query
    for i, query in enumerate(test_queries, 1):
        print(f"--- Test Case {i} ---")
        print(f"Original Query: {query}")

        # Create state and process
        state = WorkflowState(original_query=query)
        response_state = query_agent.process(state)

        print(
            f"Processed Query Length: {len(response_state.processed_query)} characters"
        )
        print(f"Processed Query Preview:")
        print(f"{response_state.processed_query}")
        print(f"Success: {'Yes' if response_state.processed_query else 'No'}")
        print()

    print("\n=== Test Suite Complete ===")


if __name__ == "__main__":
    main()
