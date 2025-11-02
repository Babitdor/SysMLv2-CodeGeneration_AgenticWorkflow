from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import Field


class SysMLExampleSearchTool(BaseTool):
    """Search for similar SysML code examples"""

    name: str = "sysml_example_search_tool"
    description: str = """
    Search / Retrieve relevant SysML v2 and KerML code examples, patterns, and known error references from the knowledge base.

    Use this tool when:
    - You need concrete SysML code patterns or syntax examples for a concept you are modeling
    - You want to compare multiple possible design representations
    - You need to enrich a user query before attempting to generate SysML code
    - You want to avoid common syntax / typing errors (Types_Errors.md reference)

    This tool retrieves:
    - Real SysML v2 code examples (.sysml)
    - KerML definitions and structures (.kerml)
    - Known error patterns and type mismatch examples (from Types_Errors.md)
    - Relevant documentation context

    Input: natural language description of what you are trying to model
    Output: enriched structured context (patterns + examples + documentation snippets)

    Examples of good queries:
    - "state machine with hierarchical substates and typed transitions"
    - "action definition that receives and emits flow properties"
    - "slot usage pattern with item flows"
    - "requirement verification + satisfaction chain"
    - "how do you define connection usage between two ports with units?"

    Use this tool BEFORE generating SysML when you need domain grounding or examples.
    """

    # Use Field to properly declare the attribute in Pydantic model
    rag_knowledge_base: Optional[object] = Field(
        default=None, description="RAG knowledge base instance for querying"
    )

    def __init__(self, rag_knowledge_base=None, **kwargs):
        """Initialize with RAG knowledge base"""
        super().__init__(rag_knowledge_base=rag_knowledge_base, **kwargs)

    def _run(self, query: str) -> str:
        """Synchronous run method"""
        if not self.rag_knowledge_base:
            return "RAG knowledge base not available."

        try:
            code_results = self.rag_knowledge_base.query(  # type: ignore
                query, n_results=3, filter_doc_type="code"
            )
            doc_results = self.rag_knowledge_base.query(  # type: ignore
                query, n_results=2, filter_doc_type="documentation"
            )

            if not code_results and not doc_results:
                return "No SysML enrichment found."

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

            return rag_context

        except Exception as e:
            return f"RAG enhancement error: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async run method"""
        return self._run(query)
