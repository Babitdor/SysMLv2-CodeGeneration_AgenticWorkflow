from langchain.tools import BaseTool

class SysMLExampleSearchTool(BaseTool):
    """Search for similar SysML code examples"""

    name: str = "sysml_example_search"
    description: str = """
    Search for similar SysML v2 code examples from the knowledge base.
    Input: Description of what you want to create (e.g., "action with flows")
    Output: Relevant example code snippets with explanations
    Use this when you need reference examples for code generation.
    """

    def __init__(self, knowledge_base):
        super().__init__()
        self.knowledge_base = knowledge_base

    def _run(self, query: str) -> str:
        """Search for examples"""
        try:
            results = self.knowledge_base.similarity_search(query, k=3)
            examples = []
            for i, doc in enumerate(results, 1):
                examples.append(f"Example {i}:\n{doc.page_content}\n")
            return "\n".join(examples)
        except Exception as e:
            return f"Error searching examples: {str(e)}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
