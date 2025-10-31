from .sysml_parser.LoggingParsers.SysMLv2LoggingParser import SysMLv2LoggingParser
from langchain.tools import BaseTool


class SysMLValidatorTool(BaseTool):
    """LangChain tool for validating SysML v2 code"""

    name: str = "SysML-Validator-tool"
    description: str = """
        Validates SysML v2 code syntax and returns detailed error information.
        Input should be a string containing valid SysML v2 code.
        Returns validation status, errors (if any), and helpful feedback.
        Use this tool whenever you need to check if SysML code is syntactically correct.
    """

    def _run(self, code: str) -> str:
        """Execute the validation"""
        try:
            parser = SysMLv2LoggingParser(input_string=code)
            parser.parse()
            errors = parser.get_errors()

            if errors == "Parsing successful!":
                return "✅ Code is valid! No syntax errors found."
            else:
                return f"❌ Validation failed:\n\n{errors}"

        except Exception as e:
            return f"Error during validation: {str(e)}"

    async def _arun(self, code: str) -> str:
        """Async version (delegates to sync for simplicity)"""
        return self._run(code)


sysml_validator_tool = SysMLValidatorTool()
