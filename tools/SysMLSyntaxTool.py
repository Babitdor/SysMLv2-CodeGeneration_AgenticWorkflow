from langchain.tools import BaseTool
import yaml
from pathlib import Path
from typing import Dict


class SysMLSyntaxTool(BaseTool):
    """Quick syntax reference lookup"""

    name: str = "sysml_syntax_reference_tool"
    description: str = """
    Get syntax reference for specific SysML v2 constructs.
    Input: Type of construct (e.g., "part definition", "part def", "connection", "state")
    Output: Syntax rules and examples for that construct
    Use this when you're unsure about correct syntax.
    """

    _syntax_db: Dict = None  # type: ignore
    _construct_map: Dict = None  # type: ignore

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_syntax_data()

    def _load_syntax_data(self):
        """Load syntax database from YAML file"""
        yaml_path = Path(__file__).parent / "db" / "syntax_db.yaml"

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self._syntax_db = data.get("syntax", {})
                self._construct_map = data.get("construct_map", {})
        except FileNotFoundError:
            raise FileNotFoundError(
                f"syntax_db.yaml not found at {yaml_path}. "
                "Please ensure the file exists in the same directory as this tool."
            )
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing syntax_db.yaml: {e}")

    def _run(self, construct_type: str) -> str:
        """Get syntax reference"""

        # Ensure data is loaded
        if self._syntax_db is None or self._construct_map is None:
            self._load_syntax_data()

        # Normalize input to lowercase
        normalized_input = construct_type.lower().strip()

        # Try to map short form to full name
        normalized = self._construct_map.get(normalized_input, normalized_input)

        # Get syntax from database
        result = self._syntax_db.get(
            normalized,
            f"Syntax reference for '{construct_type}' not found. "
            f"Available constructs: {', '.join(sorted(set(list(self._syntax_db.keys()) + list(self._construct_map.keys()))))}",
        )

        return result

    async def _arun(self, construct_type: str) -> str:
        """Async version"""
        return self._run(construct_type)


# Create tool instance
syntax_tool = SysMLSyntaxTool()
