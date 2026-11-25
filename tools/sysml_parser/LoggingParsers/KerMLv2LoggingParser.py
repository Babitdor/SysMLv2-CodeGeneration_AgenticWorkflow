from .LoggingParser import LoggingParser
from .KerMLv2.KerMLv2Lexer import KerMLv2Lexer
from .KerMLv2.KerMLv2Parser import KerMLv2Parser


class KerMLv2LoggingParser(LoggingParser):
    """Parser for validating KerML v2 specifically."""

    def __init__(self, input_string: str):
        if not input_string or not input_string.strip():
            raise ValueError("Empty input: KerML code cannot be empty")
        super().__init__(input_string)

    def get_lexer_parser_classes(self):
        return KerMLv2Lexer, KerMLv2Parser
