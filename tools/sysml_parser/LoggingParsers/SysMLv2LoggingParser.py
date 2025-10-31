from .LoggingParser import LoggingParser
from .SysMLv2.SysMLv2Lexer import SysMLv2Lexer  
from .SysMLv2.SysMLv2Parser import SysMLv2Parser

class SysMLv2LoggingParser(LoggingParser):
    """Parser for validating SysML v2 specifically"""
    
    def __init__(self, input_string: str):
        """Initialize parser with input validation"""
        if not input_string or input_string.strip() == "":
            raise ValueError("Empty input: SysML code cannot be empty")
        super().__init__(input_string)
    
    def get_lexer_parser_classes(self):
        return SysMLv2Lexer, SysMLv2Parser



