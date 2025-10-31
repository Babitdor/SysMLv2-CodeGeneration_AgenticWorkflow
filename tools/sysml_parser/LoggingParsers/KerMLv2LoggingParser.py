from LoggingParsers.LoggingParser import LoggingParser
from LoggingParsers.KerMLv2.KerMLv2Lexer import KerMLv2Lexer  
from LoggingParsers.KerMLv2.KerMLv2Parser import KerMLv2Parser 

class KerMLv2LoggingParser(LoggingParser):
    """Parser for validating KerMLv2 specifically"""
    
    def get_lexer_parser_classes(self):
        return KerMLv2Lexer, KerMLv2Parser
