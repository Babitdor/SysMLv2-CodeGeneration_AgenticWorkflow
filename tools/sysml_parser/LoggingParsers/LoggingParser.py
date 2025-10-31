from antlr4 import *  # type: ignore
from antlr4.error.ErrorListener import ErrorListener
from .ErrorLogger import ErrorLogger


class LoggingParser:
    """Base class for custom parsers that log errors."""

    def __init__(self, input_string: str):
        """
        Initializes the parser with the input string and sets up the custom error listener.
        """
        self.input_string = input_string
        self.error_logger = ErrorLogger()

    def parse(self):
        """
        Main parsing method that sets up the specific lexer and parser for each subclass.
        """
        lexer_class, parser_class = self.get_lexer_parser_classes()

        input_stream = InputStream(self.input_string)
        lexer = lexer_class(input_stream)
        token_stream = CommonTokenStream(lexer)
        parser = parser_class(token_stream)

        parser.removeErrorListeners()
        parser.addErrorListener(self.error_logger)

        tree = parser.entryRuleRootNamespace()
        return tree.toStringTree(recog=parser)

    def get_errors(self):
        """
        Retrieves the errors encountered during parsing.
        """
        errors_string = self.error_logger.get_errors_string()
        return errors_string if errors_string else "Parsing successful!"

    def get_lexer_parser_classes(self):
        """
        Abstract method to obtain specific lexer and parser classes.

        """
        raise NotImplementedError(
            "Must be implemented in the subclass to specify lexer and parser."
        )
