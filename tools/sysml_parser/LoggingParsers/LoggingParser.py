from antlr4 import *  # type: ignore
from .ErrorLogger import ErrorLogger


class LoggingParser:
    """Base class for custom parsers that log errors."""

    def __init__(self, input_string: str):
        self.input_string = input_string
        self.error_logger = ErrorLogger(input_string=input_string)

    def parse(self):
        """Set up the ANTLR lexer/parser pipeline and return the parse tree string."""
        lexer_class, parser_class = self.get_lexer_parser_classes()

        input_stream = InputStream(self.input_string)

        lexer = lexer_class(input_stream)
        lexer.removeErrorListeners()
        lexer.addErrorListener(self.error_logger)

        token_stream = CommonTokenStream(lexer)
        parser = parser_class(token_stream)
        parser.removeErrorListeners()
        parser.addErrorListener(self.error_logger)

        tree = parser.rootNamespace()
        return tree.toStringTree(recog=parser)

    def get_errors(self) -> str:
        """Return the formatted error string from the last parse."""
        return self.error_logger.get_errors_string()

    def has_errors(self) -> bool:
        """Return True if any errors were recorded during the last parse."""
        return self.error_logger.has_errors()

    def get_error_count(self) -> int:
        """Return the number of errors recorded during the last parse."""
        return self.error_logger.get_error_count()

    def get_errors_list(self) -> list[str]:
        """Return errors as a list from the last parse."""
        return self.error_logger.get_errors_list()

    def get_lexer_parser_classes(self):
        """Abstract method: return (LexerClass, ParserClass) for this language."""
        raise NotImplementedError(
            "Must be implemented in the subclass to specify lexer and parser."
        )
