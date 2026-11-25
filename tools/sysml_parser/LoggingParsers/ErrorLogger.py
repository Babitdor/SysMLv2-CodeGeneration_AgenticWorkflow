from antlr4.error.ErrorListener import ErrorListener
import re


class ErrorLogger(ErrorListener):
    """Captures every syntax error from ANTLR and formats them for human consumption."""

    def __init__(self, input_string: str | None = None):
        self.__errors: list[str] = []
        self.__input_lines: list[str] = input_string.splitlines() if input_string else []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        """Process and store syntax errors with simplified messages."""
        try:
            simplified_msg = self._simplify_error_message(msg, offendingSymbol)
            error_message = f"Syntax error at line {line}, column {column}: {simplified_msg}"
            snippet = self._get_line_snippet(line, column)
            if snippet:
                error_message += f"\n{snippet}"
        except Exception:
            error_message = f"Syntax error at line {line}, column {column}: {msg}"
        self.__errors.append(error_message)

    def _get_line_snippet(self, line: int, column: int) -> str:
        """Return the source line and a caret pointing at the error column."""
        if not self.__input_lines or line < 1 or line > len(self.__input_lines):
            return ""
        source_line = self.__input_lines[line - 1]
        caret_col = max(0, min(column, len(source_line)))
        return f"  {source_line}\n  {' ' * caret_col}^"

    def _simplify_error_message(self, msg, offendingSymbol):
        """Simplify ANTLR error messages for better readability."""
        if not msg:
            return "Unknown syntax error"

        if "mismatched input" in msg.lower():
            return self._handle_mismatched_input(msg)
        elif "extraneous input" in msg.lower():
            return self._handle_extraneous_input(msg)
        elif "missing" in msg.lower():
            return self._handle_missing_token(msg)
        elif "no viable alternative" in msg.lower():
            return self._handle_no_viable_alternative(msg, offendingSymbol)
        elif "token recognition error" in msg.lower():
            return self._handle_token_recognition_error(msg)

        return msg

    def _handle_mismatched_input(self, msg):
        """Handle 'mismatched input' error messages."""
        try:
            token_match = re.search(r"'([^']*)'", msg)
            token = token_match.group(1) if token_match else "unknown"

            if token in ("<EOF>", "EOF"):
                return "Unexpected end of file — check for unclosed braces, parentheses, or missing semicolons"

            if "expecting" in msg:
                expected = msg.split("expecting", 1)[1].strip()
                expected = self._simplify_expected_tokens(expected)
                return f"Unexpected '{token}', expecting {expected}"

            return f"Unexpected token '{token}'"
        except Exception:
            return msg

    def _handle_extraneous_input(self, msg):
        """Handle 'extraneous input' error messages."""
        try:
            token_match = re.search(r"'([^']*)'", msg)
            token = token_match.group(1) if token_match else "unknown"

            if token in ("<EOF>", "EOF"):
                return "Unexpected end of file — check for unclosed braces or missing semicolons"

            if "expecting" in msg:
                expected = msg.split("expecting", 1)[1].strip()
                expected = self._simplify_expected_tokens(expected)
                return f"Extra token '{token}' found, expecting {expected}"

            return f"Unexpected extra token '{token}'"
        except Exception:
            return msg

    def _handle_missing_token(self, msg):
        """Handle 'missing' token error messages."""
        try:
            token_match = re.search(r"'([^']*)'", msg)
            token = token_match.group(1) if token_match else "unknown"

            if token in ("<EOF>", "EOF"):
                return "Unexpected end of file — check for unclosed braces or missing semicolons"

            return f"Missing expected token '{token}'"
        except Exception:
            return msg

    def _handle_no_viable_alternative(self, msg, offendingSymbol):
        """Handle 'no viable alternative' error messages."""
        try:
            token = (
                offendingSymbol.text
                if offendingSymbol and hasattr(offendingSymbol, "text")
                else "unknown"
            )

            if token == "<EOF>":
                return "Unexpected end of file — incomplete statement or expression"

            return f"Invalid syntax near '{token}' — no valid parsing rule matches"
        except Exception:
            return "Invalid syntax — no valid parsing rule matches"

    def _handle_token_recognition_error(self, msg):
        """Handle token recognition error messages."""
        try:
            error_match = re.search(r"at: '([^']*)'", msg)
            if error_match:
                char = error_match.group(1)
                return f"Unrecognized character or sequence: '{char}'"
            return "Unrecognized character or token"
        except Exception:
            return msg

    def _simplify_expected_tokens(self, expected: str) -> str:
        """Simplify ANTLR expected-token lists into readable form."""
        try:
            expected = expected.replace("{", "").replace("}", "")

            if not expected.strip():
                return "end of statement"

            if "," in expected:
                raw_options = [opt.strip().strip("'") for opt in expected.split(",") if opt.strip()]
            else:
                raw_options = [expected.strip().strip("'")]

            readable: list[str] = []
            for tok in raw_options:
                if tok == "<EOF>":
                    readable.append("end of input")
                elif re.match(r"^T__\d+$", tok) or tok.startswith("RULE_"):
                    # Internal ANTLR anonymous/grammar-rule tokens — not user-meaningful
                    continue
                else:
                    readable.append(tok)

            if not readable:
                return "a valid continuation"

            if len(readable) > 3:
                return f"one of: {', '.join(readable[:3])}... (and {len(readable) - 3} more)"
            return f"one of: {', '.join(readable)}"
        except Exception:
            return expected

    def get_errors_string(self) -> str:
        """Return all errors as a formatted string."""
        if not self.__errors:
            return "Parsing successful!"
        header = f"Found {len(self.__errors)} syntax error(s):\n"
        return header + "\n".join(self.__errors)

    def get_errors_list(self) -> list[str]:
        """Return errors as a list."""
        return self.__errors.copy()

    def has_errors(self) -> bool:
        """Return True if any errors were recorded."""
        return bool(self.__errors)

    def get_error_count(self) -> int:
        """Return the number of recorded errors."""
        return len(self.__errors)

    def clear_errors(self):
        """Reset error state."""
        self.__errors.clear()
