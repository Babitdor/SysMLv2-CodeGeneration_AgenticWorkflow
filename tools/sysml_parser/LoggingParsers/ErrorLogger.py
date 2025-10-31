from antlr4.error.ErrorListener import ErrorListener
import re


class ErrorLogger(ErrorListener):
    """This class keeps track of every error found in the SysML code"""

    def __init__(self):
        self.__errors = []  # List to collect errors
        self.__error_count = 0  # Track total number of errors

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        """Process and store syntax errors with simplified messages"""
        try:
            simplified_msg = self._simplify_error_message(msg, offendingSymbol)

            # Store each error as a formatted string
            error_message = (
                f"Syntax error at line {line}, column {column}: {simplified_msg}"
            )
            self.__errors.append(error_message)
            self.__error_count += 1
        except Exception as ex:
            # Fallback in case simplification fails
            error_message = f"Syntax error at line {line}, column {column}: {msg}"
            self.__errors.append(error_message)
            self.__error_count += 1

    def _simplify_error_message(self, msg, offendingSymbol):
        """Simplify error messages for better readability"""
        if not msg:
            return "Unknown syntax error"

        # Handle mismatched input errors
        if "mismatched input" in msg.lower():
            return self._handle_mismatched_input(msg)

        # Handle extraneous input errors
        elif "extraneous input" in msg.lower():
            return self._handle_extraneous_input(msg)

        # Handle missing token errors
        elif "missing" in msg.lower():
            return self._handle_missing_token(msg)

        # Handle no viable alternative errors
        elif "no viable alternative" in msg.lower():
            return self._handle_no_viable_alternative(msg, offendingSymbol)

        # Handle token recognition errors
        elif "token recognition error" in msg.lower():
            return self._handle_token_recognition_error(msg)

        # Return original message if no pattern matches
        return msg

    def _handle_mismatched_input(self, msg):
        """Handle 'mismatched input' error messages"""
        try:
            # Extract token using regex to handle edge cases
            token_match = re.search(r"'([^']*)'", msg)
            token = token_match.group(1) if token_match else "unknown"

            # Handle EOF specially
            if token == "<EOF>" or "EOF" in token:
                return "Unexpected end of file - check for unclosed braces, parentheses, or missing semicolons"

            # Extract expected tokens
            if "expecting" in msg:
                expected = msg.split("expecting", 1)[1].strip()
                expected = self._simplify_expected_tokens(expected)
                return f"Unexpected '{token}', expecting {expected}"

            return f"Unexpected token '{token}'"
        except:
            return msg

    def _handle_extraneous_input(self, msg):
        """Handle 'extraneous input' error messages"""
        try:
            token_match = re.search(r"'([^']*)'", msg)
            token = token_match.group(1) if token_match else "unknown"

            if token == "<EOF>" or "EOF" in token:
                return "Unexpected end of file - check for unclosed braces or missing semicolons"

            if "expecting" in msg:
                expected = msg.split("expecting", 1)[1].strip()
                expected = self._simplify_expected_tokens(expected)
                return f"Extra token '{token}' found, expecting {expected}"

            return f"Unexpected extra token '{token}'"
        except:
            return msg

    def _handle_missing_token(self, msg):
        """Handle 'missing' token error messages"""
        try:
            token_match = re.search(r"'([^']*)'", msg)
            token = token_match.group(1) if token_match else "unknown"
            return f"Missing expected token '{token}'"
        except:
            return msg

    def _handle_no_viable_alternative(self, msg, offendingSymbol):
        """Handle 'no viable alternative' error messages"""
        try:
            token = (
                offendingSymbol.text
                if offendingSymbol and hasattr(offendingSymbol, "text")
                else "unknown"
            )

            if token == "<EOF>":
                return "Unexpected end of file - incomplete statement or expression"

            return f"Invalid syntax near '{token}' - no valid parsing rule matches"
        except:
            return "Invalid syntax - no valid parsing rule matches"

    def _handle_token_recognition_error(self, msg):
        """Handle token recognition error messages"""
        try:
            # Extract the problematic character/sequence
            error_match = re.search(r"at: '([^']*)'", msg)
            if error_match:
                char = error_match.group(1)
                return f"Unrecognized character or sequence: '{char}'"
            return "Unrecognized character or token"
        except:
            return msg

    def _simplify_expected_tokens(self, expected):
        """Simplify the list of expected tokens"""
        try:
            # Remove braces
            expected = expected.replace("{", "").replace("}", "")

            # Handle empty expectation
            if not expected.strip():
                return "end of statement"

            # Split by comma and limit options
            if "," in expected:
                options = [opt.strip() for opt in expected.split(",")]
                # Filter out very technical tokens
                options = [opt.strip("'") for opt in options if opt.strip()]

                if len(options) > 3:
                    return f"one of: {', '.join(options[:3])}... (and {len(options) - 3} more)"
                return f"one of: {', '.join(options)}"

            return expected.strip("'")
        except:
            return expected

    def get_errors_string(self):
        """Join all errors into a single string, separating them by newlines"""
        if not self.__errors:
            return "Parsing successful!"

        header = f"Found {self.__error_count} syntax error(s):\n"
        return header + "\n".join(self.__errors)

    def get_errors_list(self):
        """Return errors as a list"""
        return self.__errors.copy()

    def has_errors(self):
        """Check if any errors were encountered"""
        return len(self.__errors) > 0

    def get_error_count(self):
        """Get the total number of errors"""
        return self.__error_count

    def clear_errors(self):
        """Clear all stored errors"""
        self.__errors.clear()
        self.__error_count = 0
