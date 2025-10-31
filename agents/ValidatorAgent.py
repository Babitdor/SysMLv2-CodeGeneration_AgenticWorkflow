import os
import sys
import time
import logging
import re
from typing import List
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from states.WorkflowState import WorkflowState
from states.ValidationState import ValidationResult, ErrorInfo, ValidationStatus
from tools.SysMLValidatorTool import SysMLValidatorTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidatorAgent:
    """Agent responsible for validating SysML code using SysMLValidatorTool"""

    def __init__(self):
        """Initialize the validator agent with the SysML validation tool"""
        self.validator_tool = SysMLValidatorTool()
        logger.info("ValidatorAgent initialized with SysMLValidatorTool")

    @traceable(name="ValidatorAgent - validate")
    def validate(self, state: WorkflowState) -> WorkflowState:
        """Validate the generated SysML code using the SysMLValidatorTool"""
        logger.info("Starting validation for current SysML code")
        start_time = time.time()

        if not state.code or not state.code.strip():
            error_result = ValidationResult(
                success=False,
                errors=[
                    ErrorInfo(
                        name="EmptyCodeError", message="No code provided for validation"
                    )  # type: ignore
                ],
                execution_time=time.time() - start_time,
            )
            state.add_validation_result(error_result)
            state.is_valid = ValidationStatus.INVALID
            logger.error("Validation aborted: No code provided")
            return state

        try:
            # Use the SysMLValidatorTool to validate the code
            validation_output = self.validator_tool._run(state.code)

            execution_time = time.time() - start_time

            # Check if validation was successful
            if validation_output.startswith("✅ Code is valid!"):
                # Validation successful
                validation_result = ValidationResult(
                    success=True,
                    errors=[],
                    warnings=[],
                    output=validation_output,
                    execution_time=execution_time,
                )
                state.is_valid = ValidationStatus.VALID
                logger.info("✅ Validation successful")

            else:
                # Validation failed - parse errors from output
                errors = self._parse_validation_errors(validation_output)
                warnings = self._parse_validation_warnings(validation_output)

                validation_result = ValidationResult(
                    success=False,
                    errors=errors,
                    warnings=warnings,
                    output=validation_output,
                    execution_time=execution_time,
                )
                state.is_valid = ValidationStatus.INVALID
                state.error = self._format_errors_for_state(errors)
                logger.warning(f"❌ Validation failed with {len(errors)} error(s)")

            # Add validation result to state history
            state.add_validation_result(validation_result)
            return state

        except Exception as e:
            # Handle unexpected errors during validation
            execution_time = time.time() - start_time
            error_result = ValidationResult(
                success=False,
                errors=[
                    ErrorInfo(
                        name="ValidationException",
                        message=f"Unexpected error during validation: {str(e)}",
                    )  # type: ignore
                ],
                execution_time=execution_time,
            )
            state.add_validation_result(error_result)
            state.is_valid = ValidationStatus.ERROR
            state.error = str(e)
            logger.exception("Validation failed with exception")
            return state

    @traceable(name="ValidatorAgent - parse_errors")
    def _parse_validation_errors(self, validation_output: str) -> List[ErrorInfo]:
        """Parse error information from validation output"""
        errors = []

        # Skip if validation was successful
        if "✅" in validation_output or "No syntax errors found" in validation_output:
            return errors

        # Split output into lines for parsing
        lines = validation_output.split("\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("❌"):
                continue

            # Look for common error patterns
            # Pattern 1: "Error: <message>"
            if line.lower().startswith("error"):
                error_msg = line.split(":", 1)[-1].strip()
                errors.append(ErrorInfo(name="SyntaxError", message=error_msg))  # type: ignore

            # Pattern 2: Line contains error keywords
            elif any(
                keyword in line.lower()
                for keyword in ["syntax error", "parse error", "invalid", "unexpected"]
            ):
                # Try to extract line number if present
                line_number = self._extract_line_number(line)
                errors.append(
                    ErrorInfo(name="SyntaxError", message=line, line_number=line_number)  # type: ignore
                )

            # Pattern 3: Check for specific parser error formats
            elif "line" in line.lower() and any(
                kw in line.lower() for kw in ["error", "fail", "invalid"]
            ):
                line_number = self._extract_line_number(line)
                errors.append(
                    ErrorInfo(name="ParserError", message=line, line_number=line_number)  # type: ignore
                )

        # If no errors were parsed but validation failed, create a generic error
        if not errors and "âŒ" in validation_output:
            # Extract everything after the ❌ symbol
            error_text = validation_output.split("âŒ")[-1].strip()
            if error_text:
                errors.append(ErrorInfo(name="ValidationError", message=error_text))  # type: ignore
            else:
                errors.append(
                    ErrorInfo(
                        name="UnknownError",
                        message="Validation failed with unknown error",
                    )  # type: ignore
                )

        return errors

    def _parse_validation_warnings(self, validation_output: str) -> List[str]:
        """Parse warning information from validation output"""
        warnings = []

        lines = validation_output.split("\n")
        for line in lines:
            line = line.strip()
            if line and "warning" in line.lower():
                warnings.append(line)

        return warnings

    def _extract_line_number(self, text: str) -> int:
        """Extract line number from error text"""
        # Look for patterns like "line 5", "Line 10", etc.
        match = re.search(r"line\s+(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0

    def _format_errors_for_state(self, errors: List[ErrorInfo]) -> str:
        """Format errors into a string for the state.error field"""
        if not errors:
            return ""

        error_messages = []
        for error in errors:
            msg = error.message
            if error.line_number and error.line_number > 0:
                msg = f"Line {error.line_number}: {msg}"
            error_messages.append(msg)

        return "\n".join(error_messages)

    @traceable(name="ValidatorAgent - format_feedback")
    def format_validation_feedback(self, state: WorkflowState) -> str:
        """Format validation results for human-readable feedback"""
        latest = state.get_latest_validation()
        if not latest:
            return "No validation results available."

        if latest.success:
            return "✅ SysML code validated successfully!"

        feedback = "❌ SysML Validation Errors Found:\n\n"

        for i, error in enumerate(latest.errors, 1):
            feedback += f"Error {i}: {error.message}"
            if error.line_number and error.line_number > 0:
                feedback += f" (Line {error.line_number})"
            feedback += "\n"

        if latest.warnings:
            feedback += "\nâš ï¸ Warnings:\n"
            for warning in latest.warnings:
                feedback += f"- {warning}\n"

        return feedback


def test_validator_agent():
    """Test function for ValidatorAgent"""
    print("=" * 80)
    print("VALIDATOR AGENT TEST SUITE")
    print("=" * 80)

    # Test samples - good and bad SysML code
    test_cases = [
        {
            "name": "Valid SysML Package",
            "code": """package TestSystem {
    part def Engine;
    part def Transmission;
    
    part vehicle {
        part engine : Engine;
        part transmission : Transmission;
    }
}""",
            "description": "Valid SysML package with part definitions",
            "expected_success": True,
        },
        {
            "name": "Invalid SysML - Missing Semicolon",
            "code": """package TestSystem {
    part def Engine
    part def Transmission;
}""",
            "description": "Invalid SysML with missing semicolon",
            "expected_success": False,
        },
        {
            "name": "Valid SysML Action",
            "code": """package PictureTaking {
    part def Exposure;
    
    action def Focus { 
        out xrsl: Exposure; 
    }
    
    action def Shoot { 
        in xsf: Exposure; 
    }
    
    action takePicture {
        action focus: Focus[1];
        flow of Exposure from focus.xrsl to shoot.xsf;
        action shoot: Shoot[1];
    }
}""",
            "description": "Valid SysML action with flows",
            "expected_success": True,
        },
        {
            "name": "Invalid SysML - Unclosed Brace",
            "code": """package TestSystem {
    part def Engine;
    part def Transmission;
""",
            "description": "Invalid SysML with unclosed brace",
            "expected_success": False,
        },
        {
            "name": "Empty Code",
            "code": "",
            "description": "Empty code string",
            "expected_success": False,
        },
    ]

    # Initialize validator
    validator = ValidatorAgent()

    # Run test cases
    print("\nRunning Validation Tests")
    print("-" * 80)

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"Expected: {'SUCCESS' if test_case['expected_success'] else 'FAILURE'}")

        # Create workflow state with the test code
        state = WorkflowState(
            original_query=f"Test case: {test_case['name']}", code=test_case["code"]
        )

        # Run validation
        try:
            start_time = time.time()
            validated_state = validator.validate(state)
            test_time = time.time() - start_time

            latest_result = validated_state.get_latest_validation()

            if latest_result:
                success = latest_result.success
                actual_result = "SUCCESS" if success else "FAILURE"
                expected_result = (
                    "SUCCESS" if test_case["expected_success"] else "FAILURE"
                )
                test_passed = success == test_case["expected_success"]

                print(f"Actual: {actual_result}")
                print(f"Test Result: {'✅ PASS' if test_passed else '❌ FAIL'}")
                print(f"Execution Time: {latest_result.execution_time:.3f}s")

                if not success and latest_result.errors:
                    print("Errors Found:")
                    for error in latest_result.errors[:3]:  # Show first 3 errors
                        line_info = (
                            f" (Line {error.line_number})" if error.line_number else ""
                        )
                        print(f"  - {error.name}: {error.message}{line_info}")

                if latest_result.warnings:
                    print(f"Warnings: {len(latest_result.warnings)}")

                results.append(
                    {
                        "test": test_case["name"],
                        "expected": expected_result,
                        "actual": actual_result,
                        "passed": test_passed,
                        "time": latest_result.execution_time,
                    }
                )
            else:
                print("❌ No validation result returned")
                results.append(
                    {
                        "test": test_case["name"],
                        "expected": (
                            "SUCCESS" if test_case["expected_success"] else "FAILURE"
                        ),
                        "actual": "NO_RESULT",
                        "passed": False,
                        "time": test_time,
                    }
                )

        except Exception as e:
            print(f"❌ Exception during test: {str(e)}")
            results.append(
                {
                    "test": test_case["name"],
                    "expected": (
                        "SUCCESS" if test_case["expected_success"] else "FAILURE"
                    ),
                    "actual": "EXCEPTION",
                    "passed": False,
                    "time": 0.0,
                }
            )

        print("-" * 80)

    # Test summary
    print("\nTest Summary")
    print("=" * 80)

    passed_tests = sum(1 for r in results if r["passed"])
    total_tests = len(results)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Total Execution Time: {sum(r['time'] for r in results):.3f}s")

    print("\nDetailed Results:")
    for result in results:
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(
            f"{status} {result['test']}: {result['expected']} -> {result['actual']} ({result['time']:.3f}s)"
        )

    # Test formatting
    print("\nTesting Validation Feedback Formatting")
    print("-" * 80)

    # Use a test case with errors for formatting test
    error_test = test_cases[1]  # Missing semicolon case
    test_state = WorkflowState(
        original_query="Formatting test", code=error_test["code"]
    )
    validated_state = validator.validate(test_state)

    feedback = validator.format_validation_feedback(validated_state)
    print(feedback)

    print("\n" + "=" * 80)
    print("✅ Test suite completed!")


if __name__ == "__main__":
    """Main test runner"""
    try:
        test_validator_agent()
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed with exception: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\nExiting...")
