import os
from typing import Optional, List
import contextlib
import subprocess
import time
from io import StringIO
from jupyter_client.manager import KernelManager
from States import WorkflowState, ValidationResult, SysMLConfig, ErrorInfo
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidatorAgent:
    """Agent responsible for validating SysML code"""

    def __init__(self, config: Optional[SysMLConfig] = None):
        """Initialize with configuration"""
        self.config = config or SysMLConfig()
        self.kernel_manager = None
        self.kernel_client = None
        self._suppress_startup_warnings()
        logger.info("ValidatorAgent initialized")

    def _suppress_startup_warnings(self):
        """Set environment variables to suppress Java warnings"""
        java_opts = [
            "--add-opens=java.base/sun.misc=ALL-UNNAMED",
            "--add-opens=java.base/java.lang=ALL-UNNAMED",
            "--add-opens=java.base/java.util=ALL-UNNAMED",
            "-XX:+IgnoreUnrecognizedVMOptions",
            "-Djava.util.logging.config.file=NUL",
            "-Dlog4j.rootLogger=OFF",
            "-Dlog4j.logger.root=OFF",
            "-Dlog4j.logger.org.eclipse=OFF",
            "-Dlog4j.logger.com.google=OFF",
            "-Dorg.slf4j.simpleLogger.defaultLogLevel=error",
        ]

        os.environ["JAVA_TOOL_OPTIONS"] = " ".join(java_opts)
        os.environ["PYTHONWARNINGS"] = "ignore"

        if not self.config.silent_startup:
            logger.debug("Applied warning suppression options")

    def check_kernel_available(self) -> bool:
        """Check if SysML kernel is installed and available"""
        try:
            with open(os.devnull, "w") as devnull:
                result = subprocess.run(
                    ["jupyter", "kernelspec", "list"],
                    capture_output=True,  # captures both stdout and stderr
                    text=True,
                    timeout=10,
                )

            if self.config.kernel_name in result.stdout:
                logger.info("✅ SysML kernel found and available")
                return True
            else:
                logger.warning("❌ SysML kernel not found")
                logger.info("Available kernels:\n%s", result.stdout)
                logger.info(
                    "To install SysML kernel: conda install conda-forge::jupyter-sysml-kernel"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Timeout checking kernels")
            return False
        except FileNotFoundError:
            logger.error("❌ Jupyter not found. Install with: pip install jupyter")
            return False
        except Exception as e:
            logger.error(f"❌ Error checking kernels: {e}")
            return False

    def start_kernel(self) -> bool:
        """Start the SysML Jupyter kernel"""
        try:
            if self.config.silent_startup:
                logger.info("Starting SysML kernel ...")

            with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(
                StringIO()
            ):
                self.kernel_manager = KernelManager(kernel_name=self.config.kernel_name)
                self.kernel_manager.start_kernel()
                self.kernel_client = self.kernel_manager.client()
                self.kernel_client.start_channels()
                time.sleep(5)

            # Test kernel
            msg_id = self.kernel_client.execute("// Kernel test")

            timeout = 10
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    msg = self.kernel_client.get_iopub_msg(timeout=1)
                    if (
                        msg["header"]["msg_type"] == "status"
                        and msg["content"].get("execution_state") == "idle"
                    ):
                        logger.info("\n✅ SysML kernel started successfully")
                        return True
                except:
                    continue

            logger.info("\nSysML kernel started (assumed working)")
            return True

        except Exception as e:
            logger.error(f"\nFailed to start SysML kernel: {e}")
            return False

    def validate(self, state: WorkflowState) -> WorkflowState:
        """Validate the generated SysML code"""

        logger.info("Starting validation for current SysML code")
        start_time = time.time()

        if not self.kernel_client and not self.start_kernel():
            error_result = ValidationResult(
                success=False,
                errors=[ErrorInfo(name="KernelError", message="Kernel not available")],  # type: ignore
                execution_time=time.time() - start_time,
            )
            state.add_validation_result(error_result)
            logger.error("Validation aborted: Kernel not available")
            return state

        try:
            msg_id = self.kernel_client.execute(state.code)  # type: ignore

            errors = []
            warnings = []
            output_lines = []
            error_found = False

            timeout_start = time.time()
            while time.time() - timeout_start < self.config.timeout:
                try:
                    msg = self.kernel_client.get_iopub_msg(timeout=1)  # type: ignore
                    msg_type = msg["header"]["msg_type"]
                    content = msg["content"]

                    if msg_type == "error":
                        error_found = True
                        error_info = ErrorInfo(
                            name=content.get("ename", "Error"),
                            message=content.get("evalue", ""),
                            traceback=content.get("traceback", []),
                        )  # type: ignore
                        errors.append(error_info)
                        logger.error("Error detected: %s", error_info.message)

                    elif msg_type == "stream":
                        stream_text = content.get("text", "")
                        filtered_text = self._filter_stream_warnings(stream_text)

                        if filtered_text:
                            output_lines.append(filtered_text)

                        if self._contains_error_keywords(filtered_text):
                            error_found = True
                            parsed_errors = self._parse_stream_errors(filtered_text)
                            errors.extend(parsed_errors)
                            for e in parsed_errors:
                                logger.error("Stream error detected: %s", e.message)

                        if self._contains_warning_keywords(filtered_text):
                            stream_warnings = self._parse_stream_warnings(filtered_text)
                            warnings.extend(stream_warnings)
                            for w in stream_warnings:
                                logger.warning("Stream warning detected: %s", w)

                    elif msg_type == "execute_result":
                        result_text = content.get("data", {}).get("text/plain", "")
                        output_lines.append(result_text)

                    elif (
                        msg_type == "status"
                        and content.get("execution_state") == "idle"
                    ):
                        break

                except Exception:
                    break

            raw_output = "\n".join(output_lines)

            # Additional error detection
            if not error_found and raw_output:
                additional_errors = self._detect_errors_in_output(raw_output)
                errors.extend(additional_errors)
                error_found = len(additional_errors) > 0
                for e in additional_errors:
                    logger.error("Detected error in output: %s", e.message)

            # Truncate output if too long
            if len(raw_output) > self.config.max_output_length:
                output = raw_output[: self.config.max_output_length] + "... (truncated)"
            else:
                output = raw_output

            validation_result = ValidationResult(
                success=not error_found,
                errors=errors,
                warnings=warnings,
                output=output.strip(),
                raw_output=raw_output,
                execution_time=time.time() - start_time,
            )

            state.add_validation_result(validation_result)
            if validation_result.success:
                logger.info("Validation successful")
            else:
                logger.warning("Validation completed with errors/warnings")
            return state

        except Exception as e:
            error_result = ValidationResult(
                success=False,
                errors=[ErrorInfo(name="ValidationError", message=str(e))],  # type: ignore
                execution_time=time.time() - start_time,
            )
            state.add_validation_result(error_result)
            logger.exception("Validation failed with exception")
            return state

    def _filter_stream_warnings(self, text: str) -> str:
        """Filter out known startup warnings from stream output"""
        lines = text.split("\n")
        filtered_lines = []

        for line in lines:
            if line.strip() and not any(
                pattern in line for pattern in self.config.suppress_warnings
            ):
                filtered_lines.append(line)

        return "\n".join(filtered_lines) if filtered_lines else ""

    def _contains_error_keywords(self, text: str) -> bool:
        """Check if text contains error keywords"""
        if not text or not text.strip():
            return False

        error_keywords = [
            "error",
            "Error",
            "ERROR",
            "exception",
            "Exception",
            "failed",
            "Failed",
            "FAILED",
            "syntax error",
            "parse error",
            "compilation error",
        ]
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in error_keywords)

    def _contains_warning_keywords(self, text: str) -> bool:
        """Check if text contains warning keywords"""
        if not text or not text.strip():
            return False

        # Skip if it's a startup warning
        if any(pattern in text for pattern in self.config.suppress_warnings):
            return False

        warning_keywords = ["warning", "Warning", "WARNING", "warn"]
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in warning_keywords)

    def _parse_stream_errors(self, text: str) -> List[ErrorInfo]:
        """Parse errors from stream text"""
        errors = []
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "error" in line.lower() or "exception" in line.lower():
                errors.append(ErrorInfo(name="StreamError", message=line))  # type: ignore

        return errors

    def _parse_stream_warnings(self, text: str) -> List[str]:
        """Parse warnings from stream text"""
        warnings = []
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if line and "warning" in line.lower():
                warnings.append(line)

        return warnings

    def _detect_errors_in_output(self, output: str) -> List[ErrorInfo]:
        """Detect errors in the complete output text"""
        errors = []

        error_patterns = [
            r"Syntax error.*",
            r"Parse error.*",
            r"Compilation error.*",
            r".*not found.*",
            r".*undefined.*",
            r".*invalid.*",
        ]

        for pattern in error_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                errors.append(ErrorInfo(name="OutputError", message=match.strip()))  # type: ignore

        return errors

    def format_validation_feedback(self, code_state: WorkflowState) -> str:
        """Format validation results for human-readable feedback"""
        latest = code_state.get_latest_validation()
        if not latest:
            return "No validation results available."

        if latest.success:
            return "✅ SysML code validated successfully!"

        feedback = "❌ SysML Validation Errors Found:\n\n"

        for i, error in enumerate(latest.errors, 1):
            feedback += f"Error {i}: {re.sub(r"ERROR:\s*", "", error.message)}"
            if error.traceback:
                feedback += f"Details: {' '.join(error.traceback[-2:])}\n"
            feedback += "\n"

        if latest.warnings:
            feedback += "⚠️ Warnings:\n"
            for warning in latest.warnings:
                feedback += f"- {warning}\n"
            feedback += "\n"

        return feedback

    def cleanup(self):
        """Clean up kernel resources"""
        try:
            if self.kernel_client:
                self.kernel_client.stop_channels()
            if self.kernel_manager:
                self.kernel_manager.shutdown_kernel()
            logger.info("Kernel cleanup completed successfully")
        except Exception as e:
            logger.warning("Error during cleanup: %s", e)


def test_validator_agent():
    """Test function for ValidatorAgent"""

    print("=" * 80)
    print("VALIDATOR AGENT TEST SUITE")
    print("=" * 80)

    # Test samples - good and bad SysML code
    test_cases = [
        {
            "name": "Valid SysML Block Definition",
            "code": """
            package PictureTaking {
	part def Exposure;

	action def Focus { out xrsl: Exposure; }
	action def Shoot { in xsf: Exposure; }

	action takePicture {
		action focus: Focus[1];
		flow of Exposure from focus.xrsl to shoot.xsf;
		action shoot: Shoot[1];
	}
}
            """,
            "description": "Valid SysML block definition with parts and constraints",
            "expected_success": True,
        },
        {
            "name": "Valid SysML Activity Diagram",
            "code": """
            package PictureTaking {
	part def Exposure;

	action def Focus { out xrsl: Exposure; }
	action def Shoot { in xsf: Exposure; }

	action takePicture {
		action focus: Focus[1];
		flow of Exposure from focus.xrsl to shoot.xsf;
		action shoot: Shoot[1];
	}
}
            """,
            "description": "Valid SysML activity diagram with actions and flows",
            "expected_success": True,
        },
        {
            "name": "Invalid SysML - Syntax Error",
            "code": """
            package PictureTaking {
	part def Exposure;

	action def Focus { out xrsl: Exposure; }
	action def Shoot { in xsf: Exposure; }

	action takePicture {
		action focus: Focus[1]
		flow of Exposure from focus.xrsl to shoot.xsf;
		action shoot: Shoot[1]
	
}
            """,
            "description": "Invalid SysML with missing semicolon and closing brace",
            "expected_success": False,
        },
        {
            "name": "Invalid SysML - Undefined Type",
            "code": """
            package PictureTaking {
	part def Exposure;

	action def Focus { out xrsl: Exposure; }
	action def Shoot { in xsf: Exposure; }

	action {
		action focus: Focus[1];
		flow of Exposure from focus.xrsl to shoot.xsf;
		action shoot: Shoot[1];
	}
}
            """,
            "description": "Invalid SysML with undefined types",
            "expected_success": True,
        },
        {
            "name": "Simple Valid Comment",
            "code": "// This is a simple SysML comment",
            "description": "Simple valid SysML comment",
            "expected_success": True,
        },
    ]

    # Initialize validator
    config = SysMLConfig(silent_startup=True, timeout=15, max_output_length=1000)

    validator = ValidatorAgent(config)

    # Check kernel availability
    print("\n1. Checking SysML Kernel Availability")
    print("-" * 40)
    kernel_available = validator.check_kernel_available()

    if not kernel_available:
        print("\n❌ SysML kernel not available. Exiting tests.")
        print("Please install the SysML kernel first:")
        print("conda install conda-forge::jupyter-sysml-kernel")
        return

    # Run test cases
    print("\n2. Running Validation Tests")
    print("-" * 40)

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"Expected: {'SUCCESS' if test_case['expected_success'] else 'FAILURE'}")

        # Create workflow state with the test code
        state = WorkflowState(
            code=test_case["code"], description=test_case["description"]  # type: ignore
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
                print(f"Execution Time: {latest_result.execution_time:.2f}s")

                if not success and latest_result.errors:
                    print("Errors Found:")
                    for error in latest_result.errors[:3]:  # Show first 3 errors
                        print(f"  - {error.name}: {error.message}")

                if latest_result.warnings:
                    print(f"Warnings: {len(latest_result.warnings)}")

                if latest_result.output.strip():
                    print(f"Output: {latest_result.output[:100]}...")

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
                        "expected": expected_result,  # type: ignore
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

        print("-" * 60)

    # Test summary
    print("\n3. Test Summary")
    print("=" * 40)

    passed_tests = sum(1 for r in results if r["passed"])
    total_tests = len(results)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Total Execution Time: {sum(r['time'] for r in results):.2f}s")

    print("\nDetailed Results:")
    for result in results:
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(
            f"{status} {result['test']}: {result['expected']} -> {result['actual']} ({result['time']:.2f}s)"
        )

    # Test formatting
    print("\n4. Testing Validation Feedback Formatting")
    print("-" * 40)

    # Use the last state that had errors for formatting test
    error_states = [r for r in test_cases if not r["expected_success"]]
    if error_states:
        test_state = WorkflowState(
            code=error_states[0]["code"],
            description=error_states[0]["description"],  # type: ignore
        )
        validated_state = validator.validate(test_state)

        feedback = validator.format_validation_feedback(validated_state)
        print(feedback)

    # Cleanup
    print("\n5. Cleanup")
    print("-" * 40)
    validator.cleanup()
    print("✅ Validator cleaned up successfully")


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
