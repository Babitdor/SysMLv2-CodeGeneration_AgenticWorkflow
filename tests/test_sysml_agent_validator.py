import asyncio
import builtins
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agents.SysMLAgent import SysMLAgent
from states.ValidationState import ValidationStatus
from states.WorkflowState import WorkflowState
from tools.SysMLValidatorTool import sysml_validator_tool, validate_via_sidecar


CODE = "package Test {}"


def make_agent(model_claim: bool) -> SysMLAgent:
    response = {
        "code": CODE,
        "validated": model_claim,
        "syntax_checks_performed": [],
    }
    deep_agent = SimpleNamespace(
        invoke=lambda *args, **kwargs: {
            "messages": [SimpleNamespace(content=json.dumps(response))]
        }
    )
    with patch("agents.SysMLAgent.create_deep_agent", return_value=deep_agent):
        return SysMLAgent(llm=object())


def generate(model_claim: bool, report: str) -> WorkflowState:
    agent = make_agent(model_claim)
    state = WorkflowState(original_query="model Test", processed_query="model Test")
    with patch(
        "tools.SysMLValidatorTool.SysMLValidatorTool._run", return_value=report
    ) as validator:
        result = agent.generate_code(state)
    validator.assert_called_once_with(code=CODE)
    return result


def test_model_claim_cannot_override_validator_failure():
    report = "Validation backend: sidecar\nFound 1 semantic error(s):\nERROR: bad model"
    result = generate(model_claim=True, report=report)

    assert result.is_valid == ValidationStatus.INVALID
    assert result.error == report
    assert result.validation_history[-1].success is False
    assert result.validation_history[-1].output == report
    assert report in make_agent(True)._create_prompt_with_context(result)


def test_sidecar_success_overrides_model_self_doubt():
    report = "Validation backend: sidecar\n✅ Code is valid! No syntax or semantic errors found."
    result = generate(model_claim=False, report=report)

    assert result.is_valid == ValidationStatus.VALID
    assert result.error == ""
    assert result.validation_history[-1].success is True


@pytest.mark.parametrize(
    "report",
    [
        "Validation backend: local-semantic\n"
        "Compiler validation unavailable: Sidecar request failed (URLError)\n"
        "Local checks passed. Compiler acceptance has not been established.",
        "Validation backend: sidecar\n"
        "Validation incomplete; compiler acceptance has not been established.",
    ],
)
def test_incomplete_or_fallback_validation_does_not_approve(report):
    result = generate(model_claim=True, report=report)
    assert result.is_valid == ValidationStatus.INVALID
    assert result.validation_history[-1].success is False


def test_validator_runs_without_external_mbse_package(monkeypatch):
    original_import = builtins.__import__

    def without_mbse(name, *args, **kwargs):
        if name == "mbse" or name.startswith("mbse."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_mbse)
    monkeypatch.setitem(os.environ, "SYSML_KERNEL_SERVICE_URL", "")
    report = asyncio.run(sysml_validator_tool._arun(code=CODE))

    assert report.startswith("Validation backend: local-semantic")
    assert "Local checks passed. Compiler acceptance has not been established." in report
    assert "Error during validation" not in report


def test_sidecar_contract(monkeypatch):
    response = MagicMock()
    response.read.return_value = (
        b'{"success": true, "completed": true, "errors": []}'
    )
    response.__enter__.return_value = response
    monkeypatch.setenv("SYSML_KERNEL_SERVICE_URL", "http://sysml-kernel:8888/")

    with patch("tools.SysMLValidatorTool.urlopen", return_value=response) as request:
        result = validate_via_sidecar(CODE)

    http_request = request.call_args.args[0]
    assert http_request.full_url == "http://sysml-kernel:8888/validate"
    assert json.loads(http_request.data) == {"sysml_text": CODE}
    assert result.backend == "sidecar"
    assert result.completed is True
    assert not result.has_errors
