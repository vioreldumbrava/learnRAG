"""Exercise the installed command surface, including validation before I/O."""

import subprocess
import sys

import pytest


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "rag_app", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )


def test_cli_entrypoint_exposes_rebuild():
    result = run_cli("--help")
    assert result.returncode == 0, result.stderr
    assert "rebuild" in result.stdout


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["retrieve", " "], "blank"),
        (["query", " "], "blank"),
        (["retrieve", "question", "--top-k", "0"], "range"),
        (["inspect", "--sample", "0"], "range"),
    ],
)
def test_cli_rejects_bad_input_before_opening_config(arguments, message):
    result = run_cli(*arguments, "--config", "nonexistent-config-for-validation.yaml")
    assert result.returncode == 2
    assert message in result.stderr
    assert "Config file not found" not in result.stdout
