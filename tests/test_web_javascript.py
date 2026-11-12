import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Node is optional locally; installed in CI"
)
@pytest.mark.parametrize("script", ["test_sse.mjs", "test_citations.mjs"])
def test_browser_modules(script):
    """The dependency-free JS modules: the SSE parser and the citation splitter."""

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["node", "--test", str(root / "tests/js" / script)],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
