"""run_python timeout parametresi ve TimeoutExpired path testleri."""

import subprocess
from unittest.mock import patch

import pytest

from tools.exec_ops import run_python, TIMEOUT_SECONDS


def _fake_realpath(p):
    """Sandbox kontrolünü geçmek için WORKSPACE_ROOT altında görünen bir yol döndür."""
    from tools.file_ops import WORKSPACE_ROOT
    import os
    return os.path.join(WORKSPACE_ROOT, "fake_script.py")


def test_default_timeout_constant():
    assert isinstance(TIMEOUT_SECONDS, int)
    assert TIMEOUT_SECONDS > 0


def test_timeout_expired_returns_error_message():
    from tools.file_ops import WORKSPACE_ROOT
    import os

    with (
        patch("tools.exec_ops.os.path.realpath", side_effect=_fake_realpath),
        patch("tools.exec_ops.os.path.isfile", return_value=True),
        patch("tools.exec_ops.subprocess.run", side_effect=subprocess.TimeoutExpired("python", 5)),
    ):
        result = run_python("fake_script.py", timeout=5)

    assert "timed out after 5s" in result
    assert "FREE_PYTHON_TIMEOUT" in result


def test_custom_timeout_passed_to_subprocess():
    from tools.file_ops import WORKSPACE_ROOT
    import os

    captured = {}

    def fake_run(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")
        return proc

    with (
        patch("tools.exec_ops.os.path.realpath", side_effect=_fake_realpath),
        patch("tools.exec_ops.os.path.isfile", return_value=True),
        patch("tools.exec_ops.subprocess.run", side_effect=fake_run),
    ):
        run_python("fake_script.py", timeout=120)

    assert captured["timeout"] == 120
