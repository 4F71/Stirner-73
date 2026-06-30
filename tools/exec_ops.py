"""Sandboxed Python execution — lets agents actually run the code they write."""

import os
import shlex
import subprocess
import sys

from tools.file_ops import WORKSPACE_ROOT

# FREE_PYTHON_TIMEOUT env var ile override edilebilir; yoksa 30s varsayilan.
_ENV_TIMEOUT = os.environ.get("FREE_PYTHON_TIMEOUT")
TIMEOUT_SECONDS: int = int(_ENV_TIMEOUT) if _ENV_TIMEOUT and _ENV_TIMEOUT.isdigit() else 30


def run_python(path: str, args: str = "", timeout: int = TIMEOUT_SECONDS) -> str:
    """Runs a Python script from inside workspace/ and returns stdout+stderr."""
    target = os.path.realpath(os.path.join(WORKSPACE_ROOT, path))
    if os.path.commonpath([target, WORKSPACE_ROOT]) != WORKSPACE_ROOT:
        raise ValueError(f"run_python only allowed inside workspace/: {path}")
    if not os.path.isfile(target):
        raise ValueError(f"file not found: {path}")

    cmd = [sys.executable, target] + (shlex.split(args) if args else [])
    try:
        proc = subprocess.run(
            cmd, cwd=WORKSPACE_ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return (
            f"ERROR: execution timed out after {timeout}s. "
            "Artirmak icin: run_python(..., timeout=<saniye>) "
            "veya FREE_PYTHON_TIMEOUT=<saniye> env var."
        )

    output = (proc.stdout + proc.stderr).strip()
    return f"[exit code {proc.returncode}]\n{output}" if output else f"[exit code {proc.returncode}] (cikti yok)"


EXEC_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run a Python script inside the workspace/ sandbox and return its stdout/stderr. "
                "Use this to actually test code right after writing it with write_file/edit_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Script path relative to workspace root, e.g. test.py",
                    },
                    "args": {
                        "type": "string",
                        "description": "Optional space-separated command-line arguments.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            f"Max execution time in seconds (default {TIMEOUT_SECONDS}). "
                            "Increase for long-running scripts (e.g. training loops)."
                        ),
                    },
                },
                "required": ["path"],
            },
        },
    },
]

EXEC_TOOL_EXECUTOR = {
    "run_python": run_python,
}
