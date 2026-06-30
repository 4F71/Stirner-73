"""Sandboxed Python execution — lets agents actually run the code they write."""

import os
import shlex
import subprocess
import sys

from tools.file_ops import WORKSPACE_ROOT

TIMEOUT_SECONDS = 30


def run_python(path: str, args: str = "") -> str:
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
            encoding="utf-8", errors="replace", timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: execution timed out after {TIMEOUT_SECONDS}s"

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
                },
                "required": ["path"],
            },
        },
    },
]

EXEC_TOOL_EXECUTOR = {
    "run_python": run_python,
}
