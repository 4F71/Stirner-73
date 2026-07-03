"""Git tools — read-only context araçları + commit/push yazma araçları."""

import os
import subprocess
import tempfile

from tools.file_ops import PROJECT_ROOT

TIMEOUT_SECONDS = 15


def _run_git(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git"] + args, cwd=PROJECT_ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return "ERROR: git not installed"
    except subprocess.TimeoutExpired:
        return f"ERROR: git timed out after {TIMEOUT_SECONDS}s"

    output = (proc.stdout + proc.stderr).strip()
    return output or "(degisiklik yok)"


def git_diff(path: str = "") -> str:
    """Shows unstaged changes, optionally scoped to one path."""
    args = ["diff"]
    if path:
        args += ["--", path]
    return _run_git(args)


def git_diff_staged(path: str = "") -> str:
    """Shows staged (git add edilmis) degisiklikleri gosterir."""
    args = ["diff", "--cached"]
    if path:
        args += ["--", path]
    return _run_git(args)


def git_log(path: str = "", n: int = 10) -> str:
    """Shows recent commit history, optionally scoped to one path."""
    capped = max(1, min(n, 50))
    args = ["log", f"-{capped}", "--oneline"]
    if path:
        args += ["--", path]
    return _run_git(args)


def git_status() -> str:
    """Shows staged/unstaged/untracked file state (short format)."""
    return _run_git(["status", "--short"])


def git_add(path: str) -> str:
    """Stages a file or directory for commit."""
    return _run_git(["add", "--", path])


def git_commit(message: str) -> str:
    """Commits staged changes with the given message (UTF-8 safe, supports Turkish)."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False)
    try:
        tmp.write(message)
        tmp.close()
        return _run_git(["commit", "-F", tmp.name])
    finally:
        os.unlink(tmp.name)


def git_push(remote: str = "origin", branch: str = "") -> str:
    """Pushes commits to remote. Defaults to 'origin'; branch defaults to current branch."""
    args = ["push", remote]
    if branch:
        args.append(branch)
    return _run_git(args)


GIT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show unstaged git changes in the project (read-only). Use to see what was just edited.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional path (relative to project root) to scope the diff to.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Show recent commit history (read-only, oneline format).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional path (relative to project root) to scope the log to.",
                    },
                    "n": {
                        "type": "integer",
                        "description": "Max number of commits to show (default 10, capped at 50).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff_staged",
            "description": "Show staged (git add edilmis) changes only (read-only). Use for pre-commit review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional path to scope the diff to.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show staged/unstaged/untracked file state (read-only, short format). Use to see which files changed without showing the full diff.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

GIT_TOOLS_SCHEMA += [
    {
        "type": "function",
        "function": {
            "name": "git_add",
            "description": "Stage a file or directory for the next commit (git add).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to stage (e.g. 'agents/coder.py' or '.')"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Commit staged changes with a message. UTF-8 safe — supports Turkish characters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Commit mesajı (conventional commit formatında: 'fix(kapsam): açıklama')."},
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_push",
            "description": "Push committed changes to remote (default: origin, current branch).",
            "parameters": {
                "type": "object",
                "properties": {
                    "remote": {"type": "string", "description": "Remote adı (varsayılan: 'origin')."},
                    "branch": {"type": "string", "description": "Branch adı (boş bırakılırsa mevcut branch)."},
                },
                "required": [],
            },
        },
    },
]

GIT_TOOL_EXECUTOR = {
    "git_diff": git_diff,
    "git_diff_staged": git_diff_staged,
    "git_log": git_log,
    "git_status": git_status,
    "git_add": git_add,
    "git_commit": git_commit,
    "git_push": git_push,
}
