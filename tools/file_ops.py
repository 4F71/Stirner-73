"""Workspace-scoped file tools exposed to agents via Ollama tool-calling."""

import base64
import os

from tools.rollback_ops import record_snapshot

WORKSPACE_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "workspace")
)
PROJECT_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..")
)

def resolve_read_path(path: str) -> str:
    # Okuma islemleri tum projede (PROJECT_ROOT) yapilabilir.
    target = os.path.realpath(os.path.join(PROJECT_ROOT, path))
    if os.path.commonpath([target, PROJECT_ROOT]) != PROJECT_ROOT:
        raise ValueError(f"read path escapes project root: {path}")
    return target

def resolve_write_path(path: str) -> str:
    # Yazma islemleri SADECE workspace icinde yapilabilir! (Guvenlik)
    target = os.path.realpath(os.path.join(WORKSPACE_ROOT, path))
    if os.path.commonpath([target, WORKSPACE_ROOT]) != WORKSPACE_ROOT:
        raise ValueError(f"write path escapes workspace sandbox: {path}")
    return target


def read_file(path: str) -> str:
    target = resolve_read_path(path)
    if not os.path.isfile(target):
        raise ValueError(f"file not found: {path}")
    try:
        with open(target, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Binary/UTF-8 olmayan dosya: ham bytes'i raw olarak dondurmek yerine
        # modeli yaniltmamak icin acik bir uyari don.
        size = os.path.getsize(target)
        return (
            f"[BINARY DOSYA] '{path}' UTF-8 metin olarak okunamadi "
            f"(muhtemelen ikili/binary icerik, {size} byte). Bu arac sadece metin dosyalarini okur."
        )


def write_file(path: str, content: str) -> str:
    target = resolve_write_path(path)
    record_snapshot(path, target)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)
    return f"wrote {len(content)} bytes to {path}"


def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replaces a single, unique occurrence of old_string with new_string in a workspace file."""
    target = resolve_write_path(path)
    if not os.path.isfile(target):
        raise ValueError(f"file not found: {path}")
    with open(target, "r", encoding="utf-8") as f:
        text = f.read()

    count = text.count(old_string)
    if count == 0:
        raise ValueError("old_string bulunamadi: dosyada birebir eslesme yok")
    if count > 1:
        raise ValueError(f"old_string {count} kez geciyor, daha spesifik (benzersiz) bir metin ver")

    record_snapshot(path, target)
    new_text = text.replace(old_string, new_string, 1)
    with open(target, "w", encoding="utf-8") as f:
        f.write(new_text)
    return f"edited {path}: {len(old_string)} -> {len(new_string)} chars replaced"


def list_workspace(path: str = ".", recursive: bool = False) -> list[str]:
    target = resolve_read_path(path)
    if not os.path.isdir(target):
        raise ValueError(f"not a directory: {path}")
    if not recursive:
        entries = os.listdir(target)
        return sorted(
            e + ("/" if os.path.isdir(os.path.join(target, e)) else "")
            for e in entries
        )
    results = []
    for root, dirs, files in os.walk(target):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        rel_root = os.path.relpath(root, target)
        for f in sorted(files):
            rel = os.path.join(rel_root, f) if rel_root != "." else f
            results.append(rel.replace("\\", "/"))
    return results


def read_image_b64(path: str) -> str:
    target = os.path.abspath(path)
    if not os.path.isfile(target):
        raise ValueError(f"file not found: {path}")
    with open(target, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


FILE_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file's contents. Allowed anywhere inside the project root (not just workspace/).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the project root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (overwrite) a text file. Restricted to the workspace/ sandbox only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the workspace root (writes cannot escape workspace/).",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace one exact, unique occurrence of old_string with new_string in a workspace file. "
                "Prefer this over write_file for small/targeted changes — it won't accidentally clobber "
                "the rest of the file. Fails if old_string is not found or matches more than once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the workspace root.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to find (must be unique in the file).",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Text to replace it with.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_workspace",
            "description": "List files/directories. Allowed anywhere inside the project root. Use recursive=true to walk all subdirectories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the project root. Defaults to project root.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "If true, walk all subdirectories and return relative paths. Default false.",
                    },
                },
                "required": [],
            },
        },
    },
]

TOOL_EXECUTOR = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "list_workspace": list_workspace,
}

# Web araçlarını da birleşik şemaya ekle
from tools.web_ops import WEB_TOOLS_SCHEMA, WEB_TOOL_EXECUTOR

FILE_TOOLS_SCHEMA = FILE_TOOLS_SCHEMA + WEB_TOOLS_SCHEMA
TOOL_EXECUTOR = {**TOOL_EXECUTOR, **WEB_TOOL_EXECUTOR}
