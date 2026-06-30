"""Rollback journal for write_file/edit_file — undo recent workspace writes.

rollback() itself stays human-only (triggered via `/rollback` shell command or
`free rollback` CLI command) — undoing a write is destructive enough that it
should not be agent-callable. rollback_history() is read-only and is exposed
to agents via ROLLBACK_TOOLS_SCHEMA so they can check their own recent edits.
"""

import json
import os
import time

WORKSPACE_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "workspace")
)
ROLLBACK_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "logs", "rollback")
)
SNAPSHOT_DIR = os.path.join(ROLLBACK_DIR, "snapshots")
JOURNAL_PATH = os.path.join(ROLLBACK_DIR, "journal.jsonl")


def _read_journal() -> list[dict]:
    if not os.path.isfile(JOURNAL_PATH):
        return []
    with open(JOURNAL_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_journal(entries: list[dict]) -> None:
    with open(JOURNAL_PATH, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def record_snapshot(rel_path: str, abs_path: str) -> None:
    """Captures the pre-write state of a workspace file before write_file/edit_file
    overwrites it, so rollback() can later restore it."""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    entry_id = time.time_ns()
    existed_before = os.path.isfile(abs_path)
    snapshot_file = None

    if existed_before:
        snapshot_file = f"{entry_id}.snap"
        # Byte modunda kopyala: text mode binary dosyalarda UnicodeDecodeError firlatir
        # ve snapshot alinamadan write_file/edit_file coker.
        with open(abs_path, "rb") as f:
            content = f.read()
        with open(os.path.join(SNAPSHOT_DIR, snapshot_file), "wb") as f:
            f.write(content)

    entries = _read_journal()
    entries.append({
        "id": entry_id,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "path": rel_path,
        "existed_before": existed_before,
        "snapshot": snapshot_file,
    })
    _write_journal(entries)


def rollback(steps: int = 1) -> str:
    """Undoes the last `steps` write_file/edit_file operations, most recent first."""
    entries = _read_journal()
    if not entries:
        return "rollback gecmisi bos, geri alinacak islem yok"

    steps = max(1, min(steps, len(entries)))
    to_undo = entries[-steps:]
    remaining = entries[:-steps]

    restored = []
    for entry in reversed(to_undo):
        target = os.path.realpath(os.path.join(WORKSPACE_ROOT, entry["path"]))
        if entry["existed_before"]:
            snapshot_path = os.path.join(SNAPSHOT_DIR, entry["snapshot"])
            if os.path.isfile(snapshot_path):
                with open(snapshot_path, "rb") as f:
                    content = f.read()
                with open(target, "wb") as f:
                    f.write(content)
                restored.append(f"geri alindi: {entry['path']} (onceki haline donduruldu)")
            else:
                restored.append(f"UYARI: {entry['path']} icin snapshot bulunamadi, atlandi")
        else:
            if os.path.isfile(target):
                os.remove(target)
            restored.append(f"silindi: {entry['path']} (bu islemle yaratilmisti)")

        if entry.get("snapshot"):
            snapshot_path = os.path.join(SNAPSHOT_DIR, entry["snapshot"])
            if os.path.isfile(snapshot_path):
                os.remove(snapshot_path)

    _write_journal(remaining)
    return "\n".join(restored)


def rollback_history(n: int = 10) -> str:
    """Lists the last n journal entries, most recent first (does not undo anything)."""
    entries = _read_journal()
    if not entries:
        return "rollback gecmisi bos"

    n = max(1, min(n, len(entries)))
    lines = []
    for entry in reversed(entries[-n:]):
        action = "yazildi/duzenlendi" if entry["existed_before"] else "yeni olusturuldu"
        lines.append(f"[{entry['ts']}] {entry['path']} ({action})")
    return "\n".join(lines)


ROLLBACK_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "rollback_history",
            "description": (
                "workspace/ icindeki son N write_file/edit_file isleminin gecmisini listeler "
                "(geri almaz, sadece gosterir)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer", "description": "Gosterilecek kayit sayisi (varsayilan 10)."}},
                "required": [],
            },
        },
    }
]

ROLLBACK_TOOL_EXECUTOR = {"rollback_history": rollback_history}
