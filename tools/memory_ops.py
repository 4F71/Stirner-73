"""Proje kararlari icin kalici hafiza — chat gecmisinden bagimsiz, JSONL tabanli kayit defteri.
tools/rollback_ops.py ile ayni append-only desen, ancak bu defter insan-only degil: search_codebase
gibi ajanlarin da okuyup yazabilecegi bir "hatirlama" araci olarak tasarlandi.
"""

import json
import os
import time

MEMORY_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "logs", "memory")
)
MEMORY_PATH = os.path.join(MEMORY_DIR, "decisions.jsonl")


def _read_entries(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def remember(text: str, tag: str = "", path: str = MEMORY_PATH) -> str:
    """Bir karari/notu kalici hafizaya ekler."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entry = {
        "id": time.time_ns(),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": tag,
        "text": text,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return f"kaydedildi: {text}"


def recall(query: str = "", n: int = 10, path: str = MEMORY_PATH) -> str:
    """Kayitli kararlari arar/listeler; query bossa son n kaydi dondurur."""
    entries = _read_entries(path)
    if not entries:
        return "Hafiza bos, henuz kayitli karar yok."

    if query:
        q = query.casefold()
        entries = [e for e in entries if q in e["text"].casefold() or q in e.get("tag", "").casefold()]
    if not entries:
        return "Eslesen kayit bulunamadi."

    n = max(1, min(n, len(entries)))
    entries = entries[-n:]
    lines = []
    for e in reversed(entries):
        tag_str = f"({e['tag']}) " if e.get("tag") else ""
        lines.append(f"[{e['ts']}] {tag_str}{e['text']}")
    return "\n".join(lines)


MEMORY_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "remember_decision",
            "description": (
                "Bu projede alinan onemli bir karari, bulguyu veya kisitlamayi kalici hafizaya "
                "kaydeder. Bu hafiza chat gecmisinden bagimsizdir ve sonraki session'larda otomatik "
                "olarak sistem promptuna eklenir."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Kaydedilecek karar/not."},
                    "tag": {"type": "string", "description": "Opsiyonel kisa etiket, orn: 'model-secimi'."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_decisions",
            "description": "Daha once kaydedilmis proje kararlarini arar veya listeler.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Aranacak anahtar kelime (bossa son kayitlari listeler)."},
                    "n": {"type": "integer", "description": "Maksimum sonuc sayisi (varsayilan 10)."},
                },
                "required": [],
            },
        },
    },
]

MEMORY_TOOL_EXECUTOR = {
    "remember_decision": remember,
    "recall_decisions": recall,
}


SUMMARY_SYSTEM_PROMPT = (
    "Asagidaki konusma gecmisini, sonraki bir oturumda hatirlanmasi gereken "
    "onemli karar/bulgu/kisitlama varsa TEK CUMLEYLE Turkce ozetle. Onemli bir "
    "sey yoksa SADECE 'YOK' yaz. Baska hicbir sey yazma."
)


def summarize_and_remember(client, model: str, messages: list[dict], max_history: int = 20, path: str = MEMORY_PATH) -> str:
    """Oturum kapanirken konusma gecmisinin ozetini cikarip otomatik remember eder.

    Onemli bir sey bulunamazsa (model 'YOK' derse) hicbir sey kaydetmez ve "" dondurur.
    """
    exchanged = [m for m in messages if m.get("role") in ("user", "assistant") and m.get("content")]
    if len(exchanged) < 2:
        return ""

    summary_messages = [{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}] + exchanged[-max_history:]
    try:
        response = client.chat(model, summary_messages, options={"temperature": 0.2})
        summary = response.get("message", {}).get("content", "").strip()
    except Exception:
        return ""

    if not summary or summary.upper().startswith("YOK"):
        return ""

    remember(summary, tag="session-ozeti", path=path)
    return summary
