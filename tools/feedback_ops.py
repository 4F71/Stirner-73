"""Router düzeltme kayıtları — kullanıcının /correct ile işaret ettiği yanlış
yönlendirmeleri logs/router_feedback.jsonl'a kaydeder.

Bu dosya ileride fine-tune dataseti olarak training/data/'ya aktarılabilir.
"""

import json
import os
import time

LOG_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "logs"))
FEEDBACK_PATH = os.path.join(LOG_DIR, "router_feedback.jsonl")

VALID_INTENTS = {"code", "research", "codebase"}


def record_correction(prompt: str, correct_intent: str, wrong_intent: str = "") -> str:
    """Yanlış yönlendirilen bir prompt için düzeltme kaydeder."""
    correct_intent = correct_intent.strip().lower()
    if correct_intent not in VALID_INTENTS:
        return f"Geçersiz intent: '{correct_intent}'. Geçerli değerler: {', '.join(sorted(VALID_INTENTS))}"

    os.makedirs(LOG_DIR, exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "correct_intent": correct_intent,
        "wrong_intent": wrong_intent,
    }
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return f"✓ Kaydedildi: '{prompt[:60]}' → {correct_intent}"


def feedback_stats() -> str:
    """Kayıtlı düzeltme sayısını ve dağılımını özetler."""
    if not os.path.isfile(FEEDBACK_PATH):
        return "Henüz router düzeltme kaydı yok."

    entries = []
    with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not entries:
        return "Kayıt dosyası boş."

    from collections import Counter
    dist = Counter(e["correct_intent"] for e in entries)
    lines = [f"Toplam düzeltme: {len(entries)}"]
    for intent, count in sorted(dist.items()):
        lines.append(f"  {intent}: {count}")
    lines.append(f"\nDosya: {FEEDBACK_PATH}")
    return "\n".join(lines)
