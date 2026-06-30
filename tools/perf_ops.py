"""Per-turn model performance logging (token/s, latency).

Ollama /api/chat cevabindaki eval_count / eval_duration alanlari her tur sonunda
buraya yazilir; `free stats` komutu bu dosyayi okuyup model bazli ozet cikarir.
"""

import json
import os
import time
from collections import defaultdict

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
PERF_LOG_PATH = os.path.join(LOG_DIR, "perf.jsonl")


def log_turn(model: str, eval_count: int, eval_duration_ns: int) -> None:
    """Bir tur icin token/s verisini PERF_LOG_PATH'e ekler."""
    if eval_duration_ns <= 0 or eval_count <= 0:
        return
    tokens_per_sec = eval_count / (eval_duration_ns / 1e9)
    entry = {
        "ts": time.time(),
        "model": model,
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration_ns,
        "tokens_per_sec": round(tokens_per_sec, 2),
    }
    with open(PERF_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def perf_stats(model_filter: str = "") -> str:
    """PERF_LOG_PATH'i okuyup model bazli ozet dondurur."""
    if not os.path.exists(PERF_LOG_PATH):
        return "Henuz performans kaydi yok. `free shell` ile bir tur yaptiktan sonra tekrar dene."

    data: dict[str, list[float]] = defaultdict(list)
    with open(PERF_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = entry.get("model", "")
            tps = entry.get("tokens_per_sec", 0)
            if model_filter and model_filter.casefold() not in m.casefold():
                continue
            if tps > 0:
                data[m].append(tps)

    if not data:
        return "Eslesen performans kaydi bulunamadi."

    lines = ["Model Performans Ozeti (token/s)\n" + "─" * 50]
    for m, vals in sorted(data.items(), key=lambda x: -sum(x[1]) / len(x[1])):
        avg = sum(vals) / len(vals)
        mn = min(vals)
        mx = max(vals)
        lines.append(
            f"{m}\n"
            f"  Tur sayisi : {len(vals)}\n"
            f"  Ort token/s: {avg:.1f}  (min {mn:.1f} / max {mx:.1f})"
        )
    return "\n".join(lines)
