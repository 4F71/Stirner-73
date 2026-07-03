"""Training run karşılaştırma — training/results/ altındaki versiyonları okur,
accuracy ve edge-case metriklerini tablo olarak döndürür.
"""

import os
import re
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "training" / "results"

_RE_ACCURACY = re.compile(r"Accuracy:\s*(\d+)/(\d+)\s*\(([0-9.]+)%\)")
_RE_MODEL = re.compile(r"Model\s*:\s*(.+)")
_RE_DATE = re.compile(r"Tarih\s*:\s*(.+)")


def _parse_eval_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    result = {"file": path.name, "run": path.parent.name}

    m = _RE_DATE.search(text)
    if m:
        result["date"] = m.group(1).strip()

    # Dosyada üç blok var: eğitim seti, edge-case, toplam
    # Her "Accuracy:" satırını sırayla al
    matches = _RE_ACCURACY.findall(text)
    if len(matches) >= 1:
        correct, total, pct = matches[0]
        result["train_acc"] = f"{correct}/{total} ({pct}%)"
    if len(matches) >= 2:
        correct, total, pct = matches[1]
        result["edge_acc"] = f"{correct}/{total} ({pct}%)"
    if len(matches) >= 3:
        correct, total, pct = matches[2]
        result["total_acc"] = f"{correct}/{total} ({pct}%)"
        result["total_pct"] = float(pct)

    return result


def compare_runs(run_filter: str = "") -> str:
    """training/results/ altındaki tüm eval dosyalarını karşılaştırır."""
    if not RESULTS_DIR.exists():
        return f"Sonuç dizini bulunamadı: {RESULTS_DIR}"

    eval_files = sorted(RESULTS_DIR.rglob("eval_*.txt"))
    if run_filter:
        eval_files = [f for f in eval_files if run_filter in f.parent.name]

    if not eval_files:
        return "Karşılaştırılacak eval dosyası bulunamadı."

    rows = [_parse_eval_file(f) for f in eval_files]
    rows.sort(key=lambda r: r.get("total_pct", 0), reverse=True)

    # Kolon genişlikleri
    col_run  = max(len(r["run"]) for r in rows)
    col_date = max(len(r.get("date", "-")) for r in rows)
    col_tr   = max(len(r.get("train_acc", "-")) for r in rows)
    col_ed   = max(len(r.get("edge_acc", "-")) for r in rows)
    col_tot  = max(len(r.get("total_acc", "-")) for r in rows)

    header = (
        f"{'Run':<{col_run}}  {'Tarih':<{col_date}}  "
        f"{'Eğitim':<{col_tr}}  {'Edge-Case':<{col_ed}}  {'Toplam':<{col_tot}}"
    )
    sep = "─" * len(header)
    lines = [sep, header, sep]
    for r in rows:
        lines.append(
            f"{r['run']:<{col_run}}  {r.get('date','-'):<{col_date}}  "
            f"{r.get('train_acc','-'):<{col_tr}}  "
            f"{r.get('edge_acc','-'):<{col_ed}}  "
            f"{r.get('total_acc','-'):<{col_tot}}"
        )
    lines.append(sep)
    lines.append(f"Toplam {len(rows)} eval dosyası · {RESULTS_DIR}")
    return "\n".join(lines)
