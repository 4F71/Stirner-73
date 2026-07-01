"""
Router Evaluation — training/router-merged modeli üzerinde confusion matrix + accuracy.

Kullanım:
    python workspace/eval_router.py
    python workspace/eval_router.py --model training/router-merged
    python workspace/eval_router.py --ollama free-router   # Ollama modeli varsa

Çıktı:
    - Per-class accuracy
    - Confusion matrix (terminalde ASCII)
    - Yanlış tahminlerin listesi
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

# Val set (hold-out): train_router_lora.py tarafından oluşturulur, eğitimde görülmez
# Yoksa tam dataset kullanılır (geriye dönük uyumluluk için)
VAL_DATASET  = Path(__file__).parent / "data" / "val_dataset.jsonl"
FULL_DATASET = Path(__file__).parent / "data" / "cursor_dataset.jsonl"
DATASET      = VAL_DATASET if VAL_DATASET.exists() else FULL_DATASET
DEFAULT_MODEL = Path(__file__).parent / "router-lora"
LABELS = ["code", "research"]

# ---------------------------------------------------------------------------
# Ek test örnekleri — dataset'te olmayan, görülmemiş edge-case'ler
# ---------------------------------------------------------------------------
EXTRA_TESTS = [
    # ── KOLAY (baseline) ────────────────────────────────────────────
    ("bir fonksiyon yaz",                             "code"),
    ("quantization nedir",                            "research"),
    ("pytest testi ekle",                             "code"),
    ("flash attention nasıl çalışır",                 "research"),
    ("remove unused imports",                         "code"),

    # ── TEK KELİME / ÇOK KISA ───────────────────────────────────────
    ("refactor",                                      "code"),
    ("debug",                                         "code"),
    ("optimize",                                      "code"),
    ("düzelt",                                        "code"),
    ("test",                                          "code"),

    # ── PASİF CÜMLE (eylem yok, ama fix istiyor) ────────────────────
    ("kod çalışmıyor",                                "code"),
    ("testler patlıyor",                              "code"),
    ("import hata veriyor",                           "code"),
    ("neden hata veriyor",                            "code"),
    ("bu neden çalışmıyor",                           "code"),

    # ── SORU FORMUNDA AMA AKSIYON ────────────────────────────────────
    ("bunu nasıl fix ederim",                         "code"),
    ("bunu nasıl yapabilirim",                        "code"),
    ("can you make this work",                        "code"),

    # ── TÜRKÇE/İNGİLİZCE KARIŞIK ────────────────────────────────────
    ("bu function'ı fix et",                          "code"),
    ("type hint ekle buna",                           "code"),
    ("şunu implement et",                             "code"),

    # ── GERÇEKTEn SINIR DURUM (ikisi de olabilir) ───────────────────
    ("bu doğru yaklaşım mı",                          "research"),
    ("async kullanmalı mıyım burada",                 "research"),
    ("bu fonksiyonu optimize etmeli miyim",           "research"),
    ("bu kodu incele",                                "code"),
    ("bunu açıkla ama örnek de ver",                  "research"),
    ("bu yaklaşım doğru mu",                          "research"),
    ("overengineering mi bu",                         "research"),

    # ── YANILTICI (araştırma gibi görünüp aslında fix) ──────────────
    ("neden bu kadar yavaş",                          "code"),
    ("memory leak var sanırım",                       "code"),
    ("şu hata ne anlama geliyor ve nasıl çözerim",    "code"),
    ("bu exception'ı handle et",                      "code"),

    # ── YANILTICI (eylem gibi görünüp aslında araştırma) ────────────
    ("LoRA'yı anlat",                                 "research"),
    ("gradient checkpointing'i açıkla",               "research"),
    ("transformer mimarisini özetle",                 "research"),
    ("farkı nedir bunların",                          "research"),
    ("ne zaman kullanmalıyım bunu",                   "research"),
]


def load_dataset() -> list[tuple[str, str]]:
    samples = []
    with open(DATASET, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            msgs = obj["messages"]
            user_text = next(m["content"] for m in msgs if m["role"] == "user")
            answer = json.loads(next(m["content"] for m in msgs if m["role"] == "assistant"))
            samples.append((user_text, answer["intent"]))
    return samples


BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"


def _is_adapter(model_path: str) -> bool:
    """adapter_config.json varsa bu bir LoRA adapter'ıdır, merged model değil."""
    return (Path(model_path) / "adapter_config.json").exists()


def predict_hf(prompts: list[str], model_path: str) -> list[str]:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    SYSTEM = ('Sen bir intent sınıflandırıcısın. Kullanıcı prompt\'una göre '
              'SADECE şu 2 değerden birini JSON olarak döndür: '
              '{"intent": "code"} | {"intent": "research"}')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if _is_adapter(model_path):
        from peft import PeftModel
        print(f"LoRA adapter yükleniyor: {model_path}")
        print(f"  Base model: {BASE_MODEL}")
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16)
        model = PeftModel.from_pretrained(base, model_path)
    else:
        print(f"Merged model yükleniyor: {model_path}")
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16)

    model.eval()
    model.to(device)
    print(f"  ✓ {device.upper()} üzerinde hazır\n")

    preds = []
    for prompt in prompts:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=24, do_sample=False)
        decoded = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
        try:
            intent = json.loads(decoded)["intent"]
            if intent not in LABELS:
                intent = "?"
        except Exception:
            intent = "?"
        preds.append(intent)
    return preds


def predict_ollama(prompts: list[str], model_name: str) -> list[str]:
    import requests

    SYSTEM = ('Sen bir intent sınıflandırıcısın. Kullanıcı prompt\'una göre '
              'SADECE şu 2 değerden birini JSON olarak döndür: '
              '{"intent": "code"} | {"intent": "research"}')

    preds = []
    for prompt in prompts:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 24},
        }
        r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=30)
        r.raise_for_status()
        raw = r.json()["message"]["content"].strip()
        try:
            intent = json.loads(raw)["intent"]
            if intent not in LABELS:
                intent = "?"
        except Exception:
            intent = "?"
        preds.append(intent)
    return preds


def confusion_matrix(y_true: list[str], y_pred: list[str]) -> dict:
    matrix = {t: {p: 0 for p in LABELS + ["?"]} for t in LABELS}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def print_report(samples: list[tuple[str, str]], preds: list[str], title: str) -> None:
    texts = [s[0] for s in samples]
    truths = [s[1] for s in samples]

    correct = sum(t == p for t, p in zip(truths, preds))
    total = len(truths)
    acc = correct / total * 100

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  Accuracy: {correct}/{total}  ({acc:.1f}%)")
    print(f"{'='*60}")

    # Confusion matrix
    cm = confusion_matrix(truths, preds)
    col_w = 10
    print(f"\n  Confusion Matrix (satır=gerçek, sütun=tahmin):")
    header = f"{'':12}" + "".join(f"{l:>{col_w}}" for l in LABELS) + f"{'?':>{col_w}}"
    print(f"  {header}")
    print(f"  {'-'*len(header)}")
    for true_label in LABELS:
        row = f"  {true_label:<12}" + "".join(
            f"{cm[true_label][p]:>{col_w}}" for p in LABELS
        ) + f"{cm[true_label].get('?', 0):>{col_w}}"
        print(row)

    # Per-class accuracy
    print("\n  Per-class accuracy:")
    for label in LABELS:
        n = sum(1 for t in truths if t == label)
        c = sum(1 for t, p in zip(truths, preds) if t == label and p == label)
        bar = "█" * c + "░" * (n - c)
        print(f"    {label:<10} {c}/{n}  [{bar}]")

    # Yanlış tahminler
    errors = [(text, t, p) for text, t, p in zip(texts, truths, preds) if t != p]
    if errors:
        print(f"\n  Yanlış tahminler ({len(errors)}):")
        for text, true, pred in errors:
            print(f"    [gerçek={true:<10} tahmin={pred:<10}] {text[:60]}")
    else:
        print("\n  Hata yok!")


def main() -> None:
    import datetime, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="HF model path")
    parser.add_argument("--ollama", default=None, help="Ollama model adı (varsa)")
    parser.add_argument("--no-extra", action="store_true", help="Sadece dataset örnekleri")
    parser.add_argument("--run", default="v2_2class_300ex", help="Hangi run klasörüne kaydedilsin (training/results/<run>/)")
    parser.add_argument("--out", default=None, help="Sonuçları kaydet (--run'ı override eder)")
    args = parser.parse_args()

    dataset_samples = load_dataset()
    extra_samples = [] if args.no_extra else EXTRA_TESTS
    all_samples = dataset_samples + extra_samples

    prompts = [s[0] for s in all_samples]

    if args.ollama:
        print(f"Ollama modeli kullanılıyor: {args.ollama}")
        preds = predict_ollama(prompts, args.ollama)
        model_tag = args.ollama
    else:
        preds = predict_hf(prompts, args.model)
        model_tag = Path(args.model).name

    # Çıktıyı hem terminale hem dosyaya yaz
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = Path(__file__).parent / "results" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_{model_tag}_{ts}.txt"

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
        def flush(self):
            for f in self.files:
                f.flush()

    with open(out_file, "w", encoding="utf-8") as f:
        tee = Tee(sys.stdout, f)
        orig_stdout = sys.stdout
        sys.stdout = tee

        print(f"Model : {args.ollama or args.model}")
        print(f"Tarih : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print_report(dataset_samples, preds[:len(dataset_samples)], "Dataset Örnekleri (eğitim seti)")
        if extra_samples:
            print_report(extra_samples, preds[len(dataset_samples):], "Edge-Case Örnekleri (görmediği)")
        print_report(all_samples, preds, "TOPLAM")

        sys.stdout = orig_stdout

    print(f"\n📄 Sonuçlar kaydedildi: {out_file}")


if __name__ == "__main__":
    main()
