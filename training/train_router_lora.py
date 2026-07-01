"""
Router LoRA Fine-Tune — Qwen2.5-1.5B-Instruct
2-sınıf intent classifier: code | research

Gereksinimler:
  pip install "unsloth[cu124]" transformers datasets peft accelerate

  trl KULLANILMIYOR (Windows'ta triton uyumsuzluğu nedeniyle);
  eğitim saf transformers.Trainer ile yapılır.

Çalıştırma:
  python training/train_router_lora.py
  python training/train_router_lora.py --export-gguf   # GGUF export da yapar

Çıktı:
  training/router-lora/          ← adapter ağırlıkları
  training/router-q4.gguf        ← Ollama için Q4_K_M GGUF (--export-gguf ile)
"""

import argparse
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME   = "unsloth/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR   = Path(__file__).parent / "router-lora"
GGUF_PATH    = Path(__file__).parent / "router-q4.gguf"
DATASET_FILE = Path(__file__).parent / "data" / "cursor_dataset.jsonl"
RUN_NAME     = "v2_2class_300ex"     # results/ altında klasör adı; yeni eğitimde güncelle
RESULTS_DIR  = Path(__file__).parent / "results" / RUN_NAME

LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
MAX_SEQ_LEN  = 256

TRAIN_EPOCHS = 15         # 300 örnekle 15 epoch; 5 epoch yetmiyordu
BATCH_SIZE   = 4
GRAD_ACCUM   = 2
LR           = 3e-4

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
)


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def load_dataset(tokenizer):
    rows = []
    with open(DATASET_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Qwen2.5 chat template kullan
            text = tokenizer.apply_chat_template(
                obj["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            rows.append({"text": text})
    return Dataset.from_list(rows)


class _TokenDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        return {
            "input_ids": ids,
            "attention_mask": self.attention_mask[idx],
            "labels": ids.clone(),   # causal LM: hedef = girdi
        }


def train(model, tokenizer, dataset):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    texts = list(dataset["text"])   # Dataset column → plain list
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_tensors="pt",
    )
    torch_ds = _TokenDataset(enc)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=torch_ds,
    )
    print(f"[INFO] Training başlıyor: {len(torch_ds)} örnek, {TRAIN_EPOCHS} epoch")
    trainer.train()
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"[OK] Adapter kaydedildi: {OUTPUT_DIR}")
    return trainer


def export_gguf(model, tokenizer):
    """Adapter'ı base model ile merge et, merged/ olarak kaydet. Sonra llama.cpp ile GGUF'a çevir."""
    print("[INFO] Adapter merge ediliyor...")
    from peft import PeftModel
    merged_dir = Path(__file__).parent / "router-merged"

    # Fallback yolunda model peft sarmalıdır; değilse adapter'ı üsten yükle.
    if not isinstance(model, PeftModel):
        base = model
        merged = PeftModel.from_pretrained(base, str(OUTPUT_DIR))
    else:
        merged = model

    merged = merged.merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"[OK] Merged model kaydedildi: {merged_dir}")
    print()
    print("GGUF için sonraki adımlar:")
    print("  1. llama.cpp kur: https://github.com/ggerganov/llama.cpp")
    print(f"  2. python llama.cpp/convert_hf_to_gguf.py {merged_dir} --outfile {GGUF_PATH} --outtype q4_k_m")
    print(f"  3. ollama create free-router -f training/Modelfile.router")


def plot_loss(trainer, out_dir: Path) -> None:
    """Training loss eğrisini PNG olarak kaydet."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib bulunamadı, loss grafiği atlandı. pip install matplotlib")
        return

    history = trainer.state.log_history
    steps  = [e["step"] for e in history if "loss" in e]
    losses = [e["loss"] for e in history if "loss" in e]
    if not steps:
        print("[WARN] Loss verisi bulunamadı, grafik atlandı.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, losses, marker="o", linewidth=1.5, markersize=3, color="#4C72B0")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Router LoRA — Training Loss  ({TRAIN_EPOCHS} epoch, {len(steps)} log noktası)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    import datetime
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = assets_dir / f"loss_curve_{ts}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[OK] Loss eğrisi kaydedildi: {path}")


def run_quick_eval(model, tokenizer):
    """Training bittikten sonra birkaç örnek üzerinde hızlı doğruluk kontrolü."""
    import torch
    probes = [
        ("bu fonksiyonu yaz",        "code"),
        ("hata var düzelt",          "code"),
        ("çalışmıyor",               "code"),
        ("transformer nedir",        "research"),
        ("neden async kullanmalıyım","research"),
        ("pytest mi unittest mi",    "research"),
    ]
    correct = 0
    for prompt, expected in probes:
        messages = [
            {"role": "system", "content":
             "Sen bir intent sınıflandırıcısın. SADECE JSON döndür: "
             "{\"intent\": \"code\"} | {\"intent\": \"research\"}"},
            {"role": "user", "content": prompt},
        ]
        enc = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        # Yeni transformers BatchEncoding döndürüyor, eski sürümler tensor döndürüyordu.
        input_ids = (enc["input_ids"] if hasattr(enc, "__getitem__") else enc).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=24, temperature=0.0, do_sample=False
            )
        decoded = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        try:
            intent = json.loads(decoded).get("intent", "?")
        except Exception:
            intent = decoded[:20]
        ok = "✓" if intent == expected else "✗"
        print(f"  {ok}  '{prompt}' → {intent}  (beklenen: {expected})")
        if intent == expected:
            correct += 1
    print(f"\n  Quick-eval: {correct}/{len(probes)} ({'%.0f' % (correct/len(probes)*100)}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-gguf", action="store_true", help="Eğitim bittikten sonra Q4_K_M GGUF export yap")
    parser.add_argument("--eval-only",   action="store_true", help="Eğitim yapma, sadece kaydedilmiş adapter'ı değerlendir")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()

    if args.eval_only:
        print("[INFO] Eval-only modu — training atlandı")
        trainer = None
    else:
        dataset = load_dataset(tokenizer)
        trainer = train(model, tokenizer, dataset)
        plot_loss(trainer, RESULTS_DIR)

    run_quick_eval(model, tokenizer)

    if args.export_gguf:
        export_gguf(model, tokenizer)
