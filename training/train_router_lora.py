"""
Router LoRA Fine-Tune — Qwen2.5-1.5B-Instruct
3-sınıf Türkçe intent classifier: code | codebase | research

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
DATASET_FILE = Path(__file__).parent / "dataset.jsonl"

LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
MAX_SEQ_LEN  = 256

TRAIN_EPOCHS = 5          # 30 örnekle 5 epoch yeterli; overfitting OK (kapalı domain)
BATCH_SIZE   = 4
GRAD_ACCUM   = 2
LR           = 3e-4

# ---------------------------------------------------------------------------
# Unsloth yükleme — CPU fallback ile kompatibil
# ---------------------------------------------------------------------------
try:
    from unsloth import FastLanguageModel
    _USE_UNSLOTH = True
except (ImportError, NotImplementedError, Exception) as _e:
    print(f"[WARN] unsloth yüklenemedi ({type(_e).__name__}: {_e})")
    print("[WARN] Standart transformers kullanılıyor (daha yavaş).")
    _USE_UNSLOTH = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling


def load_model_and_tokenizer():
    if _USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LEN,
            dtype=None,          # auto-detect
            load_in_4bit=True,   # 8GB VRAM'e sığdır
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        from peft import LoraConfig, get_peft_model, TaskType
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


def export_gguf(model, tokenizer):
    """GGUF export: unsloth varsa built-in, yoksa adapter merge + llama.cpp convert."""
    if _USE_UNSLOTH:
        print("[INFO] GGUF export (Q4_K_M) başlıyor...")
        model.save_pretrained_gguf(
            str(GGUF_PATH.with_suffix("")),
            tokenizer,
            quantization_method="q4_k_m",
        )
        print(f"[OK] GGUF: {GGUF_PATH}")
        return

    # Unsloth yok — adapter'ı base model ile merge et, merged/ olarak kaydet.
    print("[INFO] Adapter merge ediliyor (unsloth yok, llama.cpp ile export edilecek)...")
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


def run_quick_eval(model, tokenizer):
    """Training bittikten sonra 3 örnek üzerinde hızlı doğruluk kontrolü."""
    if _USE_UNSLOTH:
        FastLanguageModel.for_inference(model)

    import torch
    probes = [
        ("bu fonksiyonu yaz",            "code"),
        ("run_agent_loop nerede",         "codebase"),
        ("transformer nedir",             "research"),
    ]
    correct = 0
    for prompt, expected in probes:
        messages = [
            {"role": "system", "content":
             "Sen bir intent sınıflandırıcısın. SADECE JSON döndür: "
             "{\"intent\": \"code\"} | {\"intent\": \"codebase\"} | {\"intent\": \"research\"}"},
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
    print(f"\n  Quick-eval: {correct}/{len(probes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-gguf", action="store_true", help="Eğitim bittikten sonra Q4_K_M GGUF export yap")
    parser.add_argument("--eval-only",   action="store_true", help="Eğitim yapma, sadece kaydedilmiş adapter'ı değerlendir")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()

    if args.eval_only:
        print("[INFO] Eval-only modu — training atlandı")
    else:
        dataset = load_dataset(tokenizer)
        train(model, tokenizer, dataset)

    run_quick_eval(model, tokenizer)

    if args.export_gguf:
        export_gguf(model, tokenizer)
