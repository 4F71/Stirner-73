# Router LoRA Fine-Tune

`free` CLI'ının intent router'ı için Qwen2.5-1.5B-Instruct üzerinde LoRA fine-tune altyapısı.

## Amaç

Kullanıcı promptlarını 2 sınıfa ayır:
- `code` — bir şey **yap** (yaz, düzelt, ekle, refactor et)
- `research` — bir şey **anla** (nedir, nasıl, neden, farkı ne)

`codebase` niyeti keyword tabanlı `_keyword_route()` ile yakalanır, ML'e bırakılmaz.

## Klasör Yapısı

```
training/
├── data/
│   └── cursor_dataset.jsonl     # 300 örnek (Sonnet + Kimi K2.5 + Opus 4.8)
├── router-lora/                 # Eğitilmiş LoRA adapter
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── checkpoint-N/            # Her epoch sonunda kaydedilen checkpoint
│   └── tokenizer*
├── router-merged/               # Base + adapter merge edilmiş tam model
├── results/
│   ├── v1_3class_30ex/          # İlk eğitim (3 sınıf, 30 örnek, CPU)
│   │   └── eval_*.txt
│   └── v2_2class_300ex/         # Güncel eğitim (2 sınıf, 300 örnek, RTX 4070)
│       ├── assets/
│       │   └── loss_curve_*.png # Training loss eğrisi görseli
│       └── eval_*.txt           # Eval raporu
├── train_router_lora.py         # Eğitim scripti
├── eval_router.py               # Değerlendirme scripti (confusion matrix)
├── Modelfile.router             # Ollama model tanımı
└── dataset.jsonl                # Eski 3-sınıflı dataset (kullanılmıyor)
```

## Dataset

`data/cursor_dataset.jsonl` — 300 örnek, 3 model tarafından üretildi:

| Model | Satır | Odak |
|---|---|---|
| Claude Sonnet 4.6 | 1–100 | Standart teknik sorular, açık imperatif komutlar |
| Kimi K2.5 | 101–200 | Kısa/ham promptlar, debugging, konfigürasyon |
| Claude Opus 4.8 | 201–300 | Edge case'ler, pasif/implicit niyet, sınır durumlar |

Dağılım: 150 `code` / 150 `research` — dengeli.

## Gereksinimler

```bash
pip install transformers torch peft accelerate matplotlib datasets
# CUDA için:
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Not:** `unsloth` Windows'ta şu an çalışmıyor (triton uyumsuzluğu). Eğitim `transformers.Trainer` ile yapılır.

## Eğitim

```bash
python training/train_router_lora.py
```

Parametreler (`train_router_lora.py` içinde):

| Parametre | Değer | Açıklama |
|---|---|---|
| `LORA_R` | 8 | LoRA rank |
| `LORA_ALPHA` | 16 | LoRA scaling |
| `TRAIN_EPOCHS` | 15 | Epoch sayısı |
| `BATCH_SIZE` | 4 | Batch boyutu |
| `LR` | 3e-4 | Learning rate |
| `MAX_SEQ_LEN` | 256 | Maksimum token uzunluğu |

Çıktılar:
- Adapter: `training/router-lora/`
- Loss eğrisi: `training/results/loss_curve_*.png`
- Checkpoint'ler: `training/router-lora/checkpoint-N/`

## Değerlendirme

```bash
# Merged model ile:
python training/eval_router.py

# Belirli checkpoint ile:
python training/eval_router.py --model training/router-lora/checkpoint-20

# Ollama modeli varsa:
python training/eval_router.py --ollama free-router
```

Çıktı: `training/results/eval_<model>_<tarih>.txt`  
İçerik: Accuracy, confusion matrix, per-class accuracy, yanlış tahminler.

## Ollama'ya Alma

Eğitim sonrası merged model → GGUF → Ollama:

```bash
# 1. llama.cpp kur (bir kez)
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt

# 2. GGUF'a dönüştür
python llama.cpp/convert_hf_to_gguf.py training/router-merged/ \
    --outfile training/router-q4.gguf --outtype q4_k_m

# 3. Ollama modeli oluştur
ollama create free-router -f training/Modelfile.router

# 4. Test
ollama run free-router "bu fonksiyonu yaz"
```

## Sonuçlar

### v1 — 3-sınıflı, 30 örnek, 5 epoch (CPU)
- Toplam accuracy: %46.7
- `codebase` sınıfı: %0 — öğrenemedi
- Karar: `codebase` ML'den çıkarıldı, 2-sınıfa indirildi

### v2 — 2-sınıflı, 300 örnek, 15 epoch (RTX 4070)
- Eğitim devam ediyor...
