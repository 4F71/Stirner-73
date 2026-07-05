# Changelog

## [v0.2] — 2026-07-05

### Genel

v0.2, v0.1'den bu yana eklenen tüm özellikleri ve düzeltmeleri kapsar.
Ana tema: **kararlılık + ajan yetenek genişlemesi + yerel router**.

---

### Model Stack

| Model | Konum | Kullanım |
|-------|-------|---------|
| `qwen2.5-coder-7b-abliterated` (Q6_K) | GPU VRAM | Kod üretimi, dosya düzenleme |
| `hermes3:8b` | RAM + GPU offload | Araştırma, analiz |
| `qwen3-vl:8b` | GPU VRAM | Görüntü / ekran analizi |
| `free-router` (`Qwen2.5-1.5B` LoRA) | CPU/RAM | Intent sınıflandırma (code/research/codebase) |

Router yoksa `qwen2.5:1.5b` (Ollama) devreye girer.

---

### Yeni Özellikler

#### Ajan & Çekirdek
- **Sokratik öğrenme modu** — `free explain --socratic`: ajan cevabı vermez, soruyla yönlendirir
- **Council tartışma modu** — birden fazla ajan aynı problemi farklı rollerden tartışır
- **Vision agent araç seti genişletildi** — web + dosya araçları
- **Reviewer agent** — `grep_codebase`, `run_pytest`, `git_diff` araçları eklendi
- **Debug modu** — `free debug` ayrı sistem promptuyla çalışıyor
- **Kalıcı proje hafızası** — `remember`/`recall` araçları, coder/research/reviewer'a bağlandı
- **Oturum çıkışında otomatik özet** — shell kapanırken hafıza güncelleniyor

#### Router & Yönlendirme
- `code` / `research` / `codebase` üç-sınıf intent router
- **LoRA fine-tune altyapısı** (`training/train_router_lora.py`) — `Qwen2.5-1.5B` tabanlı
- Keyword ön-filtre ile yanlış eşleşme azaltıldı
- `free-router` Ollama'da yoksa `qwen2.5:1.5b`'ye otomatik fallback
- `/correct` komutu ile hatalı yönlendirme anında düzeltilebilir, fine-tune dataseti oluşturuluyor

#### RAG & Arama
- **Yerel RAG motoru** — ChromaDB tabanlı, `free index` ile güncelleniyor
- `search_codebase` aracı coder/research/reviewer'a bağlandı
- `grep_codebase` — sembolik grep aracı tüm ajanlarda
- Shell açılışında RAG index 24 saatten eskiyse uyarı

#### CLI & Shell
- **`free watch`** — dosya değişikliklerini izleyip otomatik yeniden çalıştırır
- **`compare-runs`** — `training/results/` altındaki eval dosyalarını karşılaştırır
- **`free doctor`** — VRAM/RAM durumu, model listesi, bağımlılık kontrolü
- **`free scaffold`** — proje şablonu üretici (torch-experiment, python-package, ...)
- **`/retry`** — son komutu yeniden çalıştırır
- **`/pop`** — son mesajı geçmişten çıkarır
- **`/stats`** — performans ve `feedback` alt-komutu
- **`/exit --quiz`** — çıkışta öğrenme sorusu
- **`/ctx`** — context penceresi doluluk yüzdesi
- **`/models`**, **`/load`**, **`/status`**, **`/log`**, **`/doctor`** shell komutları
- Tab tuşu geçmiş önerisini kabul ediyor
- Context budget göstergesi toolbar'da

#### Araçlar
- `run_python` — sandbox Python çalıştırma, `FREE_PYTHON_TIMEOUT` ile ayarlanabilir
- `edit_file` + otomatik yedekleme → `rollback` aracı ile geri alma
- `git_add` / `git_commit` / `git_push` — ajanlara git yazma erişimi
- `git_status` / `git_diff` / `git_log` — salt-okunur git araçları
- `run_ruff` / `run_mypy` — statik analiz, workspace varsayılan
- `run_pytest` — test çalıştırma
- `fetch_url`, `web_search`, `whois` — web araçları
- `audit_log`, `rollback_file` — denetim izi + geri alma
- `system_info` — VRAM/RAM/CPU gerçek ölçümü
- Air-gapped mod (`--no-network`) ve kriptografik denetim izi (hash-chain)

---

### Düzeltmeler

| Alan | Sorun |
|------|-------|
| `agents/core.py` | Araç şeması (`tools_schema`) Ollama API'ye hiç gönderilmiyordu |
| `agents/core.py` | `chat_stream()` `prompt_eval_count` döndürmüyordu; `/ctx` yanlış % gösteriyordu |
| `agents/core.py` | `unload()` HTTP hatasını susturuyordu, VRAM durumu yanlış hesaplanıyordu |
| `agents/core.py` | Fallback JSON parser false-positive araç çağrısı tetikliyordu |
| `agents/core.py` | `[RETRY]` marker araç sonuçlarını silebiliyordu |
| `tools/scaffold_ops.py` | `torch-experiment` şablonunda `import os` yanlış yerdeydı → `NameError` |
| `tools/watch_ops.py` | Path separator OS uyumsuzluğu (`__pycache__` Windows'ta filtrelenmiyordu) |
| `tools/compare_ops.py` | `_RE_MODEL` dead code — model adı tablodan eksikti |
| `main.py` | `/retry` ilk prompttan önce `NameError` fırlatıyordu |
| `main.py` | `/correct` `wrong_intent` boş kaydediyordu |
| `main.py` | `/stats feedback` erişilemiyordu |
| `requirements.txt` | `lxml` eksikti → `fetch_url` her zaman çöküyordu |
| `requirements.txt` | `duckduckgo-search` (ölü paket) kaldırıldı |
| `pyproject.toml` | `ddgs`, `psutil`, `watchdog`, `lxml` eksik bağımlılıklar eklendi |
| `agents/coder.py` | KURAL 4 çelişkili sistem prompt netleştirildi |
| `agents/reviewer.py` | `REVIEWER_TOOLS_SCHEMA` prompt çağrısından önce tanımlanmıyor hatasıydı |
| router | Keyword yanlış eşleşmeler düzeltildi |
| shell | Model seçimi, çift cevap, `/remember` kalıcılık sorunları |
| shell | `codebase` modu git araçlarına erişemiyordu |

---

### Git Geçmişi Düzeltmesi

Tüm 67 commit mesajının başındaki UTF-8 BOM (`﻿`) `git filter-branch` ile temizlendi (2026-07-05).

---

## [v0.1] — 2026-06-19

İlk çalışan sürüm. Temel multi-agent CLI iskeleti kuruldu.

### Model Stack

| Model | Konum | Kullanım |
|-------|-------|---------|
| `qwen2.5-coder-7b-abliterated` (Q6_K) | GPU VRAM | Kod üretimi |
| `hermes3:8b` | RAM + GPU offload | Araştırma |
| `qwen3-vl:8b` | GPU VRAM | Görüntü analizi |

### Eklenenler

#### CLI
- `free code` — kodlama ajani
- `free research` — arastirma ajani
- `free review` — kod inceleme ajani
- Click tabanlı CLI giriş noktası (`main.py`)
- `pyproject.toml` + `requirements.txt` bağımlılık tanımları

#### Çekirdek (`agents/core.py`)
- `OllamaClient` — `localhost:11434` HTTP client, `requests` ile doğrudan çağrı
- `ModelManager` — tek-model VRAM kuralı, yükleme/boşaltma yönetimi
- `run_agent_loop` — araç çağrısı + yanıt döngüsü

#### Ajanlar
- `CoderAgent` (`agents/coder.py`) — kod üretimi ve dosya düzenleme odaklı sistem promptu
- `ResearchAgent` (`agents/research.py`) — araştırma ve analiz odaklı sistem promptu
- `ReviewerAgent` (`agents/reviewer.py`) — statik analiz araçlarıyla kod incelemesi
- `VisionAgent` (`agents/vision.py`) — görüntü + metin girişi, `qwen3-vl:8b` tabanlı

#### Araçlar (`tools/`)
- `file_ops.py` — `read_file`, `write_file`, `list_workspace` (workspace sandbox)
- `static_analysis.py` — `run_ruff`, `run_mypy` (workspace sandbox)
