"""Ollama HTTP client, VRAM/RAM model discipline, and the agentic tool-call loop."""

import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler

import requests
import agents.config as config
import tools.network_guard  # noqa: F401 — Session.request/httpx.Client.send guard'ini kurar
from tools.audit_ops import append_event
from tools.perf_ops import log_turn

# Son tur token/s — shell toolbar'inin okudugu paylaşılan durum
last_tokens_per_sec: float = 0.0

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "free.log")
logger = logging.getLogger("free")
logger.setLevel(logging.INFO)
_log_handler = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_log_handler)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Donanım dogrulamasi (HF model kartlari) sonrasi gercekci registry:
# GLM-5.2 (753B), MiniMax-M3 (428B) ve Kimi-K2.7-Code (1T) 8GB VRAM/32GB RAM'e
# sigmiyor; coder icin Qwen2.5-Coder-7B, research icin Hermes3-8B, vision
# icin Qwen3-VL-8B ile degistirildi (hepsi Ollama'da tool-calling destekli).
MODEL_REGISTRY = {
    "qwen2.5-coder": {"pool": "vram_native", "max_ctx": 8192, "role": "coder"},
    "hermes3": {"pool": "vram_heavy", "max_ctx": 16384, "role": "research"},
    "qwen3-vl": {"pool": "vram_native", "max_ctx": 6144, "role": "vision"},
    "mistral-nemo": {"pool": "vram_heavy", "max_ctx": 16384, "role": "codebase"},
}

# Registry'de olmayan modeller (orn. main.py'deki guncel coder modeli) icin
# fallback context penceresi. Bu degerler donanim olcumunden kesin turetilmedi
# (KV-cache quantizasyonu/layer sayisi canli nvidia-smi olcumu gerektirir);
# baslangic noktasi olarak isaretlenmistir — OOM/yavaslama gorulurse dusur,
# context kirpilmasi hissi olursa yukselt (--ctx ile override edilebilir).
DEFAULT_NUM_CTX = 8192

# Yaklasik VRAM kullanim tahmini (GB) — CLAUDE.md model_registry runtime_profile
# notlarindan alindi, sadece `free status` icinde bilgilendirme amacli.
MODEL_VRAM_ESTIMATES_GB = {
    "qwen2.5-coder": 5.9,
    "hermes3": 5.0,
    "qwen3-vl": 6.5,
    "mistral-nemo": 7.1,
}

DEFAULT_CODER_MODEL = "qwen2.5-coder:7b"
DEFAULT_RESEARCH_MODEL = "hermes3:8b"
DEFAULT_VISION_MODEL = "qwen3-vl:8b"
# search_codebase (RAG) sentezinde hermes3/qwen2.5-coder denemelerinden daha iyi sonuc verdi
# (bkz. agents/research.py RESEARCH_SYSTEM_PROMPT) — bu yuzden codebase sorulari icin ayri rol.
DEFAULT_CODEBASE_MODEL = "mistral-nemo:latest"

MAX_TURNS_DEFAULT = 8


def _base_name(model_id: str) -> str:
    return model_id.split(":")[0]


def _num_ctx_for(model_id: str) -> int:
    """ctx_override > registry'deki max_ctx > DEFAULT_NUM_CTX sirasiyla cozumlenir."""
    if config.ctx_override:
        return config.ctx_override
    info = MODEL_REGISTRY.get(_base_name(model_id), {})
    return info.get("max_ctx") or DEFAULT_NUM_CTX


CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_context_usage(messages: list[dict], model_id: str) -> tuple[int, int, int]:
    """Mesaj gecmisinin token tahminini, num_ctx'i ve doluluk yuzdesini dondurur.

    Ollama'dan gercek prompt_eval_count geldiyse onu kullanir (daha dograru);
    yoksa char/4 tahminiyle devam eder.
    """
    if config.last_prompt_eval_count:
        estimated_tokens = config.last_prompt_eval_count
    else:
        char_count = sum(len(m.get("content") or "") for m in messages)
        estimated_tokens = char_count // CHARS_PER_TOKEN_ESTIMATE
    ctx = _num_ctx_for(model_id)
    pct = min(100, round(100 * estimated_tokens / ctx)) if ctx else 0
    return estimated_tokens, ctx, pct


class OllamaClient:
    def __init__(self, host: str | None = None):
        self.host = host or OLLAMA_HOST

    def list_loaded(self) -> list[dict]:
        resp = requests.get(f"{self.host}/api/ps", timeout=10)
        resp.raise_for_status()
        return resp.json().get("models", [])

    def list_models(self) -> list[dict]:
        """Lists all locally pulled models (not just ones currently loaded in memory)."""
        resp = requests.get(f"{self.host}/api/tags", timeout=10)
        resp.raise_for_status()
        return resp.json().get("models", [])

    def unload(self, model_id: str) -> None:
        requests.post(
            f"{self.host}/api/generate",
            json={"model": model_id, "prompt": "", "keep_alive": 0},
            timeout=30,
        )
        logger.info("unloaded model=%s", model_id)
        if config.audit_enabled:
            append_event("model_unload", {"model": model_id})

    def embed(self, model: str, text: str) -> list[float]:
        resp = requests.post(
            f"{self.host}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("embedding", [])

    def chat(self, model: str, messages: list[dict], tools: list[dict] | None = None,
              options: dict | None = None, format: str | None = None) -> dict:
        payload = {"model": model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
        if options:
            payload["options"] = options
        if format:
            payload["format"] = format
        resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()

    def chat_stream(self, model: str, messages: list[dict],
                    options: dict | None = None) -> dict:
        """Streams tokens to stdout in real-time, returns assembled message dict."""
        payload = {"model": model, "messages": messages, "stream": True}
        if options:
            payload["options"] = options
        resp = requests.post(f"{self.host}/api/chat", json=payload,
                             timeout=600, stream=True)
        resp.raise_for_status()

        full_content = ""
        final_message = {}
        final_chunk: dict = {}
        console.print(f"\n[bold cyan]🤖 {model}[/] [dim](streaming)[/]")
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                chunk = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            token = chunk.get("message", {}).get("content", "")
            if token:
                print(token, end="", flush=True)
                full_content += token
            if chunk.get("done"):
                final_message = chunk.get("message", {})
                final_chunk = chunk
                break
        print()  # newline after streaming
        final_message["content"] = full_content
        # Streaming done chunk'undaki perf alanlarini ana response'a tasiyoruz
        # boylece run_agent_loop'taki log_turn ayni kodla calisir
        return {
            "message": final_message,
            "eval_count": final_chunk.get("eval_count", 0),
            "eval_duration": final_chunk.get("eval_duration", 0),
            "done_reason": final_chunk.get("done_reason", ""),
        }


class ModelManager:
    """Enforces: only one model loaded at a time — 8GB VRAM is a hard ceiling
    regardless of pool (vram_native models conflict with each other too, e.g.
    coder vs. vision)."""

    def __init__(self, client: OllamaClient, registry: dict | None = None):
        self.client = client
        self.registry = registry or MODEL_REGISTRY

    def _info(self, model_id: str) -> dict:
        return self.registry.get(_base_name(model_id), {"pool": "vram_heavy", "max_ctx": None})

    def ensure_loaded(self, model_id: str) -> dict:
        info = self._info(model_id)
        target = _base_name(model_id)

        try:
            loaded = self.client.list_loaded()
        except requests.RequestException as exc:
            logger.warning("could not query /api/ps: %s", exc)
            return info

        for entry in loaded:
            name = entry.get("name") or entry.get("model") or ""
            base = _base_name(name)
            if not base or base == target:
                continue
            logger.info("unloading model=%s to free VRAM for %s", base, target)
            self.client.unload(name)

        if config.audit_enabled:
            append_event("model_load", {"model": target})

        return info


from rich.console import Console
console = Console()


def _find_first_balanced_json_object(text: str) -> tuple[int, str | None]:
    """Scans for the first balanced {...} object in text, ignoring braces inside
    string literals. Returns (start_index, json_str) or (-1, None) if none found.
    Needed because a naive find('{')/rfind('}') grabs everything between the FIRST
    '{' and the LAST '}' — if the model writes two JSON blocks (e.g. an example
    followed by the real call), that span is not valid JSON and the tool call is
    silently dropped."""
    search_from = text.find('{')
    while search_from != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(search_from, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return search_from, text[search_from:i + 1]
        search_from = text.find('{', search_from + 1)
    return -1, None


def run_agent_loop(
    client: OllamaClient,
    model: str,
    messages: list[dict],
    tools_schema: list[dict] | None = None,
    tool_executor: dict | None = None,
    max_turns: int = MAX_TURNS_DEFAULT,
    options: dict | None = None,
) -> str:
    """Drives the chat/tool-call loop until the model returns a final text answer."""
    tool_executor = tool_executor or {}
    tools_schema = tools_schema or []

    if config.no_network:
        blocked = {"whois_lookup", "web_search", "fetch_url"}
        tools_schema = [t for t in tools_schema if t.get("function", {}).get("name") not in blocked]
        tool_executor = {k: v for k, v in tool_executor.items() if k not in blocked}

    options = options or {
        "temperature": 0.3,
        "repeat_penalty": 1.2,
        "top_p": 0.9,
        # RTX 4070 (Ampere): tüm katmanlari VRAM'e pin'le, CPU offload gizli gecikme ekler.
        "num_gpu_layers": 999,
        # Flash Attention — Ampere destekliyor; aynı VRAM'de daha uzun context veya
        # daha hızlı inference. Ollama 0.1.29+ gerektirir.
        "flash_attn": True,
    }
    options.setdefault("num_ctx", _num_ctx_for(model))
    # Shallow copy: run_agent_loop kendi tur mesajlarini eklese de
    # caller'in orijinal listesini (orn. council turlarini) kirletmez.
    history = list(messages)
    # Tekrar eden tool çağrılarını tespit etmek için: (name, frozen_args) → kaç kez
    _tool_repeat: dict[tuple, int] = {}
    _MAX_TOOL_REPEAT = 2  # Aynı araç+arg ikilisi bu kadar tekrar ederse kes

    for turn in range(max_turns):
        t0 = time.time()
        try:
            if config.verbose:
                console.print(f"[dim]\u2500\u2500 TUR {turn+1}/{max_turns} | model={model} | mesaj_sayısı={len(history)} \u2500\u2500[/]")
                response = client.chat_stream(model, history, options=options)
            else:
                with console.status(f"[bold cyan]{model} düşünüyor...[/]"):
                    response = client.chat(model, history, options=options)
        except Exception as _http_exc:
            err = str(_http_exc)
            # 500: Ollama model crash'i (genellikle OOM). CLI çökmemeli.
            if "500" in err:
                console.print(
                    f"[bold red]❌ Ollama 500 hatası — model çökmüş olabilir (VRAM yetersiz?).[/]\n"
                    f"[dim]Öneri: /model ile farklı bir model seç veya /ctx ile context penceresini küçült.[/]"
                )
            else:
                console.print(f"[bold red]❌ Ollama bağlantı hatası: {err}[/]")
            logger.error("ollama_request_failed model=%s error=%s", model, err)
            return f"ERROR: Ollama isteği başarısız — {err}"
        
        elapsed = time.time() - t0

        eval_count = response.get("eval_count", 0)
        eval_duration_ns = response.get("eval_duration", 0)
        prompt_eval_count = response.get("prompt_eval_count", 0)
        if eval_count and eval_duration_ns:
            log_turn(model, eval_count, eval_duration_ns)
            global last_tokens_per_sec
            last_tokens_per_sec = round(eval_count / (eval_duration_ns / 1e9), 1)
        # Ollama'nin gercek prompt token sayisini context tahminine yansit
        if prompt_eval_count:
            config.last_prompt_eval_count = prompt_eval_count

        if response.get("done_reason") == "length":
            console.print(
                "[bold yellow]⚠️  Context penceresi doldu (done_reason=length) — "
                "cevap kesilebilir. /ctx ile num_ctx artırın veya /clear ile geçmişi sıfırlayın.[/]"
            )
            logger.warning("done_reason=length model=%s turn=%d", model, turn)

        message = response.get("message", {})
        
        tool_calls = message.get("tool_calls")
        content = message.get("content", "")
        
        # Fallback parser if the model leaks the tool call JSON into content
        malformed_tool_call = False
        if not tool_calls and content:
            start, json_str = _find_first_balanced_json_object(content)
            if json_str is not None:
                try:
                    parsed = json.loads(json_str)
                    if "name" in parsed and "arguments" in parsed:
                        tool_calls = [{"function": parsed}]
                        # Remove the JSON from content so we don't display it raw
                        content = content[:start].strip()
                        message["content"] = content
                    else:
                        malformed_tool_call = True
                except Exception as e:
                    logger.warning("Fallback JSON parse error: %s", e)
                    malformed_tool_call = True

        # Append assistant message to history
        history.append(message)

        _RETRY_MARKER = "[RETRY]"
        if malformed_tool_call:
            console.print("[dim]⚠️  Geçersiz JSON araç çağrısı, tekrar isteniyor...[/]")
            if config.audit_enabled:
                append_event("tool_call_retry", {"model": model, "raw_content": content})
            # Önceki retry mesajını geçmişten çıkar — üst üste birikmesini önle.
            history[:] = [m for m in history if _RETRY_MARKER not in m.get("content", "")]
            history.append({
                "role": "user",
                "content": (
                    f"{_RETRY_MARKER} Gecersiz JSON. SADECE su format: "
                    "{\"name\": \"<arac_adi>\", \"arguments\": {...}}"
                ),
            })
            continue

        if config.verbose and content:
            console.print(f"\n[dim]⏱️  Süre: {elapsed:.1f}s[/]")
            console.print(f"[dim]── RAW JSON / TOOL PARSER SONRASI ──\n{content}[/]")
        
        # Print any conversational text if we didn't already stream it
        if content and not config.verbose:
            from rich.markdown import Markdown
            console.print(f"\n[bold cyan]🤖 {model}[/] [dim]({elapsed:.1f}s)[/]")
            console.print(Markdown(content))

        # If no tools called, we are done — return content so callers
        # (council, single-shot CLI) can display or chain it if needed.
        if not tool_calls:
            return content

        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name")
            raw_args = fn.get("arguments") or {}
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError as exc:
                    logger.warning("tool_call arg parse failed name=%s error=%s", name, exc)
                    history.append({
                        "role": "user",
                        "content": (
                            f"[SİSTEM ARACI SONUCU - {name}]:\n"
                            f"ERROR: '{name}' aracinin argumanlari gecerli JSON degil, parse edilemedi ({exc}). "
                            "Lutfen araci SADECE gecerli bir JSON 'arguments' nesnesiyle tekrar cagir.\n\n"
                            "[ZORUNLU TALİMAT]: Cevabını tamamen TÜRKÇE yaz. Kazakça, İngilizce veya başka dil YASAK."
                        ),
                    })
                    continue
            else:
                args = raw_args

            # Tekrar döngüsü tespiti
            try:
                _repeat_key = (name, str(sorted(args.items()) if isinstance(args, dict) else args))
            except Exception:
                _repeat_key = (name, str(args))
            _tool_repeat[_repeat_key] = _tool_repeat.get(_repeat_key, 0) + 1
            if _tool_repeat[_repeat_key] > _MAX_TOOL_REPEAT:
                logger.warning("tool_loop_detected name=%s, aborting turn", name)
                history.append({
                    "role": "user",
                    "content": (
                        f"[SİSTEM UYARISI]: '{name}' aracini ayni argümanlarla {_tool_repeat[_repeat_key]} kez "
                        "çağırdın, sonuç değişmiyor. Farklı bir araç dene veya mevcut bilgiyle sonuca ulaş."
                    ),
                })
                break

            logger.info("tool_call name=%s args=%s", name, args)
            if config.audit_enabled:
                append_event("tool_call", {"name": name, "args": args})
            if name in ("write_file", "edit_file") and "path" in args:
                from tools.file_ops import WORKSPACE_ROOT
                import os as _os
                full_path = _os.path.join(WORKSPACE_ROOT, args["path"])
                console.print(f"[dim]⚙️  Çalıştırılıyor: {name}({args})...[/]")
                console.print(f"[dim]📄 Dosya: {full_path}[/]")
            else:
                console.print(f"[dim]⚙️  Çalıştırılıyor: {name}({args})...[/]")

            if name in ("write_file", "edit_file") and config.confirm_writes:
                from rich.prompt import Confirm
                if not Confirm.ask(f"[yellow]⚠️  '{name}' calistirilsin mi?[/]"):
                    result = "ERROR: kullanici onaylamadi, islem iptal edildi"
                    logger.info("tool_call rejected by user name=%s", name)
                    history.append({
                        "role": "user",
                        "content": (
                            f"[SİSTEM ARACI SONUCU - {name}]:\n{result}\n\n"
                            "[ZORUNLU TALİMAT]: Yukarıdaki araç sonucunu KULLAN. "
                            "Cevabını tamamen TÜRKÇE yaz. Kazakça, İngilizce veya başka dil YASAK."
                        ),
                    })
                    continue

            executor = tool_executor.get(name)
            if executor is None:
                available = ", ".join(sorted(tool_executor.keys())) or "(yok)"
                result = (
                    f"ERROR: '{name}' adli bir arac yok. Bu modda kullanilabilir araclar: {available}. "
                    "Gorevine uygun olani sec; eger istedigin yetenek listede yoksa farkli bir mod gerekir "
                    "(orn. kod yazma icin 'code', proje analizi icin 'codebase', web/arastirma icin 'research')."
                )
                logger.warning("unknown tool requested name=%s", name)
            else:
                try:
                    result = str(executor(**args))
                except Exception as exc:
                    # Argüman özetini güvenli şekilde oluştur: büyük değerleri kırp.
                    _MASK = {"password", "token", "secret", "key", "auth"}
                    safe_args = {
                        k: ("***" if any(m in k.lower() for m in _MASK)
                            else (str(v)[:80] + "..." if len(str(v)) > 80 else v))
                        for k, v in args.items()
                    }
                    result = f"ERROR [{name}({safe_args})]: {exc}"
                    logger.warning("tool_call failed name=%s args=%s error=%s", name, safe_args, exc)

            preview = result[:500] + "..." if len(result) > 500 else result
            logger.info("tool_result name=%s result=%s", name, preview)
            if config.verbose:
                console.print(f"[dim]── ARAÇ SONUCU [{name}] ──\n{preview}[/]")

            # Pass tool result as a user message since native tools are disabled
            # Late Prompt Injection: Türkçe zorunluluğu araç sonucunun tam altına eklenir
            # Böylece model cevap üretmeden hemen önce bu kuralı okur (context fade yok)
            history.append({
                "role": "user",
                "content": (
                    f"[SİSTEM ARACI SONUCU - {name}]:\n{result}\n\n"
                    "[ZORUNLU TALİMAT]: Yukarıdaki araç sonucunu KULLAN. "
                    "Cevabını tamamen TÜRKÇE yaz. Kazakça, İngilizce veya başka dil YASAK."
                ),
            })

    logger.warning("max_turns reached (%d) for model=%s", max_turns, model)
    return f"[free] tur limitine ulasildi (max_turns={max_turns}), islem tamamlanamadan kesildi."
