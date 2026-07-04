"""free watch — watchdog tabanlı proje monitörü.

Git değişikliklerini ve Python dosyası kaydetmelerini izler; değişiklik
algılandığında terminale bildirim basar. İsteğe bağlı olarak pytest'i
otomatik çalıştırır.
"""

import subprocess
import threading
import time
from pathlib import Path

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer
    _WATCHDOG_OK = True
except ImportError:
    _WATCHDOG_OK = False

from tools.file_ops import PROJECT_ROOT


def _run_git_status() -> str:
    try:
        out = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
            encoding="utf-8", timeout=5,
        )
        return out.stdout.strip() or "(temiz)"
    except Exception as exc:
        return f"git hata: {exc}"


def _run_pytest() -> str:
    try:
        out = subprocess.run(
            ["python", "-m", "pytest", "--tb=no", "-q"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
            encoding="utf-8", timeout=60,
        )
        lines = (out.stdout + out.stderr).strip().splitlines()
        return lines[-1] if lines else "(çıktı yok)"
    except Exception as exc:
        return f"pytest hata: {exc}"


class _ChangeHandler(FileSystemEventHandler):
    def __init__(self, run_tests: bool, console_print):
        self._run_tests = run_tests
        self._print = console_print
        self._debounce: dict[str, float] = {}
        self._lock = threading.Lock()

    def on_modified(self, event: "FileSystemEvent"):
        if event.is_directory:
            return
        src = str(event.src_path)
        # Sadece .py dosyaları + .gitignore dışı değişiklikler
        if not (src.endswith(".py") or src.endswith(".md") or src.endswith(".jsonl")):
            return
        # logs/ ve __pycache__/ değişikliklerini yoksay (platform-bağımsız)
        src_path = Path(src)
        skip_parts = {"logs", "__pycache__", ".git"}
        if any(part in skip_parts for part in src_path.parts):
            return
        now = time.time()
        with self._lock:
            last = self._debounce.get(src, 0)
            if now - last < 1.5:
                return
            self._debounce[src] = now

        rel = Path(src).relative_to(PROJECT_ROOT) if Path(src).is_relative_to(PROJECT_ROOT) else src
        self._print(f"\n[bold cyan]👁  Değişiklik:[/] {rel}")
        status = _run_git_status()
        self._print(f"[dim]git status: {status}[/]")
        if self._run_tests and src.endswith(".py"):
            self._print("[dim]pytest çalışıyor...[/]")
            result = _run_pytest()
            self._print(f"[dim]pytest: {result}[/]")


def watch(run_tests: bool = False, console_print=None) -> None:
    """Projeyi izlemeye başlar; Ctrl+C ile durur."""
    if not _WATCHDOG_OK:
        if console_print:
            console_print("[bold red]watchdog kurulu değil.[/] Kurmak için: pip install watchdog>=4.0")
        return

    if console_print is None:
        from rich.console import Console
        _con = Console()
        console_print = _con.print

    handler = _ChangeHandler(run_tests=run_tests, console_print=console_print)
    observer = Observer()
    observer.schedule(handler, str(PROJECT_ROOT), recursive=True)
    observer.start()

    test_label = " + pytest" if run_tests else ""
    console_print(f"[bold green]👁  free watch başladı[/][dim]{test_label} · {PROJECT_ROOT} · Çıkmak için Ctrl+C[/]")

    try:
        while observer.is_alive():
            observer.join(timeout=1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        console_print("[dim]free watch durduruldu.[/]")
