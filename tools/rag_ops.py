"""Codebase RAG (semantic search) — local embeddings via Ollama + a local Chroma vector store.

LangChain is used here only for chunking (CLAUDE.md strict_rules #2 reserves LangChain
for RAG). Embeddings themselves still go through OllamaClient.embed, which calls the
local Ollama API directly via requests, same as every other model call in this project.
"""

import logging
import os

import chromadb
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from agents.core import OllamaClient
from tools.file_ops import PROJECT_ROOT

# chromadb 0.5.3'teki posthog surum uyumsuzlugu, anonymized_telemetry=False olsa da
# her capture() cagrisinda "capture() takes 1 positional argument but 3 were given"
# hatasini loglar (zararsiz ama gurultulu) — bu logger'i sustur.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

EMBED_MODEL = os.environ.get("FREE_EMBED_MODEL", "nomic-embed-text:latest")
INDEX_DIR = os.path.realpath(os.path.join(PROJECT_ROOT, ".rag_index"))
COLLECTION_NAME = "free_codebase"
_CHROMA_SETTINGS = chromadb.Settings(anonymized_telemetry=False)

# private/ haric tutuluyor: commit planlari, ai-log.md gibi kisisel/gizli icerik indexlenmemeli.
EXCLUDE_DIRS = {
    "private", "workspace", ".git", "venv", ".venv", "__pycache__",
    "logs", ".rag_index", "node_modules", ".pytest_cache",
}
INCLUDE_EXTENSIONS = {".py", ".md", ".toml"}

_client = OllamaClient()


def _iter_source_files():
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for name in files:
            if os.path.splitext(name)[1] in INCLUDE_EXTENSIONS:
                full = os.path.join(root, name)
                yield os.path.relpath(full, PROJECT_ROOT)


def _chunk_file(rel_path: str, text: str) -> list[str]:
    if rel_path.endswith(".py"):
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=800, chunk_overlap=100
        )
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_text(text)


def index_codebase(paths: str = "") -> str:
    """(Re)indexes the codebase — or a comma-separated subset of relative paths — into the local vector store."""
    target_paths = [p.strip() for p in paths.split(",") if p.strip()] if paths else None
    files = target_paths if target_paths else list(_iter_source_files())

    # Embed'leri koleksiyona yazmadan önce topla — hata olursa index bozuk kalmaz
    ids, documents, embeddings, metadatas = [], [], [], []
    indexed_files = 0

    for rel_path in files:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.isfile(full_path):
            continue
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        if not text.strip():
            continue

        for i, chunk in enumerate(_chunk_file(rel_path, text)):
            try:
                emb = _client.embed(EMBED_MODEL, chunk)
            except Exception as exc:
                return (
                    f"ERROR: embed modeli çalışmadı ({exc}). "
                    f"'ollama pull {EMBED_MODEL}' ile modeli çektiğinden emin ol. "
                    "Index güncellenmedi."
                )
            ids.append(f"{rel_path}::{i}")
            documents.append(chunk)
            embeddings.append(emb)
            metadatas.append({"path": rel_path, "chunk": i})
        indexed_files += 1

    # Embed başarılı → artık koleksiyonu sil ve yeniden oluştur
    chroma = chromadb.PersistentClient(path=INDEX_DIR, settings=_CHROMA_SETTINGS)
    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma.create_collection(COLLECTION_NAME)

    if documents:
        collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    return f"{indexed_files} dosya, {len(documents)} parca indexlendi -> {INDEX_DIR}"


def search_codebase(query: str, k: int = 5) -> str:
    """Semantically searches the indexed codebase and returns the top-k matching chunks."""
    if not os.path.isdir(INDEX_DIR):
        return "ERROR: index bulunamadi, once 'free index' calistirilmali"

    chroma = chromadb.PersistentClient(path=INDEX_DIR, settings=_CHROMA_SETTINGS)
    try:
        collection = chroma.get_collection(COLLECTION_NAME)
    except Exception:
        return "ERROR: index bulunamadi, once 'free index' calistirilmali"

    query_embedding = _client.embed(EMBED_MODEL, query)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    if not docs:
        return "Eslesme bulunamadi."

    parts = [f"### {meta.get('path', '?')}\n{doc}" for doc, meta in zip(docs, metas)]
    return "\n\n".join(parts)


RAG_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_codebase",
            "description": (
                "Semantically search the indexed codebase for relevant code/doc chunks. "
                "Use this instead of guessing file paths when you need to find where something "
                "is implemented or gather context spread across multiple files. Requires 'free index' "
                "to have been run at least once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language or code search query.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

RAG_TOOL_EXECUTOR = {
    "search_codebase": search_codebase,
}
