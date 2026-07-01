"""ReviewerAgent — static-analysis-backed code review (reuses the coder model)."""

from agents.core import DEFAULT_CODER_MODEL, ModelManager, OllamaClient, run_agent_loop
from tools.file_ops import FILE_TOOLS_SCHEMA, TOOL_EXECUTOR
from tools.static_analysis import STATIC_ANALYSIS_EXECUTOR, STATIC_ANALYSIS_TOOLS_SCHEMA
from tools.exec_ops import EXEC_TOOLS_SCHEMA, EXEC_TOOL_EXECUTOR
from tools.git_ops import GIT_TOOLS_SCHEMA, GIT_TOOL_EXECUTOR
from tools.rag_ops import RAG_TOOLS_SCHEMA, RAG_TOOL_EXECUTOR
from tools.memory_ops import MEMORY_TOOLS_SCHEMA, MEMORY_TOOL_EXECUTOR
from tools.system_ops import SYSTEM_TOOLS_SCHEMA, SYSTEM_TOOL_EXECUTOR
from tools.audit_ops import AUDIT_TOOLS_SCHEMA, AUDIT_TOOL_EXECUTOR
from tools.rollback_ops import ROLLBACK_TOOLS_SCHEMA, ROLLBACK_TOOL_EXECUTOR
from tools.grep_ops import GREP_TOOLS_SCHEMA, GREP_TOOL_EXECUTOR

def _tool_names(schema: list) -> str:
    return ", ".join(f"'{t['function']['name']}'" for t in schema)


def _build_reviewer_prompt() -> str:
    tool_list = _tool_names(REVIEWER_TOOLS_SCHEMA)
    return (
        "Sen 'free review' ajanisin. Kod kalitesi denetlemesi yaparsin. Once list_workspace/"
        "read_file ile ilgili dosyalari oku (gerekirse benzer/ilgili kod parcalarini bulmak icin "
        "search_codebase ile anlamsal arama yap), git_status/git_diff/git_log ile son degisiklikleri gor, sonra "
        "run_ruff ve run_mypy araclarini calistirip ciktilarini yorumla. Ilgili testler varsa run_pytest ile "
        "calistirip gecip gecmediklerini dogrula. Supheli bir davranisi "
        "run_python ile calistirip dogrula. Bulgularini onem sirasina gore (hata > uyari > stil) "
        "maddeler halinde sun; somut duzeltme onerileri ekle. write_file/edit_file'i sadece "
        "kullanicinin acikca istedigi duzeltmeleri uygularken kullan (kucuk degisiklik icin "
        "edit_file'i tercih et). Onemli bulgulari remember_decision ile hafizaya kaydet; "
        "onceki inceleme notlari icin recall_decisions kullan.\n\n"
        f"Mevcut araclar: {tool_list}.\n"
        "Bir arac cagirmak icin SADECE asagidaki JSON formatini ciktinda bulundur, BASKA HICBIR SEY YAZMA:\n"
        "{\n"
        "  \"name\": \"run_ruff\",\n"
        "  \"arguments\": {\"path\": \".\"}\n"
        "}\n"
        "Arac calistiktan sonra sonucunu goreceksin.\n\n"
        "YANITLARINI KESINLIKLE VE SADECE TURKCE DILINDE VERECEKSIN."
    )


REVIEWER_SYSTEM_PROMPT = _build_reviewer_prompt()

REVIEWER_TOOLS_SCHEMA = (
    FILE_TOOLS_SCHEMA + STATIC_ANALYSIS_TOOLS_SCHEMA + EXEC_TOOLS_SCHEMA + GIT_TOOLS_SCHEMA
    + RAG_TOOLS_SCHEMA + GREP_TOOLS_SCHEMA + MEMORY_TOOLS_SCHEMA + SYSTEM_TOOLS_SCHEMA
    + AUDIT_TOOLS_SCHEMA + ROLLBACK_TOOLS_SCHEMA
)
REVIEWER_TOOL_EXECUTOR = {
    **TOOL_EXECUTOR, **STATIC_ANALYSIS_EXECUTOR, **EXEC_TOOL_EXECUTOR, **GIT_TOOL_EXECUTOR,
    **RAG_TOOL_EXECUTOR, **GREP_TOOL_EXECUTOR, **MEMORY_TOOL_EXECUTOR, **SYSTEM_TOOL_EXECUTOR,
    **AUDIT_TOOL_EXECUTOR, **ROLLBACK_TOOL_EXECUTOR,
}


class ReviewerAgent:
    def __init__(self, model: str = DEFAULT_CODER_MODEL, client: OllamaClient | None = None):
        self.model = model
        self.client = client or OllamaClient()
        self.manager = ModelManager(self.client)

    def run(self, prompt: str) -> str:
        self.manager.ensure_loaded(self.model)
        messages = [
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return run_agent_loop(
            self.client,
            self.model,
            messages,
            tools_schema=REVIEWER_TOOLS_SCHEMA,
            tool_executor=REVIEWER_TOOL_EXECUTOR,
        )
