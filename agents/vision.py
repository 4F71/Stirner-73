"""VisionAgent — image understanding (screenshots, diagrams, UI mockups)."""

from agents.core import DEFAULT_VISION_MODEL, ModelManager, OllamaClient, run_agent_loop
from tools.file_ops import FILE_TOOLS_SCHEMA, TOOL_EXECUTOR, read_image_b64
from tools.rag_ops import RAG_TOOLS_SCHEMA, RAG_TOOL_EXECUTOR
from tools.grep_ops import GREP_TOOLS_SCHEMA, GREP_TOOL_EXECUTOR
from tools.memory_ops import MEMORY_TOOLS_SCHEMA, MEMORY_TOOL_EXECUTOR
from tools.system_ops import SYSTEM_TOOLS_SCHEMA, SYSTEM_TOOL_EXECUTOR
from tools.audit_ops import AUDIT_TOOLS_SCHEMA, AUDIT_TOOL_EXECUTOR
from tools.rollback_ops import ROLLBACK_TOOLS_SCHEMA, ROLLBACK_TOOL_EXECUTOR

VISION_TOOLS_SCHEMA = (
    FILE_TOOLS_SCHEMA + RAG_TOOLS_SCHEMA + GREP_TOOLS_SCHEMA + MEMORY_TOOLS_SCHEMA
    + SYSTEM_TOOLS_SCHEMA + AUDIT_TOOLS_SCHEMA + ROLLBACK_TOOLS_SCHEMA
)
VISION_TOOL_EXECUTOR = {
    **TOOL_EXECUTOR, **RAG_TOOL_EXECUTOR, **GREP_TOOL_EXECUTOR, **MEMORY_TOOL_EXECUTOR,
    **SYSTEM_TOOL_EXECUTOR, **AUDIT_TOOL_EXECUTOR, **ROLLBACK_TOOL_EXECUTOR,
}


def _tool_names(schema: list) -> str:
    return ", ".join(f"'{t['function']['name']}'" for t in schema)


VISION_SYSTEM_PROMPT = (
    "Sen 'free vision' ajanisin. Sana verilen goruntuyu (ekran goruntusu, diyagram, "
    "UI mockup) analiz edip kullanicinin sorusunu yanitlarsin. Gerekirse bulgularini "
    "write_file ile workspace/ icine bir rapor olarak kaydedebilirsin, ama varsayilan "
    "olarak dogrudan metin yaniti ver. Goruntudeki bir UI/diyagrami kod tabaniyla "
    "karsilastirman gerekirse search_codebase/grep_codebase kullan.\n\n"
    f"Mevcut araclar: {_tool_names(VISION_TOOLS_SCHEMA)}.\n"
    "Bir arac cagirman gerekirse SADECE asagidaki JSON formatini ciktinda bulundur, BASKA HICBIR SEY YAZMA:\n"
    "{\n"
    "  \"name\": \"write_file\",\n"
    "  \"arguments\": {\"path\": \"rapor.md\", \"content\": \"...\"}\n"
    "}\n"
    "Arac calistiktan sonra sonucunu goreceksin.\n\n"
    "YANITLARINI KESINLIKLE VE SADECE TURKCE DILINDE VERECEKSIN."
)


class VisionAgent:
    def __init__(self, model: str = DEFAULT_VISION_MODEL, client: OllamaClient | None = None):
        self.model = model
        self.client = client or OllamaClient()
        self.manager = ModelManager(self.client)

    def run(self, prompt: str, image_path: str) -> str:
        self.manager.ensure_loaded(self.model)
        image_b64 = read_image_b64(image_path)
        messages = [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt, "images": [image_b64]},
        ]
        return run_agent_loop(
            self.client,
            self.model,
            messages,
            tools_schema=VISION_TOOLS_SCHEMA,
            tool_executor=VISION_TOOL_EXECUTOR,
        )
