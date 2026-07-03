"""CoderAgent — code completion and debug/reverse-engineering modes."""

from agents.core import DEFAULT_CODER_MODEL, ModelManager, OllamaClient, run_agent_loop
from tools.file_ops import FILE_TOOLS_SCHEMA, TOOL_EXECUTOR
from tools.exec_ops import EXEC_TOOLS_SCHEMA, EXEC_TOOL_EXECUTOR
from tools.git_ops import GIT_TOOLS_SCHEMA, GIT_TOOL_EXECUTOR
from tools.rag_ops import RAG_TOOLS_SCHEMA, RAG_TOOL_EXECUTOR
from tools.memory_ops import MEMORY_TOOLS_SCHEMA, MEMORY_TOOL_EXECUTOR
from tools.grep_ops import GREP_TOOLS_SCHEMA, GREP_TOOL_EXECUTOR
from tools.system_ops import SYSTEM_TOOLS_SCHEMA, SYSTEM_TOOL_EXECUTOR
from tools.audit_ops import AUDIT_TOOLS_SCHEMA, AUDIT_TOOL_EXECUTOR
from tools.rollback_ops import ROLLBACK_TOOLS_SCHEMA, ROLLBACK_TOOL_EXECUTOR

CODER_TOOLS_SCHEMA = (
    FILE_TOOLS_SCHEMA + EXEC_TOOLS_SCHEMA + GIT_TOOLS_SCHEMA + RAG_TOOLS_SCHEMA
    + MEMORY_TOOLS_SCHEMA + GREP_TOOLS_SCHEMA + SYSTEM_TOOLS_SCHEMA
    + AUDIT_TOOLS_SCHEMA + ROLLBACK_TOOLS_SCHEMA
)
CODER_TOOL_EXECUTOR = {
    **TOOL_EXECUTOR, **EXEC_TOOL_EXECUTOR, **GIT_TOOL_EXECUTOR, **RAG_TOOL_EXECUTOR,
    **MEMORY_TOOL_EXECUTOR, **GREP_TOOL_EXECUTOR, **SYSTEM_TOOL_EXECUTOR,
    **AUDIT_TOOL_EXECUTOR, **ROLLBACK_TOOL_EXECUTOR,
}


def _tool_names(schema: list) -> str:
    """Schema listesinden araç isimlerini virgülle birleştirilmiş string olarak döndürür."""
    return ", ".join(f"'{t['function']['name']}'" for t in schema)


CODE_SYSTEM_PROMPT = (
    "Sen 'free' otonom kodlama ajanısın. Sadece konuşmakla kalmaz, EYLEM yaparsın!\n\n"
    "KURAL 1: Araç çağırmadan ÖNCE HİÇBİR ŞEY YAZMA. İlk çıktın mutlaka bir JSON araç çağrısı olmalı.\n"
    "KURAL 2: ASLA 'Ben bir yapay zekaım, dosyalara erişemem' gibi bahaneler üretme — "
    "sana araçlar verildi, kullan.\n"
    "KURAL 3: Bir şeyin nerede uygulandığını bilmiyorsan önce search_codebase, sonra read_file.\n"
    "KURAL 4: Küçük değişiklik → edit_file. Tam yeniden yazım → write_file. "
    "WORKSPACE dışındaki proje dosyaları için write_file/edit_file 'agents/', 'tools/', 'main.py' gibi "
    "YOL GİRİLMESİ GEREKTİREN ARACLAR DEĞİLDİR; bunlar workspace/ altına yazar. "
    "Proje dosyasını düzenlemek için read_file + edit_file kullan, path='agents/coder.py' gibi göreceli yol ver.\n"
    "KURAL 5: Git işlemleri için git_status → git_add → git_commit → git_push araçlarını sırayla kullan. "
    "write_file ile commit ATILMAZ.\n"
    "KURAL 6: Araç çalıştıktan sonra sonucu aynen kullan, uydurma.\n\n"
    f"Mevcut araçlar: {_tool_names(CODER_TOOLS_SCHEMA)}.\n\n"
    "Araç çağırma formatı (SADECE BU JSON, başka hiçbir şey yazma):\n"
    "{\n"
    "  \"name\": \"git_status\",\n"
    "  \"arguments\": {}\n"
    "}\n\n"
    "YANITLARINI KESİNLİKLE VE SADECE TÜRKÇE DİLİNDE VERECEKSİN."
)

DEBUG_SYSTEM_PROMPT = (
    "Sen 'free debug' ajanisin.\n\n"
    "KURAL 1: Arac cagirmadan ONCE HICBIR SEY YAZMA. Ilk ciktin mutlaka bir JSON arac cagrisi olmali.\n"
    "KURAL 2: Oncelikleri: git_status/git_diff ile son degisiklikleri gör, read_file ile hatali dosyayi oku, "
    "run_python ile hatayi dogrula, edit_file ile duzelt.\n"
    "KURAL 3: Arac sonucunu aynen kullan, uydurma.\n\n"
    f"Mevcut araclar: {_tool_names(CODER_TOOLS_SCHEMA)}.\n"
    "Arac cagirma formati (SADECE BU JSON, baska hicbir sey yazma):\n"
    "{\n"
    "  \"name\": \"git_status\",\n"
    "  \"arguments\": {}\n"
    "}\n\n"
    "YANITLARINI KESINLIKLE VE SADECE TURKCE DILINDE VERECEKSIN."
)


# Sokratik mod kasitli olarak salt-okunur: write_file/edit_file/run_python/web araclari
# yok, kullanici kodu kendi cikarimiyla incelesin diye sadece okuma+arama araclari verilir.
_SOCRATIC_TOOL_NAMES = {"read_file", "list_workspace", "grep_codebase", "search_codebase"}
SOCRATIC_TOOLS_SCHEMA = [
    t for t in (FILE_TOOLS_SCHEMA + RAG_TOOLS_SCHEMA + GREP_TOOLS_SCHEMA)
    if t["function"]["name"] in _SOCRATIC_TOOL_NAMES
]
SOCRATIC_TOOL_EXECUTOR = {
    name: fn for name, fn in {**TOOL_EXECUTOR, **RAG_TOOL_EXECUTOR, **GREP_TOOL_EXECUTOR}.items()
    if name in _SOCRATIC_TOOL_NAMES
}

SOCRATIC_SYSTEM_PROMPT = (
    "Sen 'free explain --socratic' ajanisin. Kullanici kodu EZBERLEMEK degil, "
    "PARCALAYARAK kendi cikarimiyla OGRENMEK istiyor. Bu yuzden cevabi asla direkt "
    "vermezsin.\n\n"
    "YONTEM:\n"
    "1. Once read_file/list_workspace/grep_codebase/search_codebase ile istenen kodu oku ve analiz et "
    "(bu adimda istedigin kadar arac cagirabilirsin).\n"
    "2. Analiz bittiginde, kullaniciya kodun en onemli/en az aciklayici tasarim kararini "
    "hedef alan TEK bir rehber soru sor (orn. 'bu fonksiyon neden bir generator, "
    "ne kazandiriyor?', 'bu satir olmasa ne bozulur?'). SADECE soruyu yaz, baska aciklama, "
    "ozet veya cevap YAZMA.\n"
    "3. Kullanici cevap verdiginde: cevap dogruysa kisaca onayla ve bir adim daha derine inen "
    "yeni bir soru sor; cevap yanlissa/yetersizse duzeltme SOYLEMEDEN, kullaniciyi dogru yone "
    "iten ek bir ipucu sorusu sor.\n"
    "4. Kullanici acikca 'anlat'/'cevabi ver'/'pas gecelim' derse, o zaman direkt ve net acikla.\n\n"
    f"Mevcut araclar: {_tool_names(SOCRATIC_TOOLS_SCHEMA)}.\n"
    "Bir arac cagirmak icin SADECE asagidaki JSON formatini ciktinda bulundur, BASKA HICBIR SEY YAZMA:\n"
    "{\n"
    "  \"name\": \"read_file\",\n"
    "  \"arguments\": {\"path\": \"...\"}\n"
    "}\n\n"
    "YANITLARINI KESINLIKLE VE SADECE TURKCE DILINDE VERECEKSIN."
)


class CoderAgent:
    def __init__(self, model: str = DEFAULT_CODER_MODEL, client: OllamaClient | None = None):
        self.model = model
        self.client = client or OllamaClient()
        self.manager = ModelManager(self.client)

    def run(self, prompt: str, mode: str = "code") -> str:
        if mode == "socratic":
            system_prompt = SOCRATIC_SYSTEM_PROMPT
            tools_schema, tool_executor = SOCRATIC_TOOLS_SCHEMA, SOCRATIC_TOOL_EXECUTOR
        elif mode == "debug":
            system_prompt = DEBUG_SYSTEM_PROMPT
            tools_schema, tool_executor = CODER_TOOLS_SCHEMA, CODER_TOOL_EXECUTOR
        else:
            system_prompt = CODE_SYSTEM_PROMPT
            tools_schema, tool_executor = CODER_TOOLS_SCHEMA, CODER_TOOL_EXECUTOR

        self.manager.ensure_loaded(self.model)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return run_agent_loop(
            self.client,
            self.model,
            messages,
            tools_schema=tools_schema,
            tool_executor=tool_executor,
        )
