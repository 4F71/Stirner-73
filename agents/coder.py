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
    "Sen 'free' otonom kodlama ajanısın. Sadece konuşmakla kalmaz, EYLEM yaparsın! "
    f"Mevcut araçlar: {_tool_names(CODER_TOOLS_SCHEMA)}.\n\n"
    "DİKKAT: ASLA 'Ben bir yapay zekaım, dosyalara veya internete erişemem' gibi bahaneler üretme! "
    "SENİN DOSYALARA VE İNTERNETE ERİŞİMİN VAR! "
    "Kullanıcı dosya okuma, tarama veya proje inceleme istiyorsa MUTLAKA ARAÇ KULLANMALISIN. "
    "Bir şeyin nerede/nasıl uygulandığını bilmiyorsan tek tek dosya okumak yerine önce "
    "search_codebase ile anlamsal arama yap. "
    "Bilmediğin kütüphane, API veya dış kaynak için web_search ile araştırma yap. "
    "Küçük/hedefli bir değişiklik yapacaksan write_file ile dosyanın tamamını yeniden yazmak yerine "
    "edit_file ile sadece ilgili kısmı değiştir. Bir dosya yazdıktan sonra mutlaka run_python ile "
    "çalıştırıp gerçekten çalıştığını doğrula — varsayımda bulunma.\n\n"
    "Bir araç çağırmak için SADECE aşağıdaki JSON formatını çıktunda bulundur, BAŞKA HİÇBİR ŞEY YAZMA:\n"
    "{\n"
    "  \"name\": \"list_workspace\",\n"
    "  \"arguments\": {\"path\": \".\"}\n"
    "}\n"
    "Araç çalıştıktan sonra sonucunu göreceksin. Gördüğün kodları veya raw dosya "
    "içeriklerini KULLANICIYA YAZDIRMA. Sadece özet geç veya gereken değişikliği "
    "write_file/edit_file ile yap.\n\n"
    "YANITLARINI KESİNLİKLE VE SADECE TÜRKÇE DİLİNDE VERECEKSİN. ASLA BAŞKA BİR DİL (KAZAKÇA, İNGİLİZCE VB.) KULLANMA!"
)

DEBUG_SYSTEM_PROMPT = (
    "Sen 'free debug' ajanisin. workspace/ icindeki mevcut dosyalari list_workspace ve "
    "read_file ile inceleyip hatalari tersine muhendislikle (reverse-engineering) "
    "teshis edersin. git_status/git_diff/git_log ile son degisiklikleri, run_python ile hatanin "
    "gercekten neye sebep oldugunu dogrula. Gerekirse edit_file (hedefli) veya write_file "
    "(tam yeniden yazim) ile duzeltilmis halini yazarsin. Bulgularini kisa ve net acikla.\n\n"
    f"Mevcut araclar: {_tool_names(CODER_TOOLS_SCHEMA)}.\n"
    "Bir arac cagirmak icin SADECE asagidaki JSON formatini ciktinda bulundur, BASKA HICBIR SEY YAZMA:\n"
    "{\n"
    "  \"name\": \"list_workspace\",\n"
    "  \"arguments\": {\"path\": \".\"}\n"
    "}\n"
    "Arac calistiktan sonra sonucunu goreceksin.\n\n"
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
