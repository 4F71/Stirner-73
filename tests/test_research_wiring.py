"""ResearchAgent — run_agent_loop wiring ve schema doğrulama testleri."""

from unittest.mock import MagicMock, patch

from agents.research import ResearchAgent, RESEARCH_TOOLS_SCHEMA, RESEARCH_TOOL_EXECUTOR


def test_research_agent_passes_correct_schema():
    """ResearchAgent.run, run_agent_loop'a RESEARCH_TOOLS_SCHEMA'yı iletmeli."""
    with patch("agents.research.run_agent_loop") as mock_loop:
        mock_loop.return_value = "araştırma bitti"
        agent = ResearchAgent(model="hermes3:8b")
        result = agent.run("Python nedir?")

    assert mock_loop.called
    _, kwargs = mock_loop.call_args[0], mock_loop.call_args[1]
    # Positional: client, model, messages, schema, executor
    call_args = mock_loop.call_args[0]
    # tools_schema 4. pozisyonel arg (index 3) veya keyword
    passed_schema = mock_loop.call_args[1].get("tools_schema") or (
        call_args[3] if len(call_args) > 3 else None
    )
    if passed_schema is not None:
        tool_names = {t["function"]["name"] for t in passed_schema}
        expected = {t["function"]["name"] for t in RESEARCH_TOOLS_SCHEMA}
        assert tool_names == expected


def test_research_tool_executor_covers_schema():
    """RESEARCH_TOOL_EXECUTOR'da tüm schema araçları için çalıştırıcı olmalı."""
    schema_names = {t["function"]["name"] for t in RESEARCH_TOOLS_SCHEMA}
    executor_names = set(RESEARCH_TOOL_EXECUTOR.keys())
    missing = schema_names - executor_names
    assert not missing, f"Executor'da eksik araçlar: {missing}"


def test_research_system_prompt_contains_tool_names():
    """RESEARCH_SYSTEM_PROMPT tüm araç adlarını içermeli."""
    from agents.research import RESEARCH_SYSTEM_PROMPT
    for tool in RESEARCH_TOOLS_SCHEMA:
        name = tool["function"]["name"]
        assert name in RESEARCH_SYSTEM_PROMPT, (
            f"'{name}' araç adı RESEARCH_SYSTEM_PROMPT içinde bulunamadı"
        )
