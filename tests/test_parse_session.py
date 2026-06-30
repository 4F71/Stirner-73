"""_parse_session — JSON ve Markdown format round-trip testleri."""

import json

from main import _parse_session


def test_json_format_roundtrip():
    msgs = [
        {"role": "user", "content": "merhaba"},
        {"role": "assistant", "content": "nasıl yardımcı olabilirim?"},
    ]
    raw = json.dumps(msgs, ensure_ascii=False)
    result = _parse_session(raw)
    assert result == msgs


def test_json_format_with_header_collision():
    """İçeriğinde '### user' geçen mesaj Markdown parse'da bozulurdu."""
    msgs = [
        {"role": "user", "content": "### user bu bir başlık mı?"},
        {"role": "assistant", "content": "hayır, bu normal içerik"},
    ]
    raw = json.dumps(msgs, ensure_ascii=False)
    result = _parse_session(raw)
    assert result[0]["content"] == "### user bu bir başlık mı?"


def test_legacy_markdown_format():
    raw = "### user\n\nmerhaba\n\n### assistant\n\nnasıl yardımcı olabilirim?\n"
    result = _parse_session(raw)
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "merhaba"
    assert result[1]["role"] == "assistant"


def test_broken_json_falls_back_to_markdown():
    raw = '{"broken": [}\n\n### user\n\nselam\n'
    result = _parse_session(raw)
    assert any(m["role"] == "user" for m in result)


def test_empty_raw_returns_empty_list():
    assert _parse_session("") == []
    assert _parse_session("   ") == []


def test_json_filters_invalid_entries():
    raw = json.dumps([{"role": "user", "content": "ok"}, {"no_role": "bad"}])
    result = _parse_session(raw)
    assert len(result) == 1
    assert result[0]["role"] == "user"
