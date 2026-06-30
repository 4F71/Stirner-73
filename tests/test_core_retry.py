from unittest.mock import MagicMock

import agents.config as config
from agents.core import run_agent_loop


def make_client(contents):
    """contents: list of assistant message dicts to return on successive chat() calls."""
    client = MagicMock()
    client.chat.side_effect = [{"message": c} for c in contents]
    return client


def test_malformed_tool_json_triggers_retry_then_accepts_clean_answer():
    config.verbose = False
    config.audit_enabled = False
    client = make_client([
        {"role": "assistant", "content": '{"tool": "foo", "args": {}}'},  # missing name/arguments
        {"role": "assistant", "content": "Tamamlandı, iş bitti."},
    ])
    messages = [{"role": "user", "content": "bir seyi yap"}]

    result = run_agent_loop(client, "fake-model", messages, max_turns=5)

    assert client.chat.call_count == 2
    # Final turn had no tool_calls -> content (last message text) is returned.
    assert result == "Tamamlandı, iş bitti."
    # run_agent_loop now operates on a shallow copy so caller's list is untouched.
    assert len(messages) == 1

    # Verify that the retry system message was injected into the internal history
    # by inspecting what was passed to the second chat() call.
    second_call_history = client.chat.call_args_list[1][0][1]  # positional arg: messages
    roles = [m["role"] for m in second_call_history]
    assert "user" in roles
    # The corrective [RETRY] marker must appear between the malformed turn
    # and the final clean answer request.
    contents = [m.get("content", "") for m in second_call_history]
    assert any("[RETRY]" in c for c in contents)


def test_malformed_string_arguments_trigger_retry():
    """Tool call gelir ama arguments alanı geçersiz JSON string'dir."""
    config.verbose = False
    config.audit_enabled = False

    client = MagicMock()
    # İlk yanıt: name doğru, arguments bozuk JSON string
    bad_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "agents/core.py"',  # kapanış eksik
                }
            }
        ],
    }
    good_answer = {"role": "assistant", "content": "İşte sonuç."}
    client.chat.side_effect = [{"message": bad_call}, {"message": good_answer}]

    messages = [{"role": "user", "content": "dosyayı oku"}]
    result = run_agent_loop(client, "fake-model", messages, max_turns=5)

    assert client.chat.call_count == 2
    # İkinci çağrıdaki geçmişte JSON parse hata mesajı olmalı.
    second_history = client.chat.call_args_list[1][0][1]
    error_msgs = [
        m for m in second_history
        if "gecerli JSON degil" in m.get("content", "")
        or "parse edilemedi" in m.get("content", "")
    ]
    assert error_msgs, "JSON parse hata mesaji gecmiste bulunamadi"


def test_persistent_malformed_json_hits_max_turns_without_infinite_loop():
    config.verbose = False
    config.audit_enabled = False
    client = make_client([
        {"role": "assistant", "content": '{"tool": "foo", "args": {}}'},
        {"role": "assistant", "content": '{"tool": "bar", "args": {}}'},
    ])
    messages = [{"role": "user", "content": "bir seyi yap"}]

    result = run_agent_loop(client, "fake-model", messages, max_turns=2)

    assert client.chat.call_count == 2
    assert "tur limitine ulasildi" in result
