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
    # Final turn had no tool_calls -> content already printed, empty string returned
    assert result == ""
    # run_agent_loop now operates on a shallow copy so caller's list is untouched.
    assert len(messages) == 1

    # Verify that the retry system message was injected into the internal history
    # by inspecting what was passed to the second chat() call.
    second_call_history = client.chat.call_args_list[1][0][1]  # positional arg: messages
    roles = [m["role"] for m in second_call_history]
    assert "user" in roles
    # The corrective system warning must appear between the malformed assistant turn
    # and the final clean answer request.
    contents = [m.get("content", "") for m in second_call_history]
    assert any("SİSTEM UYARISI" in c for c in contents)


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
