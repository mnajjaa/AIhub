import json

import pytest

import agent_core


def test_run_agent_happy_path(monkeypatch):
    def fake_call_openai(*args, **kwargs):
        content = json.dumps(
            {
                "summary": "Test summary",
                "keywords": ["alpha", "beta"],
            }
        )
        return None, content

    monkeypatch.setattr(agent_core, "call_openai", fake_call_openai)

    result = agent_core.run_agent("summarizer", {"text": "Hello world"})

    assert result["summary"] == "Test summary"
    assert result["keywords"] == ["alpha", "beta"]


def test_run_agent_missing_input_field():
    with pytest.raises(ValueError, match="Missing input field"):
        agent_core.run_agent("summarizer", {})
