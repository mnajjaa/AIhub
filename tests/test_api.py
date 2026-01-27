import json

from fastapi.testclient import TestClient

import agent_core


def test_api_run_summarizer(monkeypatch):
    def fake_call_openai(*args, **kwargs):
        content = json.dumps(
            {
                "summary": "API summary",
                "keywords": ["api"],
            }
        )
        return None, content

    monkeypatch.setattr(agent_core, "call_openai", fake_call_openai)

    import app as app_module

    monkeypatch.setattr(app_module, "MLFLOW_AVAILABLE", False)
    monkeypatch.setattr(app_module, "mlflow", None, raising=False)

    client = TestClient(app_module.app)
    response = client.post("/run/summarizer", json={"text": "Dummy text"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == "API summary"
    assert payload["keywords"] == ["api"]
