from types import SimpleNamespace

import app as app_module


def test_health_is_unavailable_when_model_is_missing(monkeypatch):
    def missing_model():
        raise FileNotFoundError("not trained")

    monkeypatch.setattr(app_module, "get_model", missing_model)
    response = app_module.app.test_client().get("/api/health")

    assert response.status_code == 503
    assert response.get_json()["model_loaded"] is False


def test_predict_rejects_empty_text():
    response = app_module.app.test_client().post("/api/predict", json={"text": "  "})

    assert response.status_code == 400
    assert "non-empty" in response.get_json()["error"]


def test_predict_rejects_non_json_body():
    response = app_module.app.test_client().post(
        "/api/predict", data="not json", content_type="text/plain"
    )

    assert response.status_code == 400


def test_predict_returns_label_and_scope_note(monkeypatch):
    class FakeModel:
        def predict(self, texts):
            assert texts == ["A sample news article."]
            return ["REAL"]

    monkeypatch.setattr(app_module, "get_model", lambda: FakeModel())
    response = app_module.app.test_client().post(
        "/api/predict", json={"text": "A sample news article."}
    )

    assert response.status_code == 200
    assert response.get_json()["label"] == "REAL"
    assert "independently verify" in response.get_json()["note"]
