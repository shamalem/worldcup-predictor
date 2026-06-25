"""API smoke tests using FastAPI's TestClient (no trained model required)."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "docs" in r.json()


def test_predict_without_model_is_503_or_200():
    """If no model is trained it must 503; if trained, it must return probs."""
    r = client.post("/api/predict", json={
        "team_a": "Brazil", "team_b": "Germany",
        "year": 2014, "stage": "Final", "neutral": True, "host": "none"})
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        probs = r.json()["probabilities"]
        total = probs["team_a"] + probs["draw"] + probs["team_b"]
        assert abs(total - 1.0) < 0.05
