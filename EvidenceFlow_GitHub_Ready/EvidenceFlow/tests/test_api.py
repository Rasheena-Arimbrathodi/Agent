from fastapi.testclient import TestClient
from evidenceflow.api import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["documents_indexed"] >= 4


def test_documents_endpoint():
    resp = client.get("/documents")
    assert resp.status_code == 200
    docs = resp.json()
    assert any(d["doc_id"] == "policy_remote_work" for d in docs)


def test_ask_endpoint():
    resp = client.post("/ask", json={"query": "How often must API keys be rotated?", "top_k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert "security_guidelines" in body["citations"]
    assert "90 days" in body["answer"]


def test_evaluate_endpoint():
    resp = client.get("/evaluate")
    assert resp.status_code == 200
    body = resp.json()
    assert body["passed"] == 5
    assert body["pass_rate"] == 1.0
