import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "Telecom-Support-Intelligence"}


def test_query_esim_policy():
    response = client.post(
        "/api/v1/query",
        json={"query": "What is the policy for eSIM activation?", "top_k": 2}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["sources"]) > 0
    assert data["sources"][0]["policy_id"] == "Doc_04"


def test_invalid_top_k():
    response = client.post(
        "/api/v1/query",
        json={"query": "eSIM activation", "top_k": 20}  # Exceeds max limit of 10
    )
    assert response.status_code == 422  # Unprocessable Entity