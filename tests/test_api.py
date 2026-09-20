import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["vector_store"] == "connected"
    assert data["models"] == "loaded"

def test_stats_endpoint():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_tickets" in data
    assert data["total_tickets"] > 0
    assert "queue_distribution" in data
    assert "priority_distribution" in data

def test_tickets_endpoint():
    response = client.get("/tickets?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "tickets" in data
    assert len(data["tickets"]) <= 5
    if data["tickets"]:
        ticket = data["tickets"][0]
        assert "id" in ticket
        assert "subject" in ticket
        assert "queue" in ticket
        assert "priority" in ticket

def test_triage_endpoint():
    payload = {
        "subject": "Payment failed during checkout",
        "body": "My credit card was declined and error code 4002 appeared."
    }
    response = client.post("/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "queue" in data
    assert "queue_confidence" in data
    assert "priority" in data
    assert "priority_confidence" in data
    assert data["queue_confidence"] >= 0.0

def test_chat_off_topic():
    payload = {
        "session_id": "test-pytest-1",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ]
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "cited_ticket_ids" in data
    assert "tool_trace" in data
    assert data["tool_trace"] == "None"
    assert "capital" in data["answer"].lower() or "support" in data["answer"].lower()
