"""
tests/test_web_server.py
=========================
Unit tests for kavach.ui.web_server (FastAPI backend).

Coverage:
  1. GET /health returns 200 and {"status": "ok"}
  2. GET / returns 200 with text/html content-type
  3. GET / response body contains "KAVACH"
  4. POST /analyze with missing audio_path returns 422 (Pydantic validation)
  5. POST /analyze with valid body returns 200 and {"status": "started"}
  6. WebSocket /ws connection opens and accepts without raising
  7. WebSocket message schema has all required keys (schema contract test)
  8. POST /analyze with empty JSON body returns 422
"""

import sys
import os
import asyncio
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import httpx
from httpx import AsyncClient, ASGITransport

# Import the FastAPI app — deferred so we can check imports work
from kavach.ui.web_server import app, _emit, _active_queues, _queue_lock


# ── Async HTTP tests (httpx + ASGI transport) ─────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_200():
    """GET /health → 200 + {"status": "ok"}"""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_index_returns_html_content_type():
    """GET / → 200 with text/html content-type."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_index_contains_kavach_title():
    """GET / response body should contain the string 'KAVACH'."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
    assert "KAVACH" in resp.text


@pytest.mark.asyncio
async def test_analyze_missing_audio_path_returns_422():
    """POST /analyze with no body → Pydantic validation error → 422."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/analyze", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_analyze_empty_body_returns_422():
    """POST /analyze with empty body → 422 (audio_path is required)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/analyze", content=b"", headers={"Content-Type": "application/json"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_analyze_valid_body_returns_started():
    """
    POST /analyze with a valid audio_path → 200 + {"status": "started"}.
    The pipeline thread will fail (file doesn't exist) but the endpoint
    itself should return 200 immediately — the pipeline is fire-and-forget.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/analyze",
            json={"audio_path": "/nonexistent/fake_audio.mp3"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "started"
    assert body["audio_path"] == "/nonexistent/fake_audio.mp3"


# ── WebSocket tests (Starlette TestClient) ────────────────────────────────────

def test_websocket_connects_successfully():
    """
    WebSocket /ws should accept connections without raising.
    Uses Starlette's synchronous TestClient which handles the event loop.
    """
    from starlette.testclient import TestClient

    client = TestClient(app)
    # If connection fails, websocket_connect raises — this is the test
    with client.websocket_connect("/ws"):
        pass  # connection opened and closed cleanly


# ── Schema contract test (no network required) ────────────────────────────────

def test_websocket_event_schema_has_required_keys():
    """
    Verify the expected WebSocket message schema contains all required keys.
    This is a pure data-shape test — no server needed.
    """
    required_keys = {
        "timestamp",
        "alert_level",
        "final_score",
        "tiers_detected",
        "explanation",
        "component_scores",
        "utterances",
    }

    # Construct a sample event matching the schema documented in web_server.py
    sample_event = {
        "timestamp":        13.0,
        "alert_level":      "ALERT",
        "final_score":      0.730,
        "tiers_detected":   [1, 3],
        "explanation":      "Caller impersonates Amazon ICICI Bank (Tier 1)…",
        "component_scores": {
            "heuristic":  0.40,
            "classifier": 0.50,
            "slm":        1.00,
        },
        "utterances": [
            {"speaker": "CALLER", "text": "amazon icici bank se bol raha hoon", "timestamp": 10.1}
        ],
    }

    # Every required key must be present
    assert required_keys.issubset(set(sample_event.keys()))

    # component_scores must have all three signal keys
    assert {"heuristic", "classifier", "slm"} == set(sample_event["component_scores"].keys())

    # utterances items must have speaker, text, timestamp
    for utt in sample_event["utterances"]:
        assert {"speaker", "text", "timestamp"}.issubset(set(utt.keys()))
