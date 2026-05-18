"""
Integration test: runs the full mock workflow end-to-end via the FastAPI
test client (no external network, no RDKit — uses mock mode throughout).
Verifies that every API step returns the expected shape and status transitions.
"""
import asyncio
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app, raise_server_exceptions=True)


def _poll_until(session_id: str, target_statuses: set, max_tries: int = 30) -> dict:
    """Poll session endpoint until one of target_statuses is reached."""
    import time
    for _ in range(max_tries):
        r = client.get(f"/api/session/{session_id}")
        assert r.status_code == 200
        data = r.json()
        if data["status"] in target_statuses:
            return data
        time.sleep(0.2)
    raise TimeoutError(f"Session {session_id} did not reach {target_statuses} in time. Last: {data['status']}")


class TestHealthCheck:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestLitReview:
    def test_lit_review_mock(self):
        r = client.post("/api/lit-review", json={
            "protein_name": "ABL1 kinase",
            "run_mode": "mock",
        })
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert data["status"] == "started"

    def test_lit_review_completes(self):
        r = client.post("/api/lit-review", json={
            "protein_name": "EGFR",
            "run_mode": "mock",
        })
        sid = r.json()["session_id"]
        session = _poll_until(sid, {"awaiting_approval", "error"})
        assert session["status"] == "awaiting_approval", session.get("error_message")

        lr = session["lit_review"]
        assert lr is not None
        assert lr["protein_name"] == "EGFR"
        assert isinstance(lr["existing_therapies"], list)
        assert lr["recommended_method"] in ("lbdd", "sbdd", "hybrid")
        assert isinstance(lr["known_actives_count"], int)

    def test_session_not_found(self):
        r = client.get("/api/session/nonexistent-uuid-1234")
        assert r.status_code == 404

    def test_empty_protein_name_rejected(self):
        r = client.post("/api/lit-review", json={
            "protein_name": "",
            "run_mode": "mock",
        })
        assert r.status_code == 422  # Pydantic validation


class TestFullMockPipeline:
    """Runs the complete LBDD mock pipeline end-to-end."""

    def test_lbdd_mock_pipeline(self):
        # Step 1: lit review
        r = client.post("/api/lit-review", json={
            "protein_name": "CDK2",
            "run_mode": "mock",
        })
        sid = r.json()["session_id"]
        _poll_until(sid, {"awaiting_approval"})

        # Step 2: approve with LBDD + zinc_250k
        r = client.post("/api/approve", json={
            "session_id": sid,
            "approved": True,
            "chosen_method": "lbdd",
            "dataset": "zinc_250k",
            "run_mode": "mock",
        })
        assert r.status_code == 200

        session = _poll_until(sid, {"awaiting_md_decision", "error"})
        assert session["status"] == "awaiting_md_decision", session.get("error_message")

        dr = session["discovery_result"]
        assert dr is not None
        assert dr["method_used"] == "lbdd"
        assert dr["mock"] is True
        assert dr["compounds_screened"] > 0
        assert len(dr["top_hits"]) > 0
        assert len(dr["figures"]) > 0

        vm = dr["validation_metrics"]
        assert 0.5 < vm["auc_roc"] < 1.0
        assert 0.0 < vm["bedroc"] <= 1.0
        assert vm["ef1_percent"] > 1.0

        # Step 3: skip MD
        r = client.post("/api/md", json={
            "session_id": sid,
            "run_md": False,
            "run_mode": "mock",
        })
        assert r.status_code == 200
        session = _poll_until(sid, {"manuscript_done"})
        assert session["status"] == "manuscript_done"
        assert session["manuscript"] is not None

    def test_sbdd_mock_pipeline(self):
        r = client.post("/api/lit-review", json={
            "protein_name": "BRAF",
            "run_mode": "mock",
        })
        sid = r.json()["session_id"]
        _poll_until(sid, {"awaiting_approval"})

        r = client.post("/api/approve", json={
            "session_id": sid,
            "approved": True,
            "chosen_method": "sbdd",
            "dataset": "fda_approved",
            "run_mode": "mock",
        })
        session = _poll_until(sid, {"awaiting_md_decision", "error"})
        assert session["status"] == "awaiting_md_decision"

        dr = session["discovery_result"]
        assert dr["method_used"] == "sbdd"
        vm = dr["validation_metrics"]
        assert vm.get("pose_rmsd_mean") is not None
        assert vm.get("pose_success_rate_2A") is not None

    def test_hybrid_mock_pipeline(self):
        r = client.post("/api/lit-review", json={
            "protein_name": "KRAS G12C",
            "run_mode": "mock",
        })
        sid = r.json()["session_id"]
        _poll_until(sid, {"awaiting_approval"})

        r = client.post("/api/approve", json={
            "session_id": sid,
            "approved": True,
            "chosen_method": "hybrid",
            "dataset": "zinc_250k",
            "run_mode": "mock",
        })
        session = _poll_until(sid, {"awaiting_md_decision", "error"})
        assert session["status"] == "awaiting_md_decision"
        assert session["discovery_result"]["method_used"] == "hybrid"

    def test_md_mock_pipeline(self):
        r = client.post("/api/lit-review", json={
            "protein_name": "HIV protease",
            "run_mode": "mock",
        })
        sid = r.json()["session_id"]
        _poll_until(sid, {"awaiting_approval"})

        client.post("/api/approve", json={
            "session_id": sid,
            "approved": True,
            "chosen_method": "lbdd",
            "dataset": "zinc_250k",
            "run_mode": "mock",
        })
        _poll_until(sid, {"awaiting_md_decision"})

        r = client.post("/api/md", json={
            "session_id": sid,
            "run_md": True,
            "duration": {"value": 1, "unit": "ns"},
            "run_mode": "mock",
        })
        assert r.status_code == 200

        session = _poll_until(sid, {"manuscript_done", "error"})
        assert session["status"] == "manuscript_done", session.get("error_message")

        md = session["md_result"]
        assert md is not None
        assert md["duration_ns"] == 1.0
        assert md["frames_analyzed"] > 0
        assert md["rmsd_mean_nm"] > 0
        assert len(md["figures"]) > 0

    def test_cancel_approval(self):
        r = client.post("/api/lit-review", json={
            "protein_name": "TP53",
            "run_mode": "mock",
        })
        sid = r.json()["session_id"]
        _poll_until(sid, {"awaiting_approval"})

        r = client.post("/api/approve", json={
            "session_id": sid,
            "approved": False,
            "dataset": "zinc_250k",
            "run_mode": "mock",
        })
        assert r.json()["status"] == "cancelled"


class TestRateLimiting:
    def test_lit_review_rate_limit(self):
        """Exceeding RATE_LIMIT_MAX (default 20) requests/min triggers a 429."""
        import os
        max_req = int(os.getenv("RATE_LIMIT_MAX", "20"))
        responses = []
        for _ in range(max_req + 5):
            r = client.post("/api/lit-review", json={
                "protein_name": "ABL1",
                "run_mode": "mock",
            })
            responses.append(r.status_code)
        assert 429 in responses, f"Expected a 429 among: {set(responses)}"


class TestInputSanitisation:
    def test_protein_name_length_capped(self):
        long_name = "A" * 500
        r = client.post("/api/lit-review", json={
            "protein_name": long_name,
            "run_mode": "mock",
        })
        # Should succeed — sanitisation truncates rather than rejects
        assert r.status_code == 200

    def test_path_traversal_in_session_id(self):
        r = client.get("/api/session/../../../etc/passwd")
        assert r.status_code in (404, 422)

    def test_path_traversal_in_figure_download(self):
        r = client.get("/api/figures/../../etc/passwd/test.png")
        assert r.status_code in (404, 422)
