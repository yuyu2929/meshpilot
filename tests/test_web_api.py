"""Web API 冒烟测试。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from run_web import create_app


def test_health():
    c = TestClient(create_app())
    assert c.get("/api/health").json() == {"status": "ok"}


def test_meshes_list():
    c = TestClient(create_app())
    data = c.get("/api/meshes").json()
    assert len(data["meshes"]) == 4
    assert all(m["faces"] > 0 for m in data["meshes"])


def test_optimize_endpoint():
    c = TestClient(create_app())
    r = c.post("/api/optimize", json={
        "mesh": "icosphere(2)", "budget_faces": 150, "chamfer": 0.02})
    data = r.json()
    assert "error" not in data
    assert data["before"]["n_faces"] == 320
    assert data["after"]["n_faces"] <= 150
    assert data["verification"]["passed"] is True
    assert len(data["steps"]) >= 1


def test_optimize_noop_within_budget():
    c = TestClient(create_app())
    r = c.post("/api/optimize", json={
        "mesh": "icosphere(2)", "budget_faces": 100_000, "chamfer": 0.02})
    data = r.json()
    assert data["after"]["n_faces"] == 320
    assert data["steps"] == []


def test_optimize_unknown_mesh():
    c = TestClient(create_app())
    r = c.post("/api/optimize", json={"mesh": "nope"})
    assert "error" in r.json()


def test_license_endpoint():
    c = TestClient(create_app())
    data = c.get("/api/license").json()
    assert data["tier"] == "community"
    assert set(data["features"]) == {"batch_pipeline", "ci_report", "sso"}


# ---------------- 输入校验（v0.1.1 加固） ----------------

def test_budget_bounds_rejected():
    """负数/零/超大会被 pydantic 拦成 422，而不是白跑一遍 CPU。"""
    c = TestClient(create_app())
    for bad in (0, -1, 3, 2_000_001):
        r = c.post("/api/optimize", json={"mesh": "icosphere(2)", "budget_faces": bad})
        assert r.status_code == 422, f"budget_faces={bad} 竟然通过"


def test_chamfer_bounds_rejected():
    c = TestClient(create_app())
    for bad in (-0.1, 1.5):
        r = c.post("/api/optimize", json={"mesh": "icosphere(2)", "chamfer": bad})
        assert r.status_code == 422


def test_mesh_name_length_capped():
    c = TestClient(create_app())
    r = c.post("/api/optimize", json={"mesh": "x" * 200})
    assert r.status_code == 422