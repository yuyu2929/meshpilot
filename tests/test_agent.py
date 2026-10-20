"""Agent 闭环测试：诊断/处方/执行/验证 + LLM 插拔。"""
from __future__ import annotations

from meshpilot import agent, prims


def test_full_loop_passes():
    budget = agent.Budget(max_faces=400, platform="custom",
                          max_chamfer_norm=0.02)
    result = agent.run(prims.icosphere(3), budget)
    assert result.mesh.n_faces <= 400
    assert result.passed
    assert result.verification["checks"]["faces_within_budget"]
    assert result.verification["checks"]["chamfer_within_budget"]


def test_within_budget_is_noop():
    budget = agent.Budget(max_faces=100_000, platform="custom")
    result = agent.run(prims.icosphere(2), budget)
    assert result.mesh.n_faces == prims.icosphere(2).n_faces
    assert len(result.steps) == 0
    assert result.passed


def test_weld_step_fires_for_duplicated_mesh():
    budget = agent.Budget(max_faces=100_000, platform="custom")
    result = agent.run(prims.duplicated_vertices_mesh(2), budget)
    actions = [s.step.action for s in result.steps]
    assert "weld" in actions
    assert result.mesh.n_vertices < prims.duplicated_vertices_mesh(2).n_vertices


def test_clean_step_fires_for_degenerate_mesh():
    import numpy as np
    from meshpilot.meshio import Mesh
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 1, 1], [0, 2, 3]], dtype=np.int32)
    budget = agent.Budget(max_faces=100_000, platform="custom")
    result = agent.run(Mesh(verts, faces), budget)
    actions = [s.step.action for s in result.steps]
    assert "clean" in actions
    assert result.mesh.n_faces == 2


def test_advisor_injected_and_failure_tolerated():
    budget = agent.Budget(max_faces=400, platform="custom")
    calls = []

    def advisor(prompt):
        calls.append(prompt)
        return "建议保留 UV，优先减面"

    result = agent.run(prims.icosphere(3), budget, advisor=advisor)
    assert calls, "advisor 应被调用"
    assert any("LLM 建议" in s.step.rationale for s in result.steps)

    def bad_advisor(prompt):
        raise RuntimeError("LLM 挂了")

    result2 = agent.run(prims.icosphere(3), budget, advisor=bad_advisor)
    assert result2.passed
    assert any("不可用" in s.step.rationale for s in result2.steps)


def test_platform_budgets():
    assert agent.PLATFORM_BUDGETS["web_mobile"] == 25_000
    b = agent.Budget.for_platform("pc")
    assert b.max_faces == 500_000
    try:
        agent.Budget.for_platform("nope")
        assert False
    except ValueError:
        pass


def test_verification_fails_when_budget_too_tight():
    # 极小预算 + 极严偏差 → 应不通过且原因明确
    budget = agent.Budget(max_faces=30, platform="custom",
                          max_chamfer_norm=0.0001)
    result = agent.run(prims.icosphere(3), budget)
    assert not result.passed
    checks = result.verification["checks"]
    assert not all(checks.values())
