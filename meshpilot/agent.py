"""Agent 决策层：诊断 → 处方 → 执行 → 验证。

这就是 "MeshPilot" 这个名字的含义：不是又一个减面 CLI，
而是一个能自己看网格健康状况、开优化处方、执行并验收的 agent。

工作方式（默认纯规则引擎，零外部依赖）：
1. diagnose：拓扑体检 + 与平台预算的差距
2. prescribe：把诊断转成有序优化计划，每步带理由
3. execute：按计划跑流水线，每步记录前后指标
4. verify：对照预算验收（面数 + chamfer 偏差），不过关明确报告

LLM 可插拔：advisor 参数接收一个 callable(prompt)->str。
传入后，处方理由会附带 LLM 建议（例如"这个资产建议保留 UV"）；
不传则纯规则运行——核心功能永远不依赖任何 API key。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import quality, topology as topo, weld as weld_mod
from .meshio import Mesh
from .qem import simplify

# ---------------------------------------------------------------- 预算
PLATFORM_BUDGETS = {
    "web_mobile": 25_000,      # 手机浏览器 three.js/babylon
    "web_desktop": 100_000,    # 桌面 WebGL
    "pc": 500_000,             # PC 游戏
    "console": 2_000_000,      # 主机
}


@dataclass
class Budget:
    max_faces: int
    max_chamfer_norm: float = 0.01   # 允许的最大表面偏差（相对对角线）
    platform: str = "custom"
    keep_uvs: bool = True

    @classmethod
    def for_platform(cls, platform: str,
                     max_chamfer_norm: float = 0.01) -> "Budget":
        if platform not in PLATFORM_BUDGETS:
            raise ValueError(
                f"未知平台 {platform}，可选：{list(PLATFORM_BUDGETS)}")
        return cls(max_faces=PLATFORM_BUDGETS[platform],
                   max_chamfer_norm=max_chamfer_norm, platform=platform)


# ---------------------------------------------------------------- 诊断
@dataclass
class Diagnosis:
    topo: dict
    faces_over_budget: int
    estimated_weld_savings: int
    problems: list[str] = field(default_factory=list)


def diagnose(mesh: Mesh, budget: Budget) -> Diagnosis:
    rep = topo.analyze(mesh)
    return Diagnosis(
        topo=rep.to_dict(),
        faces_over_budget=max(0, mesh.n_faces - budget.max_faces),
        estimated_weld_savings=rep.weldable_vertices,
        problems=list(rep.problems),
    )


# ---------------------------------------------------------------- 处方
@dataclass
class PlanStep:
    action: str            # weld / clean / decimate / lod
    params: dict
    rationale: str


def prescribe(diagnosis: Diagnosis, budget: Budget,
              advisor=None) -> list[PlanStep]:
    steps: list[PlanStep] = []
    if diagnosis.estimated_weld_savings > 0:
        steps.append(PlanStep(
            "weld", {"eps": 1e-6},
            f"合并 {diagnosis.estimated_weld_savings} 个重复顶点："
            "零几何损失，直接减少顶点缓冲区"))
    if diagnosis.topo.get("degenerate_faces", 0) > 0:
        steps.append(PlanStep(
            "clean", {},
            f"清除 {diagnosis.topo['degenerate_faces']} 个退化三角形："
            "零面积面会让减面器的误差度量失真"))
    if diagnosis.faces_over_budget > 0:
        # 留 10% 余量，给后续编辑/压缩留空间
        target = max(int(budget.max_faces * 0.9), 4)
        steps.append(PlanStep(
            "decimate", {"target_faces": target},
            f"面数 {diagnosis.topo['n_faces']} 超出 {budget.platform} "
            f"预算 {budget.max_faces}，QEM 减面至 {target}"))
    if advisor is not None:
        prompt = prompt_for_advisor(diagnosis, budget)
        try:
            advice = advisor(prompt)
            if advice:
                for s in steps:
                    s.rationale += f"｜LLM 建议：{advice.strip()[:200]}"
        except Exception as e:  # advisor 故障不影响主流程
            for s in steps:
                s.rationale += f"｜LLM 建议不可用（{e}），按规则执行"
    return steps


def prompt_for_advisor(diagnosis: Diagnosis, budget: Budget) -> str:
    return (
        f"你是 3D 资产优化顾问。资产体检：{diagnosis.topo}。"
        f"目标平台 {budget.platform}，面数预算 {budget.max_faces}，"
        f"最大允许表面偏差 {budget.max_chamfer_norm}（相对对角线归一化）。"
        f"请用两句话给出优化优先级建议。"
    )


# ---------------------------------------------------------------- 执行
@dataclass
class StepResult:
    step: PlanStep
    before: dict
    after: dict
    note: str = ""


@dataclass
class RunResult:
    mesh: Mesh
    steps: list[StepResult]
    verification: dict
    metrics_before: dict
    metrics_after: dict

    @property
    def passed(self) -> bool:
        return self.verification.get("passed", False)


def _metrics(mesh: Mesh) -> dict:
    rep = topo.analyze(mesh)
    d = mesh.stats()
    d.update({"nonmanifold_edges": rep.nonmanifold_edges,
              "degenerate_faces": rep.degenerate_faces})
    return d


def run(mesh: Mesh, budget: Budget, *, advisor=None,
        max_cost_ratio: float = 0.05) -> RunResult:
    """完整闭环：诊断 → 处方 → 执行 → 验证。

    max_cost_ratio：折叠代价上限（相对模型对角线平方的比例）。
    含噪资产（扫描件）的 quadric 代价天然偏高，0.05 是教学默认；
    严格模式可降到 0.01——代价超限时保守停止而非硬减。
    """
    diagnosis = diagnose(mesh, budget)
    plan = prescribe(diagnosis, budget, advisor=advisor)
    current = mesh
    steps: list[StepResult] = []
    before = _metrics(current)

    for step in plan:
        pre = _metrics(current)
        note = ""
        if step.action == "weld":
            current, st = weld_mod.weld(current, **step.params)
            note = f"顶点 {st['before']} → {st['after']}"
        elif step.action == "clean":
            current, removed = topo.remove_degenerate(current)
            note = f"移除 {removed} 个退化面"
        elif step.action == "decimate":
            current, st = simplify(current, **step.params,
                                   max_cost_ratio=max_cost_ratio)
            note = (f"面数 {st.faces_before} → {st.faces_after}，"
                    f"折叠 {st.collapses} 次，"
                    f"翻转拒绝 {st.rejected_flips} 次"
                    + (f"；{st.stopped_reason}" if st.stopped_reason else ""))
        post = _metrics(current)
        steps.append(StepResult(step, pre, post, note))

    after = _metrics(current)
    if not steps:
        # 无操作（资产本就在预算内）：自身 chamfer 只有采样底噪，无意义
        verification = {
            "passed": True,
            "checks": {"faces_within_budget": True,
                       "chamfer_within_budget": True},
            "faces": current.n_faces, "budget_faces": budget.max_faces,
            "chamfer_norm": 0.0,
            "budget_chamfer_norm": budget.max_chamfer_norm,
            "coverage": 1.0,
            "note": "资产未修改（已在预算内）",
        }
    else:
        verification = _verify(current, budget, mesh)
    return RunResult(mesh=current, steps=steps, verification=verification,
                     metrics_before=before, metrics_after=after)


def _verify(mesh: Mesh, budget: Budget, original: Mesh) -> dict:
    ch = quality.chamfer_distance(original, mesh, n_samples=8000)
    faces_ok = mesh.n_faces <= budget.max_faces
    quality_ok = ch["symmetric_norm"] <= budget.max_chamfer_norm
    checks = {
        "faces_within_budget": faces_ok,
        "chamfer_within_budget": quality_ok,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "faces": mesh.n_faces,
        "budget_faces": budget.max_faces,
        "chamfer_norm": ch["symmetric_norm"],
        "budget_chamfer_norm": budget.max_chamfer_norm,
        "coverage": ch["coverage"],
    }
