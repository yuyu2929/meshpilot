"""MeshPilot 命令行。

    python -m meshpilot analyze  model.obj [--json]
    python -m meshpilot optimize model.obj -o out.obj --platform web_mobile
    python -m meshpilot lod       model.obj -o lods/ [--ratios 1,0.5,0.25]
    python -m meshpilot demo                       # 内置演示网格
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import agent, lod as lod_mod, prims
from .meshio import load, save
from .topology import analyze as topo_analyze


def _print_topo(mesh) -> None:
    rep = topo_analyze(mesh)
    print(f"  顶点 {rep.n_vertices}  面 {rep.n_faces}  "
          f"欧拉数 {rep.euler_characteristic}")
    print(f"  闭合 {rep.closed}  非流形边 {rep.nonmanifold_edges}  "
          f"边界边 {rep.boundary_edges}")
    print(f"  退化面 {rep.degenerate_faces}  重复面 {rep.duplicate_faces}  "
          f"孤立顶点 {rep.orphan_vertices}  可合并顶点 {rep.weldable_vertices}")
    if rep.problems:
        print("  问题清单:")
        for p in rep.problems:
            print(f"    - {p}")
    else:
        print("  未发现问题 ✓")


def cmd_analyze(args) -> int:
    meshes = load(args.file)
    for m in meshes:
        print(f"[{m.name}] ({m.source_format})")
        _print_topo(m)
        if args.json:
            print(json.dumps(topo_analyze(m).to_dict(), ensure_ascii=False,
                             indent=2))
    return 0


def cmd_optimize(args) -> int:
    meshes = load(args.file)
    budget = (agent.Budget.for_platform(args.platform, args.chamfer)
              if args.platform else
              agent.Budget(max_faces=args.max_faces, platform="custom",
                           max_chamfer_norm=args.chamfer))
    for m in meshes:
        print(f"[{m.name}] 优化前：面 {m.n_faces}  顶点 {m.n_vertices}")
        result = agent.run(m, budget)
        print(f"  计划 {len(result.steps)} 步：")
        for s in result.steps:
            print(f"    {s.step.action}: {s.note}")
        v = result.verification
        print(f"  验收 {'✓ 通过' if v['passed'] else '✗ 未通过'}："
              f"面数 {v['faces']}/{v['budget_faces']}，"
              f"偏差 {v['chamfer_norm']:.5f}/{v['budget_chamfer_norm']}")
        if args.out:
            save(result.mesh, args.out)
            print(f"  已写出 → {args.out}")
        if args.report:
            report = {
                "source": str(args.file),
                "budget": budget.__dict__,
                "metrics_before": result.metrics_before,
                "metrics_after": result.metrics_after,
                "steps": [{"action": s.step.action,
                           "rationale": s.step.rationale,
                           "note": s.note,
                           "before": s.before, "after": s.after}
                          for s in result.steps],
                "verification": v,
            }
            Path(args.report).write_text(
                json.dumps(report, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8")
            print(f"  报告 → {args.report}")
    return 0 if result.verification["passed"] else 2


def cmd_lod(args) -> int:
    ratios = tuple(float(x) for x in args.ratios.split(",") if x.strip())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in load(args.file):
        print(f"[{m.name}] 生成 LOD 链：ratios={ratios}")
        chain = lod_mod.generate_chain(m, ratios, args.chamfer)
        for lvl in chain.levels:
            flag = "" if lvl.accepted else "  ← 质量不达标，已拒绝"
            print(f"  LOD{lvl.level}: {lvl.mesh.n_faces} 面 "
                  f"(ratio={lvl.face_ratio:.3f}, "
                  f"chamfer={lvl.chamfer_norm:.5f}){flag}")
            save(lvl.mesh, out_dir / f"{m.name}_lod{lvl.level}.obj")
        print(f"  已写出 → {out_dir}/")
    return 0


def cmd_demo(args) -> int:
    print("内置演示网格：")
    print(f"  icosphere(4)    {prims.icosphere(4).n_faces} 面（光滑球）")
    print(f"  noisy_sphere(4) {prims.noisy_sphere(4).n_faces} 面（扫描噪声球）")
    print(f"  torus(48,24)    {prims.torus().n_faces} 面（环面，亏格 1）")
    m = prims.noisy_sphere(4)
    print("\n诊断 noisy_sphere(4)：")
    _print_topo(m)
    print("\n以 web_mobile 预算优化：")
    budget = agent.Budget.for_platform("web_mobile", 0.01)
    result = agent.run(m, budget)
    for s in result.steps:
        print(f"  {s.step.action}: {s.note}")
    v = result.verification
    print(f"  验收 {'✓' if v['passed'] else '✗'}："
          f"{v['faces']}/{v['budget_faces']} 面，"
          f"偏差 {v['chamfer_norm']:.5f}/{v['budget_chamfer_norm']}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="meshpilot", description="自动驾驶式 3D 资产优化 Agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyze", help="拓扑体检")
    p.add_argument("file")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("optimize", help="诊断→处方→执行→验证")
    p.add_argument("file")
    p.add_argument("-o", "--out")
    p.add_argument("--platform", choices=list(agent.PLATFORM_BUDGETS))
    p.add_argument("--max-faces", type=int, default=25000)
    p.add_argument("--chamfer", type=float, default=0.01,
                   help="最大允许表面偏差（相对对角线）")
    p.add_argument("--report")
    p.set_defaults(fn=cmd_optimize)

    p = sub.add_parser("lod", help="生成 LOD 链")
    p.add_argument("file")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--ratios", default="1,0.5,0.25,0.125")
    p.add_argument("--chamfer", type=float, default=0.01)
    p.set_defaults(fn=cmd_lod)

    p = sub.add_parser("demo", help="内置演示")
    p.set_defaults(fn=cmd_demo)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
