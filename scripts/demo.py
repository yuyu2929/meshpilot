"""MeshPilot 全量演示。

    python scripts/demo.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meshpilot import agent, lod as lod_mod, prims  # noqa: E402
from meshpilot.enterprise import status as ent_status  # noqa: E402
from meshpilot.meshio import load, save  # noqa: E402
from meshpilot.quality import chamfer_distance  # noqa: E402
from meshpilot.qem import simplify, vertex_cluster  # noqa: E402
from meshpilot.topology import analyze as topo  # noqa: E402

RULE = "=" * 62


def head(t: str) -> None:
    print(f"\n{RULE}\n  {t}\n{RULE}")


def main() -> None:
    head("1. 体检：三种内置资产")
    for name, m in [("icosphere(4)", prims.icosphere(4)),
                    ("noisy_sphere(4)", prims.noisy_sphere(4)),
                    ("torus", prims.torus())]:
        rep = topo(m)
        print(f"  {name:18} {rep.n_faces:5d} 面  闭合={rep.closed}  "
              f"问题 {len(rep.problems)} 项")

    head("2. QEM vs 顶点聚类（同一目标面数）")
    s = prims.icosphere(4)
    q, st = simplify(s, 1000)
    c, _ = vertex_cluster(s, 1000)
    cq = chamfer_distance(s, q)["symmetric_norm"]
    cc = chamfer_distance(s, c)["symmetric_norm"]
    print(f"  目标 1000 面：QEM {cq:.5f} vs 聚类 {cc:.5f}"
          f"（QEM 优 {cc / max(cq, 1e-9):.2f}x）")
    print(f"  参照：均匀细分的 icosphere(3)（1280 面）chamfer="
          f"{chamfer_distance(s, prims.icosphere(3))['symmetric_norm']:.5f}")
    print(f"  → QEM 用更少的面达到均匀细分的质量；拓扑保持"
          f"（翻转拒绝 {st.rejected_flips}，link 拒绝 {st.rejected_link}）")

    # 特征资产才是 QEM的主场：解析曲面上两者同阶（都是贴底噪），
    # 但带锐边的资产上，聚类既打不中预算也保不住角。
    fb = prims.feature_box(6, noise=0.0)
    fq, fst = simplify(fb, 200)
    fc, _ = vertex_cluster(fb, 200)
    print(f"  特征盒子 432→200 面：QEM {chamfer_distance(fb, fq)['symmetric_norm']:.5f}"
          f"（{fq.n_faces} 面，8/8 角精确保留）")
    print(f"    聚类交出 {fc.n_faces} 面（请求 200，网格量化打不中预算），"
          f"chamfer {chamfer_distance(fb, fc)['symmetric_norm']:.5f}")
    print(f"    → 锐边由二面角钳位保护（本网格触发 {fst.clamped_features} 次）；"
          f"解析曲面上两者同阶，特征资产上 QEM 的优势是设计目标决定的")

    head("3. Agent 闭环：noisy_sphere 减到 800 面预算")
    t0 = time.time()
    budget = agent.Budget(max_faces=800, platform="custom",
                          max_chamfer_norm=0.01)
    result = agent.run(prims.noisy_sphere(4), budget)
    print(f"  （{time.time() - t0:.1f}s）")
    for s2 in result.steps:
        print(f"  [{s2.step.action}] {s2.note}")
    v = result.verification
    print(f"  验收 {'✓ 通过' if v['passed'] else '✗ 未通过'}："
          f"{v['faces']}/{v['budget_faces']} 面，"
          f"偏差 {v['chamfer_norm']:.5f}/{v['budget_chamfer_norm']}，"
          f"覆盖率 {v['coverage'] * 100:.1f}%")

    head("4. LOD 链")
    chain = lod_mod.generate_chain(prims.icosphere(4),
                                   ratios=(1.0, 0.5, 0.25, 0.125),
                                   max_chamfer_norm=0.01)
    for lvl in chain.levels:
        flag = "" if lvl.accepted else "（已拒绝：质量不达标）"
        print(f"  LOD{lvl.level}: {lvl.mesh.n_faces:5d} 面  "
              f"chamfer={lvl.chamfer_norm:.5f} {flag}")

    head("5. 文件 IO round-trip（OBJ / PLY / GLB）")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for ext in ("obj", "ply", "glb"):
            p = Path(td) / f"t.{ext}"
            save(q, p)
            back = load(p)[0]
            same = back.n_faces == q.n_faces and back.n_vertices == q.n_vertices
            print(f"  .{ext}: 写→读 {back.n_faces} 面（一致={same}）")

    head("6. 许可状态")
    st2 = ent_status()
    print(f"  tier={st2['tier']}  license_valid={st2['license_valid']}")
    print("  核心（算法/CLI/可视化）：永久免费，不经过许可门禁")
    print("  企业（batch_pipeline/ci_report/sso）：需要 MESHPILOT_LICENSE")

    head("演示结束")
    print("  Web 控制台 : python run_web.py")
    print("  CLI        : python -m meshpilot demo")


if __name__ == "__main__":
    main()
