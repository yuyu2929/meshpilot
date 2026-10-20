"""QEM 减面测试：面数达标、拓扑保持、质量有界、优于聚类。"""
from __future__ import annotations

import numpy as np

from meshpilot import prims
from meshpilot.quality import chamfer_distance
from meshpilot.qem import simplify, vertex_cluster
from meshpilot.topology import analyze


def test_face_count_target():
    s = prims.icosphere(3)          # 1280 面
    red, st = simplify(s, 400)
    assert red.n_faces <= 400
    assert st.faces_after == red.n_faces
    assert st.collapses > 0


def test_topology_preserved_sphere():
    s = prims.icosphere(3)
    red, _ = simplify(s, 400)
    rep = analyze(red)
    assert rep.euler_characteristic == 2
    assert rep.boundary_edges == 0
    assert rep.nonmanifold_edges == 0
    assert rep.closed
    assert rep.degenerate_faces == 0


def test_topology_preserved_torus():
    t = prims.torus(24, 12)         # 576 面
    red, _ = simplify(t, 200)
    rep = analyze(red)
    assert rep.euler_characteristic == 0
    assert rep.boundary_edges == 0
    assert rep.nonmanifold_edges == 0


def test_chamfer_bounded():
    s = prims.icosphere(3)
    red, _ = simplify(s, 400)
    ch = chamfer_distance(s, red, n_samples=1500)
    # 温和减面（1280→400）表面偏差应远小于对角线的 2%
    assert ch["symmetric_norm"] < 0.02


def test_qem_beats_clustering():
    s = prims.icosphere(3)
    q, _ = simplify(s, 400)
    c, _ = vertex_cluster(s, 400)
    cq = chamfer_distance(s, q, n_samples=1500)["symmetric_norm"]
    cc = chamfer_distance(s, c, n_samples=1500)["symmetric_norm"]
    assert cq < cc, f"QEM {cq} 应优于聚类 {cc}"


def test_noop_when_within_target():
    s = prims.icosphere(2)
    red, st = simplify(s, 10_000)
    assert red.n_faces == s.n_faces
    assert st.collapses == 0


def test_duplicated_vertices_mesh_survives():
    dup = prims.duplicated_vertices_mesh(2)
    red, st = simplify(dup, 100)
    assert red.n_faces <= 100
    rep = analyze(red)
    assert rep.nonmanifold_edges == 0


def test_cost_cap_stops_conservatively():
    s = prims.icosphere(3)
    red, st = simplify(s, 50, max_cost_ratio=0.001)
    assert red.n_faces > 50      # 代价门控提前停止
    assert st.stopped_reason
    assert "代价" in st.stopped_reason or "目标" in st.stopped_reason


# ---- 特征资产回归（三个历史 bug 的守护测试）----
# bug 1：weld 的 remap 用 first_idx[inverse]（代表顶点的原数组位置）
#        而非组号，稀疏重复时索引越界——feature_box 的构造直接踩中。
# bug 2：4×4 齐次解没除 t[3]，最优目标点被放大 ~1e3 倍后全部被
#        尺度检查拒绝，减面退化成"中点折叠"。
# bug 3：折叠后只重推改名面的边，v 的原有邻边留下陈旧低代价条目，
#        之后弹出执行高代价折叠——棱链被一次性塌掉。
def test_feature_box_builds_welded():
    fb = prims.feature_box(6)
    assert fb.n_faces == 432                 # 6 面 x 6x6x2 三角
    assert fb.n_vertices == 218              # 294 - 76 个接缝重复顶点
    rep = analyze(fb)
    assert rep.closed and rep.euler_characteristic == 2


def test_sharp_features_preserved():
    fb = prims.feature_box(6, noise=0.0)     # 无噪声：几何精确可知
    red, st = simplify(fb, 200)
    # 预算精确命中
    assert red.n_faces == 200
    # 质量贴采样底噪（修复前：0.0296，角点中位偏移 0.37）
    ch = chamfer_distance(fb, red)
    assert ch["symmetric_norm"] < 0.012
    # 8 个角点精确保留（修复前：全部磨掉）
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                        for z in (-1, 1)], dtype=float)
    for c in corners:
        assert np.linalg.norm(red.vertices - c, axis=1).min() < 1e-9
    # 拓扑保持
    rep = analyze(red)
    assert rep.euler_characteristic == 2
    assert rep.closed and rep.boundary_edges == 0
    assert rep.nonmanifold_edges == 0
    assert rep.degenerate_faces == 0
    # 锐边钳位确实在工作
    assert st.clamped_features > 0


def test_clustering_cannot_hit_budget_on_features():
    """对照组：顶点聚类的网格量化无法精确命中面数预算。"""
    fb = prims.feature_box(6, noise=0.0)
    red, _ = vertex_cluster(fb, 200)
    assert red.n_faces != 200                # 108 或 302，唯独不是 200
