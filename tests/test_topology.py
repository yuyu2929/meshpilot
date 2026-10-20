"""拓扑分析测试。"""
from __future__ import annotations

import numpy as np

from meshpilot import prims
from meshpilot.meshio import Mesh
from meshpilot.topology import analyze, remove_degenerate


def test_icosphere_is_clean_closed():
    rep = analyze(prims.icosphere(3))
    assert rep.closed
    assert rep.nonmanifold_edges == 0
    assert rep.boundary_edges == 0
    assert rep.degenerate_faces == 0
    assert rep.euler_characteristic == 2   # 球面


def test_torus_genus_one():
    rep = analyze(prims.torus())
    assert rep.closed
    assert rep.euler_characteristic == 0   # 亏格 1


def test_nonmanifold_detection():
    # 三个面共享同一条边 (0,1)
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],      # 0,1,2
        [0, 0, 1], [0, 0, -1],                # 3,4
    ], dtype=float)
    faces = np.array([
        [0, 1, 2],   # 用边 (0,1)
        [0, 1, 3],   # 再用边 (0,1)
        [0, 1, 4],   # 三用边 (0,1) → 非流形
    ], dtype=np.int32)
    rep = analyze(Mesh(verts, faces))
    assert rep.nonmanifold_edges >= 1
    assert not rep.closed


def test_degenerate_detection_and_removal():
    verts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([
        [0, 1, 2],   # 共线 → 零面积
        [0, 1, 1],   # 重复顶点
        [0, 1, 3],   # 正常
    ], dtype=np.int32)
    rep = analyze(Mesh(verts, faces))
    assert rep.degenerate_faces == 2
    cleaned, removed = remove_degenerate(Mesh(verts, faces))
    assert removed == 2
    assert cleaned.n_faces == 1


def test_weldable_detection():
    dup = prims.duplicated_vertices_mesh(2)
    rep = analyze(dup)
    assert rep.weldable_vertices > 0


def test_boundary_edges_open_mesh():
    # 单个三角形：3 条边界边
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    rep = analyze(Mesh(verts, np.array([[0, 1, 2]], dtype=np.int32)))
    assert rep.boundary_edges == 3
    assert not rep.closed
