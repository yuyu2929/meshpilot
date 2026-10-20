"""meshio 测试：round-trip 与边界。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from meshpilot import prims
from meshpilot.meshio import Mesh, MeshError, load, save


def test_obj_roundtrip():
    m = prims.icosphere(2)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.obj"
        save(m, p)
        back = load(p)[0]
    assert back.n_faces == m.n_faces
    assert back.n_vertices == m.n_vertices
    assert back.vertices.shape == m.vertices.shape
    # 几何一致（OBJ 是 ascii 6 位小数）
    assert (abs(back.vertices - m.vertices) < 1e-5).all()


def test_ply_roundtrip():
    m = prims.icosphere(2)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.ply"
        save(m, p)
        back = load(p)[0]
    assert back.n_faces == m.n_faces
    assert (abs(back.vertices - m.vertices) < 1e-5).all()


def test_glb_roundtrip():
    m = prims.icosphere(2)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.glb"
        save(m, p)
        back = load(p)[0]
    assert back.n_faces == m.n_faces
    assert back.n_vertices == m.n_vertices
    assert (abs(back.vertices - m.vertices) < 1e-5).all()


def test_mesh_rejects_bad_faces():
    with pytest.raises(MeshError):
        Mesh(vertices=[[0, 0, 0], [1, 0, 0]], faces=[[0, 1, 5]])


def test_unsupported_format():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.stl"
        p.write_text("solid x")
        with pytest.raises(MeshError):
            load(p)


def test_empty_mesh_passthrough():
    m = Mesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
    assert m.n_faces == 1
    rep = prims.icosphere(1).subset([])
    assert rep.n_faces == 0
