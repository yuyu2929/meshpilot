"""LOD 链测试。"""
from __future__ import annotations

from meshpilot import lod, prims
from meshpilot.topology import analyze


def test_chain_generation():
    chain = lod.generate_chain(prims.icosphere(3),
                               ratios=(1.0, 0.5, 0.25),
                               max_chamfer_norm=0.01)
    assert len(chain.levels) == 3
    faces = [l.mesh.n_faces for l in chain.levels]
    assert faces[0] == prims.icosphere(3).n_faces
    assert faces[1] < faces[0]
    assert faces[2] < faces[1]
    assert all(l.accepted for l in chain.levels)


def test_chain_rejects_bad_quality():
    # 极严阈值 → 深层 LOD 应被拒绝并终止链条
    chain = lod.generate_chain(prims.icosphere(3),
                               ratios=(1.0, 0.5, 0.25, 0.125, 0.0625),
                               max_chamfer_norm=0.0001)
    rejected = [l for l in chain.levels if not l.accepted]
    assert rejected, "极严阈值下应有 LOD 被拒绝"
    # 拒绝级之后不再有接受级
    first_bad = next(i for i, l in enumerate(chain.levels) if not l.accepted)
    assert all(not l.accepted for l in chain.levels[first_bad:])


def test_chain_levels_are_valid_meshes():
    chain = lod.generate_chain(prims.icosphere(2),
                               ratios=(1.0, 0.5, 0.25))
    for lvl in chain.levels:
        rep = analyze(lvl.mesh)
        assert rep.nonmanifold_edges == 0
        assert rep.degenerate_faces == 0
