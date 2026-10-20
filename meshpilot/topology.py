"""拓扑分析：把网格的"健康状况"变成可执行的诊断数据。

诊断项（都有明确工程含义）：
- 非流形边：>2 个面共享的边，GPU 与布尔运算都会出问题
- 边界边：仅 1 个面引用的边，开放网格标记
- 退化三角形：零面积 / 重复顶点，渲染与减面都会翻车
- 重复面：完全相同的三角形
- 孤立顶点：不被任何面引用的顶点，纯浪费显存
- 重复顶点（可合并）：坐标相同的不同顶点，weld 的收益来源
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .meshio import Mesh


@dataclass
class TopologyReport:
    n_vertices: int = 0
    n_faces: int = 0
    nonmanifold_edges: int = 0
    boundary_edges: int = 0
    degenerate_faces: int = 0
    duplicate_faces: int = 0
    orphan_vertices: int = 0
    weldable_vertices: int = 0
    euler_characteristic: int = 0
    closed: bool = False
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_vertices": self.n_vertices,
            "n_faces": self.n_faces,
            "nonmanifold_edges": self.nonmanifold_edges,
            "boundary_edges": self.boundary_edges,
            "degenerate_faces": self.degenerate_faces,
            "duplicate_faces": self.duplicate_faces,
            "orphan_vertices": self.orphan_vertices,
            "weldable_vertices": self.weldable_vertices,
            "euler_characteristic": self.euler_characteristic,
            "closed": self.closed,
            "problems": list(self.problems),
        }


def edge_counts(faces: np.ndarray) -> dict[tuple[int, int], int]:
    """无向边 → 引用面数。"""
    counts: dict[tuple[int, int], int] = {}
    f = faces
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    for a, b in edges:
        key = (int(a), int(b))
        counts[key] = counts.get(key, 0) + 1
    return counts


def analyze(mesh: Mesh) -> TopologyReport:
    rep = TopologyReport(n_vertices=mesh.n_vertices, n_faces=mesh.n_faces)
    if mesh.n_faces == 0:
        rep.problems.append("空网格：没有面")
        return rep

    counts = edge_counts(mesh.faces)
    rep.nonmanifold_edges = sum(1 for c in counts.values() if c > 2)
    rep.boundary_edges = sum(1 for c in counts.values() if c == 1)
    rep.closed = rep.boundary_edges == 0 and rep.nonmanifold_edges == 0

    # 退化：重复顶点索引或零面积
    f = mesh.faces
    dup_idx = ((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2])
               | (f[:, 0] == f[:, 2]))
    tri = mesh.face_vertices()
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    diag = max(mesh.diagonal(), 1e-12)
    zero_area = area2 <= (diag * 1e-9)
    rep.degenerate_faces = int(np.count_nonzero(dup_idx | zero_area))

    # 重复面（顶点集合相同）
    key = np.sort(f, axis=1)
    _, counts_faces = np.unique(key, axis=0, return_counts=True)
    dup_counts = counts_faces[counts_faces > 1]
    rep.duplicate_faces = int((dup_counts - 1).sum()) if len(dup_counts) else 0

    used = np.zeros(mesh.n_vertices, dtype=bool)
    used[f.reshape(-1)] = True
    rep.orphan_vertices = int(np.count_nonzero(~used))

    # 可合并顶点：量化坐标哈希
    rep.weldable_vertices = _count_weldable(mesh.vertices)

    rep.euler_characteristic = (mesh.n_vertices - len(counts)
                                + mesh.n_faces)

    if rep.nonmanifold_edges:
        rep.problems.append(
            f"{rep.nonmanifold_edges} 条非流形边（>2 面共享），"
            "GPU 渲染与后续处理都会出问题")
    if rep.degenerate_faces:
        rep.problems.append(f"{rep.degenerate_faces} 个退化三角形")
    if rep.duplicate_faces:
        rep.problems.append(f"{rep.duplicate_faces} 个重复面")
    if rep.orphan_vertices:
        rep.problems.append(f"{rep.orphan_vertices} 个孤立顶点（纯浪费显存）")
    if rep.weldable_vertices:
        rep.problems.append(
            f"{rep.weldable_vertices} 个顶点可合并（weld 可直接减少显存）")
    if not rep.closed:
        rep.problems.append(
            f"开放网格：{rep.boundary_edges} 条边界边"
            "（若用于 3D 打印/布尔运算需先修复）")
    return rep


def _count_weldable(vertices: np.ndarray, eps: float = 1e-6) -> int:
    """量化到 eps 网格后统计重复顶点数（相对坐标尺度归一化）。"""
    if len(vertices) == 0:
        return 0
    scale = float(np.linalg.norm(vertices.max(axis=0)
                                 - vertices.min(axis=0))) or 1.0
    q = np.round(vertices / (eps * scale)).astype(np.int64)
    _, inverse, counts = np.unique(q, axis=0, return_inverse=True,
                                   return_counts=True)
    return int(np.count_nonzero(counts[inverse] > 1))


def degenerate_mask(mesh: Mesh) -> np.ndarray:
    """返回退化面掩码（供清理用）。"""
    f = mesh.faces
    dup_idx = ((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2])
               | (f[:, 0] == f[:, 2]))
    tri = mesh.face_vertices()
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    diag = max(mesh.diagonal(), 1e-12)
    return dup_idx | (area2 <= diag * 1e-9)


def remove_degenerate(mesh: Mesh) -> tuple[Mesh, int]:
    mask = ~degenerate_mask(mesh)
    removed = int(np.count_nonzero(~mask))
    return mesh.subset(np.nonzero(mask)[0]), removed
