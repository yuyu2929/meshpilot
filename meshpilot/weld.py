"""顶点合并（weld）：把坐标相同的顶点合并成一个。

这是最安全、零质量损失的优化：GPU 顶点缓存按索引寻址，
重复顶点让顶点缓冲区白白变大。合并后面数不变、几何不变，
显存与 draw call 前的数据量直接下降。
"""
from __future__ import annotations

import numpy as np

from .meshio import Mesh


def weld(mesh: Mesh, eps: float = 1e-6) -> tuple[Mesh, dict]:
    """量化合并重复顶点。eps 相对模型对角线归一化。

    返回 (新网格, 统计)。属性（uv/normal）取合并组内首个顶点的值。
    """
    if len(mesh.vertices) == 0:
        return mesh.copy(), {"before": 0, "after": 0, "merged": 0}

    scale = float(np.linalg.norm(mesh.vertices.max(axis=0)
                                 - mesh.vertices.min(axis=0))) or 1.0
    q = np.round(mesh.vertices / (eps * scale)).astype(np.int64)
    # 用字典把量化坐标映射到代表顶点
    unique_keys, first_idx, inverse = _unique_first(q)

    # inverse[i] 是顶点 i 所属的组号；新顶点数组按组号排序排列
    # （mesh.vertices[first_idx] 的第 g 项即组 g 的代表），
    # 所以旧索引 → 新索引的映射就是 inverse 本身。
    # 注意不能用 first_idx[inverse]：那是代表顶点在"原数组"中的位置，
    # 仅当 first_idx == arange(n) 时碰巧等于新位置。
    remap = inverse.astype(np.int32)   # 旧顶点 → 新顶点
    new_faces = remap[mesh.faces]

    new_mesh = Mesh(
        vertices=mesh.vertices[first_idx],
        faces=new_faces,
        uvs=None if mesh.uvs is None else mesh.uvs[first_idx],
        normals=None if mesh.normals is None else mesh.normals[first_idx],
        name=mesh.name,
        source_format=mesh.source_format,
    )
    stats = {
        "before": mesh.n_vertices,
        "after": new_mesh.n_vertices,
        "merged": mesh.n_vertices - new_mesh.n_vertices,
    }
    return new_mesh, stats


def _unique_first(q: np.ndarray):
    """量化坐标去重，返回 (unique_keys, first_idx, inverse)。

    与 np.unique(axis=0, return_index=True) 的差异：保留"首次出现"的
    代表顶点而不是排序后的代表，属性归属更符合直觉（可选）。
    实现上直接用 np.unique 的 return_index 即可，语义等价。
    """
    keys, first_idx, inverse = np.unique(
        q, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    return keys, first_idx, inverse
