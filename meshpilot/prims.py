"""程序化测试网格生成器。

存在的理由：整个仓库（测试 + demo + Web）不依赖任何外部 .obj 资源文件——
自包含，clone 即可跑。生成的网格带可控高频细节（噪声），
让减面算法有真实的活干。
"""
from __future__ import annotations

import numpy as np

from .meshio import Mesh


def icosphere(subdivisions: int = 3, radius: float = 1.0) -> Mesh:
    """标准 icosphere 细分。极点无退化三角，适合测减面。"""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts[0])
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)

    for _ in range(subdivisions):
        verts, faces = _subdivide(verts, faces)

    return Mesh(verts * radius, faces.astype(np.int32), name="icosphere")


def _subdivide(verts: np.ndarray, faces: np.ndarray):
    cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key in cache:
            return cache[key]
        m = 0.5 * (verts[a] + verts[b])
        m /= np.linalg.norm(m)
        verts_list.append(m)
        idx = len(verts_list) - 1
        cache[key] = idx
        return idx

    verts_list = [v for v in verts]
    new_faces = []
    for a, b, c in faces:
        ab = midpoint(int(a), int(b))
        bc = midpoint(int(b), int(c))
        ca = midpoint(int(c), int(a))
        new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    return np.array(verts_list), np.array(new_faces, dtype=np.int64)


def noisy_sphere(subdivisions: int = 4, radius: float = 1.0,
                 amplitude: float = 0.15, freq: float = 6.0,
                 seed: int = 7) -> Mesh:
    """带高频径向噪声的球——模拟"扫描来的脏资产"。"""
    base = icosphere(subdivisions, radius)
    rng = np.random.default_rng(seed)
    # 多频正弦叠加：确定性的"扫描噪声"
    v = base.vertices
    phase = rng.random(3) * 2 * np.pi
    noise = np.zeros(len(v))
    for k in range(1, 5):
        w = freq * k
        noise += (np.sin(w * v[:, 0] + phase[0])
                  + np.sin(w * v[:, 1] + phase[1])
                  + np.sin(w * v[:, 2] + phase[2])) / (k * 3.0)
    noise /= np.abs(noise).max() + 1e-12
    direction = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    out = v + direction * (noise * amplitude)[:, None]
    return Mesh(out, base.faces.copy(), name="noisy_sphere")


def torus(major_segments: int = 48, minor_segments: int = 24,
          major: float = 1.0, minor: float = 0.3) -> Mesh:
    """环面——固定拓扑（亏格 1），测减面保持拓扑的能力。"""
    u = np.linspace(0, 2 * np.pi, major_segments, endpoint=False)
    v = np.linspace(0, 2 * np.pi, minor_segments, endpoint=False)
    verts = []
    for cu in u:
        for cv in v:
            x = (major + minor * np.cos(cv)) * np.cos(cu)
            y = (major + minor * np.cos(cv)) * np.sin(cu)
            z = minor * np.sin(cv)
            verts.append((x, y, z))
    verts = np.array(verts)
    faces = []
    for i in range(major_segments):
        for j in range(minor_segments):
            a = i * minor_segments + j
            b = i * minor_segments + (j + 1) % minor_segments
            c = ((i + 1) % major_segments) * minor_segments + (j + 1) % minor_segments
            d = ((i + 1) % major_segments) * minor_segments + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    return Mesh(verts, np.array(faces, dtype=np.int32), name="torus")


def duplicated_vertices_mesh(subdivisions: int = 3) -> Mesh:
    """把 icosphere 的每个顶点复制成 3 个（模拟糟糕导出的资产），
    面索引指向复制品——weld 的专用测试资产。"""
    base = icosphere(subdivisions)
    n = len(base.vertices)
    verts = np.concatenate([base.vertices] * 3)
    faces = base.faces.copy()
    for k in range(3):
        part = base.faces + k * n
        if k == 0:
            faces = part
        else:
            faces = np.concatenate([faces, part])
    return Mesh(verts, faces.astype(np.int32), name="duplicated")


def feature_box(subdiv: int = 6, size: float = 1.0,
                noise: float = 0.01, seed: int = 3) -> Mesh:
    """带锐边的盒子：6 个平面 + 轻微噪声。

    这是 QEM 的主场：平面区域应该被大块合并（quadric 误差≈0），
    12 条锐边与 8 个角被保留。顶点聚类会把角磨圆、把边糊掉——
    在解析曲面（球/环）上两者打平甚至聚类反超，
    但特征资产上 QEM 的优势是设计目标决定的。
    """
    rng = np.random.default_rng(seed)
    faces_list: list[np.ndarray] = []
    verts_list: list[np.ndarray] = []

    # 6 个面：(法向, u 轴, v 轴, 原点)
    axes = [
        ((0, 0, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)),    # +z
        ((0, 0, -1), (1, 0, 0), (0, 1, 0), (0, 0, -1)),  # -z
        ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)),    # +x
        ((-1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0)),  # -x
        ((0, 1, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0)),    # +y
        ((0, -1, 0), (1, 0, 0), (0, 0, 1), (0, -1, 0)),  # -y
    ]
    for n, u, v, o in axes:
        n = np.array(n, dtype=float)
        u = np.array(u, dtype=float)
        v = np.array(v, dtype=float)
        o = np.array(o, dtype=float) * size
        grid = []
        for i in range(subdiv + 1):
            for j in range(subdiv + 1):
                p = (o + u * (-size + 2 * size * i / subdiv)
                     + v * (-size + 2 * size * j / subdiv))
                if noise and 0 < i < subdiv and 0 < j < subdiv:
                    p = p + rng.normal(0, noise, 3)
                grid.append(p)
        base = len(verts_list) if verts_list else 0
        verts_list.extend(grid)
        idx = lambda a, b: base + a * (subdiv + 1) + b
        for i in range(subdiv):
            for j in range(subdiv):
                faces_list.append(np.array(
                    [idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]))
                faces_list.append(np.array(
                    [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]))

    m = Mesh(np.array(verts_list), np.array(faces_list, dtype=np.int32),
             name="feature_box")
    from . import weld as weld_mod
    welded, _ = weld_mod.weld(m)   # 合并 6 面接缝的重复顶点
    return welded
