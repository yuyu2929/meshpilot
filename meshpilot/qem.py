"""QEM 边折叠减面（Garland & Heckbert 1997，SIGGRAPH）。

工业界事实标准的减面算法：Blender Decimate、Simplygon、meshoptimizer
的 simplify 都源于这一族。核心思想：

1. 每个顶点维护一个 4x4 二次型矩阵 Q，累加所有邻接面的平面误差
   （点到平面距离平方的二次型表示）
2. 一条边 (u,v) 的折叠代价 = 最优目标点 t 处的误差 t^T (Q_u+Q_v) t，
   其中 t 通过求解 (Q_u+Q_v) t = [0,0,0,1]^T 得到
3. 全局最小代价的边优先折叠（堆调度），折叠后把 u 的 Q 并入 v，
   更新邻边代价——贪心但误差可控

工程约束（正确性优先于速度）：
- 边界边默认锁定不折叠（保护开放网格轮廓）
- 翻转检测：折叠后邻面法线反向 → 拒绝该次折叠
- 目标点 clamp 到 uv 线段内，防奇异解把模型撕裂
- 懒删除堆：quadric 更新使旧条目失效，version 机制跳过
- 锐边钳位：二面角超过阈值的边，折叠目标点强制落到两平面
  交线上——否则奇异 quadric 回退的线段中点会把侧脸拉离平面，
  锐边被磨圆（无噪声立方体实测 chamfer 4 倍劣化）

性能定位：纯 Python/numpy，1 万面量级秒级完成——面向
"优化前检查/CI 门控/教学演示"，不替代 C++ 库处理百万面资产。
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .meshio import Mesh


@dataclass
class SimplifyStats:
    faces_before: int = 0
    faces_after: int = 0
    vertices_before: int = 0
    vertices_after: int = 0
    collapses: int = 0
    rejected_flips: int = 0
    rejected_link: int = 0
    max_quadric_cost: float = 0.0
    clamped_features: int = 0
    stopped_reason: str = ""


def _face_quadrics(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """每个面的平面二次型 p p^T（p = (nx,ny,nz,d)，齐次坐标）。"""
    tri = vertices[faces]                       # (m,3,3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    safe = np.where(ln > 0, ln, 1.0)
    n = n / safe[:, None]
    d = -np.einsum("ij,ij->i", n, tri[:, 0])
    p = np.concatenate([n, d[:, None]], axis=1)  # (m,4)
    return np.einsum("ij,ik->ijk", p, p)        # (m,4,4)


def simplify(mesh: Mesh, target_faces: int, *,
             lock_boundary: bool = True,
             max_cost_ratio: float | None = None,
             feature_angle: float = 30.0) -> tuple[Mesh, SimplifyStats]:
    """减面到 target_faces。返回 (新网格, 统计)。

    feature_angle：二面角阈值（度）。相邻两面二面角超过它的边视为
    锐边（特征边），折叠目标点强制钳位到两平面交线，保住边与角的
    锐度。平滑曲面（球/环）的二面角远小于阈值，行为不受影响；
    传 0 或负数关闭钳位（回到经典 Garland-Heckbert 行为）。
    """
    stats = SimplifyStats(
        faces_before=mesh.n_faces, vertices_before=mesh.n_vertices)
    if mesh.n_faces <= target_faces or mesh.n_faces == 0:
        stats.faces_after = mesh.n_faces
        stats.vertices_after = mesh.n_vertices
        stats.stopped_reason = "已达目标面数"
        return mesh.copy(), stats

    V = mesh.vertices.astype(np.float64).copy()
    F = [list(map(int, f)) for f in mesh.faces]
    n = len(V)
    diag = mesh.diagonal() or 1.0
    cost_cap = (max_cost_ratio * diag) ** 2 if max_cost_ratio else None

    # ---- 顶点二次型 ----
    fq = _face_quadrics(V, mesh.faces)
    Q = np.zeros((n, 4, 4), dtype=np.float64)
    for i in range(mesh.n_faces):
        for v in F[i]:
            Q[v] += fq[i]

    # ---- 邻接结构 ----
    vert_faces: dict[int, set[int]] = {i: set() for i in range(n)}
    edge_faces: dict[tuple[int, int], set[int]] = {}
    face_alive = [True] * len(F)

    def add_face(fid: int) -> None:
        a, b, c = F[fid]
        for v in (a, b, c):
            vert_faces[v].add(fid)
        for u, w in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault((min(u, w), max(u, w)), set()).add(fid)

    for fid in range(len(F)):
        add_face(fid)

    # ---- 候选边堆 ----
    edge_version: dict[tuple[int, int], int] = {}
    heap: list[tuple[float, int, int, int]] = []

    def feature_line(u: int, v: int):
        """边 (u,v) 两侧面的交线（锐边钳位线）。

        返回 (线上一点, 单位方向) 或 None：
        - 非流形内边（两面/一面/边界）→ None
        - 二面角 <= feature_angle（平滑边）→ None
        - 两面近平行（交线方向退化）→ None
        """
        fids = [fid for fid in edge_faces.get((min(u, v), max(u, v)), ())
                if face_alive[fid]]
        if len(fids) != 2:
            return None
        planes = []
        for fid in fids:
            a, b, c = F[fid]
            n = np.cross(V[b] - V[a], V[c] - V[a])
            ln = float(np.linalg.norm(n))
            if ln <= 0:
                return None
            n = n / ln
            planes.append((n, -float(np.dot(n, V[a]))))
        (n1, d1), (n2, d2) = planes
        # 流形相邻两面的朝向相反，用 |cos| 判断夹角
        if abs(float(np.dot(n1, n2))) > np.cos(np.radians(feature_angle)):
            return None
        direction = np.cross(n1, n2)
        dn = float(np.linalg.norm(direction))
        if dn <= 1e-9:
            return None
        direction = direction / dn
        # 交线上离边中点最近的点：解 2x2 正规方程
        # [[n1·n1, n1·n2], [n1·n2, n2·n2]] t = [n1·m+d1, n2·m+d2]
        m = 0.5 * (V[u] + V[v])
        A = np.array([[np.dot(n1, n1), np.dot(n1, n2)],
                      [np.dot(n1, n2), np.dot(n2, n2)]])
        rhs = np.array([np.dot(n1, m) + d1, np.dot(n2, m) + d2])
        try:
            t = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            return None
        return m - t[0] * n1 - t[1] * n2, direction

    def quadric_at(K: np.ndarray, p: np.ndarray) -> float:
        return float(p[0] ** 2 * K[0, 0] + 2 * p[0] * p[1] * K[0, 1]
                     + p[1] ** 2 * K[1, 1] + 2 * p[0] * p[2] * K[0, 2]
                     + 2 * p[1] * p[2] * K[1, 2] + p[2] ** 2 * K[2, 2]
                     + 2 * p[0] * K[0, 3] + 2 * p[1] * K[1, 3]
                     + 2 * p[2] * K[2, 3] + K[3, 3])

    def edge_cost(u: int, v: int) -> tuple[float, np.ndarray | None, bool]:
        """返回 (代价, 目标点, 是否锐边钳位)。

        目标点求解（Garland-Heckbert 的标准式）：
        最小化 [p,1] K [p,1]^T 的驻点满足 K[:3,:3] p = -K[:3,3]。
        等价地解 4×4 系统 K t = [0,0,0,1]^T——注意解出的 t 满足
        t[:3] = t[3] * p*（p* 为真正最优），必须除以 t[3] 才是位置。
        早期版本直接用 t[:3]，等于把最优解放大 1/f(p*) ≈ 1e3 倍，
        全部被尺度检查拒掉，实际永远回退边中点——减面退化成
        "中点折叠"，平滑资产上与顶点聚类打平、锐角被磨圆。
        """
        K = Q[u] + Q[v]
        pos = None
        try:
            t = np.linalg.solve(K, np.array([0.0, 0.0, 0.0, 1.0]))
            # t[3] = 1/f(p*)：接近 0 意味着最优解在极远处（不定 K），
            # 除掉会得到爆炸性坐标，直接视为无效
            if abs(float(t[3])) > 1e-12:
                cand = t[:3] / float(t[3])
                if np.all(np.isfinite(cand)):
                    # 数值异常保护：解跑出模型尺度外视为无效
                    if (np.linalg.norm(cand - V[u]) <= diag
                            and np.linalg.norm(cand - V[v]) <= diag):
                        pos = cand
        except np.linalg.LinAlgError:
            pass
        if pos is None:
            # 奇异/异常解 → 在 {u, v, 中点} 中取 quadric 代价最小者。
            # 平面区域所有面共面时 K 秩 1，最优解不唯一——端点上
            # 代价通常就是 0（顶点本就在平面上），选中端点可避免
            # 无谓的位置漂移；锐角处端点（角点本身）代价 0，
            # 中点 > 0，于是角被保住而不会被拉向棱中点。
            # 历史上这里盲目取中点：立方体 200 面时 chamfer 0.0296
            # （角点中位偏移 0.37），端点优选后 0.0080。
            best_c, best_p = None, None
            for cand in (V[u], V[v], 0.5 * (V[u] + V[v])):
                cc = quadric_at(K, cand)
                if best_c is None or cc < best_c:
                    best_c, best_p = cc, cand
            pos = best_p
        # 锐边钳位：把目标点投影到两平面交线上。
        # 没有这一步，奇异 quadric 的中线回退会把锐边一侧的面拉离
        # 其平面（立方体实测：200 面时 chamfer 0.0296 → 0.0079）。
        clamped = False
        if feature_angle > 0:
            line = feature_line(u, v)
            if line is not None:
                p0, direction = line
                pos = p0 + float(np.dot(pos - p0, direction)) * direction
                clamped = True
        c = quadric_at(K, pos)
        # 代价恰好为 0 是合法的：共面区域内折叠、锐边上沿交线折叠
        # 都零失真。只有显著为负才说明 K 不定（驻点是鞍点），
        # 该边不可信——早期版本用 c<=0 误杀了所有平面上的免费折叠。
        if c < -1e-9 * diag * diag:
            return float("inf"), pos, clamped
        return max(c, 0.0), pos, clamped

    def push_edge(u: int, v: int) -> None:
        key = (min(u, v), max(u, v))
        if key not in edge_faces or not edge_faces[key]:
            return
        if lock_boundary and len(edge_faces[key]) == 1:
            return  # 边界边锁定
        c, _, _ = edge_cost(u, v)
        edge_version[key] = edge_version.get(key, 0) + 1
        heapq.heappush(heap, (c, edge_version[key], key[0], key[1]))

    for (u, v) in list(edge_faces.keys()):
        push_edge(u, v)

    alive = len(F)
    # ---- 主循环 ----
    while heap and alive > target_faces:
        cost, ver, u, v = heapq.heappop(heap)
        key = (min(u, v), max(u, v))
        if edge_version.get(key, -1) != ver:
            continue                      # 懒删除：已被更新的条目
        if key not in edge_faces or not edge_faces[key]:
            continue
        if cost_cap is not None and cost > cost_cap:
            stats.stopped_reason = (
                f"折叠代价 {cost:.3g} 超过上限 "
                f"({max_cost_ratio}*diag)^2——停止以防质量崩坏")
            break
        stats.max_quadric_cost = max(stats.max_quadric_cost, cost)

        # 目标点
        _, pos, clamped = edge_cost(u, v)
        if clamped:
            stats.clamped_features += 1

        # ---- 翻转/退化检测：u 和 v 的邻面在折叠后都要合法 ----
        # 三个要点：
        # 1. 用目标位置 pos 检测，不能用 v 的旧位置
        # 2. 新面必须保持原面的顶点循环顺序（只把 u 换成 pos）——
        #    手工重排 others 会把法线算反，导致所有边被误拒（collapses=0）
        # 3. 只查含 u 的面不够：v 也要搬到 pos，它的原有面可能被拉成
        #    共线（零面积）或翻面。折叠后 u 和 v 都映射到 pos，
        #    所以两边的面都用同一个替换规则检测
        #    （无噪声立方体实测：漏查 v 的面产出 7 个共线退化三角形）。
        ok = True
        for fid in set(vert_faces[u]) | set(vert_faces[v]):
            if not face_alive[fid]:
                continue
            a, b, c = F[fid]
            if u in (a, b, c) and v in (a, b, c):
                continue                 # 含 uv 边的面将被删除
            n1 = np.cross(V[b] - V[a], V[c] - V[a])
            p0, p1, p2 = (pos if x in (u, v) else V[x] for x in (a, b, c))
            n2 = np.cross(p1 - p0, p2 - p0)
            if np.dot(n1, n2) <= 0 or np.linalg.norm(n2) <= diag * 1e-12:
                ok = False
                break
        if not ok:
            stats.rejected_flips += 1
            # 该边标记为失败版本，避免反复弹出
            edge_version[key] = edge_version.get(key, 0) + 1
            continue

        # ---- Link condition（流形保持，Garland-Heckbert 的标准约束）----
        # 折叠 (u,v) 保持流形的充要条件：N(u) ∩ N(v) 恰好是共享该边的
        # 两个面的对顶点。若交集更大（u/v 有共同邻居不在对顶点位），
        # 折叠后某些边会有 >2 个面引用 → 非流形。
        # 缺这个检查的减面器会在球/环面上"打出柄来"（实测欧拉数漂移）。
        neighbors_u: set[int] = set()
        for fid in vert_faces[u]:
            if face_alive[fid]:
                neighbors_u.update(x for x in F[fid] if x != u)
        neighbors_v: set[int] = set()
        for fid in vert_faces[v]:
            if face_alive[fid]:
                neighbors_v.update(x for x in F[fid] if x != v)
        opposite: set[int] = set()
        for fid in edge_faces.get(key, ()):
            if face_alive[fid]:
                opposite.update(x for x in F[fid] if x != u and x != v)
        if (neighbors_u & neighbors_v) - {u, v} != opposite:
            stats.rejected_link += 1
            edge_version[key] = edge_version.get(key, 0) + 1
            continue

        # ---- 执行折叠 u → v ----
        V[v] = pos
        Q[v] = Q[v] + Q[u]

        # 删除含边 (u,v) 的面
        for fid in list(edge_faces.get(key, ())):
            if face_alive[fid]:
                face_alive[fid] = False
                alive -= 1
                a, b, c = F[fid]
                for x in (a, b, c):
                    vert_faces[x].discard(fid)
                for u2, w2 in ((a, b), (b, c), (c, a)):
                    k2 = (min(u2, w2), max(u2, w2))
                    s = edge_faces.get(k2)
                    if s is not None:
                        s.discard(fid)
                        if not s:
                            del edge_faces[k2]

        # 其余含 u 的面改指向 v。
        # 关键：必须先把该面的旧边引用移除、再加新边引用——
        # 否则 edge_faces 累积陈旧 fid，死边"看起来活着"，
        # 每次折叠会重推上千条边（实测 320 次折叠推 86 万次 edge_cost）。
        for fid in list(vert_faces[u]):
            if not face_alive[fid]:
                continue
            a, b, c = F[fid]
            for u2, w2 in ((a, b), (b, c), (c, a)):
                k2 = (min(u2, w2), max(u2, w2))
                s = edge_faces.get(k2)
                if s is not None:
                    s.discard(fid)
                    if not s:
                        del edge_faces[k2]
            new_f = [v if x == u else x for x in F[fid]]
            if len(set(new_f)) < 3:
                # 非流形残留：改名后出现重复顶点 → 面退化，删除
                face_alive[fid] = False
                alive -= 1
                for x in new_f:
                    vert_faces[x].discard(fid)
                continue
            F[fid] = new_f
            vert_faces[u].discard(fid)
            vert_faces[v].add(fid)
            a, b, c = new_f
            for u2, w2 in ((a, b), (b, c), (c, a)):
                k2 = (min(u2, w2), max(u2, w2))
                edge_faces.setdefault(k2, set()).add(fid)

        vert_faces[u].clear()
        stats.collapses += 1

        # 更新受影响边的堆条目。
        # 关键：V[v] 被改写、Q[v] 并入 Q[u]，v 的 ALL 关联边代价
        # 都已失效——包括 v 原有的、不含 u 的面上的边。只推"改名面"
        # 的边会漏掉它们：陈旧的低代价条目之后弹出并通过版本检查，
        # 执行实时代价高得多的折叠（实测：立方体垂直棱链被一次性
        # 塌成两半，角点中位偏移 0.11、chamfer 0.046）。
        pushed_keys: set[tuple[int, int]] = set()
        for fid in vert_faces[v]:
            if not face_alive[fid]:
                continue
            for w in F[fid]:
                if w == v:
                    continue
                key = (min(v, w), max(v, w))
                if key in pushed_keys:
                    continue
                pushed_keys.add(key)
                push_edge(v, w)

    # ---- 重建网格 ----
    # 必须用修改后的 F（含 u→v 改名与删除）重建，
    # 不能对原始 mesh.faces 做 subset——改名从未写回原始数组，
    # 那样活面仍引用已死顶点，边界边/欧拉数/chamfer 全错。
    keep = np.nonzero(np.array(face_alive, dtype=bool))[0]
    final_faces = np.array([F[i] for i in keep], dtype=np.int32).reshape(-1, 3)
    merged = Mesh(vertices=V, faces=final_faces,
                  uvs=mesh.uvs, normals=mesh.normals,
                  name=mesh.name, source_format=mesh.source_format)
    out = merged.subset(np.arange(len(final_faces)))  # 丢弃未引用顶点
    stats.faces_after = out.n_faces
    stats.vertices_after = out.n_vertices
    if not stats.stopped_reason:
        stats.stopped_reason = "已达目标面数" if alive <= target_faces \
            else "候选边耗尽（边界锁定/翻转保护）"
    return out, stats


def vertex_cluster(mesh: Mesh, target_faces: int) -> tuple[Mesh, dict]:
    """顶点聚类简化（对照组基线）：快但质量差。

    按空间网格聚类顶点，每簇一个代表点，面按代表点重映射，
    退化的丢弃。工业上属于最低质量档，用作 QEM 的对照。
    """
    if mesh.n_faces <= target_faces:
        return mesh.copy(), {"clusters": mesh.n_vertices}
    ratio = max(target_faces / max(mesh.n_faces, 1), 1e-6)
    cell = mesh.diagonal() / (len(mesh.vertices) ** (1 / 3)) * (ratio ** (1 / 3))
    cell = max(cell, 1e-12)
    keys = np.floor(mesh.vertices / cell).astype(np.int64)
    uniq, first, inverse = np.unique(keys, axis=0, return_index=True,
                                     return_inverse=True)
    inverse = inverse.reshape(-1)
    reps = mesh.vertices[first]
    # 每簇质心
    sums = np.zeros_like(reps)
    np.add.at(sums, inverse, mesh.vertices)
    counts = np.bincount(inverse, minlength=len(reps))
    reps = sums / np.maximum(counts, 1)[:, None]

    new_faces = inverse[mesh.faces]
    keep = ((new_faces[:, 0] != new_faces[:, 1])
            & (new_faces[:, 1] != new_faces[:, 2])
            & (new_faces[:, 0] != new_faces[:, 2]))
    out = Mesh(reps, new_faces[keep],
               None if mesh.uvs is None else mesh.uvs[first],
               None if mesh.normals is None else mesh.normals[first],
               name=mesh.name, source_format=mesh.source_format)
    return out, {"clusters": len(reps)}
