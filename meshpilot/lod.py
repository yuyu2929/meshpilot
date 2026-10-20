"""LOD 链生成：一次减面，多级细节。

游戏引擎的标准做法：LOD0（原始）→ LOD1 → LOD2 → LOD3，
按视距切换。本模块从同一网格逐级减面生成链条，每级独立验证，
任何一级质量不达标即从链中剔除并记录——不把烂 LOD 塞进流水线。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .meshio import Mesh
from .qem import simplify
from .quality import chamfer_distance


@dataclass
class LODLevel:
    level: int
    mesh: Mesh
    face_ratio: float
    chamfer_norm: float
    accepted: bool = True


@dataclass
class LODChain:
    levels: list[LODLevel] = field(default_factory=list)
    source_faces: int = 0

    def to_dict(self) -> list[dict]:
        return [{
            "level": l.level,
            "faces": l.mesh.n_faces,
            "vertices": l.mesh.n_vertices,
            "face_ratio": round(l.face_ratio, 4),
            "chamfer_norm": round(l.chamfer_norm, 6),
            "accepted": l.accepted,
        } for l in self.levels]


DEFAULT_RATIOS = (1.0, 0.5, 0.25, 0.125, 0.0625)


def generate_chain(mesh: Mesh, ratios: tuple[float, ...] = DEFAULT_RATIOS,
                   max_chamfer_norm: float = 0.005) -> LODChain:
    """从原网格逐级减面。每级在上一级基础上继续减（渐进网格语义）。"""
    chain = LODChain(source_faces=mesh.n_faces)
    current = mesh
    for i, ratio in enumerate(ratios):
        target = max(int(mesh.n_faces * ratio), 4)
        if target >= current.n_faces:
            lvl = LODLevel(i, current.copy(), current.n_faces / mesh.n_faces,
                           0.0)
            chain.levels.append(lvl)
            continue
        reduced, stats = simplify(current, target)
        ch = chamfer_distance(mesh, reduced)
        accepted = ch["symmetric_norm"] <= max_chamfer_norm
        if not accepted:
            # 质量崩坏：保留上一级，标记拒绝，终止链条
            lvl = LODLevel(i, reduced, ratio, ch["symmetric_norm"],
                           accepted=False)
            chain.levels.append(lvl)
            break
        chain.levels.append(
            LODLevel(i, reduced, ratio, ch["symmetric_norm"], True))
        current = reduced
    return chain
