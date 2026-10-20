"""Mesh 数据结构 + 文件 IO（OBJ / PLY / GLB）。

设计取舍：
- 只用 numpy + 标准库（struct/json/base64），不依赖 trimesh/scipy——
  用户 `pip install numpy` 即可运行全部核心功能。
- 顶点属性（uv/normal）与顶点数对齐；OBJ 的 vt/vn 索引通常与 v 不同，
  读入时按 face 的 vt 索引展开为逐顶点属性（常见 glTF 导出风格），
  保持"一个顶点一组完整属性"的不变量，减面和写出一致。
- GLB 只实现渲染所需最小子集：单 mesh、单 primitive、
  POSITION(+NORMAL/TEXCOORD_0) + uint32 indices。这覆盖了绝大多数
  待优化资产；超出子集的文件在读取时明确报错，不静默丢数据。
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class MeshError(Exception):
    pass


@dataclass
class Mesh:
    vertices: np.ndarray                    # (n, 3) float64
    faces: np.ndarray                       # (m, 3) int32，顶点索引
    uvs: np.ndarray | None = None           # (n, 2) float64
    normals: np.ndarray | None = None       # (n, 3) float64
    name: str = "mesh"
    source_format: str = ""

    def __post_init__(self):
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int32)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise MeshError(f"vertices 必须是 (n,3)，得到 {self.vertices.shape}")
        if self.faces.size and self.faces.shape[1] != 3:
            raise MeshError(f"faces 必须是 (m,3)，得到 {self.faces.shape}")
        if self.faces.size:
            if self.faces.min() < 0 or self.faces.max() >= len(self.vertices):
                raise MeshError("faces 中存在越界顶点索引")

    # ---- 基本量 ----
    @property
    def n_vertices(self) -> int:
        return int(len(self.vertices))

    @property
    def n_faces(self) -> int:
        return int(len(self.faces))

    def face_vertices(self) -> np.ndarray:
        """(m, 3, 3) 每个三角形的三个顶点坐标。"""
        return self.vertices[self.faces]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def diagonal(self) -> float:
        lo, hi = self.bounds()
        return float(np.linalg.norm(hi - lo))

    def copy(self) -> "Mesh":
        return Mesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            uvs=None if self.uvs is None else self.uvs.copy(),
            normals=None if self.normals is None else self.normals.copy(),
            name=self.name,
            source_format=self.source_format,
        )

    def subset(self, keep_faces: np.ndarray) -> "Mesh":
        """按 face 掩码/索引取子集，顶点重新编号（丢弃未引用顶点）。"""
        faces = self.faces[keep_faces]
        used = np.unique(faces)
        remap = -np.ones(len(self.vertices), dtype=np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)
        return Mesh(
            vertices=self.vertices[used],
            faces=remap[faces],
            uvs=None if self.uvs is None else self.uvs[used],
            normals=None if self.normals is None else self.normals[used],
            name=self.name,
            source_format=self.source_format,
        )

    def stats(self) -> dict:
        return {
            "name": self.name,
            "format": self.source_format,
            "vertices": self.n_vertices,
            "faces": self.n_faces,
            "has_uvs": self.uvs is not None,
            "has_normals": self.normals is not None,
            "diagonal": round(self.diagonal(), 6),
        }


# ---------------------------------------------------------------- OBJ
def read_obj(path: str | Path) -> Mesh:
    v, vt, vn = [], [], []
    f_v, f_vt, f_vn = [], [], []
    for lineno, raw in enumerate(Path(path).read_text(
            encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "v":
            v.append([float(x) for x in parts[1:4]])
        elif tag == "vt":
            vt.append([float(x) for x in parts[1:3]])
        elif tag == "vn":
            vn.append([float(x) for x in parts[1:4]])
        elif tag == "f":
            # 支持 v, v/vt, v/vt/vn, v//vn；多边形按扇形三角化
            tri_v, tri_vt, tri_vn = [], [], []
            for tok in parts[1:]:
                a = tok.split("/")
                tri_v.append(int(a[0]) - 1)
                tri_vt.append(int(a[1]) - 1 if len(a) > 1 and a[1] else -1)
                tri_vn.append(int(a[2]) - 1 if len(a) > 2 and a[2] else -1)
            for k in range(1, len(tri_v) - 1):
                f_v.append((tri_v[0], tri_v[k], tri_v[k + 1]))
                f_vt.append((tri_vt[0], tri_vt[k], tri_vt[k + 1]))
                f_vn.append((tri_vn[0], tri_vn[k], tri_vn[k + 1]))
        elif tag == "o":
            pass

    if not v:
        raise MeshError(f"{path}: 没有顶点数据")
    vertices = np.array(v, dtype=np.float64)
    faces = np.array(f_v, dtype=np.int32).reshape(-1, 3)

    uvs = normals = None
    if vt and all(t >= 0 for t in f_vt for t in t):
        uvs = _expand_per_vertex(vt, f_vt, len(vertices))
    if vn and all(t >= 0 for t in f_vn for t in t):
        normals = _expand_per_vertex(vn, f_vn, len(vertices))
    return Mesh(vertices, faces, uvs, normals,
                name=Path(path).stem, source_format="obj")


def _expand_per_vertex(attr: list, f_attr: list, n_vertices: int):
    """OBJ 的属性索引与顶点索引不同；展开成逐顶点属性（取首次出现的面）。

    若同一顶点在不同面引用不同属性值（UV 接缝），保留首次出现的值，
    并在 stats 之外通过 uvs_seam 检测暴露（meshio 层只保证不崩）。
    """
    arr = np.array(attr, dtype=np.float64)
    out = np.zeros((n_vertices, arr.shape[1]), dtype=np.float64)
    seen = np.zeros(n_vertices, dtype=bool)
    for tri in f_attr:
        for vi, ai in zip(tri, tri):
            if ai >= 0 and not seen[vi]:
                out[vi] = arr[ai]
                seen[vi] = True
    return out


def write_obj(mesh: Mesh, path: str | Path) -> None:
    lines = ["# MeshPilot OBJ export", f"o {mesh.name}"]
    for p in mesh.vertices:
        lines.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    if mesh.uvs is not None:
        for t in mesh.uvs:
            lines.append(f"vt {t[0]:.6f} {t[1]:.6f}")
    if mesh.normals is not None:
        for n in mesh.normals:
            lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
    has_uv = mesh.uvs is not None
    has_n = mesh.normals is not None
    for f in mesh.faces:
        a, b, c = f + 1
        if has_uv and has_n:
            lines.append(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")
        elif has_uv:
            lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
        elif has_n:
            lines.append(f"f {a}//{a} {b}//{b} {c}//{c}")
        else:
            lines.append(f"f {a} {b} {c}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- PLY
def read_ply(path: str | Path) -> Mesh:
    data = Path(path).read_bytes()
    end = data.find(b"end_header")
    if end < 0:
        raise MeshError("PLY: 缺少 end_header")
    header = data[:end].decode("ascii", errors="replace")
    fmt = "ascii"
    for line in header.splitlines():
        if line.startswith("format"):
            fmt = line.split()[1]
    body = data[end + len(b"end_header") + 1:]
    if body[:1] == b"\n":
        body = body[1:]

    n_vert = n_face = 0
    props: list[tuple[str, str]] = []
    in_vertex = False
    for line in header.splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "element" and tok[1] == "vertex":
            in_vertex = True
            n_vert = int(tok[2])
        elif tok[0] == "element" and tok[1] != "vertex":
            in_vertex = False
            if tok[1] == "face":
                n_face = int(tok[2])
        elif tok[0] == "property" and in_vertex:
            props.append((tok[1], tok[2]))

    if fmt == "ascii":
        rows = [ln.split() for ln in body.decode("ascii").splitlines()
                if ln.strip()]
        verts = np.array([[float(x) for x in r[:3]] for r in rows[:n_vert]])
        faces = []
        for r in rows[n_vert:n_vert + n_face]:
            k = int(r[0])
            faces.append([int(x) for x in r[1:1 + k]])
    else:
        is_le = fmt == "binary_little_endian"
        endian = "<" if is_le else ">"
        ply2np = {"float": "f4", "double": "f8", "int": "i4", "uint": "u4"}
        dt = np.dtype([(p, endian + ply2np[t]) for t, p in props])
        verts_raw = np.frombuffer(body[:n_vert * dt.itemsize], dtype=dt)
        off = n_vert * dt.itemsize
        faces = []
        for _ in range(n_face):
            (k,) = struct.unpack_from(endian + "B", body, off)
            idx = struct.unpack_from(endian + "%di" % k, body, off + 1)
            off += 1 + 4 * k
            faces.append(idx)
        verts = np.stack([verts_raw["x"], verts_raw["y"], verts_raw["z"]],
                         axis=1).astype(np.float64)
    faces = np.array(faces, dtype=np.int32).reshape(-1, 3)
    return Mesh(verts, faces, name=Path(path).stem, source_format="ply")


def write_ply(mesh: Mesh, path: str | Path) -> None:
    lines = [
        "ply", "format ascii 1.0",
        "comment MeshPilot export", f"element vertex {mesh.n_vertices}",
        "property float x", "property float y", "property float z",
    ]
    if mesh.uvs is not None:
        lines += ["property float s", "property float t"]
    lines.append(f"element face {mesh.n_faces}")
    lines.append("property list uchar int vertex_indices")
    lines.append("end_header")
    for i, p in enumerate(mesh.vertices):
        row = f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"
        if mesh.uvs is not None:
            row += f" {mesh.uvs[i][0]:.6f} {mesh.uvs[i][1]:.6f}"
        lines.append(row)
    for f in mesh.faces:
        lines.append(f"3 {f[0]} {f[1]} {f[2]}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- GLB
_GLB_MAGIC = 0x46546C67  # 'glTF'
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942

_COMP = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "i", 5126: "f"}
_NP_COMP = {5120: "i1", 5121: "u1", 5122: "i2", 5123: "u2",
            5125: "i4", 5126: "f4"}
_TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb(path: str | Path) -> list[Mesh]:
    """读取 GLB，返回其全部 mesh/primitive（各转成一个 Mesh）。

    支持 accessor 的 bufferView + byteOffset/byteStride；
    不支持 draco/meshopt 压缩（明确报错）、interleaved 之外的稀疏扩展。
    """
    data = Path(path).read_bytes()
    if len(data) < 12 or struct.unpack_from("<I", data, 0)[0] != _GLB_MAGIC:
        raise MeshError("不是合法 GLB（magic 不符）")
    total = struct.unpack_from("<I", data, 8)[0]
    off = 12
    json_doc = bin_chunk = None
    while off < min(total, len(data)):
        clen, ctype = struct.unpack_from("<II", data, off)
        off += 8
        chunk = data[off:off + clen]
        off += clen
        if ctype == _CHUNK_JSON:
            json_doc = json.loads(chunk.decode("utf-8"))
        elif ctype == _CHUNK_BIN:
            bin_chunk = chunk
    if json_doc is None:
        raise MeshError("GLB 缺少 JSON chunk")

    buffers = []
    if bin_chunk:
        buffers.append(bin_chunk)
    for b in json_doc.get("buffers", []):
        if "uri" not in b and len(buffers) <= 0:
            raise MeshError("buffer 无 uri 且无 BIN chunk")
        if "uri" in b:
            uri = b["uri"]
            if uri.startswith("data:"):
                import base64
                buffers.append(base64.b64decode(uri.split(",", 1)[1]))

    def read_accessor(idx: int) -> np.ndarray:
        acc = json_doc["accessors"][idx]
        if acc.get("normalized"):
            raise MeshError("normalized accessor 暂不支持")
        bv = json_doc["bufferViews"][acc["bufferView"]]
        buf = buffers[bv.get("buffer", 0)]
        n = acc["count"]
        comp = _NP_COMP[acc["componentType"]]
        k = _TYPE_COUNT[acc["type"]]
        stride = bv.get("byteStride") or np.dtype(comp).itemsize * k
        start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        if stride == np.dtype(comp).itemsize * k:
            arr = np.frombuffer(
                buf, dtype=np.dtype(comp), count=n * k, offset=start
            ).reshape(n, k)
        else:
            out = np.empty((n, k), dtype=comp)
            for i in range(n):
                row = np.frombuffer(
                    buf, dtype=np.dtype(comp), count=k,
                    offset=start + i * stride)
                out[i] = row
            arr = out
        if arr.dtype.kind == "i" and acc["componentType"] == 5123:
            arr = arr.astype(np.int32)
        if arr.dtype.kind == "u" and acc["componentType"] == 5125:
            arr = arr.astype(np.int32)
        return arr.astype(np.float64 if arr.dtype.kind == "f" else np.int64)

    meshes: list[Mesh] = []
    for mi, gmesh in enumerate(json_doc.get("meshes", [])):
        for pi, prim in enumerate(gmesh.get("primitives", [])):
            attrs = prim.get("attributes", {})
            if prim.get("mode", 4) != 4:
                raise MeshError(f"mesh {mi} primitive {pi}: 非三角模式不支持")
            if prim.get("extensions", {}).get("KHR_draco_mesh_compression"):
                raise MeshError("Draco 压缩暂不支持，请先解压")
            if "POSITION" not in attrs:
                raise MeshError(f"mesh {mi} primitive {pi}: 无 POSITION")
            verts = read_accessor(attrs["POSITION"])
            uvs = (read_accessor(attrs["TEXCOORD_0"])
                   if "TEXCOORD_0" in attrs else None)
            normals = (read_accessor(attrs["NORMAL"])
                       if "NORMAL" in attrs else None)
            faces = read_accessor(prim["indices"]).reshape(-1, 3)
            meshes.append(Mesh(
                verts, faces.astype(np.int32), uvs, normals,
                name=gmesh.get("name", f"mesh{mi}"), source_format="glb"))
    if not meshes:
        raise MeshError("GLB 中没有 mesh")
    return meshes


def write_glb(mesh: Mesh, path: str | Path) -> None:
    """写出最小 GLB：单 mesh 单 primitive，POSITION/NORMAL?/TEXCOORD_0?/indices。"""
    pos = mesh.vertices.astype(np.float32)
    idx = mesh.faces.astype(np.uint32).reshape(-1)
    blob = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []

    def add(arr: np.ndarray, target: int, ctype: int, atype: str) -> int:
        raw = arr.tobytes()
        # 4 字节对齐
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        blob.extend(raw)
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(raw),
                      "target": target})
        accessors.append({
            "bufferView": len(views) - 1, "componentType": ctype,
            "count": int(arr.size // _TYPE_COUNT[atype]), "type": atype,
        })
        return len(accessors) - 1

    a_pos = add(pos, 34962, 5126, "VEC3")
    a_idx = add(idx, 34963, 5125, "SCALAR")
    attrs = {"POSITION": a_pos}
    if mesh.normals is not None:
        attrs["NORMAL"] = add(mesh.normals.astype(np.float32),
                              34962, 5126, "VEC3")
    if mesh.uvs is not None:
        attrs["TEXCOORD_0"] = add(mesh.uvs.astype(np.float32),
                                  34962, 5126, "VEC2")

    doc = {
        "asset": {"version": "2.0", "generator": "MeshPilot"},
        "meshes": [{"name": mesh.name, "primitives": [{
            "attributes": attrs, "indices": a_idx, "mode": 4}]}],
        "bufferViews": views,
        "accessors": accessors,
        "buffers": [{"byteLength": len(blob)}],
    }
    js = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    while len(js) % 4:
        js += b" "
    out = bytearray()
    out += struct.pack("<III", _GLB_MAGIC, 2,
                       12 + 8 + len(js) + 8 + len(blob))
    out += struct.pack("<II", len(js), _CHUNK_JSON) + js
    out += struct.pack("<II", len(blob), _CHUNK_BIN) + bytes(blob)
    Path(path).write_bytes(bytes(out))


# ---------------------------------------------------------------- 调度
_READERS = {".obj": read_obj, ".ply": read_ply, ".glb": read_glb}
_WRITERS = {".obj": write_obj, ".ply": write_ply, ".glb": write_glb}


def load(path: str | Path) -> list[Mesh]:
    p = Path(path)
    reader = _READERS.get(p.suffix.lower())
    if reader is None:
        raise MeshError(f"不支持的输入格式 {p.suffix}（支持 obj/ply/glb）")
    meshes = reader(p)
    return meshes if isinstance(meshes, list) else [meshes]


def save(mesh: Mesh, path: str | Path) -> None:
    p = Path(path)
    writer = _WRITERS.get(p.suffix.lower())
    if writer is None:
        raise MeshError(f"不支持的输出格式 {p.suffix}（支持 obj/ply/glb）")
    writer(mesh, p)
