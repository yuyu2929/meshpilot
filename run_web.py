"""MeshPilot Web 控制台 + API。

    python run_web.py [--port 8700]

页面：/            优化控制台（内置演示网格，无需上传）
API： POST /api/optimize   诊断 → 处方 → 执行 → 验证 全流程
      GET  /api/meshes     可用演示网格列表
      GET  /api/health     健康检查
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from meshpilot import agent, prims
from meshpilot.enterprise import status as ent_status
from meshpilot.meshio import MeshError
from meshpilot.topology import analyze as topo_analyze


# 请求模型必须在模块级（from __future__ import annotations 下，
# 函数内定义的 Pydantic 模型会被 FastAPI 退化成查询参数——aria-agent/FPTES 踩过）
class OptimizeIn(BaseModel):
    mesh: str = Field(default="noisy_sphere(4)", max_length=64)
    # 上界防止一次请求把 CPU 占死（无认证的本地服务也要设防）
    budget_faces: int = Field(default=800, ge=4, le=2_000_000)
    chamfer: float = Field(default=0.01, ge=0.0, le=1.0)


MESHES = {
    "icosphere(2)": lambda: prims.icosphere(2),
    "icosphere(4)": lambda: prims.icosphere(4),
    "noisy_sphere(4)": lambda: prims.noisy_sphere(4),
    "torus": lambda: prims.torus(48, 24),
}


def create_app() -> FastAPI:
    app = FastAPI(title="MeshPilot", version="0.1.0")
    static_dir = Path(__file__).parent / "meshpilot" / "webapp" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(static_dir / "index.html"))

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/meshes")
    def meshes():
        return {"meshes": [{"name": k, "faces": f().n_faces}
                           for k, f in MESHES.items()]}

    @app.post("/api/optimize")
    def optimize(inp: OptimizeIn):
        if inp.mesh not in MESHES:
            return {"error": f"未知网格 {inp.mesh}，可选：{list(MESHES)}"}
        try:
            mesh = MESHES[inp.mesh]()
            budget = agent.Budget(max_faces=inp.budget_faces,
                                  platform="custom",
                                  max_chamfer_norm=inp.chamfer)
            result = agent.run(mesh, budget)
            before = topo_analyze(mesh)
            after = topo_analyze(result.mesh)
            return {
                "mesh": inp.mesh,
                "budget": {"faces": inp.budget_faces,
                           "chamfer_norm": inp.chamfer},
                "before": before.to_dict(),
                "after": after.to_dict(),
                "steps": [{"action": s.step.action,
                           "rationale": s.step.rationale,
                           "note": s.note,
                           "before_faces": s.before.get("faces"),
                           "after_faces": s.after.get("faces")}
                          for s in result.steps],
                "verification": {
                    k: (v.item() if isinstance(v, np.generic) else v)
                    for k, v in result.verification.items()
                    if k != "checks"
                },
                "checks": {
                    k: (bool(v.item()) if isinstance(v, np.generic) else bool(v))
                    for k, v in result.verification["checks"].items()
                },
            }
        except MeshError as e:
            return {"error": str(e)}

    @app.get("/api/license")
    def license_status():
        return ent_status()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="MeshPilot Web 控制台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8700)
    args = parser.parse_args()
    import uvicorn

    print(f"MeshPilot → http://{args.host}:{args.port}")
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
