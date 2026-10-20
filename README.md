# MeshPilot — 自动驾驶式 3D 资产优化 Agent

<div align="left">

[![CI](https://github.com/yuyu2929/meshpilot/actions/workflows/ci.yml/badge.svg)](https://github.com/yuyu2929/meshpilot/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Dual_(NC_%2F_Commercial)-blue)](LICENSE.md)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

</div>

> 💰 **个人免费，商业付费**：个人学习/研究/爱好项目永久免费；
> 公司、个体经营者等营利实体的任何业务用途需购买[商业许可](COMMERCIAL-LICENSE.md)。
> 详见 [LICENSE.md](LICENSE.md)。

> 诊断（拓扑/指标）→ 处方（优化计划）→ 执行（流水线）→ 验证（质量门槛）。
> 一个能对自己发的每一刀负责的减面器。

MeshPilot 把"3D 资产优化"包装成一个 Agent 闭环：它先体检网格
（非流形边、边界、退化面、重复顶点），再根据平台预算开处方
（weld → clean → decimate），执行减面，最后用**几何质量门槛**验收——
不达标就停下报告，而不是默默交出烂网格。

核心减面算法是 **QEM 边折叠**（Garland & Heckbert, SIGGRAPH 1997——
Blender Decimate、Simplygon、meshoptimizer 的 simplify 都源于这一族），
带二面角锐边钳位、翻转/link 保护、懒删除堆调度。

## 诚实定位

这是一个**教学级但正确**的 QEM 实现，不是 Simplygon 的竞品：

- **正确性经过实测**：拓扑保持（球 euler=2、环 euler=0、无边界/非流形/退化面）、
  预算精确命中、锐边与角点在所有预算档位精确保留（见下方基准）。
- **性能定位**：纯 Python/numpy，万面量级秒级——面向"优化前检查 /
  CI 门控 / 教学演示 / 小型资产批处理"。百万面生产资产请用 C++ 库
  （meshoptimizer 等），MeshPilot 的处方逻辑可以直接移植过去。
- **质量度量的底噪是诚实的**：点集采样的 chamfer 有 ~1/√n 的系统底噪
  （8000 采样 ≈ 0.005–0.008 归一化），文档中所有对比都标注了底噪，
  不把噪声当成优势。

## 快速开始

```bash
pip install -r requirements.txt
python scripts/demo.py          # 六段全量演示（无需任何外部资源文件）
python -m meshpilot demo        # 同上的 CLI 入口
python run_web.py               # Web 控制台 → http://localhost:8700
python -m pytest tests/ -q      # 49 个测试
```

### CLI

```bash
# 体检：拓扑问题 + 指标
python -m meshpilot analyze asset.obj

# 优化到平台预算，带质量验收（不达标 exit code 2）
python -m meshpilot optimize asset.obj -o out.obj --platform web_mobile

# 生成 LOD 链
python -m meshpilot lod asset.obj --ratios 1.0,0.5,0.25,0.125 -o lod_dir/
```

平台预算（面数上限）：`web_mobile` 25k / `web_desktop` 100k /
`pc` 500k / `console` 2M。也可 `--max-faces N --chamfer 0.01` 自定义。

### 作为库

```python
from meshpilot import prims, agent
from meshpilot.qem import simplify
from meshpilot.quality import chamfer_distance

mesh = prims.noisy_sphere(4)                 # 或 load("asset.glb")
result = agent.run(mesh, agent.Budget(max_faces=800, platform="custom"))
print(result.verification)                   # {'passed': True, 'chamfer_norm': 0.006, ...}

red, stats = simplify(mesh, 500)             # 直接调算法
print(chamfer_distance(mesh, red))
```

## 基准（实测，非宣传）

资产全部程序化生成（仓库自包含，clone 即可跑）。chamfer 为面积加权
双向点集距离，8000 采样，按对角线归一化；**采样底噪 ≈ 0.005–0.008**。

| 资产 | 方法 | 面数 | chamfer | 备注 |
|---|---|---|---|---|
| icosphere(4) 5120 面 | QEM | 1000（精确命中） | 0.00572 | ≈ 均匀细分 icosphere(3)（1280 面）的质量 |
| | 顶点聚类 | 1000 | 0.00592 | 对照组 |
| feature_box 432 面（锐边盒子） | QEM | 200（精确命中） | 0.00784 | **8/8 角精确保留**，euler=2，零退化面 |
| | 顶点聚类 | 300（请求 200） | 0.00779 | 网格量化打不中预算；超调时（108 面）角偏移 0.25 |
| noisy_sphere(4) 5120 面 | QEM | 800 | 0.00600 | Agent 闭环验收通过，覆盖率 89.2% |

**怎么读这张表**：解析曲面（球/环）上 QEM 与聚类同阶（都贴底噪）——
这不是缺陷，是数学事实；QEM 的优势在**特征资产**（锐边/角点保持）和
**预算控制**（精确命中面数，聚类不能）。把减面器用于 CAD/硬表面资产时，
这两点正是生产线关心的。

## 架构

```
meshpilot/
  meshio.py      Mesh 数据结构；OBJ / PLY / GLB 读写（GLB 最小子集，Draco 明确报错）
  topology.py    体检：非流形/边界/退化/重复顶点/欧拉数
  weld.py        顶点合并（零质量损失的第一步）
  qem.py         QEM 边折叠（核心）+ 顶点聚类（对照组基线）
  quality.py     chamfer / 覆盖率 / 质量分
  lod.py         LOD 链生成（质量不达标即截断）
  agent.py       自动驾驶闭环：diagnose → prescribe → run → verify
  licensing.py   Ed25519 许可证（MP1.<payload>.<sig>）
  enterprise/    企业功能插槽（batch_pipeline / ci_report / sso）
  cli.py         analyze / optimize / lod / demo
  webapp/        Web 控制台（FastAPI + 静态页）
scripts/         密钥生成 / 许可证签发 / 全量演示
tests/           49 个用例（算法 / 拓扑 / LOD / Agent / IO / Web / 许可）
```

## 许可模式：个人免费，商业付费

**一句话：个人免费用，公司用要买授权。**

| 你是谁 | 费用 |
|---|---|
| **个人**：学习/研究/爱好项目/业余作品 | **永久免费** |
| 非营利组织：教育、政府、公益、公共研究 | **免费** |
| **公司 / 个体经营者 / 任何营利实体** | **需购买[商业许可](COMMERCIAL-LICENSE.md)** |

- **核心层**（算法、CLI、LOD、质量验证、Web 控制台）：不设任何代码门禁，
  个人与非营利组织免费；**营利使用需商业授权**（PolyForm Noncommercial 1.0.0）。
- **企业层**（`batch_pipeline` / `ci_report` / `sso`）：需商业许可 +
  私有扩展包 `meshpilot_enterprise`，通过 `MESHPILOT_LICENSE` 环境变量挂载
  Ed25519 签名许可证。公钥随仓库分发，私钥从不入库；任何篡改静默降级社区版。

> ⚠️ 本项目**不是 OSI 开源项目**，准确说法是 **source-available**（源码可见）。
> 详见 [LICENSE.md](LICENSE.md)。

详见 [docs/BUSINESS.md](docs/BUSINESS.md)。

## 已知限制（路线图）

- 属性纹理（UV/法线）暂不参与 quadric 误差——减面可能拉扯 UV 接缝
- 单 mesh / 单 primitive 的 GLB；Draco 压缩需先解码
- 无边界 quadric 惩罚（开放网格的轮廓保护弱于闭网格）
- 性能：百万面资产需要 C++ 移植（算法本身可移植，无魔法）

数学推导与工程约束详见 [THEORY.md](THEORY.md)；开发踩坑全记录见
[DEVLOG.md](DEVLOG.md)。
