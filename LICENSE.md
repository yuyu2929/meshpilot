# License Overview 双许可总览

**一句话：个人免费用，商业用付费。**

MeshPilot 采用**双许可（Dual Licensing）**发布。使用前按你的身份选一条即可：

| 你是谁 | 适用的许可 | 费用 |
|---|---|---|
| **个人**：学习、研究、试验、爱好项目、业余作品 | [PolyForm Noncommercial 1.0.0](LICENSE-POLYFORM.md) | **永久免费** |
| 非营利组织：慈善、教育、公共研究、公共安全/卫生、环保、政府机构 | [PolyForm Noncommercial 1.0.0](LICENSE-POLYFORM.md) | **免费** |
| **公司 / 个体经营者 / 任何营利实体**（内部系统、嵌入产品、SaaS、外包交付） | [商业许可](COMMERCIAL-LICENSE.md) | **需购买授权** |

## 1. PolyForm Noncommercial 1.0.0 —— 个人与非营利组织免费

完整法律文本见 [LICENSE-POLYFORM.md](LICENSE-POLYFORM.md)。适用于**非商业目的**：

- 个人学习、研究、试验、爱好项目、业余追求
- 科研与公开发表（以公共利益为目的的研究）
- 慈善组织、教育机构、公共研究机构、公共安全/卫生组织、环保组织、政府机构
  （无论其资金来源如何）

覆盖范围包括全部核心能力：QEM 减面、拓扑诊断、LOD 链、质量验证、CLI 与本地 Web 控制台。

## 2. 商业许可 —— 企业与营利组织付费

任何**营利性实体**（公司、个体经营者及其关联机构）出于任何商业目的使用 MeshPilot
（内部资产流水线、嵌入产品、SaaS 服务、外包交付、闭源分发），都需要一份商业许可。

商业授权条款、档位与联系方式见 [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)。

## 与企业版功能的关系

MeshPilot 有两层，**许可要求不同**：

| 层 | 内容 | 个人/非营利 | 营利组织 |
|---|---|---|---|
| 核心层 | 算法、CLI、LOD、质量验证、Web 控制台 | 免费 | 需商业许可 |
| 企业层 | `batch_pipeline` / `ci_report` / `sso` 流水线周边 | — | 需商业许可 + 私有扩展包 |

企业层还额外需要 `pip install meshpilot_enterprise`（私有包）与
`MESHPILOT_LICENSE` 环境变量。核心层不经任何许可门禁，无网也能跑。

## 判断不了？看这两个问题

1. **使用 MeshPilot 的行为，最终是否服务于营利？** 是 → 需要商业许可。
2. **你是法律意义上的非商业组织吗？**（PolyForm "Noncommercial Organizations" 一节
   有逐字定义）是 → 免费许可覆盖你，与资金来源无关。

仍有疑问，欢迎开 Issue 询问；善意的不确定不会被视为违约（见 PolyForm "Violations"
一节的 32 天补救条款）。

---

Copyright (c) 2026 yuyu2929. All rights reserved.
