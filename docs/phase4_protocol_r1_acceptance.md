# 阶段 4 R1 执行协议验收记录

- 阶段：4（可解释异常增量模型）目标读取前执行勘误
- 科学协议版本：`0.4.0`
- 执行修订：`r1`
- 验收日期：2026-07-16
- 协议冻结提交：`f30d7e4ae7bb729d4ab38177a319a78957458596`
- 协议标签：`v0.3.0-anomaly-increment-protocol-r1`
- 评分代码计划标签：`v0.3.0-anomaly-increment-scoring-code-r1`
- 结果计划标签：`v0.3.0-anomaly-increment-r1`
- 正式评分尝试：`0`
- 地震目标读取：`0`
- 锁定测试：未运行

本记录只验收 R0 readiness 事故登记、R1 机器契约、四份目标盲清单、受限空间工件复现、随机输入封印和协议级测试。它不验收 Arrow 逻辑身份评分实现，不生成 R1 seal，不授权读取阶段4目标。只有本记录提交、推送、跨平台 CI 通过并发布协议标签后，才允许开始 R1 评分代码修复。

## 冻结身份

| 项目 | SHA-256 |
| --- | --- |
| 规范化 R1 协议设计 | `c15d3bbca5cef4b363a79e183d715124256a12088873d81cd77de489766b32de` |
| R1 随机输入封印 | `4d6287a2bc737f1f7050a9daf60f89af8e79cd975fc09b8a867b3ada8b24416c` |
| R1 跨清单验证内容 | `ea45ae7ba6a847c83867b988ec3e1aa4b09257e105797edbad4c2a084009be7a` |
| 折与暴露清单文件 / 内容 | `d700f15837102ab90b2c9038a6d807bd94cf2f5683a7aeb3088dc5ad9ea61788` / `bc7fc404451837f2a2d1b281c13e6f0cbb3920dfaefcdc8dee5444efd62fc6ab` |
| 特征设计清单文件 / 内容 | `f4b18c7af27a1f16b2ea7d01f48d9f1f03439860939fc75bcd969ac72313706f` / `cf9bc7b3800b6ebfd5cf0a085666f28c261ffe9ba8bf00704528dcf382a5efb3` |
| 随机性清单文件 / 内容 | `1c5c473e2a8dcf09983ae3432ae17d81360efce634f480fedc350a680b18ee30` / `4bd22487487b762312fb4f057a42b68c9df2faa25473ecdd2f9f010d4d6cf79d` |
| 空间分层清单文件 / 内容 | `aac273e9ac306568a93063fad0fe0ebc28d161189e0b4f089d657ed8d7ba2654` / `d897710b21e913bec7cf6c05c60844611ce0118f8f436aeeae22314001ebf281` |

R1 的 fold、feature、spatial 三份科学清单与 R0 文件逐字节相同。PCG64、根种子147、类型化上下文与配对规则保持不变；因 R1 协议设计和输入封印身份改变，实际 R1 随机流与参考向量从新封印确定性重派生，并仅由 R1 randomness 清单冻结。

## 受限空间工件

| 工件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| 25 km cell mapping | 42,917 | `d11a4fa571c16e80229311c9f449443634f5a1b8e6b3ca6762282b35a3699e3c` |
| entity mapping | 4,055,922 | `808c56804c51de0cd1c217765f2846e7a6b4662b7f6229e718dfe89a6badb57e` |
| connectors | 11,768 | `8f7a99aaf89e7487c71f6970b30d28263a2deec47241dcdac90633a3f1af15d4` |
| zone geometry | 233,171 | `d36295bc243f75c8f4af3dd0bfd9c9808b8e2cce158719c028ac9d0360c66d6a` |

R1 四个本地工件与 R0 文件逐字节相同。公开 spatial 清单不含坐标、几何、WKB/GeoJSON 或逐格映射；原始构造线和四个受限工件继续只保存在本地忽略路径。

## R0 事故与阶段3判定

[R0 readiness 事故记录](phase4_scoring_readiness_incident_r0.md)固定了原标签、receipt、qualification、seal 与双台账身份。旧 formal-attempt ledger 和 target-read ledger 的记录数均为0。

25 km 正式范围 153/153 期、36列的 2-worker 诊断逐列满足字段、顺序、类型、validity 与 Arrow 科学数值一致，不存在首个科学差异；因此阶段3工件和标签不修订。该诊断明确不能区分有效 `+0.0/-0.0`，而接受表含278,811个有效负零，所以 R1 评分代码完成后仍必须用最终版本化哈希在 1/2 workers 下完成153期全部有效 payload 的逐位复演。

## 验收测试

- `scripts/build_stage4_preregistration.py check`：4份清单、4个本地工件、全部跨清单约束通过，`target_read_count=0`。
- R1 协议测试：13项通过。
- 完整非目标测试：1,051项通过。
- Ruff lint/format、`git diff --check`：通过。
- mypy：211个源文件无错误。
- wheel 与 sdist：构建成功；`seismoflux --version` 通过。
- 本机绝对路径、秘密与凭据扫描：通过。
- 独立只读审计：`APPROVE`，无阻断项。
- GitHub CI run `29434361575`：Ubuntu 通过（2分42秒），Windows 通过（4分27秒）。

## 下一门

协议标签发布后才允许：

1. 新增版本化 Arrow 逻辑身份实现，保留 legacy 物理 IPC 哈希；
2. 将默认配置、授权、preflight、qualification、seal、双台账、checkpoint、模型包、报告、静态图与交互页面全部迁入 R1 命名空间；
3. 完成153期 1/2 workers 逐位复演、全部非目标测试、跨平台 CI、提交、推送和评分代码标签；
4. 评分标签之后生成并核验 R1 receipt、qualification、空双台账和 seal；
5. target-blind readiness 全绿后，才允许唯一正式目标读取。

正式计算仍锁定 CPU float64。GPU 只有在独立等价性门通过后才可作为可选加速；锁定测试继续禁止运行。
