# 阶段 4 R2 协议冻结验收记录

- 验收日期：2026-07-16
- 协议版本：`0.4.1`
- 执行修订：`r2`
- 纠正修订：`r1`
- 协议标签计划：`v0.3.1-anomaly-increment-protocol-r2`
- 评分代码标签计划：`v0.3.1-anomaly-increment-scoring-code-r2`
- 结果标签计划：`v0.3.1-anomaly-increment-r2`
- R1 formal-attempt 记录：`0`
- R1 target-read 记录：`0`
- 阶段 4 正式目标读取：`0`
- 阶段 4 正式评分：`0`
- 阶段 9 锁定测试：未运行

本验收仅覆盖目标盲的 R2 协议、路径、清单、受限空间工件、随机输入封印和治理测试。本验收不授权正式目标读取，不执行评分，不产生 G2/G3 或采用结论。

## R1 废止哈希

| 证据 | SHA-256 |
| --- | --- |
| R1 协议设计 | `c15d3bbca5cef4b363a79e183d715124256a12088873d81cd77de489766b32de` |
| R1 scoring seal 文件 | `a6e8dc9ac283813edb62e301114d4985ae332b9c607584c987a4297efe5978f3` |
| R1 formal-attempt ledger 文件 | `9ac5e5e080c1d5425f985cb3091b94c0da69d211469589d26ae8bfc314088142` |
| R1 formal-attempt ledger 内容 | `cadc80e5a0f00ffce241f910409750b01e3f410d910dda5d5aad0ff3033d2448` |
| R1 target-read ledger 文件 | `0a49450cc1006ccd0ced26fba30330417f1ec8667c5cedf0bf04242f158210c8` |
| R1 target-read ledger 内容 | `4c1fb843edfa8f59f37f137d8d68962cbdae5991cc8809302d39d240f0b395b6` |

## R2 冻结身份

生成清单并完成验收后，在本节记录：

| 项目 | SHA-256 |
| --- | --- |
| R2 协议设计 | `96a8464d7727db32668eb6fed0a2712c75667f729f996d03e3b59b79882de40b` |
| R2 随机输入封印 | `27ae5b06ede6a36fe6d9af285f22dc10e2f3b86d09f1933f1f7a8379751509a4` |
| R2 跨清单验收内容 | `59f16596ede027a5c397397f8365b30b7554b50375c1de62c277bf64582104ac` |
| fold 清单文件 / 内容 | `74bb8c73751cab71c85e20609eb70d94902634fe946a3cb4316a83d7e9b051a1` / `d3fd999d51ad66e3ea153d1b40d24d8f861993f0337d2de297ac297711f8eaf2` |
| feature-set 清单文件 / 内容 | `89eb3d73eb1ffdb554dc4e72792aedd65473711ca061d197242a6cd560ca4675` / `65b86aa50dc18ba6cab94f264c82772d588bac4ef9c466138a87a4e8e4995211` |
| randomness 清单文件 / 内容 | `0deb1928a166a0ff1c7cb744df088301328aa78a906eab30c16abaa72116d784` / `9fdfb5be0dcf02c3421099bb0efb650fdd197f45a683ec6889f691e4eda45b32` |
| spatial-strata 清单文件 / 内容 | `283a6790f6e7c16bc31d9498b2cc3cd043e19c8f141046afb898e988f25dcc83` / `0e8e1af89b802522150c9ef3b46a1ffe0d440e51f69ce4a532a6e1540934b5d2` |

## 本地受限空间工件

| 工件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| 25 km cell mapping | 42917 | `171a500de9f9dd475f2c37a5426debc7c6f2d34ddd418056729c39b27118108e` |
| entity mapping | 4055922 | `49cd56ace13680c3465b0c128f7dd9823636f6f1db7a2f39a12d1235df532170` |
| connectors | 11768 | `1f25120d9b9b15ec428efe97183179cebe1c3c5b0e022294dbf82f4c73e4e167` |
| zone geometry | 233171 | `c1c54d390bd1553c8f75b10def4898e24deb919345dd9ca0a11a02d0ff80ba70` |

## 必须通过的检查

- [x] R2 四份公开清单内容哈希和文件哈希有效。
- [x] R2 四个本地空间工件路径、字节数和 SHA 与公开摘要一致。
- [x] R1/R2 科学日历、折、特征和空间输入除版本身份外一致。
- [x] R2 PCG64 参考向量由新封印重新派生，且不等于 R1。
- [x] G2 对 dynamic 与 snapshot 都冻结 time/space `p<=0.05`。
- [x] G2 的 coverage guard 与两项 practical threshold 对 dynamic/snapshot 分别独立应用。
- [x] 四个正式 placebo 请求和四个 checkpoint 身份均已预登记。
- [x] 结果身份显式绑定四个 placebo 结果对象及完整零分布，而非只绑定最终 p 值。
- [x] CPU 上限为 6、BLAS=1、禁止嵌套并行、低优先级要求已冻结。
- [x] 地震目录 `origin_time`/`available_at` 覆盖截止 `2026-07-09T04:25:56Z`，覆盖全部冻结验证窗终点，缺失或不足时 fail-closed。
- [x] 发布限制明确：目录时效假设乐观、bootstrap 不含重拟合不确定性、ETAS 不可评估且只能相对冻结 KDE 背景声称增量。
- [x] 四个空间输出物理隔离、采用模型表、中心点警示和轴刻度要求已冻结。
- [x] public/forecast 工件冻结结构化目标载荷拒绝列表，关键词扫描或 UI 隐藏不能替代。
- [x] 展示语义冻结 coverage、聚合回溯、百分位、相对强度、采用卡、日历地标、采用变体空间图与四置乱面板边界。
- [x] protocol/governance/config/CLI 关键测试通过（171 项）；全项目非目标测试通过（1089 项）。
- [x] R1 文件未修改，R1/R2 路径互不重叠。
- [x] 正式目标读取和正式评分仍为 0。
