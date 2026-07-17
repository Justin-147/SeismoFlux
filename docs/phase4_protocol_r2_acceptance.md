# 阶段 4 R2 协议冻结验收记录

- 验收日期：2026-07-17
- 协议版本：`0.4.1`
- 执行修订：`r2`
- 纠正修订：`r1`
- 协议标签计划：`v0.3.1-anomaly-increment-protocol-r2`
- 评分代码标签预留身份：`v0.3.1-anomaly-increment-scoring-code-r2`；当前 R2 禁止发布
- 结果标签预留身份：`v0.3.1-anomaly-increment-r2`；当前 R2 禁止发布
- R2 执行状态：`blocked_before_target_read_etas_comparator_not_evaluable`
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
| R2 协议设计 | `fd51d7f19306c48f95d89905416b1e38b9f2ab0078b4d5078ab83857278e193d` |
| R2 随机输入封印 | `8f8bb386cd62f225bb48b30f24c2360a865cf36faa541d35f94060daf44e8f09` |
| R2 跨清单验收内容 | `fde7d1fdc41568ee87d2405d58ab42b3ed0026ba8bf2a15611af0f903be95cb2` |
| fold 清单文件 / 内容 | `74bb8c73751cab71c85e20609eb70d94902634fe946a3cb4316a83d7e9b051a1` / `d3fd999d51ad66e3ea153d1b40d24d8f861993f0337d2de297ac297711f8eaf2` |
| feature-set 清单文件 / 内容 | `89eb3d73eb1ffdb554dc4e72792aedd65473711ca061d197242a6cd560ca4675` / `65b86aa50dc18ba6cab94f264c82772d588bac4ef9c466138a87a4e8e4995211` |
| randomness 清单文件 / 内容 | `06e1c52d819d0159f9091d338c8a5e41484fabb1241284730ed22765e3a5d0ae` / `bb67f90c99d857eabf9182735016deea6e92ab4c7cfbc7a768587b117bd6f921` |
| spatial-strata 清单文件 / 内容 | `283a6790f6e7c16bc31d9498b2cc3cd043e19c8f141046afb898e988f25dcc83` / `0e8e1af89b802522150c9ef3b46a1ffe0d440e51f69ce4a532a6e1540934b5d2` |

## 本地受限空间工件

| 工件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| 25 km cell mapping | 42917 | `171a500de9f9dd475f2c37a5426debc7c6f2d34ddd418056729c39b27118108e` |
| entity mapping | 4055922 | `49cd56ace13680c3465b0c128f7dd9823636f6f1db7a2f39a12d1235df532170` |
| connectors | 11768 | `1f25120d9b9b15ec428efe97183179cebe1c3c5b0e022294dbf82f4c73e4e167` |
| zone geometry | 233171 | `c1c54d390bd1553c8f75b10def4898e24deb919345dd9ca0a11a02d0ff80ba70` |

## 必须通过的检查

- [x] R2 四份公开清单内容哈希和文件哈希有效；最终目标盲 `generate` 与严格 `check` 均通过，`target_read_count=0`。
- [x] R2 四个本地空间工件路径、字节数和 SHA 与公开摘要一致。
- [x] R1/R2 科学日历、折、特征和空间输入除版本身份外一致。
- [x] R2 PCG64 的 60 组参考向量（bootstrap/space/time=`12/24/24`）由终审封印重新派生；其数值载荷与 R1 及提交 `13e645d` 的标签前旧 R2 逐位置相同数均为 `0/60`。
- [x] ETAS 前置门当前为 `not_evaluable`；R2 将 dynamic/snapshot 门冻结为 `gate_reached=false,status=not_reached`，正式目标读取和全部评分状态为未授权。
- [x] 未来新执行修订把 `etas_background_no_increment` 锁为唯一主科学对照，绑定目标读取前冻结的 ETAS 工件、参数和数值资格 SHA；`evaluable` 状态本身绝不满足 G2，dynamic/snapshot 都必须有直接候选减 ETAS 的正式结果且该轨道通过后才可能采用异常模型。
- [x] 未来新执行修订必须继承 dynamic 与 snapshot 的 time/space `p<=0.05` 设计；R2 当前四组置乱的执行工件状态均为 `not_authorized_not_computed`。
- [x] 机器词表严格分域：科学门状态只允许 `passed`、`failed`、`evidence_insufficient`、`not_reached` 并禁止旧状态 `fail`；比较器可评价性只允许 `evaluable`/`not_evaluable`；未授权执行工件只允许 `not_authorized_not_computed`/`not_authorized_not_created`。
- [x] 未来设计中 dynamic 是唯一确认性入口；snapshot 仅在 dynamic 完整资格为 `passed` 且 G3 为 `failed`/`evidence_insufficient` 时到达，禁止 snapshot-only 救援；该设计仅在 ETAS 可评价且新修订授权后生效。
- [x] 未来设计要求四组以 ETAS 为对照的 observed raw statistic、完整置乱零分布和结果对象始终计算，并逐对象绑定 ETAS 工件/参数/资格 SHA；R2 当前不得越过 ETAS 前置门计算。
- [x] 未来候选门结果身份精确绑定 `gate_reached`、`gate_reached_reason` 和包含 time/space 结果对象 SHA 的 `gate_record_sha256`；R2 当前结果身份为 `not_authorized_not_created`。
- [x] 探索性 Holm 精确分为 dynamic/snapshot × time/space 四族，每族固定 2 个震级档 × 5 个窗口共 10 个成员。
- [x] G2 的 coverage guard 与两项预登记 practical threshold 对 dynamic/snapshot 分别独立应用。
- [x] 两项 practical 分支的分区、候选/对照、震级、窗口、面积选择、分母、无效分支处理、阈值和区间字段均由严格校验器完整锁定。
- [x] dynamic/snapshot 相对 ETAS、KDE 与 coverage 的三开发折稳定性全部成为候选完整资格硬门；KDE/coverage 是次级必过混杂保护，不能替代 ETAS。
- [x] 结果身份冻结完整 54 行 fold × variant × comparator × horizon 数值/事件计数、18 行折宏值/状态和 6 行候选对照汇总状态。
- [x] 两候选相对 ETAS、KDE 与 coverage 的跨区残余、95% 区间及至少两个正贡献事件区全部成为候选完整资格硬门。
- [x] 区域结果身份精确冻结 702 行窗口明细、234 行区域宏、6 行轨道汇总与 12,000 行 bootstrap；分母、贡献可复算字段、区域/全局事件数求和、零目标分母的 nullable recall 语义、事件 ID 去重并集、唯一最强区和每轨 `0..1999` 复制索引均已锁定。
- [x] G3 `failed` 与 `evidence_insufficient` 都进入同一冻结条件回退分支，不允许事后补规则。
- [x] 四个未来正式 placebo 请求和四个 checkpoint 身份均已预登记；R2 当前不创建 checkpoint。
- [x] 未来结果身份显式绑定 ETAS 工件/参数/数值资格、事件项与补偿积分对象、dynamic/snapshot 直接减 ETAS 的 G2 结果、含 ETAS 行的跨折/跨区表，以及四个逐一绑定 ETAS SHA 的 placebo 结果对象及完整零分布；R2 当前不创建结果身份。
- [x] CPU 上限为 6、BLAS=1、禁止嵌套并行、低优先级要求已冻结。
- [x] 协议标签阶段 `expected_test_count=UNFROZEN`；评分代码与非目标测试最终确定后再冻结真实 JUnit 整数，禁止复用协议验收快照数。
- [x] 本地受限空间工件及目标型回溯文件的访问权限凭据已列为目标读取/发布前硬门；五路径同句柄双采样、Windows 无继承 DACL 且 `INHERIT_ONLY_ACE` 不得证明当前对象 Full Control、POSIX 本地经典文件系统且无 ACL xattr 均 fail-closed。
- [x] 当前五个本地路径的只读实测按 ACL v3 正确 fail-closed（`Windows DACL still inherits access entries`），没有生成权限凭据；这不阻断本次 blocked-protocol 标签，但未来新执行修订在任何目标读取前必须先完成实际 ACL 硬化并重新验证。
- [x] qualification、formal-preflight receipt/resource、脚本 writer 和 canonical ledger 的公开生产 API 均以完整 R2 协议中央 guard 作为首个可执行动作；当前 blocked R2 在任何路径转换、resolve/stat/open/hash/mkdir/tempfile/lock 或文件创建前停止。
- [x] 无协议参数的评分入口已从公共生产 API 移除；私有评分 core 不在 `__all__`，仓库生产调用图仅允许 `FormalRunSession.execute` 在首语句精确执行 `formal_scoring` guard 后调用，合成测试 helper 不进入 wheel。该治理边界防止公共 API 和仓库生产调用误入，不把同进程恶意 Python（其本可直接文件 I/O）误称为安全沙箱。
- [x] 回溯 ACL 契约冻结凭据路径、父目录写前限制、零字节文件写前核验、同一持有句柄/文件身份、最终文件与 ACL descriptor SHA 绑定、禁止普通临时文件先写目标字节，且文件和凭据不得进入公开 bundle。
- [x] 地震目录 `origin_time`/`available_at` 覆盖截止 `2026-07-09T04:25:56Z`，覆盖全部冻结验证窗终点，缺失或不足时 fail-closed。
- [x] 发布限制明确：目录时效假设乐观、bootstrap 不含重拟合不确定性、ETAS 不可评估，R2 不授权任何正式增量声明；KDE 仅是目标盲诊断设计，不能替代 ETAS。
- [x] 四个空间输出物理隔离、采用模型表、中心点警示和轴刻度要求已冻结。
- [x] public/forecast 工件冻结结构化目标载荷拒绝列表，关键词扫描或 UI 隐藏不能替代。
- [x] 展示语义冻结 ETAS 主对照、KDE/coverage 次级选项、聚合回溯、百分位、相对强度、采用卡、日历地标、采用变体空间图与四置乱面板边界。
- [x] protocol/governance/config/CLI、ACL、hard-stop、formal-run 和 Windows Parquet 关键测试通过；其最终状态同时包含在下述全项目 JUnit 中。
- [x] 全项目非目标测试通过：最终复跑为 `1190 passed`、`0 failed`、`0 errors`、`0 skipped`，JUnit 为本地 `data/interim/stage4/anomaly_increment_r2/runtime_logs/protocol-r2-final-full-nontarget-20260717-rerun1.junit.xml`，文件 SHA-256=`a7a209c98fc0cfaa1b3be514b5d8ec199db4fcd38f0736d2f9e17bd8b44bda57`。该计数仅是协议验收快照，不得冻结为评分代码计数；真实 scoring-freeze JUnit 只能由未来新执行修订冻结。
- [x] `runtime_logs` 中此前失败/中断的目标盲测试日志保持原样作为历史证据；最终 JUnit 只用于本次验收，不删除、不覆盖旧日志。
- [x] R1 文件未修改，R1/R2 路径互不重叠。
- [x] 正式目标读取和正式评分仍为 0。
