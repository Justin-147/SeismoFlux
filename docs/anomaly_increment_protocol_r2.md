# 阶段 4 可解释异常增量模型：协议修订 R2

## 1. 身份与边界

- 科学协议版本：`0.4.1`。
- 执行修订：`r2`，纠正 `r1`，不覆盖或改写 R1。
- 最终目标盲冻结日：`2026-07-17`；此前的 R2 提交未打协议标签，经终审修正后才构成可签署协议。
- 机器协议：`configs/anomaly_increment_r2.yaml`。
- 协议标签：`v0.3.1-anomaly-increment-protocol-r2`。
- 评分代码标签预留身份：`v0.3.1-anomaly-increment-scoring-code-r2`；R2 禁止发布。
- 结果标签预留身份：`v0.3.1-anomaly-increment-r2`；R2 禁止发布。

本修订发生在任何阶段 4 目标读取或正式评分之前。终审确认冻结 ETAS 比较器仍为 `not_evaluable`，所以 R2 只授权目标盲的协议、清单、受限空间工件和测试；它明确阻断 qualification、seal、ledger、评分代码标签、正式目标读取和正式评分。只有另立目标盲 ETAS 数值修复协议、得到可评价且冻结的 ETAS 比较器并发布新的执行修订后，才可恢复阶段 4 评分路线。

## 2. R1 的不可变废止证据

R1 的 formal-attempt ledger 与 target-read ledger 均为 0 条记录，目标字节从未读取，正式评分从未运行。R1 文件和标签继续作为历史证据，不得移动、删除、覆盖或用于 R2 授权。

| R1 证据 | SHA-256 / 状态 |
| --- | --- |
| 规范化协议设计 | `c15d3bbca5cef4b363a79e183d715124256a12088873d81cd77de489766b32de` |
| scoring seal 文件 | `a6e8dc9ac283813edb62e301114d4985ae332b9c607584c987a4297efe5978f3` |
| formal-attempt ledger 文件 | `9ac5e5e080c1d5425f985cb3091b94c0da69d211469589d26ae8bfc314088142` |
| formal-attempt ledger 内容 | `cadc80e5a0f00ffce241f910409750b01e3f410d910dda5d5aad0ff3033d2448`；`records=[]` |
| target-read ledger 文件 | `0a49450cc1006ccd0ced26fba30330417f1ec8667c5cedf0bf04242f158210c8` |
| target-read ledger 内容 | `4c1fb843edfa8f59f37f137d8d68962cbdae5991cc8809302d39d240f0b395b6`；`records=[]` |

## 3. R2 科学门槛修正

ETAS 是唯一实施蓝图要求的主对照。R2 冻结时它因数值稳定性失败而没有合格参数，状态为 `not_evaluable`；这不是可由 KDE 替代的比较结果。该状态优先于下述所有候选门和采用矩阵：G2 只能为 `evidence_insufficient`/`not_reached`，必须保留背景模型，且不得读取正式目标来弥补缺失比较器。下述 G2/G3、placebo 与采用规则是 ETAS 前置门修复后新执行修订必须继承的预登记设计，不构成 R2 的评分授权。

未来新执行修订必须把 `etas_background_no_increment` 作为唯一主科学对照，同时保留 `kde_background_no_increment` 与 `coverage_only` 两条次级必过混杂保护轨道，再评估 `snapshot` 和 `dynamic`；KDE 和 coverage 均不得替代 ETAS。ETAS 的模型工件、参数快照和数值资格凭据必须在目标读取前冻结并逐项绑定 SHA-256，单有 `evaluable` 状态绝不构成 G2 通过。在 ETAS 前置门已通过后，G2 对 `dynamic` 与 snapshot-equivalent 分支都必须执行同一组四个正式 placebo 请求，主统计量统一为候选减 ETAS，且每个 observed/null 结果对象都绑定同一 ETAS 身份。R2 当前四组请求的执行工件状态是 `not_authorized_not_computed`：

1. `time × dynamic`；
2. `space × dynamic`；
3. `time × snapshot`；
4. `space × snapshot`。

三个机器词表严格分域：科学门状态只允许 `passed`、`failed`、`evidence_insufficient`、`not_reached`，历史拼写 `fail` 禁止；比较器可评价性只允许 `evaluable`、`not_evaluable`；尚未获授权的执行工件状态只允许 `not_authorized_not_computed`、`not_authorized_not_created`，不得冒充科学门结论。在未来 ETAS 前置门已通过、且由新执行修订授权的正式执行中，四个请求的 observed raw statistic、完整零分布及结果对象无论序贯门是否到达都必须计算并绑定。R2 当前在目标读取前停止，因此不得为了满足“总是计算”而越过 ETAS 前置门。

两种候选各自只有在时间置乱和空间置乱的单侧 Monte Carlo 值都满足 `p<=0.05` 时，才满足置乱证据条件。这是同时必须成立的 intersection-union 条件，报告用联合值为 `max(p_time,p_space)`。任一请求缺失、失败比例超限或 p 值不合格，均不得把对应模型判为通过。四个请求共享预登记的配对映射语义，但每个模型必须重新拟合自身的置乱模型；不得用 dynamic 的空间结果替代 snapshot。

G2 的主信息增益和两项预登记 practical-improvement 门槛必须对 `dynamic`、`snapshot` 分别以冻结 ETAS 为直接对照计算和判定；候选减 ETAS 的宏观信息增益 95% 下界必须大于 0。reporting-coverage 混杂保护另外要求当前候选相对 `coverage_only` 的宏观信息增益 95% 下界大于 0。协议中的 `current_evaluated_candidate_variant` 指当前分支，不得固定解释为 dynamic，也不得用 dynamic 的实用改进替 snapshot。两项 practical 分支均在成绩前完整冻结，不得只校验候选名称：第一分支是 `M5_6`、7/30/90 天等权、600,000 km² 精确完整单元前缀下相对 ETAS 的严格召回宏增益至少 5 个百分点，所有研究区目标为分母；第二分支是在 625 km² 冻结预算网格、无插值、ETAS 在 600,000 km² 的严格命中数为逐窗口参照下，同召回并集面积宏降幅至少 10%。零参照命中、960,000 km² 仍不可达或 bootstrap 分支无效都按预登记失败值处理而不得删除。两分支各自还要求双侧 95% 区间下界大于 0；两个单侧错误上界的并集不超过 0.05，禁止事后新增第三项指标。

`dynamic` 始终是唯一 H1/G2 确认性入口；只有 ETAS 前置门为 `evaluable` 且候选减 ETAS 的直接 G2 结果通过时，dynamic 才可进入 `gate_reached=true`、原因为 `primary_confirmatory_candidate` 的正式门。R2 当前 ETAS 前置门未通过，所以所有候选 qualification 均为 `not_reached`。在未来合规执行中，`snapshot` 仍不是可以与 dynamic 平行择优的第二个 alpha 入口；snapshot qualification 当且仅当 dynamic 完整资格状态为 `passed` 且 G3 为 `failed` 或 `evidence_insufficient` 时 `gate_reached=true`。其余路径一律记录 `gate_reached=false`、qualification `status=not_reached`，已计算的 snapshot raw/placebo 仅为 diagnostic，不得形成确认性结论。每个候选门记录必须把 `gate_reached`、`gate_reached_reason`、ETAS 工件/参数/数值资格 SHA、正式 G2/开发折/跨区结果 SHA，以及 time/space placebo 结果对象 SHA 一并纳入 `gate_record_sha256`，不能只绑定最终状态。

探索性 Holm 校正精确冻结为四个 candidate × kind 家族：`dynamic_time`、`dynamic_space`、`snapshot_time`、`snapshot_space`。每族成员键是 `magnitude_bin × horizon_days` 的完整笛卡尔积，震级档固定为 `M5_6`、`M6_plus`，窗口固定为 7/30/90/180/365 天，因此每族恰为 10 个成员；不得跨候选或跨置乱类型合并家族，也不得删除证据不足成员。G2 主宏端点不进入这些探索家族。

候选的“完整资格”不再只等于一次正式验证分数，而要求以下四部分全部通过：

1. 可评价且在目标读取前冻结工件、参数和数值资格凭据的 ETAS 比较器；
2. 直接候选减 ETAS 的正式验证 G2 核心；
3. 开发期跨折增量稳定性；
4. 正式验证跨区稳定性。

开发期跨折门使用三个已经冻结且目标带互斥的滚动折，只评估 `M5_6` 的 7/30/90 天等权宏平均。dynamic 与条件回退 snapshot 都必须分别相对 `etas_background_no_increment`、`kde_background_no_increment`、`coverage_only` 达到至少 2/3 折点估计为正且三折中位宏观信息增益大于 0；三条比较轨道都必须通过，KDE/coverage 是次级必过混杂保护而不是 ETAS 替代品。任一折任一主窗口零评分事件时，该候选为证据不足，禁止用部分窗口或随机切分补救。结果身份必须保存完整的 3 折 × 2 候选 × 3 对照 × 3 窗口共 54 行信息增益和受支持唯一物理事件计数，以及 18 行折宏值、正窗口数和状态、6 行候选 × 对照的正折数、中位宏值和状态；缺行或缺值即 `evidence_insufficient`。G3 仍只回答 `dynamic - snapshot` 是否跨折稳定，不能替代候选相对非异常对照的跨折门。

跨区门使用目标无关、在评分前固定的 39 个 construction zones，顺序按 `construction_zone_id`，目标不得创建、合并、拆分或重排区域。对候选 `v`、对照 `c`、区域 `r`、窗口 `h` 定义可加和贡献：

\[
C_{v,c,r,h}=\frac{\sum_{e\in r}\log\{\lambda_v(e)/\lambda_c(e)\}-\int_r(\lambda_v-\lambda_c)}{N_h},
\]

其中 `N_h` 是该窗口全局受支持的唯一物理事件数，并在同一窗口的所有区域、候选和对照间完全相同；每个窗口的 39 个区域受支持唯一物理事件数之和还必须精确等于该全局数。三个主窗口等权得到区域宏贡献，全部区域之和必须在 `1e-12` 绝对容差内等于对应全局宏观信息增益。每条候选 × 对照轨道的主统计量为 `global_macro_IG - max_region_contribution`；最大区并列时按冻结区域 ID，2,000 次联合物理事件 bootstrap 的每次复制都重新选择最大贡献区，补偿积分固定。候选相对 ETAS、KDE 和 coverage 三条轨道都必须满足：该残余点估计大于 0、95% 下界大于 0，并至少有两个“含至少一个受支持唯一物理事件且区域宏贡献大于 0”的区域。区域映射不完整、窗口不可评估或少于两个可评价事件区均记为证据不足，不得采用该候选，也不得声称稳定增量。未来结果身份精确冻结 702 行候选 × 对照 × 39 区 × 3 窗口明细、234 行区域宏值、6 行轨道汇总和 12,000 行轨道 × bootstrap 复制明细。窗口明细同时保存区域事件对数强度差、区域补偿积分差、区域和全局受支持唯一物理事件数、区域内全研究区目标数、全局全研究区目标数、候选/对照命中数、召回与召回可评价性；严格召回统一在 `600,000 km²` 报警面积预算下计算。区域 strict recall 使用本 construction zone 内的目标数为分母，该分母只在同一 region × horizon 的候选/对照间共享；分母大于 0 时召回必须等于命中数除以分母，分母为 0 时两侧命中数必须为 0、两侧召回必须为 `null`、可评价性必须为 `not_evaluable_no_region_targets`，且这一诊断空值不得使信息增益轨道自动成为证据不足。全局目标数在同一窗口的各区和各轨道间共享，39 个区域分母之和必须等于全局分母。区域宏事件数按 7/30/90 天物理事件 ID 精确去重并集。每条轨道必须恰有一个 `is_strongest_region=true`，且与冻结并列规则及轨道汇总 ID 一致。bootstrap 的 `replication_index` 对每条轨道必须是无缺失、无重复、无额外值的整数 `0..1999`。

上述结果身份仅适用于 ETAS 已可评价且另有新执行修订授权的未来执行；R2 当前状态为 `not_authorized_not_created`。届时两种候选的正式 G2 核心、完整跨折明细及宏状态、跨区贡献表与残余 bootstrap 分布、四组置乱结果、G3、固定序贯门的 reached/reason/hash、最终采用决定和“最终采用模型效果表”都必须进入结果身份、注册表、报告、模型卡、静态图和交互页。结果身份必须首先绑定 ETAS 工件、参数快照、数值资格凭据、事件项与补偿积分评分对象，以及 dynamic/snapshot 各自直接减 ETAS 的正式 G2 结果；开发折和跨区表必须含 ETAS 对照行。`time_dynamic_placebo_result_distribution`、`space_dynamic_placebo_result_distribution`、`time_snapshot_placebo_result_distribution` 与 `space_snapshot_placebo_result_distribution` 四个结果对象及完整零分布都必须绑定同一 ETAS SHA，不得只用最终 p 值间接引用。任何异常采用都要求直接 ETAS 轨道通过；若 ETAS 可评价但候选减 ETAS 不通过，则保留在阶段 4 目标读取前已冻结选择证据的最佳非异常背景，禁止利用阶段 4 目标或异常结果改选。dynamic 主候选审计表与最终采用模型表必须物理和语义上可区分；未采用候选不得标成当前预测。

## 4. 随机流和输入一致性

`protocol_version`、协议设计哈希和随机输入封印都进入类型化随机上下文，因此 R2 必须重新生成全部 PCG64 参考向量和正式随机流。任何 R0/R1 randomness、checkpoint 或置乱结果都不得复用。

R2 的科学日历、折、目标盲特征定义、预处理、背景输入、研究区和空间构造输入必须与 R1 一致；允许的差异仅限 R2 版本身份、独立路径、R2 parquet contract metadata、由新设计封印派生的哈希与随机参考向量。生成后必须用机器检查证明这一点。

R2 继续继承 `arrow_ipc_selected_table_logical_identity_r1` 算法。这里的 `r1` 是逻辑身份算法版本，不是 R1 执行命名空间，禁止误改。

## 5. 资源冻结

- 正式后端仍为 CPU float64；未经独立等价门控不得切换 GPU。
- 最多 6 个逻辑 CPU，进程优先级为 Windows `BelowNormal`。
- BLAS 每个工作线程最多 1 线程，禁止内外层嵌套并行。
- BLAS 环境必须在导入 NumPy/SciPy 前设置，并在目标读取前回读 affinity、优先级和实际线程池状态形成资源凭据。
- 所有资格测试、清单生成和未来正式运行都必须遵守该上限。
- 协议 YAML 的 `expected_test_count` 始终精确为委托标记字符串 `UNFROZEN`；只有未来新执行修订的评分代码实现和全部非目标测试最终确定后，才可把源码常量与类型化 qualification receipt 一次性冻结为真实 JUnit 整数。不得把协议 YAML 改成整数，也不得让本次协议验收快照数冒充评分冻结计数。

## 6. 地震目录覆盖冻结与解释边界

地震目录的实际覆盖证据冻结自 `docs/data_quality_report.md`：`origin_time` 与 `available_at` 的最大值均为 `2026-07-09T04:25:56Z`。R2 冻结验证窗的最晚终点为 `2025-07-18T16:00:00Z`。在获得目标读取授权后、计算第一个正式分数之前，运行时必须确认目录的两个最大时间都覆盖每一个冻结验证窗终点；字段缺失、时间不可解析或覆盖不足时必须 fail-closed，登记无效尝试且不得评分。

结果解释同时受以下限制：

- 当前 `available_at=origin_time` 是乐观的目录时效假设，不代表真实发布延迟已经建模。
- bootstrap 区间是在已拟合模型固定条件下的条件区间，不包含重新拟合带来的参数与模型选择不确定性。
- ETAS 比较器状态为 `not_evaluable`，因此 R2 不得读取阶段 4 正式目标、判定 G2、采用异常模型或声称任何正式增量；KDE 不能替代 ETAS。必须先完成单独的目标盲 ETAS 数值修复并发布新执行修订。

## 7. 空间输出物理隔离

R2 冻结四个互不相同的空间文件：

| 类型 | 静态 | 交互 |
| --- | --- | --- |
| 目标盲预测 | `outputs/visualizations/anomaly_increment_r2_forecast_spatial.svg` | `outputs/visualizations/anomaly_increment_r2_forecast_spatial.html` |
| 本地受限回溯目标叠加 | `outputs/visualizations/anomaly_increment_r2_retrospective_target_local.svg` | `outputs/visualizations/anomaly_increment_r2_retrospective_target_local.html` |

预测文件不得嵌入、延迟加载或自动跳转加载目标载荷，不得包含事件 ID、目标坐标、命中结果或目标型控制。回溯文件必须明确标记“本地受限、含目标”，保持 Git 忽略；公开注册表只能记录本地路径和 SHA，不能嵌入内容。四个受限空间输入工件在目标读取前必须是普通、非 reparse 文件并通过访问权限凭据：Windows 仅当前进程用户、LocalSystem 与内置管理员可访问且禁止继承 ACE，标记为 `INHERIT_ONLY_ACE` 的许可不得充当当前对象的用户 Full Control 证明；POSIX 仅接受可证明为本地经典权限模型的文件系统，文件/目录分别为 owner-only `0600/0700`，且任何 ACL 相关 xattr 都必须为空。五个路径都必须用同一保留句柄在进入和退出时重复采样 owner、ACL/权限和文件身份，两次规范描述符必须逐字节一致；无法验证时 fail-closed。

冻结时对当前五路径的只读实测因 Windows DACL 仍含继承条目而按预期 fail-closed，未生成权限凭据。R2 本身已在 ETAS 前置门处永久阻断目标读取，因此该现场状态不阻断 blocked-protocol 标签；任何未来新执行修订都必须先实际硬化这些路径、重新通过同一 ACL v3 验证并把凭据绑定到资格与执行封印，才可读取目标。

目标型回溯写入采用预登记的持有句柄流程，凭据固定为 `outputs/visualizations/anomaly_increment_r2_retrospective_acl_receipt.json`：先限制父目录 `outputs/visualizations`，再创建零字节目标文件并核验其 ACL；从该次核验到最终写完必须持续持有同一已核验句柄，写前后平台文件身份必须一致。普通临时文件不得先接收任何目标字节，也不得从未受限或未核验临时文件原子替换目标。每个回溯文件的凭据必须绑定相对路径、零字节核验身份、最终身份、最终文件 SHA-256、最终 ACL descriptor SHA-256 和父目录 ACL descriptor SHA-256。回溯文件及该凭据均不得进入公开 bundle；无法满足任一步即 fail-closed。

每个可发布 forecast/public 工件在发布前必须解析静态 SVG/XML DOM，并递归反序列化交互载荷；随后拒绝 `local_restricted`、`target_bearing` 分类以及 `event_id`、目标/震中坐标、`hit_status`、`target_marker` 等字段。关键词扫描或把控件在 UI 中隐藏都不能替代结构校验；任一命中必须 fail-closed 并禁止发布。

在真实裁剪单元几何尚未实现时，任何中心点图都必须醒目标注“中心点示意，非面积几何；报警面积以数值为准”。无支持区域必须标为“未评估或无支持，不代表低强度”。Molchan 横轴固定为 `0–1`，固定面积曲线横轴固定为 `0–960000 km²`；所有图必须有数值刻度、轴名和完整图例，任何截断轴必须明确标注。

展示语义也属于发布契约：`etas_background_no_increment` 必须是主科学对照，`kde_background_no_increment` 与 `coverage_only` 必须作为可见次级比较选项；聚合回溯视图必须隐藏或禁用 issue/model 控件，并标为“全部 N 个起报日汇总”。`100%` 峰值只能称为“峰值网格百分位”，不得称预测准确率；`relative_strength` 只定义为峰值积分网格强度除以平均积分网格强度。采用卡和采用变体必须显示，最新回溯地标必须称“最新冻结日历地标”，不得冒充当前预测。forecast 空间图必须渲染最终采用变体；未采用的 dynamic 只能标为研究候选。静态图和交互页都必须展示两候选相对 ETAS/KDE/coverage 的三折稳定性、区域贡献和“去掉最强区后的残余”区间。time/space × dynamic/snapshot 四个置乱静态面板必须全部位于渲染边界内，并有专门的边界测试。

## 8. 双冻结顺序

1. 生成并校验 R2 四份公开清单和四个本地受限空间输入工件。
2. 提交、推送并发布协议标签 `v0.3.1-anomaly-increment-protocol-r2`，该标签记录 `blocked_before_target_read_etas_comparator_not_evaluable`，不授权评分。
3. 返回背景阶段，另立目标盲 ETAS 数值修复协议；不得利用阶段 4 正式目标选择修复方案。
4. 只有 ETAS 比较器数值资格通过并冻结后，才可发布新的阶段 4 执行修订，并在其中实现 evaluation、scoring、deliverables 和 spatial 科学代码。
5. R2 预留的评分代码标签、qualification、seal、双台账和结果标签不得创建；R2 正式目标读取与正式评分保持为 0。

阶段 9 锁定测试在阶段 4 继续无条件禁止。
