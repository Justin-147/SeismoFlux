# 阶段4可解释异常增量模型：执行修订 R1

## 1. 身份与适用范围

- 科学协议版本：`0.4.0`，保持不变。
- 执行修订：`r1`。
- 上位科学协议：[anomaly_increment_protocol.md](anomaly_increment_protocol.md)。
- 唯一实施蓝图：[SEISMOFLUX_IMPLEMENTATION_HANDOFF.md](../SEISMOFLUX_IMPLEMENTATION_HANDOFF.md)。
- 机器契约：[anomaly_increment_r1.yaml](../configs/anomaly_increment_r1.yaml)。
- R0 就绪性事故：[phase4_scoring_readiness_incident_r0.md](phase4_scoring_readiness_incident_r0.md)。

本修订只纠正目标读取前发现的 Arrow 列身份实现缺陷，不改变科学问题、输入数据、阶段3特征定义、模型变体、预处理、超参数、折、目标窗口、背景、ETAS 状态、置乱方法、随机根种子、G2/G3 门槛或锁定测试规则。R1 继续使用相同的 PCG64、根种子和类型化派生算法，但协议设计与输入封印身份改变，因此实际 R1 随机流和参考向量将从新封印重新派生并在 R1 randomness 清单中冻结；不得复用或挑选 R0 流。R0 未读取阶段4目标、未产生模型得分，也未消耗正式验证次数。

## 2. 不可覆盖的 R0 证据

下列 R0 标签和本地证据只作为历史审计记录，不得移动、删除、覆盖或授权 R1：

- `v0.3.0-anomaly-increment-protocol`；
- `v0.3.0-anomaly-increment-scoring-code`；
- R0 formal-preflight receipt、qualification、scoring seal；
- R0 formal-attempt ledger 与 target-read ledger。

R1 使用独立配置、清单、本地执行根、封印、台账、检查点、模型包、报告和可视化路径。

## 3. R1 双冻结门

R1 协议标签为 `v0.3.0-anomaly-increment-protocol-r1`，评分代码标签为 `v0.3.0-anomaly-increment-scoring-code-r1`，结果标签为 `v0.3.0-anomaly-increment-r1`。

双冻结严格按以下顺序执行：

1. 完成 R1 协议、机器配置、随机清单、事故记录和协议测试；
2. 提交并推送 R1 协议，CI 通过后发布 `v0.3.0-anomaly-increment-protocol-r1`；
3. 该标签存在后才允许实现 R1 评分代码和逻辑身份；
4. 评分代码完成全部非目标测试、153 期逐位复演和跨平台 CI 后，提交、推送并发布 `v0.3.0-anomaly-increment-scoring-code-r1`；
5. 评分标签存在后才允许生成 R1 formal-preflight receipt、qualification、双台账和 scoring seal，并只运行 target-blind readiness check；
6. readiness 全绿且双台账仍为 0 后，才允许唯一正式运行读取阶段4目标。

锁定测试仍属于阶段9，本阶段禁止运行。

## 4. Arrow 逻辑列身份 R1

R0 的 legacy 物理 IPC 哈希保留用于历史回放，不改变其语义，也不得用于新的阶段4执行身份。R1 新增 `arrow_ipc_selected_table_logical_identity_r1`，SHA-256 域分隔为 ASCII `seismoflux.selected-table-logical-identity.r1` 后接一个 NUL 字节；机器契约逐项冻结其规范化边界。

R1 仅规范化两类与科学内容无关的表示差异：

1. 排除 Arrow 表的顶层 schema metadata；
2. 将 null 槽位下未定义的 payload 规范为零值。

以下内容仍必须严格进入身份并逐位保持：列名与顺序、Arrow 类型、nullability、字段级 metadata、有效性位图，以及全部有效值 payload。有效的 `+0.0` 与 `-0.0`、不同有效 NaN payload、字段 metadata 变化、类型变化、列顺序变化或任一有效性位变化都必须产生不同身份。布尔数据和有效性位图长度以外的 padding 位必须规范化，分块与切片布局不得改变逻辑身份。未显式支持的嵌套、字典和扩展类型必须失败关闭。

评分代码 R1 完成后，153 个正式范围历史起报日必须在 1 个与 2 个空间工作进程下用最终版本化哈希逐列、逐位复演一致。若发现任何有效 payload、有效性、字段、类型或列序差异，R1 必须停止并先建立阶段3修订版本；不得用身份规范化掩盖科学差异。R0 事故诊断中的 2-worker Arrow logical replay 只用于证明没有阶段3科学差异，不能替代这道最终门。

## 5. 计算后端与 GPU

正式 R1 仍以冻结的 CPU float64 路径为权威。当前 GPU 没有在评分代码冻结前通过既定等价性门，因此不得改变正式模型特征、目标函数、超参数、随机流或数值结果。GPU 只可在后续独立、目标盲的等价性验证通过后用于可选加速；失败时保持 CPU-only。

## 6. 输出与解释边界

无论 G2/G3 通过、失败或证据不足，都必须生成 R1 注册表、报告、模型卡、静态图和本地交互页面。静态图继续覆盖数据与模型流程、效应曲线、时空置乱、信息增益区间、地区×窗口、Molchan 和固定面积召回。交互页面必须区分回溯与真正预测；真正预测模式物理排除目标覆盖层，历史目标只能在回溯模式中单独显示。

所有输出只能称为条件强度、期望事件强度、相对强度或顺位，不得称为绝对发震概率。任何前瞻预测必须使用独立版本目录永久归档，不得按后来是否命中选择保留版本。
