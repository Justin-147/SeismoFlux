# 阶段 4 可解释异常增量模型：协议修订 R2

## 1. 身份与边界

- 科学协议版本：`0.4.1`。
- 执行修订：`r2`，纠正 `r1`，不覆盖或改写 R1。
- 机器协议：`configs/anomaly_increment_r2.yaml`。
- 协议标签：`v0.3.1-anomaly-increment-protocol-r2`。
- 评分代码标签：`v0.3.1-anomaly-increment-scoring-code-r2`。
- 结果标签：`v0.3.1-anomaly-increment-r2`。

本修订发生在任何阶段 4 目标读取或正式评分之前。它只授权目标盲的协议、清单、受限空间工件、测试和后续评分代码实现；本提交本身不得读取正式目标、运行正式评分或产生正式科学结论。

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

R2 仍比较 `background_no_increment`、`coverage_only`、`snapshot` 和 `dynamic`，但 G2 对 `dynamic` 与 snapshot-equivalent 分支都必须执行同一组四个正式 placebo 请求：

1. `time × dynamic`；
2. `space × dynamic`；
3. `time × snapshot`；
4. `space × snapshot`。

两种候选各自只有在时间置乱和空间置乱的单侧 Monte Carlo 值都满足 `p<=0.05` 时，才满足置乱证据条件。任一请求缺失、失败比例超限或 p 值不合格，均不得把对应模型判为 G2 通过。四个请求共享预登记的配对映射语义，但每个模型必须重新拟合自身的置乱模型；不得用 dynamic 的空间结果替代 snapshot。

G2 的 reporting-coverage 混杂保护与两项 practical-improvement 门槛也必须对 `dynamic`、`snapshot` 分别计算和分别判定：当前被评估候选必须相对 `coverage_only` 的宏观信息增益 95% 下界大于 0，并独立满足同一组实用改进阈值。协议中的 `current_evaluated_candidate_variant` 指当前分支，不得固定解释为 dynamic，也不得用 dynamic 的实用改进替 snapshot 通过 G2。

snapshot 的完整 G2、四组置乱结果、G3、最终采用决定和“最终采用模型效果表”都必须进入结果身份、注册表、报告、模型卡、静态图和交互页。结果身份必须显式绑定 `time_dynamic_placebo_result_distribution`、`space_dynamic_placebo_result_distribution`、`time_snapshot_placebo_result_distribution` 与 `space_snapshot_placebo_result_distribution` 四个结果对象及其完整零分布，不得只用最终 p 值间接引用。dynamic 主候选审计表与最终采用模型表必须物理和语义上可区分；未采用候选不得标成当前预测。

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

## 6. 地震目录覆盖冻结与解释边界

地震目录的实际覆盖证据冻结自 `docs/data_quality_report.md`：`origin_time` 与 `available_at` 的最大值均为 `2026-07-09T04:25:56Z`。R2 冻结验证窗的最晚终点为 `2025-07-18T16:00:00Z`。在获得目标读取授权后、计算第一个正式分数之前，运行时必须确认目录的两个最大时间都覆盖每一个冻结验证窗终点；字段缺失、时间不可解析或覆盖不足时必须 fail-closed，登记无效尝试且不得评分。

结果解释同时受以下限制：

- 当前 `available_at=origin_time` 是乐观的目录时效假设，不代表真实发布延迟已经建模。
- bootstrap 区间是在已拟合模型固定条件下的条件区间，不包含重新拟合带来的参数与模型选择不确定性。
- ETAS 比较器状态为 `not_evaluable`，因此阶段 4 只能声称相对冻结 KDE 背景的增量；不得声称优于 ETAS，也不得把 KDE 结果包装成 ETAS 替代结论。

## 7. 空间输出物理隔离

R2 冻结四个互不相同的空间文件：

| 类型 | 静态 | 交互 |
| --- | --- | --- |
| 目标盲预测 | `outputs/visualizations/anomaly_increment_r2_forecast_spatial.svg` | `outputs/visualizations/anomaly_increment_r2_forecast_spatial.html` |
| 本地受限回溯目标叠加 | `outputs/visualizations/anomaly_increment_r2_retrospective_target_local.svg` | `outputs/visualizations/anomaly_increment_r2_retrospective_target_local.html` |

预测文件不得嵌入、延迟加载或自动跳转加载目标载荷，不得包含事件 ID、目标坐标、命中结果或目标型控制。回溯文件必须明确标记“本地受限、含目标”，保持 Git 忽略；公开注册表只能记录本地路径和 SHA，不能嵌入内容。

每个可发布 forecast/public 工件在发布前必须解析静态 SVG/XML DOM，并递归反序列化交互载荷；随后拒绝 `local_restricted`、`target_bearing` 分类以及 `event_id`、目标/震中坐标、`hit_status`、`target_marker` 等字段。关键词扫描或把控件在 UI 中隐藏都不能替代结构校验；任一命中必须 fail-closed 并禁止发布。

在真实裁剪单元几何尚未实现时，任何中心点图都必须醒目标注“中心点示意，非面积几何；报警面积以数值为准”。无支持区域必须标为“未评估或无支持，不代表低强度”。Molchan 横轴固定为 `0–1`，固定面积曲线横轴固定为 `0–960000 km²`；所有图必须有数值刻度、轴名和完整图例，任何截断轴必须明确标注。

展示语义也属于发布契约：`coverage_only` 必须作为可见比较选项；聚合回溯视图必须隐藏或禁用 issue/model 控件，并标为“全部 N 个起报日汇总”。`100%` 峰值只能称为“峰值网格百分位”，不得称预测准确率；`relative_strength` 只定义为峰值积分网格强度除以平均积分网格强度。采用卡和采用变体必须显示，最新回溯地标必须称“最新冻结日历地标”，不得冒充当前预测。forecast 空间图必须渲染最终采用变体；未采用的 dynamic 只能标为研究候选。time/space × dynamic/snapshot 四个置乱静态面板必须全部位于渲染边界内，并有专门的边界测试。

## 8. 双冻结顺序

1. 生成并校验 R2 四份公开清单和四个本地受限空间工件。
2. 提交、推送并发布协议标签 `v0.3.1-anomaly-increment-protocol-r2`。
3. 标签存在后，才允许实现 evaluation、scoring、deliverables 和 spatial 的 R2 科学代码。
4. 完成目标盲测试、1/2/6 workers 一致性、四请求 checkpoint/resource 资格和跨平台 CI 后，提交、推送并发布评分代码标签。
5. 重新生成 R2 preflight、qualification、空双台账和 scoring seal；readiness 全绿且台账仍为 0/0 后，才允许唯一正式运行。

阶段 9 锁定测试在阶段 4 继续无条件禁止。
