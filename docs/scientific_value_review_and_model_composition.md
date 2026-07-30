# 科学价值复审与模型组合路线

首次形成：2026-07-29

阶段 4A 目标盲修订：2026-07-30

状态：阶段 4A 0.4.4 在真实目标读取前触发基础性 P0，异常增量路线已硬停止

## 1. 最终科学目标

模型名称和工程形式都从属于同一个目标：

> 在相同报警面积、相同支持域和严格未来隔离条件下，提高未来独立地震区域召回；或在达到相同召回时减少报警面积。

历史多发区不是需要消除的噪声。长期空间 KDE、ETAS/Hawkes、动态异常和长期构造先验都可以成为最终组合的一部分。只要滚动验证证明总效果更好，就允许采用简单加权、堆叠或其他可解释组合。

## 2. ETAS 资格的有限角色

当前五快照 × 五起点资格只判断这一份 ETAS 实现能否稳定、可重复地计算。它属于
`necessary_enabler`，不是预测效果验收，也不是整个项目能否继续的唯一门。

Q0 协议及其唯一尝试保持字节级不变，Q2 仍只输出 `evaluable` 或
`not_evaluable`：

- `evaluable`：ETAS 可以作为一个候选预测成分和异常增量主对照；后续仍须与已经通过 G1-LS 的 75 km KDE 在同一滚动验证中比较。
- `not_evaluable`：发布并封存当前 ETAS 数值不足，停止这条 ETAS 修复路线和候选迭代；不得据此否定历史多发区或地震活动的预测价值。

唯一蓝图的 H1 和 G1 允许使用“ETAS 或当前最佳合法背景模型”。因此，ETAS
`not_evaluable` 不终止整个科学问题。Q2 结果标签完成后，必须先冻结新的目标盲阶段 4
执行协议，再以 75 km KDE 作为主背景开展最简单的异常增量检验；在该新协议冻结前仍不得读取阶段 4 正式目标。

这一区分同时保留两条约束：

1. 不改写 Q0 对本次 ETAS 资格尝试的停止决定；
2. 不让一个组件的数值失败错误地阻断蓝图已经允许的 KDE 科学路线。

## 3. 模型组合与贡献拆分

后续至少比较：

1. 长期 75 km KDE；
2. 可评价时的 ETAS/Hawkes；
3. KDE + 动态异常；
4. 可评价时的 ETAS/Hawkes + 动态异常；
5. 加入长期构造先验的简单组合；
6. 完整组合及其逐组件消融。

贡献使用同一折、同一目标、同一支持域和同一报警面积下的配对差值：

- 完整组合减去动态异常；
- 完整组合减去 ETAS/Hawkes；
- 完整组合减去长期 KDE；
- 完整组合减去长期构造先验；
- 单成分、两两组合与完整组合的固定面积召回和 Molchan 曲线。

ETAS 与 KDE 高度相关时，不把顺序加入带来的边际差值误称为唯一因果贡献；同时报告单独模型、逐项消融和多个合理加入顺序。

## 4. 每阶段强制科学价值门

每个阶段性结果形成后立即记录：

1. 分类：`direct_improvement`、`necessary_enabler` 或 `no_material_progress`；
2. `evidence`：相对当前最好合法基线的直接证据，或明确写“尚无直接效果证据”；
3. `next_scientific_test`：下一项最接近最终目标、边界有限的科学检验；
4. `stop_condition`：继续投入的停止条件；
5. `decision`：`continue`、`adjust` 或 `stop`。

测试数量、代码规模、界面完成度和运行稳定性不能替代预测效果证据。若工程资产不能在紧邻下一阶段转化为科学检验，停止扩张实现并先调整路线。

只有未来隔离的滚动验证显示固定报警面积召回提高，或同召回所需面积下降，才能标记
`direct_improvement`。静态图和交互图必须同时展示基线、完整组合、消融、面积预算与失败案例，不能只展示模型自身分数。

分类与动作不得自由解释：

- `direct_improvement`：必须有未来隔离、同支持域、同面积的效果证据，并通过当期预登记的不确定性与置乱门；
- `necessary_enabler`：必须能直接进入紧邻、一次性、预登记的科学检验，并写清最长投入范围；不能把“未来也许有用”当作理由；
- `no_material_progress`：没有上述两类证据，立即停止当前工程扩张并复审数据、假设、模型组合和评价设计。

阶段验收和当前重启交接必须使用同名字段
`science_value_category`、`evidence`、`decision`、`next_scientific_test`、
`stop_condition`。字段不全时不得把阶段写成完成，也不得进入下一阶段。

## 5. Q2 实际结果与路线调整

Q2 已按冻结身份形成五个快照、25 条起点记录，并通过独立哈希和数值复算。实际结果为
`not_evaluable`，但必须精确理解：

- 25/25 个起点均在初始向量处终止，记录为 `iterations=0`、
  `function_evaluations=1`，终点等于初始点；
- 没有形成参数快照，因此 Hessian、分支比、三网格没有实际计算，fold 1、fold 3 的局部父历史敏感性也因缺少主参数而不可计算；
- 独立 verify 通过只证明封印身份、初始点数值复算和状态汇总一致，不证明优化器成功运行；
- 结果只能说明“当前冻结 ETAS 执行路径不可评价”，不能说明 ETAS 经拟合后预测效果差，也不能否定历史地震活动的预测价值。

本阶段科学价值分类为 `necessary_enabler`，具体价值是可信负结果和路线剪枝，不是预测效果直接提升。它触发的决定是 `adjust`：

1. 停止当前 ETAS 数值修复和候选迭代，不重跑本 attempt，不增加 Candidate 8；
2. 保留已经通过 G1-LS 的 75 km KDE 作为当前最好合法背景；
3. 在读取任何阶段 4 正式目标前，先冻结新的目标盲执行协议；
4. 下一项最近的科学检验只回答：动态异常加入 75 km KDE 后，是否在同支持域、同未来窗口和同报警面积下稳定提高区域召回，并优于时间/空间置乱；
5. 若最小增量检验没有稳定效果，停止异常模型工程扩张，复审异常时效、空间支持、报告可用时间和模型组合，而不是直接进入大型神经网络。

下一步被限定为 `Stage4A / S4-KDE-DEV`：在目标盲状态冻结
`configs/anomaly_increment_kde_dev.yaml` 和最小薄运行器，然后只做一次三个滚动折的开发科学
attempt。通过只授权再冻结一次独立验证协议；失败或证据不足都不授权继续堆叠异常模型。

动态轨迹不是异常路线的唯一生存条件：`B2` 若不能稳定超过 `B1`，只否定轨迹贡献；`B1` 仍须相对
KDE 和覆盖控制独立通过固定面积召回、信息增益、两类置乱和跨折/跨区门，才能作为简单快照候选保留。

Q2 的准确静态和离线交互判读分别见：

- `docs/background_etas_numerical_qualification_interpretation.svg`；
- `outputs/interactive/background_etas_numerical_qualification/interpretation.html`。

## 6. Stage 4A 可执行性审计与 0.4.3 调整（历史状态）

`0.4.2` 可执行性审计没有产生预测成绩。它发现公开背景 registry/model JSON 只有身份和数值审计
摘要，不含可推理 KDE 状态，而旧协议又禁止目录重物化；同时旧 `.jsonl` target-read ledger
措辞与已经目标盲实现的单 mapping `0→1` CAS 原语不一致。

审计结果的科学价值字段是：

- `science_value_category`: `no_material_progress`
- `evidence`: 尚无直接效果证据；若沿旧合同继续实现，只会堆叠无法进入科学评分的工程。
- `decision`: `adjust`
- `next_scientific_test`: 在零目标读取状态冻结 `0.4.3`，只允许单目录会话、固定 fold_4/75 km
  背景重物化和精确原语 allowlist，再进入薄 code freeze。
- `stop_condition`: 若必须重选带宽、改变 Mc/支持域/final_validation、第二次打开目录、复活旧 R2
  orchestrator 或改变科学门，则停止工程扩张。

`0.4.3` 修复只有经独立复审 PASS 后，有限阶段价值才可另记为 `necessary_enabler` 并决定
`continue_to_thin_code_freeze`；本轮复审 PASS 前保持 `no_material_progress + adjust`。该有限价值
只表示执行矛盾已消除、紧邻的一次 Stage 4A 科学检验仍可实施，不能把协议测试或可执行性冒充预测
提升。详细证据见 `docs/phase4_kde_development_executability_amendment.md`。

## 7. Stage 4A 0.4.4 当前科学价值状态

上一节记录的是 `0.4.3` 冻结前后的历史判断。随后对废弃一次性 runner 的独立审计确认：旧实现
没有把时间/空间置乱真正接入“重建特征—重新拟合—重新评分”，因此已停止使用；它没有打开真实
目标，也没有消耗唯一开发 attempt。目标盲 schema 预检又确认阶段 3 状态/特征 schema 与 registry
一致，两个置乱重建函数本身具备真实重建能力，剩余基础缺口只是受限空间工件定位/schema 与
exposure 日期 adapter。

当前唯一协议已调整为 `0.4.4` / `v0.3.4-kde-anomaly-increment-protocol`；`0.4.3` 与
`v0.3.3-*` 已在任何真实输入或目标读取前转为
`historical_superseded_before_target_read`，不再提供执行授权。

当前科学价值字段是：

- `science_value_category`: `necessary_enabler`
- `direct_prediction_improvement`: 尚无
- `evidence`: 已冻结四个受限空间工件的固定身份/schema、exposure/issue 日期解释和最小
  target-blind adapter 边界；科学问题、候选、面积、置乱、Bootstrap 和通过门均未改变。
- `decision`: 只继续一次纯合成的 observed/time/space 同路径重建—组装—重拟合验收与薄代码冻结。
- `next_scientific_test`: 代码标签远端回读通过后，才在唯一开发 attempt 中运行真实
  observed/time/space 重拟合和 600,000 平方公里固定面积比较。
- `stop_condition`: 若这一次纯合成验收仍发现新的基础性 P0，停止异常增量路线并保留 75 km KDE；
  不再追加工程层、复杂模型或锁定测试绕过。

这一定性只说明紧邻的科学检验获得一条受控可执行路径，不是预测效果提升。详细当前合同见
`docs/phase4_kde_target_blind_executability_amendment.md`。

## 8. Stage 4A 0.4.4 代码级终审与路线停止

纯合成薄组件和 observed/time/space 同路径链完成了 229 个工程测试，但正式 runner 设计终审发现
新的 foundational P0：受限 `cell_mapping` 中的
`cell_id → construction_zone_id` 在 adapter 内核验后被丢弃，而 `0.4.4` 又把返回值冻结为不含
该映射的四项。于是协议强制的 39 区贡献和 leave-one-region-out 没有合法计算路径；重读工件、
增加返回字段或从 geometry 重建区域都会违反冻结合同。

同一终审还确认，当前拟合草稿把每个 `issue × cell` 的 7 天 composite-midpoint 补偿项压成
单个 exposure 和 decay，不能表示非线性指数目标中的逐中点积分。纯合成链只证明代码连通，不证明
正式数学目标正确。seal 终审另有一个严格类型 P2，但由于更高层 P0 已触发停止，不再追加 seal
补丁。

本轮科学价值字段更新为：

- `science_value_category`: `no_material_progress`
- `direct_prediction_improvement`: 无
- `evidence`: 229 个工程测试通过，但区域稳健性门不可计算，时间补偿目标也不符合冻结数学定义；
  开发目标、独立验证和锁定测试仍均未读取，未产生真实效果指标。
- `decision`: `stop_anomaly_increment_and_retain_75km_KDE`
- `next_scientific_test`: `preregister_stage2s_causal_two_timescale_seismicity_screen`；只预登记
  长期 75 km KDE 与一个近期因果地震 KDE 的最小组合及一次开发检验。
- `stop_condition`: 不得通过 `0.4.5` 补丁、复杂异常模型、阶段 4B、阶段 5 或锁定测试读取复活或
  绕过当前异常路线。

未创建 attempt ledger 和 target-read ledger，唯一开发 attempt 消耗仍为 0。详细失败证据、草稿
隔离清单和恢复边界见
`docs/phase4_kde_0_4_4_synthetic_acceptance_and_foundational_stop.md`。

只读路线复审已推荐另立阶段 2S，而不是重开阶段 4A：用冻结的长期 75 km KDE 作为 `S0`，只增加
一个严格因果的近期地震 KDE 形成 `S1`，直接检验 600,000 平方公里固定面积召回、信息增益和时间
错位对照。该推荐尚未获得执行授权；必须先提交、推送当前失败验收，再另行修订唯一蓝图并完成
Stage 2S 目标盲协议验收。
