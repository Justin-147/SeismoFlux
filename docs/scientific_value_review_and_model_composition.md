# 科学价值复审与模型组合路线

首次形成：2026-07-29

阶段 4A 目标盲修订：2026-07-30

阶段 2S 唯一正式尝试停止复审：2026-07-30

阶段 2P 目标盲路线复审：2026-07-31

状态：阶段 4A 异常路线和阶段 2S 历史近期地震路线均已硬停止；当前保留长期 75 km KDE，
Stage 2P 已选择新的真正前瞻近期地震筛查；Stage 2P-1A 候选审计已 GO 并唯一跃迁为
accepted/frozen，当前 `protocol_frozen=true`、`execution_authorized=false`、
`real_issue_authorized=false`

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

只读路线复审曾推荐另立阶段 2S，而不是重开阶段 4A：用冻结的长期 75 km KDE 作为 `S0`，只增加
一个严格因果的近期地震 KDE 形成 `S1`，直接检验 600,000 平方公里固定面积召回、信息增益和时间
错位对照。该路线后来完成协议与代码冻结并进入唯一正式尝试；最终结果见第 10 节。

## 9. Stage 2S-0 最小科学合同

阶段 4A 停止提交 `414198aff611bbe1df05413e0b7bb7309c644189` 已推送后，Stage 2S-0 才开始。
路线复审最终选择更小、解释更清楚的 30 天合同，放弃 365 天指数衰减和 1,000 次随机 lag：

- `S0`：冻结长期 75 km KDE；
- `S1`：`S0` + 最近 30 天严格因果 M4+ 事件的等权 75 km KDE；
- `SP`：`S0` + 紧邻的再前一个 30 天同结构 KDE。

`SP` 是确定性过去对照，不冒充 permutation。每折两个混合权重只在 h007 fit exposures 上按固定
一维凹似然拟合；三模型共享 M5–6 日率、支持域、格网、目标和面积。两项比较分别进入信息增益和
召回的两成员 Bonferroni 同时区间。

三路独立协议审计发现并修订了会改变科学结论的基础问题：源日历 content hash 缺失、研究区/
query grid/精确面积合同不闭合、连续 KDE 与格点评分混用、支持域信息增益与全区召回分母未分开，
以及双对比区域/LORO、单次读取、Bootstrap 随机流和滚动折封印顺序不唯一。修订后只允许连续
事件坐标评分、12.5 km 归一质量聚合到 25 km 报警/区域质量、两项对比分别做固定分母可加 LORO；
三折严格按顺序执行并使用 fold-fit、issue、fold、master 四层不可改写封印，任何 assessment 成绩
都不能进入后一折拟合或预测。

2022–2023 已经用于 Stage 2R 背景和 75 km 带宽选择。因此 Stage 2S 即使通过，也只能称为
“新增 alpha 和候选图在复用开发期内的时间因果筛查”，不能称为全局目标盲或独立验证。

本阶段科学价值字段为：

- `science_value_category`: `necessary_enabler`
- `direct_prediction_improvement`: 尚无
- `evidence`: 唯一模型、对照、连续评分、窗口、权重规则、三折、固定面积、Bootstrap、双对比
  跨折召回稳定性、跨区、延迟门和滚动封印链已通过目标盲本地终审；Stage 2S 真实目录、候选目标
  指标和成绩仍未读取，远端协议标签核验前无执行授权。
- `decision`: `adjust_to_stage2s_causal_seismicity_screen`
- `next_scientific_test`: `run_one_three_fold_S1_minus_S0_and_SP_development_attempt`
- `stop_condition`: 新 foundational P0、任何门失败或证据不足都停止本路线，不改成其他窗口、
  衰减、带宽或模型重试。

Stage 2S-0 只有在协议测试、独立验收、提交、推送和协议标签远端核对全部完成后，才可进入最小
代码阶段。完整合同见 `docs/causal_seismicity_screen_protocol.md`。

## 10. Stage 2S-1 唯一正式尝试停止复审

Stage 2S-1 的代码提交 `4188523991926c51a7fbd9314d36395cc9bfad62` 及远端 code tag 已核验。
正式入口完成无目标预检、唯一 attempt 与 target read 认领、目录单次解析以及 fold4 支持域重建。
主科学路径随后在首个 fold-fit receipt 和任何预测封印前发生未留存异常；terminal/finalizer
重读 preflight receipt 时又触发 canonical JSON reserved-key 二次异常，覆盖原始异常并阻止
terminal record 写入。因此原始 science 失败类型未知且不可恢复。

没有形成 S0/S1/SP 预测、评分、Bootstrap、跨区或延迟指标，也没有 whole-run record、正式效果图
或 result tag。这不是“近期地震活动没有用”的科学负结果，而是没有可评分答案的无效实验。冻结
协议不允许在 foundational P0 后修复代码并用同一历史目标再考一次。

- `science_value_category`: `no_material_progress`
- `direct_prediction_improvement`: 无法评价
- `evidence`: 真实目录只读取一次并完成长期支持重建，但没有任何预测 seal、成绩或效果指标
- `decision`: `stop_stage2s_and_retain_75km_KDE`
- `next_scientific_test`: `target_blind_route_review_then_preregister_selected_prospective_test`；先选择
  新问题、评价门和停止条件并写入唯一蓝图，只有验收、提交和推送后才可实施
- `stop_condition`: 不得重用本次 attempt、2022–2025 目标或锁定测试进行修复后重跑、调参或
  追逐阳性结果；不得先修代码再寻找可承接的科学问题

## 11. Stage 2P-0 目标盲路线复审

本轮没有读取新目标或成绩。已有证据排序为：

1. 75 km KDE 是唯一合法背景冠军，但只通过信息增益 G1-LS，尚未证明固定面积严格召回；
2. 近期地震 Stage 2S 无成绩，假设仍未知；
3. 异常特征库已因果重建，但没有真实预测结果；
4. 构造快照缺少历史可用资格，ETAS 当前实现不可评价，复杂模型没有进入依据。

已经发生的历史目标不能再包装成真正前瞻证据。路线复审因此选择 `Stage 2P`：从未来首个合法
周起报开始同步封存因果重建的长期 KDE `P0(T)`、固定等权近期 30 天挑战者 `P1(T)` 和同一
`T` 前快照内以 `Q=T-15min` 为共同截止的前一等长 30 天起源窗对照 `PP(T)`。冻结的是 G1-LS
选出的 75 km 方法、支持域和资格规则，不是截至 2023-06-30 的陈旧密度；三模型每期都从同一
封存快照形成。

异常挑战者排第二，在 P 路线开始后、任何 P 确认性效果指标解封前另立有效日和协议，不阻塞第一期。

该决定直接服务用户要求的“先提高总效果，再通过消融看贡献”：P1 的支持域信息增益和全研究区
严格召回在同一 600,000 平方公里面积上限规则下都
同时超过 P0 与 PP，且结果不由单一震群或区域主导，才说明这一固定近期成分有增量。未来异常
挑战者加入后，只在两者较晚 `valid_from` 之后的共同新 cohort 比较 `P0`、`P0+R`、`P0+A`
和 `P0+R+A`。当前不为尚未启动的异常臂预定模型或权重。

周度 52 个按时 issue 只是归档水位，正式评分使用目标无关选择的逐窗口不重叠 exposure、唯一
去重 M5–6 主目标和配对震群块 Bootstrap。第 52 个按时期成熟且三个 horizon 可评价、达到
20 个主目标事件和 10 个独立震群块才允许一次正式判定；不足且未打开效果行时只冻结输入并继续
盲积累，第 104 个按时期成熟后允许第二次、也是最后一次输入冻结。104 的门仍不足时直接记
`evidence_insufficient`、不创建结果封印；全线确认性效果仍最多只看一次。即使通过，也只形成
`direct_improvement` 候选证据，不替代 G7/G8 或业务晋级。

- `science_value_category`: `necessary_enabler`
- `direct_prediction_improvement`: 尚无
- `evidence`: 目标盲审计确认不存在可安全宣称为真正未见的历史独立 cohort；已冻结最小前瞻
  科学问题、候选、主面积、最低样本和最长投入
- `decision`: `adjust_to_stage2p_prospective_recent_seismicity_screen`
- `next_scientific_test`: Stage 2P-1 首期前预登记源/真值快照、模型身份、非重叠 exposure、
  震群块、familywise 区间、许可边界和 P0/P1/PP 不可改写预测归档；完成验收、提交、推送与
  远端标签核验后，才从下一规则起报开始
- `stop_condition`: 首期合同不闭合则不发行；正式门失败或 104 个按时起报仍样本不足则停止 P1、
  保留 P0；不得针对已观察 cohort 改权重、窗口、带宽或模型

## 12. Stage 2P-1A 前瞻协议科学价值复审

### 12.1 外行结论

这一阶段还没有做出一张真实预测图，也没有让模型变准。完成的是把未来考试规则写死：每周同一
时刻保存长期图 P0、加入最近 30 天地震的图 P1、加入再早 30 天地震的对照图 PP，三者用同一份
当时可见的数据和同一套“不超过 600,000 平方公里”的完整格前缀规则，并记录实际完整格面积。
等未来地震成熟后，最多只看一次累计结果。

它有必要，因为过去的历史数据已经不能再当作完全没见过的考试；但它只是通往真实证据的桥梁，
不是预测提升本身。候选审计已 GO 并唯一跃迁为 `protocol_frozen=true`；执行和真实起报仍分别
保持 `execution_authorized=false`、`real_issue_authorized=false`，远端闭合后的下一步也只能
做纯合成同路径验收。

Stage 2P-1A 只冻结 schema、规范字节、工件 profile、外部信任边界和失败闭合语义，不声称真实
ASN.1/CMS、目录表、预测数组或评价字节已经实现；其状态只能是 `stage2p1b_required`。Stage
2P-1B 才用纯合成数据逐字节演练整条链。候选审计 GO 后已在同一候选中唯一改为
`accepted`、`protocol_frozen=true`，并已完成精确最终字节复验、提交、推送、annotated
协议标签和远端 tag object/peeled commit 回读，Stage 2P-1A 已正式关闭。随后 1B 开工前预检
发现的新 foundational P0 见第 13 节。

### 12.2 这次具体冻结了什么

- 三模型固定为每期重建的长期 75 km KDE P0、固定 0.5 的 P0+R30 和固定 0.5 的 P0+RP30；
- `Q=T-15min` 是共同查询结束；R30=`(Q-30d,Q]`、RP30=`(Q-60d,Q-30d]`，两窗严格等长；
  P0、R30、RP30 必须来自同一个 T 前快照；旧 Stage 2S 把 PP 可见截止移到 T-30d 的实现及其
  runner、attempt、seal、拟合、Bootstrap 和 gate 均不能复用；
- 本地目录只作截至 `2026-07-09T04:25:56Z` 的历史基线，之后的增量和真值使用同一条 USGS
  ComCat 获取/修订链，并完成 60 天同源洗脱；查询下探到 `minmagnitude=3.9`，本地严格筛
  `mag>=4.0`；count/query 的参数、规范 URL、原始响应和解析计数均需留证，count 达 20,000
  时禁止 query；
- 切源缝隙只按 300 秒、50 km、`abs(ΔM)<=0.5` 三阈值作确定性一对一去重，local anchor 优先；
- 首期不早于 `2026-09-10 00:00 Asia/Shanghai`，且是协议/代码标签远端核验后的下一规则周四；
- cohort 嵌入两个 annotated tag 的远端 object/peeled commit/receipt，并冻结从 parser 到
  evaluator/visualizer/validator 的 code manifest，后续记录必须逐项匹配；
- 正式抓取只在 Q 后开始；可安装的完整源快照、预测和报警候选必须最迟在 `T-5min` 冻结，候选
  TSA 的 `genTime<T-5min`。候选形成前失败时不生成预测；完整候选已生成但两家候选 TSA 均失败
  时，只保留本地受限、内容寻址候选并记 `prediction_generated=true`、
  `prediction_installed=false`。两类失败都最迟在 `T-4min` 冻结 missed 审计 core，并以独立
  RFC3161 请求在 T 前取得审计 token；所有远程时间戳记录的 core 只排除顶层
  `timestamp_attempt_evidence`、`remote_timestamp`、`content_sha256`，TSA 固定按 DigiCert 主、
  Sectigo 备的顺序取证；验证依赖、信任锚、OID、TSA 身份和 create-only token 附件合同也冻结；
- 只允许 `TargetCohortDefinition`、`IssueInputSnapshotRecord`、
  `MatureTruthSnapshotRecord`、`TruthRevisionRecord`、`EvaluationFreezeRecord` 五类只追加记录；
- scheduled 序号计所有规则周四且最多 130，on_time 序号只计按时 issue；52/104 检查点只按
  on_time，130 个规则周四仍不足 104 个按时期则证据不足停止，且 130 cap 优先于迟到的第 52 门；
- 真值在各窗口结束后再等 30 天，以独立于 issue 输入的请求形成时间上独立抓取的成熟快照，固定
  在 `0h、+6h、+24h、+72h、+168h` 按顺序取第一个完整成功响应并停止；只有全失败才完成五次
  并记不可评分，不得记零、临时重试或替补；preferred
  origin 修订跨 exposure 边界时按同一 formal-freeze source snapshot 确定性唯一重归属；
- 第 52 个按时期可有一次 formal-freeze；只有其基本样本门不满足且未打开效果行时，第 104 个
  按时期才允许第二次，全线最多两次 input-freeze、一次 effect look。formal-freeze 成功才写实际
  目标数、震群数和窗口成员；只有 `not_run_no_complete_scope` 的机械空 scope 可写真实 0，cap
  或 count/query/解析/派生失败都必须把科学量写 null/unavailable，保留 exposure/availability
  证据且不运行 Bootstrap，不能把“取不到”伪装成“没有地震”；
- 三路密度只在 G1-LS 支持域归一；信息增益只用支持域目标，严格召回仍以全研究区目标为分母，
  支持域外三模型共同未命中；600,000 平方公里只是完整格前缀面积上限并记录实际面积；
- 7/30/90 天只按 on_time/issue time/horizon 选择不重叠 exposure，已选真值不可用项不替补；
  主门至少 20 个唯一 M5–6 事件和 10 个独立震群块；使用 2,000 次配对震群块 Bootstrap、四端点
  Bonferroni 同时区间、39 区和最大震群稳健性门；完整 bootstrap 索引在盲态冻结且零分母不重抽；
- 第 52/104 路径的确认性输入最多冻结两次、效果最多查看一次；只有三个 horizon 均可评价、
  N>=20、B>=10 才打开效果并直接闭合唯一 `result_seal`。valid/invalid
  `ResultBundleManifest` 都以唯一规范 JSON 精确字节安装并复算文件 SHA；valid 分支必须完整绑定
  effect rows、同面积比较、Bootstrap、端点和稳健性工件，invalid 分支则按
  `effect_rows_open`、`alarm_area_comparison`、`bootstrap`、`endpoint_evaluation`、
  `robustness_evaluation`、`result_bundle_install` 或 `result_seal` 失败阶段，只绑定当时实际存在
  的审计工件，后续字段为 null，不得伪造空表、零端点或重跑。预测静态图和交互页立即展示，成熟
  回放只追加新图，累计效果在正式查看前密封；
- 默认最多 8 worker 且保留至少 2 个物理 CPU 核；GPU 只能做数学等价加速。

### 12.3 来源和结论边界

USGS 自有数据通常按美国公共领域处理，但 ComCat 的伙伴来源可能有例外。因此逐事件行和精确坐标
始终本地受限，公开事件叠加逐项复核来源与许可；这不阻断本地受控研究。

ComCat 在中国的 M4 覆盖、震级口径和修订节奏可能不同于本地目录。60 天洗脱只消除 P1/PP 内部
的换源混杂，不证明 ComCat 完全或与本地目录等价。覆盖差异只记录并限制结论，不设置效果相关
硬阈值。一次官方抓取或冻结目标无关支持证据未按时取得可记 missed；持续中断或实质变化时暂停
后续发行并作目标盲修订。两种分支都不补发，也不能查看目标后换源或调参。未来无论阳性还是阴性，
结论都只适用于冻结的 ComCat 获取/修订链、局部资格、75 km KDE 和固定 0.5 候选，不能外推为
全部近期地震模型。

### 12.4 科学价值判定

- `science_value_category`: `necessary_enabler`
- `direct_prediction_improvement`: `none`
- `evidence`: 来源/原始请求、远端标签、代码身份、时间戳、切源、模型、真值成熟、统计唯一查看、
  失败闭合、追加回放、许可和资源边界已形成冻结预登记，并完成最终验收、提交、推送与远端标签
  回读；真实 issue 数为 0，新前瞻目标读取数为 0，效果指标数为 0
- `uncertainty_change`: 降低了未来结果被事后回填、换源混杂、重复查看和单一震群支配所误导的
  风险；没有降低“近期 30 天地震是否真能提升预测”的科学不确定性
- `decision`: `continue_to_stage2p1b_synthetic_only_after_acceptance`
- `next_direct_test`: 用纯合成数据走通 `Q=T-15min` 同一快照、T-5 候选/T-4 missed 双证明、
  P0/R30/RP30、P0/P1/PP、RFC3161 core/主备假服务、五类只追加记录、时间上独立成熟抓取、
  preferred-origin 跨界修订、真值不可用且禁止替补、formal-freeze unavailable、最多两次
  input-freeze/一次 effect look、支持域 IG/全域召回精确估计量、至少 10 个震群块、四端点区间、
  valid/invalid ResultBundle 和 `input_freeze→result_seal` 密封闭环
- `stop_condition`: 合同审计失败则不进入 2P-1B；合成链出现 foundational P0 则停止并复审；
  真实起报前还必须完成代码、标签、来源身份和许可 preflight，不能把工程完成冒充科学进展

## 13. Stage 2P-1B 信任可行性预检停止复审

### 13.1 外行结论

Stage 2P-1B 原计划用假数据和假时间戳服务器验证整条链。开工前发现，已冻结的锁只接受
DigiCert/Sectigo 的真实证书链，而假服务器没有这两家机构的私钥，也没有针对本次动态请求预先
签好的合法回执。当前协议同时禁止换成测试根。因此合成成功链不是“还没写完”，而是在密码学上
无法按当前合同完成。

三路独立只读审计一致给出 foundational P0 / NO-GO。实现没有开始，真实地震目录、网络、issue、
effect rows 和锁定测试均未读取。直接删掉 validator 的关门点、跳过签名或相信记录自报的
“验证成功”都会形成虚假验收，不能推动预测效果。

### 13.2 对最终科学目标的影响

- `scientific_question`: 当前冻结协议能否在完全离线、无真实私钥的条件下完成五记录合成成功链？
- `new_evidence`: 冻结 registry 只有真实 DigiCert/Sectigo 精确锚；仓库没有相应私钥、预签 token
  或隔离的 synthetic trust profile。
- `uncertainty_change`: 将 1B 从 `ready_not_started` 收敛为
  `impossible_under_current_frozen_contract`。
- `science_value_category`: `no_material_progress`
- `direct_prediction_improvement`: `none`
- `evidence`: 没有模型预测、召回、信息增益、报警面积或置乱对照结果；只有信任边界不可满足的
  审计证据。
- `decision`: `stop_before_stage2p1b_implementation_and_reassess_protocol`
- `next_direct_test`: 新建目标盲协议版本，明确隔离 production 与 synthetic-acceptance 信任域，
  让测试 PKI 动态签发 core/nonce，但仍使用同一实际 ASN.1/CMS verifier；production 必须拒绝
  全部测试 token。
- `stop_condition`: 新协议未经测试、独立审计、提交、推送、annotated tag 和远端回读，不恢复
  1B，不读取真实数据。

该预检本身没有提高预测能力；其唯一价值是阻止不可满足的工程合同继续消耗时间，避免把测试绕过
误称为科学进展。推荐的协议修订细节见
`docs/restart_handoff_2026-07-31_stage2p1b_trust_preflight.md`。

## 14. Stage 2P-1A2 科学优先路线调整

本阶段标识为 `Stage2P-1A2`，协议版本为 0.2.5。
候选已完成独立科学公平性复验并转为 `accepted/frozen`；真实执行、联网和效果读取仍未授权。

### 14.1 外行结论

前一版花了太多精力证明“文件在某一秒以前被某张证书盖过章”，却还没有画出一张能判断方法好坏
的预测图。这条工程路线已经停止。旧 v0.2.4 不删除，作为历史证据保留；未来执行改用 v0.2.5
最小科学合同。

下一步不再讨论证书链。直接用三组合成地震检验 P0、P1、PP：一组让近期地震真的有用，一组完全
没用，一组故意误导。地图和指标必须把这三种情况区分开，证明实现至少会在应该赢时赢、没信息时
不乱报、被误导时显示变差。

### 14.2 哪些科学约束继续保留

- 同一起报时刻只用当时可见的数据；
- 三模型使用同一快照、支持域、网格和 600,000 平方公里面积上限；
- 不用真实震中制造候选或调参数；
- 以同面积严格区域召回为一级指标，以空间信息增益为支持指标；
- 同时比较 P1-P0 和 P1-PP；
- 至少 20 个唯一 M5–6 目标、10 个震群块，2,000 次配对 Bootstrap；
- 最多一次正式效果查看，失败后不针对同一 cohort 调参；
- 局部 Mc 过高只影响本地单元；
- 简单模型没有效果证据前不做大型模型。

RFC3161/TSA、证书链、逐工件注册和硬件收据不再是科学 MVP 的必经门。规范 JSON、SHA-256、只
追加记录和 GitHub 远端回读已经足以支持当前“预测是否在答案前固定”的研究追溯。

### 14.3 下一项直接检验

Stage 2P-1B MVP 必须交付三种合成情景的 P0/P1/PP 静态三联图、离线交互页、同面积报警区、未来
合成目标、严格召回、信息增益、两项模型对比和去除最大地区/震群后的稳健性。全部结果必须标注
为合成演练。如果没有形成这些可以直接判断模型好坏的结果，本路线不再增加任何工程层，先简化或
调整科学假设。

### 14.4 科学价值判定

- `scientific_question`: 最近 30 天地震活动能否在固定报警面积下提高未来独立区域召回
- `science_value_category`: `necessary_enabler`
- `direct_prediction_improvement`: `none`
- `evidence`: 旧版不可执行的信任前置已被隔离；真实预测、未来目标、召回和信息增益仍为 0
- `uncertainty_change`: 科学效果不确定性没有下降；最近一次直接合成检验现在可执行
- `decision`: `continue_to_stage2p1b_mvp_synthetic_after_acceptance`
- `next_scientific_test`: 三种合成情景的 P0/P1/PP 图、交互页和同面积指标
- `stop_condition`: 不能正确区分三种情景、没有直观结果，或再次被无关工程前置阻断时停止并简化；
  未来正式门失败时停止 P1、保留 P0

## 15. Stage 2P-1B 合成科学 MVP 复审

### 15.1 外行结论

这一步终于形成了可以直接看的结果。程序在“近期活动确实有用”时让 P1 赢，在“没有新信息”时
给出零增益，在“近期活动误导”时让 P1 明确变差。它说明方法不会只报喜不报忧，也不会靠扩大
报警面积取胜。

但未来合成地震是按已知地图关系故意摆放的，所以 100 个百分点的差不是现实预测成绩。真实世界
中最近 30 天地震究竟有没有用，仍需按时发布真实预测并等待未来地震后才能回答。

### 15.2 新证据

- P0/P1/PP 实际报警面积均为 600,000 平方公里；
- 三种情景分别产生正、零、负的召回和信息增益方向；
- 36 个目标分属 12 个震群、4 个区域，区域和震群移除不是同一项重复检查；
- 2,000 次固定 Bootstrap 与 Bonferroni 区间可确定复算；
- 审计发现并封死了直接构造窗口注入未来事件的旁路；
- 异质 horizon 零分母会返回 `evidence_insufficient` 且不重抽；
- 静态图和离线交互页均明确为纯合成，不冒充正式门通过。

### 15.3 科学价值判定

- `scientific_question`: 最近 30 天活动候选的实现和评价器能否公平区分已知正、零、负情景
- `science_value_category`: `necessary_enabler`
- `direct_prediction_improvement`: `none`
- `evidence`: 282 项回归通过；2,000 次六产物逐字节复核通过；三路独立审计 GO，P0/P1=0
- `uncertainty_change`: 已降低实现偷看未来、区域/震群重复和合成标签误导风险；未降低真实预测效果不确定性
- `decision`: `accept_stage2p1b_then_remote_close_before_real_data`
- `next_scientific_test`: Stage 2P-1C 第一张按时冻结、公开的真实 P0/P1/PP 预测图，并等待成熟真值
- `stop_condition`: 真实数据公平性、局部 Mc、支持域/39 区或按时发行任一不能闭合则不发行；
  正式门失败时停止 P1、保留 P0

本阶段达到的是“可以开始一次诚实的真实前瞻检验”，不是“已经提高地震预测”。异常表、断层和
长期危险性数据尚未进入该最小问题；待合法背景有真实前瞻证据后，再把它们作为独立挑战者逐项
加入，通过相同面积比较和消融评价各自贡献。
