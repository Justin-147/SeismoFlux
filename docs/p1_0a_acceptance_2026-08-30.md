# SeismoFlux P1-0A 真正前瞻预登记验收（2026-08-30）

## `v0.2.7` 预首次真实起报科学完整性修订

P1-0B 纯合成审计发现，原 `v0.2.6` 合同没有在 36 月累计 0 群时区分“真值已完整可用且确实
0 群”与“真值不可用或根本没有到期入选 exposure”。当前候选因此透明升级为
`v0.2.7-p1-b0-r30-protocol`，计划代码标签为 `v0.2.7-p1-b0-r30-code`。

本修订发生在首个规则起报 `2026-09-10T00:00:00+08:00` 之前；修订时真实记录数为 0、真实目标
读取数为 0，没有利用未来答案。协议 ID、`valid_from`、模型、数据、面积、目标、指标、阈值和起报日
全部不变。原 `v0.2.6` 的提交、标签、远端回读与以下历史验收证据原样保留；它们不被删除或改写。
`v0.2.7` 已由提交 `0f43f15bc983a37157f1b129976c7ec0ea47fc7d` 和注释标签
`v0.2.7-p1-b0-r30-protocol` 完成测试、验收、提交、推送和远端回读；闭合本身仍不授权真实 issue。

修订后的零群规则是：只有至少一个到期且由 30 天 guard selector 入选的 exposure、全部相应 Truth
均为 `mature_truth`、合计确实为 0 群，才可写
`report_evidence_insufficient_at_final_review`。任一到期入选 Truth 不可用，或截至终点没有任何到期
入选 exposure，都只能写 `pause_scientific_integrity_failure`；不得把未知或无可评价 exposure 当成
真实零群。JSON Schema 允许这两个结构值，跨记录的充分必要条件由记录链验证器执行。

同时澄清终止边界：任何 look 写出 `pause_scientific_integrity_failure` 都立即终止本试验；
`cluster_30` 或 `time_36_months` 的任一终判也立即关闭新 issue 与复审流。上述决定之后不得再追加
Protocol、Authorization、Forecast、Missed 或 Review，也不再延续周历 missed。终止后唯一允许追加的
记录类型是 `TruthSnapshotRecord`，而且 issue 的 Forecast 必须在终判前已经发行、该 horizon 必须已经
由对应 guard selector 入选、同一 issue×horizon 的 Truth key 尚不存在；它只用于完成已经承诺的
30/90 天随访，不得改变、重开或追加既有终判。

## 当前结论

验收：`PASS`。阶段状态：`accepted_closed`。远端闭合：`verified`。

P1-0A 候选已经把 `B0_R30` 对 `B0` 的真正前瞻科学问题、模型、数据边界、首期、面积、目标、序贯复审和停止条件写成明确的冻结合同。它目前只属于 `necessary_enabler`，真实 issue 数为 0，没有新的训练、测试或前瞻效果，`real_issue_authorized=false`。

本文件列出的本地门、第三次独立科学复审、精确提交、协议标签、推送和远端回读已经全部通过。P1-0A 已闭合，但仍不得发行真实预测；下一阶段只授权 P1-0B 纯合成双模型演练。

## 通俗验收问题

本阶段只检查一件事：我们是否在看到未来答案之前，把“哪两张图比赛、什么时候画、用什么数据、圈多大面积、未来怎样算赢”写死了。

它没有回答 `B0_R30` 未来是否真的更好。答案必须等每周按时保存的预测及其 30/90 天未来地震逐步成熟后才能形成。

## 冻结科学内容

- 对比：`B0_R30` 对 `B0`。
- 公式：`B0_R30=0.75*B0+0.25*R30`；D1 三折均选 `alpha=0.25`，P1 不再训练或调参。
- KDE：两路均为 75 km Gaussian KDE。
- `B0`：1970 年起、起源和可用时刻均不晚于 Q、位于冻结研究区/support 的 M4+ 事件等权；`common_mc=4.0`，局部 Mc 只作用对应本地单元且不重算。
- 日历：每周四 `00:00 Asia/Shanghai`，`Q=T-15min`。
- `valid_from` 和首个 scheduled issue：`2026-09-10T00:00:00+08:00`。
- 数据：本地目录只作截至 `2026-07-09T04:25:56Z` 的历史 `B0`；此后模型新增输入和真值唯一来自 ComCat；首期满足 60 天同源洗脱。
- 切源去重：候选在冻结阈值内严格按“起源时间差 → WGS84 距离 → 震级差 → stable source event ID”升序确定性一对一匹配，本地记录优先；禁止人工改配。
- 公平性：`B0` 先在 600,000 km²上限下取完整格前缀并形成 `A_ref`；`B0_R30` 只能取 `<=A_ref` 的完整格前缀，绝不比基准多圈；面积差必须小于挑战者下一格且小于 625 km²，否则该曝光不可评价。
- 主终点：30 天独立 M5–6 震群严格召回；90 天为次级。
- 震群：每个 horizon 的 selector 固定 `T_next>=T_prev+h+30d`，以 30 天 guard gap 排除跨入选窗口重复；各成熟窗口内做一次 30 天/75 km 分群，封口后固定最早事件代表和唯一 issue 归属。
- 复审：`SequentialReviewRecord` 只读取 30 天主终点；10/20 群只报告并继续，36 个月前达到 30 群或先到 36 个月即最终判定，最终 decision 不能继续。
- 统计：三次复审使用震群配对 Bootstrap 2,000 次、PCG64 种子 147 和三看 Bonferroni 双侧
  98.333333% 区间。
- 真值：30/90 天窗口在 `T+h+30d` 后由独立 ComCat 抓取成熟；不可用不能记为零或换一期替代。
- 最终解释：无正方向则停止挑战者；正向但未达到至少 +5 pp 且序贯调整区间下界大于 0 时统一为 `report_uncertain_at_final_review`；达到该门才是强确认。同召回少用约 8% 面积为次级实用成功。
- 展示：每个按时期必须有不可覆盖静态图和离线交互页；成熟后只新增回放，不修改原预测。
- 禁用：异常、断层、人工预测、树模型、神经网络、锁定测试以及旧 P0/P1/PP 合成合同均不进入本 P1。
- 退休：RFC3161/TSA、证书链、硬件收据和逐工件外部注册不再作为门控。

记录链从尚未授权的 `ProtocolDefinition` genesis 开始；未授权周四可接
`MissedIssueRecord`，P1-0B 与真实代码闭合后才可新增 `RealIssueAuthorizationRecord`。只有授权记录
已经在同一只追加链上且早于 `T`，后续才准接 Forecast；真值和序贯复审也不得另起旁链。P1-0A 本身
只冻结这些类型，`actual_record_count=0`。

30 天主终点的 10/20/30 look 使用封口震群按 `(代表时间,event_id,issue_id,cluster_id)` 的全局顺序前缀；90 天不进入序贯 decision。一次成熟批次跨过多个门时必须依 10→20→30 连续写记录。36 月先到时 `elapsed_months=36`，用全部成熟群替代下一未到 look：prior look/count 只能是 `0→0..9`、`1→10..19`、`2→20..29`，终端 look 为 `prior+1`；同刻达到 30 群只写 `cluster_30`。36 月累计 0 群时命中写 0，效果字段和 Bootstrap 区间字段写 null；只有至少一个到期入选 30 天 exposure 的 Truth 全部成熟可用且合计确实 0 群时，decision 才为 `report_evidence_insufficient_at_final_review`，任一到期入选 Truth 不可用或根本没有到期入选 exposure 时只能 `pause_scientific_integrity_failure`。全程最多三看。
旧 v0.2.5 的 P0/P1/PP、7/30/90 宏平均和旧 selector 不提供当前权限。

## missed 与继续规则

若真实执行代码和授权没有在 `2026-09-10T00:00:00+08:00` 前完成提交、推送和远端回读，则 9 月 10 日只能登记 `missed`：没有预测、没有效果，永久不得补发。协议保持完整时，排期继续到后续每个周四；待真实授权闭合后，只能从下一个尚未到达的规则周四首次 `on_time`。不得利用 9 月 10 日之后出现的目标修改协议。

未来信息污染、原预测可覆盖、数据源变为不可比较或两模型公平面积规则失效属于科学完整性失败，必须暂停并另立更晚 `valid_from`，不能用下一周补救当前 cohort。

## 精确验收范围

本阶段只验收以下文件，不纳入任何既有 Stage4 未跟踪草稿：

1. `docs/p1_b0_r30_prospective_preregistration.md`
2. `configs/p1_b0_r30_prospective.yaml`
3. `data/contracts/p1_prospective_records_v1.json`
4. `data/manifests/p1_source_boundary_manifest.json`
5. `data/manifests/p1_model_manifest.json`
6. `tests/unit/test_p1_b0_r30_preregistration.py`
7. `configs/research_protocol.yaml` 的 `p1_b0_r30_current_authority`
8. `docs/p1_0a_acceptance_2026-08-30.md`
9. `docs/restart_handoff_2026-08-30_p1.md`
10. `SEISMOFLUX_IMPLEMENTATION_HANDOFF.md`

## 最终门与实测结果

| 门 | 验收内容 | 当前状态 | 最终证据 |
| --- | --- | --- | --- |
| P1-A1 协议一致性 | 文档、YAML、JSON Schema/清单中的公式、时间、数据、面积、目标和权限逐项一致 | PASS | 聚焦测试 `14 passed in 0.67s`；第三次独立复审 `GO` |
| P1-A2 解析与结构 | YAML/JSON 可解析，Schema 和清单结构闭合 | PASS | 六类合法记录样例全部通过 Draft 2020-12 Schema；历史非法反例均被拒绝 |
| P1-A3 科学边界 | `real_issue_authorized=false`、0 条实际记录；未授权 genesis/missed、后续显式授权和单链顺序闭合；未来泄漏、回填、旧 v0.2.5 授权、锁定测试和禁用特征均失败闭合 | PASS | 未授权 Forecast、断链、错误 missed 状态、90 天混入主 look 和虚假零样本区间均被拒绝 |
| P1-A4 评价公平性 | B0 参考面积、30 天 guard-gap selector、窗口内分群/封口/归属、仅30天序贯、稳定 look 前缀、批量跨门和36月零群 null 语义逐项冻结 | PASS | 挑战者多圈面积、错误 prior/look/count、终判继续等反例均被拒绝 |
| P1-A5 资源与工程相称性 | 本阶段不运行大计算；无后台 P1 真实抓取/发行进程；不触碰 Stage4 草稿 | PASS | 无 Python/P1 后台；48 逻辑处理器，三次总 CPU 采样约 17.58%/4.96%/9.54%；15 个 Stage4 草稿保持未跟踪且未纳入 |
| P1-A6 文本与差异 | 格式/差异检查通过，精确暂存只含上列 10 个文件 | PASS | Ruff、format、strict Mypy、`git diff --check` 全部通过；治理回归 `43 passed in 48.73s` |
| P1-A7 远端闭合 | 提交、推送并远端回读到同一提交；闭合前真实发行仍禁止 | PASS | 当前协议提交 `0f43f15bc983a37157f1b129976c7ec0ea47fc7d`；标签对象 `947cccf2406e58e4b12cb44fbe86905731addd3c`，解引用同为 `0f43f15bc983a37157f1b129976c7ec0ea47fc7d`；原 `v0.2.6` 证据继续保留为历史 |

建议的相称最终命令范围为：聚焦 P1-0A 单元测试、配置/JSON 解析、与本改动相关的治理回归、Ruff/格式（如适用）和 `git diff --check`。不得运行锁定测试，也不得把全仓测试数量当作预测效果。

## 当前验收决定

- `acceptance`: `PASS`
- `stage_status`: `accepted_closed`
- `remote_closure`: `verified`
- `real_issue_authorized`: `false`
- `next_stage_authorized`: `P1-0B_synthetic_only`

注意：P1-0A 闭合后仍然不授权真实 issue。下一步 P1-0B 必须只做双模型纯合成演练，不读新增真实目录、不联网、不读未来真实目标；P1-0B 自身验收、提交和推送后，才可另行审议真实发行授权。

## 科学价值复审

- `science_value_category`: `necessary_enabler`
- `direct_prediction_improvement`: `none`
- `evidence`: 冻结了 D1 最佳简单候选的未来公平检验规则；真实 issue、成熟目标和新效果均为 0。
- `uncertainty_change`: 尚未降低 `B0_R30` 是否真正提高未来召回的不确定性，只降低了未来试验被事后改规则的风险。
- `decision`: `P1_0A_closed_proceed_P1_0B_synthetic_only`
- `next_scientific_test`: P1-0B 纯合成双模型正/零/负演练和静态/离线交互展示。
- `stop_condition`: 合同不一致或合成链不能正确反映预设结果时不进入真实发行；不得以更多工程证明或更复杂模型绕过。
