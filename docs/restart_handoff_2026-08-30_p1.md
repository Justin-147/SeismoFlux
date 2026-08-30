# SeismoFlux P1 中断恢复交接（2026-08-30）

## 一句话状态

整个项目尚未完成。D1 已由提交 `0dab57fd1491b5f4924cbae87c0b2001c6fc6b24` 完成验收、提交、推送和远端回读，科学结论是：简单模型 `B0_R30` 在真实历史三折时间外推中比 `B0` 多命中 4/21 个独立震群；异常组件没有显示额外增量，因此停止异常复杂化，不打开锁定测试。

P1-0A 已由冻结提交 `d793d3359e7caf71efbef32b76bb887f5511ad78` 和标签 `v0.2.6-p1-b0-r30-protocol` 完成验收、推送和远端回读。它没有真实 issue 和新效果，科学价值是 `necessary_enabler`，`real_issue_authorized=false`。当前紧邻工作是 P1-0B 纯合成双模型演练。

## 外行版：现在在做什么

我们已经从历史数据里看到“长期多发区 + 最近 30 天活动”可能比只看长期多发区更好。现在不能继续用已经看过的历史答案证明自己，而要在未来地震发生前把两张地图保存下来，再等未来 30/90 天检验。

P1-0A 就是在开考前把规则写死：每周什么时候画图、两张图怎么算、只能用哪些数据、最多圈多大面积、怎样把一串余震算作一个独立震群，以及到多少样本才判断。当前还只是准备考试，不是已经考出新成绩。

## 完整工作线

| 工作线 | 状态 | 科学含义 |
| --- | --- | --- |
| S0 全数据与路线复审 | 已闭合 | 确认科学优先和真实证据边界 |
| D1 历史因果回放与异常置乱 | 已闭合并推送 | `B0_R30` 是当前最佳简单候选；异常不再复杂化 |
| P1-0A 前瞻预登记 | 已闭合并远端回读 | 只冻结规则，无真实效果 |
| P1-0B 双模型纯合成 | 下一阶段，已获启动授权 | 只检验计算和图件逻辑，不读新增真实目录、不联网 |
| P1 真实按时发行 | 未授权、0期 | 必须等待前两步分别闭合后另行授权 |
| P1 30/90天成熟评价 | 未开始 | 只能评价未来按时保存的真实预测 |
| 锁定测试 | 未打开 | 当前继续禁止 |

## P1-0A 已冻结的科学方案

- 比较 `B0` 与 `B0_R30=0.75*B0+0.25*R30`；`alpha=0.25` 是 D1 三折一致选择，未来不调。
- `B0` 使用 1970 年起、起源和可用时刻不晚于 Q、位于研究区/support 的 M4+ 事件等权构造 75 km Gaussian KDE；`common_mc=4.0`，局部 Mc 只作用相应本地单元且不重算；`R30` 使用相同 KDE。
- 每周四 `00:00 Asia/Shanghai` 起报，共同查询截止 `Q=T-15min`。
- `valid_from` 和首个 scheduled issue 固定为 `2026-09-10T00:00:00+08:00`。
- 本地目录只作截至 `2026-07-09T04:25:56Z` 的历史 `B0`；此后新增输入和真值唯一来自 ComCat，首期已有 60 天同源洗脱。
- 切源候选严格按“起源时间差 → WGS84 距离 → 震级差 → stable source event ID”升序确定性一对一去重，本地记录优先，不能人工换配。
- 同一期使用同快照、同支持域、同 25 km 网格和同完整格面积规则。`B0` 先在 600,000 km²上限下形成参考实际面积 `A_ref`，`B0_R30` 只能取 `<=A_ref` 的完整格前缀；面积差必须小于挑战者下一格且小于 625 km²，否则不可评价。
- 主评价是未来 30 天 M5–6 独立震群严格召回，90 天为次级。
- 每个 horizon 只选满足 `T_next>=T_prev+h+30d` 的下一 on-time issue；30 天 guard gap 使震群不能跨入选窗口重复。各成熟窗口内做 30 天/75 km 分群，最早事件作代表并唯一归属该 issue。
- 序贯复审只使用 30 天主终点；10/20 群只报告并继续，36 个月前达到 30 群或先到 36 个月即最终判定，终判不能继续积累。
- 无正方向就停止 `B0_R30`；正向但未达到至少 +5 pp 且序贯调整区间下界大于 0 时统一为 `report_uncertain_at_final_review`；达到该门才称强确认。
- 达到相同召回少用约 8% 面积是次级实用成功。
- 每个按时期的原静态图和离线交互页不可覆盖；真值成熟后只新增回放。
- 不使用异常、断层、人工预测、树模型、神经网络或锁定测试。
- RFC3161/TSA、证书链和硬件收据已经退休，不再耗时建设。

## 9 月 10 日与中断规则

P1-0A、P1-0B、真实代码和授权若没有在某个起报 `T` 前分别完成验收、提交、推送和远端回读，则该期不能事后补做。

具体到首期：若真实执行代码和授权未在 `2026-09-10T00:00:00+08:00` 前闭合，9 月 10 日永久登记为 `missed`，没有预测、没有效果。协议科学完整时，后续周四仍继续排期；真实授权闭合后只能从下一个尚未到达的规则周四首次 `on_time`。不得查看 9 月 10 日之后的地震再改公式、窗口、面积或判断门。

若发生未来泄漏、原预测可覆盖、两模型使用不同快照/支持/面积规则等科学完整性失败，则暂停 P1 并另立更晚的 `valid_from`，不能把它当作普通 missed 后继续。

P1 只有一条只追加记录链：后续记录实现验收后，先如实创建尚未授权的 `ProtocolDefinition`
genesis；未授权规则期继续接 `MissedIssueRecord`，但不得倒签时间。P1-0B、真实代码和授权以后闭合时，再新增
`RealIssueAuthorizationRecord`；只有它早于某个 `T`，该 `T` 才可接 Forecast。预测、missed、真值和
序贯复审都不能另起旁链。P1-0A 当前只是定义合同，实际记录仍为 0。

当前 selector 对 30/90 天分别从 on-time issue 按时间机械选择带 30 天 guard gap 的窗口；旧 v0.2.5 的三模型、
7/30/90 宏平均和旧 selector 只作历史。30 天封口震群按 `(代表时间,event_id,issue_id,cluster_id)`
排序，look 永远取前 10/20/30；90 天不进入 `SequentialReviewRecord`。一次成熟批次跨多个门时按
10→20→30 连续写记录。36 月先到时 `elapsed_months=36`，prior look/count 只能为
`0→0..9、1→10..19、2→20..29`，终端 look=`prior+1`；同刻达到 30 群只写 `cluster_30`。36 月 0 群
时命中写 0，效果字段和 Bootstrap 区间字段写 null，decision=`report_evidence_insufficient_at_final_review`。

## 当前精确文件

P1-0A 候选范围固定为：

- `docs/p1_b0_r30_prospective_preregistration.md`
- `configs/p1_b0_r30_prospective.yaml`
- `data/contracts/p1_prospective_records_v1.json`
- `data/manifests/p1_source_boundary_manifest.json`
- `data/manifests/p1_model_manifest.json`
- `tests/unit/test_p1_b0_r30_preregistration.py`
- `docs/p1_0a_acceptance_2026-08-30.md`
- `docs/restart_handoff_2026-08-30_p1.md`
- `SEISMOFLUX_IMPLEMENTATION_HANDOFF.md`

旧 `v0.2.5-prospective-science-mvp-*` 和旧 P0/P1/PP 合成结果只作历史记录，不能直接授权本 P1。现存 Stage4 未跟踪草稿不属于本阶段，禁止删除、移动、暂存或提交。

## 中断后如何继续

工作树：`D:\AIPred\SeismoFlux\data\interim\worktrees\science_first`。

1. 运行 `git status --short`，确认 P1-0A 已闭合，且原有 15 个 Stage4 未跟踪草稿仍未暂存；不要回退或纳入它们。
2. 核验分支历史包含冻结提交 `d793d3359e7caf71efbef32b76bb887f5511ad78`，协议标签解引用到同一提交。
3. 紧邻下一步只做 P1-0B 双模型纯合成：不联网、不读本地目录截止之后的真实新增目录、不读未来真实目标。
4. P1-0B 必须覆盖正、零、负效果和 36 月零群边界，生成静态 SVG 与完全离线交互 HTML，并验证挑战者面积不超过 B0 参考面积。
5. P1-0B 自身验收、提交、推送和远端回读前，`real_issue_authorized`、`real_catalog_read_authorized` 和 `real_network_fetch_authorized` 均保持 false。

## 机器可检索的当前决定

- `science_value_category`: `necessary_enabler`
- `direct_prediction_improvement`: `none`
- `evidence`: D1 的 `B0_R30` 历史外推由 5/21 提高到 9/21；P1-0A 只冻结未来公平检验，真实 issue 和新效果为 0。
- `decision`: `P1_0A_closed_start_P1_0B_synthetic_only`
- `next_scientific_test`: 双模型纯合成正/零/负情景的同面积地图、静态图、离线交互页和评价结果。
- `stop_condition`: 合同或合成公平性不能闭合则不发行真实 P1；不以异常、复杂模型、锁定测试或工程证明链绕过。
