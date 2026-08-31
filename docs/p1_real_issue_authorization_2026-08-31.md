# SeismoFlux P1 真实前瞻用户授权证据（2026-08-31）

## 用户授权原文

我授权依据冻结的 P1 v0.2.7 协议创建 RealIssueAuthorizationRecord，并从下一个合法规则时刻开始真实前瞻预测；不得补发、不得修改冻结模型或利用未来地震信息。

## 授权边界

- 授权对象：冻结的 `P1 v0.2.7`，即 `B0` 与 `B0_R30` 的真实前瞻比较。
- 冻结科学代码：`c71c97790adcf33f6c8121e367317857dc8dff31`，标签
  `v0.2.7-p1-b0-r30-code`。
- 冻结协议：`0f43f15bc983a37157f1b129976c7ec0ea47fc7d`，标签
  `v0.2.7-p1-b0-r30-protocol`。
- 已验收发行通路：`db93e9ac8cd2997859fdd8815fa4d026b10fd95e`，标签
  `v0.2.7-p1-b0-r30-ops`；公开分支和标签剥离值均已远端回读到该提交。
- 本次生成的 `ProtocolDefinition` 内容哈希：
  `1a8e44f4a56fde0129e913a95818e05f07b0b1a9f4f2b256bb7dddcedced89c3`，记录时间
  `2026-08-31T04:32:53.005504Z`；其中 `real_issue_authorized=false`，用于先冻结规则，再由下一条
  `RealIssueAuthorizationRecord` 引用本文件的远端提交后显式启用。
- 首个获授权的规则起报时刻：`2026-09-09T16:00:00Z`
  （北京时间 `2026-09-10T00:00:00+08:00`）。
- 对应共同查询截止：`2026-09-09T15:45:00Z`
  （北京时间 `2026-09-09T23:45:00+08:00`）。
- 如果不能在该期 `T` 前公开闭合预测包，只能记为 `missed`，不得事后补发；协议仍完整时再等待
  下一个尚未到达的合法规则时刻。

## 科学含义

这份授权只允许开始一场真正的未来检验，不代表模型已经取得新的真实预测效果。冻结历史开发证据
仍是同约 600,000 km² 报警面积下 `B0 5/21 → B0_R30 9/21`。授权后必须先在地震发生前保存地图，
再等待 30/90 天真值成熟，才能判断加入近期活动是否提高独立 M5–6 震群召回。

- `science_value_category`: `necessary_enabler`
- `direct_prediction_improvement`: `none_new`
- `decision`: `authorize_frozen_real_prospective_only`
- `next_scientific_test`: `first_on_time_issue_then_wait_30_90_day_truth`
- `stop_condition`: `no_backfill_no_model_change_no_future_information`

## 链上授权记录

- 公开授权证据提交：`a27417eaeb89955dd0e688d9d2c1b5f19f3da302`。
- `RealIssueAuthorizationRecord` 内容哈希：
  `184673b3b9e11e099bd8dcb4188b5d2e6c1497623519abcc2e36258a482600b1`。
- 远端核验与记录时间：`2026-08-31T04:36:13.436543Z`。
- 状态：`real_issue_authorized=true`，从 `2026-09-09T16:00:00Z` 起只允许按冻结协议真实前瞻。
- 截至记录形成时，真实 Forecast 仍为 0 期；没有提前读取首期目录或未来真值。
