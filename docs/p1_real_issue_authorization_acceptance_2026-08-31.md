# SeismoFlux P1 真实前瞻授权科学验收（2026-08-31）

## 外行版结论

已经把用户的授权、冻结规则和允许开始的日期连成一条公开且不能随意改写的记录。它相当于给未来
预测比赛盖了“从 2026 年 9 月 10 日零时开始、不能补交、不能改题”的印章。现在还没有生成第一张
真实预测图，也没有新的命中成绩；下一项真正科学工作是在 9 月 9 日 23:45 用当时可见的目录保存
首期预测，然后等待未来地震检验。

## 验收身份

- 冻结科学代码：`c71c97790adcf33f6c8121e367317857dc8dff31`。
- 冻结协议：`0f43f15bc983a37157f1b129976c7ec0ea47fc7d`。
- 已验收发行通路：`db93e9ac8cd2997859fdd8815fa4d026b10fd95e`，标签
  `v0.2.7-p1-b0-r30-ops` 已远端回读。
- 公开授权证据与 genesis 提交：`a27417eaeb89955dd0e688d9d2c1b5f19f3da302`。
- `ProtocolDefinition` 哈希：
  `1a8e44f4a56fde0129e913a95818e05f07b0b1a9f4f2b256bb7dddcedced89c3`。
- `RealIssueAuthorizationRecord` 哈希：
  `184673b3b9e11e099bd8dcb4188b5d2e6c1497623519abcc2e36258a482600b1`。
- 远端核验与记录时间：`2026-08-31T04:36:13.436543Z`。
- 授权生效 T：`2026-09-09T16:00:00Z`；首期 Q：`2026-09-09T15:45:00Z`。

## 科学边界

- 只授权冻结的 B0 与 B0_R30；75 km KDE、30 天 R30、0.75/0.25 权重、支持域、格网、至多
  600,000 km² 和 30/90 天指标全部不变。
- 授权记录形成期间没有获取首期真实 ComCat 目录、没有读取未来 M5–6 真值、没有运行锁定测试。
- 预测不能补发；若首期不能在 T 前公开闭合，只能记录 missed 并等待下一合法期。
- 当前仍只有历史开发证据 `B0 5/21 → B0_R30 9/21`。授权本身不增加预测效果。

## 科学价值复审

- `acceptance`: `PASS`
- `stage_status`: `authorization_record_ready_for_remote_closure`
- `science_value_category`: `necessary_enabler`
- `direct_prediction_improvement`: `none_new`
- `evidence`: `public_authority_and_frozen_start_boundary`
- `decision`: `stop_authorization_engineering_after_remote_readback_then_issue_first_forecast`
- `next_scientific_test`: `2026-09-09T15:45:00Z_first_real_issue`
- `stop_condition`: `no_backfill_no_model_change_no_future_information`

本阶段提交和远端回读完成后，不再继续扩展授权基础设施；资源直接转向首期真实前瞻和随后 30/90 天
成熟真值。最早 30 天结果为北京时间 2026-11-09，最早 90 天结果为 2027-01-08；可靠结论仍需积累
足够多的独立 M5–6 震群，不能把“授权完成”说成“预测完成”。
