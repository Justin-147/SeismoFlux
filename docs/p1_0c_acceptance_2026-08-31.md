# SeismoFlux P1-0C 科学验收（2026-08-31）

> 本记录是 P1-0C 最终科学验收与远端闭合结论。代码、图件和记录已经提交、推送并远端回读；
> 这仍不构成真实起报授权，`real_issue_authorized=false` 保持不变。

## 一句话结论

P1-0C 科学验收和远端闭合通过。每一个命中都能从 exact raw truth bytes 重新经过“标准事件、合法暴露、
独立震群、代表事件落格、固定报警格、命中和复审”机械算回；坐标、格号、评分、复审时间或原始
字节被协调替换都会失败。这个阶段是未来真实试验的必要准备，不是新的预测效果；
`real_issue_authorized=false`，锁定测试未运行。

## 外行版：这次到底做了什么

可以把它理解成给未来比赛加了一条不可跳过的验算链。以前一张“命中 10 个”的汇总表只要连同
自己的校验码一起替换，通用检查仍可能被蒙混过去。现在系统会从原始地震记录重新做一遍：哪些
记录当时已经看得到、哪些属于评价窗口、哪些只是同一场地震序列、代表震落在哪个固定格子、这个
格子是否在当时保存的报警区，最后才得到命中数。中间任何一处被替换都会失败。

同时生成了两类直观图件：

- 起报静态图与离线交互页：只显示起报时已经可知的输入、单位面积相对强度、排名和报警格，绝不
  显示未来地震；
- 成熟回放静态图与离线交互页：另存一份合成已知答案，绿色圆点表示命中，红叉表示漏报，用来证明
  原始字节到分群和评分的机械链条能正确工作。它不是实际预测成绩。

## 使用了哪些数据

本阶段只读取三份冻结的本地输入：

1. 历史地震目录：
   `data/processed/stage1/debc98054172a4a1/earthquake_event.parquet`，40,898 条，文件 SHA-256 为
   `2193514eec2889dbf4ae9598c5d45ef8961a8f3fcd26c7183b233dbe20842347`；目录最大
   origin/available 时刻均为 `2026-07-09T04:25:56Z`。
2. 研究区：`data/processed/china_mainland.geojson`，SHA-256 为
   `5e5dcf012e080882161c95bf592a1ee39a0f0fdad7114bcff58d645aeb30bb02`。
3. 冻结支持域清单：`data/manifests/background_local_support_manifest.json`，support_id 为
   `local-support-f6816ab6c6581306`；61 个 500 km 单元中 52 个 supported、9 个 indeterminate、
   0 个 unsupported，保留面积约 9,415,305.754 km²，公共完整震级阈值 Mc=4.0。

局部支持状态只约束自己的 500 km 单元，不会把某个局部 Mc 或不确定性传播到其他区域。本阶段没有
读取异常表、断层特征、人工预测、网络目录、起报截止后的未来目标或锁定测试。

## 数据怎样变成两张预测图

- 起报演练时刻 `T=2026-09-09T16:00:00Z`，查询截点 `Q=T-15min`。
- `B0` 使用 Q 前合法可见、达到 Mc 的长期地震，75 km Gaussian KDE 形成长期背景。
- `R30` 只使用 Q 前最近 30 天的同口径地震。
- 挑战者固定为 `B0_R30 = 0.75*B0 + 0.25*R30`，没有再训练或调参。
- 在冻结的 15,697 个 25 km 裁剪格上，按“格内标准化质量 / 该格真实面积”排序；两模型各自只能取
  不超过 600,000 km² 的完整前缀，挑战者不能比 B0 多圈面积。
- 图上颜色也使用同一个“质量 / 真实面积”量，并让两模型共享色标；相对强度不是绝对发震概率。

## 真实冻结历史适配演练结果

| 项目 | B0 | B0_R30 |
| --- | ---: | ---: |
| 入选历史事件 | 5,991 | B0 5,991；R30 0 |
| 完整报警格 | 995 | 995 |
| 实际报警面积 | 599,494.373 km² | 599,494.373 km² |

冻结目录止于 2026-07-09，而演练查询截点是 2026-09-09，因此最近 30 天窗口没有事件。按照预登记，
空 R30 时挑战者必须逐位退回 B0；本次两张图的标准化质量逐位相同，面积差为 0。这是对真实历史输入
接线和公平性的检查，不是“新模型没有用”的效果结论，更不是一次真实起报。

局部高 Mc 不会影响其他区域：25 km 格只继承自己所属 500 km 单元的支持状态；负列号边界也按数学
向下取整映射。本次 15,697 个格中 15,008 个属于 supported 单元、689 个属于 indeterminate 单元、
0 个属于 unsupported 单元。

## 原始字节到评分的已知答案

合成成熟回放从 3,063 字节原始响应重新得到 10 个独立震群；B0 与 B0_R30 均命中 6 群、漏报 4 群，
召回 60%，差值 0 个百分点。之所以是零差异，是因为它绑定到这次空 R30 的起报图；目的只是验证
“原始响应→分群→命中”的链条，而不是制造好看的效果。

主动反例已经证明以下替换都会被拒绝：

- 同时替换得分清单、声明哈希和复审；
- 替换震群成员或代表事件；
- 换成同面积但不同位置的报警格；
- 替换原始目录字节；
- 保持其余标识不变，把代表事件改到另一合法格；
- 让平面坐标、经纬度和格号不再由同一位置往返得到；
- 把回放月数改为 3.0 并同步重建整条 `cluster_10` 复审；
- 把 Q 之后才可见的事件塞进预测；
- 把真值抓取后才出现的修订塞进评分；
- 用缺失真值冒充“成熟后确实 0 群”。

已在抓取时可见、但位于评价窗外、震级不在 M5–6 或研究区外的完整响应行可以保留，只是不参与
目标计分；这样既不丢原始证据，也不会误算。

## 图件与可复算产物

- 静态起报图：[`p1_0c_real_history_forecast.svg`](p1_b0_r30_preflight/p1_0c_real_history_forecast.svg)
- 离线交互起报页：[`p1_0c_real_history_forecast.html`](p1_b0_r30_preflight/p1_0c_real_history_forecast.html)
- 静态合成成熟回放：[`p1_0c_synthetic_mature_replay.svg`](p1_b0_r30_preflight/p1_0c_synthetic_mature_replay.svg)
- 离线交互成熟回放：[`p1_0c_synthetic_mature_replay.html`](p1_b0_r30_preflight/p1_0c_synthetic_mature_replay.html)
- 科学结果：[`p1_0c_preflight_result.json`](p1_b0_r30_preflight/p1_0c_preflight_result.json)
- 逐文件清单：[`p1_0c_preflight_manifest.json`](p1_b0_r30_preflight/p1_0c_preflight_manifest.json)

起报 SVG SHA-256 为
`ed465504c262666af0076603dcb7f850962af041177930ffa1e61ff6cbdf5bcb`；结果 JSON SHA-256 为
`c6e04cc52b7bca284e9ae6dedb44edfe07e923d73b391221aab8cda2162dc7b6`；manifest 文件自身 SHA-256 为
`034ed00bd16e91d7786046d1ff6f6444d2abbde750ebf0491d84874999acce99`。

6 个受 manifest 管理的科学产物加 manifest 自身，共 7 个文件；它们在约定的单进程、数值库单线程
环境中独立生成两次，逐字节一致。静态 SVG、起报交互页和成熟回放页均完成实际渲染目视检查；
交互页不引用外部网络资源。

## 验收证据与科学价值复审

- P1 联合科学检查：`127 passed in 228.40s`，单进程且数值库单线程。
- Ruff 格式与静态检查通过；严格 Mypy 检查 5 个源文件和 3 个测试文件通过；
  `git diff --check` 通过。
- 最终独立科学审计：39 项针对性检查通过，结论 `GO`，`P0=0`、`P1=0`；协调修改回放月数、
  代表格、坐标、评分和复审均被拒绝。
- `network_accessed=false`、`locked_test_run=false`、`real_issue_authorized=false`。

科学价值分类为 `necessary_enabler`，不是 `direct_prediction_improvement`。它没有增加新的真实命中
证据，但消除了“汇总成绩可被整套替换”的科学阻断，并把真实冻结目录接到了可核验的双模型地图。
历史开发证据仍然只是 D1 的 `B0 5/21 → B0_R30 9/21`；是否在真正未来仍有提升，只能由后续按时
保存、到期后再评价的预测回答。

## 阶段决定

- `acceptance`: `P1_0C_PASS`
- `stage_status`: `accepted_committed_tagged_pushed_remote_readback_complete`
- `science_value_category`: `necessary_enabler`
- `direct_prediction_improvement`: `none_new`
- `real_issue_authorized`: `false`
- `science_commit`: `c71c97790adcf33f6c8121e367317857dc8dff31`
- `annotated_tag`: `v0.2.7-p1-b0-r30-code`
- `science_remote_branch_readback_before_closure_docs`: `c71c97790adcf33f6c8121e367317857dc8dff31`
- `remote_tag_peeled_readback`: `c71c97790adcf33f6c8121e367317857dc8dff31`
- `closure_record_commit`: `24ea3627f9d10d254658d211fdae8d3fcf56f700`
- `decision`: `P1_0C_closed_stop_engineering_wait_for_separate_real_issue_authorization`
- `stop_condition`: 没有另行显式 `RealIssueAuthorizationRecord` 不得进入真实起报。
