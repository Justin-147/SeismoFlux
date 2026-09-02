# SeismoFlux重启续接：审计纠偏与C2A有限对照（2026-09-02）

> 当前最高入口。替代上一份`restart_handoff_2026-09-02_s1c2_protocol_pause.md`的暂停状态，
> 不删除旧记录。总蓝图第1.11节为当前执行权威。

## 1. 外行版结论

项目还没有全部完成。已有目录模型在相同约60万km²、30天条件下，将147个Ms5–6独立震序的命中
从32个提高到46个、再到54个。这是2000—2019开发历史中的线索，不是最终独立证实。

本次重启后查清两件事：

1. 95%规则没有改坏此前的预测分数，但挡住了C1尚未开始的预测；单纯降到80%仍会被早期稀疏地区
   挡住。后续改为直接比较局地处理的效果，不再全国一票否决。
2. 用户两份目录默认都是Ms。这只补全类型说明，没有改变计算，所以入库、样本数、旧预测和分数
   都不需要重跑。机器空值原样保留，以来源补充说明解释，不随意转换Mw。

补充对照C2A已经完成：同一147个震序、约60万km²，原长期＋近期模型命中54个，A仅去掉确定高Mc
地区后命中52个，B再去掉Mc未知地区后命中53个。主指标没有提升。B的长期模型在30万km²有多命中
1个的小正结果，其余面积无稳定优势，也完整保留。保留C0基线，结束这条筛选分支，不再重跑C2A。

下一项是C2B有限覆盖面板与模型比较，随后进入多数据研究。当前并没有完成整个项目。

## 2. 工作位置与基准身份

- 工作树：`D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata`
- 分支：`codex/p2r-multitask-multidata`
- 恢复起点提交：`173e9f140ffe60332eba912401fa18770bbfd636`，本轮开始时与远端一致且工作树干净。
- S1-C0科学结果：`33ad8bc8aaa76f7e85d3949bfe82471c42b7521f`；闭合：
  `d5ddfa62edc69eb5c1c3c1d2de7f4f0c1056c23f`。
- S1-C1诊断：`5a99f4ce446320cd955e47966cb060ab1252067a`；闭合：
  `3339931796fa0ff0921c6d2b5fab350b110cde7e`。
- 本轮审计、协议和下一执行阶段的实际闭合状态见本文第7节，不能把起点SHA冒充新提交。

## 3. 现在必须读的文件

1. `SEISMOFLUX_IMPLEMENTATION_HANDOFF.md`第1.9—1.11节；
2. `docs/s1_support_threshold_magnitude_audit_2026-09-02.md`；
3. `data/manifests/catalog_magnitude_semantics_2026-09-02.json`；
4. `configs/multitask_s1_c2a_input_sensitivity.yaml`；
5. `docs/s1c2a_scientific_results_2026-09-02.md`（真实结果、图件、科学价值复审与验收）；
6. `docs/SEISMOFLUX_MULTITASK_MULTIDATA_SCIENCE_PLAN_2026-09-01.md`总路线；
7. 上一暂停交接第4—7节的覆盖文献与后续C2方案，尚未冻结的模型参数仍不能冒充已决定。

旧文档中“95%未达所以不能再研究”和“本地震级类型未知”的当前解释，均以本轮补充为准。原文件
和原产物不改，避免把现在的认识写回当时的研究记录。

## 4. 已完成的C2A精确范围

- 仍是2000—2004、2005—2009、2010—2014、2015—2019四折；每折29个30天主曝光，共116期。
- Ms4+、1970起全源权威目录。每期只用当期前24小时已发生且可获得的事件。
- 使用C0已由更早内层选定的L1/L2/L3参数：四折L1收缩年数5/1/1/5，L2带宽均75km，L3近期权重
  均0.25。它们在C2A不重新选择。
- 两种处理：A仅排除C1确定Mc>4的训练中心、保留未知格；B再排除未知格。每种处理的三个模型
  使用同一输入。C1四个外折起点掩膜整折固定，不重新估Mc。
- 复用C0不加掩膜的参照预测。C2A仅保存位置质量数组，不重算C0五时限、时间或震级模块。
- 全域15,697个25km格、同一147个Ms5–6固定首震锚点、原五档面积、严格0km命中。
- 六条新曲线、四折全部保存并核对后才统一评分；C0目标身份只在score阶段读取。
- 输出根：`outputs/multitask_s1/s1c2a_input_sensitivity_v1`，不覆盖C0/C1/P1。
- 这是开发期固定参数输入敏感性，不是重新优化后的最好能力，也不是最终独立验证。

## 5. 其后怎么做

C2A完整报告已经形成，这项局地筛选问题到此结束。C2剩余工作称为C2B：有限的覆盖面板对照、
多尺度KDE、指数时间衰减和低维ridge组合；按问题分别设计，不能与两个掩膜做全交叉搜索。
当前C2B精确配置尚未冻结，也没有开始训练或查看任何C2B候选成绩。先把文献已支持的机制和
有限候选落实，完成协议验收与提交推送后运行；不要把这段准备扩展成新一轮完整性门控。

数据面板比较用同一固定模型；模型改进比较要用相同数据基线；若有新超参，只在对应外折之前的
内层选择。D1面板同时改变来源成员和起始年代时，只能称“面板整体差异”；若不增加桥接对照就
不能声称已独立分离年代效应。ridge不做去掉M5特征的对照就不能把整体增益单独归因于M5。

开发期允许依据新认识修订方案并记录，不把项目当比赛；但不打开最终留出调参，不选择性隐藏负结果。
C2目录问题完成后转入S2断层、危险性和应变，再做S3异常，不无限拖延多数据研究。

## 6. S0—S7和资源

| 阶段 | 通俗任务 | 当前状态 |
| --- | --- | --- |
| S0 | 查清数据、规定时间/地点/震级任务与样本 | 已完成 |
| S1 | 比较目录模型及目录处理是否真的提高预测 | C0、C2A已完成；C1无预测停止；C2B有限模型比较待开展 |
| S2 | 加断层、危险性、应变，看中长期增量 | 未开始 |
| S3 | 加205期异常，看是否超过目录及置乱对照 | 未开始 |
| S4 | 有限树模型与一个小型神经/多任务模型 | 未开始 |
| S5 | 组合互补模型，拆分数据和特征贡献 | 未开始 |
| S6 | 稳健性、静态图、交互页、论文、PPT、可编辑poster | 未完成，已有早期图件不是最终结论 |
| S7 | 保存不回填的真实未来预测 | 并行，不阻塞历史研究 |

心跳`seismoflux`已恢复ACTIVE，每30分钟；每次说明用了哪些数据、方法、训练/开发/独立检验效果、
完整S0—S7线以及离最终目标还差什么，并检查偏离与纠偏。未有新预测成绩时要直接说没有。

机器为24物理/48逻辑核心。C2A预测实际2线程、BelowNormal优先级，抽样占整机CPU约3.97%；
评分单进程单数值线程，抽样约2.08%。计算与绘图均已退出，末次检查没有Python训练/评分进程。
新任务默认2、最多3个折工作线程，数值库单线程，至少保留2个物理核心。

## 7. 本轮阶段验收与安全恢复点

当前状态：`S1_C2A_SCIENTIFIC_COMPLETE_PENDING_RESULT_COMMIT_PUSH`。
审计与协议已由提交`3c2948d125a532672df65c3da543113ac7c79bb8`完成验证、独立科学复审、推送与
远端回读；验收记录为`docs/s1c2a_audit_protocol_acceptance_2026-09-02.md`。最小位置对照实现
验收为`docs/s1c2a_implementation_acceptance_2026-09-02.md`，25项聚焦合成验证和
真实文件只读预检通过；实现提交`35e33f1895cb8f275e879732acea184345a67e1f`已推送并远端回读。
真实预测已于北京时间18:12:49启动，PID为31500，2个折线程、BelowNormal优先级。
18:14:08抽样为整机CPU3.97%、内存约304.5MiB；这不是所有程序的总占用。
四个外折已于北京时间18:16:47完成保存（116/116期、100%）。预测清单SHA-256为
`68a5e7f0c06b8b0ec10bb9cbc580471bdfa5a60f66d7a7ff1eefd5915509986f`。
18:17:14启动的首次评分在校验旧C0目标起报期时停止，报错为
`extra, duplicate, or non-development target issue`。根因为评分适配把同一个日期的纳秒整数误当微秒，
不是目标改变或重复。已只修复该转换，新增独立日期常量回归，31项聚焦测试通过；独立复核116期、
34空期及147个锚点与预测完全一致，未重算预测、未改变评价。
评分于18:22:24以单进程PID32428恢复，18:26:22成功结束；该进程已退出。主指标C0为32/46/54，
A为32/46/52，B为32/46/53。真实产物独立复核、两张静态图目视验证、离线数据与JS语法检查通过。
浏览器安全策略拒绝本地file地址，点击交互未实测；不绕过限制，也不把它变成科学研究停止门。
本次运行日志为`data/interim/c2a_logs/predict_20260902T181249.stdout.log`及对应stderr文件。
真实成绩与验收见`docs/s1c2a_scientific_results_2026-09-02.md`。待本轮结果提交、推送与远端回读后，
进入C2B有限协议设计；此处不预先填入尚未产生的科学结果提交SHA。

```yaml
audit_decision: retain_existing_results_and_run_finite_new_input_sensitivity
catalog_values_changed: false
S1_C0_rerun_required_for_95_or_Ms: false
S1_C1_Mc_recomputation_required: false
S1_C2A_predictions_created: complete_four_of_four_folds_116_issues
S1_C2A_scores_read: true
S1_C2A_score_status: complete_same_predictions_same_targets
S1_C2A_scientific_acceptance: PASS_no_new_stable_predictive_gain
S1_C2A_result_commit_pushed: false
S1_C2B_protocol_frozen: false
S1_C2B_predictions_created: false
heartbeat_status: ACTIVE
holdout_opened: false
audit_2023_plus_opened: false
locked_test_run: false
science_first_Stage4_drafts_touched: false
```

安全顺序：核验已完成结果，不重跑C2A、不改C0/C1/P1；确认本轮结果已推送后，转入第5节的C2B。
若发现中断留下未提交文件，只核对本轮已列明代码/图件/文档，不使用`git add -A`带入旧草稿。

已有图件入口（相对于本工作树）：

```text
outputs/multitask_s1/s1c2a_input_sensitivity_v1/visualization/01_main_anchor_hits.png
outputs/multitask_s1/s1c2a_input_sensitivity_v1/visualization/02_l3_area_curves.png
outputs/multitask_s1/s1c2a_input_sensitivity_v1/visualization/seismoflux_s1c2a_replay.html
```

预测清单SHA：`68a5e7f0c06b8b0ec10bb9cbc580471bdfa5a60f66d7a7ff1eefd5915509986f`。
评分摘要SHA：`4ff2ad36df4d85fb958599631ca59182e8f741253bcc7860484afa293da6432e`。
评分清单SHA：`0651d3483acf6f48a3b287d0e1f11aba26425e67b63126b8f89eee3e5b02221c`。
逐事件资料与离线页仅本机保存；公开同步只含聚合结果、静态图、代码及文档。

## 8. 最新科学价值复审

```yaml
science_value_category: direct_ablation_evidence_without_new_predictive_gain
evidence: same_147_anchors_C0_54_A_52_B_53_at_600000km2_30d
decision: retain_C0_reference_close_fixed_parameter_mask_branch
next_scientific_test: finite_C2B_coverage_and_model_comparison_then_S2_multidata_increment
stop_condition: no_further_C2A_threshold_mask_or_parameter_search
project_completed: false
```

本轮排除的是一个没有稳定增量的固定参数处理分支，不是否定全部完整性方法。下一步必须贡献新的
模型或数据效果证据，不能继续在95%、类型空值、日期适配或图件工程上循环。
