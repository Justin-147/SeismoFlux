# 最新续接：S1-C2B有限目录模型比较（2026-09-02）

> 当前最高恢复入口，接替`restart_handoff_2026-09-02_s1c2_resumed.md`；旧结果保留。
> 21:19统一评分完成；最终静态图、离线回放和科学复审均已通过。主任务由54/147提高到58/147。
> 当前为`S1_C2B_SCIENCE_ACCEPTED_PENDING_GIT_CLOSURE`；完成本次提交推送后关闭S1，进入S2。

## 1. 先用通俗话说明

项目没全部完成。我们已有一个“历史地震热点＋近期活动”的目录参考模型：在2000—2019开发
历史、30天、约60万km²报警面积下，命中147个独立Ms5–6震序中的54个。简单地区频率模型为32个，
长期平滑模型为46个。它们不是最终独立测试，也不能把这147个当全部目录的可用样本。

95%门和震级类型的审计已完成。两份用户目录均按Ms解释，旧数值与分数不用重跑；95%没有改坏
C0分数，但曾把C1挡在预测之前。补充C2A已经实际比较：原模型54个、仅删确定高Mc地区52个、再
删未知地区53个，没有稳定提高。这个分支已结束，不再反复调整全国比例或目录类型。

本轮已回答：改变热点范围和近期事件表示确实有开发期增益。30天、Ms5–6、同样60万km²预算，
原参考命中54/147，多尺度58/147、相对年龄加权57/147；多尺度新增14个、丢失10个，净增4个。
180天Ms5–6由70/237提高到88/237，但全年Ms≥6由11/45降到10/45，不能称所有任务都更好。
训练只用更早历史，较晚的2000—2019分四折评价；这是开发证据，不是最终独立检验或未来保证。
数据、训练、所有正负结果及图件见`docs/s1c2b_scientific_results_2026-09-02.md`。

现在不再调目录门槛或继续扩张本轮参数。下一项真正科学工作是加入用户已有断层资料，比较在同一
目录背景上加/不加断层后，特别是90/180/365天及大震任务，能否进一步减少漏报。

## 2. 工作位置、必读文件与不能动的边界

- 工作树：`D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata`
- 分支：`codex/p2r-multitask-multidata`
- Python：`D:\AIPred\SeismoFlux\.venv\Scripts\python.exe`
- 根库、science_first既有Stage4未跟踪草稿、冻结P1：不修改。
- 不读取2020—2022留出和2023+审计目标，不运行锁定测试，不联网取未来目录。
- 历史C0/C1/C2A预测、评分和冻结配置不回写；禁止`git add -A`混入其它工作。

继续前读：

1. 总蓝图`SEISMOFLUX_IMPLEMENTATION_HANDOFF.md`第1.9—1.13节；
2. 本文；
3. `docs/s1c2b_methods_protocol_2026-09-02.md`；
4. `configs/multitask_s1_c2b_catalog_models.yaml`；
5. `docs/SEISMOFLUX_MULTITASK_MULTIDATA_SCIENCE_PLAN_2026-09-01.md`（S0—S7，不重跑已完成S0）；
6. 若需要旧审计/真实结果，读`docs/s1_support_threshold_magnitude_audit_2026-09-02.md`及
   `docs/s1c2a_scientific_results_2026-09-02.md`，不要因此重开C2A计算。

已推送C2A科学提交`4f5a2bea66516d5e580ffc3ee3ba8839f424f5e7`，闭合提交
`b1c14e1b2d70c6df9b279365fd575eee8e9c05ff`。本轮C2B提交状态见第5节，不能用旧SHA冒充新提交。

## 3. 本轮完成和下一步

### 已完成的必要科学准备

- 只读核对三个目录训练面板及24个历史时间节点，未训练模型、未读取新外折目标：
  D0（1970全源Ms4+）、D1（1980小震来源成员Ms4+）、D2（1950大震来源成员Ms5+）。
  例如2000年起报前可用事件为2895/1494/1184，2020年前为5237/3836/1631；这是训练数量，不是成绩。
- 经典文献依据和简化边界落盘：固定/多尺度高斯核、时空核思想、条件点过程对数线性组合。
- 精确冻结九模型、五时限、两震级档、内层选择和相同目标/面积比较。C2B的D1/D2仅为局部面板名，
  不与总方案里的近期/断层消融编号混淆。
- 独立科学复审已完整阅读精确协议，因果性、空块处理、面积离散差与归因界限通过；不设95%否决。

### 下一安全动作

预测、评分、最终图件与科学验收均已完成，详情见第9节。**不得再次启动C2B预测、评分或为进度条
重跑任何旧任务。** 21:44只读核验没有相关Python后台进程，整机CPU约9.5%，不是任务卡住。
当前先完成结果提交、推送与远端闭合，再进入S2。若重启发生在推送之前，核对分支状态和第9节，
仅补齐尚未完成的提交推送；不重新计算已核验的成绩。

建议工作拆分：

1. 位置数学：`src/seismoflux/multitask_s1/c2b_models.py`已实现，稳定log空间高斯核、有限混合、
   训练期加权标准化及条件ridge目标。
2. 因果面板和训练预测：`c2b_inputs.py`与`c2b_predict.py`已实现源成员映射、内层/外折日历、
   各h严格过去选择、每折每时限检查点；不调用旧完整模型跑法。
3. 评价：`c2b_score.py`已实现全四折完成后才复用C0目标；九模型及C0参考，全部h/震级/面积，
   配对新增与丢失。严格0km为主，70km仅辅助投影距离敏感性。
4. 结果一出来就解释增量，完成最小有用的静态图与离线回放，不在展示工程上拖延多数据研究。

模型细节以精确YAML为准。指数近期权重只表示相对年龄，不能宣称发震率会随安静期自动减弱。
D2特征对照只归因于整个大震历史空间特征；D0本来已含1970后的大震，不能称此前完全没用大震。
格前缀控制预算，但不同排序的实际面积可相差不足625km²，必须报告；最终最多10区仍待后续处理。

## 4. S0—S7完整工作线与科学复审

| 阶段 | 通俗任务 | 当前进展 |
| --- | --- | --- |
| S0 | 摸清数据、时间/地点/震级任务和可用样本 | 已完成 |
| S1 | 先把目录模型做成可信参考，比较目录处理与有限新模型 | C0/C2A完成；C1无预测停止；C2B科学验收通过、待本次提交推送闭合 |
| S2 | 加断层、危险性、应变，看是否改善中长期位置 | 下一阶段；S1远端闭合后先做断层几何配对 |
| S3 | 加205期动态异常，检验其超出目录的贡献 | 未开始 |
| S4 | 有限树模型和一个小型神经/多任务模型 | 未开始 |
| S5 | 组合互补方法，用消融区分数据和特征贡献 | 未开始 |
| S6 | 稳健性、直观图件、交互、论文、PPT、可编辑poster | 未完成，早期展示不是最终论文证据 |
| S7 | 保存不可回填的未来预测 | 并行，不阻塞历史研究 |

距离最终目标还差的是多数据真正能提高多少、不同预测时限/震级是否互补，以及最终独立检验，
不是等某一个未来日期。不能保证这些尝试一定提高，但可以通过有限对照尽快知道哪些有用。

```yaml
science_value_category: direct_development_predictive_improvement
evidence: main_anchor_54_to_58_of_147_and_180_day_70_to_88_of_237_with_M6_year_negative_retained
decision: close_finite_S1_after_result_git_closure_then_start_S2
next_scientific_test: paired_catalog_background_with_and_without_fault_geometry
stop_condition: do_not_extend_closed_catalog_hyperparameter_search
off_target: false
engineering_loop_detected: false
plan_adjustment: retain_multiscale_and_age_complements_move_to_new_data
project_completed: false
```

## 5. 验收、运行与恢复记录

当前状态：`S1_C2B_SCIENCE_ACCEPTED_PENDING_GIT_CLOSURE`；最终验收见第9节。以下保留协议和运行沿革。
12项配置/账本聚焦测试、Ruff与
独立科学复审已通过，见`docs/s1c2b_protocol_acceptance_2026-09-02.md`。协议提交
`b35d8a760fee1443211e619e2c7d96a97892b899`已推送并由`git ls-remote`回读确认；允许进入有限实现。
位置数学、因果面板与位置专用训练预测/评分路径已实现；合计53项聚焦验证、Ruff和独立科学复审
通过，真实数据只读预检通过。见`docs/s1c2b_implementation_acceptance_2026-09-02.md`。
实现提交`fcce264465627cf5cdeb2b361618de0b33a6fe4c`已推送并通过`git ls-remote`确认。
2026-09-02北京时间19:40:28已启动真实历史训练与预测（PID33996）。

```yaml
C2A_closed_and_pushed: true
C2B_training_panel_ledger_created: true
C2B_protocol_scientific_review: PASS
C2B_protocol_tests: PASS_12_focused_tests
C2B_protocol_git_closure: complete_remote_verified
C2B_protocol_commit: b35d8a760fee1443211e619e2c7d96a97892b899
C2B_model_implementation_status: accepted_pushed_remote_verified
C2B_implementation_commit: fcce264465627cf5cdeb2b361618de0b33a6fe4c
C2B_model_implementation_complete: true
C2B_prediction_run_started: true
C2B_saved_outer_prediction_issue_horizon_pairs_at_1942: 0
C2B_completed_horizon_checkpoints_at_1942: 0
C2B_total_horizon_checkpoints: 20
C2B_causal_component_dates_saved_at_1942: 21
C2B_causal_component_dates_total: 413
C2B_scores_created: true
C2B_actual_predictive_improvement: positive_development_location_gain_with_uncertainty
C2B_final_prediction_issue_horizon_pairs: 396
C2B_final_completed_horizon_checkpoints: 20
C2B_final_score_curve_rows: 1400
C2B_final_pairing_summaries: 918
C2B_final_static_and_offline_render: accepted_rendered_v2
C2B_scientific_acceptance: PASS
C2B_results_git_closure: pending
heartbeat_status: ACTIVE
holdout_opened: false
audit_2023_plus_opened: false
locked_test_run: false
frozen_P1_modified: false
science_first_Stage4_drafts_touched: false
```

账本SHA-256：`b08de8b0a8d2b84d581a5f0b5c4b2bb37d9d7e3f58a73f4e4e99373266ee1ed6`。
协议SHA-256：`4f497690643a64466cbd9eec358977f1c8c1d4655bc4cc9ca5a65dbc95859243`。
脚本：`scripts/audit_multitask_s1_c2b_panels.py`；12项边界/配置测试及Ruff通过。账本是聚合计数，
可随本阶段同步；原始事件与本地逐事件产物不外传。

心跳保持每30分钟。机器24物理/48逻辑核心，日常检查单进程；大任务默认2最多3折线程、数值库
单线程、BelowNormal，至少保留2个物理核心。19:12检查没有Python/pythonw后台进程，两CPU负载
31%/5%（当时全机约18%，尚未训练）。19:40已启动唯一C2B进程PID33996，2折线程、BelowNormal；
19:41两秒采样占整机CPU4.13%（约1.98个逻辑核心）、常驻内存277.9 MiB。启动前可用内存约45.7GiB。
19:42 stderr为0字节。没有完整外折检查点是因为正在先计算更早训练期，不是界面卡住。

```text
进程：33996（实际恢复前重新核验，不能仅相信旧PID）
stdout：data/interim/c2b_logs/predict_20260902T194028.stdout.log
stderr：data/interim/c2b_logs/predict_20260902T194028.stderr.log
同一输出：outputs/multitask_s1/s1c2b_catalog_models_v1
因果基础核：component_cache/issue_<microseconds>.npz
预测检查点：folds/<fold>/horizon_<ddd>/horizon_manifest.json
全部完成标志：prediction_manifest.json（必须核验四折后才能评分）
```

每个起报时点核缓存包含全部所需长期/近期空间基础，跨模型和时限复用，不包含未来目标。413个
不同历史日期包含内层与外层，不是独立震例数；其百分比也不是外折预测已完成百分比。
真实逐事件/数组产物只存本机，不能`git add -A`或把整个输出目录公开推送。

同一预测任务的恢复命令（先确认不存在相关活跃进程，不另建输出目录）：

```powershell
$env:PYTHONPATH = 'D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata\src'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'D:\AIPred\SeismoFlux\.venv\Scripts\python.exe' scripts/run_multitask_s1_c2b.py --phase predict --project-root 'D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata' --data-root 'D:\AIPred\SeismoFlux\data' --workers 2
```

正式后台启动必须设置隐藏窗口和BelowNormal。脚本在加载数值库前设单线程；输出目录的进程锁
避免重复预测。以上预测恢复命令现仅作历史说明，全部预测已经完成，不应再次启动。

## 6. 心跳检查快照：2026-09-02 19:54（北京时间）

- 已只读核验全机Python命令行：仅有本轮预测进程PID33996，仍以`--workers 2`运行；没有重复启动。
- 因果基础核已保存119/413个不同历史日期（28.8%），最近文件写入19:54:13；这是训练准备进度，
  不是独立震例数，也不是预测或项目完成率。前次19:47的81个已继续增加，任务没有卡住。
- 四折五时限预测检查点仍为0/20，最终`prediction_manifest.json`尚不存在，未调用评分、未读外折目标。
- 两秒采样：进程约占整机CPU 2.11%（1.01个逻辑核心当量），常驻内存505.4 MiB；低于普通优先级，
  数值库单线程。stderr仍为0字节。无须中止或改动这次运行。
- 95%/Ms审计和C2A补充对照已经闭合，不再重做；本次没有改变模型、输入、面积、指标或运行代码，
  没有触碰根库、冻结P1和science_first的Stage4草稿。

科学复审：本次是让已登记的九模型对照继续产出证据，尚不是预测效果提升；没有偏离受控面积下
提高独立震区召回的目标，也没有转入与预测无关的工程优化。当前不需要追加模型或调整参数。
下一步仍是等待全部预测完成、核验后统一评分；有限比较结束即转S2多数据研究，不继续扩展目录分支。
项目未完成，心跳保持开启。本快照仅为及时恢复记录，不代表新的科学阶段验收或新的推送。

## 7. 心跳检查快照：2026-09-02 20:21（北京时间）

| 较晚历史预测段 | 已保存时限 | 已保存起报—时限对 |
| --- | --- | ---: |
| 2000—2004 | 7、30、90、180、365天，5/5 | 99/99 |
| 2005—2009 | 7天，1/5 | 44/99 |
| 2010—2014 | 0/5，基础计算进行中 | 0/99 |
| 2015—2019 | 0/5，尚未完成检查点 | 0/99 |

正式预测合计143/396（36.1%），时限检查点6/20（30%）；基础核296/413（71.7%）。这些百分比
分别表示不同工作量，不表示整个项目完成率。最近基础核写入20:21:40，最近时限预测保存20:19:13；
第一折完整记录在20:18:31写入。最终预测manifest尚不存在，尚无统一分数，不提前读取外折目标。

20:21:08只读核验仍只有PID33996这一个Python预测实例，命令行、2折线程与同一输出目录不变。
两秒采样整机CPU占用4.16%（2.00个逻辑核心当量）、内存710.4 MiB、BelowNormal；stderr为0字节。
四个运行模块与协议的SHA-256均与已验收冻结版本相同。没有重启、重复训练、修改模型或新增参数。

科学价值复审：已经从训练准备推进到真实历史预测保存，但在统一评价前仍不能声称提高了召回。
当前继续同一有限比较符合目标，没有偏离，也没有陷入工程细节；无须纠偏或停止正在正常推进的
计算。全部预测完成后才统一评价，保留小增益及负结果，完成图件、回放和科学验收后关闭S1。
本次没有读留出/审计/锁定测试，没有改P1或science_first的Stage4草稿。心跳继续开启。

为避免S1结束后再次停在资料查找，本次另做了只读S2入口定位（不代表已启动S2）：简化断层、属性与
危险性已在`src/seismoflux/data/geology.py`及`pipeline.py`入库，可复用`fault_segment`；源身份见
`data/manifests/source_inventory.csv`。GSRM/GNSS仍为计划候选，尚无已入库产品。S2首个既定比较是
同一目录参考加/不加断层几何，随后分开检验危险性与应变；地质快照的历史使用局限需明确标注，
不把现代快照冒充当时已知输入，也不让外部资料获取阻塞已有数据的有限比较。

本记录已本地落盘，尚未构成新科学阶段验收；运行产物仍仅保留本机，不将逐事件数组公开推送。

## 8. 全部历史预测完成，统一评分启动：2026-09-02 21:00（北京时间）

最后一折在20:54:26完成。根代理随后执行`verify_prediction_manifest`，四折均99个起报—时限对，
合计396，全部五时限及数组身份/形状/归一性检查通过；此后才允许读取C0开发目标。原预测PID33996
已经退出，只读进程核验未发现其他Python实例，未重新训练。20:50最后一次预测CPU约2.10%、内存
520.8 MiB，stderr为0字节。

21:00:14启动唯一统一评分进程，初始PID15244，BelowNormal、数值库单线程：

```text
脚本：scripts/run_multitask_s1_c2b.py --phase score
工作树：D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata
data-root：D:\AIPred\SeismoFlux\data
stdout：data/interim/c2b_logs/score_20260902T210014.stdout.log
stderr：data/interim/c2b_logs/score_20260902T210014.stderr.log
评分完成标志：outputs/multitask_s1/s1c2b_catalog_models_v1/score_phase/score_manifest.json
汇总：同目录summary.json（须等完整manifest及全部产物核验）
```

评分运行时不得启动第二个评分实例；若中断，先核对进程、stderr及不完整产物，再决定同一输入下
的恢复，不能回写预测。新图件仅由统一评分结果生成，主指标为30天/Ms5–6/60万km²/严格0km的
独立震区召回，同时保留全部时限、震级、面积及负结果。图件与离线页正在复用旧渲染器做薄适配，
不改变模型、指标或评分实现。当前仍无最终C2B分数，不能把预测保存完成等同于模型有效。

本次评分检查（21:09）：PID15244累计CPU时间仍持续增加，约单个逻辑核心，21:05两秒采样为
整机2.12%；21:09常驻内存约1776.6 MiB，无stderr错误。评分器在全组计算结束后才写结果，
目前没有逐折评分检查点或可靠百分比；日志暂时为空不代表停止。不得为增加进度条重启或改代码。
基础核最终413/413，预测检查点20/20，两者完成率100%，但不等于项目或科学效果已完成。

全部20份真实训练元数据已只读复审，结论与边界已写入
`docs/s1c2b_scientific_results_2026-09-02.md`：每次最终拟合167—1516个Ms≥4目标、12—131个起报期，
所有拟合正常，内层存在改进但不可替代较晚评价。该报告的开发结果节仍明确留待统一评分，不填推测。
薄渲染器`scripts/render_multitask_s1_c2b.py`初稿已落盘，仍由独立worker完成合成测试；真实渲染未运行，
不能把脚本存在当静态图或离线页已经验收。后续先查当前文件和测试状态，不覆盖未提交工作。

心跳已更新为“评分中”的安全恢复流程，仍每30分钟。科学复审：本次完成的是整组预测与训练证据，
对最终目标属必要使能；尚无新增召回证据，没有偏离方案或陷入工程优化。无需改参数，下一步是
按冻结比较读出真实增量，生成最小有用图件后关闭S1转S2。

## 9. 最终结果与科学验收：2026-09-02 21:45（北京时间）

本节取代上文各历史快照的“评分中/尚无新成绩”。20:54预测完成，21:19:13统一评分完成；四折
各99个起报—时限对，共396个，五时限20份检查点和413份基础核全部完成。评分覆盖1400条曲线
与918个预定配对，六份评分产物的哈希均与manifest一致，未重训或回写任何历史预测。

### 9.1 科学结论与下一步

- 主任务30天、Ms5–6、60万km²、严格0km：原最强L3为54/147，多尺度58/147、年龄57/147。
  多尺度新增14、丢失10，四折净增+1/0/+3/0；年龄新增6、丢失3，四折+1/+1/0/+1。
  召回差分别+2.72与+2.04个百分点，95%区间跨零；保留小增益，不宣称已独立确认。
- 多尺度180天Ms5–6为88/237，对原L3的70/237有增益；Ms≥6全年却为10/45，对照11/45。
  年龄模型在30/45万km²较小预算有互补价值。ridge主任务49/50低于原L3，但大震历史特征
  对较长时限有任务专用贡献，全部负结果保留，不能宣布一个模型包办全部任务。
- 主平均实际面积为L3 599,698、多尺度599,626、年龄599,708km²。均在同一预算以内；
  完整格排序的微小面积差如实报告，不改变评价目标，不通过删除困难地区制造增益。
- 独立只读科学复审及终稿事实复核通过。属于**直接开发期地点预测提升**，不是仅完成工程。
  没有偏离科学目标；现在应停止本轮目录搜索，转向尚未使用的新增数据，而非追求更漂亮旧分数。
- S2采用多尺度为主要地点背景，年龄为小面积互补，原L3为固定历史对照；D1 K75和四特征ridge
  仅保留大震/长时限参考，不做所有模型与所有特征的全交叉。先比较加/不加纯断层几何，
  再单列危险性和应变；方法先查经典文献，现代静态快照用于历史回放的局限必须明示。

### 9.2 图件、交互与必要验收

本机最终目录：`outputs/multitask_s1/s1c2b_catalog_models_v1/rendered_v2`。

- `01_main_anchor_recall.png/.svg`：同一主任务全部模型成绩；
- `02_grouped_area_curves.png/.svg`：报警面积变化下的得失；
- `03_paired_contributions.png/.svg`：时限/震级分开看各数据与模型贡献；
- `04_selected_gain_and_failure_local_only.png/.svg`：同一候选成功与失败地图，仅本机；
- `seismoflux_s1c2b_replay.html`：离线历史开发回放，仅本机，含全部模型/时限/震级/预算与空期。

主代理已目视检查四张静态图；第二版仅避免成功与失败展示期重复，并分别突出新增和丢失，
不改变任何分数。九个展示产物哈希核验通过。浏览器像素级自动验收受既有策略限制，未绕过；
采用静态图、保存数据和Node离线交互检查，全部1400种模型/时限/震级/预算组合可操作。
53项已有协议/模型/预测/评分聚焦测试再次通过；9项最终渲染测试通过（18.55秒），Ruff通过。
这些验证只证明结果读取和展示没有发现错误，预测效果证据是上面的真实历史命中差异。

### 9.3 结果身份、资源和中断恢复

- 预测manifest：`110c795bacae7aa4e00fe72eb3195fee947c1c913d8752da721d9e122db2a01c`；
- 评分manifest：`ae96f95f2e7fbd92c8f6f692b0f4f95f7dd48787d5bcfc2cdf53e0138dde8dc5`；
- 聚合summary：`29445b03aaa7aac01878711c0fb97561fe6881cbe28084d72de44d6c6b2331e3`。

21:44:56只读核验没有SeismoFlux相关Python实例；两CPU负载8%/11%，整机约9.5%，没有占满核心。
原预测PID33996及评分PID15244均已退出，不按旧PID重启。根库、冻结P1、science_first的Stage4
草稿未被本次修改；未打开2020—2022留出、2023+审计或锁定测试，未获取未来目录。

科学验收：`PASS`。本次结果提交和远端闭合：待完成，恢复时先查实际git状态。只提交渲染代码、
测试、报告/交接/蓝图、聚合summary/score_manifest及六张公开聚合PNG/SVG；含逐事件数据的
NPZ/parquet、案例地图、HTML、render_manifest及合成QA目录保留本机。禁止整体添加输出目录。
阶段推送完成后才进入S2。心跳保持每30分钟，以最新结果和S2为下一工作，不能仍按旧评分中提示恢复。
