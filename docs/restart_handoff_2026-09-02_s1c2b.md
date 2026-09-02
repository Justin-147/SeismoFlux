# 最新续接：S1-C2B有限目录模型比较（2026-09-02）

> 当前最高恢复入口，接替`restart_handoff_2026-09-02_s1c2_resumed.md`；旧结果保留。
> C2B协议和实现均已验收推送，19:40启动真实历史训练/预测。尚无C2B新成绩，不能声称效果提高。

## 1. 先用通俗话说明

项目没全部完成。我们已有一个“历史地震热点＋近期活动”的目录参考模型：在2000—2019开发
历史、30天、约60万km²报警面积下，命中147个独立Ms5–6震序中的54个。简单地区频率模型为32个，
长期平滑模型为46个。它们不是最终独立测试，也不能把这147个当全部目录的可用样本。

95%门和震级类型的审计已完成。两份用户目录均按Ms解释，旧数值与分数不用重跑；95%没有改坏
C0分数，但曾把C1挡在预测之前。补充C2A已经实际比较：原模型54个、仅删确定高Mc地区52个、再
删未知地区53个，没有稳定提高。这个分支已结束，不再反复调整全国比例或目录类型。

已经启动的新比较将问：热点范围画得大些还是小些、近期地震的先后顺序，以及更早的大震历史，
有没有帮助？具体是三种训练资料、九个有限位置模型，覆盖五种预测时限和两种震级档。模型训练
与较晚历史评价分开，数据变化与模型变化分别比较。19:42已完成21/413个不同历史起报时点的因果
基础核计算（约5.1%），尚未完成外折预测检查点；**还没有C2B新成绩或提升结论**。

## 2. 工作位置、必读文件与不能动的边界

- 工作树：`D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata`
- 分支：`codex/p2r-multitask-multidata`
- Python：`D:\AIPred\SeismoFlux\.venv\Scripts\python.exe`
- 根库、science_first既有Stage4未跟踪草稿、冻结P1：不修改。
- 不读取2020—2022留出和2023+审计目标，不运行锁定测试，不联网取未来目录。
- 历史C0/C1/C2A预测、评分和冻结配置不回写；禁止`git add -A`混入其它工作。

继续前读：

1. 总蓝图`SEISMOFLUX_IMPLEMENTATION_HANDOFF.md`第1.9—1.12节；
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

协议和专用位置路径均已验收推送，当前安全动作是检查唯一后台进程、因果缓存与各时限预测检查点，
不要重复启动。若进程已退出且全折未完成，先查stderr与已有输出，在相同代码/协议/输入下恢复
同一个输出目录；不运行C0完整runner，也不改变模型或目标。全四折五时限完成后才进入评分。

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
| S1 | 先把目录模型做成可信参考，比较目录处理与有限新模型 | C0/C2A完成；C1无预测停止；C2B真实训练/预测运行中 |
| S2 | 加断层、危险性、应变，看是否改善中长期位置 | 未开始，C2B后立即转入 |
| S3 | 加205期动态异常，检验其超出目录的贡献 | 未开始 |
| S4 | 有限树模型和一个小型神经/多任务模型 | 未开始 |
| S5 | 组合互补方法，用消融区分数据和特征贡献 | 未开始 |
| S6 | 稳健性、直观图件、交互、论文、PPT、可编辑poster | 未完成，早期展示不是最终论文证据 |
| S7 | 保存不可回填的未来预测 | 并行，不阻塞历史研究 |

距离最终目标还差的是多数据真正能提高多少、不同预测时限/震级是否互补，以及最终独立检验，
不是等某一个未来日期。不能保证这些尝试一定提高，但可以通过有限对照尽快知道哪些有用。

```yaml
science_value_category: necessary_enabler
evidence: three_causal_panel_counts_and_nine_finite_models_defined_no_new_C2B_skill
decision: continue_running_registered_C2B_predictions_then_measure_paired_skill
next_scientific_test: same_targets_same_alarm_budget_independent_region_recall
stop_condition: finite_C2B_comparison_done_then_close_S1_and_start_S2
off_target: false
engineering_loop_detected: false
plan_adjustment: clarify_age_weighting_and_M5_feature_attribution_no_new_model_family
project_completed: false
```

## 5. 验收、运行与恢复记录

当前状态：`S1_C2B_PREDICTION_RUNNING`。12项配置/账本聚焦测试、Ruff与
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
C2B_scores_created: false
C2B_actual_predictive_improvement: unknown
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
避免重复计算。现有完整时限预测不重跑；全折完成前不得调用`--phase score`。
