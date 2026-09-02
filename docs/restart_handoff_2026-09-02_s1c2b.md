# 最新续接：S1-C2B有限目录模型比较（2026-09-02）

> 当前最高恢复入口，接替`restart_handoff_2026-09-02_s1c2_resumed.md`；旧结果保留。
> 本轮进展是下一项科学比较的训练资料核对和协议，不是新模型已提高预测的证据。

## 1. 先用通俗话说明

项目没全部完成。我们已有一个“历史地震热点＋近期活动”的目录参考模型：在2000—2019开发
历史、30天、约60万km²报警面积下，命中147个独立Ms5–6震序中的54个。简单地区频率模型为32个，
长期平滑模型为46个。它们不是最终独立测试，也不能把这147个当全部目录的可用样本。

95%门和震级类型的审计已完成。两份用户目录均按Ms解释，旧数值与分数不用重跑；95%没有改坏
C0分数，但曾把C1挡在预测之前。补充C2A已经实际比较：原模型54个、仅删确定高Mc地区52个、再
删未知地区53个，没有稳定提高。这个分支已结束，不再反复调整全国比例或目录类型。

正在准备的新比较将问：热点范围画得大些还是小些、近期地震的先后顺序，以及更早的大震历史，
有没有帮助？具体是三种训练资料、九个有限位置模型，覆盖五种预测时限和两种震级档。模型训练
与较晚历史评价分开，数据变化与模型变化分别比较；已有训练账本，**还没有C2B新预测或成绩**。

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

协议阶段验收推送后，实现并验证专用位置模型路径，再执行真实训练/预测。不得直接调用C0完整
runner，因为它还会进入与本轮无关的负二项、震级和联合拟合。应复用既有日历、格网、因果历史和
位置评价，只新增所需空间核/少量组合逻辑。拟建模块名称不是已有产物，恢复时先检查实际文件。

建议工作拆分：

1. 位置数学：稳定log空间高斯核、有限混合、训练期加权标准化及条件ridge目标。
2. 因果面板和训练预测：源成员映射、已有内层/外折日历、各h严格过去选择、四折检查点。
3. 评价：四折完成后才复用C0开发目标；九模型及C0参考，全部h/震级/面积，配对新增与丢失。
4. 结果一出来就解释增量，完成最小有用的静态图与离线回放，不在展示工程上拖延多数据研究。

模型细节以精确YAML为准。指数近期权重只表示相对年龄，不能宣称发震率会随安静期自动减弱。
D2特征对照只归因于整个大震历史空间特征；D0本来已含1970后的大震，不能称此前完全没用大震。
格前缀控制预算，但不同排序的实际面积可相差不足625km²，必须报告；最终最多10区仍待后续处理。

## 4. S0—S7完整工作线与科学复审

| 阶段 | 通俗任务 | 当前进展 |
| --- | --- | --- |
| S0 | 摸清数据、时间/地点/震级任务和可用样本 | 已完成 |
| S1 | 先把目录模型做成可信参考，比较目录处理与有限新模型 | C0/C2A完成；C1无预测停止；C2B协议已落盘，计算待启动 |
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
decision: proceed_to_small_location_only_implementation_and_real_comparison
next_scientific_test: same_targets_same_alarm_budget_independent_region_recall
stop_condition: finite_C2B_comparison_done_then_close_S1_and_start_S2
off_target: false
engineering_loop_detected: false
plan_adjustment: clarify_age_weighting_and_M5_feature_attribution_no_new_model_family
project_completed: false
```

## 5. 验收、运行与恢复记录

当前状态：`S1_C2B_PROTOCOL_ACCEPTED_GIT_CLOSURE_PENDING`。12项配置/账本聚焦测试、Ruff与独立
科学复审已通过，见`docs/s1c2b_protocol_acceptance_2026-09-02.md`。完成提交与远端回读后进入实现。

```yaml
C2A_closed_and_pushed: true
C2B_training_panel_ledger_created: true
C2B_protocol_scientific_review: PASS
C2B_protocol_tests: PASS_12_focused_tests
C2B_protocol_git_closure: pending
C2B_model_implementation_complete: false
C2B_predictions_created: false
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
31%/5%（全机约18%，不是本项目在训练）。此时还没有C2B训练后台进程，不能将协议编写声称为
“模型正在训练”。继续时先查进程和输出，避免重复实例，并及时把完成比例/检查点写入本文。
