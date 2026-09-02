# S1-C2B有限位置比较：实现验收（2026-09-02）

## 科学结论与许可范围

**PASS：可以启动冻结协议内的四折历史训练和位置预测。当前没有C2B新预测成绩。**
本轮只将已验收的九模型科学比较变成可运行路径，不改变数据、候选模型、评价目标或面积。
实现提交推送后运行；四折五时限全部保存完毕才统一读取C0开发目标评分。

协议提交`b35d8a760fee1443211e619e2c7d96a97892b899`已验收、推送并远端确认。
精确方法仍为`configs/multitask_s1_c2b_catalog_models.yaml`及对应方法文档，SHA未改。

## 已落实的科学流程

1. 三目录训练面板按规范事件及任一可见来源成员选择，事件和来源均早于起报前24小时；保留Ms
   原数值，不重新去重，不用Mc掩膜或95%/80%全国门。
2. 多尺度核、相对年龄核、三/四特征ridge只利用每次起报已经可见的历史。新参数由外折之前的
   内层选择；ridge两条前推训练链分别拟合标准化，训练标签结束和可见性均隔离到验证起点前30天。
3. 存储有限log空间质量，避免窄核远场浮点下溢影响似然；无新概率地板。空近期回同一长期模型，
   空训练目标或预定数值不合格回K75，保留外期。
4. 日历沿用四开发折和五时限396个起报—时限对。每折每时限保存独立预测检查点，随后保存全折
   完成记录；中断恢复复用同身份已完成记录。未封口但已写好的数组只允许与同输入确定性重算
   完全一致后原样封口，不覆盖或挑选后来更好看的版本。
5. 评分先核验全部预测，再复用C0五参考模型和固定开发目标；预计792个起报—时限—震级键，
   空期保留。按时限和震级分别列严格召回、辅助容差、四种事件/震序视角、面积曲线与配对。

辅助70 km的具体实现补充：在既有等积投影下，事件到已选裁切格多边形集合的距离≤70 km。
这是投影距离敏感性，不额外圈入计费面积、不替代严格0 km主指标；基准和新模型同口径。
对原C0质量排序保持原算法，新模型使用log密度排序，避免先取指数损失窄核尾部信息。

## 验证证据

独立只读科学/接口审查通过：逐项核对因果输入、内层标签、两次前推标准化、固定参数外推、
五时限数量和四折完成后才能评分。没有发现须重开95%、Ms、C2A或扩大候选的理由。

本轮聚焦验证：**53项合成/配置测试通过，3.98秒；所有新增位置模块、脚本与测试的Ruff通过。**
包括手算核与面积归一、解析梯度、可学合成信号、极小尾部、空期、微秒边界、迟到标签、
内层前推隔离、五时限日期、预测文件保存/回读后进入位置评分，以及全四折不完整禁止读目标。
这不等于实际预测性能已经通过。

真实文件只读预检也通过：

```text
2020年前读取的目录记录：34898
全国网格：15697
主起报—时限对：396
2000年起报前D0/D1/D2训练数量：2895 / 1494 / 1184
实际模型拟合次数：0
新外折目标或分数读取：false
```

模块身份（本机精确字节）：

| 模块 | SHA-256 |
| --- | --- |
| c2b_predict.py | b01dd4ea731780bf0494396573d8c4c4fd5ba6d5edff3eaa86077aa57c977392 |
| c2b_inputs.py | d433b4bcbfe6198ef2ca30b99590f61265d9a95d08fb8570a50a2437ce87076d |
| c2b_models.py | da3e0cbcdf4f6d8df0f7d3003e5da520d317fb690b50d40184c49d51e527b7e8 |
| c2b_score.py | 6843b5fdbe63ced5319ebb496b70f5d637bda05fdc26959df714084b900439ce |

## 启动和完成后的决定

运行入口为`scripts/run_multitask_s1_c2b.py`，明确`--phase predict`与`--phase score`，不能混在
同一阶段中读取新目标。真实启动必须用隐藏后台进程、BelowNormal，默认2折线程、数值库单线程，
至少保留2物理核心。先查进程，再取得同一输出目录的OS进程锁，避免重复实例。

真实产物根：`outputs/multitask_s1/s1c2b_catalog_models_v1`。每期因果空间核为可复用缓存；
每折各时限的`horizon_manifest.json`是预测检查点，全折`prediction_manifest.json`出现并校验
成功才允许评分。日志中的空间核完成数是训练准备进度，不得冒充已完成外折预测比例。

结果出来后立即回答：相对原模型新增/丢失多少独立震区，在哪些时限/震级/面积有用；保留局部
小增益及负结果。有限比较结束后关闭S1进入S2，不能继续增加目录候选或改95%标准。

```yaml
science_value_category: necessary_enabler
evidence: finite_location_only_implementation_causal_synthetic_checks_and_real_input_preflight
decision: run_registered_four_fold_C2B_predictions_then_paired_scoring
next_scientific_test: measured_independent_region_recall_gain_by_panel_and_model
stop_condition: complete_finite_C2B_then_move_to_S2_multidata
implementation_acceptance: PASS
C2B_real_predictions_created_at_acceptance: false
C2B_scores_created_at_acceptance: false
git_closure: complete_remote_verified
implementation_commit: fcce264465627cf5cdeb2b361618de0b33a6fe4c
holdout_opened: false
locked_test_run: false
frozen_P1_modified: false
science_first_Stage4_drafts_touched: false
```

验收之后的运行状态：实现提交推送及远端确认完成后，于北京时间19:40:28启动PID33996的唯一
后台预测进程，2折线程、BelowNormal。最新检查点、资源及恢复入口见最新交接；此处的
“验收时尚未预测”是历史状态，不代表后台未启动。
