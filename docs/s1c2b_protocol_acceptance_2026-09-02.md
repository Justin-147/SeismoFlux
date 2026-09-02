# S1-C2B训练面板与有限协议验收（2026-09-02）

## 结论

**PASS：可进入有限位置模型实现。尚未产生C2B预测、成绩或新的效果提升。**
该结论只验收研究问题、资料边界与可执行比较，不代替模型结果验收；提交推送后才进入实现。

## 科学检查

- 三个面板的24个历史训练节点均已核对，保持规范事件去重身份；来源使用任一当时可见成员。
  原始数值按用户说明解释为Ms，未重写原始文件；不把说明扩展到公开外部目录。
- 九个位置模型、四个多尺度候选、十个相对年龄组合、两种ridge三档lambda均明确，没有无界搜索。
- 固定模型比较训练面板，固定主面板比较模型；D2特征使用加入/不加入的成对模型，归因仅限该
  大震历史空间特征整体，D0原已含1970后的大震。
- 内层按时间前推，标签截止与30天隔离均明确；ridge每次标准化仅用合法训练数据，未用较晚
  内层或C0外折最终选择参数给较早验证提供信息。
- 全国域、25 km格网、五时限、两震级档、起报日期和目标不因模型改变；面积按相同上限比较，
  实际离散面积差须报告。空期保留，不用目标稀疏或未知Mc一票否决全国。
- 指数核仅是相对年龄加权的位置分布；不宣称时间发震率衰减、绝对发震概率或已经改进次数/震级。
- 国内经典目录覆盖、国际核平滑/点过程及混合预测文献已审阅，实际采用与简化差异均写明。

独立审查者完整只读复核精确YAML和方法文档，科学结论为PASS；其关于D2归因的非阻塞补充已采纳。
不新增强显著性门，不为理论简化继续扩大模型族；C2B有限结果形成后关闭S1、进入S2。

## 已执行的必要验证

北京时间19:11附近，在数值库单线程环境执行：

```text
python -m pytest tests/unit/test_audit_multitask_s1_c2b_panels.py tests/unit/test_multitask_s1_c2b_protocol.py -q
12 passed in 1.47s
python -m ruff check scripts/audit_multitask_s1_c2b_panels.py tests/unit/test_audit_multitask_s1_c2b_panels.py tests/unit/test_multitask_s1_c2b_protocol.py
All checks passed!
git diff --check
无错误
```

验证覆盖源成员、Ms数值边界、起报前24小时含微秒边界、三个年代面板、24行聚合账本及其身份，
并仅根据日期重建396个起报—时限对。没有读取新的外折目标、留出或审计期内容，没有运行模型。

- 协议SHA-256：`4f497690643a64466cbd9eec358977f1c8c1d4655bc4cc9ca5a65dbc95859243`
- 账本SHA-256：`b08de8b0a8d2b84d581a5f0b5c4b2bb37d9d7e3f58a73f4e4e99373266ee1ed6`
- 19:12进程检查：没有Python/pythonw进程；两CPU共24物理/48逻辑核心，采样负载31%和5%
  （这是全机其它活动，约18%整机；不是SeismoFlux训练消耗）。没有新后台计算。
- 本轮只编辑本工作树列明的新协议、账本、测试和交接入口；未写science_first既有Stage4草稿、
  冻结P1或根库历史工作。

## 科学价值与下一个安全点

```yaml
science_value_category: necessary_enabler
evidence: source_aware_training_panels_and_finite_causal_comparison_protocol
decision: accept_protocol_then_implement_small_location_only_models
next_scientific_test: same_area_independent_region_recall_by_model_panel_horizon_magnitude
stop_condition: no_more_protocol_expansion_close_S1_after_finite_C2B_results
C2B_prediction_gain_demonstrated: false
protocol_acceptance: PASS
publication_scope: aggregate_ledger_code_and_documents_only
git_closure: complete_remote_verified
protocol_commit: b35d8a760fee1443211e619e2c7d96a97892b899
```

注意：测试通过只说明可以开展这项科学比较，不能当成预测改善证据。
