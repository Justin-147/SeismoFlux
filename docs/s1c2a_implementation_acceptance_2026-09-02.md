# S1-C2A最小位置对照实现验收

结论：通过。本提交推送后可运行真实历史预测；尚未产生新的预测效果证据。

协议依据：`3c2948d125a532672df65c3da543113ac7c79bb8`，配置SHA-256为
`4aa9070c190f0e599870bde93899bcb2153f34e5069bf923a2499bce8e9c64bd`。

## 对照没有变形

- 仍是C0相同四折、116个30天曝光、原25km全国格网与五档面积。
- A/B只改变训练中心；各处理内部L1/L2/L3共用输入，参数固定为C0更早内层所选。
- C1外折起点mask复用，没有重估Mc，没有80%或95%全国门。
- 因果窗口保持T前24小时；近期只取`(T−30天,T−24小时]`。
- 不变化的输入复用原预测；变化的输入调用原L1/KDE/L3公式，不改其平滑、收缩或混合形式。
- 全国质量归一化和原完整格面积排序不变，实际面积逐模型记录。
- 新预测四折全部完成后才允许读取原C0相同目标。评分核对116期（含空期）和147个唯一首震锚点。
- 正、零、负结果都保留；配对区间按锚点重采样，不把格网或同震序成员数冒充独立样本。

## 完成的检查

1. 真实文件只读预检通过：四个C1外折mask与全国空间分区一致；C0每折29期、三模型、15,697格
   预测和固定参数可直接读回。预检未生成新预测，未读取目标或成绩。
2. 25项聚焦合成验证通过，包括因果时间、布尔mask、边界归属、原公式一致、输入未变时逐位复用、
   空近期退回长期背景、按折恢复、同目标、空期、配对方向、单次只运行指定阶段。
3. 新增执行代码与测试的Ruff检查通过；`git diff --check`通过。
4. 根代理复审预测路径，独立审计复审评分路径，未发现影响本项科学结论的阻断。

预测模块最终SHA-256：`c57f57ddeaff249ce88f8f29f6f743929cd3395540c1ec1a07aeb408a5b9334a`。
未修改旧C0/C1/P1实现、协议和产物，未接触holdout/audit/locked test或Stage4旧草稿。

## 科学价值与下一步

这是直接数据消融的必要实现，不是模型已变好。现在停止扩展实现，运行既定六曲线、四折预测，再
统一评分；若结果没有改善，也会完整报告，不另调百分比或Mc格尺度。图件和离线页只负责展示真实
结果，不作为新的科学门。

运行入口：`scripts/run_multitask_s1_c2a.py`，明确`--phase predict`或`--phase score`。默认2个折
工作线程、数值库单线程；每5期输出进度，每折完成保存检查点。不启动重复实例。

```yaml
acceptance: PASS
science_value_category: necessary_implementation_for_fixed_input_ablation
new_prediction_effect_at_acceptance: none
next_after_commit_push: run_C2A_prediction_then_score_after_all_four_fold_checkpoints
holdout_audit_locked_test_authorized: false
```
