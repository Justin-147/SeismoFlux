# ETAS 数值资格检验

总状态：**not_evaluable**。这只是拟合稳定性检验，不是预测命中率。

| 快照 | 收敛起点 | Hessian | 分支比 | 三网格 | 结论 |
|---|---:|---|---|---|---|
| fold_1 | 0/5 | 未通过 | 未通过 | 未通过 | not_evaluable |
| fold_2 | 0/5 | 未通过 | 未通过 | 未通过 | not_evaluable |
| fold_3 | 0/5 | 未通过 | 未通过 | 未通过 | not_evaluable |
| fold_4 | 0/5 | 未通过 | 未通过 | 未通过 | not_evaluable |
| final_validation | 0/5 | 未通过 | 未通过 | 未通过 | not_evaluable |

五个起点检查是否找到同一稳定解；Hessian 检查解是否清楚；分支比防止模型失稳；三网格检查结论是否依赖网格粗细。局部高 Mc 只影响对应固定单元。

本报告不含事件编号、坐标、异常特征、评分或锁定测试信息。
