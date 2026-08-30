# SeismoFlux D1 最终科学归因与阶段验收（2026-08-28）

## 验收结论

D1 的冻结真实历史开发回放、区域与逐震群稳健性、时间/空间置乱、最终静态图和离线交互回放均已完成，本地阶段验收通过。状态为 `accepted_for_commit_and_push`：本文件及下列精确范围成功提交、推送到 `origin/codex/stage2-etas-science-first` 并完成远端回读后，D1 阶段闭合。整个 SeismoFlux 项目尚未完成。

锁定测试没有读取或运行；D1 全部结果仍标为 `retrospective_development_only`，相对条件强度和顺位不称为绝对发震概率。

## 使用的数据和方法

- 冻结地震目录 40,898 条；起报日前 M4+ 地震用于 75 km `B0`、近 30 天 `R30` 和训练目标，未来窗口 M5–6 地震只在报警冻结后评分。
- 205 个真实异常报告期、3,217,885 行因果异常特征、15,697 个固定约 25 km 网格，以及 39 个目标无关构造区。
- 30 天一级终点包含 24 个起报日和 21 个独立 M5–6 震群；90 天次级终点包含 9 个起报日和 22 个独立震群。
- 比较六个简单模型：`B0`、`B0_R30`、`B0_C`、`B0_C_A_snapshot`、`B0_C_A_dynamic`、`B0_R30_C_A_dynamic`。各折只用评估期之前的数据预处理、选参和重拟合。
- 未使用人工预测地点/震级/时间、真实震中生成候选或区域、未来报告、断层或长期危险性、树模型、神经网络、锁定测试。

## 预测效果

在 30 天、600,000 km²、21 个独立震群上，`B0_R30` 相对 `B0`：

- 命中由 `5/21` 提高到 `9/21`；
- 召回由 23.81% 提高到 42.86%，增加 4 群和 19.05 pp；
- 2,000 次配对震群 Bootstrap 95% 区间为 `[+4.76,+38.10]` pp，`P(增益>0)=0.9905`；
- 三折命中由 `3/8、2/6、0/7` 变为 `5/8、4/6、0/7`，3/3 折不变差；
- 在 300,000 km²已命中 7/21，高于 `B0` 在 600,000 km²的 5/21。

这属于真实历史时间外推中的 `direct_improvement`，使 `B0_R30` 成为当前最佳简单候选；它不是锁定确认或真正前瞻证明。

## 最终异常归因

冻结的时间和空间置乱均已完成：六个 `kind × fold` 各为 200/200，总计 1200/1200。最终文件为 `data/interim/d1/placebo_078e950/d1_placebo_result.json`，SHA-256 为 `f9fd81887863ac8f6ac174e346a015e7e18cb0f3b9a906b1deea0c4013ec3ec9`，绑定模型提交 `078e950a2b4a837f2ebaaed0d62708012c6e6e23`。

- 三个预登记异常对比的观测召回增量全部为 0。
- 时间置乱科学拟合失败 `58/200 = 29.0%`，三折为 `57/1/0`。
- 空间置乱科学拟合失败 `11/200 = 5.5%`，三折为 `10/0/1`。
- 两类置乱均超过冻结的 5%失败上限；三个对比全部为 `evidence_insufficient`、`promising=false`。
- 完整组合与 `B0_R30` 在一级终点都命中 9/21，不能把提升归因于异常。

因此，本次冻结方案没有支持异常带来额外预测能力。停止继续复杂化当前异常组件，但不把证据不足夸大为“异常在所有条件下永远无效”。

## 最终图件与回放

最终不可覆盖版本位于 `outputs/visualizations/d1_078e950_final_attribution/`：

- `d1_effects.svg`：30/90 天召回—面积、Molchan、Bootstrap 和最终异常归因；
- `d1_maps.svg`：三折同面积报警区、成熟目标与命中/漏报案例；
- `d1_explorer.html`：完全离线切换起报日、模型、窗口和面积；
- `d1_science_report.md`、`d1_science_summary.json` 和产物清单。

验收值：`placebo_attribution_complete=true`、`robustness_diagnostics_complete=true`、`final_attribution_ready=true`、`best_intermediate_model=B0_R30`。清单身份为 `cf9dc84fa0018254f0dde53cf7f4847cdde4a2a445c37773fd3afacab94782e3`，清单文件 SHA-256 为 `1d000592961ca5220e292bd6d7024dad367e25ff9f4106b05c0790b88159f523`。两张 SVG 已目视核查；HTML 脚本语法、四个选择控件和无 HTTP/HTTPS、`fetch` 外部依赖检查通过。旧 observed 图件未覆盖。

渲染审计还发现并修复了一个完成态文案问题：置乱已完成时，旧模板仍提示“必须完成置乱”。修复只影响科学状态表达，不改变数据、模型、指标或结果，并增加了回归测试。

## 测试与边界验收

- D1 聚焦单元测试：`73 passed in 11.17s`。
- 完成态渲染回归测试在格式化后复测：`13 passed in 6.36s`。
- 唯一蓝图及相邻科学治理文档回归：`41 passed in 4.40s`。
- Ruff check：通过。
- Ruff format check：通过。
- 严格 Mypy：`Success: no issues found in 21 source files`。
- `git diff --check`：通过。
- 本机没有可调用的 `uv`，因此未把 `uv lock --check`冒充为已执行；本轮使用既有冻结 `.venv` 完成全部聚焦测试和静态检查。
- 没有运行锁定测试，没有修改模型、折、指标、种子、复本数或输入，也没有触碰既有 Stage4 未跟踪草稿。

## 阶段决定

- D1 本地验收：`PASS`。
- 异常机制：`evidence_insufficient`，停止复杂化，不进入 D2 异常组合拆分。
- 当前保留候选：`B0_R30`。
- 下一科学阶段：单独预登记新的 P1 真正前瞻比较，不回填历史 issue。
- 锁定测试：保持未打开。

## 机器可检索的科学价值复审

- `science_value_category`: `direct_improvement`
- `evidence`: `B0_R30` 在 30 天、600,000 km²、21 个独立震群的真实历史三折时间外推中由 5/21 提高到 9/21，增加 4 群和 19.05 pp；Bootstrap 95%区间 `[+4.76,+38.10]` pp，`P(增益>0)=0.9905`，3/3 折不变差。异常三个观测增量均为 0，时间失败 58/200、空间失败 11/200，全部归因为 `evidence_insufficient`、`promising=false`。
- `decision`: `retain_B0_R30_stop_anomaly_complexification_and_close_D1_after_remote_readback`
- `next_scientific_test`: 在新的 `valid_from` 下冻结模型、参数、首期、数据源、30 天窗口、600,000 km²面积和独立震群评价规则，真正前瞻比较 `B0_R30` 与 `B0`。
- `stop_condition`: 不再为当前异常组件增加特征、模型或敏感性；若 `B0_R30` 在预登记最大前瞻样本或最长时间仍无正方向，则停止该挑战者并退回 `B0`。
