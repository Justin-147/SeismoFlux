# SeismoFlux 重启续接交接：科学路线复审（2026-07-31）

> 本文件保留S0历史。D1-0及后续恢复请以
> `docs/restart_handoff_2026-08-03_d1_0.md`为当前入口。

## 恢复时先看这一段

权威工作树：

`D:\AIPred\SeismoFlux\data\interim\worktrees\science_first`

分支：

`codex/stage2-etas-science-first`

S0 科学复审提交：

`71c5ab2`（已推送至 `origin/codex/stage2-etas-science-first`）

恢复命令：

```powershell
$wt = 'D:\AIPred\SeismoFlux\data\interim\worktrees\science_first'
git -C $wt status --short --branch
git -C $wt log -5 --oneline --decorate
```

先确认当前分支和远端一致，再阅读：

1. `SEISMOFLUX_IMPLEMENTATION_HANDOFF.md` 的 `1.1 2026-07-31 科学路线重设`；
2. `docs/ALL_DATA_SCIENTIFIC_INVENTORY_2026-07-31.md`；
3. `docs/SCIENTIFIC_REASSESSMENT_2026-07-31.md`；
4. `docs/scientific_reassessment_overview.svg` 和
   `docs/scientific_reassessment_explorer.html`；
5. 本交接。

若 Codex 的 Git 写权限已经恢复，先只暂存以下八个文件，不得使用 `git add .`：

```powershell
git add -- AGENTS.md SEISMOFLUX_IMPLEMENTATION_HANDOFF.md `
  docs/ALL_DATA_SCIENTIFIC_INVENTORY_2026-07-31.md `
  docs/SCIENTIFIC_REASSESSMENT_2026-07-31.md `
  docs/restart_handoff_2026-07-31_scientific_reassessment.md `
  docs/scientific_reassessment_acceptance_2026-07-31.md `
  docs/scientific_reassessment_overview.svg `
  docs/scientific_reassessment_explorer.html
git diff --cached --check
git diff --cached --name-status
```

确认暂存清单恰为上述八项后，提交并推送；15 个未跟踪 Stage 4 草稿必须保持未暂存。

## 当前阶段

- 完整工作线：S0 全数据和路线复审 → D1 真实历史开发回放 → D2 贡献拆分 →
  P1 真正前瞻确认（与 D1 并行准备）→ F1 构造慢窗口 → 锁定确认 → 持续预测展示。
- 当前阶段：`D1-0_executable_scientific_contract`
- S0 状态：`closed`；数据盘点、路线复审、验收、提交和推送均已完成。
- 当前状态：开始冻结不含模型成绩的 D1 样本水位、特征、模型公式和统计规则。
- 恢复判据：远端分支应至少包含 `71c5ab2`；不得重复提交 S0，也不得跳过 D1-0 直接打开成绩。
- 锁定测试：未运行。
- 新真实模型训练：尚未启动。
- 新真实预测成绩：尚未产生。

## 为什么重设路线

当前唯一真实正证据是 75 km KDE 相对均匀背景的信息增益；异常、近期地震、构造和组合均还没有
真实效果。过去把开发筛查也做成一次性最终考试，工程故障多次发生在预测和评分前，却消耗了大量
时间。

新路线把“历史开发筛选”和“真正前瞻确认”分开。历史开发允许小而稳定的正提升继续，不再要求
全部窗口和诊断同时过硬门；最终结论仍以同面积、无未来信息的前瞻结果为准。

## 全部数据位置

原始只读来源：

`D:\AIPred\LocationPred`

来源清单：

`D:\AIPred\SeismoFlux\data\manifests\source_inventory.csv`

当前 Stage 1 标准化数据：

`D:\AIPred\SeismoFlux\data\processed\stage1\debc98054172a4a1`

当前 Stage 3 异常特征包：

`D:\AIPred\SeismoFlux\data\processed\stage3\anomaly_history\anomaly-feature-bundle-de7547faa9f87541`

真实行级数据没有复制进 science-first 工作树；运行 D1 时必须把路径解析到仓库根目录的
`data\processed`，或使用只读目录连接。不得复制 4.077 GiB 特征表进 Git 工作树。

## 已完成的科学盘点

- 原始输入：7 类、216 文件、43,610,994 字节；216/216 SHA-256 一致；
- 地震：43,785 来源行，去重后 40,898 个事件，29,637 个在研究区；
- 异常：205 期、59,904 条、166,189 状态行、3,217,885 特征行；
- 构造：519 段、7,339 点、7,216 条真实迹线；
- 当前唯一真实正模型：75 km KDE；
- KDE 最终验证：96 个支持域 M≥4 目标，IG `+0.40215`，逐事件 Bootstrap 条件 95% CI
  `[+0.23564,+0.56286]`；这些不是 96 个 M5–6 独立震群，同期只有 8 个 M5–6 和 1 个 M6+；
- 尚无固定 600,000 平方公里召回成绩；
- ETAS 为 `not_evaluable`；
- Stage 2S 和 Stage 4A 没有形成预测成绩；
- Stage 2P-1B 只使用合成数据；
- 真正前瞻 issue 和锁定测试均为 0。

## 下一阶段 D1 的精确科学问题

> 在三个真实历史时间外推折、相同 600,000 平方公里报警面积下，最近 30 天地震活动、报告活动
> 控制、单期异常和动态异常能否单独或组合提高未来 30 天独立 M5–6 震群严格召回？

比较六个模型：

1. `B0`：75 km 长期 KDE；
2. `B0+R30`；
3. `B0+C`；
4. `B0+C+A_snapshot`；
5. `B0+C+A_dynamic`；
6. `B0+R30+C+A_dynamic`。

主评估起报区间（`Asia/Shanghai`）：

- `[2023-07-01, 2024-04-01)`，90 天目标成熟到 2024-06-30；
- `[2024-07-01, 2025-04-01)`，90 天目标成熟到 2025-06-30；
- `[2025-07-01, 2026-04-01)`，90 天目标成熟到 2026-06-30。

训练最后起报不晚于评估起点减 90 天。首轮不能用几十个原始特征直接拟合稀少 M5–6；
应把 200 km 特征压缩为最多 10 个预定义组分数，用未来 M4+ 网格计数训练
`log(base_density×30天)` 偏置的 Poisson ridge，再只在 M5–6 独立震群上评价。
不加交互、树或神经网络。

一级指标是 30 天、600,000 平方公里独立震群召回；同召回所需面积为并列重要结果。

## 下一步不是看成绩，而是完成 D1-0 合同

D1-0 必须在任何模型效果打开前冻结：

- 精确起报日历、端点、成熟日和跨折唯一目标/震群归属；
- 每折拟合 M4+、评价 M5–6 事件和 30 天/75 km 震群水位；
- `B0` 每期因果重建、R30 的凸混合公式和 `α={0,0.25,0.5,0.75}`；
- 最多 10 个异常组分数的精确源列和公式；
- Poisson exposure/offset、`ridge alpha={0.1,1,10}` 和训练内选择；
- 2,000 次配对震群 Bootstrap、根种子 147、各 200 次时间/空间置乱；
- 300/450/600/750/960 千平方公里面积档和 `row,column,cell_id` 并列规则；
- 新 P1 只可并行准备数据源与发行；真实 issue 必须等简化协议和模型完成验收、提交、推送。

D1-0 完成验收、提交和推送前，不得查看六个模型的命中、召回、IG 或效果地图。

## D1-0 开始前的最小检查

1. 核对工作树、分支、最新提交和远端；
2. 确认 15 个未跟踪 Stage 4 草稿仍不在 D1 提交范围；
3. 确认根目录真实 Parquet 只读可访问；
4. 只读取日期、震级档样本水位、模式、行数和目标无关特征列，冻结完整可执行合同；
5. 检查 CPU 使用率；默认最多 4 个 worker 起步，确认系统空闲后才提高到 8；
6. 设置 BLAS/OpenMP 内层线程为 1，始终至少保留 2 个物理核心；
7. GPU 只在与 CPU 数学结果一致时加速，不因使用 GPU 改模型；
8. 每完成一个时间折就保存检查点、状态和图件草稿。

审计时机器有 24 个物理核心、48 个逻辑线程，且系统 CPU 一度约 52%–57%。因此不能直接启动
占满 CPU 的任务；每次重启后必须重新现场检查，不复用这个历史百分比。

## 工作树隔离警告

以下 15 个未跟踪 Stage 4 草稿不是本次 S0 文档工作的一部分，不得编辑、暂存、删除或还原：

- `src/seismoflux/anomaly_increment/kde_dev_background.py`
- `src/seismoflux/anomaly_increment/kde_dev_calendar.py`
- `src/seismoflux/anomaly_increment/kde_dev_fit.py`
- `src/seismoflux/anomaly_increment/kde_dev_inputs.py`
- `src/seismoflux/anomaly_increment/kde_dev_placebo_mapping.py`
- `src/seismoflux/anomaly_increment/kde_dev_seal.py`
- `src/seismoflux/anomaly_increment/kde_dev_statistics.py`
- `tests/unit/test_stage4_kde_dev_background.py`
- `tests/unit/test_stage4_kde_dev_calendar.py`
- `tests/unit/test_stage4_kde_dev_fit.py`
- `tests/unit/test_stage4_kde_dev_inputs.py`
- `tests/unit/test_stage4_kde_dev_placebo_mapping.py`
- `tests/unit/test_stage4_kde_dev_seal.py`
- `tests/unit/test_stage4_kde_dev_statistics.py`
- `tests/unit/test_stage4_kde_dev_synthetic_chain.py`

仓库根目录 `D:\AIPred\SeismoFlux` 绑定另一工作树和分支，并有不相关 ETAS 改动。只把它当真实
处理数据存放点，不在根工作树编辑或提交。

## 中断恢复规则

- 不根据旧 PID 判断任务是否仍在运行；
- 先检查现场进程、CPU、GPU、最后更新时间和阶段状态；
- 已完成折的输出必须只读保留；
- 若运行在评分前因实现问题停止，记 `invalid_run`，修复最小问题后可在开发模式重跑；
- 若已经打开正式锁定结果，则不得重跑或针对结果调参；
- 每个阶段结束后立即更新本交接或创建下一份日期明确的权威交接。

## 本轮科学价值

- `science_value_category`: `necessary_enabler`
- `evidence`: 全部真实输入已核对；真实证据边界已澄清；D1 的数据、模型、时间折和主指标已确定
- `decision`: `finish_s0_acceptance_commit_push_then_freeze_d1_executable_contract`
- `next_scientific_test`: D1-0 冻结精确样本水位、公式和统计后，运行六个模型的 30 天固定面积独立震群召回
- `stop_condition`: 若所有增量平均方向不正或与置乱无区别，停止对应组件复杂化并保留 75 km KDE
