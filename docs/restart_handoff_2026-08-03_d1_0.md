# SeismoFlux 重启续接交接：D1-0合同闭合后直接运行D1

## 一句话状态

S0全数据与科学路线复审已在提交`71c5ab2`完成并推送。D1-0已冻结真实历史回放合同和精确样本
水位，仍没有模型成绩；若远端分支已包含提交信息`docs: freeze D1 causal replay contract`，D1-0即
已闭合，恢复后不要重做文档，直接进入D1六模型历史回放。

## 工作位置

- 工作树：`D:\AIPred\SeismoFlux\data\interim\worktrees\science_first`
- 分支：`codex/stage2-etas-science-first`
- 远端：`origin/codex/stage2-etas-science-first`
- 唯一蓝图：`SEISMOFLUX_IMPLEMENTATION_HANDOFF.md`第1.1节
- 机器合同：`configs/d1_retrospective_development.yaml`
- 水位清单：`data/manifests/d1_fold_water_level_manifest.json`
- 通俗科学合同：`docs/d1_executable_scientific_contract.md`
- 本阶段验收：`docs/d1_0_acceptance_2026-08-03.md`

## 恢复时先做

```powershell
Set-Location D:\AIPred\SeismoFlux\data\interim\worktrees\science_first
git status --short --branch
git log -3 --oneline --decorate
git ls-remote origin refs/heads/codex/stage2-etas-science-first
```

判定：

1. 远端含`docs: freeze D1 causal replay contract`：D1-0已闭合，直接执行下一节；
2. 本地有该提交但远端没有：只补推送并回读，不重做合同；
3. 本地也没有：只验收并提交本交接列出的D1-0文件，不能把下述旧Stage4草稿混入提交。

## 下一项直接科学检验

用三个时间外推折比较六模型：

1. 75 km长期地震背景`B0`；
2. 加最近30天地震`B0_R30`；
3. 加报告覆盖`B0_C`；
4. 加单期异常`B0_C_A_snapshot`；
5. 加动态异常`B0_C_A_dynamic`；
6. 全组合`B0_R30_C_A_dynamic`。

训练使用未来30天M4+的条件空间位置；一级评价是30天、600,000 km²下21个规则化独立M5–6
震群的严格召回，并同时报告同召回所需面积。结果必须含六模型、三折、时间/空间置乱、原始命中群数、
不确定性、静态图和可切换issue/模型/horizon/面积的离线交互页。

## 数据和资源

- 地震：`data/processed/stage1/debc98054172a4a1/earthquake_event.parquet`，40,898行；
- 报告日历：同目录`anomaly_report_period.parquet`，205行；
- 特征：`data/processed/stage3/anomaly_history/anomaly-feature-bundle-de7547faa9f87541/`，
  3,217,885行特征和166,189行状态；
- 本地空间置乱映射：`data/interim/stage4/anomaly_increment_r2/`，只读取四个已哈希工件，不碰临时文件；
- 默认最多4个worker，每个BLAS/OpenMP线程为1，至少保留2个物理核心；GPU仅在float64等价时加速。

检查点按一个“折×模型”或25个置乱复本落盘。合同和输入哈希不一致时不得混接旧结果；评分前实现
错误记`invalid_run`并做最小修复，不能从多次重试中挑最好。

## 不得碰的现场草稿

工作树中另有7个未跟踪`src/seismoflux/anomaly_increment/kde_dev_*.py`和8个对应测试，它们属于此前
路线的未验收草稿，不是D1-0提交内容。不要编辑、删除、暂存或据此推断科学结果。

## 汇报和中断要求

每完成一个折或一批置乱，更新当前状态与恢复入口。D1阶段汇报固定说明：用了/没用什么数据、怎样
防止未来泄漏、九类线索、六模型训练、训练收敛、真实时间外推命中与面积、置乱和失败案例，以及它
对最终预测目标是直接提升、可信负结果还是证据不足。工程测试数不能代替真实效果。

## 当前科学价值状态（机器可检索）

- `science_value_category`: `necessary_enabler`
- `evidence`: D1-0只冻结真实样本水位、六模型和因果评价规则；没有打开任何模型成绩。
- `decision`: 远端确认D1-0闭合后直接运行D1，不再增加无关工程前置。
- `next_scientific_test`: 六模型三折真实历史回放，一级比较30天、600,000 km²下的命中震群数与面积效率。
- `stop_condition`: 异常增量方向不正、与置乱无区别或完全依赖单一区域/震群时，停止对应组件复杂化；
  小而稳定的正提升则进入真正前瞻积累。
