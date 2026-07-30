# 阶段 2S-1 因果近期地震活动代码验收

- 验收日期：2026-07-30
- 阶段：`Stage2S-1 code stage`
- 协议版本：`0.2.3`
- 实验 ID：`stage2s-causal-seismicity-development-v1`
- 协议提交：`98e21573057d9a73d552b0cbac7a64f5206b3546`
- 协议标签：`v0.2.3-causal-seismicity-screen-protocol`
- 代码标签：`v0.2.3-causal-seismicity-screen-code`
- 本地结论：独立终审 `GO`
- 执行权限：只有本提交、分支和代码标签均推送并远端核验后，才允许进入唯一真实 one-shot

## 1. 外行能听懂的结论

代码已经具备参加一次正式历史考试的条件，但目前还没有新的真实预测成绩。

这套方法把“长期地震多发背景”与“最近 30 天已发生的小一些地震”叠加，再与长期背景和再前一个
30 天对照做公平比较。三种方法报警面积相同，程序先把每个起报日的图封存，再打开后来发生的
M5–6 地震评分，避免先看答案后画图。

如果新方法通过，只能说明这条时间因果信号值得另做前瞻封存验证；如果失败或证据不足，就停止这条
30 天路线，不能看成绩后换成 60/90/365 天再试。

## 2. 数据、模型和评价

- `S0`：G1-LS 已通过的长期 75 km KDE；
- `S1`：`S0` 加起报前最近 30 天、届时已经可用的等权 M4+ 地震 KDE；
- `SP`：`S0` 加紧邻的再前 30 天同结构 KDE；
- 目标：三个滚动折中的未来 M5–6 地震；
- 时间窗：7/30/90 天；
- 正式面积：600,000 平方公里；
- 指标：连续空间信息增益、同面积严格召回、跨折、跨区、去主导区、震群 leave-out 和 1/7 天
  可用性延迟；
- 随机性：冻结 PCG64 namespace，2,000 次配对物理事件块 Bootstrap。

ETAS 在既有阶段中仍为 `not_evaluable`，本阶段不改写该结论；因此与当前最好合法背景 `S0`
比较，并用完全同结构的过去窗口 `SP` 检验“最近 30 天”是否真的优于一般空间成团。
本阶段不使用异常表、人工预测、故障、危险性或大型神经网络。

## 3. 防泄漏与一次机会治理

正式顺序固定为：

```text
远端 code tag 核验
  -> 无目标空间 preflight receipt
  -> O_EXCL attempt ledger
  -> O_EXCL target-read receipt
  -> 真实目录只物理打开一次并读入不可变 bytes
  -> fold 1 fit/issue/fold seals
  -> fold 2 fit/issue/fold seals
  -> fold 3 fit/issue/fold seals
  -> master prediction seal
  -> 才开放 assessment memberships/coordinates 并统一评分
```

非目标 preflight 已绑定 50/25/12.5 km 三层格网、面积、代表点、父子映射和区域关系。正式入口在
attempt 前检查全部 receipt、seal、result、terminal 和六个展示工件均不存在。中断后只允许读取
不可变账本形成终态，禁止重读目录或重跑。

## 4. 验收证据

- Stage 2S 精确测试集：`172 passed in 45.84s`
- 同一 production 科学路径纯合成端到端：`1 passed in 4.95s`
- Ruff：33 文件全部通过
- Ruff format：33 文件已格式化
- strict mypy：33 文件无问题
- `git diff --check`：通过
- 数据—方法—因果时间线确定性校验：通过
- 正式资源预检：24 个物理核心、1 worker、六类数值线程均为 1、保留 2 个物理核心
- 两个离线 HTML：Node 语法通过、单内联、无网络依赖
- 静态 PNG：人工检查无面板重叠、无近零科学计数偏移
- 独立终审：未发现新 P0/P1；三项发布阻断均由负向 canary 闭合，结论 `GO`

浏览器本地 URL 被应用安全策略阻断，因此尚未完成实际点击式 QA；这是已登记 P2，不影响 code
tag。正式结果形成后应在允许本地离线页面的受控浏览器补抽查。

## 5. 当前真实数据状态

截至验收：

- 没有打开、统计或枚举真实 `data/processed`；
- non-target preflight receipt 不存在；
- formal attempt ledger 不存在；
- target-read receipt 不存在；
- master prediction seal、whole-run record、terminal 和正式输出目录均不存在；
- 唯一真实 Stage 2S 开发机会没有消耗。

## 6. 科学价值复审

- `science_value_category`: `necessary_enabler`
- `evidence`: 已证明冻结科学链可在生产路径中因果隔离、一次性执行、完整评分和可视化，但所有
  成绩仍来自合成验收，不能说明真实预测效果提高
- `decision`: 不再增加工程功能；提交、推送并远端核验 code tag 后，立即进行唯一真实历史筛查
- `next_scientific_test`: 在冻结三折与同一 600,000 平方公里报警面积下，一次性比较
  `S1-S0` 和 `S1-SP` 的信息增益与严格召回
- `stop_condition`: 新 foundational P0、门失败、证据不足或无效均停止 Stage 2S；不得基于结果
  换窗口、带宽、面积、区域、模型或随机规则重试

## 7. 精确提交边界

本阶段只提交 `.gitignore`、本验收/交接/时间线、三个 Stage 2S 脚本、独立
`src/seismoflux/stage2s/` 模块和 Stage 2S 测试。

以下未跟踪草稿必须继续隔离：

- `src/seismoflux/anomaly_increment/kde_dev_*.py`
- `tests/unit/test_stage4_kde_dev_*.py`
- `tests/unit/test_stage4_kde_dev_synthetic_chain.py`

禁止 `git add .`，禁止把本地正式 attempt/receipt/seal/result/terminal 加入公开仓库。
