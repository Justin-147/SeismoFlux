# Stage 2S-1 代码发布与正式评估续接交接

- 更新日期：2026-07-30
- 工作树：`D:\AIPred\SeismoFlux\data\interim\worktrees\science_first`
- 分支：`codex/stage2-etas-science-first`
- 协议提交：`98e21573057d9a73d552b0cbac7a64f5206b3546`
- 协议标签：`v0.2.3-causal-seismicity-screen-protocol`，已远端核验
- 预期代码标签：`v0.2.3-causal-seismicity-screen-code`
- 当前阶段：`Stage2S-1` 代码发布
- 当前结论：独立终审 `GO`；只允许精确提交 Stage 2S-1 文件并核验远端代码标签
- 真实科学结果：尚无
- 唯一正式 attempt：尚未消耗

## 1. 外行能听懂的现状

程序已经通过“闭卷考试系统”的最终检查，但真正的历史考试还没有开始。

本阶段比较三张未来地震相对强度图：

- `S0`：长期 75 km 地震多发背景；
- `S1`：长期背景加上起报前最近 30 天、当时已经知道的 M4+ 地震；
- `SP`：长期背景加上再前一个 30 天的同结构对照。

三张图使用同一未来 M5–6 总率和同一 600,000 平方公里报警面积。因此它只回答
“最近地震能不能帮助判断未来更可能落在哪里”，不回答具体哪天发震、总数会不会增加，也不输出
绝对发震概率。

截至本文件落盘：

- Stage 2S 尚未打开真实地震目录；
- 正式 non-target preflight receipt、attempt ledger、target-read receipt、master seal、结果和
  terminal 均不存在；
- 合成测试和图件只证明实现可运行、可审计，不能证明预测已经变好。

## 2. 已完成的实现与审计

1. 真实目标坐标和 assessment membership 只有在三折 master prediction seal 后才能开放。
2. 最近 30 天 `R` 与再前 30 天 `RP` 均同时受 `origin_time_utc` 和 `available_at` 截止约束。
3. `S0/S1/SP` 共享支持域、共同 M5–6 率、连续密度算子和精确报警面积；补偿差独立复算。
4. alpha 使用无 floor、无 clipping 的稳定 log-density 求解，覆盖极端密度比、双下溢和近抵消。
5. 2,000 次配对物理事件块 Bootstrap、39 区闭合、去最大贡献区、震群 leave-out 和 1/7 天延迟
   均实现并进入机器记录。
6. 非目标 preflight 在 attempt 前绑定 50/25/12.5 km 三层格网 ID、逐层面积、代表点身份、
   12.5→25 与 25→50 父子映射及面积闭合。
7. 一次性 transaction 覆盖全部正式 sink 初始缺失、`O_EXCL`、单次目录 bytes、异常/中断终态和
   重启只读 finalizer；不会因重启偷偷重跑。
8. 静态图和两个离线交互页同时展示 S0/S1/SP、数据截止、面积、失败案例、区域/LORO、延迟和
   震群主导诊断；standalone 校验器强制核对 whole-run hash、六工件 hash 和 frame/seal 身份。
9. 独立终审未发现新 P0/P1，原三项 P1 已闭合，结论为 code-tag `GO`。

## 3. 最终统一验收证据

- 精确 15 个 Stage 2S 测试文件：`172 passed in 45.84s`
- 真实 production 科学路径的纯合成端到端 canary：`1 passed in 4.95s`
- Ruff：33 个 Stage 2S source/script/test 文件全部通过
- Ruff format：`33 files already formatted`
- strict mypy：`Success: no issues found in 33 source files`
- 因果时间线确定性重建：通过
- `git diff --check`：通过
- 正式 launcher：24 个物理核心、1 worker、六类数值线程均为 1、保留 2 个物理核心
- PNG 人工检查：面板无重叠，近零召回不再显示科学计数偏移
- 两个离线 HTML：各一个内联 script，Node 语法通过，无外链、`fetch`、XHR 或 WebSocket

残余 P2：应用内浏览器的本地 URL 安全策略阻止了真实点击式 QA。Node、离线性、字段/控件单测和
静态人工检查均通过；这不阻断 code tag，但正式结果形成后仍应在可用的受控浏览器中补交互抽查。

## 4. CPU 与进程边界

- 所有验收和正式执行使用单 worker。
- `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、
  `NUMEXPR_NUM_THREADS`、`VECLIB_MAXIMUM_THREADS`、`BLIS_NUM_THREADS` 均固定为 `1`。
- 本机 24 个物理核心，正式入口验证可用 22 个、保留至少 2 个。
- 最近三次系统总 CPU 采样约为 `49.9% / 48.9% / 52.1%`，主要负载来自其他任务。
- PID `61524` 及其它非本阶段 Python 进程不终止、不接管。

## 5. 下一步严格顺序

1. 只按白名单暂存 Stage 2S-1 文件，排除全部未跟踪 Stage 4 草稿。
2. 提交并推送分支。
3. 创建并推送 `v0.2.3-causal-seismicity-screen-code`，远端核对 peeled commit。
4. 再检查正式 receipt、attempt、seal、result、terminal 和六工件路径全部缺失。
5. 正式入口先完成无目标 study-area/grid/cell-zone preflight。
6. 只有 preflight 通过才原子创建唯一 attempt ledger 和 target-read receipt，并物理读取真实目录
   一次。
7. 无论门控为通过、失败、证据不足或无效，都停止调参，保留原始结果和封印，生成静态/交互回溯
   与科学价值复审，提交、推送并打 result tag。

禁止在看到结果后改变 30 天窗口、75 km 带宽、600,000 平方公里面积、区域、候选或随机规则。
Stage 2S 禁止生成“当前预测”图；前瞻预测必须另立不可回填协议。

## 6. 提交边界

允许：

- `.gitignore` 中 Stage 2S attempt ledger 规则；
- `docs/phase2s1_causal_seismicity_code_acceptance.md`；
- 本交接文档和 `docs/stage2s_data_method_causal_timeline.svg`；
- 三个 `scripts/*stage2s*` 脚本；
- `src/seismoflux/stage2s/` 下 15 个模块；
- 14 个新增 Stage 2S 单元、泄漏和集成测试。

禁止：

- `src/seismoflux/anomaly_increment/kde_dev_*.py`；
- `tests/unit/test_stage4_kde_dev_*.py`；
- 任何正式 receipt、attempt、seal、result 或 terminal；
- `git add .`。

## 7. 科学价值复审

- `science_value_category`: `necessary_enabler`
- `evidence`: 主科学链、因果隔离、一次性治理和结果展示已经通过 172 项统一测试与独立终审；尚无
  真实历史效果指标
- `decision`: 结束工程扩写，完成代码标签后立即进入唯一真实历史开发筛查
- `next_scientific_test`: 在冻结三折、同一 600,000 平方公里下，一次性检验 S1 是否同时超过 S0
  和 SP
- `stop_condition`: 任一新 foundational P0、正式门失败、证据不足或无效都停止 Stage 2S，
  保留 75 km KDE，不得基于结果换窗口、模型或面积重试
