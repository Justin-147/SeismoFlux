# 阶段 2 ETAS 最小数值资格验收

## Q0 协议冻结

日期：2026-07-29

状态：`PASS_LOCAL`（等待外部 Git 发布门）

### 科学边界

- [x] 唯一蓝图未修改，且新协议明确服从蓝图。
- [x] G1-LS 仍记录为由 75 km KDE 通过；ETAS 资格不是新增 G1 门。
- [x] 五个快照、25 个起点、模型、边界、优化器和阈值与 R4 相同。
- [x] 每个快照同时限制 `origin_time` 和 `available_at` 不晚于 `fit_end`。
- [x] 异常、阶段4目标、既有分数和锁定测试读取均为禁止。
- [x] 正负结果都冻结；不得新增 Candidate 8 或事后放宽。

### 工程最小性

- [x] Q0 只新增配置、协议、验收和协议测试四个文件。
- [x] 未复制 R5 的 launcher、evidence、IO、Win32 句柄或事务实现。
- [x] 结果要求包含中文报告、静态 SVG 和离线交互 HTML。
- [x] 单进程、BLAS 单线程并至少保留两个物理核心。

### 验证

- [x] 协议测试全部通过：`6 passed in 2.06s`。
- [x] YAML 可由拒绝重复键的加载器解析。
- [x] `ruff check`、`ruff format --check`、严格 `mypy` 和 `git diff --check` 全部通过。
- [x] 工作树除 Q0 四个文件外无其他差异。
- [x] 独立只读科学门审计复验为 `PASS`。
- [x] 独立只读最小实现审计复验为 `PASS`。

### 本地验收结论

“科学边界”“工程最小性”和“验证”三组已全部满足，本文件据此标记为 `PASS_LOCAL`。

提交、推送和 annotated tag 是本文件冻结后的外部发布门，不能写成冻结提交对自身未来状态的声明。Q1 只有在以下命令事实均由 Git 远端和当前交接文档记录后才能开始：

1. 冻结提交存在且工作树干净；
2. 分支已推送；
3. `v0.2.2-background-etas-qualification-protocol` 为指向该提交的远端 annotated tag。

## 阶段性科学价值门

本门是 Q1、Q2 以及后续各阶段验收的强制组成，不得用工程完成度替代。

每次阶段性结果形成后必须立即记录：

1. 对最终目标的作用分类：`direct_improvement`、`necessary_enabler` 或 `no_material_progress`；
2. 与当前最好合法基线在同一支持域、同一未来窗口和同一报警面积下的证据；
3. 对长期 KDE、ETAS/Hawkes、动态异常和长期构造先验的可归因贡献，后续用冻结消融和配对比较核验；
4. 下一项最接近科学目标的检验，以及继续投入当前路线的停止条件；
5. 决策：`continue`、`adjust` 或 `stop`。

Q0 当前分类为 `necessary_enabler`：它冻结了一次不读阶段 4 正式目标的 ETAS 可计算性检验，但没有产生预测效果提升证据。Q1 只有在把该检验实现为最小、可独立复算且可停止的工具时仍属于必要使能；若实现继续扩张却不能直接进入唯一 Q2 资格，则应判为 `no_material_progress` 并停止扩张。

Q2 无论得到 `evaluable` 或 `not_evaluable` 都必须立即复审：

- `evaluable` 只表示 ETAS 组件可进入与长期 KDE、动态异常及其组合的滚动预测比较，不等于 ETAS 已提高召回；
- `not_evaluable` 表示停止当前数值修复路线，但不得解释为历史多发区或地震活动对预测无用；75 km KDE 仍是合法有效成分。Q2 结果标签后先冻结新的目标盲阶段 4 执行协议，再按唯一蓝图允许的“最佳背景模型”路线继续最简单的 KDE + 异常增量检验；
- 真正的 `direct_improvement` 只能由未来隔离的滚动验证证明：在固定报警面积下召回提高，或达到同召回所需面积下降，并通过预登记消融说明各模型贡献。

完整路线及 ETAS 正负结果的分流见
`docs/scientific_value_review_and_model_composition.md`。该澄清不修改 Q0 协议字节、不读取阶段 4 正式目标，也不授权在新阶段 4 协议冻结前读取目标。

## Q0 外部发布闭环记录

Q0 冻结后的外部事实已经完成：

- 冻结提交：`56f49035488a24f94d3e08d9f4781a4c4cd8d9e0`；
- 分支：`codex/stage2-etas-science-first`，已推送；
- 远端 annotated tag：`v0.2.2-background-etas-qualification-protocol`；
- 远端 peeled commit 与冻结提交一致。

因此 Q1 可以合法开始。

## Q1 最小代码冻结

日期：2026-07-29

状态：`PASS_LOCAL_WITH_RECORDED_DEVIATION`（等待外部 Git 发布门）

### 科学与防泄漏边界

- [x] 运行路径只构造 ETAS 拟合事件、因果父历史、75 km KDE 和拟合区间积分问题，不构造阶段 4 正式目标、评分问题或命中。
- [x] 五个快照均同时执行 `origin_time <= fit_end` 和 `available_at <= fit_end`。
- [x] 五快照 × 五冻结起点、种子协议 `0.2.1` 和每个起点 float hex 逐项核对。
- [x] fold 1、fold 3 的 `unsupported` 父历史敏感性只在主参数处重算 objective，新增优化调用数为 0。
- [x] 三网格门使用拟合问题，不使用未来评估问题。
- [x] Q0 协议文件按完整 SHA-256 绑定；实际模型、边界、优化器、阈值和积分容差与 Q0 机器协议逐项核对。
- [x] Q0 协议字节和唯一蓝图冻结副本均未修改。

### 工程边界

- [x] `ETASParameterBounds.from_transformed` 只加入严格变换边界检查和 exact-endpoint 映射；没有 clip、容差、边界扩张或目标函数变化。
- [x] runner 在任何 `seismoflux` 导入前固定本工作树 `src`，并核验模块路径，不能误载根工作树草稿。
- [x] runner 只接受唯一固定协议路径、annotated code tag、tagged HEAD 和无非输出代码差异的工作树。
- [x] 单进程、BLAS 单线程；Windows API 实测 24 个物理核心，启动 CPU 占用必须低于 70%。
- [x] 快照按同目录临时文件加 create-if-absent 落盘；receipt 阻止删除已完成快照后重新拟合。
- [x] 合法中断只补从未落盘的快照；已完成快照重新打开后独立复算。
- [x] attempt environment seal 绑定 Q0 协议、`uv.lock`、Python、NumPy 和 SciPy 版本，禁止跨数值环境混合恢复。
- [x] verify 独立重建输入清单、结果清单、objective、梯度、Hessian、分支比、三网格、敏感性和三份公开视图。
- [x] 中文 Markdown、静态 SVG 和完全离线 HTML 不含事件编号、坐标、绝对本地路径、异常、目标或评分。

### 验证

- [x] 聚焦 Q1、Q0 协议和端点测试：`49 passed`。
- [x] Q1 相关局部支持、ETAS、KDE、网格和数值回归集：`154 passed`。
- [x] 可确认不含正式数据、阶段 4 目标和锁定测试入口的广泛背景回归集：`348 passed`。
- [x] `ruff check`、`ruff format --check`、五文件 strict `mypy` 和 `git diff --check` 全部通过。
- [x] 清除 `PYTHONPATH` 后，runner 仍从本 science-first 工作树导入。
- [x] 独立最小工程复审：`PASS_WITH_RECORDED_DEVIATION`。
- [x] 独立科学复审：`PASS`；透明偏差不构成科学污染或 Q1 阻断。

### 非阻断治理偏差

Q1 配置烟测曾误调用会验证数据引用的旧 loader。它只对隔离工作树内不存在的
`data/processed/china_mainland.geojson` 执行了一次 `Path.is_file()` 存在性检查，随后立即以
`missing` 失败：

- 打开或读取文件字节：`0`；
- 哈希、目录枚举、事件、几何、异常、目标或评分读取：`0`；
- 根项目真实 `data/processed` 接触：`0`；
- Q2 优化调用和锁定测试消费：`0`。

根因是配置烟测错误使用 `load_project_background_config`。实现已改为
`load_config + load_background_protocol` 的目标盲加载，并把真实输入的精确路径和哈希核验推迟到
code tag 之后；相应回归测试已经加入。独立工程复审判定该事件没有产生科学信息、没有污染目标盲性，
因此记录为 `non_blocking_governance_deviation`，不得删除或改写。

### 科学价值复审

- 分类：`necessary_enabler`，不是 `direct_improvement`。
- 实质作用：把当前 ETAS 实现压缩为一次可运行、可停止、可独立复算的 Q2 数值资格；没有声称提高预测召回。
- 最近科学证据：唯一一次 Q2 资格。Q2 后不得继续扩张本工具。
- 停止条件：Q2 无论正负都结束本数值资格路线；负结果停止 ETAS 修复，正结果只允许进入与 KDE 及组合的滚动效果比较。
- 决策：`continue`，但只继续到 Q2；随后立即重新执行科学价值复审。

Q1 提交、推送和 `v0.2.2-background-etas-qualification-code` annotated tag 是本地验收后的外部发布门。三者完成并由远端回读核验前，不得读取真实阶段 2 输入或启动 Q2。
