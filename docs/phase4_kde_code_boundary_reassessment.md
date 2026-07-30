# Stage 4A 代码边界科学价值复审

- 日期：2026-07-30
- 协议：`stage4-kde-development-v1` / `0.4.3`
- 真实开发输入读取：0
- 唯一科学 attempt 消耗：0
- 结论：保留科学实验，调整代码职责；停止继续扩张独立统计 reducer
- 后续状态：本文件规定的唯一集成审计已于 2026-07-30 发现 foundational P0 并触发停线；
  generic builder 不再继续修补；当时唯一许可方案见
  `docs/phase4_kde_integrated_audit_stop_and_one_shot_contract.md`。该 one-shot 方案随后也在唯一独立
  审计中发现新的 foundational P0 并停止；当前唯一许可方向见
  `docs/phase4_kde_one_shot_audit_failure_and_science_first_simplification.md`

## 1. 为什么触发复审

首轮 score-blind 代码已经实现封印、随机映射、统计门和 `fold_4` 背景重物化。独立审计先后发现并
修复了缺失 seal binding、派生结论可绕过、输入顺序漂移、背景冻结身份可覆盖和可变序列 TOCTOU。

最后一轮统计窄审仍指出：如果 `Stage4ARawEvidence` 只接收调用方声明的 SHA-256，统计 reducer 最多
只能检查字符串格式、相等、互异和顺序，不能仅凭摘要证明摘要确实来自对应的事件成员、Bootstrap
draw 或 placebo mapping。继续在 reducer 内增加自声明哈希，只会把真实性问题向上游递归移动，
不会更接近固定报警面积下的预测效果检验。

该结果触发用户要求的停止规则：不再用更多孤立工程补丁掩盖职责边界不清，先调整实现方案。

## 2. 科学方案是否错误

当前没有证据表明核心科学实验错误。以下内容保持不变：

- 75 km 历史地震 KDE 是当前最好合法背景；
- `B0/C0/B1/B2` 在相同支持域、目标、窗口、积分域和 600,000 平方公里下比较；
- 只允许三个滚动折、7/30/90 天和一次开发科学 attempt；
- 2,000 次物理事件块 Bootstrap、时间/空间各 1,000 次成对置乱；
- 信息增益、固定面积严格召回 `+5 pp`、同时区间、`maxT` 和跨折/跨区门不变；
- 失败或证据不足时停止异常模型扩张，不用更复杂模型补救。

问题出在“谁负责证明统计值来自哪一个原始 draw/mapping”，不是科学问题、数据或评价指标本身。

## 3. 调整后的唯一职责边界

### 3.1 canonical provenance builder

code-sealed 薄 runner 必须提供唯一 canonical builder，并直接从实际运行对象计算来源身份：

- 去重事件并集与折—窗口 membership；
- 事件到预冻结区域的绑定；
- 每个 Bootstrap replication 的 event-block draw；
- 每个时间置乱 replication 的 `TimeMappingDTO`；
- 每个空间置乱 replication 的 `SpaceMappingDTO`。

每个摘要使用域分离 canonical payload，至少包含
`kind + plan identity + replication_index + canonical payload`。plan 必须绑定完整有序
`replication_index → draw/mapping identity` 向量。正式运行器只能通过该 builder 构造
`Stage4ARawEvidence`；静态 AST/import 测试禁止在正式路径直接实例化 raw evidence 或调用私有派生门。

### 3.2 statistics reducer

统计模块只承担：

- 深度不可变和维度/复本数校验；
- 同一 builder receipt 的身份一致性检查；
- 共享 Bonferroni 区间、成对 `maxT`、候选门、动态门和 overall decision；
- 非有限、零事件和证据不足的 fail-closed 语义。

它不再被要求仅凭调用方给出的摘要反向证明未提供的原始 payload。这一真实性由同一 code tag 下的
canonical builder、checkpoint receipt、code seal 和正式路径静态禁单共同承担。

## 4. 有限验收与硬停止条件

调整后只允许一次端到端纯合成验收，同时验证：

1. 相同原始 payload 在不同载入顺序和 worker 数下得到同一身份；
2. 交换任意两个 replication 的 draw/mapping 或数值，builder receipt/正式装配失败；
3. 修改事件 membership、区域绑定、draw 或 mapping 任一项都会改变对应 plan/receipt；
4. 正式 runner 没有绕过 canonical builder 的入口；
5. `B0/C0/B1/B2` 仍共享背景、rate head、目标集合、格网和固定面积；
6. 统计唯一入口从 builder 产物生成门控，不接受外部 decision、CI 或 `maxT`。

若这一次集成审计仍发现新的基础来源合同缺口，立即暂停 Stage 4A 实现，重新设计更简单的实验执行
合同；不得继续给独立模块增加新的自声明身份字段。

## 5. 科学价值复审

- `science_value_category`: `no_material_progress`
- `evidence`: 本轮没有产生任何预测成绩；连续 reducer 补丁没有提高固定面积召回，也不能仅凭摘要
  证明上游来源真实性，继续同方向优化会成为无休止工程。
- `decision`: `adjust`
- `next_scientific_test`: 只实现 sealed runner 的 canonical provenance builder，并执行一次端到端
  纯合成审计；通过后才完成其余薄 runner，仍不得打开真实输入。
- `stop_condition`: 集成审计仍发现新的基础来源合同缺口，或需要改变候选、特征、面积、统计门、
  读取次数和 attempt 身份时，停止 Stage 4A 代码扩张并重审整个执行协议。

该调整不是预测效果提升，也不授权真实数据读取。它的目的只是把剩余工程压缩为一次能够直接通向唯一
Stage 4A 科学检验的、职责明确的实现。
