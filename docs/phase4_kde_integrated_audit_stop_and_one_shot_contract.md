# Stage 4A 集成审计停线与 one-shot 执行合同

- 日期：2026-07-30
- 后续状态：本合同规定的唯一 one-shot 独立审计已发现新的 foundational P0 并判定 `FAIL`；
  one-shot whole-record 路线已停止，不再提供执行授权。证据与科学优先简化方案见
  `docs/phase4_kde_one_shot_audit_failure_and_science_first_simplification.md`。
- 协议：`stage4-kde-development-v1` / `0.4.3`
- 审计性质：score-blind、纯合成、只读独立审计
- 真实开发输入读取：0
- 独立验证读取：0
- 锁定测试读取：0
- 唯一科学 attempt 消耗：0
- 当时决定（现已废止）：停止 generic provenance builder 的补丁式扩张，改成单入口、单记录的
  封闭执行合同
- 前置复审：`docs/phase4_kde_code_boundary_reassessment.md`

## 1. 已完成但不能当作科学结果的工作

canonical builder 草稿通过了完整 `2000/1000/1000` 纯合成路径测试；seal、mapping、
fold-4 background、statistics 与 runner 的联合回归为 `177 passed`，Ruff、format、strict mypy
和 `git diff --check` 也通过。

这些结果只说明已有测试覆盖的代码能够运行，不说明异常信息提高了地震预测效果，也不能覆盖独立审计
发现的来源合同缺口。

## 2. 唯一集成审计的失败证据

### 2.1 foundational P0：完整 receipt 没有绑定主成绩

当前 `Stage4AProvenanceReceipt` 绑定了 membership、region、三个 replication plan 和复本结果，
却没有绑定：

- `B1/B2` 的 observed metrics；
- `fit_and_score_valid`；
- 三折 × 三窗口的信息增益和固定面积召回；
- 区域贡献与 leave-one-region-out 数值；
- 由这些值派生的 gate。

独立审计在纯合成输入上只把 `B1.fit_and_score_valid` 从 `True` 改为 `False`，继续传入原完整
receipt，调用仍被接受；receipt 保持完全相同，而 overall gate 从 `passed` 变成 `invalid`。

因此同一张所谓“完整凭证”可以对应相反科学结论。该问题不是测试数量能够弥补的局部格式错误，而是
证据边界错误。

### 2.2 foundational P0：置乱来源仍可由调用方替换

generic builder 仍接受调用方预构造的 Bootstrap payload、`TimeMappingDTO`、`SpaceMappingDTO` 和
统计 callback。当前 canonical payload 没有完整绑定：

- 六个固定 `fold × fit/assessment` 时间池及各自 `Stage4AStreamContext`；
- 全部实际 `issue × frozen stratum` 空间原始组及 joint context；
- 从冻结 randomness manifest 到 draw/mapping 再到统计值的封闭调用链。

测试中的任意伪 DTO 可以进入 builder；每个时间 replication 只含任意一个非空 scope 也会被接受，
空间 replication 之间也不要求使用同一完整 group key 集。它能证明“收到的对象被哈希”，不能证明
“正式协议规定的全部原始池被正确置乱并用于该数值”。

### 2.3 审计结论

本次唯一集成审计为 `FAIL`。上述任一 P0 都足以触发预登记硬停止条件。当前 generic builder：

- 不得进入 code freeze；
- 不得作为正式 Stage 4A 执行入口提交；
- 不授权打开真实开发目标；
- 不授权消耗 `stage4-kde-development-v1-attempt-1`；
- 不再通过增加 detached SHA、receipt 层或 callback 身份字段继续修补。

## 3. 通俗解释

当前实现相当于给实验附件贴了封条，却没有把主成绩单一起封进去。换掉主成绩单后，封条仍然一样；
而且部分置乱附件还是由外部递进来的，程序不能确认它们确实来自规定的原始池和随机流。

这不说明 KDE 或异常模型一定无效，只说明现在还不能信任这条执行路径给出的“通过/失败”结论。

## 4. 历史 one-shot 合同（已废止）

本节至第 8 节只保留为“当时准备怎样执行”的历史审计证据，不构成当前授权。该合同规定的唯一独立
审计已经失败，禁止据此 code freeze、打开真实目标或运行真实 attempt。

当时设计的正式路径只保留一个公开入口：

```python
def run_stage4a_once(*, repository_root: Path) -> Stage4AWholeRunRecord:
    ...
```

调用方不得再传入：

- observed metrics、raw evidence 或 gate；
- Bootstrap draw、time/space mapping 或 replication 统计值；
- `statistic_callback`；
- `expected_receipt`；
- 可替换的 work item、并行 hash/value 数组或通用 checkpoint。

runner 在一个 code-sealed 调用链内完成：

1. 核验协议、source commit、code tag、code seal、输入/随机清单、环境锁和零态 ledger；
2. 注册唯一 attempt，并把 target-read ledger 从零态原子 CAS 到唯一打开态；
3. 只打开一次目录；先冻结 `B0/C0/B1/B2` 预测和 25 km、600,000 平方公里完整前缀，再读取目标评分；
4. 从实际运行对象构造事件 membership、区域绑定、六个时间池和全部空间原始组；
5. 用冻结的完整 RNG context 内部生成 2,000 个事件块 draw、1,000 组时间 mapping 和 1,000 组
   空间 mapping；
6. 在同一私有函数内完成“生成 → 重建/重拟合 → 计算复本统计”，不跨越可替换 callback；
7. 从实际 observed evidence 和全部复本重建 `Stage4ARawEvidence`，只调用一次正式统计门；
8. 原子写出唯一 `Stage4AWholeRunRecord`。

如需容忍断电，只允许 runner 私有、code-sealed、append-only 的进度日志。它不是科学证据 API，
调用方不能构造、替换或提供预期值；最终记录必须从日志的完整实际行重新汇总。

## 5. 唯一 whole-run record

`Stage4AWholeRunRecord` 同时保存：

1. **冻结身份**：协议、实验/门/attempt、source commit、code tag、code seal、输入清单、frozen
   input seal、randomness manifest、环境锁和两个 ledger 的前后身份；
2. **实际输入身份**：fold-4 背景 receipt、历史训练行、25 km 有序质量、预测 bundle、事件
   membership、区域绑定、每个时间池和每个空间组的 canonical 身份与行数；
3. **不可分离复本行**：kind/index、完整 RNG context、原始池身份、实际 draw/mapping、实际统计值
   和 failure code 必须位于同一行；
4. **主成绩**：`B0/C0/B1/B2` 全部逐折逐窗 observed metrics、四个正式比较、区域/LORO 结果、
   同时区间输入、两组 `maxT` 输入、CI、p 值、candidate truth table、overall status 和
   selected/fallback variant；
5. **运行与展示产物**：受控 CPU/GPU 元数据、静态图和交互页的 artifact SHA-256。

最终只计算一个：

```text
run_record_sha256 =
SHA256(canonical_json(Stage4AWholeRunRecord_without_run_record_sha256))
```

record 本身就是唯一 receipt，不再创建 detached provenance receipt。验证器必须从 record 内原始
证据重新计算 gate，并要求逐字段等于保存的 gate；仅重算外层 SHA 不算通过。

## 6. code seal 与 record 的责任边界

code seal 负责冻结计算语义和控制流，包括：原始池提取、排序、RNG 派生、mapping 构造、特征重建、
重拟合、评分、固定面积前缀、未支持目标计漏、唯一目录会话、预测先冻结后评分，以及 raw evidence
和 gate 的唯一装配路径。

whole-run record 负责绑定本次实际输入、实际数值和实际结论。它不再声称仅凭摘要能反向证明计算
语义；语义由 code-sealed 的封闭调用链承担。

## 7. 历史合同的有限验收（已失败）

新合同只允许一次纯合成端到端验收：

1. 精确完成 `2000/1000/1000`，并从 record 内原始值重算 gate；
2. 输入载入顺序和 worker 数变化不改变 scientific payload、observed metrics 和 gate；
3. 分别修改 membership/region、time pool、space pool、RNG context、draw/mapping、observed
   metric、replication statistic 或 gate 任一项，都必须使验证失败或 whole-record hash 改变；
4. 交换任意两行数值或完整行都必须因 index→context→pool→mapping 日程不符而失败；
5. AST/import 验收确认不存在 callback、`expected_receipt`、外部 replication payload、外部 raw
   evidence/gate、动态导入或第二 target reader；
6. 最多 6 workers、每进程 BLAS 1 thread、至少保留 2 个物理核心；
7. 一次独立集成审计通过后，才允许 code freeze。

若该新合同再次出现 foundational P0，立即停止 Stage 4A 工程，不再增加 provenance 层；保留当前
KDE 合法基线，回到异常数据、假设和更简单科学实验设计的复审。

该条件已经在唯一独立审计中触发。以下第 8 节是触发前的历史复审；当前决定和下一科学试验以
`docs/phase4_kde_one_shot_audit_failure_and_science_first_simplification.md` 为准。

## 8. 本阶段科学价值复审

- `science_value_category`: `no_material_progress`
- `evidence`: `177 passed` 没有产生任何真实预测成绩；同一 receipt 可对应相反 gate，且置乱
  来源仍未形成封闭调用链，当前 builder 不能可靠回答最终科学问题。
- `decision`: `stop_generic_builder_and_adjust_to_one_shot`
- `next_scientific_test`: 只实现上述单入口、单 whole-run record 的最小纯合成路径，并进行一次
  独立集成审计；通过前不打开真实输入。
- `stop_condition`: 新路径仍需外部 draw/mapping/observed/gate/callback/expected receipt，无法把
  主成绩和全部复本封入同一记录，或再次发现 foundational P0 时，停止 Stage 4A 实现并复审科学
  路线。

该调整仍不是 `direct_improvement`。它的唯一合理价值，是把下一段代码直接压缩到下面这个问题：

> 在相同背景、支持域和 600,000 平方公里报警面积下，异常信息是否使独立物理区域召回稳定提高
> 至少 5 个百分点，并同时优于覆盖控制、时间置乱和空间置乱？

本合同最终没有通过，所以上述 code seal 和唯一真实 Stage 4A attempt 均未获授权、均未执行。禁止
沿本合同继续；当前下一步是新文档规定的 manifest/schema-only、target-blind 预检。
