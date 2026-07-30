# Stage 4A 目标盲输入与真实置乱预检

- 日期：2026-07-30
- 唯一蓝图：`SEISMOFLUX_IMPLEMENTATION_HANDOFF.md`
- 当前协议：`stage4-kde-development-v1` / `0.4.3`
- 预检范围：只读 tracked manifest、config、schema 构建代码和测试
- `data/processed` 真实字节读取：0
- 开发目标读取：0
- 独立验证读取：0
- 锁定测试读取：0
- 唯一开发科学 attempt 消耗：0
- 结论：核心 Stage 3 输入和真实置乱原语可用，但当前执行合同缺少两个目标盲适配定义，尚不能
  code freeze 或打开真实目标

## 1. 通俗结论

冻结输入的身份声明完整，表结构可由 code + dictionary 重建并与 registry 对上；真实文件是否仍
存在、字节是否匹配尚未核验。程序已经有“真正打乱异常时间/位置并重新计算异常特征”的能力。

现在缺的不是新模型，而是两张准确的说明书：

1. 空间分层文件到底从哪个固定路径读取、每一列是什么；
2. 折叠清单里的训练/评估 ID 怎样唯一换算成起报时间和未来窗口。

这两项没写死就直接跑，可能把正确数据接错。所以下一步只补这两个目标盲接口，不再增加封条系统，
也不改变任何科学门。

## 2. 已确认 READY 的部分

### 2.1 Stage 3 冻结表身份和物理 schema

公开 registry 冻结了：

- `anomaly_state_history.parquet`：79 列；
- `anomaly_feature_store.parquet`：1,637 列；
- 两文件的 file/content/schema SHA-256、行数和 row group 数；
- 冻结特征字典及其列、类型、缺失语义和因果来源。

在不读取 Parquet 字节的前提下，使用当前 Stage 3 code、冻结 dictionary、`storage.py` 固定 metadata
和 sort keys 重建两个 Arrow schema，结果为：

| 数据集 | registry schema SHA-256 | 重建 SHA-256 | 结论 |
|---|---|---|---|
| state history | `b6986a78912d34ed90e83a6d72fc3dfbf8b4c0edf59217f969ae5e7fb33152d5` | 相同 | `READY` |
| feature store | `0d7a9eab211324fcf3d03f6a55ef0f5136a6132f0ce0c56a72cb0a92912e6808` | 相同 | `READY` |

真实 placebo rebuild 要求的 79 个 feature-store 列全部存在；`coverage_only`、`snapshot`、
`dynamic` 三套冻结 design contract 的 source columns 也全部存在，缺失数均为 0。

### 2.2 四个模型的固定特征

- `B0`：KDE，无异常增量特征；
- `C0`：9 个覆盖控制逻辑特征；
- `B1`：17 个覆盖 + 单期异常逻辑特征；
- `B2`：22 个覆盖 + 单期异常 + 动态轨迹逻辑特征。

特征列、变换、缺失处理、截断、标准化和 penalty 已在
`anomaly_increment_r2_feature_set.json` 中冻结，并由 `0.4.3` 精确 allowlist 继承。不得搜索或更换
特征。

### 2.3 三个滚动折和时间置乱池

`joint_macro_rolling_folds` 已给出三个折的：

- `fold_index`、`fit_scope_id` 和训练截止时间；
- 7 天训练 exposure IDs；
- 7/30/90 天评估 exposure IDs；
- fit/assessment 两个互不交叉的 time-permutation issue pools。

三折、三个窗口和时间池本身为 `READY`。

### 2.4 25 km 背景质量和 600,000 平方公里前缀

该部分应在单目录会话中从冻结 `fold_4`、75 km KDE 运行时重物化，不要求预存成绩文件：

- 25 km cell ID、行列和裁剪面积来自冻结 local-support manifest；
- cell mass 由已允许的 Poisson/KDE 原语生成；
- 运行时必须核对有序 cell-mass identity；
- 报警区按强度降序和固定 tie-break 取完整 cell 前缀，累计面积不得超过
  600,000 平方公里，不得选部分 cell 或跳过首个超预算 cell。

因此输入合同为 `READY`，实际组装代码尚待后续最小 runner 实现。

### 2.5 真实时间和空间置乱原语

`rebuild_time_placebo_features` 为 `READY`：

- 保留 recipient 的背景、网格和覆盖控制；
- 在各折 fit/assessment 池内移动快照值及其缺失性；
- 不复制 donor 预存轨迹，而是在 recipient 时间轴上从伪历史重新计算动态轨迹。

`rebuild_space_placebo_features` 为 `READY`：

- 只在同一 `issue × frozen construction stratum` 内交换 eligible entity 坐标；
- 不改变非坐标属性、ineligible entity、背景或覆盖控制；
- 重新计算 200 km 单期异常和动态轨迹。

两个函数都不接收地震目录、目标、成绩或文件路径。它们足以生成真实 `B1/B2` placebo 特征，但当前
runner 尚未把这些输出接到同一重训和评分路径。

## 3. 当前阻断项

### 3.1 `MISSING`：空间受限工件缺少当前授权的 path/schema

`anomaly_increment_r2_spatial_strata.json` 只公开四个受限工件的 byte count、media type 和 SHA-256：

- cell mapping；
- entity mapping；
- connectors；
- zone geometry。

`0.4.3` 允许继承这些摘要，却没有重新声明当前 Stage 4A 可用的固定路径和列 schema。旧 R2 config
曾声明路径，但旧执行授权明确禁止复用，不能把旧路径默认为当前授权。

空间 placebo 至少需要 entity mapping 的：

`state_id, anomaly_id, issue_time_utc, construction_stratum_id,
coordinate_pair_sha256, outside_study_area`

区域稳健性还需要 cell mapping 与 zone geometry。没有当前目标盲定位合同就必须失败关闭。

### 3.2 `AMBIGUOUS`：exposure ID 到时间窗的解析合同未显式冻结

当前 allowlist 允许 `joint_macro_rolling_folds`，其中已有全部 exposure ID，但旧 manifest 的
`horizons` 明细不在允许范围。必须在当前合同内明确定义：

- `development-h007/h030/h090-YYYY-MM-DD` 的解析；
- 对应 `anomaly-issue-YYYY-MM-DD`；
- `Asia/Shanghai` 本地起报日到 UTC 起报时刻；
- 目标窗严格为 `(T, T+h]`；
- fit 只用 7 天 exposure，且训练目标结束严格早于 assessment 开始。

在这些规则冻结前，不得由 runner 自行猜测日期或继承旧 R2 exposure 对象。

### 3.3 `MISSING`：合法实际 assembly 尚未实现

当前未提交 runner 是失败的纯合成实现，不能使用。合法实际路径尚缺：

`冻结表/折/背景 → B0/C0/B1/B2 矩阵 → 同一预处理与模型拟合 → 目标盲格网预测冻结 →
真实 target 只评分 → 实际 time/space rebuild → 同路重训/冻结/评分`

这是下一代码阶段的最小范围，不得加入新模型、特征、面积、折、统计门或 provenance/receipt 层。

## 4. 唯一允许的目标盲可执行性修订

下一协议修订应为 `0.4.4`，标签为 `v0.3.4-kde-anomaly-increment-protocol`。它只修复以下执行接口，
科学问题和门全部保持 `0.4.3` 不变：

1. 在唯一蓝图、config 和 input contract 中重新声明四个 spatial local artifacts 的固定路径、
   现有 manifest SHA-256、精确列 schema 和“只在 code tag 后读取非目标输入”的边界；
2. 冻结 exposure ID parser 和本地日历到 UTC 的唯一映射；
3. 精确授权目标盲 loader 所需的 Stage 3 反序列化与 snapshot/grid 构造符号，或把同等逻辑封入
   一个 code-sealed 薄 adapter；不得调用旧 R2 orchestrator、gate、randomness 或 result identity；
4. 增加新 DTO 直接传入两个 rebuild 函数的纯合成集成测试；
5. 增加 `rebuild → 9/17/22 列组装 → 同一 ridge Poisson 重训 → assessment statistic` 的小型
   纯合成等价测试。

本修订只允许一轮。通过、提交、推送和协议 tag 完成前，仍不得打开任何真实输入字节；代码 tag 完成
后可核验非目标 Stage 3/空间工件的 path、file hash 和 schema。开发目标只允许在预测已经冻结的唯一
注册 attempt 内打开。

## 5. 不变的科学门

以下内容不得因本修订改变：

- 75 km KDE 背景；
- `B0/C0/B1/B2` 四成员；
- 三个滚动折、7/30/90 天；
- 600,000 平方公里主报警面积和严格召回 `+5 pp`；
- 2,000 次物理事件块 Bootstrap；
- 时间/空间各 1,000 次置乱；
- 信息增益、同时区间、`maxT`、跨折和跨区门；
- 唯一开发 attempt、独立验证和锁定测试隔离。

## 6. 硬停止条件

出现任一情况即停止当前异常路线并保留 KDE：

- 一轮 `0.4.4` 修订后仍无法唯一定位和解析空间工件或 exposure；
- 需要读取目标后才能决定路径、schema、日期、特征、分层或模型；
- 新 DTO 不能直接驱动实际 rebuild；
- observed 与 placebo 不能经过同一组装、预处理、重训、预测冻结和评分路径；
- 又开始增加新 provenance/receipt 层，却没有形成上述实际科学链；
- 下一次唯一纯合成端到端审计再次发现 foundational P0。

## 7. 即时科学价值复审

- `science_value_category`: `necessary_enabler`
- `evidence`: 本轮没有产生真实预测成绩；但已用可复算 hash 证明 Stage 3 冻结表结构和全部模型列
  合同一致，确认两个 rebuild 原语能真正重建 `B1/B2`，并把剩余阻断收敛为一个空间
  locator/schema 和一个 exposure parser，而不是继续泛化工程；真实文件存在性和字节仍待 code
  tag 后核验。
- `decision`: `one_bounded_target_blind_executability_amendment`
- `next_scientific_test`: 完成并独立验收 `0.4.4` 的两个目标盲接口与
  `rebuild → assemble → same-model refit → statistic` 纯合成等价链。
- `stop_condition`: 一轮修订后仍有新的基础输入/调用链缺口，或不能紧邻进入实际 Stage 4A
  runner 时，停止异常工程路线并保留 KDE。

本阶段仍不是 `direct_improvement`。只有唯一 Stage 4A 开发结果在相同 600,000 平方公里内稳定提高
召回、信息增益和置乱/跨折/跨区门，才算对最终科学目标形成直接推动。
