# 阶段 4A 目标盲可执行性修订

- 日期：2026-07-30
- 新协议版本：`0.4.3`
- 新协议/代码/结果标签：`v0.3.3-kde-anomaly-increment-{protocol,code,result}`
- 被替代版本：`0.4.2` / `v0.3.2-kde-anomaly-increment-*`
- 目标读取：0
- 正式评分：0
- 状态：目标盲协议修订；不得据此开始真实输入或目标读取

## 1. 为什么必须修订

`0.4.2` 把公开 `background_local_support_model_registry.json` 和历史声明的
`manifest.json`、`poisson_kde.json` 当成可以直接提供 Stage 4A 空间 KDE 的冻结 payload，同时
禁止从目录重建。审计确认公开 registry 只包含身份、选择和数值审计摘要；后两条历史路径在本
worktree 不存在，也不包含于运行时输入。它们只保留为“历史未跟踪声明”，不要求存在、不得加入
expected-input manifest、运行时不得读取。继续按旧合同实现只能制造一个无法执行的薄 adapter，
属于 P0 合同矛盾。

同一轮审计还确认：现有目标盲 seal 原语把 target-read ledger 实现为单个 canonical JSON
mapping 的 `0→1` 原子 compare-and-swap；`0.4.2` 却声明为 append-only `.jsonl`。为了避免再造
第二套账本工程，本修订把合同收敛到现有最小状态机，同时不改变一次 attempt 和一次逻辑读取上限。

旧 `0.4.2` 与 `v0.3.2-*` 只保留为历史目标盲证据，状态为
`historical_superseded_before_target_read`，不得作为当前执行授权，也不得覆盖或改写其历史验收。

## 2. 唯一允许的 KDE 重物化

公开注册表只作预期身份与审计摘要，不是验证 payload；历史未跟踪模型路径不是运行时输入。唯一注册
目录读取会话打开地震目录后，必须在内存中完成以下固定过程，并让同一个内存 catalog 继续供折内
rate-head 训练和目标评分；禁止第二次打开目录：

1. 读取并核验
   `data/manifests/background_local_support_manifest.json`，整文件 SHA-256 必须为
   `632278416dfc717dbcb9d2eae048a4f13cdf7737a31e6e5e704a9dd17d7cef8d`，
   `manifest_id` 必须为 `local-support-bundle-69cecbee9093a21d`；
2. 先用 `origin_time_utc` 和 `available_at` 均不晚于
   `2019-12-31T16:00:00Z`、位于研究区内的全部震级目录事件重建完整度与 `fold_4` 支持；在此步
   之前做 `M>=4.0` 预筛选明确禁止；
3. 将实际 `fold_4` manifest 与公开 bundle 的固定 cells 和 `fold_4` decision 完整展开后，按全部
   字段和有序 cells 逐项相等核验；随后才用 `historical_training_mask` 选择 `M>=4.0` 且属于重建
   保留支持域的 KDE 训练事件；
4. 只用已经选定的 75 km 带宽、`chunk_size=256` 重物化 Gaussian KDE；禁止重新选带宽、改变
   Mc、支持域、积分域或使用 `final_validation`；
5. 独立按上游 canonical payload schema 从运行时值重算 training evidence ID、parameter snapshot
   ID 和 compensator domain ID，并核验三网格全部字段；公开 registry 值只能作预期交叉核对，不能
   复制到 receipt。

必须逐项相等的身份是：

| 字段 | 冻结值 |
| --- | --- |
| training event count | `5237` |
| training duration days | `18261.666666666668` |
| rate per day | `0.2867755772565483` |
| training evidence ID | `ed59aa557a816e43dd0af8f321ca689cee3ba6deac81eafa2b9e66f5b208af29` |
| parameter snapshot ID | `83a0c60d4b62ba6a6e849ac2d5f430001d054b7aec3af40f76193180a18bf4c5` |
| 75 km normalization mass | `0.9180536964403374` |
| normalization cell-mass sum | `1.0` |
| 25→12.5 km coarse total | `1.000341874485772` |
| support ID | `local-support-788851371baf0e3b` |
| compensator domain ID | `33a9095704a09f8661c48061f9febec0342a9db671d6384fe7dcbeb3cf3aed55` |
| common Mc | `4.0` |
| supported area fraction | `1.0` |
| supported area km² | `9415305.754432771` |
| 25 km cell count | `15697` |

截止日之后新增、删除或移动的目录文件/行不得改变上述背景训练子集身份；任何历史入选行、可用时间、
震级、坐标或支持域归属变化都必须失败关闭。不得通过枚举注册目录或运行旧背景 orchestrator
恢复模型。

运行时还必须在背景进入训练或评分前生成并绑定 `historical_training_rows_sha256`（按源目录行序
封印 event ID、origin/available 时间、坐标、震级与支持归属）和
`ordered_25km_cell_id_mass_sha256`（封印有序 25 km cell ID 与重物化质量）。两者只能在 code
tag 后的单次目录会话中生成，本次目标盲协议修订不得读取真实输入来预填。

## 3. 最小代码边界

新 allowlist 只增加重物化所需的目标盲背景原语：

- `background.catalog.load_study_area/load_earthquake_catalog`；
- `background.workflow.catalog_completeness_events/historical_training_mask`；
- `background.local_support.build_local_support_snapshot/build_local_support_manifest/`
  `LocalSupportCellLocator`；
- `background.local_support_manifest.load_background_local_support_manifest/`
  `validate_background_local_support_study_area`；
- `background.poisson.SpatialQuadrature/fit_spatial_poisson_family/`
  `evaluate_spatial_poisson_family_cell_masses`；
- `background.grid.build_equal_area_grid_family/diagnose_three_grid_convergence`；
- `background.artifacts.canonical_json_bytes`。

正式运行只能由上述原语构造 `fold_4`，不得导入或调用完整
`background.local_support_runtime.build_local_support_runtime`；后者仅可在纯合成等价性测试中作为
逐字段参照。

Stage 3 置乱只允许直接调用整文件哈希封印的
`anomaly_increment.placebo_features.rebuild_time_placebo_features` 和
`rebuild_space_placebo_features`。其直接传递依赖只获得 import-only 身份，不获得 API 执行授权。
旧 R2 scoring/formal orchestrator、`placebo_source.py`、`placebo_runtime.py`、
`select_kde_bandwidth`、旧 gate、旧随机流和旧 adoption semantics 继续禁止。`placebo.py` 仅保留为
哈希封印的 import-only 传递依赖；直接调用其 `TimeBijection`、`SpaceBijection` 和 build 函数禁止。
正式置乱映射由 Stage 4A namespace 的 `TimeMappingDTO`、`SpaceMappingDTO` 承担，只满足
`placebo_features` 的结构合同并从 `0.4.3` RNG 上下文构造。直接 allowlist 的完整内部静态 import
closure（含 package `__init__` 与 contracts）必须逐文件哈希封印；任何闭包外内部 import 失败关闭。

## 4. 一次性目标读取状态机

本地账本路径改为 `target_read_ledger.json`。它是单个 canonical JSON mapping：

1. preflight 原子创建并哈希绑定 `logical_open_count=0` 的零态；
2. 先 CAS 注册相同协议、代码、输入、随机和 attempt 身份的唯一 attempt；
3. target adapter 打开前，CAS 必须再次核验 attempt ledger 已注册且身份相同；
4. target-read ledger 只能从零态原子转换到唯一打开态，记录 `entry_hash`、
   `previous_zero_ledger_hash` 和 `logical_open_count=1`；
5. 打开态不可改写、追加第二次打开或回退到零态。

目录打开一次后，KDE 重物化、rate-head 训练和评分共享同一个内存 catalog。科学 attempt 数、
逻辑目标读取数和恢复限制均未改变。

## 5. 不变的科学合同

科学问题、`B0/C0/B1/B2`、三个滚动折 × 7/30/90 天、600,000 平方公里、2,000 次事件块
Bootstrap、1,000 次时间置乱、1,000 次空间置乱、候选族 `maxT`、家族同时区间、固定面积
`+5 pp` 主门、跨折/跨区门、单 attempt、独立验证与锁定测试禁读限制全部不变。本修订没有增加
候选、参数搜索、面积预算、目标读取或成绩。

## 6. 可执行性审计的科学价值复审

- `science_value_category`: `no_material_progress`
- `evidence`: 审计没有产生任何预测成绩；它只证明 `0.4.2` 的公开摘要/禁止重建合同和 `.jsonl`
  ledger 合同不可执行，继续实现会增加不能转化为科学检验的工程。
- `decision`: `adjust`
- `next_scientific_test`: 完成 `0.4.3` 目标盲协议、测试、提交、推送和协议标签后，只实现受精确
  allowlist 约束的薄 KDE 重物化与单会话 adapter，再完成 code freeze；仍不得提前打开真实输入。
- `stop_condition`: 若薄实现需要重新选带宽、改变 Mc/支持域/final_validation、第二次打开目录、
  复活旧 R2 orchestrator/placebo/runtime/随机语义，或不能逐项复现冻结背景身份，则停止工程扩张，
  不进入开发评分。

这一定性要求立即调整合同，防止继续无效工程；它不是 Stage 4A 预测效果结论。

## 7. 独立复审通过后的有限阶段价值

- `science_value_category`: `necessary_enabler`
- `evidence`: `0.4.3` 只在目标盲状态消除了两项执行矛盾，使下一步仍可保持为一次受限的薄 code
  freeze；工程与科学两路独立终审均为 `PASS`，`P0/P1/P2 = 0/0/0`；尚无直接预测效果证据。
- `decision`: `continue_to_thin_code_freeze`
- `next_scientific_test`: 远端核验 `v0.3.3-kde-anomaly-increment-protocol` 后，冻结薄实现和
  `v0.3.3-kde-anomaly-increment-code`；随后才允许唯一 `S4-KDE-DEV` attempt。
- `stop_condition`: code freeze 不能严格保持单次 catalog 会话、固定 75 km 重物化身份、精确
  allowlist 和不变科学门时停止，不以新增工程绕过。

该 `necessary_enabler` 结论只表示紧邻的一次 Stage 4A 科学检验已经重新具备可执行路径；上一节
“可执行性审计没有产生预测进展”的 `no_material_progress + adjust` 结论仍保持成立，不能把协议
通过、工程审计或代码冻结冒充预测效果提升。

## 8. 本地验收证据

本修订只执行目标盲协议测试，没有打开或枚举 `data/processed`、开发目标、独立验证目标或锁定测试：

```powershell
D:\AIPred\SeismoFlux\.venv\Scripts\python.exe -m pytest `
  tests/unit/test_stage4_kde_development_protocol.py `
  tests/unit/test_stage4_anomaly_increment_protocol.py `
  -p no:cacheprovider -q `
  --junitxml=data/interim/stage4/kde_dev_protocol_amendment_full.junit.xml
```

- 结果：`24 passed, 1 skipped in 5.25s`；
- 唯一 skip：旧 R2 的四个本地 restricted 空间工件在本 worktree 不可用，不影响本修订的目标盲
  合同；
- JUnit SHA-256：
  `709e1904a70145f27eb194f8e971eeeeda50f2a98559a7d2799c4a6b965a3d0f`；
- Ruff check：通过；
- Ruff format check：通过；
- strict mypy：通过；
- `git diff --check`：提交前必须再次通过；
- 外部闭环：尚待 protocol commit、push、annotated tag 与远端 peeled commit 回读，完成前不得进入
  源代码提交 `S`。
