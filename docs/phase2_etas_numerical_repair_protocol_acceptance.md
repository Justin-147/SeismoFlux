# 阶段 2 ETAS 数值修复协议验收记录

## 验收对象

- 工作线：阶段 2 ETAS 数值修复（`0.2.2`）
- 协议标签：`v0.2.2-background-etas-repair-protocol`
- 唯一实施蓝图：`SEISMOFLUX_IMPLEMENTATION_HANDOFF.md`
- 当前分支：`codex/stage2-etas-numerical-repair`
- 比较基线：`dae6403`
- 状态：通过（本地协议工程验收；提交、推送和远端标签核验属于发布闭环）

本次验收只冻结目标盲修复设计、25 个既有优化初值、输入/执行封印、调用回执、适配边界和停止条件。它不实施修复、不生成真实 fit-only 输入包、不运行真实 ETAS 资格拟合，也不创建新的阶段 4 执行修订。

## 协议包边界

本次协议标签的规范 protocol package 恰好包含以下六个路径，缺失、额外、拆分提交或标签 blob 不一致均不得验收：

- `.gitignore`
- `configs/background_etas_numerical_repair.yaml`
- `data/manifests/etas_numerical_repair_start_manifest.json`
- `docs/background_etas_numerical_repair_protocol.md`
- `docs/phase2_etas_numerical_repair_protocol_acceptance.md`
- `tests/unit/test_background_etas_numerical_repair_protocol.py`

`tests/unit/test_stage4_anomaly_increment_runtime.py` 仅含严格 mypy 所需的机械类型标注，不改变测试逻辑；重启交接文档记录现场，但二者都不是上述六文件 protocol package 的组成部分。

## 蓝图边界

- Stage 4 formal target consumer 调用：`0`
- Stage 4 assessment 行物化：`0`
- 正式评分与信息增益计算：`0`
- `Score ID` 创建：`0`
- 阶段 9 锁定测试：未运行
- 真实 fit-input bundle：未生成
- 资格参数、数值负结果、adapter 工件和全局 comparator receipt：均未生成
- 历史 `0.2.0`、`0.2.1` 结果、注册表和标签：未覆盖、未删除、未改判

协议草拟早期曾对预登记阶段 2 目录源执行一次只读文件级 SHA-256 核验，用于确认交接文档中的冻结文件身份；没有解析或返回数据行，没有构造 fit/assessment cohort，没有调用 Stage 4 正式消费者。当前冻结协议明确禁止在修复代码标签远端核验前再次打开、stat、hash、查询或检查任何真实 fit 源/输入包；后续全部访问必须进入两阶段 append-only 源访问账本。

## 冻结输入与方法

- 五个主快照：`fold_1`、`fold_2`、`fold_3`、`fold_4`、`final_validation`
- 每快照起点：索引 `0..4`，共 `25`
- seed 协议：沿用父版本 `0.2.1`，禁止用 `0.2.2` 重新抽样
- start manifest 文件 SHA-256：`e674ca7e02f1da9fc5afd87812e34e2ae9447f47e496d3f772277b6cd68ef05e`
- start vector payload SHA-256：`29275083fb4a6a1209ba2c6e0e1e6033ab5bb51dab7b2ffba689b04784d3acd5`
- 唯一科学修复：`ETASParameterBounds.from_transformed` 的精确 transformed 端点映射
- 禁止：放宽边界、容差 contains、越界 clip、`nextafter` 扩域、更换初值/优化器、修改 objective 或父选择路径

## 新增的可审计闭包

- `scientific_fit_input_sha256` 与父提交全值重放采用完全相同的精确字段集合。
- 修复代码运行必须绑定 Python/NumPy/SciPy/Shapely、实际 L-BFGS-B 文件和 GEOS 原生依赖的 runtime code seal。
- baseline、qualification 和公开 runtime Schema 必须同时保存完整发行包/运行文件/callable/Shapely-GEOS map 及可从相邻完整对象重算的 sibling SHA；固定双坐标 `Point` 构造只归 synthetic warmup，但共享 `numpy.array` 与每个 multithreading wrapper 的 `numpy.ndarray` class 检查必须同时标记 three-grid；生产闭包只把已有 Point 的 `x/y` 与 `equals` 作为 Shapely 根。每条链通过有序 record-ID/SHA 前像、wrapper、`__wrapped__`、executed/inert 类型化 closure cell 和 native ufunc 分层闭合。
- 每快照直接调用冻结 `fit_etas` 一次；透明 wrapper 仅观察并原样委托 SciPy `minimize`，25 次调用必须全部正常返回。
- 每个快照各有 1 个 opening call receipt、5 个 invocation receipt、1 个 closing call receipt、5 行诊断、可空 three-grid evidence、snapshot gate 和 fit-attempt snapshot；五快照合计 5/25/5/25 条相应回执，采用冻结无环哈希构造顺序。
- 初始 objective 跳过调用、调用异常、缺失/重复调用、代码/闭包/参数漂移或 wrapper 未恢复均为 `invalid_execution`，不能发布资格结果。
- 正资格参数目录采用严格文件和递归字段 allowlist；`not_evaluable` 与 `invalid_execution` 分支禁止参数目录存在。
- snapshot gate 按 count → gradient → 两个 spread 兄弟门 → Hessian minimum → Hessian condition → branching → 两个 grid 兄弟门唯一短路；未满足前置条件的门必须为 `not_run_upstream_gate`、metric 为 null 且不产生 failure code。
- 只有正资格后才允许冻结 adapter code；其完整 allowlist 代码 payload 和独立 adapter runtime preflight/seal 必须在 opening 前重算，并封印与资格执行相同的完整 `isolated_launcher_identity`；七字段 Python runtime identity 保持 nested object，不得与 hex scalar schema 混用。固定 geometry warmup 按 buffer→normalize→WKB round-trip→equals→covers 执行，exact input/output 与两个 sibling SHA 均须独立重算。六个 pre-closing staged 对象固定为 opening seal、artifact、global receipt、报告、静态 SVG、离线 HTML，closing seal 是第七个 staged 文件；七个非-publication 文件必须在 `completed` 前物化并 reopen 逐字节验证，publication manifest 在 `completed` 后原子写入。
- typed parent/retrospective target、128 个真正前瞻 ETAS 模拟批次、catalog/projection receipt 和全局 `FrozenETASComparatorReceipt` 均冻结了无文件 I/O、无目标回填和无哈希环的数据流。

## 资源约束

- SciPy L-BFGS-B 使用 CPU float64；GPU 不用于本次数值修复，以保持冻结实现一致。
- BLAS/OMP/MKL/NumExpr 单线程。
- 最多使用 6 个逻辑处理器，优先 affinity `0x3F0000`，进程优先级 `BelowNormal`。
- 至少保留 2 个物理核心；整机 CPU 使用率不低于 70% 时不启动重型拟合。

## 验收证据

### 测试与静态检查

| 项目 | 最终结果 |
|---|---|
| 协议定向测试 | `23 passed` |
| 旧 ETAS、受影响 Stage 4 守卫与协议联合回归 | `118 passed in 38.85s` |
| 全量非目标测试 | `1213 passed in 329.82s`；`failures=0`、`errors=0`、`skipped=0` |
| 全量 JUnit | `data/interim/stage2/etas_numerical_repair/runtime_logs/protocol-freeze-final-full-nontarget-20260719-rerun1.junit.xml` |
| JUnit 字节身份 | size `193506`；SHA-256 `406edffc53ad4f49a83638611d9aa376026882c89e67d87adcc5712da9dc2916` |
| Ruff | `ruff check .` 通过；`ruff format --check .` 为 `224 files already formatted` |
| Mypy | `mypy src tests`：`Success: no issues found in 216 source files` |
| Git 空白检查 | `git diff --check` 通过 |

全量测试只运行仓库 `tests` 中的合成、单元、泄漏守卫、回归和包契约；未调用正式目标消费者，未物化 assessment 行，也未运行阶段 9 锁定测试。启动前整机 CPU 抽样为 `12.67%`，执行进程使用 affinity `0x3F0000`（逻辑核 16–21）、`BelowNormal` 优先级，并强制 OMP/OpenBLAS/MKL/NumExpr/BLIS/VECLIB 单线程。

### 独立只读审计

| 审计线 | P0 | P1 | P2 | 结论 |
|---|---:|---:|---:|---|
| Runtime/Shapely 闭包与四链前像 | 0 | 0 | 0 | 通过 |
| Adapter runtime、九门真值表与七文件 DAG | 0 | 0 | 0 | 通过 |
| 六文件 protocol package、引用和三文档一致性 | 0 | 0 | 0 | 通过 |

整包终审还确认：13 个 YAML 均通过重复键拒绝加载器；协议中的 209 个单数/复数 `*_ref(s)` 全部可达；11 个公开输入绑定逐字节匹配；start manifest 恰含 5 个快照、25 个起点，文件 SHA 和向量 payload SHA 与本文件冻结值一致。

### 禁止项与结果解释

- Stage 4 formal target consumer 调用：`0`
- Stage 4 assessment 行物化：`0`
- 正式评分与信息增益计算：`0`
- `Score ID` 创建：`0`
- 阶段 9 锁定测试：未运行
- 真实 fit-input bundle、资格结果、参数工件和 adapter 工件：均未生成

因此，本次通过只表示阶段 2 ETAS 数值修复的**协议工程冻结**满足本地验收，不表示 ETAS 已稳定拟合，更不表示预测效果或科学门控成功。下一合法动作是提交六文件协议包及同批机械修复/交接记录，推送分支并创建 annotated tag `v0.2.2-background-etas-repair-protocol`；只有远端标签解析到该提交后，才允许进入 1 ULP 修复代码子阶段。

协议包自身包含本验收文件，因此不能在文件内部嵌入最终提交、标签或自身 blob SHA 而制造自引用。六个精确路径已由测试锁定，最终 Git blob/tree/commit 身份由提交与远端 annotated tag 在外层冻结并可重复查询。
