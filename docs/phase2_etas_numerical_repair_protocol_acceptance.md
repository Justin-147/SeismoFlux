# 阶段 2 ETAS 数值修复协议验收记录

## 验收对象

- 工作线：阶段 2 ETAS 数值修复（`0.2.2`）
- 原始协议标签：`v0.2.2-background-etas-repair-protocol`（保持不可变）
- R1 勘误标签：`v0.2.2-background-etas-repair-protocol-r1`（保持不可变）
- 当前勘误标签：`v0.2.2-background-etas-repair-protocol-r2`
- 唯一实施蓝图：`SEISMOFLUX_IMPLEMENTATION_HANDOFF.md`
- 当前勘误分支：`codex/stage2-etas-repair-protocol-r2`；后续实现主分支：`codex/stage2-etas-numerical-repair`
- 原始协议比较基线：`dae6403`
- R1 勘误比较基线：`b7c70aced16a6bd57bf8f86f2680687e36b7710d`
- R2 勘误比较基线：`da916454c908e0cbe4a7526f56a8f837331a3c7c`
- 状态：通过（本地协议工程验收；提交、推送和远端标签核验属于发布闭环）

本次验收只冻结目标盲修复设计、25 个既有优化初值、输入/执行封印、调用回执、适配边界和停止条件。它不实施修复、不生成真实 fit-only 输入包、不运行真实 ETAS 资格拟合，也不创建新的阶段 4 执行修订。

## R2 勘误边界

R1 远端冻结后，Stage 2R-A 实现审计发现资格执行只声明“closing 前使用 Git 忽略 staging”，却没有冻结 qualification attempt 的 staged root、common/evaluable 逐文件 staged→final 路径、closing 后完整 qualification manifest 的独立 staged 路径及公开复制失败后的回滚/重试语义。R2 只关闭该 publication 工程歧义：

- staged root 恰好为既有 fit-input 本地根派生的 `data/processed/stage2R/etas_numerical_repair_fit_input/attempts/{attempt_id}/staged_public`，必须 Git 忽略、attempt 独占、原先不存在并新建；`attempt_id` 必须完整匹配安全单一 ASCII 组件 `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`，并与既有同 attempt fit-input 目录大小写精确一致；
- pre-closing identity 只含 7 个 common staged 文件；`evaluable` 另含 2 个参数文件，`not_evaluable` 不得存在参数目录；
- pre-closing ordered path→SHA map 的 key 精确等于相应 final path，value 精确等于同 mapping staged path 的完整 reopened bytes SHA；size、完整 bytes 和声明 Schema/可视化合同都须复验后再重算 aggregate；
- qualification closing seal 和 closing 后完整 qualification manifest 各有独立 staged 路径，二者都不得进入 pre-closing identity，避免哈希环；
- common 9 个最终文件和 evaluable 额外 2 个文件都有逐项精确 `{staged_path, final_path}` 映射；每个 staged 路径都必须等于 staged root 加冻结 final repository-relative path；
- 所有路径拒绝 symlink、junction、mount/reparse、UNC、drive、`..`、分隔符/大小写别名及根逃逸；所有 staged/final 文件只能独占新建、flush/fsync、no-clobber install、reopen 并逐字节核验；
- repair code tag 中全部 final 路径必须不存在，物化前仓库重新 clean 且 HEAD/upstream 不变，成功后这些路径只能是 Git `A`，禁止覆盖、删除、rename 或其他修改；
- 公开复制失败只逆序清除本次调用新建、仍与同 attempt staged bytes 完全相同且可安全重开的 final，保留全部 staging、closing、manifest 与失败证据；
- `not_evaluable` 物化顺序固定为 common 7→closing→manifest，`evaluable` 固定为 common 7→parameter 2→closing→manifest；rollback 只能是已创建前缀的精确逆序；
- 每次 post-closing failure 都追加严格的 attempt-local、六位连续序号、own-SHA 链 publication failure receipt，绑定 closing、manifest staging state、按状态可空的 manifest SHA、完整 staged size/SHA map、创建/回滚路径、仓库身份和 retry eligibility；manifest 尚未 reopened-valid 的 construction/temp/install/reopen/schema-byte-cross-file failure 一律 `retry_eligible=false`；
- post-closing publication failure 只有在 manifest 已 reopened-valid、完整回滚、仓库恢复 clean、HEAD/upstream 未变、final 全空且 staged 9(+2)、closing、manifest 字节和哈希全部未变时，才允许同一 attempt 仅重试 byte-exact materialization；不得重跑 fit、重算/替换 staged payload、改 closing/manifest 或另选结果，否则该 attempt 永久不得发布。

R2 不改变 R1 的 signed `row/column`，也不改变事件、快照、几何、网格、KDE、初值、objective、优化器、数值门限、源访问、目标盲边界或 adapter 合同。该缺口仍在任何真实 fit 源再次 open/stat/hash/query 或 bundle inspection 之前发现；原始和 R1 标签均不移动、不覆盖、不删除。

## R1 勘误边界（历史，继续有效）

代码标签前审计发现，原始协议在 `scientific_fit_input_record_schemas.integer_encoding` 中把所有整数统写为非负，但既有冻结网格的原点固定索引公式、带符号 `cell_id` 模板和 `grid.py` 均明确允许 `row/column` 为有符号整数；中国大陆等面积投影中位于中央经线西侧的固定格不能用非负列号无损表示。R1 仅作以下字段级澄清：

- `ordered_quadrature_containers.cells.row` 与 `.column` 为严格 Python `int` 的有符号 base-10 整数，bool 必须拒绝；
- 计数、`start_index`、优化器迭代上限和其余整数仍严格非负；
- `cell_id/row/column` 必须逐格匹配既有冻结网格，不允许移位、取绝对值、重编号或改网格；
- 不改变事件集合、坐标、几何、KDE、初值、objective、优化器、门限、目标盲边界或任何科学拟合规则。

该冲突在任何真实 fit 源再次 open/stat/hash/query 或 bundle inspection 之前发现并修订；R1 验收期间真实 fit 源、阶段 4 正式目标和阶段 9 锁定测试均保持未访问。原始协议标签不移动、不覆盖、不删除；R1 已由提交 `da916454c908e0cbe4a7526f56a8f837331a3c7c` 和远端 annotated tag 冻结。

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

### R2 publication-path 勘误复验

| 项目 | R2 结果 |
|---|---|
| 协议定向测试 | 首轮通过；审计修复并回填最终状态后的验收运行：`25 passed in 27.47s`；文档回填后再次完整复验：`25 passed in 24.98s` |
| 协议、固定网格与 local-support manifest 联合回归 | `57 passed in 24.68s` |
| 新增路径闭包预复验 | 最终文档状态写入前其余 `24 passed, 1 deselected`；新增 staging/YAML/dotted-ref 三项单独复验 `3 passed` |
| `.gitignore` / R1 tree 可执行闭包 | 对全部 11 个格式化 staged path 的 `git check-ignore --no-index` 均返回 ignored，对全部 11 个 final path 均返回 not ignored；`git ls-tree` 证明全部 final path 在 R1 基底提交中不存在 |
| 隔离 public worktree 全量非目标测试 | `1213 passed, 2 skipped in 363.04s`；`failures=0`、`errors=0` |
| R2 JUnit | `data/interim/protocol-r2-full-nontarget.junit.xml` |
| R2 JUnit 字节身份 | size `195115`；SHA-256 `ec40ea946c976d362b3b1313fc31c3f318eb4b13dc2717fcd63bf3e1754e9aaf` |
| Ruff | 协议测试 `ruff check` 与 `ruff format --check` 通过 |
| Mypy | `Success: no issues found in 1 source file` |
| Git 空白检查 | `git diff --check` 通过 |
| 独立只读审计 | 首轮 `P0=1, P1=4, P2=1`；修复 manifest failure 窗口后第三轮终审 `P0=0, P1=0, P2=0`，`FINAL_SCHEMA_AUDIT=PASS` |

R2 没有重跑真实拟合；全量非目标回归仅运行仓库 `tests` 中的合成、单元、泄漏守卫、回归和包契约。两个 skip 仍来自隔离 public worktree 有意不分发的本地受限 Stage 3 feature store / Stage 4 spatial artifacts；没有读取、复制或检查这些受限工件内容。最终只读审计还对 R1→R2 YAML 做 normalized deep compare，确认改动局限于 revision/tag metadata、qualification publication path/rollback/retry 工程闭包、相应文档和机械协议测试，且精确只修改五个允许文件。

全量运行启动前 5 次整机 CPU 抽样均值为 `45.19%`、峰值为 `47.85%`；执行进程使用 affinity `0x3F0000`（6 个逻辑核）、`BelowNormal` 优先级，并强制 OMP/OpenBLAS/MKL/NumExpr/BLIS/VECLIB 单线程。运行中复核均值约 `48.75%`、峰值约 `51.66%`，未占满所有核心。

### R1 勘误复验

| 项目 | R1 结果 |
|---|---|
| 协议定向测试 | `24 passed` |
| 协议、固定网格与 local-support manifest 联合回归 | `56 passed in 24.34s` |
| 隔离 public worktree 全量非目标测试 | `1212 passed, 2 skipped in 351.64s`；`failures=0`、`errors=0` |
| R1 JUnit | `data/interim/protocol-r1-full-nontarget.junit.xml` |
| R1 JUnit 字节身份 | size `194932`；SHA-256 `d463853cf3b010e79ac9ef3646dfc0cb6332128e1405cd1fd0b206cf6702a259` |
| Ruff | R1 协议测试 `ruff check` 与 `ruff format --check` 通过 |
| Mypy | R1 协议测试：`Success: no issues found in 1 source file` |
| Git 空白检查 | `git diff --check` 通过 |

两个 skip 都来自 clean public worktree 有意不分发的本地受限 Stage 3 feature store / Stage 4 spatial artifacts；没有读取、复制或检查这些受限工件内容，也没有用它们验证 R1。新增一项协议测试后，本次共收集 `1214` 项：`1212` 项执行并通过、`2` 项按既有公开仓库保护条件跳过。第一次隔离全量 runner 因未把仓库根加入 `sys.path`，在收集期以 `scripts` import error 退出且没有执行测试；修正 runner 后才得到上述有效结果。

R1 独立只读审计对原协议与勘误 YAML 做深比较，确认除 revision metadata、R1 tag/comparison base 和 quadrature `row/column` integer encoding 外，其余协议语义完全相同；审计提出的 package 清单、证据状态、comparison-base、revision-reason 与 bool 拒绝回归均已补齐并复跑。

### 原始协议冻结证据（历史，不替代 R1 复验）

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

因此，本次勘误即使通过，也只表示阶段 2 ETAS 数值修复的**协议工程路径闭包**满足本地验收，不表示 ETAS 已稳定拟合，更不表示预测效果或科学门控成功。原始和 R1 annotated tag 保持不动；R2 完成定向/回归/静态检查与独立审计后，下一合法动作才是提交 R2 协议勘误、推送并创建 annotated tag `v0.2.2-background-etas-repair-protocol-r2`。只有远端 R2 标签解析到该勘误提交后，才允许恢复 1 ULP 修复代码子阶段。

协议包自身包含本验收文件，因此不能在文件内部嵌入最终提交、标签或自身 blob SHA 而制造自引用。六个精确路径已由测试锁定，最终 Git blob/tree/commit 身份由提交与远端 annotated tag 在外层冻结并可重复查询。
