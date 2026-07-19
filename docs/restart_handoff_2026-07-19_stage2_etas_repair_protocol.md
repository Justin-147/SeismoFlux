# SeismoFlux 续接交接：阶段 2 ETAS 数值修复协议冻结

更新时间：2026-07-19（Asia/Shanghai）

## R1 协议勘误续接覆盖（当前有效）

原始协议标签 `v0.2.2-background-etas-repair-protocol` 已提交、推送并完成远端核验，保持不可变。其后在修复代码标签前、且未再次访问任何真实 fit 源或阶段 4 正式目标时，审计发现科学输入的全局“非负整数”声明与既有有符号固定网格 `row/column` 契约冲突。当前插入一个最小协议勘误门控：

- 勘误分支：`codex/stage2-etas-repair-protocol-r1`；实现主分支仍为 `codex/stage2-etas-numerical-repair`；
- 勘误基线：`b7c70aced16a6bd57bf8f86f2680687e36b7710d`；
- 待冻结 annotated tag：`v0.2.2-background-etas-repair-protocol-r1`；
- 唯一语义改动：`ordered_quadrature_containers.cells.row/column` 明确为拒绝 bool 的严格有符号 Python `int`；所有其他整数继续严格非负；
- 禁止借勘误修改网格、事件、KDE、初值、优化器、objective、门限、目标边界或科学输入内容；
- 必须先完成 R1 定向/回归/静态验收、提交、推送与远端标签核验，再恢复 Stage 2R-A；
- Stage 2R-A 的在建代码保留为未提交工作树，不得在 R1 标签之前生成 runtime baseline、code diff receipt、真实 bundle 或资格结果。

原协议冻结过程仅作为后文历史证据保留；当前续接必须执行本节和第 5 节的 R1 步骤。其余目标盲、资源、停止条件和后续阶段边界继续有效。Stage 4 formal target consumer 调用仍为 0，assessment row 物化仍为 0，阶段 9 锁定测试：未运行。

## 1. 工作目标与唯一蓝图

唯一实施蓝图仍为仓库根目录的 `SEISMOFLUX_IMPLEMENTATION_HANDOFF.md`。本工作线的目标不是证明 ETAS 预测有效，而是在不读取阶段 4 正式目标、不运行阶段 9 锁定测试的前提下：

1. 只修复 `ETASParameterBounds.from_transformed` 在物理上界精确映射时的 1 ULP 数值解码问题；
2. 冻结五个主快照、每快照五个起点，共 25 个真实 SciPy L-BFGS-B 调用及完整可重算诊断；
3. 在相同目标盲输入上重新判定 ETAS 是 `evaluable` 还是稳定的 `not_evaluable`；
4. 仅在正资格结果冻结后实现纯 ETAS comparator adapter；
5. 为后续新的阶段 4 修订提供静态 SVG、离线交互 HTML、可回溯证据和真正前瞻预测接口；
6. 每个阶段严格执行“测试 → 验收 → 提交 → 推送 → 远端标签核验”，完成后才进入下一阶段。

科学优先级仍是：在受控报警面积下提高独立物理地震区域召回。相对强度、评分和排序不得称作绝对发震概率。

## 2. 当前工作线和仓库现场

- 工作线：阶段 2 ETAS 数值修复（R1 协议勘误子阶段）
- 当前勘误分支：`codex/stage2-etas-repair-protocol-r1`；后续实现主分支：`codex/stage2-etas-numerical-repair`
- 原始协议基底：`dae6403`（`Finalize Stage 4 R2 blocked protocol`）；R1 勘误基底：`b7c70aced16a6bd57bf8f86f2680687e36b7710d`
- 计划中的本阶段标签：`v0.2.2-background-etas-repair-protocol-r1`
- 协议状态：原协议已远端冻结；R1 本地协议工程验收已通过，提交、推送和远端标签核验尚未完成，续接时必须先查询本地与远端 Git 状态
- 真实阶段 2 拟合源：本工作线尚未重新打开或查询
- Stage 4 formal target consumer 调用：0
- assessment row 物化：0
- 阶段 9 锁定测试：未运行

当前 R1 未提交文件：

- `configs/background_etas_numerical_repair.yaml`
- `docs/background_etas_numerical_repair_protocol.md`
- `docs/phase2_etas_numerical_repair_protocol_acceptance.md`
- `docs/restart_handoff_2026-07-19_stage2_etas_repair_protocol.md`
- `tests/unit/test_background_etas_numerical_repair_protocol.py`

规范 protocol package 仍是以下六个精确路径：

- `.gitignore`
- `configs/background_etas_numerical_repair.yaml`
- `data/manifests/etas_numerical_repair_start_manifest.json`
- `docs/background_etas_numerical_repair_protocol.md`
- `docs/phase2_etas_numerical_repair_protocol_acceptance.md`
- `tests/unit/test_background_etas_numerical_repair_protocol.py`

其中 `.gitignore` 与 start manifest 在 R1 中保持与原协议 blob 完全相同，配置、主协议、验收记录和协议测试随勘误更新。本交接文档位于 package 外，只提供恢复现场。上述五项 R1 改动须在同一次勘误提交中完整保留，不得在验收前推送标签。

## 3. 已完成内容

### 3.1 冻结输入和修复边界

- 五个主快照与支持域、补偿域、固定 50/25/12.5 km 网格已冻结；
- 每快照五个 PCG64 起点已冻结，起点清单文件与向量 payload 均有固定 SHA；
- 只允许在 `ETASParameterBounds.from_transformed` 中实施精确端点 1 ULP 修复；
- 禁止放宽边界、裁剪越界值、改变 objective、优化器、随机数、父历史、KDE、网格或门限；
- 资格执行必须调用五次 `fit_etas` 和 25 次真实 SciPy `minimize`，完整保存 opening/invocation/closing/diagnostic/gate/fit-attempt 证据。

### 3.2 数据隔离和可回溯约束

- 阶段 2 目录读取与 Stage 4 正式目标读取严格分离；
- 科学输入包只可在修复代码标签远端核验后构造；
- 源访问使用 append-only 两阶段账本，禁止先物化整表再过滤；
- 动态异常必须由完整历史报告期重建，不能只用最新一期；
- 参数、诊断、三网格证据、资格 seal、负结果证据和发布清单均设计为可重算哈希闭包。

### 3.3 运行时与资源控制

- 资格进程固定隔离启动器和单线程 BLAS/OMP/MKL/NUMEXPR/BLIS/VECLIB；
- 大型任务最多使用 6 个逻辑核，优先绑定逻辑核 16–21，并至少保留 2 个物理核心；
- 总 CPU 使用率达到 70% 时不启动重拟合；
- 当前没有本工作线的 Python 重任务在运行；
- 运行时 baseline 已设计覆盖 Python、NumPy、SciPy、stdlib、PE image 和原生依赖的文件身份。

### 3.4 R1 勘误验收证据

- 协议定向：`24 passed in 26.62s`；
- 协议、固定网格与 local-support manifest 联合回归：`56 passed in 24.34s`；
- clean public worktree 全量非目标回归：`1212 passed, 2 skipped in 351.64s`，`failures=0`、`errors=0`；新增一项协议测试后共收集 `1214` 项；
- 两个 skip 仅因 public worktree 不分发本地受限 Stage 3 feature store / Stage 4 spatial artifacts，未读取或复制这些工件；
- R1 JUnit：`data/interim/protocol-r1-full-nontarget.junit.xml`，size `194932`，SHA-256 `d463853cf3b010e79ac9ef3646dfc0cb6332128e1405cd1fd0b206cf6702a259`；
- Ruff check/format、单文件严格 mypy、`git diff --check` 均通过；
- 独立只读深比较确认，除 revision metadata、R1 tag/comparison base 与 quadrature `row/column` integer encoding 外，其余协议语义完全相同；审计提出的 package 清单、证据状态、comparison-base、revision-reason 和 bool 拒绝回归均已修复并复跑。

### 3.5 原始协议冻结证据（历史，不替代 R1）

- 最近一次中间全量非目标测试：`1205 passed in 396.88s`；
- JUnit：`data/interim/stage2/etas_numerical_repair/runtime_logs/protocol-final-full-nontarget-rerun1.junit.xml`；
- JUnit SHA-256：`cb10c074917be784d4a9b3802fa4f297b3fd2e8f36e7a791d52bcd49c21aacd4`；
- 该结果是协议继续修改前的中间基线，不是最终验收；
- 最终协议定向测试：`23 passed`（使用仓库 `.venv`；默认 Anaconda 解释器未安装项目包，故不得用于验收）；
- 同次 Ruff format/check：通过；
- 该定向结果覆盖三网格/Shapely 闭包、adapter 本地受限 payload、runtime schema、七文件 DAG 和门短路语义；
- 较早的旧 ETAS 19 项与协议 22 项联合回归：`41 passed in 16.32s`；
- 当前旧 ETAS、受影响 Stage 4 守卫与协议 23 项联合回归：`118 passed in 38.85s`；
- 当前全仓 Ruff check 通过、Ruff format check 为 `224 files already formatted`，`git diff --check` 通过；
- 严格 `mypy src tests`：`Success: no issues found in 216 source files`。过程中修复了新协议 YAML loader 的 2 个类型标注，以及基底 Stage 4 守卫测试的 4 个机械类型标注；没有更改生产代码或测试语义；
- 最终全量非目标测试：`1213 passed in 329.82s`，`failures/errors/skipped` 均为 `0`；JUnit 为 `data/interim/stage2/etas_numerical_repair/runtime_logs/protocol-freeze-final-full-nontarget-20260719-rerun1.junit.xml`，size `193506`，SHA-256 `406edffc53ad4f49a83638611d9aa376026882c89e67d87adcc5712da9dc2916`；
- 三条独立只读终审均为 `P0=0, P1=0, P2=0`：Runtime/Shapely 闭包、Adapter/九门/七文件 DAG、六文件整包与三文档一致性全部通过。

## 4. 中断前审计发现及当前修复状态

### 4.1 Runtime/Shapely 闭包：最终复审 P0/P1/P2 均为 0

配置、协议和定向测试现已把 `_grid_gate_evidence` 的完整运行时调用图封入 code-tag baseline/runtime seal，包括：

- `_grid_gate_evidence` 到网格、求积、ETAS expected mass、KDE、收敛诊断和规范哈希的全部递归项目依赖；
- evidence/quadrature/expected-mass dataclass 的 `__post_init__`；
- `passed`、`failure_reasons`、`numerical_evidence_id`、`cell_ids`、`GridCell.id`、`GridSpec.cell_size_mm` 等属性链；
- SciPy `cKDTree` 构造器和 `query_ball_point`；
- Shapely 固定双坐标 `Point` 构造只归 synthetic warmup；三网格生产闭包只归已有 Point 的 `Point.x/y` 与 `BaseGeometry.equals`。两类闭包分别冻结 public alias、descriptor、deprecation/multithreading wrapper、`__wrapped__`、类型化 closure cell、wrapped Python function 和 `shapely.lib` native ufunc 分层链；
- 固定构造路径显式覆盖 `ndarray.ndim/dtype`、返回后 Point class 自检；x/y/equals 的每个 multithreading wrapper 都显式覆盖实际执行的 `numpy.ndarray` class 检查，不再把只存在于代码对象中的名称误当作完整运行闭包；
- `numpy.array` 同时属于 synthetic warmup 和三网格 expected-mass 构造；deprecation wrapper 的 `warn_from` 是固定路径实际执行的比较值，只有未进入的 `category/make_msg` 警告分支按 inert closure 封口；
- Shapely 2.1.2 的完整 distribution `RECORD`、文件和 GEOS DLL 验证；
- baseline 与 qualification 完全相同的无源、无目标 Shapely warmup，防止第一次三网格重建才加载新模块/DLL而使 runtime map 漂移；
- 以 `canonical_binding_path + callable_layer` 唯一标识依赖记录，显式绑定 alias、closure cell、ufunc 签名、RECORD 文件和 own SHA；constructor/x/y/equals 各链均保存有序 `{record_id, record_sha}` 前像，可独立重算聚合 SHA；
- baseline、qualification runtime 和公开 Schema 同时保存完整发行包/运行文件/callable/Shapely-GEOS map 与 sibling SHA，并定义 opening triple 到 runtime pair/content 的逐字段 crosswalk；
- 独立 `three_grid_runtime_dependency_closure_sha256`，并与代码标签 baseline、资格 runtime seal 和公开 runtime Schema 逐值闭合。

拒绝重复 YAML key、单数/复数 dotted ref 全量解析、四链 reference digest 和 Shapely 实对象执行均已进入协议测试；独立终审为 `P0=0, P1=0, P2=0`。这只关闭协议工程缺口，仍不允许在协议提交和远端标签核验前开始真实资格拟合。

### 4.2 Adapter、九门和七文件 DAG：最终复审 P0/P1/P2 均为 0

配置已新增：

- 五个 `adapter_immigrant_density_artifact_payload`；
- 五个 `adapter_propagation_domain_artifact_payload`；
- 一个严格的 `adapter_local_restricted_payload_manifest`；
- 每个 payload 的 schema version 1、exact schema、own SHA、文件字节 SHA、来源、attempt、adapter runtime seal、scientific input 和 parameter snapshot 交叉约束；
- 统一路径根：`data/processed/stage2R/etas_numerical_repair_adapter_payload/attempts/{adapter_artifact_attempt_id}/local_restricted`；
- adapter code payload 对完整 allowlist 路径逐项绑定 Git blob OID、文件 SHA、size 和规范聚合 SHA；
- opening 前执行固定 box/point 的 buffer/normalize/WKB round-trip/equals/covers `adapter runtime preflight`；exact input/output schema、真实派生输出与两个 sibling payload SHA 均可独立重算，并完整绑定资格执行同款 `isolated_launcher_identity` 以及 Shapely/GEOS/RECORD/callable/runtime map；
- adapter 的七字段 Python runtime identity 明确保持 nested object，已移除与单一 hex scalar 的 schema 冲突，并由 scalar/object 字段全集互斥测试锁定；
- 六个 pre-closing staged 文件为 opening seal、artifact、global receipt、报告、静态 SVG、离线 HTML，closing seal 使用第七个 staged 路径；七个非-publication 文件必须在 ledger `completed` 前物化并 reopen 逐字节验证，publication manifest 在 `completed` 后原子写入；
- 公开 artifact manifest 只保存两张五快照 content-SHA map、adapter runtime seal SHA 和 aggregate local payload content SHA，不公开密度、几何、事件行或本地路径。

上述代码/runtime/preimage、七文件路径映射、唯一 DAG 和跨文件等式均已新增单元测试；协议定向、联合、全量回归及独立终审均已通过。

### 4.3 九门依赖真值表已闭合并通过终审

唯一顺序为 count → gradient → 两个 spread 兄弟门 → Hessian minimum → Hessian condition → branching → 两个 grid 兄弟门：

- count 始终评估；后续门只在全部前置门通过时评估；
- 兄弟门共享前置条件，但在开始后各自独立通过或失败；
- 任一上游失败使全部依赖门为 `not_run_upstream_gate`，对应 snapshot-gate metric 为 null，且不追加其 failure code；
- closing stability 中已有的诊断可以保留，但不能倒灌为未运行的 gate metric；
- `ordered_gate_status_records` 是按冻结九门顺序排列的九项 `{gate_name, gate_status}` 列表，避免 canonical mapping 键排序造成顺序歧义；`ordered_failure_codes` 只投影实际 `failed` 门。

因此 count、gradient、spread、Hessian、branching 或 grid 任一层失败均只有一个无歧义的下游状态，不会制造重复或虚构失败码。

### 4.4 修复代码边界与文档漂移保护已通过最终复跑

- 协议测试从冻结 Git blob 和当前 `etas_fit.py` 中分别剔除唯一允许修改的 `ETASParameterBounds.from_transformed` 完整源码区间；其余源码字节与 AST 必须完全相等，未来修复不能暗改同类其他成员或模块其他节点；
- 验收记录明确列出恰好六个 protocol package 路径；
- 主协议、验收记录和本交接对目标盲零计数、九门短路、adapter runtime seal、七文件 staged/materialization DAG 和后续阶段顺序采用同一稳定边界，并由文档一致性测试回归。

## 5. 精确续接步骤

1. 先执行 `git status --short`、`git rev-parse HEAD`、`git ls-remote origin` 和标签查询；原标签 `v0.2.2-background-etas-repair-protocol` 只能核验、不得移动；
2. 重跑 R1 协议文档完整性、signed row/column、固定网格回归、Ruff、mypy、全量非目标测试和独立只读审计；
3. 把上述五项勘误作为一个提交推送到 `origin/codex/stage2-etas-repair-protocol-r1`；
4. 创建并推送 annotated tag `v0.2.2-background-etas-repair-protocol-r1`，再用网络查询确认远端勘误分支和 peeled tag commit 均等于本地 `HEAD`；
5. 只有第 4 步已由远端证据满足后，才把 R1 勘误提交整合到实现主分支并继续阶段 2R-A；不得在此之前打开真实阶段 2 fit 源或阶段 4 目标；
6. 阶段 2R-A 仍须独立完成测试、验收、提交、推送和代码标签核验，不能把本次协议验收当作代码实现验收。

## 6. 后续阶段计划

### 阶段 2R-A：修复代码

- 实施唯一允许的 1 ULP endpoint repair；
- 保存 runtime baseline、repair diff receipt 和新测试；
- 测试、验收、提交、推送并冻结 `v0.2.2-background-etas-repair-code`。

### 阶段 2R-B：五快照数值资格

- 在修复代码标签远端核验后才允许访问真实 fit-only 输入；
- 受控资源运行五快照 × 五起点；
- 生成静态 SVG 和离线交互 HTML 数值诊断；
- `evaluable` 或稳定 `not_evaluable` 都必须冻结结果标签；
- 若 `not_evaluable`，正式停止 ETAS 路线并保持阶段 4 阻塞，不针对结果继续调参。

### 阶段 2R-C：纯 ETAS comparator adapter（仅正资格）

- 冻结 adapter code tag；
- 生成五快照密度/传播域本地制品、参数工件、全局 comparator receipt；
- 实现回溯 likelihood 接口和 128 条 PCG64 前瞻传播目录接口；
- 生成静态方法图和离线交互展示；
- 测试、验收、提交、推送并冻结 comparator result tag。

### 新阶段 4 修订

- 仅在 comparator receipt tag 远端核验后创建；
- 正式目标读取前完成全部 seal 和绑定；
- ETAS、静态 KDE 和异常置乱对照并列比较；
- 重点展示报警面积、独立物理区域召回、时间回溯和真正前瞻起报；
- 不复写历史前瞻预测，不按后来是否命中选择保留版本。

## 7. 当前明确禁止事项

- 不得读取阶段 4 正式目标或 assessment rows；
- 不得运行阶段 9 锁定测试；
- 不得使用真实震中生成候选、网格加密位置或区域边界；
- 不得把人工预测地点、震级、时间作为特征或标签；
- 不得绕过未闭合 P0/P1/P2直接提交协议；
- 不得在简单模型未通过前引入大型神经网络；
- 不得删除、覆盖或挑选旧 attempt；
- 不得把相对强度或顺位表述为绝对发震概率。

## 8. 交接完成判据

本交接文档记录可续接现场。本地协议工程验收的前四项已满足；阶段 2 协议冻结只有在六项全部成立时才完成：

- 三网格/Shapely runtime closure 通过机械测试和独立审计；
- adapter local payload 和 gate-skip closure 通过机械测试和独立审计；
- 最终全量非目标测试通过；
- 正式验收文档填写真实、无未来承诺；
- 提交和推送成功（本文件写入时待外层 Git 操作确认）；
- annotated protocol tag 推送成功且远端解析到预期提交（本文件写入时待外层网络核验）。
