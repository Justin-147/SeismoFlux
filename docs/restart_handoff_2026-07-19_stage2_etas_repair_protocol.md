# SeismoFlux 续接交接：阶段 2 ETAS 数值修复协议冻结

更新时间：2026-07-23（Asia/Shanghai）

## R3 协议勘误续接覆盖（当前有效）

原始协议标签 `v0.2.2-background-etas-repair-protocol`、R1 标签 `v0.2.2-background-etas-repair-protocol-r1` 与 R2 标签 `v0.2.2-background-etas-repair-protocol-r2` 均已提交、推送并完成远端核验，保持不可变。R2 annotated tag object 为 `903c80ed64295311f8d7870b4847f56d67caee51`，精确 peel 到提交 `5a5902a83645c217ea11a3bd99eb70b535f0e4df`。Stage 2R-A 随后仍在未访问真实 fit 源或阶段 4 正式目标时发现：完整 three-grid 13 字段 evidence 只有进程内 preimage，没有 attempt-local create-once 文件和跨重启磁盘复验合同。当前插入第三个最小协议勘误门控：

- 勘误分支：`codex/stage2-etas-repair-protocol-r3`；实现主分支仍为 `codex/stage2-etas-numerical-repair`；
- 精确勘误基线：R2 tag object `903c80ed64295311f8d7870b4847f56d67caee51` / peeled commit `5a5902a83645c217ea11a3bd99eb70b535f0e4df`；
- 待冻结 annotated tag：`v0.2.2-background-etas-repair-protocol-r3`；
- R3 的持久化改动：冻结 `attempts/{attempt_id}/local_restricted/three_grid_gate_evidence/{snapshot_id}.json` 的 6 字段 complete envelope；它嵌入平台文件身份、sealed 既有 13 字段、三个 6 字段 resolution 前像、与 `pipeline_etas.py` 逐字段一致的 `numerical_evidence_id` crosswalk 和 envelope own SHA；outer own SHA 的 canonical JSON v1 前像恰好是显式前五字段 `envelope_identity_fields_exact`，排除第六个 own-SHA 字段；
- R3 的显式 hash-binding 语义勘误：snapshot gate、fit-attempt 和五快照 presence map 中既有 nullable SHA 字段改为锚定包含安装身份的 outer envelope SHA；字段名、类型、公开 Schema、路径和文件数不变，embedded 13-field own SHA 只在 envelope 内重算；
- Windows 资格运行唯一选择 `windows_ntfs_ntcreatefile_filerenameinfo_v1`，每个初始安装或 fresh checkpoint session 只 bootstrap 一次绝对 NT workspace root，以下全部 handle-relative、case-sensitive、`OBJ_DONT_REPARSE`；directory/temp/final DesiredAccess 精确为 `0x001201bf/0xc0110080/0x80100080`，目录禁止 GENERIC 高位，CreateDisposition/CreateOptions 分开冻结；ShareAccess、`FileFsDeviceInformation` 本地非远程谓词、parent `FileIdInfo` 128-bit FileId/u64/null 分支、temp/preinstall/final identity、no-replace rename 和 kernel32 `FlushFileBuffers` 必须全通过。POSIX `linkat` profile 只定义不选择；
- `installed_file_identity` 还冻结从 workspace root 到 evidence directory 的九级有序目录身份链及其 SHA；可信本地 supervisor 必须恰好一次传入 `--workspace-root`，opening seal 将其绑定到相同 HEAD/upstream/code-tag/protocol blobs。Windows 严格使用 `\\?\Volume{GUID}` canonical path 并唯一转换为 `\??\Volume{GUID}` NT ObjectName，`FILE_ID_128` 按 `Identifier[0..15]` 原始顺序；六类 `NtCreateFile` 调用使用完整参数矩阵。POSIX fresh session 从固定 root 绝对 open 一次、以下只 `openat`，九级记录逐 live fd `fstat`；
- R1 signed `row/column` 与 R2 publication-path 合同继续有效；禁止借 R3 修改网格、事件、KDE、初值、优化器、objective、门限、源访问、adapter 合同、目标边界、科学输入、公开 Schema 或公开路径；
- 必须先完成 R3 定向/静态验收、normalized R2→R3 deep compare、提交、推送与远端标签核验，再恢复 Stage 2R-A；
- Stage 2R-A 的在建代码不得在 R3 标签之前生成 runtime baseline、code diff receipt、真实 bundle 或资格结果。

原协议、R1 和 R2 冻结过程仅作为后文历史证据保留；当前续接必须执行本节和第 5 节的 R3 步骤。其余目标盲、资源、停止条件和后续阶段边界继续有效。Stage 4 formal target consumer 调用仍为 0，assessment row 物化仍为 0，阶段 9 锁定测试：未运行。

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

- 工作线：阶段 2 ETAS 数值修复（R3 three-grid local persistence 协议勘误子阶段）
- 当前勘误分支：`codex/stage2-etas-repair-protocol-r3`；后续实现主分支：`codex/stage2-etas-numerical-repair`
- 原始协议基底：`dae6403`（`Finalize Stage 4 R2 blocked protocol`）；R1 提交 `da916454c908e0cbe4a7526f56a8f837331a3c7c`；R3 精确基底为 R2 tag object `903c80ed64295311f8d7870b4847f56d67caee51` / peeled commit `5a5902a83645c217ea11a3bd99eb70b535f0e4df`
- 计划中的本阶段标签：`v0.2.2-background-etas-repair-protocol-r3`
- 协议状态：原协议、R1、R2 已远端冻结；R3 第一至第四轮及 pre-bind P1 均保留为负证据。construction snapshot 已提前到 source factory 完成时，最终限定与静态检查通过，同一五文件快照的语义、公式/R2 差异、平台三路终审均为 `P0=0/P1=0/P2=0`；唯一最终受控全量 `1258 passed, 2 skipped`、无 failure/error。本地工程验收允许提交；R3 提交、推送和远端 annotated tag 尚未完成，故 Stage 2R-A 仍暂停
- 真实阶段 2 拟合源：本工作线尚未重新打开或查询
- Stage 4 formal target consumer 调用：0
- assessment row 物化：0
- 阶段 9 锁定测试：未运行

当前 R3 只允许修改且尚未提交的文件：

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

其中 `.gitignore` 与 start manifest 在 R3 中保持与 R2 blob 完全相同，配置、主协议、验收记录和协议测试随勘误更新。本交接文档位于 package 外，只提供恢复现场。上述五项 R3 改动须在同一次勘误提交中完整保留，不得在验收前推送标签。

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
- 最后核验于 2026-07-23：新增目录链/checkpoint 定向启动前整机 CPU 三次采样为 `53.2%/48.8%/54.0%`，低于 70% 阻断线；pytest 采用单进程、数值库单线程，结束后没有本工作线 Python 重任务；
- 运行时 baseline 已设计覆盖 Python、NumPy、SciPy、stdlib、PE image 和原生依赖的文件身份。

### 3.4 R3 three-grid complete envelope 勘误修复状态

- complete envelope 路径固定为 `data/processed/stage2R/etas_numerical_repair_fit_input/attempts/{attempt_id}/local_restricted/three_grid_gate_evidence/{snapshot_id}.json`，五个 filename 只能是冻结快照 ID；outer 6 字段中嵌入 sealed 原 13 字段、三个 resolution 前像、严格 crosswalk、安装身份和 envelope SHA；outer SHA 的 `envelope_identity_fields_exact` 前像恰为前五字段，排除第六 own-SHA 字段；
- fresh reopen 必须从完整 bytes 重建真实 resolution/convergence/gate dataclass，核对 sealed 顶层与 nested snapshot/parameter 身份，再重算三个 resolution SHA、两个 pair SHA、13 字段 own SHA、源码 property 与独立 crosswalk 的 `numerical_evidence_id`、outer SHA；不得 evaluator replay；
- temp exclusive create 后、首字节写入前把 profile、parent identity、temp/final leaf bytes、final path/case、VolumeSerial/FileId/link count 写入 envelope；安装和每个 checkpoint 核对同一身份与最终单链接，防止同字节替换和 hardlink；
- Windows 用唯一 NTFS profile：每个初始安装或 fresh checkpoint session 只允许一次绝对 workspace-root bootstrap，`RootDirectory=NULL`、OA 含 `OBJ_DONT_REPARSE` 且无 `OBJ_CASE_INSENSITIVE`；`CreateDisposition=FILE_OPEN` 与 `CreateOptions=0x00200021` 分开冻结。attempt root 以下 `NtCreateFile(RootDirectory=parent)`；directory/temp/final DesiredAccess 为 `0x001201bf/0xc0110080/0x80100080`，目录禁止 GENERIC 高位且目录 flush 必含 specific-rights `0x00120116` 子集，ShareAccess 固定 `0x3` 且无 DELETE share；`GetVolumeInformationByHandleW` 与 `FileFsDeviceInformation` 同时证明 local NTFS/nonremote，parent 直接由 `FileIdInfo` 取得 canonical u64 volume serial + 128-bit lowercase FileId；关闭初始 temp 后用完整 preinstall options 相对重开，rename 后 flush file/parent，保留 parent chain 做 final 相对重开验证，最后关闭全部句柄；
- complete envelope 的安装身份嵌入九级 `ordered_directory_identity_chain` 与链 SHA；immediate parent 精确投影自最后一项。Windows 每个 FileId 按 raw `FILE_ID_128.Identifier[0..15]` 顺序编码，canonical path 只接受完整 Volume GUID；workspace bootstrap、目录 reopen/create、temp create/preinstall reopen、final reopen 六种 `NtCreateFile` 调用逐字段冻结 OA、Unicode string、allocation/attributes、EA、access/share、disposition/options 与 IoStatus；
- `--workspace-root` 只允许由本地 supervisor 恰好传一次，必须指向 opening seal 核验同一 HEAD/upstream/远端 code tag/协议 blobs 的 worktree，绝对路径不公开。fresh checkpoint verifier 在任何 evidence stat/open/read 前自行调用受控单次 provider 从 root 捕获九级 live chain，再读取未信任 envelope；outer SHA 与独立三锚一致后才认证 embedded chain，随后比较预捕获 live chain，最后接受。provider 构造器只接收由独立 raw handle values 建立的受控 source，不接收预制 records；source/provider/receipt 必须是 factory exact class，拒绝 subclass 与结构伪装。source/provider 绑定和 single-use 状态由外部对象身份 registry 持有；exact source factory 完成构造时，unbound registry 立即冻结全部 type/value、path object identity 与 override bytes，provider 构造器先消费并比较该 snapshot，拒绝 pre-bind path→raw/scalar mutation。provider capture 再消费自身注册项并复核 exact source；source 仅接受其绑定 provider 且只观察一次，并在 path stat/构造记录前再比较同一 snapshot，拒绝 post-bind 变更。回执含唯一单次 capture ID、实际 observation count 和受控 factory 注册的不透明 capability，receipt registry 还绑定原始对象、source kind、ID、count 及签发时 canonical bytes，首次注册原件验证先消费再校验；直接/浅/深拷贝 embedded chain、重包装、pre-bind/post-bind source swap/config mutation、used reset、窃取 capability、签发后改写标量或 records、重复提交或不足九级均拒绝；
- POSIX profile 只作 portability 定义且本次不选择：single writer、verified parent、`linkat` EEXIST no-clobber、link count `1→2→1`、exact `readdir` 与 parent `fsync`，普通 rename 禁止；
- POSIX fresh session 还固定为 root 绝对 `open(O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)` 一次、以下只用 `openat` 逐层下降，九级身份全部来自对应 live directory fd 的 `fstat(st_dev, st_ino)`；禁止 chdir、环境、当前目录、envelope path 或 path-stat 充当身份源；
- presence/SHA/null 真值表只有“前置门通过→文件存在且 outer envelope SHA 与 gate/fit/presence 三路非空值相等，grid 字段来自 embedded 13 字段”和“任一前置门未通过→文件缺席且全部为 null”两种状态；五快照 nullable map 与 content SHA 必须执行 mixed/all-present/missing/extra 测试；
- finalize、staged-local identity、closing、manifest construction、manifest staged reopen、首次 materialization、每次 retry 和 `before_seal_return` 是八个独立 fresh-reopen checkpoint；每个 session 先关闭旧句柄、只重开 root 一次再逐层相对重走，不得用内存缓存、rerun evaluator、recompute/reseal/substitute 或 alternate path；R3 durable 写入合同只管 local envelope，R2 staging 整个对象必须逐值不变；
- 外锚 DAG 为 `identity+sealed13+preimages+crosswalk → outer SHA → gate/fit/presence → staged-local aggregate → closing → manifest/commit/tag`，无反向引用；安装成功但首锚未耐久落盘即中断时，同 attempt 永久失效，禁止跨重启补锚；
- invalid execution、publication failure、成功公开、结果提交和远端标签后都保留 evidence；公开 Schema、字段名/类型、路径和文件数保持 R2 不变，但 three-grid SHA 值来源是显式 R3 语义差异；
- R2→R3 deep compare 必须直接枚举精确差异路径，不得以 `pop` 或整棵 `outputs` 相等断言掩盖变化；只允许 revision/tag metadata、IO/evidence allowlist 职责、本地 persistence subtree和少数明确的 hash-source 节点。

首轮终审失败与全量停止证据必须保留：首次 `-I -m pytest -q tests` 因 worktree root 未进入 `sys.path`，收集时 `ModuleNotFoundError: scripts`，`1 error in 14.17s`、执行数 0，JUnit 只有 collection error；第二次 Git-ignored runner 修正 `sys.path` 后，在 stdout 约 70%、无失败标记且 stderr 为空时收到 P0，主动 terminate，无有效最终 JUnit/summary，确认无孤儿 Python 进程。两次均 INVALID/STOPPED、不计验收；修复后的定向/关联/静态复验与第二轮独立终审清零前不得再次运行全量，清零后由外层只运行一次最终受控全量并回填。

第二轮终审失败也必须保留：公式/持久化 `P0=0/P1=2/P2=1`，证据闭包 `P0=0/P1=2/P2=0`，平台合同 `P0=0/P1=3/P2=0`。公式审计确认主公式正确但真实 dataclass/sibling binding/presence map 执行闭包不足；证据审计复现 `PUBLIC_CROSSWALK_UNCHANGED=True`、replacement identity 改变且级联重写文件仍被旧 verifier 接受；平台审计确认 DesiredAccess、本地非远程谓词和 parent identity 类型不完整。三路均 NO-GO。

第二轮后的上一版修订曾完成协议 `36 passed`、关联 `68 passed`，但第三轮独立语义 `P0=1/P1=2/P2=1` 与平台 `P0=1/P1=4/P2=1` 终审否决；随后版本完成完整协议 `58 passed in 60.16s`、关联 `90 passed in 66.95s`/`90 passed in 60.85s`，又被第四轮语义 `P0=0/P1=1/P2=1`、公式/差异 `P0=0/P1=0/P2=1`、平台 `P0=0/P1=3/P2=2` 否决。第四轮后再修订先行通过关键反例 `33 passed, 34 deselected in 32.34s`、完整协议 `67 passed in 60.76s`、关联 `99 passed in 61.24s`；新增目录链/root/platform 后的中间版本为完整协议 `70 passed in 67.63s`、关联 `102 passed in 67.52s`。external-registry 初版定向 `7 passed, 63 deselected in 11.85s`、完整协议 `70 passed in 67.16s`、关联 `102 passed in 69.30s`，但平台终审以 `P0=0/P1=1/P2=0` 否决其 pre-bind source mutation 窗口。当前 construction-snapshot 版本定向 `7 passed, 63 deselected in 11.48s`、完整协议 `70 passed in 67.56s`、关联 `102 passed in 70.99s`；Ruff check/format、单文件 strict mypy、YAML parse 与 `git diff --check` 全部通过。最终同哈希三路终审全部 `P0=0/P1=0/P2=0`；唯一最终受控全量 `1258 passed, 2 skipped in 427.68s`、无 failure/error。JUnit 路径 `data/interim/protocol-r3-final-full-nontarget.junit.xml`，XML `tests=1260/failures=0/errors=0/skipped=2/time=427.638`，size `205485`、SHA-256 `e02d6d050101f3f07e9a62fe418a6d2fdcde3084a0c77c5011f7dfec67f908ef`。两个 skip 仅为未分发的本地受限 Stage 3/4 工件；启动 CPU `55.7%/50.3%/47.3%`，affinity `0x3F0000`、`BelowNormal`、数值库单线程。工作树仍只允许精确五个 tracked 文件，下一步是最终文档守卫、提交、推送和远端标签核验。

### 3.5 R2 publication-path 勘误状态（历史，继续有效）

- attempt-local staged root 固定为 `data/processed/stage2R/etas_numerical_repair_fit_input/attempts/{attempt_id}/staged_public`；`attempt_id` 只允许安全单一 ASCII 路径组件并须与既有 fit-input attempt 目录大小写精确一致；
- pre-closing staged identity 只覆盖 common 7，`evaluable` 另加参数文件 2 个；closing seal 与 closing 后完整 qualification manifest 各自使用独立 staged 路径；
- pre-closing path→SHA map 已逐项绑定 final path 到同 mapping staged path 的 reopened exact bytes SHA，并要求 size/bytes/Schema 复验后重算 aggregate；
- common 9 与 evaluable additional 2 的每一项都冻结精确 staged→final 路径，staged 路径恰等于 root 加 final repository-relative path；
- repair code tag 的全部 final 路径须不存在，成功物化后的 Git 状态只能为 `A`；`not_evaluable` 参数目录始终缺席；
- path/reparse/no-clobber/reopen/byte-exact/逆序回滚与 post-closing 同字节 materialization-only 重试语义已经写入配置、主协议和测试；
- branch 顺序固定为 not-evaluable common7→closing→manifest / evaluable common7→parameter2→closing→manifest；post-closing failure receipt 以六位连续序号和 own-SHA 链记录 manifest staging state、按状态可空 SHA、staged byte map、创建/回滚路径与 retry eligibility；manifest 未 reopened-valid 时禁止同 attempt 纯物化重试；
- 协议定向最终验收运行：`25 passed in 27.47s`，文档回填后再次完整复验：`25 passed in 24.98s`；最终状态写入前其余测试为 `24 passed, 1 deselected`，新增 staging/YAML/dotted-ref 单独复验为 `3 passed`；
- 协议、固定网格与 local-support manifest 联合回归：`57 passed in 24.68s`；
- `.gitignore` 可执行闭包逐项证明 11 个 staged path 被忽略、11 个 final path 不被忽略；`git ls-tree` 逐项证明 final path 在 R1 基底不存在；Ruff check/format、严格单文件 mypy 和 `git diff --check` 均通过；
- 独立审计首轮为 `P0=1, P1=4, P2=1`，修复 path→SHA、exact A、failure receipt、顺序、manifest failure state 后，第三轮终审为 `P0=0, P1=0, P2=0`，`FINAL_SCHEMA_AUDIT=PASS`；
- clean public worktree 全量非目标回归：`1213 passed, 2 skipped in 363.04s`，`failures=0`、`errors=0`；两个 skip 仅因本地受限 Stage 3 feature store / Stage 4 spatial artifacts 未在 public worktree 分发；
- R2 JUnit：`data/interim/protocol-r2-full-nontarget.junit.xml`，size `195115`，SHA-256 `ec40ea946c976d362b3b1313fc31c3f318eb4b13dc2717fcd63bf3e1754e9aaf`；stderr 为空；
- full-run 启动前 5 次整机 CPU 均值 `45.19%`、峰值 `47.85%`，执行 affinity `0x3F0000`、优先级 `BelowNormal`、数值库单线程；运行中复核均值约 `48.75%`、峰值约 `51.66%`；
- normalized R1→R2 深比较与精确五文件 allowlist 均通过；本隔离工作树仍不得自行提交、推送或打标签，发布闭环由外层执行。

### 3.6 R1 勘误验收证据（历史，继续有效）

- 协议定向：`24 passed in 26.62s`；
- 协议、固定网格与 local-support manifest 联合回归：`56 passed in 24.34s`；
- clean public worktree 全量非目标回归：`1212 passed, 2 skipped in 351.64s`，`failures=0`、`errors=0`；新增一项协议测试后共收集 `1214` 项；
- 两个 skip 仅因 public worktree 不分发本地受限 Stage 3 feature store / Stage 4 spatial artifacts，未读取或复制这些工件；
- R1 JUnit：`data/interim/protocol-r1-full-nontarget.junit.xml`，size `194932`，SHA-256 `d463853cf3b010e79ac9ef3646dfc0cb6332128e1405cd1fd0b206cf6702a259`；
- Ruff check/format、单文件严格 mypy、`git diff --check` 均通过；
- 独立只读深比较确认，除 revision metadata、R1 tag/comparison base 与 quadrature `row/column` integer encoding 外，其余协议语义完全相同；审计提出的 package 清单、证据状态、comparison-base、revision-reason 和 bool 拒绝回归均已修复并复跑。

### 3.7 原始协议冻结证据（历史，不替代 R1/R2/R3）

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

### 4.0 Three-grid 完整 preimage 持久化：R3 当前门控

R2 已把 gate/fit-attempt/staged-local 中的 nullable three-grid own SHA 交叉闭合，但完整前像没有固定磁盘落点。首轮 R3 又错误地只持久化裸 13 字段，无法从文件重建 `ETASGridGateEvidence.numerical_evidence_id`；第二轮进一步证明把 identity 与 outer SHA 只放在同一文件中仍无法抵抗跨重启替换。当前修订使用 6 字段 outer envelope，complete-envelope own SHA 只覆盖显式前五字段，完整嵌入三个 resolution 前像、sealed 13 字段、源码同公式 crosswalk 和安装身份，并让三个既有外部 nullable SHA 字段锚定 outer SHA。这样替换身份必改变 outer SHA并被旧 gate/fit/presence 拒绝。状态窗口互斥：install 成功至 post-install flush/parent-sync/final-reopen 完整验证之间为终态 `indeterminate_after_install`，只允许只读取证，禁止同 attempt resume/reanchor/publish/推进资格；完整验证后至首锚耐久落盘前为 `invalid_execution`，不得 resume/reanchor；首锚后 envelope/anchor 任一漂移也为 `invalid_execution`，不得同 attempt 修复。八 checkpoint 只读 fresh reopen，anchor 后 expected SHA 只能来自独立冻结外部记录。R2 staged/public 安装协议保持不变，但 hash-binding 来源是显式 R3 语义勘误。

### 4.0a Qualification publication 路径闭包：R2 历史门控

R1 中 `staged_public_payload_identity` 已正确冻结 pre-closing 7(+2) 文件及无环哈希顺序，但 qualification execution 没有像后续 adapter 那样给出 attempt-local staged root 和逐文件 mapping。R2 明确三类产物：pre-closing 7(+2)、独立 staged closing seal、closing 后独立 staged qualification manifest；path→SHA value 绑定 mapped staged reopened bytes，全部 staged 9(+2) 文件通过 final clean/absent/reopen 检查后才能 byte-exact 公开物化。post-closing 失败保留 staged closing、已安装 manifest 或 manifest temp 失败证据，不再误分类为“没有 closing seal”的 pre-closing `invalid_execution`，并以严格 append-only publication failure receipt 状态机记录。只有 manifest reopened-valid、完整安全回滚且所有身份未变时允许同一 attempt 重试纯复制，任何 fit 或 staged payload 重算都禁止。

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
- 主协议、验收记录和本交接对目标盲零计数、qualification common 9(+2) staged/materialization/rollback、九门短路、adapter runtime seal、adapter 七文件 DAG 和后续阶段顺序采用同一稳定边界，并由文档一致性测试回归。

## 5. 精确续接步骤

1. 先执行 `git status --short`、`git rev-parse HEAD`、R2 tag object/peeled commit 核验和远端标签查询；原始、R1、R2 标签只能核验、不得移动；
2. R3 专属、完整协议、关联回归、Ruff、mypy、`git diff --check`、同哈希三路终审和唯一最终受控全量均已完成；不得再次运行全量。提交前只运行文档守卫/静态检查并确认精确五文件 diff；
3. 把上述五项勘误作为一个提交推送到 `origin/codex/stage2-etas-repair-protocol-r3`；
4. 创建并推送 annotated tag `v0.2.2-background-etas-repair-protocol-r3`，再用网络查询确认远端勘误分支和 peeled tag commit 均等于本地 `HEAD`；
5. 只有第 4 步已由远端证据满足后，才把 R3 勘误提交整合到实现主分支并继续阶段 2R-A；不得在此之前打开真实阶段 2 fit 源或阶段 4 目标；
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
- 当前尚未生成任何实际模型预测效果静态图或交互页；R3 只冻结诊断/展示合同，不得把协议示意图冒充效果结果；
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

本交接文档记录可续接现场。R3 阶段 2 协议勘误只有在以下各项全部成立时才完成：

- three-grid 6 字段 complete envelope（含原 13 字段、三个 resolution 前像、源码同公式 crosswalk、持久文件身份和九级目录身份链）、presence/SHA/null 真值表、独立 live-chain/root source、平台目录/文件 durability 和八个 fresh-reopen 节点通过机械测试；
- normalized R2→R3 deep compare 证明科学规则、公开 Schema/字段类型/路径/文件数精确未变，并只放行明确登记的 outer-SHA hash-source 语义节点；
- qualification common 9(+2) staged/final/rollback/retry 路径闭包通过机械测试和独立审计；
- 三网格/Shapely runtime closure 通过机械测试和独立审计；
- adapter local payload 和 gate-skip closure 通过机械测试和独立审计；
- 本轮专属、协议文件、关联与静态限定复验通过，新的三路独立终审全部清零；两次无效/停止全量尝试保留但不计验收，随后由外层一次最终受控全量通过并回填；
- 正式验收文档填写真实、无未来承诺；
- 提交和推送成功（本文件写入时待外层 Git 操作确认）；
- annotated protocol tag 推送成功且远端解析到预期提交（本文件写入时待外层网络核验）。
