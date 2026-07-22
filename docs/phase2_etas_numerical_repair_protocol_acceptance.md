# 阶段 2 ETAS 数值修复协议验收记录

## 验收对象

- 工作线：阶段 2 ETAS 数值修复（`0.2.2`）
- 原始协议标签：`v0.2.2-background-etas-repair-protocol`（保持不可变）
- R1 勘误标签：`v0.2.2-background-etas-repair-protocol-r1`（保持不可变）
- R2 勘误标签：`v0.2.2-background-etas-repair-protocol-r2`（保持不可变）
- 当前勘误标签：`v0.2.2-background-etas-repair-protocol-r3`
- 唯一实施蓝图：`SEISMOFLUX_IMPLEMENTATION_HANDOFF.md`
- 当前勘误分支：`codex/stage2-etas-repair-protocol-r3`；后续实现主分支：`codex/stage2-etas-numerical-repair`
- 原始协议比较基线：`dae6403`
- R1 勘误比较基线：`b7c70aced16a6bd57bf8f86f2680687e36b7710d`
- R2 勘误比较基线：`da916454c908e0cbe4a7526f56a8f837331a3c7c`
- R3 勘误精确基线：R2 annotated tag object `903c80ed64295311f8d7870b4847f56d67caee51`，peel 到提交 `5a5902a83645c217ea11a3bd99eb70b535f0e4df`
- 状态：未通过（第四轮独立终审仍发现 P1/P2，作为历史负证据保留；当前仅因提交、推送和远端 annotated tag 尚未完成。pre-bind P1 已修复，最终同哈希三路复审已全部 `P0=0/P1=0/P2=0`，唯一最终受控全量已通过，本地工程验收允许进入提交门，但远端冻结前仍不得恢复 Stage 2R-A）

本次验收只冻结目标盲修复设计、25 个既有优化初值、输入/执行封印、调用回执、适配边界和停止条件。它不实施修复、不生成真实 fit-only 输入包、不运行真实 ETAS 资格拟合，也不创建新的阶段 4 执行修订。

## R3 勘误边界

R2 远端冻结后，Stage 2R-A 实现审计发现每快照完整 three-grid 13 字段 payload 只存在于进程内；公开 gate、fit-attempt 和 staged-local identity 虽保存可空 own SHA，却没有冻结 SHA preimage 的 attempt-local 文件路径、durable create-once 语义或跨进程重开验证点。R3 只关闭该本地证据持久化歧义：

- 完整文件路径精确为 `data/processed/stage2R/etas_numerical_repair_fit_input/attempts/{attempt_id}/local_restricted/three_grid_gate_evidence/{snapshot_id}.json`，五个 filename 只允许冻结快照 ID；文件留在同一 attempt root、Git 忽略且禁止公开；
- 同一文件是严格 6 字段 local-restricted envelope：安装文件身份、既有 sealed 13 字段 evidence、三个精确 6 字段 resolution SHA 前像、逐字段对齐 `pipeline_etas.py` 的 `numerical_evidence_id` crosswalk 和 envelope own SHA；outer own SHA 的 canonical JSON v1 前像恰好是 `envelope_identity_fields_exact` 前五字段，明确排除第六个 own-SHA 字段；13 字段 own SHA 只在 envelope 内重算，包含平台身份的 outer envelope SHA 是 gate、fit-attempt 和 presence map 的外部锚；
- fresh reopen 必须从 envelope bytes 重建真实 `ETASGridResolutionEvidence`、`ThreeGridConvergenceGateEvidence` 和 `ETASGridGateEvidence`，核对顶层/nested snapshot 与 parameter 身份，再重算三个 resolution SHA、两个 pair SHA、sealed 13 字段 own SHA、源码 property 与独立 crosswalk 的 `numerical_evidence_id`、envelope own SHA，以及 gate/fit-attempt/staged-local presence crosswalk；不得重跑 evaluator；
- 每个文件 create-once、append-only、no-clobber；任何既有文件即使字节相同也不得被下一次 seal 采用，缺失/损坏/漂移文件不得在同一 attempt 中重建、替换、reseal 或另选路径；
- presence/SHA/null 真值表冻结为两种合法状态：three-grid 前置门全部通过时文件必在且 outer envelope SHA 等于 gate、fit-attempt、staged-local presence map 的非空值，三个 grid 公开字段来自同一 embedded 13 字段对象；前置门任一失败/未运行时文件必无且这些 SHA/metric 全部为 null；
- qualification finalize、staged-local identity、closing、manifest construction、manifest staged reopen、首次 materialization、每次 publication retry 和 `before_seal_return` 是八个分离的只读 fresh-reopen checkpoint；它们不改变 R2 的 staged/closing/manifest/failure/public 安装协议，禁止 rerun evaluator 或用 fit/gate/manifest/public SHA substitute preimage；
- 冻结 Windows 资格运行只选择 `windows_ntfs_ntcreatefile_filerenameinfo_v1`：每个初始安装或 fresh checkpoint session 只允许一次可信 workspace root 绝对 NT namespace bootstrap；bootstrap 的 `RootDirectory=NULL`，OA 必含 `OBJ_DONT_REPARSE` 且不含 `OBJ_CASE_INSENSITIVE`，`CreateDisposition=FILE_OPEN` 与 `CreateOptions=0x00200021` 分开冻结。attempt root 以下全部使用 `NtCreateFile` parent `RootDirectory` 相对下降；每个 fresh session 先关闭旧句柄，再从 root 唯一重开逐层重走并复验。directory/temp/final DesiredAccess 分别固定为 `0x001201bf`、`0xc0110080`、`0x80100080`；目录禁止 GENERIC 高位，交给 `FlushFileBuffers` 的目录句柄必须含展开后的 `FILE_GENERIC_WRITE` specific-rights 子集 `0x00120116`，文件句柄则须含 `GENERIC_WRITE`。全部 ShareAccess 固定为 `0x3` 且不含 `FILE_SHARE_DELETE=0x4`；exact NTFS 之外还须由 `FileFsDeviceInformation` 证明 `FILE_DEVICE_DISK` 且 `FILE_REMOTE_DEVICE` 位为 0；parent 身份直接来自每层目录 handle 的 `FileIdInfo`，Windows volume serial 为 canonical u64 base-10、FileId 为 lowercase 32-hex，并与 child volume/case components 交叉一致；另一平台分支严格为 null；
- `installed_file_identity` 必须嵌入从 `.` 到 `three_grid_gate_evidence` 的九级 `ordered_directory_identity_chain` 和其 canonical JSON SHA；每项绑定仓库相对路径、精确大小写分量、canonical absolute path 与平台身份，最后一项必须等于 immediate parent。初装 expected chain 来自写入前逐层 live handle；fresh checkpoint expected chain 来自已被 gate/fit/presence 外锚冻结的 envelope，launcher root 由 opening `HEAD` 绑定的固定参数提供，不得从 envelope、环境或当前目录反推；
- Windows canonical path 只接受严格 `\\?\Volume{xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}\...` grammar，拒绝 drive path、不完整 GUID、dot segment 与重复分隔符；`FILE_ID_128` 只按 `Identifier[0]..Identifier[15]` 原始数组顺序逐字节 lowercase hex，禁止 integer/GUID endian conversion。workspace bootstrap、目录 reopen/create、temp exclusive create、temp preinstall reopen 与 installed-final read-only reopen 六类调用都必须由逐项结构化矩阵冻结 OA/Unicode string 长度、allocation/file attributes、EA、access/share、disposition/options 和 `IoStatusBlock.Information`，不得靠拼接字符串或隐式默认值；
- 可信根只能来自本地 supervisor 传入且恰好一次出现的 `--workspace-root`：它必须指向 opening seal 已核验相同 `HEAD`、upstream、远端 code tag 和协议包 blobs 的 worktree，绝对值禁止进入公开工件；Windows 先把严格 `\\?\Volume{GUID}\...` 按唯一规则转换为 `\??\Volume{GUID}\...` 的 `NtCreateFile` ObjectName，live root handle 身份最终必须等于 outer-SHA 已认证目录链的第 0 项；
- fresh checkpoint 顺序固定为：verifier 自己在 stat/open/read 前调用受控、单次的 live-capture provider，从 launcher root 独立捕获九级 live handle 链；随后 handle-relative 打开并完整读取但暂不信任文件；重算 outer SHA 并匹配独立冻结的 gate/fit/presence 三锚；再把预先捕获的 live 链逐项对比已认证 embedded expected chain；最后才接受文件。provider 构造器只接受受控 live-handle source，绝不接受预制 records；source、provider 与 receipt 都要求 factory exact class，拒绝 subclass 或结构伪装。source/provider 的 factory 绑定和 single-use 状态保存在外部对象身份 registry，不依赖可重置的实例 flag。exact source factory 完成构造时，unbound registry 立即冻结全部构造字段 type/value、path 对象身份与 identity-override canonical bytes；provider 构造器先原子消费该 unbound 项，再比较原始 construction snapshot，任何 path→raw mode 或 raw scalar 的 pre-bind 篡改都失败且不可重试。provider 调用再原子消费自身注册项、逐对象复核构造时绑定的 exact source；source observation 也只接受绑定 provider 且只消费一次，并在任何 path stat 或记录构造前再次比较同一快照，拒绝绑定后变更。source 只能由独立 raw handle-source 参数构造，并在 provider 调用期逐项观察，自动生成进程内唯一且只消费一次的 capture ID、实际 observation count 和不透明一次性 runtime capability。receipt registry 同时绑定原始对象、source kind、capture ID、count 与签发时 canonical record bytes；确认是注册原件后，第一次验证尝试须先原子消费 ID/capability，再校验 count/content。直接传 embedded chain、浅/深拷贝、重包装深拷贝、构造后/绑定前或绑定后替换 source/配置、重置 used 状态、窃取 capability 另包 receipt、签发后改写任一标量或嵌套 records、重复提交或不足九级捕获都必须拒绝；
- 未选 POSIX profile 同样冻结 fresh root 绝对 bootstrap 一次、以下仅 `openat` 逐层下降，每个九级目录身份都来自对应 live fd 的 `fstat(st_dev, st_ino)`；禁止 chdir、环境、当前目录、envelope path 或 path-stat 充当身份源；
- temp 创建后、首字节写入前把 profile、parent、temp/final leaf、final path/case、VolumeSerial/FileId/link count 写入 outer envelope；写入/flush 后关闭初始 temp 句柄，再以 `FILE_OPEN | FILE_NON_DIRECTORY_FILE | FILE_OPEN_REPARSE_POINT | FILE_WRITE_THROUGH | FILE_SYNCHRONOUS_IO_NONALERT` 相对重开并验证，由该句柄 no-clobber rename；安装后 flush file/parent，关闭 renamed file 但保留 parent chain，再 final handle-relative reopen 完整验证后关闭全部句柄。同字节替换、hardlink、preexisting final 均失败。失败状态互斥：install 成功至完整 post-install flush/parent-sync/final-reopen 验证之间为终态 `indeterminate_after_install`，只允许只读取证，禁止同 attempt resume/reanchor/publish/推进资格；验证完成至首个外锚耐久落盘前为 `invalid_execution`，不得 resume/reanchor；首锚之后任何 envelope/anchor 漂移也为 `invalid_execution`，不得同 attempt 修复。POSIX `linkat` profile 仅作未选 portability 定义，明确 `1→2→1` link count、single writer、parent identity、exact `readdir` 和 parent `fsync`；
- 限定验收必须实际执行 synthetic canonical envelope round-trip、真实 dataclass 语义重建、五快照 nullable presence content SHA、cascade rehash tamper、missing/extra/no-clobber 及同字节替换/hardlink identity tamper；协议期 synthetic 安装只证明 schema/hash/verifier，不冒充 selected Windows 原生 profile 实现测试；
- 外锚 DAG 固定为 `installed identity + sealed13 + resolution preimages + crosswalk → envelope SHA → gate/fit-attempt/presence → staged-local aggregate → closing seal → manifest/result commit/tag`，无反向引用；安装成功但首个 outer-SHA 外锚未耐久落盘即中断时，同一 attempt 永久失效，禁止跨重启从文件自报身份补锚；
- invalid execution、publication failure、成功公开、结果提交和远端标签后均保留完整 attempt-local evidence，不得清理；R3 不增加公开文件、不改变公开 Schema、字段名、类型、路径或文件数，但明确把既有 three-grid SHA 字段的 hash-binding 来源由 embedded 13-field SHA 改为 outer envelope SHA。

R3 还把 IO/evidence 模块 allowlist 职责精确扩展到上述 envelope 的 create-once 持久化和磁盘重开验证。R2 的 `qualification_public_result_staging` 整个对象必须逐值相等；deep compare 直接枚举 revision、local persistence 和少数 hash-source 语义差异路径，不得通过 `pop` 或整棵 `outputs` 归一化隐藏变化。事件、快照、科学输入、signed grid、模型、KDE、初值、objective、优化器、门限、源访问、目标盲边界、adapter 合同、公开 schema/path 均不可改变。该缺口仍在真实 fit 源再次 open/stat/hash/query、bundle inspection 或任何真实资格 fit 之前发现；原始、R1、R2 标签均不移动、不覆盖、不删除。

### R3 首轮终审失败与无效全量尝试（保留审计）

- 首轮 R3 包虽曾写为本地 PASS，但独立终审给出 `P0=1/P1=4/P2=1`；旧 PASS 当场撤销。核心 P0 是裸 13 字段文件不能重建 `numerical_evidence_id`，其余问题包括平台原子持久化、身份跨重启、staging 越界和 checkpoint 合并。该版本不得冻结。
- 首次 runner 本质命令为 `.venv/Scripts/python.exe -I -B -m pytest -q tests --junitxml=data/interim/protocol-r3-full-nontarget.junit.xml`；隔离模式未把 worktree root 放入 `sys.path`，收集 `test_stage4_anomaly_increment_runtime.py` 时出现 `ModuleNotFoundError: scripts`，`1 error in 14.17s`，测试执行数为 0。CPU precheck average `47.99%`、peak `50.68%`，affinity `0x3F0000`、`BelowNormal`；该 JUnit 只有 collection error，判为 INVALID，不计验收。
- 修正后的 Git-ignored runner 为 `data/interim/run_protocol_r3_full_nontarget.py`，命令 `D:\AIPred\SeismoFlux\.venv\Scripts\python.exe -I -B data\interim\run_protocol_r3_full_nontarget.py`，内部调用 `pytest -q tests --junitxml=data/interim/protocol-r3-full-nontarget.junit.xml`。数值库全单线程，PowerShell/child affinity `0x3F0000`、`BelowNormal`；收到 P0 时 stdout 约 70%、无失败标记、stderr 为空，CPU 三次采样 average `22.82%`、peak `24.47%`。进程被主动 terminate，无有效最终 JUnit/summary，不计验收；随后确认无该 Python 解释器孤儿进程。

### R3 第二轮独立终审失败与随后修订（保留审计）

- 公式/持久化审计：`P0=0/P1=2/P2=1`。完整前像与主公式正确，但 verifier 未执行真实 resolution/order/pair dataclass 约束、未闭合 sealed 顶层与 nested snapshot/parameter 身份，五快照 mixed present/null map 与 content SHA 仍缺执行测试。
- 证据闭包审计：`P0=0/P1=2/P2=0`。同一 scientific sealed payload 可换新 FileId/inode、重写 envelope identity 并级联重算 outer SHA，而旧外部 crosswalk 不变；另指出 Windows access/share 合同和 selected-profile 验收表述不足。
- 平台合同审计：`P0=0/P1=3/P2=0`。缺少逐调用 DesiredAccess、机器可判定的 local/nonremote predicate，以及 parent Windows FileId/volume 与 POSIX dev/ino 的严格类型/null 分支。
- 三路结论均为 NO-GO，所以当时没有启动唯一最终受控全量，也没有提交、推送或打标签。随后修订采用 outer envelope SHA 三路外锚、真实 dataclass 重建、五快照 presence 执行测试、Windows access/share/nonremote/parent identity 闭包；该版后来仍须重新定向复验并重新做独立终审，旧 `31/63` PASS 只能作为被后续审计否决前的历史记录。

### R3 第三轮独立终审失败与随后再修订（保留审计）

- 上一版修订曾取得协议 `36 passed`、关联 `68 passed`，但随后独立语义闭包审计给出 `P0=1/P1=2/P2=1`，独立平台合同审计给出 `P0=1/P1=4/P2=1`，因此这些限定测试只保留为被最新终审否决前的历史工程证据，不构成 R3 PASS。
- 语义 P0 是 complete envelope own SHA 的 `includes_ref` 仍可被解释为包含自身 SHA 的六字段自引用；P1/P2 涉及科学 13 字段不变、仅换 FileId/inode 后级联重算 outer SHA 的三路旧外锚拒绝、pre-anchor crash、all-present presence 与文档一致性。
- 平台 P0/P1/P2 涉及目录 `NtCreateFile` 错用 GENERIC 高位、句柄生命周期、post-install 状态重叠、parent `FileIdInfo`/u64 边界、bootstrap 和 preinstall reopen 参数。本轮已把 outer SHA 前像收窄为显式五字段，目录 access 改为 `0x001201bf`，拆开 bootstrap disposition/options，闭合 terminal no-resume 状态，并新增身份/anchor/presence 永久反例。
- 2026-07-23 中断恢复后，该版首批关键定向为 `15 passed, 33 deselected in 20.85s`；进一步关闭 disposition/options、terminal 状态、u64 和严格身份解析后，扩大定向为 `25 passed, 33 deselected in 24.25s`，随后完成协议 `58 passed` 与关联 `90 passed`。这些结果已被第四轮独立终审拒绝，不计最终验收。

### R3 第四轮独立终审失败与当前修订（当前状态）

- 语义闭包终审为 `P0=0/P1=1/P2=1`：旧身份替换反例同时改变 POSIX inode 与 temp leaf，未隔离证明 FileId/inode 单字段进入 outer SHA；文档进度也已漂移。
- 公式/差异终审为 `P0=0/P1=0/P2=1`：主 SHA 公式、R2 精确基线与 deep-diff allowlist 通过，唯一问题是验收文档仍把已经完成的限定复验写为未完成。
- 平台合同终审为 `P0=0/P1=3/P2=2`：fresh session 缺少完整目录 expected-identity 链及来源；`NtCreateFile` 缺少六类调用的完整参数矩阵；canonical path verifier 接受 drive path/宽松 GUID；另未冻结 `FILE_ID_128.Identifier[0..15]` 原始顺序，文档状态过期。
- 当前修订已分别增加 Windows FileId-only 与 POSIX inode-only 单字段替换，保持 sealed 13 字段精确不变，并验证旧 gate/fit/presence 三锚整体与逐锚拒绝、全部新锚通过；同时新增九级目录身份链/链 SHA、严格 Volume GUID grammar、raw FileId fixture 和六类 `NtCreateFile` 结构化调用矩阵。
- live-chain 执行闭包补齐前的中间快照曾为完整协议 `67 passed in 60.76s`、关联 `99 passed in 61.24s`；随后禁止 embedded-chain copy 的版本为完整协议 `70 passed in 67.63s`、关联 `102 passed in 67.52s`。外部 registry 初版虽达到定向 `7 passed, 63 deselected in 11.85s`、完整协议 `70 passed in 67.16s`、关联 `102 passed in 69.30s`，平台终审仍以 `P0=0/P1=1/P2=0` 指出 source 构造后、provider 绑定前可改变配置，因此同样只保留为被否决的工程证据。当前修订让 unbound registry 在 factory 构造完成时立即冻结 snapshot，provider 先消费并比较后才绑定；新增 pre-bind path→raw 与 raw-scalar mutation 反例后的最终限定为定向 `7 passed, 63 deselected in 11.48s`、完整协议 `70 passed in 67.56s`、协议+固定网格+local-support manifest `102 passed in 70.99s`，均零失败；Ruff check/format、单文件 strict mypy、YAML parse 和 `git diff --check` 也通过。随后同一五文件快照的三路最终复审全部 `P0=0/P1=0/P2=0`，唯一最终受控全量也已通过；当前只待提交、推送和远端标签核验。

## R2 勘误边界（历史，继续有效）

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

R2 不改变 R1 的 signed `row/column`，也不改变事件、快照、几何、网格、KDE、初值、objective、优化器、数值门限、源访问、目标盲边界或 adapter 合同。该缺口仍在任何真实 fit 源再次 open/stat/hash/query 或 bundle inspection 之前发现；原始和 R1 标签均不移动、不覆盖、不删除。R2 已由 annotated tag object `903c80ed64295311f8d7870b4847f56d67caee51` 冻结，并精确 peel 到提交 `5a5902a83645c217ea11a3bd99eb70b535f0e4df`。

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

### R3 three-grid local persistence 勘误复验

| 项目 | R3 结果 |
|---|---|
| 首轮独立终审 | FAIL：`P0=1/P1=4/P2=1`；旧 PASS 撤销，旧测试数字不再作为 R3 接受证据 |
| 第二轮独立终审 | FAIL：公式 `P0=0/P1=2/P2=1`；证据闭包 `P0=0/P1=2/P2=0`；平台 `P0=0/P1=3/P2=0`。当前仍为 NO-GO |
| 第三轮独立终审 | FAIL：语义闭包 `P0=1/P1=2/P2=1`；平台合同 `P0=1/P1=4/P2=1`。上一版 `36/68` 限定复验被否决，当前再修订仍为 NO-GO |
| 第四轮独立终审 | FAIL：语义闭包 `P0=0/P1=1/P2=1`；公式/差异 `P0=0/P1=0/P2=1`；平台合同 `P0=0/P1=3/P2=2`。`58/90` 限定复验被否决，当前仍为 NO-GO |
| 两次全量尝试 | INVALID/STOPPED；一次 collection error、一次在约 70% 主动终止，均无有效最终 summary/JUnit，不计验收；本轮定向/关联/静态复验及新的三路独立终审清零前不得再次运行全量，清零后只允许外层运行一次最终受控全量并回填 |
| 第二轮终审前的历史限定复验 | 专属 `10 passed in 13.31s`；协议 `31 passed in 35.49s` 与 `31 passed in 35.10s`；关联 `63 passed in 33.55s`。已被第二轮 NO-GO 否决，不计最终验收 |
| 第三轮终审前的历史 outer-anchor/dataclass/presence/platform 协议定向 | 主代理独立复跑 `36 passed in 37.93s`；修订执行者复跑 `36 passed in 39.00s`。已被第三轮 NO-GO 否决，不计最终验收 |
| 第三轮终审前的历史协议、固定网格与 local-support manifest 关联回归 | 主代理独立复跑 `68 passed in 39.18s`。已被第三轮 NO-GO 否决，不计最终验收 |
| 本轮再修订的中断恢复定向 | 首批关键合同 `15 passed, 33 deselected in 20.85s`；扩大身份/平台合同定向 `25 passed, 33 deselected in 24.25s`。尚未构成阶段 PASS |
| 第四轮终审前再修订完整协议 | `58 passed in 60.16s`；零失败，但已被第四轮终审否决 |
| 第四轮终审前协议+固定网格+local-support manifest 关联回归 | 首次 `90 passed in 66.95s`；文档与永久证据断言回填后 `90 passed in 60.85s`；均已被第四轮终审否决 |
| 当前目录链/平台/provenance 再修订最终限定 | 关键反例 `33 passed, 34 deselected in 32.34s`；首次独立 live-chain/root/POSIX 定向 `7 passed, 63 deselected in 10.52s`；R2→R3 exact deep-diff 与平台核心定向 `2 passed, 68 deselected in 7.63s`；中间 live-chain 版本 `70 passed in 67.63s`、关联 `102 passed in 67.52s`；被 pre-bind P1 否决的 registry 初版为定向 `7 passed, 63 deselected in 11.85s`、完整协议 `70 passed in 67.16s`、关联 `102 passed in 69.30s`；当前 construction-snapshot 版本定向 `7 passed, 63 deselected in 11.48s`、完整协议 `70 passed in 67.56s`、关联 `102 passed in 70.99s`，均零失败 |
| 精确 R2 基线 | annotated tag object `903c80ed64295311f8d7870b4847f56d67caee51`，peel 到 commit `5a5902a83645c217ea11a3bd99eb70b535f0e4df` |
| 五快照本地路径闭包 | 五个格式化 evidence path 均由 `git check-ignore --no-index` 判定为 ignored；路径逐项等于同 attempt root 下冻结 snapshot filename |
| 最终三路独立终审 | 同一最终五文件快照的语义、公式/R2 差异、Windows/POSIX 平台三路均为 `P0=0/P1=0/P2=0`；五文件二次 hash 均与指定值一致，允许启动唯一最终全量 |
| R2→R3 差异路径 deep compare | 最终公式审计确认 normalized deep compare 恰为 24 个允许路径、unexpected/missing 均为 0，其中 root-binding 恰为 5 项；`qualification_public_result_staging` 整体逐值等于 R2 |
| 公开合同边界 | 公开 Schema、字段名/类型、路径和文件数不变；three-grid nullable SHA 的值来源显式由 embedded 13-field SHA 改为 outer envelope SHA，不能再声称 `outputs` 完整对象逐值等于 R2 |
| 唯一最终受控全量 | `1258 passed, 2 skipped in 427.68s`；JUnit `tests=1260`、`failures=0`、`errors=0`、`skipped=2`、`time=427.638`；这是三路终审全零后唯一一次计入 R3 验收的全量运行 |
| 最终 JUnit | `data/interim/protocol-r3-final-full-nontarget.junit.xml`；size `205485`；SHA-256 `e02d6d050101f3f07e9a62fe418a6d2fdcde3084a0c77c5011f7dfec67f908ef` |
| 最终 stdout/stderr | stdout size `2323`、SHA-256 `fae4ec842ee3a7e9a71bfb0985faf4261d012d0e53ac6550da1cf9c2be8b5d98`；stderr size `0`、SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| 最终全量资源门控 | 启动前 CPU `55.7%/50.3%/47.3%`（均值 `51.1%`、峰值 `55.7%`）；PID `30660`，affinity `0x3F0000`（6 逻辑核）、`BelowNormal`、OMP/OpenBLAS/MKL/NumExpr/BLIS/VECLIB 单线程 |
| 最终两项 skip | `test_actual_first_stage3_row_group_passes_exact_grid_identity_bridge`：公开仓库未分发 accepted local Stage-3 feature store；`test_cli_check_local_restricted_artifact_integration_or_explicit_skip`：本地受限 Stage-4 spatial artifacts 未分发；均未读取正式目标，且与历史公开 worktree skip 边界一致 |
| Ruff | 当前协议测试 `ruff check` 与 `ruff format --check` 通过 |
| Mypy | 当前协议测试 `Success: no issues found in 1 source file` |
| Git 空白检查 | `git diff --check` 通过 |
| 精确改动范围 | 仅修改配置、主协议、验收记录、重启交接和协议测试五个授权 tracked 文件；未提交、未推送、未打标签 |

R3 修复、限定复验和最终全量没有打开、stat、hash、查询或检查真实 fit 源/真实 bundle，没有运行 ETAS 拟合、阶段 4 正式目标消费者或阶段 9 锁定测试。第二、第三、第四轮及 pre-bind P1 依次否决旧限定结果；当前 construction-snapshot 快照已完成限定、静态、三路独立终审全零和唯一最终受控全量。本地工程验收已经通过，允许进入精确五文件提交、推送和远端 annotated tag 核验；在远端 R3 标签完成前仍不视为阶段冻结完成。

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

### 原始协议独立只读审计（历史）

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

当前 R3 阶段仍标记未通过，仅因为提交、推送和远端 annotated tag 尚未完成；本地验收已满足：最终限定复跑和静态检查已经完成，三路独立终审对同一五文件快照全部 `P0/P1/P2=0`，唯一最终受控全量 `1258 passed, 2 skipped` 且无 failure/error。现在授权把精确五个 tracked 文件作为一个提交推送，并创建/核验 `v0.2.2-background-etas-repair-protocol-r3`；不得夹带 ignored JUnit/runner 或主工作树 Stage 2R-A 草稿。这不表示 ETAS 已稳定拟合，更不表示预测效果或科学门控成功；当前没有实际预测效果静态图或交互页。R3 只冻结诊断/展示合同；Stage 2R-B 才可在代码标签远端核验后生成 25 起点数值诊断 SVG 与离线 HTML，真正回溯和前瞻效果图还需正资格 comparator receipt 和后续新 Stage 4 协议。原始、R1、R2 annotated tag 保持不动；只有远端 R3 标签解析到本次勘误提交后，才允许恢复 1 ULP 修复代码子阶段。

协议包自身包含本验收文件，因此不能在文件内部嵌入最终提交、标签或自身 blob SHA 而制造自引用。六个精确路径已由测试锁定，最终 Git blob/tree/commit 身份由提交与远端 annotated tag 在外层冻结并可重复查询。
