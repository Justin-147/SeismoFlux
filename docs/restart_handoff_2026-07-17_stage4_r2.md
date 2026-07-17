# SeismoFlux 重启续接交接：阶段 4 R2 协议审计

> 日期：2026-07-17（Asia/Shanghai）
> 本文仅用于中断恢复，受仓库根目录 `SEISMOFLUX_IMPLEMENTATION_HANDOFF.md` 唯一实施蓝图约束；如有冲突，以唯一实施蓝图为准。

## 1. 工作目标与当前结论

当前目标不是运行阶段 4 正式评分，而是完成 R2 的目标盲协议终审、硬停治理、清单重生成和协议标签验收。

终审确认 ETAS 比较器仍为 `not_evaluable`，因此 R2 必须在正式目标读取前永久停止：

- 阶段 4 正式目标读取：`0`；
- 阶段 4 正式评分：`0`；
- 阶段 9 锁定测试：未运行；
- R2 formal preflight、qualification、seal、attempt ledger、target-read ledger、checkpoint、result 和图件：均不存在；
- 不允许发布 R2 scoring-code 标签或 result 标签；
- 只允许完成并发布 blocked-protocol 标签 `v0.3.1-anomaly-increment-protocol-r2`，且必须先通过全部机械验收。

R2 当前机器状态：

`blocked_before_target_read_etas_comparator_not_evaluable`

## 2. Git 现场

- 仓库：`D:\AIPred\SeismoFlux`
- 分支：`codex/stage4-anomaly-increment-scoring-code`
- 本地与远端共同基点：`13e645d1afb5a47cce53d88a36c169fb71cd9c7c`
- 远端：`Justin-147/SeismoFlux`
- Draft PR：`#9`，`https://github.com/Justin-147/SeismoFlux/pull/9`
- 当前改动尚未提交、尚未推送、尚未打 R2 标签；这是有意保留的未完成验收现场。
- 重启前 `git diff --check`：通过。
- 重启前没有 SeismoFlux 后台 Python 进程。

主要未提交文件以 `git status --short` 为准。新增但未跟踪的核心文件：

- `src/seismoflux/anomaly_increment/restricted_access.py`
- `tests/unit/test_stage4_restricted_access.py`
- 本交接文档

不得 reset、checkout 或覆盖当前工作树。

## 3. 已完成的审计与修复

### 3.1 科学协议语义

- 将 R2 明确冻结为 ETAS 不可评价导致的目标读取前硬停，而不是 KDE 替代 ETAS 的结果。
- 机器词表分为科学门状态、比较器可评价性和执行工件状态，避免把 `not_authorized_*` 冒充科学结论。
- 未来新执行修订必须以 `etas_background_no_increment` 为主科学对照；KDE 与 coverage 是次级必过混杂保护，均不得替代 ETAS。
- 单有 ETAS `evaluable` 状态绝不满足 G2；dynamic/snapshot 都必须产生并通过直接候选减 ETAS 的正式 G2 轨道。
- 四个 time/space × dynamic/snapshot placebo 的 observed/null 统计量都以 ETAS 为对照，并逐对象绑定 ETAS 工件、参数和数值资格 SHA。
- 开发折比较扩展为 ETAS/KDE/coverage 三轨，冻结计数为 `54/18/6`。
- 跨区比较扩展为三轨，冻结计数为 `702/234/6/12000`。
- 区域目标分母为 0 时，hits 必须为 0、recall 必须为 null、状态为 `not_evaluable_no_region_targets`，不得因此把 IG 轨道自动判为证据不足。
- 每窗口 39 区受支持唯一物理事件数之和必须等于全局数。
- 当前规范化协议设计 SHA-256：

`fd51d7f19306c48f95d89905416b1e38b9f2ab0078b4d5078ab83857278e193d`

- 重启前 `validate_stage4_r2_execution_contract()`：通过。

### 3.2 目标读取前硬停

已在共享工作树中加入或加强以下防线，但重启后仍须做一次完整旁路复审：

- formal preflight、qualification、scoring seal、formal run、formal readiness 和 target consumer 的 guard-first；
- `formal_target_read` 与 `formal_scoring` 中央动作均为当前 R2 禁止；
- canonical attempt/target-read ledger 创建受协议动作门控制；
- Windows canonical ledger 名称大小写、尾随点、尾随空格和已有路径解析别名纳入阻断；
- target consumer 不再只信任可导入 sentinel，而会在目标路径、账本和哈希前重查完整协议；
- formal materialization 和 formal publication 要求完整协议及真实授权能力，并在工作前执行中央 guard。

仍需收口的已知 P1：

1. `scoring_pipeline.run_stage4_in_memory_pipeline` 仍是公开导出的无协议入口，可接受 target-derived plan 并触发 checkpoint 回调。应私有化为 guarded formal-run 之后才能调用的 core，并调整合成测试入口。
2. 重新审计 qualification/preflight/seal/canonical-ledger 的所有 public load/write/reserve/complete API，证明 canonical R2 路径不存在绕开生产入口的调用方式。
3. 用公开 sentinel 伪造授权、Windows 路径别名、访问即抛错的 target/path 对象做全链对抗测试，断言任何 stat/open/hash/mkdir/tempfile/ledger/scoring 前即硬停。

### 3.3 受限本地空间工件 ACL v3

- ACL v3 使用同一保留句柄做入口/出口 owner、权限描述符、文件身份和状态复核。
- Windows 使用 handle-based `GetSecurityInfo`；禁止继承 ACE、未知/拒绝主体，并要求当前用户显式 Full Control。
- `INHERIT_ONLY_ACE (0x08)` 不能充当当前对象 Full Control 证明；已加对抗测试。
- POSIX 仅允许可证明的本地经典 mode-bit 文件系统，文件/目录为 `0600/0700` 且无 ACL xattr。
- 专测此前结果：`45 passed`，但后续协议改动后仍需最终重跑。
- 当前五个真实本地路径的只读实测正确 fail-closed：`Windows DACL still inherits access entries`。
- 当前没有 ACL receipt。实际 ACL 硬化不是 blocked-protocol 标签的前置项，但必须在任何未来新执行修订读取目标前完成。

## 4. 清单重生成中断点

最后一次目标盲重生成命令使用低优先级、BLAS=1、逻辑核 16–21，未访问地震目标表。运行约 119 秒后失败：

`PermissionError: [WinError 32] ... construction_zone_entity_mapping.parquet`

根因是 Parquet round-trip 检查读取后，Windows 文件句柄仍存活，导致临时文件无法原子替换。已修改 `preregistration.py`：通过显式打开/关闭输入流、`use_threads=False`、`pre_buffer=False` 完成 round-trip 检查，但尚未补完回归测试，也尚未重新生成。

失败现场遗留的 Git-ignored 临时文件：

`data/interim/stage4/anomaly_increment_r2/.construction_zone_entity_mapping.parquet.ryve69sd.tmp`

不要在重启后的第一步盲目删除；先确认无进程持有并把失败事实保留在验收记录。它不是正式执行工件，也没有目标数据。

当前公开 randomness manifest 与最新设计封印不一致，协议定向测试因此预期失败。最近一次协议+ACL 定向运行结果是 `3 failed, 56 passed`，三个失败均为“randomness manifest 尚未绑定最新 frozen design”；这不是最终验收结果。

## 5. 重启后的精确续接顺序

### A. 恢复现场

```powershell
Set-Location D:\AIPred\SeismoFlux
git status --short --branch
git diff --check
```

确认分支、基点和 dirty files 与本文一致，不要 reset。

### B. 复核协议哈希和严格合同

```powershell
$env:PYTHONPATH='src'
$env:OPENBLAS_NUM_THREADS='1'
@'
from pathlib import Path
import yaml
from seismoflux.anomaly_increment.config import validate_stage4_r2_execution_contract
from seismoflux.anomaly_increment.preregistration import protocol_design_sha256
p = yaml.safe_load(Path('configs/anomaly_increment_r2.yaml').read_text(encoding='utf-8'))
validate_stage4_r2_execution_contract(p)
print(protocol_design_sha256(p))
'@ | .\.venv\Scripts\python.exe -
```

预期 SHA：`fd51d7f19306c48f95d89905416b1e38b9f2ab0078b4d5078ab83857278e193d`。

### C. 完成 Windows Parquet 句柄回归测试

为 entity/cell/zone 三种 Parquet writer 增加“同一路径连续写两次且无残留锁/临时文件”的实际测试，先运行该测试、Ruff 和 mypy，再重跑生成器。

### D. 完成硬停旁路收口

先解决第 3.2 节列出的 scoring-pipeline 与 public artifact I/O 问题；然后运行目标零接触的对抗测试。此步不能读取真实 earthquake target。

### E. 低资源重生成与检查

保持 BLAS=1、低优先级和逻辑核 16–21；不要使用逻辑核 0–15，因为重启前另一个长期任务曾占用该范围。

```powershell
$env:PYTHONPATH='src'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$p = Start-Process -FilePath '.\.venv\Scripts\python.exe' -ArgumentList @(
  'scripts/build_stage4_preregistration.py',
  'generate',
  '--external-root',
  'D:\AIPred\LocationPred'
) -NoNewWindow -PassThru
$p.PriorityClass = 'BelowNormal'
$p.ProcessorAffinity = [IntPtr]0x3F0000
$p.WaitForExit()
$p.ExitCode
```

随后运行：

```powershell
.\.venv\Scripts\python.exe scripts/build_stage4_preregistration.py check
```

生成和检查通过后：

- 记录四份公开清单的文件 SHA 和内容 SHA；
- 记录随机输入封印和跨清单 validation content SHA；
- 将新 60 组随机向量与 R1 及提交 `13e645d` 的旧 R2 逐位置比较；
- 更新 `docs/phase4_protocol_r2_acceptance.md` 中全部 `PENDING_FINAL_REGENERATION` 和临时未勾选项。

### F. 最终测试与独立复审

按顺序运行：

1. protocol/governance/config/CLI/ACL/hard-stop 定向测试；
2. 全项目非目标测试（不得运行锁定测试）；
3. `ruff check scripts src tests`；
4. `mypy src` 及项目规定的脚本检查；
5. `git diff --check`；
6. 独立只读代理复审语义、旁路、目标零读取、工件零存在和标签边界。

所有较大测试继续使用单一低优先级进程、BLAS=1、最多 6 个逻辑核，至少保留 2 个物理核心。

### G. 只有全部验收通过后才发布

1. 更新最终验收文档；
2. 提交 R2 blocked-protocol 审计与清单；
3. 推送当前分支；
4. 只发布 `v0.3.1-anomaly-increment-protocol-r2`；
5. 不发布 scoring-code 标签和 result 标签。

随后回到阶段 2，另立目标盲 ETAS 数值修复协议：修复 transform 上界 1 ULP 解码问题、保存全部 25 个 start 诊断、五个快照全部重跑、实现真实 ETAS comparator adapter。只有 ETAS 可评价并冻结后，才可创建新的阶段 4 执行修订。

## 6. 重启前安全确认

- [x] 没有 SeismoFlux 后台 Python 进程；
- [x] 没有 formal R2 执行工件；
- [x] 正式目标读取和评分仍为 0；
- [x] 锁定测试未运行；
- [x] strict R2 contract 通过；
- [x] `git diff --check` 通过；
- [x] 没有提交、推送或打标签；
- [x] dirty worktree 与未完成门控已记录。

## 7. 重启后续接收口（2026-07-17）

本节是重启快照的后续记录，不改写上文当时仍未完成的事实。续接后已完成：

- Windows Parquet 同路径连续原子写回归通过，旧 WinError 32 未复现；
- 无协议参数的评分入口已从公共 API 移除；仓库生产调用图只允许在 `formal_scoring` 中央 guard 后调用不导出的私有 core；
- qualification、formal-preflight、canonical ledger 和相关脚本的公开 I/O 全部改为 guard-first，并通过零路径解引用对抗测试；
- 最终目标盲 `generate` 与严格 `check` 通过，正式目标读取仍为 `0`；
- 协议设计 SHA 为 `fd51d7f19306c48f95d89905416b1e38b9f2ab0078b4d5078ab83857278e193d`；
- 随机输入封印为 `8f8bb386cd62f225bb48b30f24c2360a865cf36faa541d35f94060daf44e8f09`，跨清单验收内容 SHA 为 `fde7d1fdc41568ee87d2405d58ab42b3ed0026ba8bf2a15611af0f903be95cb2`；
- 60 组 PCG64 参考向量按 `12/24/24` 重生，相对 R1 与提交 `13e645d` 的旧 R2 逐位置相同数均为 `0/60`；
- 全项目非目标测试最终复跑为 `1190 passed`、`0 failed`、`0 errors`、`0 skipped`；Ruff、Ruff format、mypy 117 个源文件和 `git diff --check` 均通过；
- 八类 R2 正式工件仍全部不存在，正式评分仍为 `0`，阶段 9 锁定测试仍未运行。

因此上文第 3.2 节列出的评分入口和 public artifact I/O 收口问题已经解决。仍需等待最终独立只读审计通过后提交、推送，并且只发布 `v0.3.1-anomaly-increment-protocol-r2`。不得发布 scoring-code 或 result 标签；发布协议标签后返回阶段 2 执行目标盲 ETAS 数值修复。
