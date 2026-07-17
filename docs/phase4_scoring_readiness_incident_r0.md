# 阶段4 R0 评分就绪性事故记录

## 1. 事故定性

- 发生边界：评分代码冻结和本地封印之后、首次目标授权之前。
- 分类：`pretarget_readiness_implementation_incident`。
- 正式验证尝试：0。
- 阶段4目标读取：0。
- 锁定测试接触：否。
- 模型得分、信息增益、命中、置乱 p 值：均未生成、未读取。

本事故不构成正式验证失败，也不消耗预登记的一次正式科学运行。它暴露的是 Arrow 表身份实现与已声明逻辑身份语义不一致。R0 封印因此失效，只保留为历史审计证据，不得授权任何修复后的代码。

## 2. R0 不可变身份

| 证据 | 身份 |
|---|---|
| 协议标签对象 | `2eb6f6d84483f6edd2610b1bbba749292a1bba44` |
| 协议标签提交 | `b6447ec4fc80699a2590cb12b31c37f0424d5435` |
| 评分代码标签对象 | `32870febb95cb5cb67bdf7d910bbb51f0fcfa975` |
| 评分代码标签提交 | `e04226566be46119d934174a4f9336d99d7d0e67` |
| formal-preflight receipt 内容 SHA-256 | `9943644eef078e68f2f842963a3be627232c6b52c52d8a2b11972f4f00f3628f` |
| formal-preflight receipt 文件 SHA-256 | `918dbf7b1433888a45c3688684c58d208410a98f1448862cc722962ea1992d73` |
| qualification 内容 SHA-256 | `4a83cfeaa2f90dd80d032b2bc9c906668995d73f40d910e4a8fe0601c8afc0e9` |
| qualification 文件 SHA-256 | `dbe518b0c930405875c4cb98614121db05b715325766f27a4ba127aea22fba18` |
| scoring seal ID / 内容 SHA-256 | `0b5089327f23446ffe4677c1e0fe2c43b22f780a1272e18fd6a74b7e95de8969` |
| scoring seal 文件 SHA-256 | `c3f3241ccd081440e8203f2e02d8d1c899e8d982034522adae11ea7809df59ae` |
| formal-attempt ledger 内容 SHA-256 | `ee931b1e5fb5208d617df75b8ae89c6f311a1f01dcd4fee256ba109d8ad1e552` |
| formal-attempt ledger 文件 SHA-256 | `d5eff0a2acf2a215b26ef5f80a1fe03f3b890366c51f20efbe358e4938bd05fb` |
| target-read ledger 内容 SHA-256 | `8a8ad69d69c46a9a3dc6b74a720ab332cecd31fd41edf365b85e8ffa5baf00a1` |
| target-read ledger 文件 SHA-256 | `e615f54720205dc119008b5fc36623145fd4516a0b78d91e820957271bb9f9c4` |

两本 R0 台账的 `records` 都为空。qualification 记录 `formal_attempt_count=0`、`target_read_count=0`、`locked_test_contacted=false`、正式后端 `cpu_float64`；用户请求使用 GPU，但冻结状态为 `blocked_no_frozen_backend`。

## 3. 触发过程

第一次 target-blind readiness check 因外层超时遗留两个已核实为 `run_stage4_formal.py check` 的进程，导致新鲜内存观测不能支持封印的空间置乱并发。只终止这两个已确认的 check 进程后，可用内存恢复；两本台账仍为 0，且没有进入目标授权。

随后重新执行 readiness check，在目标读取之前停止于：

```text
formal_production.py:340 build_target_blind_convergence_inputs
convergence.py:778 assert_selected_columns_exact
grid_features.py:376
ValueError: identity reconstruction differs from the accepted stage-3 columns
```

该检查运行约 1,194 秒后在第一期 25 km 重建断言处停止。它没有执行 `run`，没有取得目标能力，也没有写入正式尝试或目标读取记录。

## 4. 根因与首期证据

R0 receipt 声明的方法是“Arrow 逻辑值、有效性、类型与列序，null payload 归零规范化”，但收敛代码实际调用 legacy 物理 Arrow IPC 哈希。该物理哈希还包含：

1. 顶层 table schema metadata；
2. null 槽位下 Arrow 明确不定义的 payload 字节；
3. 与逻辑长度无关的位图 padding 表示。

已接受的 Parquet 表顶层 metadata 是落盘内容哈希/契约/排序键；内存重建表顶层 metadata 是特征字典/语义/布局。字段 schema 在忽略顶层 metadata 时一致，字段 metadata 无差异。

首个正式范围 issue（`2022-07-20T16:00:00Z`，25 km，9 个身份列与 27 个动态源列）在 1 个和 2 个空间工作进程下均逐列 `ChunkedArray.equals=true`。剥离顶层 metadata、只将 null 下 payload 规范为零并保留有效性后，接受表与重建表身份均为 `4c6f4e2c2e2b91781ded4e7b00ff34a1ee3230c39ba42c4fbd0c9b5d70540c9d`。因此首期没有有效值、有效性、类型、字段或列序差异；失败来自表示层而非科学内容。

## 5. 全量逻辑复演门

正式范围 153 期的 25 km 全量复演已在 2 个空间工作进程下完成：

- 通过：`153 / 153`；
- 每期选择列：36（9 个身份列、27 个动态源列）；
- 字段、列序、类型差异：0；
- null bitmap 差异：0；
- 有效槽科学数值差异：0（按 Arrow logical equals）；
- 首个真实科学差异：不存在；
- 复演耗时：2,808.5 秒；含输入加载总耗时：2,887 秒；
- 目标读取：`false`。

153 期已接受表的 canonical baseline 聚合 SHA-256 为 `27be4e1b6b6e3af624d8610b5a9cee25045ba7fed440545968f187798d817060`。该规范按 formal issue 顺序，对36列移除顶层 schema metadata、将 null 槽 payload 置零并保留原 validity、字段、列序和类型，以 Arrow IPC 计算逐期 SHA-256 后再用 canonical JSON 聚合。首期身份为 `4c6f4e2c2e2b91781ded4e7b00ff34a1ee3230c39ba42c4fbd0c9b5d70540c9d`，末期身份为 `38e797beef7aaf4743ea08ebb6cdc6ea527e31230f3e2d06325becc6fcd29d0e`。

严谨限定：全153期逐列比较使用 Arrow logical equals；它把有效 `+0.0` 与 `-0.0` 视为相等。已接受列中存在 278,811 个有效 `-0.0`，除首期以外，本次诊断没有保留全部重建表做有效 payload 的逐位 signed-zero 审计。因此该诊断足以排除阶段3科学值、validity、字段、类型与列序差异，但不能代替 R1 版本化哈希对153期全部有效 payload 的逐位复核。后者仍是评分代码 R1 的必过门。

该证据证明 R0 readiness 失败没有掩盖阶段3科学内容差异，因此阶段3工件与标签保持不变，R1 可以作为纯执行勘误继续。评分代码 R1 必须使用最终版本化实现重新完成153期逐位门控；本诊断不能代替后续正式 preflight、qualification 或 seal。

## 6. 纠正决定

1. 保留 R0 legacy 哈希、标签、receipt、qualification、seal 与双台账原样不动。
2. 科学协议版本保持 `0.4.0`，新增执行修订 `r1`。
3. R1 使用新协议、评分和结果标签，以及独立配置、清单、本地执行根、封印、台账、检查点、模型包、报告和可视化路径。
4. 新增显式版本化逻辑身份方法；仅排除顶层 metadata、规范 null 下未定义 payload 和长度外 padding，严格保留字段与全部有效科学内容。
5. R1 重新完成非目标测试、跨平台 CI、预检、qualification、双台账与 scoring seal，再通过 target-blind readiness check；在此之前禁止目标读取。
6. 正式路径继续锁定 CPU float64。GPU 只可在独立等价性门通过后作为可选加速，不得改变科学结果。
