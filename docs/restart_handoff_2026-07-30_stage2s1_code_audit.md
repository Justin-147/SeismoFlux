# Stage 2S-1 无效正式尝试与项目续接交接

- 更新日期：2026-07-30
- 工作树：`D:\AIPred\SeismoFlux\data\interim\worktrees\science_first`
- 分支：`codex/stage2-etas-science-first`
- 代码提交：`4188523991926c51a7fbd9314d36395cc9bfad62`
- 代码标签：`v0.2.3-causal-seismicity-screen-code`，已远端核验
- 当前阶段：`Stage2S-1` 正式尝试收尾
- 当前结论：`STOP`
- 正式状态：attempt 已消费，formal run `invalid`
- 科学成绩：没有形成，不能判断 S1 好于或差于 S0/SP
- 当前最好合法模型：继续保留 G1-LS 的长期 75 km KDE `S0`

## 1. 外行能听懂的结论

这次正式历史考试已经开始，但在长期背景重建之后、第一份可评分答案形成之前程序出错了。程序随后
尝试记录失败时又发生第二个错误，把第一个错误的具体原因遮住了。

程序在出错前完成了：

1. 核对公开代码和标签；
2. 核对研究区、三层格网和 39 个区域；
3. 登记唯一考试机会；
4. 认领并只读取一次真实地震目录；
5. 验证目录并重建长期背景支持域。

但它还没有形成任何可评分的预测图，也没有打开“新方法比旧方法好多少”的成绩。因此不能说近期
30 天地震有用，也不能说没用。这不是可信负结果，而是一次无效实验。

按预先冻结的规则，唯一机会已经用掉，不能修复后拿同一批答案再考一次。Stage 2S 到此停止。

## 2. 用了哪些数据

本次正式入口实际使用：

- 研究区 `china_mainland.geojson`：用于无目标几何、面积和格网核验；
- 25 km cell-zone 映射：用于核对 15,697 个格与 39 个区域；
- fold4 本地支持清单：用于重建并核对长期 75 km KDE 支持域；
- 冻结地震目录：40,898 行，只物理读取并解析一次。

本次没有使用：

- 异常表、观测报告或人工预测；
- 断层、危险性、应力或大型神经网络特征；
- 2024–2025 作为新的独立 Stage 2S 验证；
- 锁定测试。

未使用异常数据是 Stage 2S 的冻结协议边界，不是本次遗漏；本次运行没有核验异常数据自身的
完整性或位置。

## 3. 已封存的不可变证据

以下本地文件必须保留，不得删除、覆盖或提交到公开仓库：

| 文件 | SHA-256 |
|---|---|
| non-target preflight receipt | `e216194a6ec035a2788d3f55b7e949f5afd21e9584d6d422de6c9a61baa9d71f` |
| formal attempt ledger | `fef0606d68aa6d1da721f634c04fd53df10500b522aa6b38307415194b560f5d` |
| target-read receipt | `cd1482d63dc8add9354f005b647781371eec62ee882fe7fac23adc3c0ff71cfe` |

成功进度：

```text
remote_tags_verified
  -> non_target_preflight_passed
  -> attempt_claimed
  -> target_read_claimed
  -> catalog_parsed
  -> fold4_support_rebuilt
```

缺失且必须保持缺失：

- fold fit receipt；
- issue/fold/master prediction seal；
- `stage2s_whole_run_record.json`；
- 六个正式结果工件；
- result tag。

`terminal_failure_record.json` 也缺失。它不是“没有失败”，而是失败记录器重读 preflight receipt
时触发保留键错误；再次启动时也会在同一路径失败。不得手工伪造 terminal，也不得再次运行
launcher。

## 4. 已知异常链与未知原始原因

最后一个成功进度是 `fold4_support_rebuilt`。主科学路径随后在 fold-fit receipt 和任何预测封印
形成前抛出异常，但正式 CLI 只输出最终异常，没有保存原始 traceback；因此不能诚实确定第一个
异常发生在 S0 构造、折内拟合还是两者之间的哪个步骤。

异常处理器随后准备写 terminal record。它为汇总现有 seal 状态而重读 preflight receipt，又把
其中带 `$seismoflux_type` 的已编码浮点 marker 送入 canonical encoder，触发可确认的第二个错误：

```text
reserved canonical JSON key at
$.bindings.aligned_grid_identity.layers.12.5.cell_size_km:
$seismoflux_type
```

第二个错误覆盖了原始 science 异常，并阻止 terminal 写入。这是 code tag 内可确认的
foundational P0；原始 science 异常则保持 `unknown`，不得猜测或把第二个错误误写成模型计算的
主失败。不能用新增测试数量掩盖这次科学尝试没有结果的事实。

## 5. CPU 与本地数据挂载

- 正式执行为 1 worker，六类数值线程均固定为 1；
- 本机 24 个物理核心，保留至少 2 个；
- 没有终止或接管其它 Python 进程；
- 为隔离工作树恢复了三个本地忽略路径：
  - `data/processed/china_mainland.geojson`：HardLink；
  - `data/processed/stage1`：Junction；
  - `data/interim/stage4/anomaly_increment_r2`：Junction。

这些链接不进入 Git，不改变源文件内容。续接时不得据此重开地震目录。

## 6. 科学价值复审

- `science_value_category`: `no_material_progress`
- `evidence`: 真实目录只读一次并完成 fold4 支持重建，但在任何预测封印和评分前失败；没有
  S1-S0 或 S1-SP 的效果指标
- `decision`: 停止 Stage 2S，不再修复后重跑历史目标，继续保留 S0
- `next_scientific_test`: 先做不读取目标成绩的科学路线复审；若复审仍选择近期地震假设，再另立
  前瞻封存协议，从本次之后尚未发生的 cohort 开始同步封存候选模型
- `stop_condition`: 不得再次使用本次 target-read、2022–2025 或锁定测试为 Stage 2S
  调参/恢复；任何新路线必须先冻结新协议、代码和不可回填归档

## 7. 后续计划

当前不应继续堆工程功能。下一步顺序固定为：

1. 保持 S0 为当前合法背景；
2. 先做不读取任何目标成绩的科学路线复审，比较“近期地震前瞻封存”和其他最小合法问题；
3. 把复审选定的新问题、模型/对照、评价门和停止条件写入唯一实施蓝图，独立验收、提交并推送；
4. 只有新协议授权后，才修正必要的序列化/terminal 边界并实现新实验；不得恢复本次 attempt；
5. 若选定前瞻路线，在每个未来起报日前封存候选图、数据截止和 hash，同时提供静态图与离线
   交互页；未来目标发生前不声称效果提升；
6. 累积预登记数量的未来事件后，按相同报警面积评价并立即复审科学价值。

不得在本次 Stage 2S 分支上直接续跑，也不得先修代码再寻找可承接的科学问题。
