# 阶段 2S-1 因果近期地震活动无效尝试停止验收

- 验收日期：2026-07-30
- 实验 ID：`stage2s-causal-seismicity-development-v1`
- attempt ID：`stage2s-causal-seismicity-development-v1-attempt-1`
- 代码提交：`4188523991926c51a7fbd9314d36395cc9bfad62`
- 代码标签：`v0.2.3-causal-seismicity-screen-code`
- 最终状态：`invalid`
- 路线决定：`STOP`
- result tag：不创建；没有 whole-run result

## 1. 验收结论

唯一正式历史开发 attempt 已消费。最后一个成功进度为 `fold4_support_rebuilt`，随后主科学路径
在 fold-fit receipt 和第一批预测封印前发生未知异常；失败记录器又因 canonical JSON 双编码缺陷
发生第二个异常并覆盖原始原因。没有形成任何模型评分、预测效果图或离线历史回溯成绩，因此本文件
既不是正结果，也不是负结果。

Stage 2S 按 foundational P0 停止。不得修改 code tag 后重开目录、补预测封印、恢复当前 attempt
或对同一目标重新评分。

## 2. 阶段证据链

通过：

- 远端 protocol/code tag；
- code import closure；
- 24 物理核心、1 worker、六类数值线程为 1；
- 无目标 study-area、50/25/12.5 km 格网、15,697 个 25 km 格、39 区映射；
- preflight receipt；
- attempt ledger；
- target-read receipt；
- 真实目录单次读取、冻结文件/内容/schema/40,898 行校验；
- fold4 支持域重建。

未形成：

- fold fit receipt；
- issue/fold/master prediction seal；
- S0/S1/SP 预测评分；
- Bootstrap、区域、震群或延迟结果；
- whole-run record；
- 六个正式结果工件；
- result tag。

## 3. 不可变本地凭据

| 记录 | 文件 SHA-256 |
|---|---|
| non-target preflight | `e216194a6ec035a2788d3f55b7e949f5afd21e9584d6d422de6c9a61baa9d71f` |
| formal attempt | `fef0606d68aa6d1da721f634c04fd53df10500b522aa6b38307415194b560f5d` |
| target read | `cd1482d63dc8add9354f005b647781371eec62ee882fe7fac23adc3c0ff71cfe` |

凭据均位于 Git ignore 路径。不得公开其完整 payload，不得删除或覆盖。

## 4. 失败与恢复审计

最后一个成功进度之后的原始 science 异常没有被 CLI 保存，现已不可恢复。不能把具体原因猜成
S0 构造、折内拟合或预测封印失败。

可确认的 terminal/finalizer 二次失败为：

```text
reserved canonical JSON key at
$.bindings.aligned_grid_identity.layers.12.5.cell_size_km:
$seismoflux_type
```

异常捕获器准备 terminal 时先重读同一 preflight receipt，把已编码的浮点 marker 再次送入
canonical encoder，触发保留键拒绝并覆盖原始异常。因此 terminal 缺失是已确认的 foundational
治理异常；它不授权再次启动程序，也不授权手工补写。

## 5. 科学价值复审

- `science_value_category`: `no_material_progress`
- `evidence`: 没有预测 seal、评分或效果指标
- `decision`: 停止 Stage 2S，保留长期 75 km KDE
- `next_scientific_test`: 先做目标盲科学路线复审；若仍选择近期地震假设，再在新前瞻协议下从
  未发生的新 cohort 起报并封存候选模型
- `stop_condition`: 禁止重用本次 attempt 或既有 2022–2025 目标进行修复后重跑

## 6. 对整个项目的含义

这次失败没有证明近期地震活动无用，也没有证明长期 75 km KDE 足够好。它只说明当前 Stage 2S
没有产生可用科学证据。下一项工作必须先做目标盲科学路线复审，而不是继续扩大测试、界面或模型
复杂度；只有复审选定并预登记前瞻路线后，才可生成不可回填的真正前瞻预测。
