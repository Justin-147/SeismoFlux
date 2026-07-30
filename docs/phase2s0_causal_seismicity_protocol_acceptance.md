# 阶段 2S-0 因果近期地震活动协议验收

- 验收日期：2026-07-30
- 阶段：`Stage2S-0`
- 协议版本：`0.2.3`
- 实验 ID：`stage2s-causal-seismicity-development-v1`
- 本地结论：协议内容通过
- 执行权限：无；提交、推送及远端标签
  `v0.2.3-causal-seismicity-screen-protocol` 核验前不得进入代码实现或读取真实目录/目标

## 1. 外行能听懂的结论

这个阶段没有训练模型，也没有得到“预测变好了”的结果。它完成的是一份不能看答案再改规则的
实验约定：

1. 用长期 75 km KDE 作为旧方法 `S0`；
2. 用“长期底图 + 最近 30 天已知 M4+ 地震”作为新方法 `S1`；
3. 用“长期底图 + 再前一个 30 天地震”作为过去对照 `SP`；
4. 三版使用相同报警面积，只看新方法能否在未来地震位置排序和区域覆盖上稳定超过两项对照。

三版共享未来 M5–6 总率，因此本实验主要判断“更可能发生在哪里”，不判断未来地震总数是否增加，
也不提供具体发震日期或绝对概率。

## 2. 数据与时间边界

- `S0` 只重物化已通过 G1-LS 的 `fold_4` 长期 75 km KDE；
- `R_T` 只使用 `T-30d < origin_time_utc <= T` 且 `available_at <= T` 的 M4+ 地震；
- `RP_T` 只使用 `T-60d < origin_time_utc <= T-30d` 且
  `available_at <= T-30d` 的 M4+ 地震；
- 评价为三个滚动折、7/30/90 天、`M5_6=[5,6)`、600,000 平方公里；
- 2022–2023 已用于 Stage 2R 背景和带宽选择，只能称复用开发期历史筛查，不能称独立验证；
- Stage 2S 开发目标读取、独立验证目标读取和锁定测试读取均为 0。

本次验收没有打开或探查 `data/processed`。attempt ledger、非目标预检 receipt、target-read receipt
和 master seal 均不存在，因此唯一真实开发 attempt 未消费。

## 3. 防止偷看答案的封印

执行顺序冻结为：

```text
fold fit receipt
  -> 每个起报日 issue prediction seal
  -> fold prediction seal
  -> 三折 master prediction seal
  -> 才能构造 assessment memberships 和统一评分
```

后一折只有在前一折封印后才能取得自己的合法 fit view。较早评价期的地震以后可以作为已发生历史
或后一折成熟 fit 标签，但不得携带先前 assessment membership、命中、成绩或指标进入后续模型。

## 4. 验收证据

稳定机器身份：

- `configs/causal_seismicity_screen.yaml`：
  `a85df78348c0f033444db4c9e3edc81b70ef436da3b108139feab39cd49d8c42`
- fold manifest：
  `c3e2444e8892addd03d4c57526c007e2a861137dac50d5abe2e53bac004456e6`
- target-blind input contract：
  `50117a0c0cda0d14bd467b8f0d1032855cb5afab0aa2d968370313a933a95ff6`

本地回归：

- Stage 2S 协议：`27 passed`
- fold4/local-support 治理联合回归：`45 passed`
- Ruff：全仓通过，`245 files already formatted`
- strict mypy：`Success: no issues found in 236 source files`
- `git diff --check`：通过
- 受控执行：单 worker、各 BLAS 库单线程，至少保留 2 个物理核心

三路独立终审对同一科学内容快照均未发现科学 P0/P1；最终治理终审另提出 1 个“草案状态尚未
转为本地验收通过”的文档状态 P1。随后只完成这项状态转换，没有改变模型、数据、窗口、门槛或
随机规则；状态转换后的全部本地回归再次通过，该治理 P1 已闭合。

## 5. 科学价值复审

- 分类：`necessary_enabler`
- 对预测效果的直接提升：尚无
- 实质推动：把一个可证伪、无未来泄漏、不能看结果改规则的近期地震活动实验冻结下来
- 不确定性：2,000 次 Bootstrap 的尾部精度有限；区间条件于固定日历和已拟合折模型，不证明
  地震序列独立；39 个构造区只是稳健性分区；严格门可能造成较高假阴性
- 决策：停止继续扩写协议，进入最小实现和纯合成/非目标预检
- 停止条件：出现新 foundational P0、任何科学门失败或证据不足时，停止本路线并保留 75 km KDE

## 6. 本次精确提交白名单

只允许逐路径暂存以下 10 个文件，禁止 `git add .`：

1. `SEISMOFLUX_IMPLEMENTATION_HANDOFF.md`
2. `configs/research_protocol.yaml`
3. `docs/research_protocol.md`
4. `docs/scientific_value_review_and_model_composition.md`
5. `configs/causal_seismicity_screen.yaml`
6. `data/manifests/causal_seismicity_screen_fold_manifest.json`
7. `data/manifests/causal_seismicity_screen_target_blind_input_contract.json`
8. `docs/causal_seismicity_screen_protocol.md`
9. `docs/phase2s0_causal_seismicity_protocol_acceptance.md`
10. `tests/unit/test_stage2s_causal_seismicity_protocol.py`

所有 `src/seismoflux/anomaly_increment/kde_dev_*.py` 和
`tests/unit/test_stage4_kde_dev_*.py` 草稿都必须继续隔离，不论其 Git 跟踪或暂存状态。

## 7. 下一阶段

远端协议标签核验后进入 `Stage2S-1`：

1. 实现独立 `seismoflux.stage2s` 最小链，不导入 Stage 4 草稿；
2. 先交付不依赖目标成绩的 `stage2s_data_method_causal_timeline.svg`；
3. 用纯合成数据证明 `S0/S1/SP`、权重、共同率、补偿、报警、Bootstrap、区域、震群和封印全链；
4. code tag 后只做研究区、格网和 cell-zone 的非目标预检；
5. 预检通过后才登记唯一 attempt、认领唯一 target read，并执行一次真实历史开发筛查；
6. 结果必须同时交付静态对比图和离线交互回溯页；若通过，也只能授权新的前瞻封存协议。
