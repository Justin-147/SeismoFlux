# 阶段 4A 0.4.4 纯合成验收与基础性停止

- 日期：2026-07-30
- 唯一蓝图：`SEISMOFLUX_IMPLEMENTATION_HANDOFF.md`
- 协议：`stage4-kde-development-v1` / `0.4.4`
- 协议提交：`f2ba7a7383a1c7ab7900f3881497a0c9c8e7be9a`
- 协议标签：`v0.3.4-kde-anomaly-increment-protocol`
- 代码阶段总验收：`FAIL`
- 失败级别：新的 foundational P0
- 开发目标读取：0
- 独立验证目标读取：0
- 锁定测试读取：0
- 唯一开发科学 attempt 消耗：0
- 预测效果结论：无

## 1. 一句话结论

纯合成代码可以跑通，但正式上场前的独立审计发现：程序核验了“25 km 格子属于哪个预冻结区域”，
随后却丢掉了这张对应表，因而无法完成协议强制的跨区域和“去掉贡献最大区域”检验；当前拟合还把
7 天积分错误压成每格一行。按唯一蓝图的硬停止条款，本异常增量路线到此停止，保留 75 km KDE，
不得继续补丁、代码冻结或读取真实结果。

## 2. 中断恢复后的工程复核

当前工作树从已经推送的协议提交恢复，分支和远端跟踪引用均指向
`f2ba7a7383a1c7ab7900f3881497a0c9c8e7be9a`。在
`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`
下，八个未跟踪薄组件及纯合成链回归结果为：

```text
229 passed
Ruff check: PASS
Ruff format check: PASS
strict mypy (15 files): PASS
```

测试覆盖 calendar、input adapter、background、placebo mapping、fit、statistics、seal 和
observed/time/space 同路径纯合成链。身份置乱精确复现 observed，非身份置乱会改变特征、设计矩阵、
预测和无目标统计量，coverage 保持不变。

这些结果只说明局部工程行为能运行。它们没有读取真实目标，没有产生信息增益、固定面积召回、
Molchan 曲线或置乱显著性，因此不能称为预测效果。

## 3. foundational P0：区域映射在正式接口中丢失

协议同时冻结了两个互相冲突的要求：

1. `configs/anomaly_increment_kde_dev.yaml:357-368` 冻结 39 个非空区域，
   `configs/anomaly_increment_kde_dev.yaml:1051-1077` 强制按区计算贡献并执行
   leave-one-region-out；
2. `configs/anomaly_increment_kde_dev.yaml:386-393` 又把正式 adapter 返回值锁成
   `issue_tables`、`snapshots_by_issue_id`、`query_grid` 和
   `construction_stratum_by_state_id` 四项。

`src/seismoflux/anomaly_increment/kde_dev_inputs.py:1006-1058` 会加载并核验
`cell_mapping` 中的 `cell_id → construction_zone_id`，但构造四字段返回值时把该对应关系丢弃。
`src/seismoflux/features/anomaly/grid.py:28-39` 定义的 `query_grid` 不含区域字段；
`construction_stratum_by_state_id` 只描述异常实体，不能给全部地震和格网分区。

在 `0.4.4` 不变时，正式 runner 没有合法补救：

- 重读 `cell_mapping` 会破坏单一受控 adapter 边界；
- 增加第五个返回字段会修改冻结接口；
- 用 geometry 重新生成区域会违反“只核验、不返回、不重建区域”；
- 隐藏缓存或从目标推断区域同样不合法。

因此强制区域门不可计算，不存在合法正式 runner。根据
`SEISMOFLUX_IMPLEMENTATION_HANDOFF.md:723-726` 和
`configs/anomaly_increment_kde_dev.yaml:1224-1227`，这已经单独触发异常路线硬停止。

## 4. 次级数值闭包缺陷：7 天补偿积分没有按冻结数学定义执行

冻结目标需要对每个 `issue × cell` 展开 7 天 composite-midpoint 项，再计算：

```text
sum_t width_t * background_mass * exp(decay_t * Xβ)
```

公共原语 `src/seismoflux/anomaly_increment/integration.py:152-195`
已经提供 `expand_midpoint_compensator_terms`。但是未跟踪的
`src/seismoflux/anomaly_increment/kde_dev_fit.py:718-788` 只给每个拟合行传入一个 exposure
和一个 decay，然后进入非线性的指数目标。先合并 exposure 再做一次指数，通常不等于逐日中点
指数项之和。

现有纯合成测试使用任意 decay 向量，只证明调用链连通，没有证明正式 Poisson 目标函数正确。
该项本可在代码冻结前、不改变科学协议地修正，因此不把它列为触发硬停止的第二个 P0；但即使区域
P0 不存在，当前 fit 也不能进入正式代码冻结。现在决定性 P0 已经触发，不再追加这项修补。

## 5. 独立 seal 终审的未修复 P2

seal 只读终审结论为 `P0/P1/P2 = 0/0/1`。唯一 P2 是
`schema_version` 和 ledger entry `sequence` 使用普通 `== 1`，使 JSON boolean `true`
可被 Python 当作整数 `1` 接受，违反精确 schema 的 fail-closed 类型要求。

该缺口不会增加 attempt 或 target-open 次数，所以不升为 P1。由于更高层 foundational P0 已触发
整条路线停止，本轮不再为 seal 增加补丁；该 seal 不纳入代码冻结。

## 6. 硬停止边界

立即冻结以下决定：

- 不创建或提交正式 `kde_dev_runner.py`；
- 不提交、推送或打 `v0.3.4-kde-anomaly-increment-code/result` 标签；
- 不修订为 `0.4.5` 来复活同一异常路线；
- 不打开真实目录、开发目标、独立验证或锁定测试；
- 不注册或消耗 `stage4-kde-development-v1-attempt-1`；
- 不进入阶段 4B、阶段 5 或大型模型；
- 不把纯合成图或工程测试图冒充真实方法效果；
- 保留已经通过 G1-LS 的 75 km KDE 作为当前最好合法背景。

当前两个冻结 ledger 文件都尚未创建：

```text
data/manifests/anomaly_increment_kde_dev_attempt_ledger.json
data/interim/stage4/anomaly_increment_kde_dev/target_read_ledger.json
```

因此没有已注册 attempt，也没有 target-read 的 `0 → 1` 转换。

## 7. 未跟踪草稿隔离清单

以下草稿只用于解释本次失败，保持未跟踪、不得纳入当前或后续代码提交：

| SHA-256 | 文件 |
| --- | --- |
| `10677cda5337fe4e0ca7ddcfff5a0f763db5007eda7a5d25227a05f8238792ed` | `src/seismoflux/anomaly_increment/kde_dev_background.py` |
| `bfd485e886ebadc7022b315c965bb4ddb15495c0bede9d39a61145ab163492f4` | `src/seismoflux/anomaly_increment/kde_dev_calendar.py` |
| `1032d109010686f339665b9f42e09c7618a1d4383b2df32c89a56f7ccd57a029` | `src/seismoflux/anomaly_increment/kde_dev_fit.py` |
| `407406aa2b1916bbc08141a0ba2e23435987646d2a0ab5be03e82bf30f1c3d4f` | `src/seismoflux/anomaly_increment/kde_dev_inputs.py` |
| `8280e95a7bbd6174b5dace8692c0fbd7e829e1204d1ca6e39619fab3ba95726d` | `src/seismoflux/anomaly_increment/kde_dev_placebo_mapping.py` |
| `670c687030faab76d61c54272713cdc33a8aefc1ca00f6af67a66b2a0aec1845` | `src/seismoflux/anomaly_increment/kde_dev_seal.py` |
| `8730b1d8fc5753825570d8ac2e34c563be1bfcc3f3abb5c7779af323157e0de6` | `src/seismoflux/anomaly_increment/kde_dev_statistics.py` |
| `c6482b383d443ebf25b9a44c7d4fb9715c6b04c27e0a0d781ae7606d46f6ae6f` | `tests/unit/test_stage4_kde_dev_background.py` |
| `2a836cc7c7f3be97eb49755c038932b34efcfc4a1e438ed635349ebfb5972441` | `tests/unit/test_stage4_kde_dev_calendar.py` |
| `eb37107a692b7f568b21579c62d738e9951eb3d0ffd6bbd53b0fed6a5c7794f3` | `tests/unit/test_stage4_kde_dev_fit.py` |
| `795c13f78e6092310d3313fb96d77113f8227a37651e97141fc06221adb17446` | `tests/unit/test_stage4_kde_dev_inputs.py` |
| `c2bc8eed50be2e2e8d68b4f4f9a29c8926667b0814f76d889263b9990a415bcd` | `tests/unit/test_stage4_kde_dev_placebo_mapping.py` |
| `1cbcdbfb8357ab73914b230609ba4f28c9c05fe94a5a51d621b2dabe2e1420e0` | `tests/unit/test_stage4_kde_dev_seal.py` |
| `582b28d99bb10c7c09007087f61ed703e9075707e2efc20e6ab3cd031db730be` | `tests/unit/test_stage4_kde_dev_statistics.py` |
| `8c1fae822614a2f220e86b70e434a107d06531b58d13cc094b53abe4606f907d` | `tests/unit/test_stage4_kde_dev_synthetic_chain.py` |

这些哈希只是中断恢复和防止误提交的工作树证据，不是科学来源证明。

## 8. 即时科学价值复审

- `science_value_category`: `no_material_progress`
- `direct_prediction_improvement`: 无
- `evidence`: 229 个工程测试通过，但正式协议要求的区域稳健性不可计算，时间补偿目标也没有按
  冻结数学定义实现；没有读取真实目标或产生任何预测效果指标。
- `decision`: `stop_anomaly_increment_and_retain_75km_KDE`
- `next_scientific_test`: `preregister_stage2s_causal_two_timescale_seismicity_screen`；只冻结长期
  75 km KDE 与一个近期因果地震 KDE 的最小组合及一次开发检验，不重开异常路线。
- `stop_condition`: 不得以修改 `0.4.4` 接口、增加来源封条、复杂异常模型、阶段 4B、阶段 5 或
  锁定测试读取绕过本停止结论。

本阶段的有效产出是及时排除一条不能合法评分的路线，避免浪费唯一真实 attempt；它不是预测效果
提升。静态图和交互展示继续保留为真实未来隔离结果的交付物，不能在没有结果时制造好看的假结论。

## 9. 只读路线复审后的推荐（尚未开始新阶段）

在不读取 `data/processed` 字节、真实目标或成绩的前提下，已比较三条路线：

1. 长期 75 km KDE 加一个近期因果地震活动 KDE；
2. 长期构造先验；
3. 修补 `0.4.4` 异常合同。

推荐第一条，拟另立“阶段 2S：因果双时间尺度地震活动筛查”。理由是 75 km KDE 已有真实正证据：
四个开发折信息增益均为正，最终验证为 `+0.40215 nats/事件`，95% 区间
`[+0.23564,+0.56286]`；但它还没有证明固定面积区域召回提高。最近的直接科学问题应是：

> 在长期 75 km KDE 不变时，只加入起报日前已经可用的近期地震，能否在 600,000 平方公里固定
> 报警面积下提高未来 M5–6 区域召回，并优于时间错位对照？

构造先验暂不推进，因为当前断层和危险性快照只能证明在 2026-07-13 可用，
`historical_model_eligible=false`，不能合法回填到更早的历史起报日；修补异常合同则直接违反本次
硬停止。

这只是下一协议阶段的推荐，不是授权。必须先完成本失败阶段的验收、提交和推送；随后另行把
`S0=长期 75 km KDE`、`S1=长期 KDE+一个近期因果 KDE`、时间错位对照、固定面积门、区域稳健性
和停止条件写入唯一蓝图并独立验收。在此之前不得写 Stage 2S 模型代码或读取目标。
