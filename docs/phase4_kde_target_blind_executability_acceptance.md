# 阶段 4A 0.4.4 目标盲可执行性修订验收

- 日期：2026-07-30
- 协议：`stage4-kde-development-v1` / `0.4.4`
- 计划标签：`v0.3.4-kde-anomaly-increment-protocol`
- 本地结论：`PASS`
- 独立工程审计：`PASS`，`P0/P1/P2 = 0/0/0`
- 独立科学审计：`PASS`，`P0/P1/P2 = 0/0/0`
- 开发目标读取：0
- 独立验证目标读取：0
- 锁定测试读取：0
- 预测效果结论：无

## 1. 本次验收了什么

本次只验收两个目标盲接口已经写清且可机械检查：

1. 四个受限空间工件的固定项目相对路径、byte count、整文件 SHA-256、Arrow/JSON schema、
   CRS、值约束和跨工件关系；
2. 开发 exposure/issue ID、`Asia/Shanghai` 到 UTC 的转换、`(T,T+h]` 窗口、三折训练/评估
   membership 与时间置乱 pool。

同时确认：

- 旧一次性 runner 不得使用；
- 现有会带入旧 qualification/runner/目标模块的 feature adapter/grid feature import closure
  不得直接进入新路径；
- 新薄代码只能按冻结 manifest 原序拼接 9/17/22 个逻辑特征，并让 observed/time/space 使用
  同一条重建—预处理—ridge 重拟合—prediction 冻结路径；
- 若下一次唯一纯合成端到端验收仍发现新的基础性 P0，停止异常路线并保留 75 km KDE。

## 2. 没有改变什么

以下科学合同保持不变：

- `B0/C0/B1/B2` 与 75 km KDE；
- 三个滚动折和 7/30/90 天；
- `M5_6=[5.0,6.0)`；
- 600,000 平方公里固定报警面积；
- 2,000 次事件块 Bootstrap；
- 1,000 次时间置乱和 1,000 次空间置乱；
- 候选族单步配对 `maxT`；
- 固定面积严格召回至少 `+5 pp`，且家族同时 95% 下界排除无改善；
- 跨折、跨区、去最大贡献区、单一开发 attempt；
- 独立验证和锁定测试继续禁读。

## 3. 本地验证

当前目标盲协议与置乱重建原语测试：

```text
24 passed in 9.03s
```

命令只运行：

```text
tests/unit/test_stage4_kde_development_protocol.py
tests/unit/test_stage4_anomaly_increment_placebo_features.py
```

并固定 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、
`OPENBLAS_NUM_THREADS=1`。此外：

- Ruff check：通过；
- Ruff format check：通过；
- strict mypy：通过；
- `git diff --check`：通过；
- 测试后 CPU 总占用抽样约 21.3%，没有残留 Python 进程。

本轮没有打开 `data/processed`，没有读取四个受限工件的内容，没有读取目标、命中、漏报、成绩或
未跟踪的废弃 runner 草稿。一次补充的旧协议兼容性测试得到 `37 passed, 1 skipped`；唯一 skip
只报告旧 R2 四工件在该 worktree 不可用，没有读取其内容，不作为本验收的通过依据。

## 4. 冻结文件身份

| 文件 | SHA-256 |
| --- | --- |
| `SEISMOFLUX_IMPLEMENTATION_HANDOFF.md` | `656bb6173379fbd9473fd2e8597a9645356ba5c0dc844b415e0aaa8c292facf6` |
| `configs/anomaly_increment_kde_dev.yaml` | `cdc5d1328d687e96952791d07f0e6a7b11b2b6b73370df1f751475199718ab3c` |
| `configs/research_protocol.yaml` | `e03ca0bd19d306b961ce252cf6eb4fbee85f85b5ea456c38ed63bef0dcecfeb1` |
| `data/manifests/anomaly_increment_kde_dev_inherited_contracts.json` | `9f4ef201008a1f1bd2d6e8b8d1c1ed4adee2855779747702bb4e93305c3ecbbe` |
| `docs/anomaly_increment_kde_dev_protocol.md` | `544a14a189d153a624c470667f90f5f4fab4f767453100d66d70bf6b57ff4dc7` |
| `docs/phase4_kde_target_blind_executability_amendment.md` | `c54d6acea35f3a2814b8e00942b70e496eee2d092056e324481c81f137816de5` |
| `docs/research_protocol.md` | `4e0c7a5187388b7af30aaa5cbb1d211313562f7050dfb0509b051e5a36f4c58c` |
| `docs/scientific_value_review_and_model_composition.md` | `cab97a794edc93c92354da3dddeda390415738a440d223e5ac822deb0f5d2c0b` |
| `tests/unit/test_stage4_kde_development_protocol.py` | `ea8df8566ac5dcf3be6b0e56cc28ab0c514ef2a779e06121898326985cc131ea` |

## 5. 科学价值复审

- `science_value_category`: `necessary_enabler`
- `direct_prediction_improvement`: 尚无
- `evidence`: 真实重训链所需的输入和日期接口已冻结，且两路独立审计均通过；尚未运行真实数据。
- `decision`: 允许提交、推送和冻结 `0.4.4` 协议；之后只进入一次纯合成薄代码验收。
- `next_scientific_test`: 纯合成 observed/time/space 同路径重建、9/17/22 特征组装、fit-only
  预处理、ridge 重拟合与 prediction 冻结。
- `stop_condition`: 该验收若出现新的基础性 P0，停止异常增量路线并保留 KDE，不再追加工程修订。

该结论只表示下一项科学检验已有受控入口，不表示异常提高了地震预测效果。

## 6. 外部门

本地验收通过后仍必须依次完成：

1. 精确提交并推送当前协议文件；
2. 创建并推送 annotated tag `v0.3.4-kde-anomaly-increment-protocol`；
3. 回读远端 tag 对象和 peeled commit，确认与本地一致。

三项全部完成前，不得进入真实输入核验或代码冻结。
