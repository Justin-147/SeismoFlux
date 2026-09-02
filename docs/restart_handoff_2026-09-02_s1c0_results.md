# SeismoFlux S1-C0 科学成绩恢复交接（2026-09-02）

> 这是当前最高优先级恢复入口。科学效果优先；不得用工程工作量代替预测证据。

## 2026-09-02 S1-C0 最终科学安全点

- S1-C0科学结果提交：`33ad8bc8aaa76f7e85d3949bfe82471c42b7521f`；已推送至
  `origin/codex/p2r-multitask-multidata`并远端逐字回读一致。
- 封存与文件完整性仍为`GO`、P0/P1=`0/0`，attempt2预测和原始成绩不修改、不覆盖。
- 成绩解释复审发现的两项P1已经由读结果前冻结的诊断和独立重算关闭，最终门控P0/P1=`0/0`：
  1. L0等密度的5/147来自固定row/column并列顺序，不是均匀随机基线；599,982.100 km²占全国
     研究区6.3724%，随机面积期望约9.37/147。L0固定前缀只保留作审计，不再用其CI作主证据。
  2. 震级比较的445个M5+事件映射到314个固定首震震序；20,000次震序聚类重采样95%CI为
     0.006184–0.016340 nats/事件，严格高于0。去掉19事件的最大震序后均值仍为0.011210。
- 地点正方向没有消失：L2相对可解释的L1区域常率净增14个/+9.52个百分点，L3净增22个/
  +14.97个百分点，四个年代折均为正；但最终泛化仍必须由S1-C1和后续留出确认。
- S1-C0科学签收为
  `GO_S1_C0_SIGNOFF_AND_PROCEED_S1_C1_NO_CHAMPION_NO_HOLDOUT`。签收对象是“冻结attempt2 +
  science_diagnostic_v1控制性解释附录”的复合记录；不重训、不改模型、不读取holdout/audit/locked。
- 诊断配置SHA-256：`2b47b432d63522edf43c4062f740ea7d4a1ec3a9990dc38be14ce96fba74cdf5`；
  结果SHA-256：`86e448dc9c2fef2b054ec58dff9ba50986a670c0f4c87f974d1d928658a744cd`；
  清单SHA-256：`66bafa35de218a7833f63f0b8fde413bcfb4be855f96bf8acef6aad69f838300`。
- 五张静态图和完全离线交互回放已经生成、哈希复核和目视核验；完整索引见
  `docs/s1c0_scientific_acceptance_2026-09-02.md`。交互页SHA-256为
  `e9180c0b4c3c4a3abab23ab1c340f8405acec03bca14a1a701c4fd1ae2b96ad1`。
- 最终串行S1回归`136 passed`；Ruff、格式、strict mypy和差异检查通过。没有后台训练进程。

## 当前已完成

- 工作树：`D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata`
- 分支：`codex/p2r-multitask-multidata`
- 冻结代码：`538241f9be6ad1a3ea424adac1a342de5ea0683a`，已推送并远端回读。
- S1-C0 `attempt2` 已完成四折 `predict → 封存 → score`；总封印：
  `858923507066e5a1d2a605baf723d5300233b96d43fdd401b7482d59dbefe5a8`。
- 原始成绩88,136行；科学摘要SHA-256：
  `a419959acfb6b0b1eb7ff91726f2c8911e20ebaed3b85a8808e247cd13185871`。
- 独立复核结论：`GO`，P0/P1=`0/0`；未打开holdout、audit或locked test。

## 首批科学结果

主地点口径为24小时目录延迟、未来30天M5–6、600,000 km²报警面积、严格同格命中、
30天/75 km固定首震锚点，共147个独立震序：

| 模型 | 命中 | 召回 |
| --- | ---: | ---: |
| L0 等密度固定并列前缀（仅审计） | 5 | 3.40% |
| L1 区域常率 | 32 | 21.77% |
| L2 长期因果KDE | 46 | 31.29% |
| L3 长期背景+最近30天地震 | 54 | 36.73% |

L2和L3相对可解释的L1在四个年代折均为正，说明目录的因果空间结构在同面积下有直接开发价值。
但L3仍漏掉93/147个震序；L3相对L2在25个时长×面积组合中为10胜、8平、7负，近期分量只能视为
短期有希望，不能称为最终冠军。负二项时间增量不稳定，停止围绕T1追参；1900+ M5+长历史对震级
尾部有经震序依赖校正后仍为正的小幅改善，但不能替代完整M4+震级分布。

本轮只用了地震目录。静态构造、动态异常、地球物理场、代表复杂模型和多数据组合仍待S2–S5检验。
完整解释见`docs/s1c0_scientific_results_2026-09-02.md`。

## 下一恢复顺序

1. S1-C0已经科学验收、提交、推送并远端回读，禁止重新优化或扩展该结果；
2. 进入S1-C1时，先在读取任何新成绩前写死因果局部完整度盲协议；
3. 保持四个开发折、147个目标震序、全国评分域、30天主时长和600,000 km²报警面积不变，只改变
   训练目录的局部完整度处理；
4. 局部Mc过高只影响对应固定支持范围的训练资料，不缩小全国预测与评分区域；
5. 检验L2/L3相对L1的正方向能否保留；无稳定增量则停止该方向，转向下一项最有科学价值的数据。

禁止读取或运行2020–2022 holdout、2023+ audit、locked test；不得触碰`science_first`既有Stage4草稿。
`seismoflux`心跳为`ACTIVE`、每30分钟一次。最近检查没有SeismoFlux后台Python进程，CPU约26.8%，
GPU约1%；大型任务最多3个外折线程、数值库单线程、至少保留2个物理核心。

```yaml
current_stage: S1_C0_remote_closed_ready_for_S1_C1_blind_protocol
science_value: direct_positive_development_evidence_with_mandatory_followup
next_scientific_test: S1_C1_causal_local_completeness_main
holdout_allowed: false
audit_allowed: false
locked_test_allowed: false
heartbeat: ACTIVE
```
