# 最新交接：S2-A断层几何增量

更新时间：2026-09-02 22:55（北京时间）。当前：`S2A_IMPLEMENTATION_ACCEPTED_PENDING_SYNC`。
本文件接替`docs/restart_handoff_2026-09-02_s1c2b.md`；旧文档与结果保留。

## 1. 一句话状态

项目尚未全部完成。S1目录比较已验收推送，现有小幅地点提升保留；现在检验加入断层图能否少漏震区。
首个有限S2-A协议已经通过配置检查和独立科学复审，并完成提交推送。新断层层与只读复用
目录的预测/评分接线也已完成，71项必要合成验证及独立科学复核通过，正提交推送。
没有S2-A真实训练/评分进程，也没有新的预测效果。
验收见`docs/s2a_protocol_acceptance_2026-09-02.md`。

S0数据/任务已完成；S1有限目录基准已完成；S2断层/危险性/应变正在做首轮几何；S3异常、S4有限
机器学习、S5组合消融、S6综合稳健性和论文展示尚未完成；S7外部检验并行，不能阻塞历史研究。

## 2. 已完成证据和不再重做的事

- S1科学提交`544360e34b139581508deba5c62d14bccf46d535`，闭合`7164e3b8cd2b8af939f18e40337d6c89ac476295`，
  已推送并远端回读一致。主任务30天Ms5–6、60万km²，L3为54/147，多尺度58/147，年龄57/147。
  多尺度新增14/丢失10，差值区间跨零；这是开发期小增益，不能称最终确认。
- 95%/Ms审计已结束。两份本地目录原数值无转换、无因类型空缺删事件，不需全部重跑。C1被过严
  百分比门挡在预测前；已用C2A做实际固定参数敏感性，主结果54→52/53，没有稳定收益，分支结束。
- S1最终图和离线回放位于`outputs/multitask_s1/s1c2b_catalog_models_v1/rendered_v2`，不要重新运行。
- 不重开95%/80%比例、旧ETAS/负二项、C2A掩膜或C2B参数搜索。不可把未完成的S2写成已提升。

## 3. S2-A精确入口

工作树：`D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata`；
分支：`codex/p2r-multitask-multidata`；Python：`D:\AIPred\SeismoFlux\.venv\Scripts\python.exe`；
数据根：`D:\AIPred\SeismoFlux\data`。

续接先读总蓝图第1.9—1.14节、本文件、以下两个协议及研究起点：

1. `configs/multitask_s2_a_fault_geometry.yaml`；
2. `docs/s2a_fault_geometry_methods_2026-09-02.md`；
3. `docs/s2_static_data_research_start_2026-09-02.md`。

已有519简化段、7,215可用精细迹线，只取几何；2026收集快照是历史描述性研究，不改原历史不可用
标记，不混入上次强震年、当前离逝率、危险性或速率。复用15,697格和39个目标盲固定块。
六新模型：每源纯断层、目录+精细断层、目录+粗化断层；三个距离尺度、每混合13候选，不扩张。
内层I2目录明确按原规则回退K75，I3只用更早数据选择；混合只看I2/I3且标签在外折前成熟。
旧413份核和外折目录预测只读复用，输出新目录`outputs/multitask_s2/s2a_fault_geometry_v1`。

## 4. 最高优先级安全动作

先完成协议的配置检查、独立科学复审与提交推送，再做最小实现。随后必要合成测试及实现验收推送，
立即运行真实有限对照；不要创建新的地图清洗系统、PSHA平台或反复溯源工程。
四折预测全部保存后才统一评分。每个检查点落盘即更新此文档；旧完成产物不覆盖。

目前没有S2实现/运行PID。后续恢复必须先确认唯一进程和检查点，不按旧S1 PID重启。
2折线程默认、最多3折，BelowNormal、数值库单线程、至少保留2物理核。日常检查单进程。
不读2020—2022留出、2023+审计、锁定测试；不取未来目录，不改根库、冻结P1、science_first
的Stage4未跟踪草稿。代码/文档/聚合证据可公开，逐事件数组/案例地图/离线HTML仅本机。

科学复审：本次是新增资料有限检验的必要准备，尚无S2效果；方向仍是充分用数据提升预测，没有
偏离目标。今后的最小证据是相同面积下多命中/多漏几个独立震区，而不是新增多少代码或通过多少测试。
心跳保持每30分钟，阶段变化后同步更新其恢复入口，不能继续指向旧S1运行。

## 5. 22:36协议验收

六项配置检查通过、独立科学复审无阻断问题。协议SHA-256为
`d6e19dca67063030e8eafdfd766f13f2310a5e52790cc4b2b7fe8707cf58b5c9`。
22:33资源核验无SeismoFlux Python任务，两CPU16%/6%，总约11%，没有占满核心。
本轮先同步协议/方法/配置测试/验收/交接及蓝图，再分别实现几何层和预测/评分接线。
当前尚无S2预测器和成绩；不得把本节误当已可恢复真实计算。

## 6. 22:42已进入有限实现

协议提交`5ffd1013640ccea220a23cc3e47dcb4e163e8f95`已推送至
`origin/codex/p2r-multitask-multidata`；22:36:54远端回读一致。上节“待同步”由本节取代。

新模块位于`src/seismoflux/multitask_s2/`。几何模块已完成合成验证，预测与评分模块仍在实现，
不得因文件存在便启动真实计算。命令入口`scripts/run_multitask_s2a.py`分别支持predict/score，
输出仅限`outputs/multitask_s2/`；旧目录结果只读。必要实现测试/科学复核/验收提交推送后才能运行。
评分将按折留检查点，防止中断后整组重做。代码和测试开发不构成新的预测证据。

## 7. 22:55实现验收，待同步后立即计算

最小几何/预测/评分与CLI已完成，见`docs/s2a_implementation_acceptance_2026-09-02.md`。
主代理集成验证71项通过（9.51秒）、Ruff通过；独立科学复核未发现阻断问题。没有改变协议参数、
数据、原目录结果或目标口径。实现提交远端确认后，不再扩展实现，立即启动真实有限计算。

预测入口（从活动工作树、设置PYTHONPATH=src及数值库单线程后，由隐藏BelowNormal进程运行）：

```powershell
& 'D:\AIPred\SeismoFlux\.venv\Scripts\python.exe' scripts/run_multitask_s2a.py --phase predict --project-root 'D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata' --data-root 'D:\AIPred\SeismoFlux\data' --workers 2
```

只有`outputs/multitask_s2/s2a_fault_geometry_v1/prediction_manifest.json`完整并核验后，才将phase改为
`score`，逐折评分；两阶段共用同一run.lock，不能同时启动。还没有运行PID或真实检查点。
图件/交互薄渲染器正在另行准备，只用合成数据，不阻塞预测，不将草稿当最终交付。
