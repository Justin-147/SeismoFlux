# 最新续接：S3-A异常地点/时间试验

更新时间：2026-09-03（本轮用户明确授权并更新最高科研准则后）。
状态：`S3A_REFERENCE_PREPARATION_RUNNING_TRAINING_PRIMITIVES_ACCEPTED_RUNNER_PENDING`。
接替`docs/restart_handoff_2026-09-03_s2c.md`，旧文档保留S2结果与95%/Ms审计证据。

**最高优先级：公开确认与推送已解决，直接续接S3真实日期背景和训练装配，不再等待同一授权。**
本轮用户紧接载荷说明明确回复“授权。”；精确记录为 `s3_public_aggregate_authorization_2026-09-03.md`。
既有输入提交 `08e8d1ce1707076de5104281fb7449b8d0b1634c` 原样保留，允许向公开
`Justin-147/SeismoFlux` 推送其中的聚合统计及本轮指导/交接补记。此前拒绝是历史状态，不是当前阻碍。
本次授权不包含原始记录、逐事件数组、震例地图和含事件坐标的页面。不要重算已完成输入。

最高科研准则已更新到唯一蓝图1.22、AGENTS和总方案第10节：肯定并按适用任务采纳可复核的小幅
或局部进步，不以统一显著性、跨全部折稳定或最小增益否决；科学证据阶段与局限仍如实说明。
心跳保持每30分钟，按 `SCIENCE_DECISIONS_AND_REPORTING.md` 去重，只报变化与当前工作。
本轮更新尚不产生新的预测成绩；S3训练执行器还需装配，不能说153期预测已经完成。
最近进程核对（11:02）无Python/pythonw；开始新计算前仍先检查重复实例。
11:24:52远端已回读一致：`85e223d53bb915513fcc0bf8ef5bd26302aef4a8`，包含既有输入提交和本轮
最高科研准则、授权及去重文档。公开等待点闭合，不再作为心跳阻碍或复述旧等待经过。

本轮已新增 `multitask_s3/preparation.py`，为全部153个真实A起报生成可复用的目录背景和20列
报告特征缓存；不是重跑输入水位，也不产生异常模型成绩。每期保存三个目录尺度、R30和全国
长期率，五个时限复用，避免每种模型重复计算。最多3线程，当前计划2线程，数值库1线程。
本机目录为 `outputs/multitask_s3/s3a_prepared_v1`，恢复命令在首次启动命令后加 `--resume`；
若遗留 `preparation.lock`，先核对PID和命令不存在后再处理锁，不创建重复实例。
纯缓存往返/身份与形状检查已通过；这不是预测效果证据。输入模块原78项检查本轮再次通过。
纯内存 `training.py` 和 `targets.py` 已实现并通过合成验收，尚未启动真实异常拟合；训练执行器
仍需接入两模块与缓存，再按全部外折预测保存后统一评分。不能把缓存完成称为S3训练完成。

11:31:59准备代码提交 `a4165f8ed5faab18d462e96efca39a2cb3e291cf` 已推送并回读一致。
11:32:26启动唯一准备任务PID42188，BelowNormal、Hidden、2工作线程、数值库1线程。
11:33:36检查点完成4/153期（2.6%），0失败；11:33:47进程活跃，整机CPU瞬时18%。
日志在本工作树 `data/interim/s3/s3a_prepared_v1.stdout.log` 和同名 `stderr.log`；
进度在 `outputs/multitask_s3/s3a_prepared_v1/preparation.json`。首次启动后不得再运行无`--resume`
的新实例。旧PID只是本次检查记录，后续心跳必须核对当前进程；完成后不重启准备。

训练与标签原语验收见 `s3a_training_assembly_acceptance_2026-09-03.md`。本轮相关合成检查132项
通过；Ruff通过。下个最小工作是新增薄`runner.py`和真实评分衔接，不是继续抽象训练框架。
已确认接口：

- `targets.prepare_anchor_ids(frame)`：对完整限定历史一次建立两个震级带的固定首震；
- `targets.build_window_targets(frame, issue_time=..., horizon_days=..., available_by=..., cell_indices=..., cell_count=..., anchor_ids_by_band=...)`
  返回`spatial_counts_ms4`、`count_ms5plus`和`bands`内事件ID/格号/首震掩膜，仅本机保存；
- `training.S3TrainingSample(issue_time_utc, features, background_log_mass, offset_ms5plus, spatial_event_counts, count_ms5plus)`；
  features是既有20列已变换矩阵，不能再次log/asinh；
- `training.S3InnerBlock(block_id, training_samples, validation_samples)`，
  `training.select_and_fit(training_samples, inner_blocks=..., design='COV'/'SNAP'/'DYN', areas_km2=...)`；
  内块训练标签用各`label_fit_cutoff`，内验证标签最多到外层`label_fit_cutoff`，不要把未来标签提前可用；
- 拟合对象`predict_log_mass(features, background_log_mass)`、`predict_log_mean(features, offset_ms5plus)`、
  `predict_calibrated_log_mean(offset_ms5plus)`、`to_dict()`；两个正式震级共享乘子，分带log次数应从
  Ms5+的log均值加各带目录基准率比例得到，不能声称新学得震级比例。

准备缓存每期保存`features/kernel_25/kernel_75/kernel_150/r30_log_mass`及三个震级带每日期望次数；
用既定权重按h混合、次数乘h。只读核缓存，无需重跑KDE；样本按calendar选择，所有外折预测保存
后才接外层评分。`runner.py`、`scoring.py`与本轮置乱执行器尚未实现，不假定它们已存在。

## 1. 当前目标、完成情况

目标是用尽可能充分的数据改善地点、时间、震级，尤其在同样报警面积下少漏独立地震区域。
项目未完成。S0数据/切分、S1目录有限比较、S2几何/速率/应变有限比较完成；S3现在进入异常新路线，
S4有限树/小网络，S5组合消融，S6综合图/论文/PPT/可编辑poster待开展；S7独立检验并行，不等9月9日。

已有主30天Ms5–6、60万km²、147锚点开发成绩：原目录54，多尺度58，515段等权几何混合60。
60仍漏87个，不是可靠预测或盲测。应变主混合58，与目录相同；长窗口小面积有限正线索已保留。
S2-C科学提交`dbda2cba6bc418682d92ce9b1042e549c37edde5`、闭合`da37b13e6739b285f1f8870bb70c9cab778bb9f7`
均已推送回读。不重跑任何S1/S2或旧D1。

已关闭事项见 `SCIENCE_DECISIONS_AND_REPORTING.md`，不重复复核，也不在例行心跳重播。

## 2. 本次新落地，尚无新成绩

读过最新交接、总方案S3及已实现源码，核对国内经典/国际经典/2025增量原文后写入：

- `configs/multitask_s3_anomaly.yaml`：两A开发折、16+4特征、地点/次数两个低维ridge输出、成熟与置乱；
- `docs/s3_anomaly_methods_2026-09-03.md`：通俗假设、引用与来源阅读边界、公式和解释限制；
- 唯一蓝图第1.21节：新S3角色与续接授权边界。

协议和纯数学验收见`docs/s3a_protocol_acceptance_2026-09-03.md`；提交
`ea9b025cc86ab77d269ba20f95703f0c884b3949`已于03:58:41推送回读一致，数据角色已开放。
随后按范围读取了开发样本用于计数及首期异常/背景接线，产物
`outputs/multitask_s3/s3a_input_v1/input_waterlevel.json`于04:11:45落盘。
**新异常参数尚未训练、外层效果尚未评分，不是153期预测已完成。** 78项相关合成/配置检查通过；
输入验收见`docs/s3a_input_acceptance_2026-09-03.md`。不要重跑协议或输入水位；直接接真实训练执行器。

本轮153个合法报告起报（北京时间2022-07-21至2025-06-26），不是全205期。
30天两折训练分别9/19个起报、72/195个Ms4事件；主评价分别6/6次、9/2个Ms5–6首震。
全部30天报告回放48/47次，覆盖16/8个Ms5–6首震，但重叠窗口不能当独立重复。
7/90/180天主评价首震合计3/17/9；365天两外折无完整起报，记NA而非失败。
一年外折无法完整容纳365天报告窗口是本轮限制，后续长窗需另行预定更长分段，不在看过当前结果后改本折。

## 3. 最高优先级安全续接

1. 输入阶段已远端闭合。先查本机 `s3a_prepared_v1/preparation.json` 的完成期数、时间和实际PID；
   未启动则在实现必要检查/提交推送后运行 `python -m seismoflux.multitask_s3.preparation`，
   参数为本工作树 `--project-root`、数据根 `--data-root`、上述 `--output-dir`、`--workers 2`。
   已运行不重复启动，已完成直接接训练，不再索要授权、不重复输入水位或扩大冻结协议。
2. 薄输入已实现并接通，直接编写/执行最小训练预测装配，不要重复造输入层：
   `calendar.build_fold_calendar`给训练、内验证、全部和主评价起报，`time_null_partitions`给每h反事实池；
   `features.read_report_issue_metadata/load_issue_features`按指定日期列读取，只给授权范围；
   `catalog_background.build_catalog_background_components(...).for_horizon(h)`每期核只算一次；
   `input_waterlevel.load_development_catalog`给按角色筛过的目录（每次训练再严格按label availability切）；
   `models.fit_training_area_imputer/fit_offset_poisson_ridge/fit_offset_poisson_intercept`复用，
   空间用旧C2B `fit_spatial_ridge`，不能把它当计数似然。
3. 首期真实格点对齐与背景检查已完成，但未缓存全153期特征/背景。继续按协议做空间/次数预测，
   只在训练内拟合缺失/标准化，计数评分使用`predict_log_mean`防止下溢丢信息。
   全部外折预测保存后统一评分，不能先看第一折成绩改第二折。
4. 时间和空间各200复本/折，仍按原数量，严格限制开发来源池；先有可恢复检查点，不重复实例。
5. 真实结果出来立即科学复审、静态图和本机回放，验收同步后续S3-B有限尺度/学科消融或S4，不无限扩模型。

目录空间参数是最新旧C折迁移：四个短中窗K25/K75权重各0.5，365天三核各1/3，不能新选参数。
计数用1970—Q历史同带次数/天数乘h，直接Ms5+拟合异常乘子，两正式带共享，不说震级分布已经学会。
仅截距校准保留，避免把基础发震率校准误说异常有效。COV5/SNAP16/DYN20严格嵌套。

重要修正：旧D1时间置乱的真实来源可能晚于伪起报，是离线反事实，不是因果预测。
S3只能在本协议A开发来源内、且所有实际拟合/验证起报切点分池；不调用全205期加载器。
真实预测绝不使用未来报告。覆盖固定也不是完整条件随机化检验，不报精确p值。

## 4. 资源、展示与访问边界

工作树`D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata`，
分支`codex/p2r-multitask-multidata`，数据根`D:\AIPred\SeismoFlux\data`，
Python`D:\AIPred\SeismoFlux\.venv\Scripts\python.exe`，PYTHONPATH为工作树src。
特征库4.09GB，按期/所需列读取，勿全列装载。报告来源仅[2022-07-01,2025-07-01)，各外折再限制。
输入核对PID28600已退出；04:14左右检查无Python后台进程、整机CPU约8%。心跳`seismoflux`已更新
为本S3入口并确认ACTIVE，每30分钟；
不能建立重复自动化。每次实际检查进程和CPU，不能引用旧PID。
机器24物理/48逻辑核；默认2最多3折线程、数值库1线程、BelowNormal、Hidden，至少留2物理核。

A2025+审计、C留出/审计分数、锁定测试保持关闭；不同角色的同一事件不是独立新样本。
不取未来目录，不改P1/旧预测，不触碰根旧工作树和science_first的Stage4未跟踪草稿。
数据/逐事件数组/震例地图/含事件HTML仅本机；仅代码/文档/聚合结果与无事件坐标图白名单提交。
此前真实浏览器file URL受安全策略限制，不能绕过，也不构成科学阻断；不再建立展示平台。

## 5. 每次心跳科学复审

用通俗话说清：本轮使用报告信息而非原始仪器振幅，153期是本轮范围，11个只是30天保守评价首震，
不是全部数据只有11个；长期目录还提供数千个背景事件。正在检验额外价值，尚无新真实成绩。
S0—S7用一行定位；只在新阶段结果出现时展开数据、训练和较晚效果，不重播已报事实，不用测试数冒充提升。复审：
①当前是否增加预测证据或紧邻的必要使能；②是否偏离多数据/三任务；
③是否陷于无益工程；④该继续、缩小还是停止。若无增量要结束当前变体，不重复95%/Ms讨论。
本次方案明确地点/次数分工和覆盖对照，属必要使能；紧接真实训练才有科学价值，不应停留在文档。
