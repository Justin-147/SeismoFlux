# 最新续接：S3-A异常地点/时间试验

更新时间：2026-09-04 10:27（北京时间；8路实际计算并新增252块，切换验收闭合）。
状态：`S3A_NULL_PREDICTIONS_RUNNING_STRATA_COMPLETE`。
接替`docs/restart_handoff_2026-09-03_s2c.md`，旧文档保留S2结果与95%/Ms审计证据。

## 最新安全点（本节覆盖下方历史过程状态）

### 2026-09-04 10:27 八路已实际运行（最高优先级安全点）

已进入`predicting_nulls`；10:25:16已存 **2265/4000块（56.625%）**、失败0，较上次已报2013块
新增252块。时间完整复本两折 **200/200、170/200**，空间 **83/200、0/200**。当前8路正在完成
第二折剩余时间复本，之后依固定待办队列转空间；空间水位此时不变是队列顺序，不是卡住。

唯一协调器PID **28084**，8个实际计算子进程 **12684、27764、37132、38252、40528、48728、
53348、54352**，父PID均28084，均BelowNormal；stderr为空。10:23:47采样整棵试验进程树CPU
约 **17.44%**，整机CPU **36%**，可用内存 **22.96GiB**。子进程工作集约2.91—3.68GiB，但包含
共享映射，不相加当独占内存；采样窗口不同，不能保证后续占用。8路切换已有实际新结果，不仅是配置。

此后只检查原进程树和实际推进，不再清锁、重启、重复讲切换实现或扩展性能工程；新日志继续为
`parallel8_20260904_094803`两文件。仍有`outer_effect_scores_computed=false`，暂无最终归因结果。
新增252块是计算推进，不是预测效果提升，也不能用当前时间对照速度外推空间对照的剩余耗时。
下一科学动作：全部4000块终态后统一归因评分、完整S3-A复审，再按既有有限S3-B推进。现阶段仍
紧邻时间/地点信息增量证据，未偏离路线；没有新科学结果要求改模型、加试验或改采用标准。

切换闭合只读复核通过：科学identity与备份完全一致，原2013个completed条目缺失0、改变0，
前序清单SHA与执行历史一致，新4模块执行SHA相符，所有任务仍在原注册集合。原NPZ字节已在
启动加载阶段核验，本次不重复全盘扫描；没有读取效果分数。8路切换事项闭合，科学归因仍未完成。

### 八路切换历史记录（已实际启动，后续不重复汇报）

用户“调整8路”已落实到新协调器；实现、113项合成核对及独立复核通过，提交
`ae8be18db1b806a4cbdb74daffde796cc8818807`已推送、远端回读一致。蓝图1.23与
`docs/s3_parallel_execution_2026-09-04.md`记录仅执行方式修订。原科学文件/YAML/数据不改。

09:47:13核对旧PID9792仅有自带conhost、无计算子进程后停止，确认旧计算已退出；切换点为
**2013/4000块（50.325%）**、失败0，最后结果09:46:45，实际2013个NPZ与清单相符，无完整孤儿。
原清单备份在`data/interim/s3/parallel_migration_20260904_094713/null_prediction_manifest.before.json`，
SHA256 `2fb7093ad24acd41059293dc8f01869b86f1bfac8cc99e3c05972a2ee12d3b13`。
仅清除已退出9792的旧锁，没有删改任何NPZ。09:48:03启动唯一协调器PID **28084**，
新入口`seismoflux.multitask_s3.null_parallel --workers 8`，Hidden、BelowNormal、数值库1线程。
新日志`data/interim/s3/s3a_null_v1.parallel8_20260904_094803.stdout.log`及同名stderr。

09:49—09:54曾处于`loading_parallel_frozen_inputs`；10:25已确认加载通过，8个worker均活跃并
新增222块。**不要另起协调器，也不要退回旧null_runner恢复。** 后续核对PID28084、上述日志、
manifest.worker_pids与子进程树；不再重复启动加载的说明或当时尚未就绪的旧待办。
数值失败收据已补齐，遇中断后不重复已失败任务；已完成训练/图件/回放/分层不重做，不提前评分。
心跳已更新为识别1协调+8计算进程并汇总占用、看整机可用内存，仍每30分钟。
本次是加快既定科学对照的执行调整，暂无新预测效果证据，不扩展性能工程。

### 2026-09-04 09:20 当前最高优先级安全点

唯一S3进程PID **9792** 已通过恢复加载并进入`predicting_nulls`，时间和空间两分支均新增结果，
stderr为空。09:17:28已保存 **1991/4000块（49.775%）**、失败0，比上次已报1970块新增21块；
时间完整复本两折 **200/200、118/200**，空间 **80/200、0/200**。当前首折在生成下一空间复本，
次折继续时间复本。09:19:52采样进程CPU约2.54%、内存3.64GiB、整机CPU16%（采样窗口不同），
BelowNormal、两个折线程；只有一个相关S3进程。上述资源数值是当时样本，不是后续占用保证。

**本次恢复事项已闭合。** 后续检查实际新增块，不再重复恢复审计、清锁或重启；日志仍是下方
`resume_20260904_083138`两文件。原数据、已存1970块、冻结模型/折/指标/200+200试验均保留，
`outer_effect_scores_computed=false`，尚无最终归因结论，不能把计算推进称为预测提升。
下一科学动作仍是完成原时间/空间对照后统一评分与完整S3-A复审，再按既有有限S3-B推进。
目前紧邻预测增量的归因证据，未偏离路线、未扩展工程，无新证据要求调整冻结试验；已完成
训练、评分、图件、回放和分层不重做，不触碰关闭的测试/审计或其它工作树。

### 2026-09-04 08:33 中断与启动续跑记录（09:20确认恢复闭合，后续不重复汇报）

用户报告刚刚重启。08:26及08:30核对：旧PID **44528** 已不存在，所有相关S3 Python进程均为0，
旧stderr为空；系统启动时间仍为09-02 17:27，无法据此确认这次是电脑还是应用重启，不猜测退出原因。
最后新增结果检查点为 **08:23:25，1970/4000块（49.25%）**、失败0；时间两折完整复本
**200/200、117/200**，空间 **77/200、0/200**。相对上次08:14已报状态新增13块。

有限只读复核：1970个登记NPZ全部存在，无孤儿或临时残片；冻结协议、10个置乱实现、7个原模型
实现及参考预测清单哈希相符。仅清除指向旧PID44528的陈旧锁，未修改manifest、NPZ或原数据。
08:31:38再次排重后，用下方原命令加`--resume`启动唯一PID **9792**，Hidden、BelowNormal、
两个折线程、数值库单线程。新日志为`data/interim/s3/s3a_null_v1.resume_20260904_083138.stdout.log`
及同名stderr；旧日志保留。未改种子、模型、输入、折、指标或200+200复本数。

08:31:57 manifest已登记新PID，状态 **`loading_frozen_inputs`**；1970块仍保留、失败0，
08:32 stderr为空。此时只证明续跑启动并存活，尚未证明本次加载结束或新增预测块；逐块字节哈希
交由原续跑程序核验。08:32:16采样进程CPU约0.80%、内存0.15GiB、整机CPU3.5%，唯一实例，
BelowNormal；进程和整机采样窗口不同，不能作为后续占用保证。

当时的下一步是检查PID9792、上述新日志与manifest，等待冻结输入加载完成；加载时百分比不增加
不等于卡住。09:20已确认`predicting_nulls`且新增21块，本次恢复闭合，不再重复加载时间排查。
已完成的真实训练、评分、图件、回放、分层均不重做；未运行锁定测试或读取未来目标。
本次只恢复原科学对照，不扩展工程；它继续解释局部预测收益与异常时间/地点的关系，不是新增
预测提升。没有新科学证据需要改变路线；完成全部对照后统一评分与S3-A复审，再按既有S3-B推进。

### 2026-09-03 中断与原检查点恢复记录（已闭合，后续不重复汇报）

原归因在23:09:18保存到 **1575/4000块（39.375%）**、失败0后意外停止：旧PID39156不存在，
原stderr为空，manifest未写成终态。时间两折完整复本为 **200/200、92/200**，空间两折为
**23/200、0/200**；较22:43已报状态新增23块，但尚无新增预测效果或最终归因结论。

23:45再次按完整命令行核对，除检查命令自身外没有任何`null_runner`实例；1575个已登记块均存在，
无孤儿或临时残片，冻结协议、输入、参考预测及实现哈希一致，磁盘空间充足。只移除指向旧PID39156的
陈旧`null_prediction.lock`，于23:49:44用原命令加`--resume`安全续接；未删除或覆盖manifest/NPZ，
未换种子、模型、数据、折、指标、复本数或输入。新唯一PID为 **44528**，Hidden、BelowNormal、
两个折线程、数值库单线程；新日志为`data/interim/s3/s3a_null_v1.resume_20260903_234944.stdout.log`
及同名stderr。

23:54新进程仍存活，manifest已写入新PID并处于`loading_frozen_inputs`，1575块保持不变，stderr为0；
23:51采样进程CPU约0.70%、内存0.64GiB、整机CPU10%。当前正在做恢复时预定的冻结输入和哈希复核，
不是重新训练；通过后会进入`predicting_nulls`并从未登记块继续。不要再次启动、改模型或提前评分。
2026-09-04 00:15已确认通过加载并新增检查点，恢复闭合。无需改变研究路线，不重开同一恢复审计。

进入空间对照后的减速已作一次有限只读核对，不是整体停滞：首折空间r001/r002/r003完成时刻为
19:49:01、19:58:14、20:07:29，约9分钟/复本；第二折时间r082与r083也持续保存块。
空间每复本逐报告重建全国特征（`features/anomaly/spatial.py:1017`），而时间主要重排已有数组；
两折同进程线程（`multitask_s3/null_runner.py:587`）中的Python循环争用可能进一步拖慢另一折，
这是代码和占用相容的解释，未作性能剖析定论。所审路径无死锁或共同输入改写迹象；检查点在特征
重建之后才更新。保留原进程/2线程/冻结试验，不裁剪字段、不换进程池；以后不再按纯时间阶段速度
估剩余时长。无新异常时不重复排查或复述该原因；只看两分支是否继续推进。

本机回放已于 **14:01:37** 完整保存，生成进程PID45680已正常退出，stderr为空，不再启动：

- 页面：`outputs/multitask_s3/s3a_score_v1/replay_v1/seismoflux_s3a_replay.html`；
- 清单：同目录`replay_manifest.json`；页面约50.3MB，包含全部324个起报×时限回放帧、
  648个分震级窗口及38个不同地震，338个分震级窗口无地震也保留。这些不是新增独立样本数。
- 页面SHA256：`bb8f7fa5df1b34dad27031b7651b9e91a87542fab3fdc44d8b37ae730139686a`。
- 原预测manifest SHA256仍为`6470226f92bc23f0556e4bae267ff754ed44a4cb9c00b004c2a1140a864d3f2d`；
  没有重训/重评分、读取新目标或置乱结果。生成代码为已推送的`7ea3ad8f1ef6ba7850a1f7482aa0b7969c8c7c60`。
- 实际页面的内嵌数据经Node只读执行核对：20种折/时限/起报轴视图、两震级日期/事件对应、
  全部六对比×五预算×两判定的切换通过。真实浏览器像素效果未声称完成，不绕过已记录的访问限制。

震序/地区分层已完成，不再是待装配任务。已推送代码`325e65b5c83e7f08d55517855907fad26d8425f3`
于14:51:38以PID17216启动，14:53:44保存全部15块和最终`strata_manifest.json`，进程正常退出。
位置：`outputs/multitask_s3/s3a_score_v1/strata_v1/`；本机解释：`STRATA_SCIENCE_REVIEW.md`。
全国与地区的权重、候选/参考命中及正负变化独立对账通过；两正式震级各自全历史建震序，权重
不在窗口/地区内重归一化，空窗、空区和365天NA保留。主轴全国两类原定不确定性区间已补齐，
全部报告轴与地区仍明确为描述性。验收和精确恢复入口见`docs/s3a_strata_acceptance_2026-09-03.md`。
新增静态图已于15:05保存并实际目视核对：`strata_v1/rendered/05_strata_recall_changes.png`
及可编辑文本SVG，同目录`render_manifest.json`记录原汇总与图件源文件哈希。固定展示动态相对
目录的主轴严格判定，不选择获益任务；四视图、所有预报时限/震级/面积及样本数、0/NA均保留。
它只解释已有预测，未重训或增加独立证据。图件核对到此结束，不再重新生成或扩展新UI。

新增科学解释：部分小面积改善在震序降权后仍保留，而某些大面积损失也保留；动态相对快照有
任务专用增益，也有损失。保留局部候选和模型互补性，不统一替代目录，不把这次汇总称为新训练
进步或新独立确认。具体分母、数值、地区和区间只读本机解释与结果，心跳首次报告后不重复旧分数。

下一科学工作：同一200+200归因继续，全部预测保存后统一评分，再做完整S3-A科学复审。
不要重做准备、训练、主评分、震例账本、离线回放或已完成分层；S3-A完成后按既有有限S3-B
学科/尺度方案推进，不直接宣称整个S3结束。帮助理解震序权重的对照图已完成，不扩展新UI。
当前未偏离科学目标；不继续围绕无益实现、美化或浏览器限制投入。
S3-A初步评分、归因完成、整个S3完成仍须分别表述。页面与全部事件位置只留本机。

### 13:59 回放启动过程（已结束，仅供追溯）

回放代码提交`7ea3ad8f1ef6ba7850a1f7482aa0b7969c8c7c60`已推送并回读一致。13:58:53确认
没有重复回放实例后，启动唯一`replay_runner` PID **45680**，Hidden、BelowNormal、数值库1线程。
原`null_runner` PID39156保留。回放只从已有预测恢复显示，不重新训练/评分；运行日志见下方。
此时两个不同职责进程是有意安排，不是重复置乱实例；不要把回放PID当归因PID处理。
若回放日志暂未更新，先核对进程和stderr；它先核验预测并加载静态裁剪格，随后每个折×时限打印
一次显示进度。原预测/评分/事件账本均已完成，不重跑。页面生成完成后只核验产物、补交接。

### 13:51 回放接线核验（已完成，仅供追溯）

原归因PID39156保持运行，13:49:07已存111/4000块（2.775%），0失败；时间两折完整复本16/200、
6/200，空间两折0/200，仍按既定顺序排队。13:42:10唯一S3实例，进程CPU4.59%、内存3.55GiB，
整机CPU14%，BelowNormal、stderr为空。没有重启或修改本次归因实现，也没有读取其效果成绩。

利用等待期补现有结果的本机离线回放，新增`replay_runner.py`和`replay_html.py`：只读原预测、
诊断、震例账本和静态几何，不重新拟合、评分或读取目录目标。按原排序恢复报警网格，逐预算核对
实际成本；命中标志直接读原数组。保留全部日期、两震级档、空窗、正负案例和365天NA。
双图显示同一预算下参考/候选，事件点击及局部放大只改视图。次数表保留本档全部事件。
16项合成核验、Ruff、类型检查及独立科学接线复审通过；包含Node严格DOM的全部筛选轴/空窗/NA/
双图视域测试。它是交互逻辑核验，不冒称真实浏览器视觉验收；沿用已记录的file访问限制，不绕过。

完成代码白名单提交推送后，单进程、数值库1线程、Hidden、BelowNormal运行一次：

```powershell
python -m seismoflux.multitask_s3.replay_runner --project-root D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata --data-root D:\AIPred\SeismoFlux\data --prediction-dir D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata\outputs\multitask_s3\s3a_fit_v1 --score-dir D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata\outputs\multitask_s3\s3a_score_v1 --case-dir D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata\outputs\multitask_s3\s3a_score_v1\case_ledger_v1 --output-dir D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata\outputs\multitask_s3\s3a_score_v1\replay_v1
```

先查无重复`replay_runner`，现有`null_runner`必须保留。输出`replay_v1/seismoflux_s3a_replay.html`
和`replay_manifest.json`；若已完整存在，不再生成。日志`data/interim/s3/s3a_replay_v1.stdout.log`
及同名stderr。全部页面与含坐标载荷仅本机，公开只提交代码/合成测试/交接文档。
这是让既有效果可直接查看与追溯，不是新预测提升；原定归因仍是主要科学计算。

### 13:33 原定归因任务运行记录（最新水位见顶部）

代码提交`4fae83e963e6c74011d05e8b67bd289f981f4d12`已推送并回读一致。确认无重复实例后，
13:17:58以唯一PID **39156**启动下方同一命令，两个折线程、数值库1线程、Hidden、BelowNormal。
全部合法报告的恒等时间重建，以及首/中/末报告的恒等空间重建均通过；这核实了未置乱时输入等价。
任务当前为`predicting_nulls`，不是等待启动，不得再次启动或重跑原准备、训练、主评分与震例账本。

13:32:31检查点：已保存 **45/4000个预测块（1.125%）**，失败0；完整复本另列，不能混为分块数：

|归因对照|2023—2024折|2024—2025折|
|---|---:|---:|
|时间置乱|6/200|2/200|
|空间置乱|0/200|0/200|

每折先时间、后空间，空间暂为0是既定顺序，不是卡住；尚未完成整个复本的已存块也计入45块。
13:33:06只读核对仍为1个S3实例：进程CPU约4.30%（以整机48逻辑核归一）、内存3.57GiB，
整机CPU13.5%，BelowNormal，stderr为空。24个物理核心中绝大多数可供其它任务使用。
这些是带时间戳的样本，不是后续资源占用保证。

当前尚无置乱对照最终效果，`outer_effect_scores_computed=false`。不要因部分复本完成就提前选择
结果或调整模型；全部预测保存/失败如实登记后统一比较。这一步检验已见局部收益是否依赖异常
发生的正确时间和地点，是归因解释，不是重新设置进步采纳门槛。当前未偏离科学目标，无需改计划。

下一步：检查原任务的新检查点及唯一进程，保持同一冻结的200+200继续；完成后接统一归因评分。
本机离线交互回放、震序等权/区域分层与完整S3-A复审仍待完成，可在不干扰本任务的前提下推进。
纯描述汇总组件`null_summary.py`已补齐并经4项合成检查；未读取或计算真实置乱效果，不是归因完成。
正在运行的manifest绑定了实现哈希，**本任务结束前不得修改其中绑定的实现文件**；新增纯汇总与
文档不属于本次预测身份。中断时只按下方原目录`--resume`，先查PID/命令，不能另开新实例或换种子。

### 13:17 归因运行接线（已完成，仅供追溯）

原定两类置乱已接上薄`null_runner.py`；空间纯重建`null_space.py`及仅合法日期状态读取
`null_state_inputs.py`已完成。259项S3合成检查与Ruff通过，独立科学接线审查通过。
没有重训/重评分原结果，也没有新增模型或改变采纳标准。推送回读后已启动下列唯一任务：

```powershell
python -m seismoflux.multitask_s3.null_runner --project-root D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata --data-root D:\AIPred\SeismoFlux\data --prepared-dir D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata\outputs\multitask_s3\s3a_prepared_v1 --reference-prediction-dir D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata\outputs\multitask_s3\s3a_fit_v1 --output-dir D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata\outputs\multitask_s3\s3a_null_v1 --kind both --workers 2
```

使用本工作树src和根venv，数值库线程1、BelowNormal、Hidden，先确认无重复S3实例。
检查点`outputs/multitask_s3/s3a_null_v1/null_prediction_manifest.json`；日志
`data/interim/s3/s3a_null_v1.stdout.log`及同名stderr。中断后保持目录与命令，添加`--resume`；
若`null_prediction.lock`遗留，先核对其PID和命令确实不在，再处理锁，禁止重复实例。

总量为每折时间200、空间200；两个折、五时限共4000个预测块（365天保留明确NA）。
按折依次做时间复本，再做空间复本；空间每个复本的同一场供五时限共用，时间按h对应分池。
根seed147的索引命名空间已固定写入manifest，不因中断另抽种子。
先检查全报告恒等时间重建、首/中/末三个报告恒等空间重建；这是输入等价核对，不计入复本。
`by_kind_fold`分开记录completed/failed复本，`completed_blocks/failed_blocks`是分块数，不能混叫。
`terminal_percent`包含已登记失败，不是成功率；所有块落盘/失败如实记录后，才另行统一评分。
目前尚无新置乱效果；不要说归因完成或S3完成。磁盘/权限/内存/身份异常是中断，不能当科学负结果。

### 先前完成状态（不重复计算或复述成绩）

12:46本次心跳从归因/震例环节继续，没有重做准备、拟合、评分或初步效果复审。
新增`case_ledger.py/case_runner.py`，只把已有逐事件命中对应到受限目录，保留全部六对比、五预算、
严格/70km的新增、丢失、共同命中、共同漏报；不挑案例，不重算命中，不读取新评价角色。
新增`null_features.py/null_inputs.py`，接通九快照与两基础序列同来源时间置乱、伪历史动态重建和
仅指定合法报告行组的两基础列读取。复用既有时间公式，不调用旧D1全205期入口。
221项S3相关合成检查和Ruff通过；时间重建另经独立只读数值复核通过。
本次只完成归因组件，**200+200置乱尚未启动，不能报为完成复本**；仍缺空间实体重建和归因运行装配。
代码提交`380c4b97c8c11a8353f18d9edb24b534c34ac900`已推送回读一致；随后单进程、数值库1线程、
BelowNormal/Hidden运行`multitask_s3.case_runner`，12:48:27完成并正常退出。
输出`outputs/multitask_s3/s3a_score_v1/case_ledger_v1/case_ledger_local.json`已存在且完整，
不要重跑。`CASE_REVIEW.md`已落实全部获益/丢失震例与地区集中性解释，不再只展示整体分数。
图代码`f6ed3c7fab59aff9315a6f7ea222dd2f99f5639a`已推送回读一致，合成矩阵核验与Ruff通过；
本机`case_ledger_v1/rendered/04_all_changed_anchor_cases.png/.svg`已生成并可视核对，包含
全部发生得失变化的主轴首震和全部预设面积，保留共同命中/漏报及未落入该主轴的区分。
本轮是对已有增量的科学解释，不是新一轮效果提升或独立验证。图、事件账本及CASE_REVIEW只留本机。
12:52核对无S3后台进程，整机CPU21%；不是中断。交互地图、时间/空间复本仍未开始，不误报。

**12:52当时的下一动作（现已由顶部运行中安全点接替）：读取本机CASE_REVIEW与本文件，装配原定
时间/空间归因执行，补离线交互回放。不要重复case_runner、初步评分或训练。** 当时时间归因已就绪；接线依据见
下方。近期目录与异常互补、次数任务限制见旧初步复审，不复播。

### 归因最小接线依据（13:17已实现，仅供追溯）

- `null_inputs.load_radius_bases`只读3身份列+两原始radius列；`null_features.permute_time_features`
  接已缓存20列，九快照/16和18缺失同donor，覆盖12—15/19留recipient，三动态/17由完整伪历史重建。
  必须传该折全部合法报告轴，不是仅训练/主评分抽样日期；调用已有`calendar.time_null_partitions`。
- 空间坐标仅复用`d1_replay.placebos.permute_d1_coordinates_within_zones`纯双射函数，输入每期
  `D1CoordinateEntity(state_id,construction_stratum_id,longitude,latitude)`，保留inside/outside后缀。
  仅替换状态的经纬度，再`spatial_entity_arrays`→`compute_selected_spatial_features`，只取200km
  九快照与两radius；不能用旧Stage4快速快照（缺本轮学科/source_new列）。三动态复用新函数。
- 状态文件只读授权issue的row groups，再`states_from_records`→`build_issue_snapshots`。
  完整AnomalyState schema只在已授权的行组中读取，不能加载全205期再过滤。
- 分层映射路径/哈希见`configs/d1_retrospective_development.yaml`；仅借用静态分区身份。
  entity mapping用Arrow授权日期predicate只读state_id/issue_time_utc/construction_stratum_id，
  cell mapping只取construction_zone_id；不要调用D1完整加载、verify/preflight或prepare入口。
- 根seed147，需在新薄执行器明确记录kind/fold/h/replicate索引；不复用旧D1 fold_1/2/3命名。
  复本仍为每折时间200、空间200，失败不删/不补抽；每次重新填补/标准化/内选择/拟合。
  模型纯拟合复用S3`predict_block`或training，不改冻结文件；保存复本预测后才评分。
- 先小合成核验/提交推送，再做原身份准备和原规模复本；两折线程、库1线程、隐藏低优先级。
  不为做归因另建通用平台，不因无显著性否决已可复核局部收益。

11:54已接通真实预测执行器`multitask_s3.runner`与纯评分组件`multitask_s3.scoring`。
178项相关合成/配置检查和独立科学接线审查通过；见`s3a_runner_acceptance_2026-09-03.md`。
11:55:55代码与本轮文档提交`4b69df56cc959f0d162b31b01f7d9308ed8d6852`已推送并回读一致。
准备已于11:58:07完成153/153、0失败，原PID42188已退出。不要重新启动准备。
11:58:44确认无重复实例后启动真实拟合PID11488，BelowNormal、Hidden、2折线程、数值库1线程。
12:01:02已完成10/10块、0失败；全部预测随后独立读回验证通过，PID11488已退出。
24个可拟合的设计×折×时限组合均完成空间与次数拟合；训练损失有改善，但这不是较晚预测提升证据。
12:01整机CPU8%；当前无持续拟合任务。365天两个块均为NA，不是拟合失败。
最小评分入口`score_runner.py`和合成检查已于12:10通过独立复核；185项S3相关检查通过。
12:10:58评分/图代码及交接提交`440c0f843735ec3bdc2c7d3c3252661fa242ce00`已推送回读一致。
12:11:39启动首次较晚历史评分PID31268，12:17:13完成全部10块。训练、准备和评分进程均已退出。
12:28整机CPU4%，无S3相关后台进程。这是正常完成，不是中断；不因没有Python进程重启旧计算。
预测manifest的SHA仍为`6470226f92bc23f0556e4bae267ff754ed44a4cb9c00b004c2a1140a864d3f2d`，
与评分前一致。真实结果已做独立科学复审，三张静态PNG/SVG已生成并可视核对。

先前初步复审见本机 `outputs/multitask_s3/s3a_score_v1/SCIENCE_REVIEW.md` 和
`science_scores.json`；本次已继续到震例复审，最高优先级以本节顶部新安全点为准。
本轮已按用户原则保留局部研究候选，未整体升级全部任务；首次结果已向用户报告，之后只讲新增证据。
具体聚合成绩、结果说明、图和事件诊断只在本机；本次公开提交仍仅代码/测试/交接，不添加新派生结果载荷。
心跳已去除硬编码的过时恢复点，继续ACTIVE、每30分钟；总按本文件实时安全点判断，不因旧提示重跑。

下一阶段边界：S3-A初步效果已完成，整个S3未完成。保持原有数据/模型/指标/200+200置乱不变，
补时间/空间归因、震序等权/区域分层、获益及丢失震例本机交互回放，再完成完整S3-A科学验收；
随后才按总方案推进S3-B有限尺度/学科消融、S4机器学习、S5组合、S6成果、S7独立检验。

下列是已完成的启动命令，仅供追溯，不再执行。若未来确有损坏/中断证据，先复核后再决定续接，
不能因目前无Python进程而重跑：

```powershell
python -m seismoflux.multitask_s3.runner --project-root D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata --data-root D:\AIPred\SeismoFlux\data --prepared-dir D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata\outputs\multitask_s3\s3a_prepared_v1 --output-dir D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata\outputs\multitask_s3\s3a_fit_v1 --workers 2
```

沿用本工作树src/Python环境，数值库线程1、BelowNormal、Hidden。不是当前shell的任意python。
中断后只加`--resume`续同目录，遗留`prediction.lock`必须先核对PID/命令再处理。
检查点为`s3a_fit_v1/prediction_manifest.json`，每折每h保存一个块，共10块（其中365天明确NA）。
日志为本工作树`data/interim/s3/s3a_fit_v1.stdout.log`和同名`stderr.log`。首块未完成时
`completed_blocks/last_checkpoint_utc`可能尚未写，读`completed`空字典得0，以进程CPU变化和
日志判断活跃，不能因界面未变化而重启。
全部预测落盘后才接外层评分。纯评分组件不是完整结果，尚需评分装配、震序等权/区域汇总、
原定200+200置乱、静态图/本机回放与科学价值复审。不要因为10块完成就说整个S3或项目完成。
只新增本机拟合目录忽略规则；旧Stage4/旧实验逐事件文件未改，不用git add -A。

评分装配入口已完成，调用`runner.verify_complete_predictions`核验十块后构造外层targets。
不改已完成的模型/runner或输入，不复跑已有预测。以下接线说明供后续分层/回放使用，
不是重新做一次主评分的指令。纯评分结果含`_local`仅本机。
70km可复用`c2b_score.projected_near_cells(STRtree(domain.locator.clipped_geometries), x_m, y_m)`，
坐标用既定`EQUAL_AREA_CRS`投影；每正式带用S0`build_episodes`完整限定历史的成员映射，
episode等权采用`1/global_member_count`，不在窗口内重组。39块只读既有冻结的cell→construction_zone
映射，不重跑S0；映射hash见`configs/multitask_s0.yaml`。主起报配对区间可复用纯
`c2b_score.exposure_bootstrap`，物理震序区间可复用`development_summary._bootstrap_episode_ratio`；
绝不调用旧完整C评分入口或旧`science_gate`。区间用于说明不确定性，不作采纳硬门。
已执行的评分CLI为`python -m seismoflux.multitask_s3.score_runner --project-root <本工作树>
--data-root D:\AIPred\SeismoFlux\data --prediction-dir <本工作树>/outputs/multitask_s3/s3a_fit_v1
--output-dir <本工作树>/outputs/multitask_s3/s3a_score_v1`；单进程单数值线程，合成核对/提交推送后执行。
该评分命令已于12:11:39运行。日志为`data/interim/s3/s3a_score_v1.stdout.log`与同名`stderr.log`。
产物为本机`science_scores.json`和`event_diagnostics_local.json`；
前者预计含两折和跨折的主不重叠/全报告描述对比，后者含事件诊断，仅本机保存。
三张静态图在`s3a_score_v1/rendered`：固定60万预算命中率、次数信息变化、全部五预算净命中热图；
最后一张同时展示局部收益与损失，不能只展示有利预算。当前尚未生成本轮交互页面。

## 本轮历史过程（已关闭事项不向用户反复复述）

**公开确认已解决，不再等待同一授权。**
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
11:36:37检查点完成24/153期（15.7%），0失败；11:36:42进程仍活跃、BelowNormal，
累计CPU约464秒、工作集约4.5GB。最近整机CPU采样为11:33:47的18%，不把累计CPU秒当使用率。
日志在本工作树 `data/interim/s3/s3a_prepared_v1.stdout.log` 和同名 `stderr.log`；
进度在 `outputs/multitask_s3/s3a_prepared_v1/preparation.json`。首次启动后不得再运行无`--resume`
的新实例。旧PID只是本次检查记录，后续心跳必须核对当前进程；完成后不重启准备。

训练与标签原语验收见 `s3a_training_assembly_acceptance_2026-09-03.md`。本轮相关合成检查132项
通过；Ruff通过。下个最小工作是新增薄`runner.py`和真实评分衔接，不是继续抽象训练框架。
11:35:45训练/目标原语及验收、交接已推送并回读一致：`9f08ffca9c22bbc749d9acb2ff6aded1f7feb5ae`。
本段最新运行状态是推送后的本地交接补记；代码已同步，不因此重复验收或重新启动已有准备任务。
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
后才接外层评分。`runner.py`和纯`scoring.py`已于11:54验收；外层评分装配和本轮置乱执行器尚待完成。

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
2. 薄输入与训练预测执行器已实现并接通，按顶部命令执行，不要重复造输入/训练层：
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
