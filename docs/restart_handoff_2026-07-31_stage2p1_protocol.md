# SeismoFlux 重启续接交接：Stage 2P-1A 前瞻协议

- 交接日期：2026-07-31
- 工作树：`science_first` 隔离工作树（本地绝对路径不进入公开提交）
- 分支：`codex/stage2-etas-science-first`
- 本轮开始 HEAD：`79ab606c68c73fb1e8ed8fa4efa5146c1b181ce6`
- 当前阶段：`Stage2P-1A complete`；下一阶段 `Stage2P-1B ready_not_started`
- 文档状态：`complete_remote_closed`
- 科学价值分类：`necessary_enabler`
- 直接预测提升：无
- `protocol_frozen=true`
- `execution_authorized=false`
- `real_issue_authorized=false`
- 真实 issue 数：0
- 新前瞻目标读取数：0
- 锁定测试读取数：0
- 协议冻结提交：`5417838e3fef3c3ed74a1eb6f6c7d719326561c5`
- annotated 协议标签：`v0.2.4-prospective-seismicity-protocol`
- 远端 tag object：`df1ca98785d3e5fcbf48ed955f0a86c247f85b2d`
- 远端 peeled commit：`5417838e3fef3c3ed74a1eb6f6c7d719326561c5`
- 远端回读时间：`2026-07-31T08:11:46+08:00`
- 远端回读规范 LF 响应 SHA256：
  `8d409c5c6bfada0d894b1d69c83c2dd8e7c53736043e083e43a78dd32c9dc1f8`

## 1. 外行版

项目还没有证明预测变准。本轮完成的是把下一场真正的“未来考试”规则写清楚：以后每周四同时
保存长期多发区图 P0、加入最近 30 天地震的图 P1、加入再早 30 天地震的对照图 PP。三张图用
同一时刻的数据和同一套“不超过 600,000 平方公里”的完整格前缀规则，各自记录实际完整格面积，
保存后不能回头修改。未来地震成熟后，累计效果最多只正式查看一次。

这一步是必要准备，不是效果成果。当前仍不能运行真实预测。候选审计已 GO，并已唯一把
`protocol_frozen` 改为 true；`execution_authorized` 和 `real_issue_authorized` 仍保持
false。精确最终字节复验、提交、推送和远端标签回读闭合后的下一步仅是用纯合成数据演练整条链。

边界必须保持清楚：1A 只冻结 schema、规范字节、工件 profile、信任边界、状态机和统计语义，
实现状态只能是 `stage2p1b_required`；它不声称 ASN.1/CMS、目录表、预测数组或评价字节已经能够
重建。1B 才以纯合成数据实现和逐字节演练全链。1A 候选审计已 GO 并完成唯一状态跃迁，但不等于
阶段完成；最终字节仍须复验、提交、推送、打 annotated 协议标签并完成远端回读。

## 2. 唯一科学目标

在 G1-LS 冻结支持域、严格未来隔离和相同报警面积上限规则下，检验固定候选
`P1=0.5*长期75km KDE+0.5*最近30天地震KDE` 是否同时超过：

- 每期重建的长期 75 km KDE `P0`；
- 同一起报快照内“再早 30 天”地震形成的公平对照 `PP`。

只有支持域信息增益和全研究区严格召回都通过预登记门，至少含 20 个唯一 M5–6 事件和 10 个
独立震群块，且不由单一地区或震群支配，才能形成近期成分提供增量的候选证据。无论正负，结论
只适用于冻结的 ComCat 获取/修订链和固定 0.5 候选，不能外推为全部近期地震模型。

## 3. 本轮已形成的冻结内容

以下均为 Stage 2P-1A 已通过候选审计并完成状态跃迁的冻结内容；尚未声明最终字节复验、提交或
推送完成，必须按此精确清单审计和暂存：

1. `configs/prospective_recent_seismicity.yaml`：唯一机器可执行前瞻协议；
2. `data/contracts/stage2p_prospective_records.json`：五类记录及
   `EvaluationFreezeRecord.input_freeze→result_seal` schema；
3. `docs/phase2p1_prospective_preregistration.md`：完整逐字段科学预登记；
4. `SEISMOFLUX_IMPLEMENTATION_HANDOFF.md`：唯一实施蓝图中的 Stage 2P-1A 段；
5. `configs/research_protocol.yaml`：阶段状态、授权和协议身份；
6. `docs/research_protocol.md`：人类可读研究协议；
7. `docs/scientific_value_review_and_model_composition.md`：Stage 2P-1A 科学价值复审；
8. `docs/restart_handoff_2026-07-31_stage2p1_protocol.md`：本交接；
9. `docs/stage2p1_prospective_evidence_flow.svg`：仓库内静态证据流图；
10. `tests/unit/test_stage2p_preregistration_protocol.py`：新协议/schema 一致性测试；
11. `tests/unit/test_stage2s_causal_seismicity_protocol.py`：阶段状态回归测试；
12. `tests/unit/test_background_protocol.py`：从冻结标签核验旧背景环境，不误绑本阶段新锁文件；
13. `src/seismoflux/stage2p/__init__.py`：Stage 2P 语义验证公开入口；
14. `src/seismoflux/stage2p/validation.py`：五类记录跨字段、跨记录因果验证；
15. `tests/unit/test_stage2p_semantic_validation.py`：五类完整合成生命周期和绕过反例；
16. `pyproject.toml`：固定 JSON Schema、RFC3161 CMS/证书链验证依赖；
17. `uv.lock`：锁定上述依赖及其传递依赖，供验收环境逐字节复现。

仓库外另有会话内交互展示 `stage2p1-evidence-clock.html`；其本机绝对路径和会话标识不得写入
公开提交，也不得把该文件误当作仓库交付或添加到 Git。

以下未跟踪 Stage 4 草稿明确排除在本阶段之外，禁止审计者顺手修改、暂存、提交或删除：

- `src/seismoflux/anomaly_increment/kde_dev_background.py`
- `src/seismoflux/anomaly_increment/kde_dev_calendar.py`
- `src/seismoflux/anomaly_increment/kde_dev_fit.py`
- `src/seismoflux/anomaly_increment/kde_dev_inputs.py`
- `src/seismoflux/anomaly_increment/kde_dev_placebo_mapping.py`
- `src/seismoflux/anomaly_increment/kde_dev_seal.py`
- `src/seismoflux/anomaly_increment/kde_dev_statistics.py`
- `tests/unit/test_stage4_kde_dev_background.py`
- `tests/unit/test_stage4_kde_dev_calendar.py`
- `tests/unit/test_stage4_kde_dev_fit.py`
- `tests/unit/test_stage4_kde_dev_inputs.py`
- `tests/unit/test_stage4_kde_dev_placebo_mapping.py`
- `tests/unit/test_stage4_kde_dev_seal.py`
- `tests/unit/test_stage4_kde_dev_statistics.py`
- `tests/unit/test_stage4_kde_dev_synthetic_chain.py`

本阶段**禁止运行 `git add .`**。只能对上述 17 个 Stage 2P-1A 仓库文件使用显式路径暂存，并在
提交前核对 `git diff --cached --name-only` 不含任何 Stage 4 草稿或其它用户文件。

## 4. 已冻结协议的关键决定

### 4.1 数据与切源

- 本地地震目录只作 P0 历史基线，截止 `2026-07-09T04:25:56Z`。
- 此后的近期增量和未来真值使用 USGS ComCat。
- 查询下探至 `minmagnitude=3.9`，解析后本地严格筛 `mag>=4.0`。
- count/query 的精确参数、规范 URL、响应头、原始响应和解析计数均留证；count 达 20,000 时
  禁止 query，不能用自报小计数绕过。
- 完整 60 天同源洗脱只消除 R30/PP 换源混杂，不证明 ComCat 在中国的 M4 覆盖完整。
- 覆盖差异只记录并限制结论，不设效果相关硬阈值。
- 一次官方抓取或冻结目标无关支持证据未按时取得可记 missed；持续中断或实质变化则暂停后续
  发行并作目标盲修订，不能补发或看效果后换源。
- 切源缝隙按 300 秒、50 km、`abs(ΔM)<=0.5` 三阈值确定性一对一匹配，local anchor 优先。

### 4.2 起报与证据链

- 规则时刻：每周四 `00:00 Asia/Shanghai`。
- 每期共同查询结束为 `Q=T-15min`；正式抓取在 Q 后开始、T 前完成。
- P0 截止 Q；R30=`(Q-30d,Q]`，RP30=`(Q-60d,Q-30d]`，两个窗口精确等长。
- 首期不得早于 `2026-09-10 00:00 Asia/Shanghai`。
- 实际首期必须是协议和代码标签远端对象及 peeled commit 核验后的下一规则时刻。
- cohort 嵌入两个 annotated tag 的远端 object/peeled commit/核验响应 receipt，并冻结
  parser、模型、评价、展示和 validator 的结构化 code manifest。
- 可安装的完整源快照、P0/P1/PP 预测和报警候选必须最迟在 `T-5min` 冻结，候选 TSA 的
  `genTime<T-5min`；成功才原子安装为 `on_time`。
- 候选形成前失败时不生成预测。完整候选已生成但两家候选 TSA 均失败时，保留本地受限、内容寻址
  的 `FailedCandidateOnTimeCore`，记 `prediction_generated=true`、
  `prediction_installed=false`，但不得安装或公开其中的预测。
- 上述两类失败都最迟在 `T-4min` 冻结最终 `missed_issue` 审计 core，再以与候选尝试分离的
  RFC3161 请求在 T 前取得审计 token；审计 token 也失败时不安装漏期记录并暂停 cohort。
- RFC3161 TSA 固定 DigiCert 主、Sectigo 备；主站有可验证失败证据后才试备站。
- 所有 schema 要求 remote timestamp 的正式记录均用同一 RFC3161 core，只且恰好排除顶层
  `timestamp_attempt_evidence`、`remote_timestamp`、`content_sha256`；禁止递归排除或嵌套
  proof/token，最终 content hash 另按只排除自身复算。
- TSA token 是记录的绑定附件，不是第六类记录。
- RFC3161 固定使用锁文件中的 Python `cryptography`/`asn1crypto` 路径，并冻结 trust/OID/TSA/
  create-only `.tsq/.tsr` 附件合同。

### 4.3 五类只追加记录

1. `TargetCohortDefinition`
2. `IssueInputSnapshotRecord`
3. `MatureTruthSnapshotRecord`
4. `TruthRevisionRecord`
5. `EvaluationFreezeRecord`（`phase=input_freeze|result_seal`）

`IssueInputSnapshotRecord.status=on_time|missed_issue`。on_time 内嵌源快照和预测 seal；missed
不安装 source snapshot 或 prediction seal。候选形成前失败时不引用候选；候选 TSA 失败时必须
引用 T-5 前冻结的本地受限失败候选，但公开记录不能包含其密度或报警格。
`scheduled_issue_sequence` 包含每个规则周四和 missed；
`on_time_issue_sequence` 只在按时封存时递增，missed 为 null，52/104 只按 on_time 计数。所有
记录只追加、不覆盖；scheduled 最多 130，届时不足 104 个按时 issue 则证据不足停止；后续真值
修订不能回填或重算已封存模型版本。
第 130 期 cap 优先；当时只有 52–103 个按时期也不得开启第 52 期正式 look。

### 4.4 真值与统计

- 7/30/90 天目标窗在结束后再等 30 天成熟。
- 真值以独立于 issue 输入的请求形成 temporally independent mature snapshot，本地严格过滤
  `T<preferred_origin_time<=T+h`；按 `0h、+6h、+24h、+72h、+168h` 顺序取第一个完整成功响应
  并停止，只有全失败才完成 5 次并记 `truth_snapshot_unavailable`，不可评分、不能记零或用
  后续 issue 替补。
- preferred origin 修订跨 `T`、`T+h` 或相邻 exposure 边界时，按 input freeze 绑定的同一
  formal-freeze source snapshot 确定性唯一重归属，不能双计。
- 每个实际 input-freeze 检查点只允许一次 formal-freeze：第 52 个按时期最多一次；只有其完整
  基本门未满足且从未打开效果行时，第 104 个按时期才允许第二次，全线最多两次。
- formal-freeze 成功时才从同一完整响应写实际目标数、震群数、目标集合和窗口成员。只有
  `not_run_no_complete_scope` 的机械空 scope 可以写真实 0；第 130 期 cap 或任何 count/query/
  解析/本地派生失败都表示这些科学量不可获得，必须写 null/unavailable，不得伪装成观测到 0；
  exposure 选择和既有 truth availability 仍留证，Bootstrap 记未运行且不能打开效果行。
- 各 horizon 只按 on_time/issue time/horizon 选择时间不重叠 exposure，并保留零事件 exposure；
  目标数量、坐标、命中和分数不得影响选择。
- 三模型只在 G1-LS 支持域归一；IG 只用支持域目标，严格召回使用全研究区目标分母，unsupported
  三模型共同未命中。
- 600,000 平方公里只是完整格前缀面积上限；逐模型记录完整排名、实际前缀、格数和实际面积。
- 主门至少 20 个唯一去重 `M5_6=[5.0,6.0)` 事件和 10 个独立震群块。
- IG=`Σln(f_A/f_B)/N_supported` nats/event，密度在事件投影坐标直接求值；召回=
  `Σhit/N_all_region`；逐 horizon 计算后再 7/30/90 等权宏平均。无支持目标、分母为零、已选
  真值不可用或密度零/非有限时不可评价，不得加 floor。
- 使用 30 天/75 km 震群连通块和 2,000 次配对 Bootstrap；有 B 个块时有放回抽 B 个，相同抽样
  用于全部模型/对比/horizon，重复块按乘数计并重算分母，分位数使用 NumPy `method=linear`。
- PCG64 直接以 147 初始化，namespace 只作身份；效果解封前冻结完整索引矩阵，任何零分母复本
  在盲态判证据不足，不重抽。
- 四个端点使用 Bonferroni 95% familywise 同时区间，并通过 39 区、最大区域和最大震群稳健性。
- 第 52/104 路径最多两次 input-freeze、一次 effect look；先写不含效果的 input freeze，只有
  三个 horizon 均可评价、N>=20、B>=10 才打开效果，并立即追加一个直接闭合的 `result_seal`；
  中途累计效果始终密封。
- horizon 可评价性由 exposure/真值可用/分母/震群/密度证据重算；评价代码和环境在 input freeze
  冻结。valid/invalid ResultBundle 都以唯一规范 JSON 精确字节安装并复算其文件 SHA；valid
  ResultBundle 必须完整绑定 effect rows、同面积比较、Bootstrap、端点和稳健性工件；
  invalid ResultBundle 按 `effect_rows_open`、`alarm_area_comparison`、`bootstrap`、
  `endpoint_evaluation`、`robustness_evaluation`、`result_bundle_install` 或 `result_seal`
  记录失败阶段、错误码及当时实际存在的工件，尚未生成的后续字段为 null，不能伪造空表、零端点
  或稳健性数值，也不得重跑。

### 4.5 展示、资源和许可

- 每个按时 issue 立即生成静态图和无外网依赖交互页；只显示相对空间强度，不称绝对概率。
- 真值成熟/修订后通过新记录追加回放 SVG/离线 HTML，引用原图哈希且不修改原预测图；逐事件
  叠加默认只存本地受限范围。
- 默认最多 8 worker，至少保留 2 个物理 CPU 核；限制 BLAS/OpenMP 内层线程。
- GPU 只作与 CPU 数学等价的加速，不得改变候选、模型或报警前缀。
- USGS 自有数据默认公共领域，但 ComCat 伙伴来源可能例外。
- 逐事件行、精确坐标和原始响应始终本地受限；公开叠加逐项复核许可。
- 公开限制不阻断本地受控研究。

## 5. 重启后的严格续接顺序

1. 只审计上述 17 个 Stage 2P-1A 仓库文件的差异、内部一致性、官方链接、JSON Schema、SVG 和
   Markdown；明确跳过全部 Stage 4 草稿。
2. 确认没有未解决 P0/P1 后只形成“候选审计 GO”；它不等于阶段完成，所有授权仍为 false。
3. 候选审计 GO 后，在同一候选中把状态唯一跃迁为 `accepted`、`protocol_frozen=true`，仍保持
   execution/real 为 false，并针对这组精确最终字节重新跑完整验收与独立审计。
4. 只在最终字节复验 GO 后提交、推送、创建 annotated 协议标签，并回读远端 tag object 与
   peeled commit；远端闭合后才声明 1A 完成，标签之后不得再改本阶段协议内容。
5. 另行实现 Stage 2P-1B 纯合成同路径链，不接本地真实目录、不访问 ComCat 真实数据、不发行 issue。
6. 2P-1B 必须覆盖 `Q=T-15min` 同一快照、T-5 候选/T-4 missed 双证明、P0/R30/RP30、
   P0/P1/PP、RFC3161 core 与 TSA 主备假服务、scheduled/on_time 双序号、五类记录、时间上独立
   成熟抓取、preferred-origin 跨界修订、真值不可用且禁止 exposure 替补、formal-freeze
   unavailable 分支、最多两次 input-freeze/一次 effect look、支持域 IG/全区召回精确估计量、
   至少 10 个震群块、四端点区间、valid/invalid ResultBundle、`input_freeze→result_seal` 闭环
   和累计密封展示。
7. 合成链验收后立即做科学价值复审；它仍是必要使能，除非产生真实未来效果证据，否则不能称为
   预测提升。

## 6. 停止条件

- 本阶段合同、来源身份、记录 schema、统计规则或公开/本地边界不闭合：不进入 2P-1B。
- 2P-1B 出现新的 foundational P0：停止实现并复审，不接真实数据。
- 覆盖差异本身只记录；一次官方/目标无关证据供给失败可 missed，持续中断或实质变化则 pause
  后续发行并目标盲修订。
- 不得复用旧 Stage 2S 的 RP 选择器、runner、attempt、seal、拟合、Bootstrap 或 gate。
- 不得读取锁定测试，不得用真实震中生成候选或调网格。
- 不得因为后续来源或真值修订回填历史预测或更换模型版本。
- Stage 2P-1A 暂存禁止 `git add .`；只显式添加清单内 17 个文件，Stage 4 草稿始终排除。
- 不得为使用 GPU 引入神经网络或新候选。

## 7. 待主代理完成

- [x] 独立审计结论回收并关闭所有 P0/P1；
- [x] `git diff --check`、关键词一致性和文件范围审计；
- [x] 候选审计 GO；此时不得声称阶段完成；
- [x] 同一候选唯一设置 `status=accepted`、`protocol_frozen=true`，保持两个授权为 false；
- [x] 对状态跃迁后的精确最终字节重新完整验收和独立审计；
- [x] 最终复验 GO 后提交、推送并创建 annotated 协议标签；
- [x] 回读远端提交、tag object 和 peeled commit，闭合后才声明 Stage 2P-1A 完成；
- [x] 保持 `execution_authorized=false`、`real_issue_authorized=false`；
- [ ] 再进入 Stage 2P-1B 纯合成链。

## 8. 状态跃迁后的验收与远端闭环证据

- 三路独立复审最终结论均为 GO/STABLE，残余 P0=0、P1=0；
- 单进程、内层线程为 1 的最终测试：机器协议 19/19、语义状态机 183/183、既有背景与 Stage 2S
  协议回归 43/43，总计 245/245；
- Ruff、严格 YAML 重复键检查、JSON Schema 2020-12 metaschema、Markdown 本地链接与围栏、
  SVG XML、依赖锁结构与哈希、`git diff --check` 均通过；
- 17 个 Stage 2P-1A 文件范围闭合；15 个既有 Stage 4 草稿继续明确排除；
- 真实 issue=0、新前瞻目标读取=0、锁定测试读取/运行=0，真实目录和网络授权仍为 false；
- 科学价值仍是 `necessary_enabler`，没有真实效果指标，不能声称预测已经变准；
- 精确暂存的 17 个文件已提交并推送；远端分支、annotated tag object 和 peeled commit 已按
  上述身份逐项回读一致。Stage 2P-1A 因而正式关闭；
- 下一步仅允许进入 Stage 2P-1B 纯合成同路径实现与验收。1B 仍不得读取真实目录、访问 ComCat
  真实网络、发行真实 issue 或打开任何效果/锁定结果。
