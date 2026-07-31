# Stage 2P-1A 真正前瞻近期地震预登记

- 阶段：`Stage2P-1A`
- 文档状态：`accepted`
- 科学价值分类：`necessary_enabler`
- 直接预测提升：尚无
- `protocol_frozen=true`
- `execution_authorized=false`
- `real_issue_authorized=false`
- 下一阶段：`Stage2P-1B` 纯合成同路径验收
- 锁定测试读取次数：0
- 新前瞻目标读取次数：0

本文件是 Stage 2P 首期前的冻结协议。它只规定以后怎样按时保存预测、怎样等未来地震成熟、
以及最终怎样只看一次结果；它没有运行模型，也没有证明预测变好。2026-07-31 三路候选审计已
GO，同一候选已唯一跃迁为 accepted/frozen；随后仍必须对精确最终字节重跑验收，再提交、推送、
创建 annotated tag 并回读 peeled commit。只有这些步骤全部完成才可声明阶段完成。审计 GO 和
状态跃迁本身都不是阶段完成；当前不得发布真实 issue。

阶段边界固定如下：

- `Stage2P-1A` 只冻结 schema、唯一规范字节、数据表/数组/manifest profile、TSA 信任根、远端
  发布身份、状态机和统计语义，并证明合同会失败闭合；不声称真实 ASN.1/CMS、目录表、预测数组
  或评价字节已经实现；
- `Stage2P-1B` 才用纯合成数据实现并逐字节演练
  acquisition→表→P0/P1/PP→静态/交互展示→成熟真值→revision→input freeze→result seal；
- 只有 1B 合成全链验收并形成远端代码标签后才可能进入真实 shadow issue；在此以前
  `real_issue_authorized=false`。

外部 `TrustedReleaseManifest` 必须从已远端回读的 annotated tag 和 peeled commit 树构造，不能
从当前记录或未验证工作树自证。1A 状态只能是 `stage2p1b_required`；1B 合成验收后才允许
`stage2p1b_synthetic_accepted`。代码提交中的本配置和记录 schema 必须与协议提交逐字节相同，
并分别保存两个提交中的 blob/hash 身份。

## 1. 外行能听懂的版本

我们准备每周四同时保存三张不能事后修改的图：

1. `P0`：长期哪些地方更常发生地震；
2. `P1`：长期图再加上最近 30 天的地震活动；
3. `PP`：长期图再加上更早的 30 天地震活动，作为公平对照。

三张图使用同一时刻拿到的同一份地震目录、同一个研究区和同一套“不超过 60 万平方公里”的完整
格前缀规则；每张图实际选中的完整格、格数和精确裁剪面积都要原样记录，不能为凑满 60 万平方公里
切格或扩区。未来地震发生后，才检查 `P1` 是否同时胜过 `P0` 和 `PP`。这样可以区分“最近 30 天
确实有帮助”和“任何一段历史地震聚集看起来都有帮助”。

预测图和交互页在按时封存后可以立即查看，但累计命中率、累计信息增益和显著性在唯一正式判定前
保持密封。第一个检查点是第 52 个按时周预测；正式解封还要求三个 horizon 都可评价、至少 20 个
唯一去重 M5–6 地震和至少 10 个独立震群块。若门未满足，继续盲积累，最长到第 104 个按时
issue。这意味着短期内能看到的是诚实的预测图，不是“已经预测有效”的结论。

## 2. 科学问题和允许的模型

科学问题是：

> 在相同数据快照、冻结支持域、相同 7/30/90 天未来窗口和相同“完整格累计面积不超过
> 600,000 平方公里”规则下，最近 30 天 M4+ 地震活动能否让未来唯一去重 M5–6 地震的支持域
> 空间信息增益和全研究区严格召回同时超过长期 75 km KDE 及旧 30 天时窗对照？

`T` 表示一期规则起报时刻；`Q=T-15min` 是三组源事件共同且唯一的查询结束和起源时间截止。
只允许以下三个确认性模型：

- `P0(T)`：冻结 G1-LS 选出的投影、训练起点、局部 Mc/资格、支持规则、边界归一、网格和
  75 km Gaussian KDE 方法；每期从同一 `T` 前快照中截止 `Q` 的事件重新计算长期密度，不沿用
  2023 年旧密度。
- `P1(T) = 0.5 × P0(T) + 0.5 × R30(T)`。
- `PP(T) = 0.5 × P0(T) + 0.5 × RP30(T)`。

三组源事件必须由同一个选择函数、同一个快照和同一个可见截止产生，唯一允许变化的是起源时间窗：

| 组件 | 起源时间窗 | 可见性 |
| --- | --- | --- |
| `P0` | 冻结训练起点至 `Q` | 已存在于本期封存快照，且在 `T` 前已可用 |
| `R30` | `(Q-30d, Q]` | 与 P0 完全相同 |
| `RP30` | `(Q-60d, Q-30d]` | 与 P0 完全相同 |

`R30` 和 `RP30` 都是精确 30 天的半开左、闭右窗口。`RP30` 是“在 `T` 时已经知道、但起源较早”
的对照，不是 `T-30d` 的反事实快照。旧 Stage 2S
的 RP 实现把可见截止移到 `T-30d`，与本协议不同，明确禁止复用。旧 Stage 2S runner、已消费
attempt、fold/master seal、alpha 拟合和 Bootstrap 也全部禁止复用。

`R30` 没有合格事件时，`P1` 必须逐数组精确等于 `P0`；`RP30` 为空时，`PP` 必须逐数组精确
等于 `P0`。固定 `0.5` 只是两个已归一空间密度的混合系数，不是变量贡献率、最优权重或绝对发震
概率。任何目标出现后都不得改变它。

## 3. 数据源和 60 天洗脱

### 3.1 本地历史基线

现有本地地震目录只承担长期 P0 的历史基线角色，固定截止为：

`2026-07-09T04:25:56Z`

其路径、文件哈希、内容哈希、schema、行数、去重身份和许可状态必须在机器合同中绑定。该目录：

- 不得提供截止之后的前瞻增量；
- 不得作为未来目标真值；
- 不得因为后来看到 USGS 修订而回写；
- 不得再被包装为新的独立验证 cohort。

### 3.2 USGS ComCat 同源前瞻流

截止之后的预测增量和未来目标真值统一来自 USGS ANSS Comprehensive Earthquake Catalog
（ComCat）。正式实现必须保存请求参数、响应状态和头、抓取起止时刻、原始响应字节及 SHA-256，
不能只保存解析后的事件。

为避免 API `minmagnitude` 边界、显示舍入或服务端比较口径造成 M4 漏边，源查询固定下探至
`minmagnitude=3.9`，解析后再对数值震级执行本地严格过滤 `mag >= 4.0`；不能直接信任显示字符串。
HTTP 204 只表示本次查询合法返回空事件集，必须封存为空窗并让相应近期成分精确退化为 P0，
不能误记为抓取失败或借机改用其他来源。204 的响应体必须为 0 字节、所有事件/行计数必须为
0，并且 `geojson_parse_verified=false`，因为空响应体并不是可解析的 GeoJSON；只有成功的
HTTP 200 JSON/GeoJSON 响应才允许 `geojson_parse_verified=true`。

权威入口和政策：

- [USGS FDSN Event Web Service](https://earthquake.usgs.gov/fdsnws/event/1/)
- [USGS ComCat 说明](https://earthquake.usgs.gov/data/comcat/index.php)
- [ComCat DOI](https://doi.org/10.5066/F7MS3QZH)
- [ANSS Data and Products Policy](https://www.usgs.gov/media/files/anss-data-and-products-policy)
- [USGS Data Licensing](https://www.usgs.gov/data-management/data-licensing)

预测增量与真值使用同一来源，是为了避免一个目录的定位、震级或收录制度差异被误认为模型增益。
本地目录截止以后先执行完整 60 天同源洗脱，使首期 `R30` 与 `RP30` 都只由 ComCat 事件构成。
但 ComCat 在中国的 M4 级事件覆盖、震级口径和修订节奏可能与本地目录不同；60 天洗脱只消除
R30/PP 内部的换源混杂，不证明 ComCat 完全，也不证明其与本地目录等价。覆盖差异只作记录和
结论适用范围限制，不设置效果相关覆盖阈值。一次正式抓取失败、官方服务暂时不可用或冻结的
目标无关支持证据无法按时取得时，该期可以记 `missed_issue`；若官方来源、政策、schema 或
目标无关支持证据持续中断或发生实质变化，则暂停后续 issue，先形成不读取效果行的目标盲修订。
两种分支都不得补发，也不得通过查看未来目标来选择新源、改阈值或调模型。
正式 `count` 和 `query` 请求都以冻结顺序规范化 GET URL，并保存完整参数对象、URL UTF-8
SHA-256、响应头和原始响应哈希。`count` 与 `query` 使用相同的时间、空间、震级和事件类型选择，
但 count 明确排除 `orderby/limit/offset/includeallorigins/includeallmagnitudes`；解析计数必须
与 attempt/acquisition 的计数相等。count 达到 20,000 时不再发 query，直接闭锁失败，禁止把
真实大计数自报成较小值。count 固定 `format=geojson`，成功 HTTP 200 必须是 JSON/GeoJSON
内容类型并通过 GeoJSON 解析。区间请求不发送只适用于 `eventid` 的 `includesuperseded`；
`reviewstatus` 也不发送而使用官方默认 `all`，两者都不得进入规范 URL 参数序列。

切源边界的跨源重复只按以下冻结规则处理：两条记录起源时间差不超过 300 秒、WGS84 椭球距离
不超过 50 km、震级差绝对值不超过 0.5 时视为候选同一事件；按时间差、距离、震级差、稳定事件
ID 依次排序作一对一确定性匹配，保留 local 记录为 anchor，并把 ComCat ID 只作为 crosswalk。
不满足全部阈值不得人工合并。该规则只解决切源缝隙，不能用真实目标扩张为新的全局去重模型。

规则起报为每周四 `00:00 Asia/Shanghai`。首期必须同时满足：

1. 不早于 `2026-09-10 00:00:00 Asia/Shanghai`
   （`2026-09-09T16:00:00Z`）；
2. Stage 2P-1 协议标签和代码标签已经推送并完成远端对象及 peeled commit 回读核验；
3. 取上述核验完成后的下一次规则起报，不能把核验当周已经错过的时刻补发。

因此首期公式为：`max(2026-09-10规则时刻, 两个远端标签核验完成后的下一规则时刻)`。
cohort 记录必须嵌入两个 annotated tag 的远端 URL、tag object id、peeled commit、核验时刻、
原始回读响应 SHA-256 和 receipt SHA-256；peeled commit 分别严格等于 cohort 的 protocol/code
commit。cohort 还冻结 parser、规范化、去重、修订、模型、评价、展示、语义 validator 和环境锁
的结构化代码清单，后续五类记录逐项匹配，不能在看到未来目标后换代码。

### 3.3 每期源快照

每期 P0、R30 和 RP30 必须从同一个不可变 `T` 输入快照产生。固定
`query_end_utc=Q=T-15min`；正式抓取只能在 `Q` 到达后开始，并且完整按时预测候选必须在
`T-5min` 前完成、封印和取得远程时间戳。源快照至少绑定：

- 数据源、机构、FDSN 端点、请求方法和完整规范化参数；
- 规范请求身份、响应头、原始响应字节和抓取时刻；官方接口不提供可锁定的 provider-native
  catalog version，禁止伪造“来源版本”；
- 抓取开始与完成 UTC、HTTP 状态和影响内容解释的响应头；
- 原始字节数、SHA-256 和本地受限保存位置的相对身份；
- parser/schema、配置、首次可用规则和去重算法的代码提交与哈希；
- 解析、规范化、去重前后计数及内容哈希；
- 查询结束、内容新鲜度、快照截止和 `issue_T_utc`；
- P0/R30/RP30 事件 ID 有序集合、计数和集合哈希；
- 上一期 `IssueInputSnapshotRecord` 哈希。

快照封印和预测计算必须都在 `T` 前完成。只能使用快照中实际存在且
`origin_time_utc <= Q`、`available_at < T`、`first_seen_at < T` 的事件；不能用起源时间落在
`(Q,T]` 的事件，也不能把请求结束以后才出现的修订回填当期。封印与 T 之间的客观数据空档必须
原样记录，不能把最后事件起源时间冒充获取时间。

### 3.4 RFC3161 远程时间戳

每个按时 `IssueInputSnapshotRecord` 的 RFC3161 候选 core 规范字节摘要必须在 `T-5min` 前提交给
预登记的 RFC3161 Time-Stamp Authority。TSA 顺序固定为 DigiCert 主、Sectigo 备；每期只能先尝试
主站，主站产生可验证失败证据后才能尝试备站，不能临时换第三家或改变顺序。机器合同在更早公开的
协议标签中固定两者端点、允许 policy OID、精确 DER 信任锚和注册表哈希，并保存和验证完整
timestamp token、消息摘要算法、离线证书路径、TSA policy OID、`genTime` 和响应字节 SHA-256。
验证只证明 token 在 `genTime` 上可沿固定锚建立离线路径；不联网检查 CRL/OCSP，因此不得声称
当前未吊销或完整 PKI 非吊销保证。

所有 schema 要求 `remote_timestamp` 的正式记录，包括目标 cohort、Issue、成熟真值、真值修订
和评价冻结，
都使用同一个 RFC3161 core 规则：完整记录按 `seismoflux_canonical_json_v1` 编码，但**只且
恰好**排除三个顶层字段：
`timestamp_attempt_evidence`、`remote_timestamp`、`content_sha256`。排除只发生在顶层，禁止在
任何嵌套对象再放 RFC3161 proof/token 字段，也禁止递归删除同名字段。TSA 的 message imprint
必须等于该 core 的 SHA-256；最终记录的 `content_sha256` 仍按记录合同的“仅排除顶层
`content_sha256`”规则独立复算，不能把两种摘要混为一项。
每类记录还必须把 `record_core_frozen_at_utc` 与 `timestamp_deadline_utc` 放在未排除的顶层
core 中；因此 deadline 在发出 TSA 请求前已由 token 的 message imprint 承诺，不能在收到
token 后回填。

有效条件是：

- token 验证的消息摘要等于该记录的规范字节摘要；
- 离线证书路径、固定 DER 锚和策略符合机器合同；
- proof deadline 必须等于顶层 `timestamp_deadline_utc`；
- 每次主站或备站尝试都必须有非空 `attempt_completed_at_utc`，并满足
  `core_frozen <= request_started <= attempt_completed <= deadline`；收到 HTTP 响应的尝试
  必须有 `response_received == attempt_completed`，网络失败则必须是
  `response_received=null`，以实际确认失败的时刻作为 `attempt_completed`；
- 备站的 `request_started` 必须不早于主站的 `attempt_completed`，从而排除主备并发、倒序
  或未形成主站失败证据就启用备站；
- 选中尝试严格满足
  `core_frozen <= request_started <= genTime <= response_received == attempt_completed
  <= deadline`，且 `genTime < deadline`；
- cohort 的 deadline 精确等于 `valid_from`，Issue 按时候选精确等于 `T-5min`；成熟真值、真值修订和评价
  冻结的 deadline 都精确等于 `record_core_frozen_at_utc + 5min`；
- 即使 `remote_timestamp=null`，也必须先验证记录类别的 deadline、core 和全部失败尝试时序；
  失败记录不能绕过这些检查；
- token 原始字节作为记录的哈希绑定附件永久保存。

验证路径固定为锁文件中的 Python `cryptography==49.0.0` 与 `asn1crypto==1.5.1`，不调用未锁定
的系统 OpenSSL CLI。cohort 冻结验证代码、信任锚、允许策略 OID、TSA 身份和附件安装合同哈希。
每次 `.tsq/.tsr` 在本地受限且被 Git 忽略的
`data/interim/stage2p/timestamp_attachments/` 下，按 record type、preimage SHA-256 和
attempt 序号 create-only 保存，并重新
打开核验字节。每次失败尝试还必须记录 HTTP/内容类型/响应字节/解析或信任链失败的互相一致
证据；不能把一个实际成功响应自报为网络失败来启用第二 TSA。

Issue 使用两段不可混淆的证明：

- 最迟 `T-5min` 冻结“可按时安装的完整候选”并完成候选 TSA；成功才安装为 `on_time`；
- 候选形成前失败时 `prediction_generated=false`；完整候选已形成但候选 TSA 失败时，保留其
  本地受限、内容寻址字节并记 `prediction_generated=true`、`prediction_installed=false`；
- 两种失败都必须最迟 `T-4min` 冻结最终 `missed_issue` 审计 core，再用与候选尝试分离的
  RFC3161 请求在 T 前取得审计 token；
- 审计 token 也失败时，不安装正式漏期记录并暂停 cohort，不能靠本机时钟或事后补写维持连续性。

因此 `missed_issue` 可以证明“曾按时生成但未安装”的失败候选存在，却绝不能把该候选当作正式
预测、密度或报警格发布。RFC3161 token 是记录附件，不另造第六类正式状态记录。

## 4. 局部 Mc、空间模型和报警面积

局部 Mc 过高只影响对应本地支持单元的资格，禁止把局部最大 Mc 提升为全国统一阈值。研究区、
等面积投影、39 区映射以及 50/25/12.5 km 对齐网格在首期前冻结：

- 三模型的 KDE 训练事件只取 G1-LS 冻结支持格内合格事件，并只在该支持域上作连续边界归一；
  支持域外密度为不可支持而不是补一个极小值；
- 局部支持状态只按冻结合同影响本地事件资格或本地可评价性；
- 不得因真实震中、后来命中或局部 Mc 改变网格、边界、候选区或其它区域；
- 所有 KDE 固定为 75 km、等权、边界归一的连续二维 Gaussian 密度；
- 12.5 km 用于归一，质量对齐聚合至 25 km 用于报警；
- 50/25/12.5 km 必须通过冻结的总量和 L1 收敛门；
- 三模型使用相同事件率，因此只比较空间相对强度，不预测总数或具体日期；
- 信息增益只使用震中位于冻结支持域内的唯一 M5–6 目标；严格召回的分母始终是全研究区唯一
  M5–6 目标，支持域外目标对 P0/P1/PP 一律记为共同未命中，不能从召回分母删除。

报警统一按 25 km 格的“质量除以精确裁剪面积”排序，以 row、column、cell ID 作固定并列规则，
选择累计精确裁剪面积不超过 600,000 平方公里的完整前缀。600,000 是上限，不是必须达到的
实际面积。每个模型都必须封存完整有序排名、入选完整格前缀、入选格数、实际累计面积及“下一格
会超限或支持域已耗尽”的终止原因；不得切割最后一格、跳过放不下的一格继续挑后面的格，也不得
为了提高召回扩大面积。未耗尽时还必须封存下一排名格的位置、cell ID、精确裁剪面积和行哈希，
并机械验证剩余预算小于该格面积；耗尽时这些字段必须为 null。单个 25 km 完整格面积不超过
625 平方公里，三模型实际报警面积的最大差也不得超过 625 平方公里且必须由三路面积重算。

input freeze 必须嵌入目标盲 `AlarmAreaManifest`：每个被任一入选 exposure 使用的 prediction
seal 恰好一项，按 scheduled issue 顺序绑定 ForecastBundle、P0/P1/PP 三路实际 float64 面积和
三路最大差。结果阶段的 `AlarmAreaComparison` 只能由这份清单重算；缺期、替换 seal 或只有裸
SHA-256 摘要都失败闭合。

## 5. 每周状态机和五类只追加记录

正式状态只允许以下五类，文件创建后不得覆盖、截断、删除、重命名或重排：

1. `TargetCohortDefinition`：首期前冻结目标、成熟、去重、震群、区域和评价规则。
2. `IssueInputSnapshotRecord`：每个规则 T 恰好一条，`status=on_time|missed_issue`。
3. `MatureTruthSnapshotRecord`：按预登记成熟与抓取节奏保存一期 exposure 的 temporally
   independent mature snapshot（时间上独立抓取的成熟快照）或 `truth_snapshot_unavailable`。
4. `TruthRevisionRecord`：保存后续允许的真值修订或正式评价冻结前的预登记全量复核。
5. `EvaluationFreezeRecord`：用 `phase=input_freeze|result_seal` 闭合唯一正式查看；仍只有这一类
   记录，不增设第六类。

每个规则 T 都分开记录两个序号：`scheduled_issue_sequence` 对每个规则周四递增，包括
`missed_issue`；`on_time_issue_sequence` 只对成功按时封存的 issue 递增，missed 时必须为 null。
第 52/104 检查点只按 `on_time_issue_sequence` 计数，漏期既不能冒充按时 issue，也不能消耗
104 个按时 issue 的上限。为防止在长期供数中断下无限延长，`scheduled_issue_sequence` 最多
130；若第 130 个规则周四结束时尚未取得第 104 个按时 issue，直接
`evidence_insufficient_and_stop`。
130 上限优先：若第 130 期才累计到 52–103 个按时 issue，也不得触发第 52 期 look，只能停止。

五类正式记录都必须携带同一个 `code_manifest_sha256` 并与 cohort 中的结构化 code manifest
一致。manifest、远端 tag receipt、source snapshot、prediction seal、truth snapshot、预测
visualization evidence 和 replay evidence 的内容寻址 SHA-256 都使用明确的规范前像：只排除
该对象自己的 ID/hash 字段；语义校验器必须重算，禁止循环自引用或任意填 64 位十六进制。

`IssueInputSnapshotRecord.status=on_time` 时必须嵌套：

- 完整 source snapshot 身份；
- P0/P1/PP 模型身份、连续密度哈希和完整 25 km 报警前缀；
- 各模型实际报警面积、完整有序排名和完整格前缀、7/30/90 天有效期、`valid_from`；
- 代码、协议、环境锁、输入、事件集合、模型和输出哈希；
- 上一期记录哈希及 RFC3161 token 附件身份。

`status=missed_issue` 不安装 source snapshot 或 prediction seal。候选形成前失败时
`prediction_generated=false`；候选 TSA 失败时必须引用 T-5 前的本地受限失败候选，但仍固定
`prediction_installed=false`。正式漏期记录禁止包含预测密度、报警格或伪造的 `on_time` 字段；
错过的 T 永久缺失，不补图、不借用前后期预测。

所有记录从原始 Python/类型对象只规范编码一次。禁止把已经规范编码的 JSON 读回后再送入规范
编码器；复核已有记录不仅要检查原始文件字节哈希和严格 schema，还必须验证
`raw_bytes == canonical_json_bytes(strictly_parsed_object)`。带额外空白、不同键序或等价转义的
非规范字节即使解析值相同也必须失败，避免多个文件共享同一规范内容身份，并避免重演 Stage 2S
`$seismoflux_type` 二次编码故障。

## 6. 未来真值、成熟和固定重试

目标真值与预测增量同样来自 ComCat，但真值必须通过与 issue 输入抓取分离的、在预定成熟时刻
重新发起的请求取得，称为 **temporally independent mature snapshot（时间上独立抓取的成熟
快照）**。这里的“独立”只指抓取时刻和响应工件独立，不声称地震事件在统计上相互独立，也不得
复用 issue 输入响应充当真值。目标为研究区内唯一去重
`M5_6=[5.0,6.0)` 物理事件；M6+ 只作探索，永远不能满足主样本门。边界使用数值
`5.0 <= magnitude < 6.0`，M5.9 和 M6.0 不得按显示字符串或四舍五入混淆。

每个 `(issue, horizon)` 的目标窗是 `(T,T+h]`；FDSN 查询边界若为包含式，必须再在本地严格过滤
`T < preferred_origin_time_utc <= T+h`，不得复用 issue 的 cutover-to-Q 查询。在 `T+h+30d`
才首次成熟抓取，固定尝试时刻为：

`0h、+6h、+24h、+72h、+168h`

按顺序取第一个完整成功响应并停止后续重试；只有前四次都失败才会尝试第五次，不临时增加。每次
都保存请求与响应证据。五次全部失败时，追加
`MatureTruthSnapshotRecord.status=truth_snapshot_unavailable`，该 exposure 不可评分；不得把不可用
当作零事件，也不得临时换数据源。

成功真值记录绑定独立请求及响应、原始/规范化/去重三层哈希与计数、ComCat event ID、首见/更新
时间、该精确响应中的 preferred origin、preferred magnitude、位置和身份修订字段。跨快照同一
ComCat event ID 优先视为同一物理事件；其它疑似重复只按首期前冻结的确定性规则处理，人工预测
字段不参与。

每个实际 `input_freeze` 检查点只允许一次预登记的 formal-freeze 尝试：第 52 个按时 issue
最多一次；只有 52 的基本样本门未满足且从未打开效果行时，第 104 个按时 issue 才允许再有一次。
全线最多两次，禁止重试、替换失败响应或按结果调整请求；52 的 manifest/revision 在 104 形成后
仍永久只追加保留。若当次复核成功，只能新增 `TruthRevisionRecord`，不能覆盖
`MatureTruthSnapshotRecord`。正式评价的事件窗口、研究区、震级档和位置使用当次 `input_freeze`
绑定的 formal-freeze preferred origin/magnitude/location/identity；评价级震群则在该检查点全部
入选窗口的唯一 M5–6 目标并集上统一计算，不进入单个 target/revision 身份。若 preferred origin
使事件跨过 `T`、`T+h` 或相邻非重叠 exposure 边界，必须明列旧归属和修订后唯一归属；禁止双计
或人工挑选有利版本。任何后续修订都不得回填、重算或更换已经封存的模型、输入、预测数组和报警
前缀。

每次 formal freeze 用一条覆盖当次全部入选 exposure 的规范查询取得唯一一份原始 GeoJSON 响应；
官方接口不提供可供锁定的 provider-native catalog version，所以身份由规范请求、响应头、原始
响应字节和抓取时刻共同给出。禁止动态分页、分批拼接或临时换源。原始响应冻结后，在本地一次性
生成五张规范 JSONL 表：全响应规范化、全局去重、preferred 字段、全部窗口归属、窗口目标绑定；
每张表固定 row schema、排序、序列化、行数和文件/内容哈希。

当前发生变化的 revision 必须绑定当前 manifest；未变化窗口可引用已存在的历史链 tip，但历史
revision 继续绑定其原 manifest，不能改写成当前 manifest。当前 `EvaluationFreezeRecord` 只绑定
当前 manifest，并通过窗口目标绑定表明确历史链 tip 与当前 target set 的 changed/unchanged
关系；禁止跨 manifest 拼接表。无法从前后可观测行区分的“晚报”或“源修正”不得伪装成可验证
原因。TruthRevision attempt 独立于初始成熟真值的 `0/6/24/72/168h` retry，并在当次 input
freeze 前封印。

## 7. 统计设计和唯一正式查看

### 7.1 非重叠 exposure

7、30、90 天分别只根据 `on_time` 状态、`issue_time` 和该 horizon，按时间升序贪心选择：

1. 选择该层第一个合格 issue；
2. 下一期必须满足 `next_T >= previous_T + horizon`；
3. 保留所有入选 exposure，包括零事件 exposure；
4. 同一物理事件在同一 horizon 最多归属一个 exposure。

选择顺序在读取目标事件数、坐标、命中或分数前即可确定，以上任一目标内容都不得影响选择。若一个
已经入选的 exposure 最终为 `truth_snapshot_unavailable`，禁止用后续 issue 替补、平移锚点或
重跑贪心；该 horizon 在该检查点为不可评价。第 52 个按时 issue 时继续盲积累，第 104 个按时
issue 时记 `evidence_insufficient`。零事件但真值抓取成功的 exposure 仍完整保留，不能与
`truth_snapshot_unavailable` 混淆。

每次 input freeze 必须先嵌入 `SelectedExposureManifest`。它逐行记录 horizon、选择顺位、
scheduled issue、issue time、prediction seal、窗口、选中的真值记录及可用状态，并从全部按时
prediction seal 的冻结候选轴机械重放贪心选择。三个 horizon 的 exposure 摘要、truth
availability 摘要以及成功 formal freeze 的窗口绑定表都必须是该清单的确定性投影；禁止只填
64 位摘要后换 exposure。

每个 horizon 先形成自己的信息增益和严格召回，再以固定 `1/3、1/3、1/3` 权重形成三窗口宏平均。
不允许按结果、事件数或窗口表现改变权重。

### 7.2 去重和震群块

主样本门是全部正式 exposure 联集中至少 20 个唯一去重 M5–6 物理事件，不是 20 个重叠窗口命中。
同一联集还必须至少形成 10 个不同的震群连通块；事件数或震群块数任一不足都不得解封确认性效果。
震群块冻结为 30 天、75 km 的无向连通分量：

- 两事件起源时间差不超过 30 天且 WGS84 椭球测地距离不超过 75 km 时连边；
- 连通关系可传递；
- 孤立事件为单例块；
- component ID 是固定命名空间、NUL 分隔与排序唯一成员 ID 的规范字节 SHA-256，不使用
  任一真实震中派生边界；
- 同一事件跨 issue、horizon 和模型的全部贡献永远属于同一块。

时间比较精确为 `abs(origin_time_i-origin_time_j)<=2592000` 秒；空间比较固定使用
`pyproj.Geod 3.7.2`、`ellps=WGS84`、`inv` 的 float64 米距离 `<=75000.0`，比较容差为 0，再取
无向图传递闭包。

### 7.3 Bootstrap 和四端点同时区间

四个端点的点估计量先精确冻结如下。对 horizon `h` 和模型对比 `A-B`：

- 支持域信息增益为该 horizon 全部支持域内唯一 M5–6 目标在各自唯一 exposure 预测上的
  `Σ ln(f_A(x_e)/f_B(x_e)) / N_h_supported`，单位 `nats/event`；`f` 是在 G1-LS 支持域连续
  边界归一的 75 km KDE 在投影后事件坐标处的密度，禁止用格查找、插值或格均值代替。三模型
  共同事件率和积分补偿项相消，禁止再人为加入或删除补偿；
- 严格召回为 `Σ hit_M(e) / N_h_all_region`。分母是该 horizon 全研究区唯一 M5–6 目标，报警区是
  模型 `M` 在 600,000 平方公里上限下的实际完整格前缀；支持域外目标对三模型的 `hit` 都固定为
  0；
- 端点先在每个 horizon 分别形成上述模型差，再对 7/30/90 天三个 horizon 等权 `1/3` 宏平均。
  分母不得改成 exposure 数、震群数、命中数或三个 horizon 的事件合并数。

任一 horizon 没有至少一个可评分的支持域唯一 M5–6 目标、存在已选
`truth_snapshot_unavailable`、全研究区分母为零，或任一模型在目标坐标的密度为零/非有限值时，
该 horizon 不可评价并记 `evidence_insufficient`，禁止加任意密度 floor。事件对数比使用
deterministic float64 pairwise summation，等值绝对容差固定 `1e-12`。

使用 2,000 次配对震群块 Bootstrap。三个模型、两个对比、三个窗口和同一事件的所有贡献联合
重采样。若原始样本有 `B` 个震群块，每次严格有放回抽取 `B` 个块；同一抽样索引同时用于全部
模型、对比和 horizon，重复抽到的块按其抽样乘数重复全部事件贡献，并据此重新计算每个 horizon
的支持域事件分母、全研究区事件分母和三窗口宏平均。零事件 exposure 仍保留在冻结 cohort 中，
不因抽样被删除。随机流精确由
`numpy.random.Generator(numpy.random.PCG64(147))` 初始化；Stage2P namespace 只作身份标签，
不得再派生另一个 seed。打开效果行前一次生成并冻结完整 2,000×B 索引矩阵，抽样顺序按
component ID UTF-8 排序，禁止丢弃或重抽复本。任一复本在任一 horizon 的支持域或全区分母为零，
立即在盲态记 `evidence_insufficient`，不得先看效果。

一个确认性 family 包含四个端点：

1. `P1-P0` 宏信息增益；
2. `P1-PP` 宏信息增益；
3. `P1-P0` 在 600,000 平方公里上限完整格前缀下的严格召回增益；
4. `P1-PP` 在同一上限规则下的严格召回增益。

对四端点使用 Bonferroni 95% familywise 双侧同时区间。每个端点使用 98.75% percentile 区间，
即对 2,000 个重采样估计按 NumPy percentile `method=linear` 取 `0.625%` 和 `99.375%` 分位数。
四个下界必须同时大于 0，并且
`P1-P0` 严格召回点估计至少为 `+5` 个百分点。

### 7.4 39 区和最大震群稳健性

每个端点必须按冻结的 39 个 construction zone 做可加闭合。零事件区保留但不能充当正区域。
对每个端点分别移除最大正贡献区域，并保持原固定分母，剩余点估计必须大于 0。

对每个端点也分别移除最大正贡献震群块并保持原固定分母，剩余点估计必须大于 0。任一端点无法
闭合、只由单一区域或单一震群支撑，正式门即失败，不能降格为通过。

### 7.5 52/104 的唯一 look

- 第 52 个 `on_time` issue 的全部 90 天窗口完成成熟等待后，先追加
  `EvaluationFreezeRecord.phase=input_freeze`。它只能绑定 cohort definition、formal-freeze
  truth/revision 链、目标无关选出的 exposure、三个 horizon 可评价性、事件数、震群块数、全部
  预测哈希、代码/协议/环境、随机 namespace 和输出目标位置，禁止包含四端点效果行。
- 只有三个 horizon 都可评价、主样本 `N>=20` 且独立震群块 `B>=10`，该 `input_freeze` 才能
  授权打开效果行；随后必须立即追加且只能追加一个直接引用其哈希的
  `EvaluationFreezeRecord.phase=result_seal`，一次性封存四端点点估计、2,000 次区间、39 区/
  最大区域/最大震群稳健性、总门和终止决定。`input_freeze` 与 `result_seal` 之间不得插入第二套
  输入、第二次计算或可见的中间成绩。
- `horizon.evaluable` 由完整 exposure 数、真值可用 exposure 数、两个目标分母、震群数和全部
  预测密度有限且为正机械重算；false 必须给出与证据一致的结构化原因，不能人工自报绕过 52 门。
- 评价代码 commit/hash 以及 `uv.lock`、`pyproject.toml` 哈希在 input freeze 就固定；result seal
  必须逐项复用冻结输入并使 previous/input/nested input hash 闭合。
- input freeze 的选中 TSA 响应必须先完成，才能记录 `effect_rows_opened_at_utc`；随后才允许
  打开效果行，且 result core freeze 和 result TSA 必须更晚。result seal 仅允许改变 phase、
  sequence、input/previous 链哈希、结果与封印时刻/证明字段；其余科学输入逐项规范字节相等。
- 正式生命周期校验必须同时取得由记录哈希引用的本地受限 target、cluster、point/region/cluster
  contribution、bootstrap、selected-exposure、alarm-area、effect/result manifest 工件；先复算
  工件哈希，再从行值重算 union N/B、端点、闭合、最大正贡献和并列规则。只有 64 位摘要而缺少
  对应工件时失败闭合，不能把自报计数当成已重算。
- formal freeze 成功时 target、window、N/B、cluster 和 horizon 目标字段必须为实际值；只有
  机械证明 selected exposure 为空时才允许规范空集。count/query/parse/本地冻结失败或 scheduled
  cap 未形成 formal freeze 时，相关科学字段必须为 null/unavailable，禁止用 0 冒充观测零。
- 若授权后执行失败，唯一 result seal 引用精确 invalid `ResultBundleManifest`。它按
  `effect_rows_open→alarm_area_comparison→bootstrap→endpoint_evaluation→robustness_evaluation
  →result_bundle_install→result_seal` 固定真值表，只允许保留失败前已完整封存的槽；部分产物仅
  进入排序去重 audit 哈希，未完成端点/稳健性槽必须为 null。仍永久停止，不能重跑。
- 若第 52 个按时 issue 的样本门未满足，只在 `input_freeze` 封存水位和失败原因，不创建
  `result_seal`，不能构造、解封或展示对比行；继续盲积累至第 104 个按时 issue。
- 第 104 个 `on_time` issue 的全部 90 天窗口完成成熟等待后同样先写 `input_freeze`。若仍有
  horizon 不可评价、`N<20` 或 `B<10`，状态为 `evidence_insufficient` 并停止 P1，不创建确认性
  `result_seal`。
- 若第 52 期已经实际打开对比，无论通过或失败，都不得在第 104 期再看一次。

因此任一路径最多存在一个确认性 `result_seal`，且它必须由唯一授权的 `input_freeze` 直接闭合。
闭合失败即停止，不得以第二次运行补救。预测地图、单期数据质量和单期成熟事件叠加不属于
正式 look；累计 P1-P0/P1-PP 命中、召回、信息增益、区间和显著性全部属于密封内容。

## 8. 静态图和离线交互

每个 `on_time` issue 在 T 前完成并随预测封印：

- 一张 P0/P1/PP 同域静态对比图；
- 一张三模型“不超过 600,000 平方公里”的实际完整报警前缀图；
- 一个无外部网络依赖的离线交互页，可切换模型、7/30/90 天有效期、R30/RP30 窗口、数据截止、
  实际面积和归档哈希；
- 清楚标注“相对空间强度，不是绝对发震概率”。

图件可以在 T 后立即查看。目标成熟后可以通过新记录追加单期事件叠加，但不得改写原始 HTML、
图片或预测数组。唯一正式 look 前，界面不得显示累计对比、滚动胜负、累计命中率、累计信息增益、
置信区间、p 值或任何暗示 P1 已经更好的颜色/徽章。
`MatureTruthSnapshotRecord` 和 `TruthRevisionRecord` 必须嵌入 append-only replay evidence：
引用原 prediction visualization 哈希和前一回放哈希（首个成熟回放为 null），绑定成熟目标集、
新 SVG/离线 HTML/script/overlay 哈希，证明原图未修改、零外链、零远程脚本、零网络抓取、零
未成熟目标叠加。prediction visualization 和每次 replay 都有排除自身 hash 字段后的规范内容
SHA-256；revision 的前一 replay 哈希必须沿 `previous_truth_record_sha256` 链闭合。逐事件回放
默认 `local_restricted_only`，公开前另做逐项许可复核。

## 9. 计算资源与可重复性

- 运行时至少为用户保留 2 个物理 CPU 核；
- 默认 `worker_count = min(8, max(1, physical_core_count - 2))`；
- 禁止外层 worker 与 BLAS/OpenMP 内层同时占满核心；
- 记录物理核数、worker 数、线程环境、内存、GPU 型号、驱动和耗时；
- GPU 只允许作为与 CPU 数学等价的加速器，不得改变模型、精度、排序、并列规则或统计门；
- 若 CPU/GPU 输出不能在预冻结容差内等价并生成相同离散报警前缀，正式结果以 CPU 路径为准，
  GPU 路径停用；
- 不得为了使用 GPU 引入神经网络或新增候选。

## 10. 公开与本地受限边界

公开仓库允许：

- 代码、机器配置、协议、测试和环境锁；
- URL、许可说明、请求参数模板；
- 原始/规范化/去重数据的哈希与计数；
- 聚合 25 km 相对强度、完整报警前缀和许可允许的事件叠加；
- 五类记录的公开安全字段、RFC3161 token、代码/协议/输出身份；
- 静态图和不含受限明细的离线交互页。

默认只存本地受限目录：

- ComCat 原始响应字节；
- 规范化和去重逐事件行；
- 精确坐标和完整事件清单；
- 本地历史目录及其逐行内容；
- 可能受来源条款或安全边界约束的详细元数据。

USGS 自有数据默认按美国公共领域处理，但 ComCat 汇集的伙伴来源可能存在例外。因此逐事件行和
精确坐标始终按本地受限处理，任何公开事件叠加都必须逐项完成来源与许可复核；未通过时只公开
哈希、计数和聚合结果。该发布限制不阻断本地受控科学研究。不得用“公开仓库”倒推原始数据可
公开；API 密钥、凭据、本机绝对路径和受限目录结构永不入库。

## 11. 停止条件

以下任一情况立即停止相应动作：

- Stage 2P-1A 合同、官方来源、目标无关支持证据、许可、RFC3161、五类记录或统计定义不能闭合：
  不进入 2P-1B；
- 2P-1B 纯合成链出现新的 foundational P0：停止实现并复审，不接真实数据；
- 协议/代码标签未远端核验、早于首期下限或错过规则 T：写 `missed_issue`，不得补发；
- 源快照过旧、正式抓取/TSA 失败、官方服务或目标无关支持证据未能按时供给、封印不早于 T：
  候选形成前失败时不生成预测；完整候选已形成但候选 TSA 失败时保留本地受限失败候选并写
  `prediction_generated=true`、`prediction_installed=false`，但绝不安装为正式预测。两类都按
  T-4 missed 审计分支处理；持续或实质性中断则暂停后续发行并作目标盲修订；
- 任一期混入 Q 后起源事件或 T 后可用数据、三模型不是同一快照/同一路径/同一面积上限规则、
  未记录各自实际完整前缀面积、历史预测被改写或记录哈希不可复算：当前版本 `invalid`，停止 P1；
- 真值在五个预定时点均未取得完整成功响应：该 exposure 标记不可评分，不得记零、临时加试或用
  后续 issue 替补；
- 第 52 期实际打开后正式门失败，或第 104 期仍有 horizon 不可评价、事件少于 20 或独立震群块
  少于 10：停止 P1、保留 P0；
- 结果由最大震群或最大区域支配而稳健性门失败：停止 P1，不改震群、区域或分母；
- ComCat/ANSS 来源、schema、许可、preferred-event 规则或冻结的目标无关支持证据发生实质变化：
  暂停后续 issue，先形成
  新的目标盲修订；不回填已经错过的期；
- ComCat 覆盖差异必须记录但不设效果相关硬阈值；供给失败按预登记的 missed/pause 分支处理，
  且不得查看目标后换源或调参；
- 禁止针对已观察 cohort 改 30 天窗、0.5 权重、75 km 带宽、面积、模型、Bootstrap 或
  familywise 构造，也禁止用旧 Stage 2S、异常、Hawkes、神经网络或锁定测试补救。

## 12. 科学价值复审

- `science_value_category`: `necessary_enabler`
- `direct_prediction_improvement`: `none`
- `evidence`: 只完成了未来检验的来源、时间、模型、归档和统计规则设计；尚无一条未来预测或效果
  指标
- `decision`: `continue_to_stage2p1b_synthetic_only_after_acceptance`
- `next_scientific_test`: 用纯合成目录完整演练 `Q=T-15min` 同一快照的 P0/R30/RP30、
  P0/P1/PP、按时与漏期、RFC3161 core/主备假服务、五类只追加记录、时间上独立成熟抓取、
  preferred-origin 跨边界修订、不可用 exposure 禁止替补、支持域 IG/全域召回精确估计量、至少
  10 个震群块、四端点同时区间和 `input_freeze→result_seal` 密封闭环
- `stop_condition`: 合成链任何因果、不可改写、同路径、统计或资源门失败即停止；在协议与代码标签
  远端核验前始终保持 `execution_authorized=false`、`real_issue_authorized=false`

未来若通过本协议得到阳性或阴性结论，结论范围也只覆盖冻结的 ComCat 获取/修订链、局部资格规则、
75 km KDE 和固定 0.5 候选，不能外推为“全部近期地震模型有效或无效”。

## 13. 待验收

本段由主代理在实际完成后补写，当前不得写“已验收、已提交或已推送”：

- [x] 三路独立审计无未解决 P0/P1；
- [x] 机器配置与本文逐字段一致；
- [x] 协议测试、链接检查、Markdown 检查和 `git diff --check` 通过；
- [ ] 科学价值复审字段齐全；
- [x] 候选审计 GO 后，在同一候选唯一一次把状态改为 accepted/frozen；
- [x] 对上述精确最终字节重跑全部验收；
- [ ] Stage 2P-1A 验收提交与推送完成；
- [ ] 协议标签远端对象与 peeled commit 回读一致；
- [x] `execution_authorized=false`、`real_issue_authorized=false` 始终保持不变；
- [ ] 仅在以上全部完成后，授权进入 Stage 2P-1B 纯合成链；
- [ ] 本阶段不授权真实 issue。
