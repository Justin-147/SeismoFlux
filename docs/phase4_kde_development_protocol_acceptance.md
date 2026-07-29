# 阶段 4A KDE + 异常开发科学协议验收

- 日期：2026-07-30
- 协议 ID：`stage4-kde-development-v1`
- 门控：`S4-KDE-DEV`
- 协议标签：`v0.3.2-kde-anomaly-increment-protocol`
- 当前范围：只验收目标盲协议，不验收模型效果
- 开发目标读取：0
- 锁定测试读取或运行：0

## 1. 验收边界

本次只回答：下一步是否已经被压缩为一次能够直接检验最终科学目标的最小开发实验。它不打开真实
Stage 4 开发目标，不运行任何模型评分，不生成事件数、信息增益、召回、p 值或采用结论。

旧 R2 协议和其 ETAS 阻断记录保持不变。本协议另立 `stage4-kde-development-v1` namespace，
仅把旧 R2 中目标盲冻结的三个滚动折、特征定义和空间分层作为科学定义输入；不继承旧 R2 的执行授权、
attempt、seal、ledger 或随机流。

## 2. 协议身份

| 文件 | SHA-256 |
| --- | --- |
| `configs/anomaly_increment_kde_dev.yaml` | `a3e2bdd6a8b14fc4d64958c8dc3586a3ac86d637710d6869d26508cefc2a3631` |
| `docs/anomaly_increment_kde_dev_protocol.md` | `7caa02a9e9b877850f276f65e836ee622b0b809cb46d367961ef47fce43ce74d` |
| `tests/unit/test_stage4_kde_development_protocol.py` | `f2bc4f752b58d77b80fb11d1a354b434c4eaa90e4aad655d0e6a7d857284ec3e` |
| `data/manifests/anomaly_increment_kde_dev_inherited_contracts.json` | `9d3b2585ce068ad13f80dac7489d6358a96b2adfc6fb4d30e5d9dd274653c99e` |
| `SEISMOFLUX_IMPLEMENTATION_HANDOFF.md` | `34eeabc70a96582055aa9cf9d53c526dbdd6dd110b6853b964523efc5828ba77` |
| `configs/research_protocol.yaml` | `c265d7fb039071a8e3750e2b9807ea4a1dbd91d00ac1d20c0d6a90bc7ac374e7` |
| `docs/research_protocol.md` | `167c20f6a331e3b58516d4d62c70909a0184161177dfe7e8eaef1c7bf9bf5598` |
| `docs/scientific_value_review_and_model_composition.md` | `83cf71bf4f6fdb185107fc92d5fc66209c8a04720f47cb2694d065a56464c2eb` |

机器协议逐项绑定：

- Q2 ETAS `not_evaluable` 结果；
- G1-LS 已通过的 75 km KDE 注册表和本地模型声明身份；
- Stage 3 已验收特征注册表、字典、特征库与状态库声明身份；
- 三折日历、9/17/22 个逻辑特征和 39 个非空目标无关空间区，并由新 namespace allowlist
  明确拒绝旧 R2 授权字段；
- 地震目录、研究区和环境锁文件声明身份；
- 计划 protocol/code/result 三个互不相同的标签。

## 3. 科学合同验收

- [x] 唯一问题直接对应“同一 600,000 平方公里内是否提高未来经目录去重的 M5–6
  物理地震事件严格召回”；“去重”不声称事件在统计上相互独立。
- [x] `B0/C0/B1/B2` 共享背景、支持域、目标、格网、未来窗、积分域、训练事件率和面积。
- [x] 开发折只使用截至 2019-12-31 的 `fold_4` KDE；截至 2023-06-30 的验证快照禁止回填。
- [x] 通过 `C0-B0`、`B1-C0`、`B2-C0`、`B2-B1` 分别识别覆盖制度、快照异常、
  全部异常和动态轨迹贡献。
- [x] 7/30/90 天、三个滚动折、2,000 次 Bootstrap、1,000 次时间置乱和 1,000 次空间置乱在看分前固定。
- [x] 九个“折 × 窗口”单元先窗后折等权；零事件单元不得删除，直接判证据不足。
- [x] Bootstrap 按“折 × 7/30/90 天三位成员签名”分层、层内同样本量有放回抽样；同一事件
  multiplicity 配对用于全部模型与指标，并保持每折—窗口事件边际数。
- [x] IG 与固定面积召回分别对 `B1/B2 × B0/C0` 四对比构造 Bonferroni 家族 95% 同时
  percentile 区间；通过门使用同时下界，普通边际 95% 区间只作诊断。
- [x] 实际意义门固定为 600,000 平方公里严格召回 `+5 pp`，且相对 `B0/C0` 的家族 95%
  同时区间下界均大于 0。
- [x] 同召回面积曲线按完整格前缀和 625 平方公里预算步长计算，但只作描述。
- [x] 同时要求信息增益、两类置乱、跨折、覆盖混杂和跨区稳健门。
- [x] `B1/B2` 的时间、空间置乱分别使用同一复本内单步 `maxT` 校正；不能比较两个裸 `p`
  后选择较好候选。`maxT` 只对 `B1-C0/B2-C0` 的覆盖控制外异常增量各算一次；相对 `B0`
  和 `C0` 的方向、区间、固定面积与稳健性门仍分别满足，`B2-B1` 只承担组件采用门。
- [x] `B2` 动态门失败只否定轨迹；`B1` 若独立通过全部共同门，仍可保留。
- [x] 候选状态先逐个判定再汇总；未采用候选的较差状态不能覆盖已通过候选。`B2` 动态增量门
  **失败**时仍可回退已通过的 `B1`；但 `B2` 本身拟合/评分 **无效**时，不能把 `maxT` 或同时区间
  缩成单候选，整体固定为证据不足且不采用异常候选。共享泄漏、身份或 `B0/C0/B1` 无效始终先于
  候选采用判为整体无效。
- [x] Stage 4A 通过只授权再冻结一次独立验证，不宣称 G2/G3 或业务可用。
- [x] 失败和证据不足都停止异常模型扩张，不授权看分后换模型、换折、换面积或进深度模型。

## 4. 防泄漏与执行验收

- [x] 协议冻结时成绩、目标计数、开发目标读取和锁定测试读取均为 0。
- [x] code commit/tag 远端核验前，真实 processed 输入字节和开发目标字节均禁止。
- [x] 目标只允许在唯一 attempt 内用于评分命中/漏报，禁止建格、空间加密、画边界、选阈值或调参。
- [x] 人工预测字段、未来异常回填、随机切分和目标派生空间设计全部禁止。
- [x] 旧 R2 的 ETAS 执行授权、attempt、seal、ledger 和随机流明确不复用。
- [x] code tag 必须冻结新 namespace 的预期输入清单、input identity seal、code seal、零操作
  attempt ledger、随机清单和输出 schema。
- [x] code freeze 使用无自引用双提交：源提交 `S` 冻结代码/schema；seal 提交 `C` 的唯一父提交为
  `S`，只准改四个固定清单路径，tag 指向 `C`；seal 禁止绑定 `C` 或自身哈希，preflight 核验
  `C→S`、精确 tree diff 与无环交叉哈希。
- [x] preflight receipt 不读目标，并绑定预先创建/核验为零的 target-read ledger 哈希与计数；
  随后才以 CAS 登记唯一 attempt。唯一 target adapter、hash-chain target ledger 和 checkpoint
  共同机械限制一个逻辑读取会话和一个 attempt。
- [x] 技术中断只能在同代码、配置、输入和随机身份下恢复同一 attempt，最多一次；代码或身份改变不得恢复。
- [x] 随机根种子 147 作为 `root_seed_decimal` 进入规范上下文；候选、比较器、窗口和指标不进入
  seed 字段，以保持配对。
- [x] 旧 R2 的 `scoring_pipeline`、formal orchestrator、旧门、五窗口 Bootstrap、placebo
  和 `Stage4SeedContext` 禁用；只按精确 symbol allowlist 复用低层科学原语。
- [x] worker 运行时公式固定为 `max(1,min(6,physical_cores-2))`，BLAS 每进程 1 线程、
  `BelowNormal`，物理核心少于 3 个时失败关闭。
- [x] GPU 只允许使用 code tag 前已证明等价的路径；不得为本门扩写新后端。

## 5. 展示验收

- [x] 公开聚合页和同 schema 静态图显示最终状态、采用候选、失败门、科学价值决定、基线、消融、
  同时/边际区间、两类置乱、`B1/B2` 各自的时间/空间 `maxT p_adj`（裸 `p` 仅诊断）、固定面积、
  完整面积—召回、Molchan、样本分母、跨折/跨区和特征组解释。
- [x] 聚合页与静态图都有外行可读的数据/方法流程面板；系数只报三折点估计和中位/范围，不虚构
  置信区间，组置零固定报告宏 IG、600,000 平方公里召回变化与格网强度秩变化。
- [x] 205 快照/3,217,885 行明确标为 Stage 3 源库总量，不冒充本 attempt 样本量；结果逐折报告
  实际读取的快照/特征行数及因未来期或正式验证保留而未用的原因。
- [x] Molchan 横轴固定为面积×时间曝光占全研究区时空曝光比例，纵轴为全区漏报率，unsupported
  目标计漏报；按九个折—窗口等权，只作描述、不参与通过门。
- [x] 公开和本地 target-free 页明确标为“历史/开发重放，不是前瞻证据”，禁止目标、命中和漏报；
  公开页只允许栅格化概化空间展示。
- [x] 通过时预测页主层为选中候选；失败/证据不足时回退 `B0`，异常层默认关闭并标未采用；
  `invalid` 时不发布预测页。
- [x] 本地回溯页才可叠加目标、命中、漏报和失败案例，并必须醒目标注“历史回溯，不是真正预测”。
- [x] 本地另提供精确 25 km、仍不含目标的历史/开发重放页，不把它冒充真正预测。
- [x] 真正前瞻 issue 只能在结果标签后按时生成，必须记录数据截止、输入/模型/配置/代码身份、
  生成时间和不可变归档路径，且禁止回填、覆盖或预载目标。
- [x] 公开产物禁止原始事件/异常 ID、原始坐标、受限边界和本地路径。
- [x] 所有强度只称相对条件强度，不称绝对发震概率。

## 6. 本地验证

最终目标盲协议回归命令为：

```powershell
D:\AIPred\SeismoFlux\.venv\Scripts\python.exe -m pytest `
  tests/unit/test_stage4_kde_development_protocol.py `
  tests/unit/test_stage4_anomaly_increment_protocol.py `
  -p no:cacheprovider -q `
  --junitxml=data/interim/stage4/anomaly_increment_kde_dev/protocol_tests.junit.xml
```

- 结果：`24 passed, 1 skipped in 4.07s`；
- 唯一 skip：旧 R2 的四个本地 restricted 空间工件在本 worktree 不可用，不影响本协议目标盲合同；
- JUnit SHA-256：
  `ab0b49b4c97f77d720741cbff3e4c1fe44c0d39628515252512354df292b4bb5`；
- Ruff check：通过；
- Ruff format check：通过；
- strict mypy（新协议测试）：通过；
- `git diff --check`：通过。

上述测试只读取 tracked 公开元数据和 synthetic fixture，没有打开或枚举 `data/processed`、正式
Stage 4 目标或锁定测试。

## 7. 独立复审

第一轮独立复审已明确 `FAIL`，并保留为审计历史：

- 科学复审：`P0/P1/P2=0/5/2`；
- 工程治理复审：`P0/P1/P2=3/6/2`；
- 可解释性复审：发现三项阻断，包括九单元宏统计量不唯一、同召回面积算法未冻结，以及预测页与漏报
  展示合同冲突。

修订没有增加候选模型，而是补齐因果 KDE 快照、覆盖度归因、快照回退、固定面积唯一主门、统计公式、
多候选误选控制、状态真值表、无自引用 `S→C` seal/ledger/checkpoint、资源公式和三页面 schema。

最终第二轮独立复审结论：

- 科学设计复审：`PASS`，`P0/P1/P2=0/0/0`；
- 工程治理复审：`PASS`，`P0/P1/P2=0/0/0`；
- 可解释性与输出复审：`PASS`，`P0/P1/P2=0/0/0`。

三项审计在 config/test 状态提升前的完整实质字节
`a2ce8770… / 7caa02a9… / 9a986d5f… / 9d3b2585…` 上完成；其后唯一变更是把机器状态及对应测试
从“待复审”机械提升为“本地验收通过、待远端标签”。上述最终回归已在提升后的表列哈希上通过。

## 8. 本阶段科学价值复审

- `science_value_category`: `necessary_enabler`
- `evidence`: 本阶段没有预测效果证据；实质作用是把下一步压缩为一次直接比较 KDE 与快照/动态异常、
  固定面积召回、两类置乱和跨折/跨区稳定性的有限科学 attempt。
- `decision`: `continue`
- `next_scientific_test`: 在 protocol commit/push/tag 远端核验后，只实现并冻结薄运行器和
  target-blind KDE adapter；code tag 后执行唯一 `S4-KDE-DEV` attempt。
- `stop_condition`: 若实现不能保持为复用现有纯科学原语的薄层，或需要扩张旧 R2/ETAS 正式工程链、
  改动科学门、提前打开真实输入/目标，则停止代码扩张并重新审视方案。

这一定性不是 `direct_improvement`。真正的直接进展必须来自唯一 Stage 4A 结果：在同支持域和固定
600,000 平方公里下提高严格召回，并同时通过预登记不确定性、置乱、覆盖控制和稳定门。

## 9. 外部发布门

本文件只能记录冻结提交之前已经完成的本地事实。以下外部事实必须在提交后写入 Git-ignored 当前
重启交接，不能让本提交对自身未来状态作自证：

1. 协议提交已推送到 `origin/codex/stage2-etas-science-first`；
2. annotated tag `v0.3.2-kde-anomaly-increment-protocol` 已推送；
3. 远端分支、tag object 和 peeled commit 与本地完全一致；
4. 提交后工作树干净。

四项完成前，Stage 4A 代码、真实输入打开和开发目标读取继续禁止。
