# SeismoFlux 重启续接交接：Stage 2P-1B 信任预检 NO-GO

- 交接时间：2026-07-31
- 分支：`codex/stage2-etas-science-first`
- 预检起点提交：`13c9380b41e5f05d39d62b8f64f4ca48df78d2d4`
- 已关闭阶段：`Stage2P-1A complete`
- 当前阶段：`Stage2P-1B preflight_foundational_P0_no_go`
- 实现状态：`not_started`
- 下一步：`target_blind_protocol_amendment_requires_decision`
- 科学价值分类：`no_material_progress`
- 直接预测提升：`none`
- 真实目录读取：0
- 真实网络请求：0
- 真实 issue：0
- 新前瞻目标读取：0
- effect rows 打开：0
- 锁定测试读取/运行：0/0
- Stage 4 未跟踪草稿：保持原样，未修改、未暂存、未删除

## 1. 外行版结论

Stage 2P-1A 已经完成：未来预测的考试规则已经锁好并推送到远端。

Stage 2P-1B 原计划用完全虚构的数据和两台“假时间戳服务器”演练整条流程。但开工前发现：

- 锁只认 DigiCert 和 Sectigo 两家的真实钥匙；
- 演练又要求假服务器现场生成能打开这把锁的新钥匙；
- 仓库里没有两家机构的私钥，也没有事先签好的合法测试回执；
- 规则还禁止换成我们自己的测试锁。

因此，当前规则下的合成成功链在密码学上不可能完成。继续写代码只会得到两种坏结果：永远失败，
或者偷偷跳过验证后假装通过。两者都不能推动地震预测目标。

本轮已按停止条件在读取真实地震目录、访问真实网络或生成真实预测之前停下。当前没有模型效果
结果，也不声称预测已经变好。

静态图见 `docs/stage2p1b_trust_preflight_no_go.svg`。

## 2. 已确认的硬冲突

### 2.1 冻结信任边界只允许真实机构

`configs/prospective_recent_seismicity.yaml` 的 `remote_timestamp.trusted_registry` 固定：

- 主站 `http://timestamp.digicert.com`；
- 备站 `http://timestamp.sectigo.com`；
- 两家各自允许的 TSTInfo policy OID；
- 两张精确 DER 信任锚及其 SHA-256；
- 禁止由记录提供替代根、policy 或 authority mapping。

`data/contracts/stage2p_prospective_records.json` 又把这些 authority、状态、proof 和
`offline_trust_path_valid=true` 的成功语义固定下来。

### 2.2 合成验收要求真正的密码学成功

Stage 2P-1B 不是只检查几个布尔字段。冻结合同要求实际重建并验证 RFC3161 ASN.1/CMS：

- token 签名和 signer/intermediate 证书链必须到达精确冻结锚；
- signer 必须满足时间戳 EKU、KeyUsage 和 CA 约束；
- message imprint 必须匹配本次记录动态生成的 core SHA-256；
- nonce、policy OID、`genTime` 和截止时刻必须匹配；
- 主站形成可验证失败证据后才可尝试备站；
- 成功的 cohort、on-time issue、成熟真值、修订和评价封印都必须具有非空合法 proof。

只演练两家都失败的分支不能形成完整五记录成功生命周期，也不能将
`implementation_status` 改为 `stage2p1b_synthetic_accepted`。

### 2.3 缺少合法离线成功材料

只读扫描已确认仓库没有：

- DigiCert/Sectigo 对应私钥或可签发中间 CA 私钥；
- `.tsq`、`.tsr`、CMS、TSTInfo、`.der`、`.crt`、`.pem` 等预签名测试向量；
- 独立的 `synthetic_acceptance` 信任注册表；
- 合成根、合成 signer、合成 policy 或 production/synthetic trust mode；
- 对精确动态 core 和 nonce 预先签发并冻结的真实 token。

没有合法私钥，离线假服务无法生成能通过真实固定锚验证的新签名。重放任意旧 token 也不能匹配
本次动态 core、nonce、`genTime` 和 deadline。直接把
`offline_trust_path_valid` 写成 true、跳过签名或注入测试根，都是绕过冻结信任边界。

## 3. 独立审计结论

三路只读审计分别从合同要求、验证器实现和可复用材料三个角度复核，结论一致：

- `FOUNDATIONAL P0`
- `NO-GO`
- 当前冻结协议不能同时满足“离线假 TSA”和“沿真实固定锚完成成功 ASN.1/CMS 验证”
- 不得开始 Stage 2P-1B 实现
- 不得接入真实数据、真实网络、真实 issue 或效果行

现有 `validate_prospective_lifecycle()` 的
`stage2p1b_validator_not_implemented` 关门点不能直接删除。即使补齐其余表、预测、评价和记录
重建，没有可满足冻结信任边界的成功 token，整条链仍不可能验收。

## 4. 科学价值复审

- `scientific_question`：冻结的 Stage 2P-1A 合同能否在不读取真实数据、不联网的条件下完整演练
  Stage 2P-1B 成功生命周期？
- `new_evidence`：三路审计确认没有可签到冻结真实锚的私钥、预签 token 或独立合成信任域。
- `uncertainty_change`：从 `Stage2P-1B ready_not_started` 收敛为
  `Stage2P-1B impossible_under_current_frozen_contract`。
- `science_value_category`：`no_material_progress`。
- `direct_prediction_improvement`：`none`。
- `governance_value`：及时阻止无效实现和虚假合成验收。
- `decision`：`stop_before_implementation_and_reassess_protocol`。
- `next_direct_test`：先冻结一个与 production 密码学隔离的 synthetic-acceptance 信任域，再用
  同一 ASN.1/CMS verifier 演练动态 request、主失败、备成功和五记录全链。
- `stop_condition`：未形成新目标盲协议版本、独立验收、提交、推送、annotated tag 和远端回读
  之前，不得恢复 1B，不得读取真实输入。

这一结论没有回答“近期地震是否提升预测”，也没有产生模型成绩。它只是证明当前实验装置还不能
合法开机。

## 5. 推荐的最小合法修订

推荐另立目标盲协议修订阶段，保留并绝不改写
`v0.2.4-prospective-seismicity-protocol`，在新版本中显式增加两种互斥信任模式：

1. `production`
   - 继续只接受现有 DigiCert/Sectigo URL、policy OID 和精确 DER 锚；
   - 明确拒绝所有 synthetic root、signer、authority ID 和 token；
   - 真实运行仍不得使用测试私钥或测试 fixture。
2. `synthetic_acceptance`
   - 冻结专用测试 root/intermediate/TSA signer 的 DER、SHA-256、policy OID 和 authority ID；
   - 使用明确标注、只供测试的私钥动态签发每个 core/nonce；
   - 演练主站网络失败、主站无效响应、备站成功和两家都失败；
   - 与 production 走同一个 ASN.1/CMS、imprint、nonce、policy、time 和 path verifier；
   - 每份记录、manifest 和 token 都携带不可混淆的 synthetic mode 身份；
   - production validator 必须用负向测试证明会拒绝全部 synthetic token。

新修订还必须：

- trust mode 只能由已核验 protocol tag、release manifest 和受信执行上下文共同固定，禁止由单条
  记录自报或自由选择；
- 为合成 PKI、请求、响应和 manifest 冻结精确生成规则与哈希；
- 证明测试私钥即使公开也只对 synthetic trust domain 有效；
- 维持 `execution_authorized=false`、`real_issue_authorized=false`、
  `real_catalog_read_authorized=false`、`real_network_fetch_authorized=false`；
- 先测试、验收、独立审计、提交、推送、创建新 annotated protocol tag 并远端回读；
- 只有新协议门通过后才重新进入 Stage 2P-1B-a。

不推荐只加入一个 `test_only_skip_signature=true`，也不推荐让 validator 信任记录自报的
`offline_trust_path_valid=true`。这两种做法都不会验证真正的时间戳链。

## 6. 恢复后的最短顺序

1. 经用户确认后，建立目标盲协议修订候选；不读取真实数据或效果。
2. 添加严格隔离的 synthetic trust profile、测试 PKI 和 production 拒绝测试根的负向合同。
3. 运行协议/schema/验证器测试和独立 P0/P1 审计。
4. 验收通过后提交、推送、新建 annotated protocol tag 并远端回读。
5. 重新开始 Stage 2P-1B-a：规范字节、create-only 安装、动态 RFC3161 主备假服务和同路径验证。
6. 每个 1B 子门均测试、验收、提交、推送后再进入下一门。
7. 1B 全链通过后仍只属于 `necessary_enabler`；只有未来真实前瞻 cohort 的唯一锁定检验才可能
   形成 `direct_improvement` 证据。

## 7. 重启续接命令与边界

在 `science_first` 工作树中先执行只读检查：

```powershell
git status --short --branch
git log -5 --oneline --decorate
git diff --check
```

必须继续排除 15 个未跟踪 Stage 4 草稿。不得使用 `git add .`，只允许显式暂存本 NO-GO
交接阶段列出的文件。

在用户确认协议修订方向之前，允许的下一动作仅是只读复核、文档验收、提交和推送本 NO-GO
交接；不允许改动冻结配置/schema 或实现 Stage 2P-1B 代码。
