# S1-C0 评分主暴露接线修复记录（2026-09-02）

## 外行版结论

四折模型预测已经完整写出并封存，但第一次“阅卷”在真正产生分数前发现拿错了答题清单：程序把
5,215个每周日期全部送去阅卷，而预登记的正式样本只有其中396个互不重叠日期。程序因此安全停止，
没有生成任何模型成绩。修复只让阅卷器接收早已冻结的396个日期，不改题目、模型、答案、面积或指标。

## 事实与边界

- 成功预测代码提交：`8a237b6601437631ccb1e594c831bc2188c19226`；已推送并远端回读。
- 旧输出根：`outputs/multitask_s1/s1c0_all_m4_screen_v1`，此后整根只读。
- 旧总封印SHA-256：
  `2605fd3f20174875bfbcddd970af34ad3bca3df4a9dc11c2dadbeaaf8a95ca55`。
- 旧评分授权SHA-256：
  `f2e2f924876c9e543dba7b0e29f1f26818a214fe196bf26da6d44e840760c0db`。
- 旧评分区没有`raw_scores.parquet`或`development_summary.json`，没有可查看或据此调参的成绩。
- 新输出根：`outputs/multitask_s1/s1c0_all_m4_screen_v1_attempt2`；必须从零生成四折预测和新封印。
- holdout、2023+ audit、locked test均未打开；既有Stage4未跟踪草稿未触碰。

## 原因与最小修复

真实输入加载器按合同返回完整5,215条周四起报账本，其中只有
`primary_exposure_selected=true`的396条是预登记的互不重叠正式样本（四折各99条）。预测阶段本来
就只对这396条作答；评分官方入口却把完整账本直接交给严格的正式目标构造器，后者正确地拒绝了
非主暴露行。

修复位于官方评分入口，只按冻结的`primary_exposure_selected`标记筛选后再构造目标。不能改为仅按
`mature`筛选，因为成熟账本仍包含大量相互重叠周窗，会改变样本定义并虚增样本。目标构造器继续
拒绝任何“被选中但未成熟”的异常行，且继续逐折、逐时长核对完整冻结主轴。

## 验证与科学价值

- 混合账本测试确认：成熟主暴露、成熟非主暴露、未成熟非主暴露中只有主暴露进入目标构造；
- 被选中但未成熟仍必须失败；
- 完整S1集合：`129 passed in 53.80s`；Ruff、格式、strict Mypy和`git diff --check`通过；
- 独立只读复核：`GO`，P0/P1=`0/0`。

本修复属于`necessary_enabler`，没有直接提高预测效果。它只保证下一次评分严格对应预登记样本，
防止因接线错误改变实验。下一项真正科学工作是从新根重跑四折预测、核验新总封印、单独授权评分，
然后比较同面积独立震序召回以及时间、震级和联合效果。

```yaml
science_value_category: necessary_enabler
evidence: "首次评分只写授权便安全停止，无raw_scores或summary；最小主暴露筛选修复经129项回归与独立复核通过"
decision: GO_COMMIT_PUSH_THEN_FRESH_ATTEMPT2_PREDICT_AND_SCORE
old_attempt_code_oid: 8a237b6601437631ccb1e594c831bc2188c19226
old_attempt_master_seal_sha256: 2605fd3f20174875bfbcddd970af34ad3bca3df4a9dc11c2dadbeaaf8a95ca55
old_attempt_score_authorization_sha256: f2e2f924876c9e543dba7b0e29f1f26818a214fe196bf26da6d44e840760c0db
old_attempt_has_raw_scores: false
old_attempt_has_development_summary: false
new_attempt_root: outputs/multitask_s1/s1c0_all_m4_screen_v1_attempt2
next_scientific_test: "从零生成四折预测并封存后，首次对冻结396个主暴露评分"
```
