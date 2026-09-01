# S0 空间外推经典方法复审（2026-09-01）

## 1. 复审原因

S0初稿曾把“39个线划区质心固定做5折、训练区再退让100 km”直接写成空间外推合同。这个口径虽不读
目标成绩，但`5折`和`100 km`都不是CSEP或空间交叉验证文献规定的统一标准，而且现有L1/L2线划来源
权威性尚未核实。按“先调研国内外经典方法、再决定实施”的新硬门，本项目在任何模型成绩出现前撤回
这项未经充分论证的冻结，并完成如下复审。

## 2. 国内外经典方法回答的是不同问题

### 2.1 CSEP：主问题是固定区域内的未来预测

CSEP的核心不是随机或空间留区训练，而是事先固定测试区域、格网、震级范围、时间窗、目录和指标，
再评价未来或伪前瞻地震。中国CSEP2.0同样强调统一空间范围和预测格式，以避免地震发生后再主观挑区。
因此，SeismoFlux的主证据应继续是全国固定支持域上的时间向前评价；空间留区不能替代它。

主要依据：

- [Zhang et al. (2024), The Collaboratory for the Study of Earthquake Predictability in China: Experiment Design and Preliminary Results of CSEP2.0](https://doi.org/10.1111/1755-6724.15250)
- [“地震可预测性国际合作研究”——1.0阶段工作理念及成果（2021）](https://dizhen.ief.ac.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=1504)
- [Schorlemmer et al. (2007), Earthquake Likelihood Model Testing](https://doi.org/10.1785/gssrl.78.1.17)
- [Zechar et al. (2010), Likelihood-Based Tests for Evaluating Space–Rate–Magnitude Earthquake Forecasts](https://doi.org/10.1785/0120090192)
- [Tsuruoka et al. (2012), CSEP Testing Center and the First Results of the Earthquake Forecast Testing Experiment in Japan](https://doi.org/10.5047/eps.2012.06.007)
- [pyCSEP评价](https://docs.cseptesting.org/concepts/evaluations.html)与[预定义区域](https://docs.cseptesting.org/concepts/regions.html)

### 2.2 空间阻塞/留区：次问题是能否迁移到新地区

空间交叉验证用于存在空间自相关的数据，估计模型向未参与拟合地区迁移时的性能。经典文献允许使用
已有多边形作原子块，也允许坐标分块或坐标k-means；块数不必等于折数，折数、缓冲和预测距离都应
匹配具体外推问题。它可能比已知地区内预测更苛刻，因此只能称迁移压力测试。

主要依据：

- [Roberts et al. (2017), Cross-validation Strategies for Data with Temporal, Spatial, Hierarchical, or Phylogenetic Structure](https://doi.org/10.1111/ecog.02881)
- [Valavi et al. (2019), blockCV](https://doi.org/10.1111/2041-210X.13107)
- [Schratz et al. (2019), Spatial Hyperparameter Tuning and Performance Assessment](https://doi.org/10.1016/j.ecolmodel.2019.06.002)
- [Brenning (2023), Spatial Machine-Learning Model Diagnostics](https://doi.org/10.1080/13658816.2022.2131789)
- [Wadoux et al. (2021), Spatial Cross-Validation Is Not the Right Way to Evaluate Map Accuracy](https://doi.org/10.1016/j.ecolmodel.2021.109692)

大震稀疏时还必须报告检验功效，不能把某篇模拟研究的事件数直接抄成普适阈值：

- [Khawaja et al. (2023), Statistical Power of Spatial Earthquake Forecast Tests](https://doi.org/10.1093/gji/ggad030)

## 3. SeismoFlux冻结决定

### 3.1 主科学轨道

- 在固定中国大陆支持域上按时间向未来检验；
- 区域、格网、震级档、时长、报警面积和指标均在评分前冻结；
- 主要概率评分使用完整目录，不机械删除余震；
- 并列报告全部事件、固定锚点episode、episode平衡事件和后续事件。

### 3.2 次级空间迁移轨道

- 使用现有25 km查询格已有的39个目标盲原子块，逐块留出；
- 现阶段只称“构造线划原子块”，来源核实前不得称“官方构造区”；
- 每个固定格恰好获得一个未用本块目标拟合的OOF分数，39份结果拼回一张全国OOF面；
- 每个全国报警面积档只对拼接后的全国OOF面施加一次，不能给每个小块各用一份全国预算；
- 块内历史输入仍按起报时刻因果可见；这检验参数迁移，不假装现实中看不到当地既往地震；
- 0 km为主缓冲，75/200 km只作本项目既有局地/异常尺度的严格敏感性，并报告实际训练—测试距离；
- M6+不得用于选择或平衡空间折；允许单块零事件，池化后确认并诚实报告不确定性，不得事后并块。

冻结输入身份：

- 空间manifest文件SHA-256：`283a6790f6e7c16bc31d9498b2cc3cd043e19c8f141046afb898e988f25dcc83`；
- L1/L2线划SHA-256：`30d7fabbea95040fed596a37dfd07970c6e7699187c27ad064471db25ef5d5cd` / `189b81655411225ad3d7a1860829835ad23b843c239134529bbad9f2d8d98c33`；
- zone集合SHA-256：`294e070ca6c0ff68d74d6638ab8ec3cda08d3469b99a3958f4d21afa4fae4a62`；
- 本地受限25 km cell→zone文件SHA-256：`171a500de9f9dd475f2c37a5426debc7c6f2d34ddd418056729c39b27118108e`。

受限几何、坐标和逐格映射不得写入公开产物；公开账本只记录身份、计数和科学状态。

### 3.3 模型选择用的较少空间折

S0不再为了凑齐“5折”立即生成任意映射。39→较少折的几何分组延至S1，但必须在任何模型成绩出现
前完成并封存：

1. 采用冻结等面积投影下的目标盲坐标块/坐标k-means++经典方案；
2. 候选`k=5/4/3`，只使用冻结几何、最早评价期以前可见的设计信息和目标盲背景功效模拟；
3. 选择达到预登记M5–6功效的最大k；M6+不得参与选择；
4. 固定种子147、`n_init=100`，发布精确zone→fold表及哈希；
5. 禁止使用评价期M5+/M6+震中、模型分数、命中或漏报；
6. 若k=3仍证据不足，模型选择只走时间折，空间结果降为描述性。

## 4. 科学价值判断

这项纠偏不会直接提高模型分数，但防止把任意空间折和缓冲误写成国际规范，也防止用未来大震位置平衡
测试折。它属于必要科学设计：主证据保持与实际全国未来预测一致，空间压力测试则单独回答跨区迁移，
两者互不冒充。下一项直接科学工作仍是完成S0样本/空间水位，再进入S1完整目录基线比较。
