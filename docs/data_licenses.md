# 数据来源、授权与再分发说明

核验日期：2026-07-13。

用户已明确确认并授权公开 SeismoFlux 仓库，包括其中的代码、配置、文档及原始文件清单元数据。该授权允许公开仓库内这些材料，但不构成原始数据、标准化行级衍生数据或研究区派生几何的再分发许可，也不自动授予第三方开源软件许可证。

阶段 1 对仓库外原始输入进行了只读解析和质量审计。当前仍未发现足以证明下列数据可公开再分发的许可证或授权文本，因此统一采用最保守策略：**授权状态未知，仅限已有授权范围内的内部研究使用，默认禁止再分发。**

| 数据源 ID | 候选来源 | 内容 | 授权证据 | 当前允许用途 | 原始数据入库/再分发 |
| --- | --- | --- | --- | --- | --- |
| `anomaly_tables` | `LocationPred/anomaly/` | 动态异常周报表 | 未提供 | 内部研究、可追溯清点与解析 | 禁止 |
| `earthquake_catalog_m3_plus` | `LocationPred/china3new.eqt` | 1970 年以来 M3+ 目录 | 未提供 | 内部研究、可追溯清点与解析 | 禁止 |
| `earthquake_catalog_m5_plus` | `LocationPred/1900年以来5级以上地震目录_simple.xlsx` | 1900 年以来 M5+ 目录 | 未提供 | 内部研究、可追溯清点与解析 | 禁止 |
| `fault_coordinates` | `LocationPred/FaultCord.xlsx` | 简化活动断层坐标 | 未提供 | 内部研究、可追溯清点与解析 | 禁止 |
| `fault_attributes` | `LocationPred/FaultAttri_ALLV4_修改.xlsx` | 活动断层属性 | 未提供 | 内部研究、可追溯清点与解析 | 禁止 |
| `long_term_hazard` | `LocationPred/weight_analysis-修改newV4_修改.xlsx` | 长期危险性结果 | 未提供 | 内部研究、可追溯清点与解析 | 禁止 |
| `plotbd_basemap_and_fault_traces` | `LocationPred/PlotBD/` | 底图与真实断层迹线 | 未提供 | 内部研究、可追溯清点与解析 | 禁止 |

## Git 公开边界

允许提交并公开：

- 代码、配置、文档和数据契约；
- `data/manifests/source_inventory.csv` 中的数据源 ID、源内相对路径、字节数、UTC 修改时间、文件类型、SHA-256 和授权状态；
- `data_catalog.json`、质量报告等不含原始行级内容的小型验收摘要。

禁止提交或公开再分发：

- 七类原始输入及其内容副本；
- 阶段 1 生成的 11 个标准化 Parquet 数据集；
- 从受限原始几何生成的研究区 GeoJSON 或其他行级、点级、线级衍生数据。

绝对候选路径只出现在 `configs/data_sources.yaml` 迁移配置中；若设置 `SEISMOFLUX_SOURCE_ROOT`，环境变量明确优先于配置默认值。PlotBD 清单只纳入 `.gmt` 与 `.dat` 原始几何，明确排除 `All_abn.png` 等旧生成图。标准化数据只写入被 Git 忽略的本地处理目录，`data_catalog.json` 中的 `standardized_data_committed_to_git` 必须保持为 `false`。

在获得权利人、来源机构、许可证版本、地域/用途限制及衍生数据再分发条款的书面证据前，不得改变上述数据状态。公开可见不等于获得复制、修改或再分发许可；项目软件若要授予开源许可证，仍须单独决定并记录。
