# 数据来源、授权与再分发说明

核验日期：2026-07-13。

本项目阶段 0 只对仓库外的候选原始输入做字节级清点与 SHA-256，不解析内容、不复制旧代码，也不把原始文件提交到 Git。当前未发现足以证明下列数据可公开再分发的许可证或授权文本，因此统一采用最保守策略：**授权状态未知，仅限已有授权范围内的内部研究使用，默认禁止再分发。**

| 数据源 ID | 候选来源 | 内容 | 授权证据 | 当前允许用途 | 原始数据入库/再分发 |
| --- | --- | --- | --- | --- | --- |
| `anomaly_tables` | `LocationPred/anomaly/` | 动态异常周报表 | 未提供 | 内部研究、可追溯清点 | 禁止 |
| `earthquake_catalog_m3_plus` | `LocationPred/china3new.eqt` | 1970 年以来 M3+ 目录 | 未提供 | 内部研究、可追溯清点 | 禁止 |
| `earthquake_catalog_m5_plus` | `LocationPred/1900年以来5级以上地震目录_simple.xlsx` | 1900 年以来 M5+ 目录 | 未提供 | 内部研究、可追溯清点 | 禁止 |
| `fault_coordinates` | `LocationPred/FaultCord.xlsx` | 简化活动断层坐标 | 未提供 | 内部研究、可追溯清点 | 禁止 |
| `fault_attributes` | `LocationPred/FaultAttri_ALLV4_修改.xlsx` | 活动断层属性 | 未提供 | 内部研究、可追溯清点 | 禁止 |
| `long_term_hazard` | `LocationPred/weight_analysis-修改newV4_修改.xlsx` | 长期危险性结果 | 未提供 | 内部研究、可追溯清点 | 禁止 |
| `plotbd_basemap_and_fault_traces` | `LocationPred/PlotBD/` | 底图与真实断层迹线 | 未提供 | 内部研究、可追溯清点 | 禁止 |

绝对候选路径只出现在 `configs/data_sources.yaml` 迁移配置中；若设置 `SEISMOFLUX_SOURCE_ROOT`，环境变量明确优先于配置默认值。被跟踪的 `data/manifests/source_inventory.csv` 只包含数据源 ID、源内相对路径、字节数、UTC 修改时间、文件类型、SHA-256 和授权状态，不泄露原始内容。PlotBD 只清点 `.gmt` 与 `.dat` 原始几何，明确排除 `All_abn.png` 等旧生成图。

在获得权利人、来源机构、许可证版本、地域/用途限制及衍生数据再分发条款的书面证据前，不得改变上述状态。项目软件自身的开源许可证也尚未授予；创建公开发行版前必须单独决定并记录。
