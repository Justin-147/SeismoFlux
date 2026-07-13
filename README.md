# SeismoFlux

融合动态异常、地震活动背景和活动断层先验的时空地震活动预测与有限面积危险区研究系统。

> Anomaly-informed spatiotemporal seismicity forecasting under alarm-area constraints.

SeismoFlux 研究条件地震活动强度和受面积约束的关注区域，不提供确定性地震预报，也不把相对强度、模型评分或颜色解释为绝对发震概率。

## 当前状态

项目严格按 [`SEISMOFLUX_IMPLEMENTATION_HANDOFF.md`](SEISMOFLUX_IMPLEMENTATION_HANDOFF.md) 从零实施。阶段 0 已闭环；阶段 1 数据契约版本为 `0.1.0`，本地数据接入与质量审计门控已通过：11 个标准化数据集重复运行的文件、内容和模式哈希全部一致，未来信息泄漏违规为 0，当前 106 项测试通过。

阶段 1 的 PR、远端 CI、合并和 `v0.1.0-data-contract` 标签仍待发布闭环；完成前不会进入阶段 2。完整证据与待办状态见 [`docs/phase1_acceptance.md`](docs/phase1_acceptance.md)。

## 本地环境

要求 64 位 Python 3.11 和 `uv` 0.8.6：

```powershell
uv sync --locked --all-groups
uv run seismoflux --help
```

生成只读原始输入的确定性清单：

```powershell
uv run seismoflux inventory --config configs/base.yaml --dry-run
uv run seismoflux inventory --config configs/base.yaml
```

执行阶段 1 数据接入与复核：

```powershell
uv run seismoflux ingest --config configs/base.yaml --dry-run
uv run seismoflux ingest --config configs/base.yaml
uv run seismoflux validate-data --config configs/base.yaml
```

执行本地验收：

```powershell
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
uv build
```

## 阶段 1 数据契约

- [`data/manifests/data_catalog.json`](data/manifests/data_catalog.json) 记录 11 个标准化数据集的行数、模式、路径和确定性哈希。
- [`docs/data_quality_report.md`](docs/data_quality_report.md) 是独立生成的质量报告；机器可读版本位于 [`data/manifests/data_quality_report.json`](data/manifests/data_quality_report.json)。
- 研究区冻结为 `PlotBD/CN-border-L1.gmt` 中按等面积最大、再按段号最小选出的连续大陆闭合环，排除海南、台湾及其他离岛；边界点按区内处理。
- 原始目录缺少可靠时区时统一采用固定 `UTC+08:00` 假设，同时保留原始字段并标记该假设。
- 标准化 Parquet 和研究区派生几何只在本地生成，不提交 Git；仓库只跟踪契约、清单元数据、配置、代码、文档和小型质量摘要。

## 数据与许可

原始输入保留在仓库外。用户已授权公开本仓库中的代码、配置、文档及原始文件清单元数据；该授权不等于原始数据或标准化行级衍生数据的再分发许可。详细边界见 [`docs/data_licenses.md`](docs/data_licenses.md)。

## 科学协议

预登记问题、假设、切分、指标、门控和停止条件见 [`docs/research_protocol.md`](docs/research_protocol.md)。锁定测试只允许正式运行一次。
