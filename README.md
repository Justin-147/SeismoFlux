# SeismoFlux

融合动态异常、地震活动背景和活动断层先验的时空地震活动预测与有限面积危险区研究系统。

> Anomaly-informed spatiotemporal seismicity forecasting under alarm-area constraints.

SeismoFlux 研究条件地震活动强度和受面积约束的关注区域，不提供确定性地震预报，也不把相对强度、模型评分或颜色解释为绝对发震概率。

## 当前状态

项目正按 [`SEISMOFLUX_IMPLEMENTATION_HANDOFF.md`](SEISMOFLUX_IMPLEMENTATION_HANDOFF.md) 从零实施。阶段 0 只建立仓库、科学契约、数据源字节清单、最小命令行、测试和持续集成；不解析原始数据，也不实现模型。

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

执行阶段 0 验收：

```powershell
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
uv build
```

## 数据与许可

原始输入保留在仓库外，Git 只跟踪配置、文件级 SHA-256 清单、代码、文档和小型验收摘要。数据授权状态和再分发限制见 [`docs/data_licenses.md`](docs/data_licenses.md)。未确认授权的数据不得再分发。

## 科学协议

预登记问题、假设、切分、指标、门控和停止条件见 [`docs/research_protocol.md`](docs/research_protocol.md)。锁定测试只允许正式运行一次。
