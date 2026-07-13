# 阶段 0 验收记录

- 阶段：0（仓库和科学契约）
- 状态：本地、远端 CI 与 `main` 保护验收通过；PR 合并待完成
- 标签解释：按首轮执行清单 §40，把第一个标签 `v0.1.0-data-contract` 留到阶段 1；阶段 0 以提交固定基线，不另冒充数据契约标签

## 验收范围

- Python 3.11 独立环境、`pyproject.toml` 与锁文件；
- 推荐目录骨架、最小 CLI、配置、测试和双平台 CI；
- 研究协议、数据源文件级 SHA-256 清单及许可证说明；
- 公开 GitHub 仓库和 `main` 分支保护；
- 不包含 LocationPred 旧代码、原始数据、标准化数据、模型或结果。

## 本地验收证据

| 项目 | 结果 |
| --- | --- |
| Python | 3.11.5，64 位 |
| `uv` | 0.8.6 |
| 依赖锁 | `uv lock --check` 通过；`uv.lock` SHA-256 `dd3cde3a700b7e1f778b9f9a146b6c61ac0e6eb779a82e6e6bc091f6412b6e6d` |
| 代码质量 | `ruff check`、`ruff format --check`、严格 `mypy` 全部通过 |
| 测试 | Windows 上 49 项全部通过 |
| 构建 | sdist 与 wheel 构建成功 |
| 干净安装 | 新建独立环境安装 wheel，`seismoflux --version` 与 `inventory --dry-run` 通过 |
| 原始输入清单 | 7 类、216 个原始文件、43,610,994 字节；旧生成图 `PlotBD/All_abn.png` 已排除 |
| 清单完整性 | 216 个 SHA-256 均合法，路径均为源内相对路径，无原始数据入库 |
| 清单可复现性 | 连续运行两次字节完全一致，SHA-256 `69d8c0126d7aebbecfa253b175eda57c8a530de102095d9fd0d85dbc33e5d93c` |
| 阶段边界 | 只有 `inventory` 可实际执行；阶段 1–10 命令只允许无副作用 dry-run，正常执行明确失败 |

构建产物不提交 Git。sdist 与 wheel 均以最终工作树重新构建，并在独立环境完成 wheel 冒烟验收。

## 阶段 1 待审计事项

- 异常周报候选文件名显示 2024 年第 36 期和 2025 年第 44 期缺失；阶段 0 不解析或补齐，阶段 1 在质量报告中正式确认。
- 原始候选目录未设置 Windows 只读属性；阶段 0 代码只以二进制只读方式打开并在哈希前后核验文件稳定性，阶段 1 仍需固化受控只读流程。

## 远端闭环

- 阶段实现提交：`908facf`（`build: establish stage 0 scientific contract`）。
- 草稿 PR：[`#1 Stage 0: establish repository and scientific contract`](https://github.com/Justin-147/SeismoFlux/pull/1)。
- GitHub Actions：[`quality (ubuntu-latest)` 与 `quality (windows-latest)`](https://github.com/Justin-147/SeismoFlux/pull/1/checks) 均通过。
- `main` 分支保护回读：强制 PR、严格同步、Ubuntu/Windows 必需检查、管理员约束、线性历史、会话解决以及禁止强推/删除均已启用。
- PR 合并及 `main` 远端提交回读：待本次记录通过最终 CI 后执行。
