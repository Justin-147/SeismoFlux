# 阶段 4A 目标盲可执行性最终修订

- 日期：2026-07-30
- 当前协议版本：`0.4.4`
- 当前协议/代码/结果标签：`v0.3.4-kde-anomaly-increment-{protocol,code,result}`
- 当前协议标签：`v0.3.4-kde-anomaly-increment-protocol`
- 直接替代：`0.4.3` / `v0.3.3-kde-anomaly-increment-*`
- 开发目标读取：0
- 独立验证目标读取：0
- 锁定测试读取：0
- 状态：仅冻结输入适配和日期解释；尚不授权打开真实输入

## 1. 为什么只做这一次小修订

对废弃的一次性 runner 做独立审计后发现，它没有把时间或空间置乱真正接入
“重建特征—重新拟合—重新评分”链，只是在固定玩具统计量上移动标记，因此不能回答异常是否真的
提高预测效果。该 runner 已停止使用，没有打开真实目标，也没有消耗唯一开发 attempt。

随后完成的目标盲预检确认：

1. 阶段 3 状态表和特征表的冻结 schema 可以从代码与字典独立重建，所得 SHA-256 与公开 registry
   完全一致；
2. `rebuild_time_placebo_features` 和 `rebuild_space_placebo_features` 会真正重建置乱后的当期与
   轨迹特征；
3. 当前缺口只剩两个：四个受限空间工件的固定路径/schema 没有在当前协议中直接声明；开发
   exposure ID 到起报时刻、窗口和 issue ID 的解释规则没有冻结为当前 adapter 合同。

因此 `0.4.4` 只补这两个接口，不改科学问题、模型候选、滚动折、目标、面积、阈值、置乱次数、
Bootstrap、随机根种子或通过门。`0.4.3` 的审计证据保持不可变，但在任何真实输入或目标读取前转为
`historical_superseded_before_target_read`，不再提供执行授权。

本次是异常路线最后一次目标盲基础合同修订。若随后的一次纯合成端到端验收仍发现新的基础性 P0
矛盾，立即停止异常增量实现，保留 75 km KDE，不再堆叠 seal、receipt、runner 或更复杂模型。

## 2. 四个受限空间工件

下面四条路径由仓库根目录与固定相对路径拼接，调用者不得覆盖。它们只在代码标签远端回读通过后
打开；打开时先核验 byte count 和整文件 SHA-256，再解析。当前修订没有检查这些文件是否存在，
也没有读取其字节。

| ID | 固定相对路径 | byte count | SHA-256 |
| --- | --- | ---: | --- |
| `cell_mapping` | `data/interim/stage4/anomaly_increment_r2/construction_zone_cell_mapping.parquet` | 42,917 | `171a500de9f9dd475f2c37a5426debc7c6f2d34ddd418056729c39b27118108e` |
| `entity_mapping` | `data/interim/stage4/anomaly_increment_r2/construction_zone_entity_mapping.parquet` | 4,055,922 | `49cd56ace13680c3465b0c128f7dd9823636f6f1db7a2f39a12d1235df532170` |
| `connectors` | `data/interim/stage4/anomaly_increment_r2/construction_zone_connectors.json` | 11,768 | `1f25120d9b9b15ec428efe97183179cebe1c3c5b0e022294dbf82f4c73e4e167` |
| `zone_geometry` | `data/interim/stage4/anomaly_increment_r2/construction_zones.parquet` | 233,171 | `c1c54d390bd1553c8f75b10def4898e24deb919345dd9ca0a11a02d0ff80ba70` |

这些身份必须与已认证的
`data/manifests/anomaly_increment_r2_spatial_strata.json` 的 `local_artifacts` 四项逐项一致。
不得从旧 R2 配置动态寻找目录，也不得调用旧 R2 orchestrator 重建拓扑。

### 2.1 `cell_mapping` 精确 schema

字段顺序和 Arrow 类型固定为：

```text
grid_id:string
cell_id:string
cell_row:int64
cell_column:int64
query_x_m:float64
query_y_m:float64
construction_zone_id:string
```

schema metadata 精确为：

```text
seismoflux_contract=0.4.1-local-construction-zone-cell-mapping
seismoflux_license=unknown_no_redistribution
seismoflux_publication=forbidden_contains_coordinates_and_per_cell_mapping
```

行数必须为 15,697，全部字段非空，`cell_id` 唯一，并与已认证阶段 3 表中的
`grid_id/cell_id/cell_row/cell_column/query_x_m/query_y_m` 按行相等。

### 2.2 `entity_mapping` 精确 schema

```text
state_id:string
anomaly_id:string
issue_time_utc:timestamp[us,UTC]
construction_stratum_id:string
coordinate_pair_sha256:string
outside_study_area:bool
```

schema metadata 精确为：

```text
seismoflux_contract=0.4.1-local-construction-zone-entity-mapping
seismoflux_license=unknown_no_redistribution
seismoflux_publication=forbidden_contains_per-entity-stratification
```

行数必须为 165,841，`state_id` 唯一；它必须且只覆盖阶段 3 snapshot 中全部
`spatial_eligible=true` 的 state，并逐项核验 `anomaly_id` 与 `issue_time_utc`。

### 2.3 `zone_geometry` 精确 schema

```text
construction_zone_id:string
geometry_wkb_equal_area_m:binary
```

schema metadata 精确为：

```text
seismoflux_contract=0.4.1-local-construction-zone-geometry
seismoflux_license=unknown_no_redistribution
seismoflux_publication=forbidden_contains_restricted_geometry
```

固定 65 行，按 `construction_zone_id` 升序，字段非空。adapter 只核验 ID、schema、SHA 和跨工件
引用，不解码、不返回、也不把几何当作特征或候选生成输入。

### 2.4 `connectors` 精确 JSON

顶层键必须且只能是：

```text
connector_count
connectors
coordinate_crs
license_status
maximum_connector_distance_m
protocol_version
publication
schema_version
```

`connector_count=40`，`coordinate_crs` 精确为
`+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs +type=crs`，
`license_status=unknown_no_redistribution`，
`maximum_connector_distance_m=100000.0`，`protocol_version=0.4.1`，
`publication=forbidden_contains_restricted_coordinates`，`schema_version=1`。
每个 connector 的键必须且只能是：

```text
coordinates_equal_area_m
endpoint_index
length_m
source_line_sha256
target_kind
```

坐标必须是两个有限二维点；`endpoint_index` 只能为 0 或 1；`length_m` 必须位于
`(0,100000]`；source hash 必须为小写 SHA-256；`target_kind` 只能是
`study_area_boundary` 或 `other_construction_line`。该文件也只作身份和关系核验，不进入模型。

## 3. exposure 与 issue 的唯一解释

adapter 只能用 ASCII `fullmatch` 接受：

```text
\Adevelopment-h(?P<horizon>007|030|090)-(?P<local_date>[0-9]{4}-[0-9]{2}-[0-9]{2})\Z
\Aanomaly-issue-(?P<local_date>[0-9]{4}-[0-9]{2}-[0-9]{2})\Z
```

日期经 `date.fromisoformat` 解析后必须能原样 `isoformat()` 回环。拒绝前后空白、Unicode 数字、
错误大小写、非法日期，以及任何 `validation`、`formal` 或未知 horizon ID。

对日期 `D`：

1. `T_local = D 00:00:00 Asia/Shanghai`，必须 `fold=0` 且 UTC offset 精确为 `+08:00`；
2. `T_utc = T_local.astimezone(UTC)`，例如 `2020-01-01` 对应
   `2019-12-31T16:00:00Z`；
3. exposure `development-hHHH-D` 只映射到 `anomaly-issue-D`；
4. 目标窗固定为 `(T_utc, T_utc + h days]`。

三个 rolling fold 必须恰为 1、2、3。训练 exposure 只允许 `h007`；每折训练窗终点必须与
manifest 声明相等，且严格早于 assessment band 起点。assessment map 的键必须恰为
`7/30/90`，ID 内 horizon 必须与键一致，窗口必须完整落在该折 band 内；跨折 assessment ID 不得
重复。

时间置乱的 fit/assessment issue pool 必须分别唯一、严格递增、互不相交，且全部 fit issue
早于全部 assessment issue。每个 exposure 映射的 issue 必须存在于相应 pool；两个 pool 的并集
必须恰好覆盖传给时间重建函数的 issue tables。任何日期、时区、重复、horizon、pool、fold 或
窗口不一致都失败关闭。

## 4. 最小直接代码边界

`0.4.4` 继续只直接调用现有的
`rebuild_time_placebo_features/rebuild_space_placebo_features`，并只把以下已哈希封印的
目标盲底层符号提升为直接调用：

- `features.anomaly.state.states_from_records`；
- `features.anomaly.snapshot.build_issue_snapshots`；
- `features.anomaly.grid.Stage3QueryGrid`。

现有 `feature_adapter.py` 和 `grid_features.py` 会经 import closure 带入旧 qualification、runner 与
目标相关模块，因此不获得直接执行授权。新的薄 `kde_dev_fit.py` 只能从已认证 feature manifest
读取 9/17/22 组冻结列名，按原序从已认证 Arrow tables 拼接 value/mask design，再调用已开放的
preprocessing/model 原语；不得导入上述两个旧模块。其纯合成验收必须逐列与 feature manifest 和
独立直接拼接结果相等。

`Stage3QueryGrid` 只能由已经认证的阶段 3 表实例化并逐字段核验；不得调用
`build_stage3_query_grid` 从研究区重新生成格网。正式 adapter 只能返回：

```text
issue_tables
snapshots_by_issue_id
query_grid
construction_stratum_by_state_id
```

其参数和返回值不得包含地震目录、目标、event label、score、模型成绩、connector 坐标或 zone
geometry。`connectors` 与 `zone_geometry` 只核验不返回。

当前禁用清单全部不变，尤其禁止旧 `scoring_pipeline.py`、`formal_preflight.py`、
`formal_run.py`、`formal_production.py`、`placebo_source.py`、`placebo_runtime.py`、
`local_support_runtime.build_local_support_runtime`、旧 placebo DTO/builders、
`preregistration.Stage4SeedContext`、旧 G2/G3 gate、旧随机流、带宽选择和拓扑重建。

## 5. 代码标签前唯一允许的纯合成验收

代码阶段必须先用内存合成数据证明：

1. 新 Stage 4A DTO 直接进入两个 rebuild，不构造旧 DTO；
2. observed、time placebo、space placebo 都走同一个
   `rebuild → 9/17/22 特征组装 → 仅 fit 行拟合预处理器 → 同一 ridge 重新拟合 →
   冻结 assessment prediction → target-free statistic` 函数路径；
3. identity mapping 精确复现 observed；非恒等 mapping 必须改变实际重建表、design 和 statistic，
   不能只改变 marker；
4. 时间置乱保留 recipient coverage 并从 pseudo-history 重建 trajectory；空间置乱保留每个
   issue/stratum 的坐标多重集、coverage 与非坐标属性，并重算 200 km snapshot/trajectory；
5. 9/17/22 个逻辑特征的列名、顺序、source/design 列计数与冻结 feature manifest 完全一致；
6. 改变 assessment 数值不能改变只在 fit 行拟合的 preprocessor SHA；
7. 独立 direct replay 与 adapter 输出在严格容差内一致；
8. 合成四工件的 SHA、Arrow metadata、字段顺序、JSON 额外键和跨工件 zone 引用任一被篡改时
   都失败关闭；
9. fixture、adapter 和 statistic 不得出现真实数据路径、真实目标/score 列或旧 R2 orchestrator；
10. worker 固定为 1，OMP/MKL/OPENBLAS 每库 1 线程。

这一步的 statistic 只能是 target-free 的数值链路检查，不能伪装成信息增益、召回、预测成绩或
科学通过。

## 6. 不变的科学合同

`B0/C0/B1/B2`、75 km KDE、三个滚动折、7/30/90 天、M5–6、600,000 平方公里固定面积、
2,000 次事件块 Bootstrap、1,000 次时间置乱、1,000 次空间置乱、候选族 `maxT`、跨折/跨区、
固定面积至少 `+5 pp` 且同时区间排除无改善、单一开发 attempt、独立验证与锁定测试禁读均不变。
根种子仍为 147，生成器仍为 PCG64；版本升级只会机械改变包含协议版本的派生上下文，不是换随机
规则或重试成绩。

## 7. 科学价值复审

- `science_value_category`: `necessary_enabler`
- `direct_prediction_improvement`: 尚无
- `evidence`: 阶段 3 schema 已独立重建并与 registry 一致；两个置乱重建函数具备真实重建能力；
  本修订只补齐真实重训链必需的固定输入与日期接口。
- `decision`: 仅继续一次纯合成端到端验收与薄代码冻结。
- `next_direct_scientific_test`: 代码标签远端回读通过后，才打开并核验冻结输入，在唯一开发 attempt
  内完成真实 observed/time/space 重新拟合和 600,000 平方公里比较。
- `hard_stop`: 若一次纯合成验收仍发现新的基础性 P0，停止异常路线，保留 KDE；不得再用工程修订、
  深模型或锁定测试寻找阳性结果。

这不是预测效果结论。真正回答“异常有没有用”的证据，只能来自后续唯一开发 attempt 的真实
重训、置乱与固定面积结果。
