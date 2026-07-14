# SeismoFlux 阶段2背景基线报告

本报告由已验证的不可变注册表确定性渲染。所有数值术语仅表示条件强度, 相对强度和信息增益, 不作绝对风险解释。

## 科学身份

- 协议版本: `0.2.0`
- 协议指纹: `f386f0d6abd5b7ca0e31e073ce0f74da812fb561052639d45227e8f339ff9032`
- 代码提交: `509057592e2abe0474c75e43b5e8bbb7cac87f53`

## 冻结科学摘要

- 科学结果状态: 科学门控失败
- 失败阶段: `completeness`
- 失败代码: `estimate_above_frozen_maximum`
- 失败原因: spatial/500km/r+7/c-3: completeness estimate 4.5 exceeds frozen maximum candidate 4.0
- 失败证据: `4e14abc393dd74fa7da6610355f0f8887ce33aed37a4c057c10a6f5789d36081`
- 最终选择的 Mc: 未定义
- 空间 KDE 带宽: 未定义 km
- 代表日条件强度: 未生成 (科学门控失败)

### 固定快照评分

| 模型 | 快照 | 状态 | 目标事件数 | 信息增益 (nats/event) | Score ID |
|---|---|---|---:|---:|---|
| uniform_poisson | fold_1 | 未运行 | — | 未定义 | — |
| uniform_poisson | fold_2 | 未运行 | — | 未定义 | — |
| uniform_poisson | fold_3 | 未运行 | — | 未定义 | — |
| uniform_poisson | fold_4 | 未运行 | — | 未定义 | — |
| uniform_poisson | final_validation | 未运行 | — | 未定义 | — |
| spatial_poisson | fold_1 | 未运行 | — | 未定义 | — |
| spatial_poisson | fold_2 | 未运行 | — | 未定义 | — |
| spatial_poisson | fold_3 | 未运行 | — | 未定义 | — |
| spatial_poisson | fold_4 | 未运行 | — | 未定义 | — |
| spatial_poisson | final_validation | 未运行 | — | 未定义 | — |
| etas | fold_1 | 未运行 | — | 未定义 | — |
| etas | fold_2 | 未运行 | — | 未定义 | — |
| etas | fold_3 | 未运行 | — | 未定义 | — |
| etas | fold_4 | 未运行 | — | 未定义 | — |
| etas | final_validation | 未运行 | — | 未定义 | — |

### 验证段 bootstrap

| 模型 | 状态 | 点估计 | 下限 | 上限 | 重采样次数 | 置信水平 | 跳过原因 |
|---|---|---:|---:|---:|---:|---:|---|
| spatial_poisson | 跳过 | 未定义 | 未定义 | 未定义 | — | 未定义 | spatial/500km/r+7/c-3: completeness estimate 4.5 exceeds frozen maximum candidate 4.0 |
| etas | 跳过 | 未定义 | 未定义 | 未定义 | — | 未定义 | spatial/500km/r+7/c-3: completeness estimate 4.5 exceeds frozen maximum candidate 4.0 |

### 延迟与时窗覆盖

| 模型 | 状态 | 比较数 | 跳过原因 |
|---|---|---:|---|
| spatial_poisson | 跳过 | 0 | spatial/500km/r+7/c-3: completeness estimate 4.5 exceeds frozen maximum candidate 4.0 |
| etas | 跳过 | 0 | spatial/500km/r+7/c-3: completeness estimate 4.5 exceeds frozen maximum candidate 4.0 |

### 未来集合覆盖

- 状态: 跳过
- 起报日数量: 0
- 跳过原因: spatial/500km/r+7/c-3: completeness estimate 4.5 exceeds frozen maximum candidate 4.0

## 不可变产物

| 类型 | Artifact ID | Manifest SHA-256 |
|---|---|---|
| processed | `969a8e1f3de2278a` | `6b1942434b82b96c8b13021309a5fd9c361fc72a259da134c4316d48a9f15812` |
| model | `66eccb7719a85486` | `0b018eebb84f6875ece2219d2b08e2b2eb0e4b6de230ec28913507e254ce01b6` |
| backtest | `26ed112e452110b9` | `b360f352826d48aa7eb1f747054517238aac294befc8fbc2456b30070509d0e9` |
| experiment | `65c7bc81fdcc1f64` | `bdc4fe75fa556db6bd0fa55eb255c5f46bdcf973ffbad9f553d4b0dfc01b6bad` |

## 正式模型尝试

| 模型 | 快照 | 状态 | 失败原因数 | 门控数 | Score ID 数 |
|---|---|---|---:|---:|---:|
| uniform_poisson | fold_1 | 未运行 | 1 | 1 | 0 |
| uniform_poisson | fold_2 | 未运行 | 1 | 1 | 0 |
| uniform_poisson | fold_3 | 未运行 | 1 | 1 | 0 |
| uniform_poisson | fold_4 | 未运行 | 1 | 1 | 0 |
| uniform_poisson | final_validation | 未运行 | 1 | 1 | 0 |
| spatial_poisson | fold_1 | 未运行 | 1 | 1 | 0 |
| spatial_poisson | fold_2 | 未运行 | 1 | 1 | 0 |
| spatial_poisson | fold_3 | 未运行 | 1 | 1 | 0 |
| spatial_poisson | fold_4 | 未运行 | 1 | 1 | 0 |
| spatial_poisson | final_validation | 未运行 | 1 | 1 | 0 |
| etas | fold_1 | 未运行 | 1 | 1 | 0 |
| etas | fold_2 | 未运行 | 1 | 1 | 0 |
| etas | fold_3 | 未运行 | 1 | 1 | 0 |
| etas | fold_4 | 未运行 | 1 | 1 | 0 |
| etas | final_validation | 未运行 | 1 | 1 | 0 |

## G1 与选择结论

- G1: 未评估 (上游科学门控失败)
- G1 通过模型: 无
- 模型选择状态: 未评估 (上游科学门控失败)
- 验证段最佳模型: 无
- 最终选择模型: 无
- 阶段3: 停止
