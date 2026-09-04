# S3-A八路计算调整与续接

2026-09-04，用户明确授权“调整8路”。本文件只调整执行方式，不改变科研方案。

## 目的与范围

原程序按两个外折顺序计算，参数增加也只能同时执行两折。现在把尚未完成的独立复本分配给
8个计算进程，由一个轻量协调进程登记进度。每个复本的5个时限仍顺序处理，空间置乱场只生成
一次并跨时限共享；种子不含PID、任务顺序或工作进程编号。小幅、局部收益的采纳原则不变。

原 `configs/multitask_s3_anomaly.yaml`、`null_runner.py` 及其冻结依赖不修改；所有数据、两A折、
时限、震级、面积、模型、200+200数量不变。A/C关闭的审计与锁定测试不运行，不取新目标。

只新增4个薄适配模块：`null_parallel`协调、`null_parallel_inputs`只读加载、
`null_parallel_context`共享只读数组、`null_parallel_worker`原复本计算。原预测identity保留，
新执行身份另存 `parallel_execution_identity`，每次启动追加 `execution_history`，记录旧清单SHA、
保留块数、新实现SHA与提交。以后续跑既校验旧科学身份，也校验新执行身份，不放宽旧哈希核验。

## 切换与恢复

1. 新实现合成核对通过并白名单提交推送后，核对旧PID命令行，停止本轮旧计算；其它任务不动。
2. 确认协调进程及其全部计算子进程已退出，备份切换瞬间清单；核对旧成果数量与SHA。
3. 仅移除已退出进程留下的锁。原NPZ不删不覆盖；完整孤儿文件依原身份复用，并标记来源为旧文件。
4. 以下入口只能续接已有原试验；默认8路，`--workers`允许1—8以便用户机器需要时降载：

```powershell
python -m seismoflux.multitask_s3.null_parallel --project-root D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata --data-root D:\AIPred\SeismoFlux\data --prepared-dir D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata\outputs\multitask_s3\s3a_prepared_v1 --reference-prediction-dir D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata\outputs\multitask_s3\s3a_fit_v1 --output-dir D:\AIPred\SeismoFlux\data\interim\worktrees\p2r_multitask_multidata\outputs\multitask_s3\s3a_null_v1 --workers 8
```

仍使用根目录 `.venv\Scripts\python.exe`，PYTHONPATH指向本工作树src；OMP/OPENBLAS/MKL/
NUMEXPR/VECLIB/BLIS均为1，PyArrow CPU与IO线程为1，Hidden、BelowNormal。新日志使用独立时间戳。
协调器先核验并加载一次原输入，再将大数组保存为本机只读映射文件，供8进程共享；不引入新依赖。
每个时限NPZ仍原子落盘，整复本完成时由主进程登记。如果中途停止，完整但未登记的NPZ按旧入口复用。
输入/文件/内存错误属于中断，不算科学失败；数值失败按原规则登记，不换种子重抽。每个数值失败
即时原子保存`taskkey.failure.json`，含原身份、任务、种子及错误；即使同复本后续时限中断，
恢复也会核验并复用该失败，不会将已失败任务当未做任务重算。收据与成功NPZ冲突时拒绝继续。

## 监控与交接

清单仍是 `outputs/multitask_s3/s3a_null_v1/null_prediction_manifest.json`。`active_pid`是协调器，
`worker_pids`是计算进程；Windows spawn子进程命令行不一定带SeismoFlux模块名，须结合PID、父PID、
启动时间与完整命令行核对。1+8进程属于一个试验实例。CPU报告应汇总该树，内存映射共享页不能
简单将工作集相加当独占内存，同时看整机可用内存。保留至少2物理核心；目前本机24物理/48逻辑核。

`loading_parallel_frozen_inputs`时尚未启用8个计算进程，不冒称已8路计算；看到8个worker和新增
预测块后才确认切换成功。新原子NPZ先于复本清单登记不等于重复实例。全部4000块终态前不评分。

## 有限验收与科学价值

合成核对包括：4000任务唯一分配、完成/失败均跳过、部分复本与孤儿恢复、空间同场、固定种子、
NA和数值/资源错误分流、原顺序与新worker的time/space预测数组及模型元数据完全相同、Windows
实际8个spawn进程读取共享数组、只读数组及新执行身份变化拒绝恢复。它们是计算正确性核对，
不是新的地震预测成绩。实际切换水位、PID和资源以最新重启交接顶部为准。

本次合成联合核验113项通过，Ruff与格式检查通过。独立复核指出的“早时限数值失败、后时限
中断导致失败记录遗失”已用原子失败收据补齐并回归通过；无其它实质切换阻碍。

实际执行验收（2026-09-04 10:27）：新协调器PID28084及8个计算子进程均实际活跃；
10:25:16原2013块切换点后已新增252块，达到2265/4000（56.625%）、失败0，stderr为空。
进程树CPU约17.44%，整机可用内存22.96GiB，均为当时抽样。8路切换可结束实施，继续原试验，
不再扩展性能优化；最终科学对照尚未完成。新代码提交ae8be18，启动交接提交ae1dc19均已推送。
切换闭合复核确认原2013个completed条目缺失0、改变0，科学身份未变、新4模块执行身份与前序
清单SHA一致；所有完成/失败/运行任务属于原注册集合。没有重扫原NPZ或读取任何效果分数。

此次必要使能仅为加快原科学检验，完成切换后不继续扩展调度系统或性能工程。原试验最终仍需
统一比较、解释增益与损失，形成S3-A复审后按原有限S3-B推进。
