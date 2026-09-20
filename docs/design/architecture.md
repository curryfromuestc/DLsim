# 架构

## 分层

```
AgentX trace
    |
[状态层]        N 个常驻 session, 虚拟时钟, 调度, KV 缓存层级, P/D 分离, 路由
    |           每次迭代给出逐请求的 (new_tokens, past_kv_len), 不取均值
[单步时延层]    模型 -> 每层算子列表 -> 算子时延与通信时延沿关键路径组合
    |
[算子时延层]    有实测的器件: 查表与插值
    |           无实测的器件: 跨器件缩放分解, 带不确定度
[配置]          device / fabric / mapping / stack
```

反向求解和参数 sweep 位于状态层之上，把整条正向路径当作被调用的函数。

每一层是独立、可替换、可单独验证的单元。层间接口是窄接口：

| 接口 | 调用方 | 输入 | 输出 |
| --- | --- | --- | --- |
| OpLatencySource | 单步时延层 | 器件、算子类别、dtype、shape | 时延、来源标注、不确定度 |
| StepLatency | 状态层 | worker 的并行配置，按 attention DP rank 分组的逐请求 (new_tokens, past_kv_len) 列表 | 本次迭代的时延及其按资源的分解 |
| Simulation | 反向求解、sweep | 四类配置与 trace | InferenceX 行与按资源的时间占比 |

OpLatencySource 有三种实现：实测表、跨器件缩放分解、闭式 roofline。以后自定义器件的周期级性能模型输出以第一种实现的数据格式接入。

## 四类配置对象

| 对象 | 内容 |
| --- | --- |
| device | 各精度算力、各级存储的容量与带宽、功耗、成本、算子时延数据的引用。允许同时定义多种器件 |
| fabric | scale-up 域的规模、每器件带宽、每消息延迟；scale-out 的每器件带宽、每消息延迟；机柜边界；host 存储层及其链路 |
| mapping | 模型的哪个阶段或哪一部分运行在哪个器件组上，组内并行方式（TP、EP、DP attention、PP），worker 数，路由策略 |
| stack | 软件栈能力：特性开关（MTP、P/D 分离、宽 EP、KV offload、整步 graph 重放、chunked prefill、前缀缓存策略）和每步固定开销 |

stack 与 device 分开，是为了支持三种对比：假设软件栈相同的纯硬件对比，按当前实际软件栈的对比，逐项开关特性的消融。

配置文件用 YAML。器件规格的字段与 AISimulate 的 `systems/*.yaml` 保持可对应，并扩展多级存储与互联。

## 两级精度

第一级是闭式估算，用于筛选：

```
每个驻留 session 的平均需求: 算力 c, 存储带宽 b, 存储容量 m, 互联 l
单个器件组可承载的 session 数 N = min( C*eta/c , B/b , (M - weights)/m , L/l )
```

第二级是完整的事件驱动重放，只对筛选后的候选运行。

## 语言与依赖

C++20，CMake 构建。

| 库 | 用途 | 使用位置 |
| --- | --- | --- |
| simdjson | 读 trace JSONL 和 InferenceX 快照 | 核心 |
| yaml-cpp | 读配置、器件规格、InferenceX recipe | 核心 |
| Eigen | 最小二乘拟合 | 核心 |
| Arrow/Parquet C++ | 读 AISimulate 的 parquet 算子表 | 仅导入工具 |

Arrow 是重依赖，隔离在导入工具中：

```
dlsim-import   parquet -> 本仓库自有的二进制算子表, 记录来源 commit 与文件
dlsim          读二进制算子表、trace 缓存和 YAML 配置
```

绘图不在本仓库实现。输出为 JSON 或 CSV。

## 运行时间预算

一次 AgentX 运行是 N 个常驻 session 持续 1800 s。按 GB300、并发 1152、40 GPU 的公开点推算，一次运行约 6e4 个全局迭代；若每个 DP rank 单独组 batch，约 7e5 个 rank 迭代。目标是单次运行在秒级完成，整条曲线的各点用线程并行。

为达到这一预算采用三个建模选择，详见 [state-engine.md](state-engine.md)：前缀缓存按 session 的连续段记录而不是逐块记录；算子时延按分桶的 shape 缓存；DP rank 可用一个代表 rank 加顺序统计量修正。

trace 首次解析后缓存为二进制格式。

## 目录规划

目录随实现逐步创建，不预先建立空目录。

```
CMakeLists.txt
src/trace        Weka 加载, 二进制缓存, session 依赖图
src/perfdata     算子表, 插值, 跨器件缩放分解
src/model        模型到算子图
src/step         单步时延
src/engine       事件循环, 调度, KV 缓存, P/D, 路由
src/fabric       互联
src/metrics      InferenceX 行输出
src/solver       反向求解与 sweep
src/cli          命令行入口
tools/import     依赖 Arrow 的导入工具
configs          device / fabric / mapping / stack
tests
docs/design
third_party      参考仓库 (submodule)
traces           下载的数据 (不进版本库)
```
