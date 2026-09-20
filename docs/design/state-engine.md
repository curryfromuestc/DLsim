# 状态引擎

状态引擎在虚拟时钟上重放 AgentX，决定每个时刻每个 worker 执行哪些请求的哪一部分，并维护 KV 在各级存储中的位置。

## 事件循环

单一虚拟时钟和一个事件堆。时间推进到下一个事件的时刻。事件类型：

| 事件 | 含义 |
| --- | --- |
| RequestReady | 某个请求的全部前提已满足且延迟已到 |
| PassComplete | 某个 worker 的一次前向迭代结束 |
| TransferComplete | 一次 KV 传输（层间或 P/D 交付）结束 |
| IdleShift | 全系统无活跃和就绪请求，待发定时器整体前移 |
| EndOfRun | 到达基准时长 |

每个 worker 的每次前向迭代产生一个 PassComplete。迭代在开始时刻确定 batch 并调用 StepLatency，状态在结束时刻提交。事件顺序由时间、类型优先级和序号唯一确定，同一输入和随机种子下结果逐位一致。

## session 依赖图

每条 trace 构造成一张依赖图，节点是请求，边有四种：同一会话内的顺序边、spawn、join、后台分支。边上带延迟，取上一请求结束到本请求开始的录制间隔。

构造规则以 agentx-harness 的 Weka 加载器为准，包括：

`type: "subagent"` 的条目成为子会话；父会话在其之前的那一轮上产生 spawn，在其之后的那一轮上产生 join。之后没有父轮次的 subagent 是后台分支。之前没有父轮次的 subagent 被丢弃。

被展平的并行 agent 由 `hash_ids` 的最长公共前缀证据识别：完全延伸某条链尾部、开始于该尾部区间结束之后、且模型相同的请求是该链的下一轮；只保留尾部前缀的请求，若更长的状态此后不再被访问，视为同一 agent 的上下文压缩，否则是从共享前缀分叉出的新 agent。新链按规则分类为独立 agent、并行 worker 组和辅助调用。这一识别在 subagent 内部嵌套进行。

验收方法是对同一批 trace 比较 DLsim 与 harness 产生的会话数、每个会话的请求序列和依赖边。

## lane 与运行控制

N 个 lane，每个 lane 同时只有一棵 session 树。树的全部节点结束后，该 lane 取下一条 trace。trace 被重复使用时带 cache-bust，使其前缀与先前的实例不共享。

每棵树从采样的起点 t* 开始。起点之前的请求属于 warmup：它们建立缓存状态，不计入指标。

## 调度

调度器是本仓库自己实现的行为模型，建模对 AgentX 两个核心指标有一阶影响的行为：

| 行为 | 参数 |
| --- | --- |
| continuous batching | 每次迭代重新组 batch |
| token 预算 | max_num_batched_tokens、max_num_seqs |
| chunked prefill | chunk 大小；prefill 与 decode 是否可在同一迭代内混合 |
| 准入 | 按 KV 可用容量；容量不足时等待 |
| 抢占 | 容量不足时按策略回收运行中的请求 |
| attention DP | 各 rank 各自组 batch，迭代结束时刻取各 rank 的最大值 |

各框架的差异（例如 prefill 优先且不混合的策略）通过 stack 配置的开关表达，参数取自 InferenceX 的 recipe。

attention DP 下可选用一个代表 rank 加顺序统计量修正来替代完整模拟全部 rank，以减少迭代数；该近似是否采用由验证结果决定。

## KV 缓存

缓存状态按 session 的连续段记录，不逐块记录。依据有两条：trace 的块标识是链式前缀哈希且命名空间限于单个 session 文件；InferenceX 的 recipe 使用按会话复用的策略。一个请求的可命中长度等于它与本 session 已驻留段的最长公共前缀。

每个段记录所在的存储层和最近访问时刻。存储层由 device 与 fabric 配置给出，例如器件本地的高带宽存储、器件本地的大容量存储、host 内存、远端存储。层间搬运的时间为

```
t = alpha_tier + bytes / bandwidth_tier
```

同一链路上的搬运排队。搬运与计算是否重叠由 stack 配置声明。需要的段正在从下层加载时，请求不被准入。

淘汰策略按层配置，默认 LRU，可附加驻留时长上限。驻留时长是一阶参数：间隔超过驻留时长的轮次需要整段重算。

对只有单一大容量主存、没有独立 host 层的器件，存储层退化为一层，命中时没有层间拷贝。

## P/D 分离

prefill 池和 decode 池是两组 worker，共用一个时钟。交付的 token 数可以是完整上下文或目的端缺失的部分，由 stack 配置决定；InferenceX 的 TRT-LLM recipe 中 prefill 侧开启按会话的块复用、decode 侧不复用，对应每轮交付完整上下文（这一点是由配置推断，未经实测验证）。

交付时间为 alpha + bytes / L，经过的链路由 fabric 决定，同一链路上的交付排队，并与 EP 通信共享带宽。P/D 分离与 KV offload 可以同时开启。

alpha 与 L 可由 CollectiveX 的 kv-dsv4 传输实测标定，见 [data-sources.md](data-sources.md)：gb300 上 nixl 经 MNNVL 拉取 524,288 token 的 KV（2,945 MB）p50 为 4.21 ms，约 699 GB/s；经 RDMA 为 39.7 ms，约 74 GB/s；mooncake 经 RDMA 为 62.0 ms。实测的每 token KV 字节数约 5,620 B（fp8），用于核对模型描述中的 bytes_per_token。按此量级，254k 上下文每轮整段交付经 MNNVL 约 2 ms，经 RDMA 约 20 ms。

## 路由

多个 worker 时，请求到 worker 的放置由策略决定：轮转，或 KV 感知。KV 感知策略的代价函数为

```
cost = w_prefill × max(0, 需要 prefill 的块数 − 该 worker 上可复用的块数) + w_decode × 该 worker 的 decode 负载 + w_active × 活跃请求数
```

选择代价最小的 worker。session 换 worker 时，已驻留的段需要经链路迁移或重算。

## 输出

逐请求记录：到达、准入、首 token、各 token、结束的时刻，命中各存储层的 token 数，所在 worker，交付耗时。由这些记录计算 [workload-and-metrics.md](workload-and-metrics.md) 定义的全部指标和按资源的时间占比。
