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

agentx-harness 加载器还有三条改变图形状的规则，DLsim 按同样的常量实现（`src/trace/weka_graph.cpp` 顶部）：分类阈值，单请求链若跨模型或首请求输入小于 max(16384, 0.10 × 主链峰值 ISL) 归为辅助调用，同模型且输出小于 4000、输入不小于 16384、输入大于 20 倍输出的链为归约调用，也按辅助调用处理；preamble 规则，最早的请求若与其余请求的 hash 没有公共前缀，且输出不超过 64 或块完全不相交，不单独建链而挂回主链；seam 门槛，间隔超过 3600 s 且重叠低于 0.5 的续接不拼接为同一链。全量数据上这些规则产生 7,004 个辅助调用 agent、707 个独立 agent 和 42 个 worker 组。对照结果：393 条 trace 上 DLsim 与 harness 的会话数、每会话请求序列和依赖边（含延迟）完全一致，对照脚本为 `tools/reference/weka_graph_dump.py` 与 `compare_graph_dump.py`。

验收方法是对同一批 trace 比较 DLsim 与 harness 产生的会话数、每个会话的请求序列和依赖边。

## lane 与运行控制

N 个 lane，每个 lane 同时只有一棵 session 树。树的全部节点结束后，该 lane 按数据集行序取下一条 trace 从头重放。trace 被重复使用时带 cache-bust，使其前缀与先前的实例不共享。

初始的每棵树从采样的起点 t* 开始，抽样规则与 warmup 的两段（primer 与 10 个无延迟续接请求）见 [workload-and-metrics.md](workload-and-metrics.md) 的重放语义；起点之前的请求与 warmup 续接请求都不计入指标。每棵树的空闲间隔封顶 300 s，整个系统的空闲间隔封顶 10 s。

## 调度

调度器是本仓库自己实现的行为模型，建模对 AgentX 两个核心指标有一阶影响的行为：

| 行为 | 参数 |
| --- | --- |
| continuous batching | 每次迭代重新组 batch |
| token 预算 | max_num_batched_tokens、max_num_seqs |
| chunked prefill | chunk 大小；prefill 与 decode 是否可在同一迭代内混合。注意力 DP 上 prefill 不是每次迭代都排：SGLang 的 `--prefill-decode-interval`（B200 为 24，B300 为 20）、vLLM 的 `--prefill-schedule-interval 8`，由点的 `prefill_interval` 表示。有 decode 在跑时，每 N 次迭代才把整份 token 预算给 prefill 一次 |
| 准入 | 按 KV 可用容量；容量不足时等待。请求的 KV 需求（聚合与 decode 侧为 ISL 加输出长度，分离的 prefill 侧为 ISL）超过该 rank 整个池时永远无法准入：这样的请求直接判为失败并计数（oversized_requests），不进入指标，会话继续；validate-points 里带此计数的点记为容量不足，不参与比较 |
| 抢占 | 容量不足时按策略回收运行中的请求 |
| attention DP | 各 rank 各自组 batch，迭代结束时刻取各 rank 的最大值 |

各框架的差异（例如 prefill 优先且不混合的策略）通过 stack 配置的开关表达，参数取自 InferenceX 的 recipe。

请求因容量不足而未能准入时，释放本次查询对前缀缓存的锁定；等待队列不保留尚未使用的缓存。否则排队请求会阻止运行中的请求淘汰缓存，导致抢占后反复重算而无法完成。加载缓存时先保护整条匹配前缀，再检查 HBM 空间，防止腾挪空间时把同一前缀的父节点搬走、增加实际搬回量。加载完成后释放锁定并重新排队，准入时再锁定实际使用的前缀。抢占后重算的输出 KV 长度是已生成 token 数减一，最后一个已输出 token 尚未进入下一次前向计算。

attention DP 下可选用一个代表 rank 加顺序统计量修正来替代完整模拟全部 rank，以减少迭代数；该近似是否采用由验证结果决定。

## KV 缓存

缓存状态按 session 的连续段记录，不逐块记录。依据有两条：trace 的块标识是链式前缀哈希且命名空间限于单个 session 文件；InferenceX 的 recipe 使用按会话复用的策略。一个请求的可命中长度等于它与本 session 已驻留段的最长公共前缀。

每个段记录所在的存储层和最近访问时刻。存储层由 device 与 fabric 配置给出，例如器件本地的高带宽存储、器件本地的大容量存储、host 内存、远端存储。层间搬运的时间为

```
t = alpha_tier + bytes / bandwidth_tier
```

同一链路上的搬运排队。搬运与计算是否重叠由 stack 配置声明。需要的段正在从下层加载时，请求不被准入。前缀完整命中时最后一块重算，命中只到它的前一块；被重算的那一块所在的段即使已在下层也不加载，只有命中链上位于下层的段才触发加载。

淘汰策略按层配置，默认 LRU，可附加驻留时长上限。驻留时长是一阶参数：间隔超过驻留时长的轮次需要整段重算。

对只有单一大容量主存、没有独立 host 层的器件，存储层退化为一层，命中时没有层间拷贝。

## P/D 分离

prefill 池和 decode 池是两组 worker，共用一个时钟。请求先在 prefill 池计算，交付 KV 后进入 decode rank 的等待队列，decode 每 rank 的序列上限为 stack 的 max_num_seqs（recipe 里 decode 侧每 rank 的 max_batch_size）；每个 pass 先装入正在 decode 的序列，剩余名额才按到达顺序准入等待队列里的请求，已准入的序列不会被新请求挤出。这是 Dynamo 的顺序：router 先发远端 prefill，再把结果路由给 decode worker。客户端由 decode worker 收到流式响应，因此分离部署里首 token 的时间戳打在 decode 准入时刻（prefill 内完成的单 token 请求除外），prefill 之后等待 decode 槽位的时间计入 TTFT 而不是 ITL。gb300 dynamo-trt 并发 388 的点上 decode 每 rank 上限 4、共 128 槽位，实测 TTFT 中位 7.6 s 正是这一排队。交付的 token 数可以是完整上下文或目的端缺失的部分，由 stack 配置决定；InferenceX 的 TRT-LLM recipe 中 prefill 侧开启按会话的块复用、decode 侧不复用，对应每轮交付完整上下文（这一点是由配置推断，未经实测验证）。

交付时间为 alpha + bytes / L，经过的链路由 fabric 决定，同一链路上的交付排队，并与 EP 通信共享带宽。P/D 分离与 KV offload 可以同时开启。

alpha 与 L 可由 CollectiveX 的 kv-dsv4 传输实测标定，见 [data-sources.md](data-sources.md)：gb300 上 nixl 经 MNNVL 拉取 524,288 token 的 KV（2,945 MB）p50 为 4.21 ms，约 699 GB/s；经 RDMA 为 39.7 ms，约 74 GB/s；mooncake 经 RDMA 为 62.0 ms。实测的每 token KV 字节数约 5,620 B（fp8），用于核对模型描述中的 bytes_per_token。按此量级，254k 上下文每轮整段交付经 MNNVL 约 2 ms，经 RDMA 约 20 ms。

## 路由

多个 worker 时，请求到 worker 的放置由策略决定：轮转，或 KV 感知。KV 感知策略的代价函数为

```
cost = w_prefill × max(0, 需要 prefill 的块数 − 该 worker 上可复用的块数) + w_decode × 该 worker 的 decode 负载 + w_active × 活跃请求数
```

选择代价最小的 worker。该池关闭了 block reuse 时（InferenceX 的 trtllm decode 池），worker 上没有前缀索引，dynamo-router 靠自己的路由历史和 session 亲和判断局部性：同一棵树的请求固定在它先前请求所在的 worker；新树放到当前存活树最少的 worker，相同时轮转。worker 内有多个 attention DP rank 时，KV 感知策略再选持有最长可复用前缀的 rank，相同时选活跃请求最少的 rank；轮转策略只选活跃请求最少的 rank。session 换 worker 或换 rank 时，已驻留的段重算，不做迁移。InferenceX 的部署都带前缀感知的路由（gb300 上是 dynamo-router），因此 `configs/points` 里的点全部使用 KV 感知策略。

## 输出

逐请求记录：到达、准入、首 token、各 token、结束的时刻，命中各存储层的 token 数，所在 worker，交付耗时。由这些记录计算 [workload-and-metrics.md](workload-and-metrics.md) 定义的全部指标和按资源的时间占比。
