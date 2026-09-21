# 单步时延

本页定义 StepLatency：给定一个 worker 的并行配置和本次迭代中按 attention DP rank 分组的逐请求 (new_tokens, past_kv_len) 列表，返回该次前向迭代的时延及其按资源的分解。

## 模型到算子图

每个模型由一个 C++ 描述给出每层的算子列表，分 prefill 路径和 decode 路径。每个算子只按一层的 shape 查询一次，再乘以层数。第一阶段只实现 DeepSeek-V4-Pro。

DeepSeek-V4-Pro 的一层包含：

| 部分 | 算子 |
| --- | --- |
| attention | 按层的 compress_ratio 分组的模块级算子：ratio 为 4 的 CSA（含 indexer 与 top-k），ratio 为 128 的 HCA；纯滑窗层并入后者 |
| 残差路径 | mHC 的 pre 与 post |
| MoE | router GEMM、共享专家 GEMM、dispatch、专家计算（MoE 或 MegaMoE）、combine |
| 首尾 | embedding、logits GEMM |

算子清单以 AISimulate 的 `aisimulate_core/sdk/models/deepseek_v4.py` 为参考逐项核对，对应的实测表为 dsv4_csa、dsv4_hca、mhc_module、gemm、moe、wideep_moe、moe_a2a。attention 使用模块级实测表，不拆成单个 kernel。

每新增一个模型需要新写一份描述。

## 并行

并行约束为 tp × attention_dp × cp = moe_tp × moe_ep。TP 和 EP 体现为算子 shape 的变化和追加的通信算子：

| 配置 | 追加的通信 |
| --- | --- |
| TP > 1 | 每层两次 all-reduce（attention 后、MoE 后），消息为 tokens × hidden。查 custom_allreduce 表时 vllm 与 sglang 按 graph_replay 选 graph 或 eager lane；版本链只向旧版本回退，若当前链没有该卡数（gb300 的 vllm 0.25.0 链只到 4 卡），改查 next 槽位的同一 kernel 实测，再不命中才用闭式公式 |
| attention DP > 1 且 moe_ep = 1 | MoE 前 all-gather、MoE 后 reduce-scatter |
| moe_ep > 1（宽 EP） | dispatch 与 combine 的 all-to-all，node_num = ceil(ep × tp / 每节点器件数)；moe_ep = 1 时 MoE 按 TP 切分，没有 all-to-all |
| PP | 级间 P2P，次数为 pp − 1 |

MoE 的负载不均衡是实测表的一个键（balanced、power_law_1.01、power_law_1.2），DeepSeek 系列取 power_law_1.01。

## 通信时延

```
t_comm = alpha × 消息数 + 字节数 / L
```

alpha 和 L 取决于该通信落在 scale-up 域内还是域外，规则见 [fabric-and-heterogeneity.md](fabric-and-heterogeneity.md)。每个通信算子先查实测表，命中就用表值；表不覆盖时才用闭式公式，结果标注 closed-form。实测覆盖范围：节点内用 AISimulate 的 nccl、custom_allreduce、moe_a2a 表（最多 8 卡，gb300 上 4 卡，全部是单节点），EP8 与 EP16 用 CollectiveX 的 dispatch 与 combine 实测（含 2 节点跨 RDMA 与 4 节点 MNNVL）。EP32 及以上和跨机柜没有实测，只有闭式公式，结果标注为外推。覆盖范围的细节见 [operator-latency.md](operator-latency.md)。

decode 阶段的 EP all-to-all 每步有上百次小消息，每消息延迟对步时延的影响大于峰值带宽；prefill 和 KV 交付受带宽限制。因此 alpha 和 L 必须分开给出。

## 组合规则

算子沿关键路径组合。串行的算子时延相加；可重叠的分支取各分支时延之和的最大值。第一阶段建模的重叠只有 decode 阶段 routed MoE 与共享专家之间的重叠。通信与计算的一般性重叠由 stack 配置中的开关声明，默认不重叠。

每步另加固定开销 t_step_fixed，来自 stack 配置，表示 kernel launch、host 调度、采样和整步 graph 重放是否可用带来的差别。算子实测表是在 CUDA Graph 重放下采集的，不含这部分。每个请求另有流水线固定时延 request_overhead_ms，同样来自 stack 配置，由状态层加在请求到达与进入 prefill 队列之间，表示路由、P/D 交接和传输建立；它进入 TTFT 与 E2E，不进入单步时延。

## 逐请求计价

mixed batch 分三部分计价：

每 rank 的 token 数为该 rank 的 prefill 新 token 数加 decode 请求数乘 (nextn + 1)。routed MoE 与 dispatch、combine 按全部 rank 合计的 token 数查询，因为路由后的专家计算汇聚了 attention DP 各 rank 的 token；mHC、共享专家、router、embedding、logits 在 attention DP 下每 rank 只处理本 rank 的 token，按各 rank 的最大值查询。attention 算子按 rank 分别计价，取各 rank 的最大值；decode 的 generation attention 每 rank 只查一次表，batch 为该 rank 的序列数、step 为它们上下文长度的平均值，因为实测表在固定 batch 下对 kv 长度近似线性、另有一项每 batch 的常数，按上下文分桶多次查表会把这项常数重复计入（gb300 dynamo-trt 并发 388 上曾使 ITL 高估约 2 倍）。

prefill attention 逐请求计价。请求 i 的新 token 数为 n_i、已缓存前缀为 p_i 时，直接以 (batch, isl = n_i, step = p_i) 查询模块表，第一个模型的所有 attention 表都有 step 这一前缀轴。前缀超出实测范围时走 [operator-latency.md](operator-latency.md) 的序列轴外推规则。(full_s² − p_i²) / full_s² 只是稠密注意力的闭式代价结构，不用于 CSA 与 HCA：它们的增量代价对 n_i 线性，在 n_i 远小于 full_s 时该比例约为线性比例的 2 倍。

decode attention 逐请求计价，使用各请求自己的 past_kv_len。为控制查询次数，past_kv_len 按对数分桶，同桶请求合并为一次带 batch 的查询，代表值取桶内 past_kv_len 的均值，因为 decode attention 对 kv_len 近似线性。

这与 AISimulate 用 batch 的平均 ISL 和平均上下文长度的做法不同。AgentX 的上下文长度跨度为 64k 到 900k，逐请求计价是必需的。

## 投机解码

MTP 由 stack 配置给出草稿长度 nextn 和接受长度的分布。GEMM、MoE 与通信按 token 数计价，decode 的 token 数为请求数乘 (nextn + 1)；attention 按序列数查表，因为一个序列的 nextn + 1 个校验 token 共用一次 KV 读取，而实测表没有查询长度这一维（按 token 数查表会把 sglang 草稿长度 6 的 decode attention 高估约 7 倍）；草稿层的代价按 generation 算子的 (nextn + L)/L 倍计入；每个请求每步前进 1 + accepted 个 token。接受长度必须由配置给出，DLsim 不内置接受率。

## 内存与 KV 容量

权重字节数由算子的权重 shape 和并行方式得到。每 token 的 KV 字节数由模型描述给出（DeepSeek 系列的压缩 KV 不按 TP 切分）。激活和通信库保留量来自 stack 与 device 配置。某一级存储可用于 KV 的 token 容量为

```
kv_tokens = floor( (capacity × 可用比例 − weights − activations − reserved) / bytes_per_token )
```

## 输出的分解

除总时延外，返回本次迭代在算力受限、存储带宽受限、通信和固定开销四部分上的时间，供状态层累计成按资源的时间占比。
