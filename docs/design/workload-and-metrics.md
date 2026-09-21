# 负载与指标

## 负载来源

AgentX 是 SemiAnalysis 在 InferenceX 中使用的 agentic 推理基准。trace 是匿名化的 Claude Code 会话，保留了 token 数量、KV 块的前缀共享结构、subagent 扇出和时间关系，不含文本内容。

数据集为 HuggingFace 上的 `semianalysisai/cc-traces-weka-062126`，Apache-2.0，主体是 `traces.jsonl`（1,847,151,435 字节，393 行）。位置与版本见 [data-sources.md](data-sources.md)。

## Weka trace 格式

每行是一个 session：

```
{ id, models[], block_size: 64, hash_id_scope: "local",
  requests: [
    { type: "s" | "n", t, model, in, out, hash_ids[], api_time, ttft?, think_time? },
    { type: "subagent", t, agent_id, subagent_type, status, duration_ms,
      tool_use_count, total_tokens, models[], requests: [ ...同上的请求 ] }
  ] }
```

已核实的字段语义：

| 字段 | 语义 |
| --- | --- |
| `t` | 请求开始时刻，单位秒，以 session 起点为零点。subagent 内部请求的 `t` 也是 session 内的绝对时刻 |
| `api_time` | 原始服务的请求耗时，单位秒。只用于推出“上一请求结束时刻”，不是要复现的目标 |
| `in` | 输入长度，等于哈希块数乘 64，不是 tokenizer 的真实计数。数据集说明平均偏差约 1.00 倍，尾部最多高估约 26 万 token |
| `out` | 输出 token 数 |
| `hash_ids` | 64 token 粒度的 KV 块标识序列。同一文件内共享一个命名空间。标识是链式前缀哈希：我们在样本上检查了 506,225 次引用，同一标识总是出现在同一位置，因此标识相等意味着前缀相同 |
| `type` | `s` 为流式，`n` 为非流式，重放时两者处理相同 |
| `status` | subagent 的结束状态，全量数据里只有 `completed`。加载器不读它：父会话是否等待某组 subagent，只由该组的 `t + duration_ms / 1000` 与父会话后续轮次的 `t` 比较决定 |

`ttft` 和 `think_time` 是原始服务侧和客户端侧的记录值。

## 重放语义

以 agentx-harness 为准，出处见其 `docs/benchmark-modes/semianalysis-agentx-faq.md` 和 `docs/tutorials/weka-trace.md`。DLsim 的状态层必须实现下列语义，结果才可与公开数据比较。

负载由并发数 N 控制，含义是同时存活的 session 树的数量，不是请求到达率。trace 的选取是顺序采样器（Weka 加载器的 preferred sampling strategy 为 SEQUENTIAL，InferenceX 不覆盖）：按数据集行序，lane i 初始取第 i 条；一棵树（根会话及其全部 subagent）全部结束后，该 lane 立即取序列中的下一条 trace 从头重放，序列用完后从第一条循环。运行固定时长，默认 3600 s（InferenceX agentic 矩阵的 DEFAULT_AGENTIC_DURATION_SECONDS），下限 900 s。

轮间延迟取上一轮结束到下一轮开始的间隔，始终生效。两级空闲封顶：一棵树没有在途请求时，若它最早的待发请求距当前超过 300 s（`--trace-idle-gap-cap-seconds 300`），该树的全部待发定时器整体前移，使最早的一个在 300 s 时到达，请求顺序与依赖不变；整个重放没有任何活跃或就绪的请求时，所有待发定时器整体前移，使下一个请求在 10 s 内到达。

每个初始 lane 的树从采样的起点 t* 开始，范围是树内全部请求开始时刻跨度的 0.25 到 0.75（`--trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75`）。抽样按 harness 的规则精确复现：seed 取 sha256("42:{trace id}:{lane}") 前 8 字节的大端整数，t* = lo + u·(hi−lo)，u 是 numpy `default_rng(seed).uniform` 的首个值；SeedSequence 与 PCG64 在 `src/engine/harness_rng.cpp` 中重实现，对照值见 `tests/harness_rng_test.cpp`。warmup 分两段：t* 之前每个 agent 的最后一个请求作为 primer 在 t=0 无依赖发出，建立缓存状态；随后每个 lane 按时间顺序取 t* 之后的 10 个请求（`--warmup-requests-per-lane 10`）作为无延迟的 warmup 续接，依赖关系保持；profiling 从其后的状态继续，指标只统计 profiling 阶段。交接规则与 harness 一致：每个 lane 的下一个请求把它在 trace 里的完整轮间延迟带入 profiling，等待全局 warmup 屏障的时间不消耗该延迟；全部 lane 中最早的一个在 profiling 时刻 0 发出，其余保持相对间隔；此时没有在途请求的树立即套用 300 s 的树空闲封顶。lane 结束后取的新树从头重放，不再抽样 t*。

首轮前缀中插入 cache-bust 标记，使被重复使用的 trace 之间不共享缓存。

`ignore_eos` 为真，输出长度严格按 trace 执行。

subagent 不占并发额度。重叠的 subagent 并发发出；父会话的某一轮可能以一组 subagent 全部完成为前提；父会话之后没有后续轮次的 subagent 作为后台分支运行，父会话不等待它。

录制中被展平的并行 agent 由加载器根据 `hash_ids` 的最长公共前缀证据识别并拆成子会话，分类为独立 agent、并行 worker 组和辅助调用。这是一个确定的算法，DLsim 需要按同样的规则构造 session 依赖图，见 [state-engine.md](state-engine.md)。

理论前缀命中率的定义：每个 trace 文件一个共享的已见块集合，按全局时间顺序处理请求，统计每个请求从头开始连续命中的块数。它是无限容量理想前缀缓存的命中率，用作缓存模型的上界对照。

## 负载特征

下列数字由我们对全量数据的统计得到（数据集自带的 `stats.txt` 给出了同样的计数和 token 总量）。理想命中率按上一节的全局时间顺序定义计算，`dlsim trace-stats` 的输出与此一致。

| 量 | 数值 |
| --- | --- |
| session / 请求 | 393 / 98,827（主 agent 56,798，subagent 组 1,697，组内请求 42,029） |
| 输入 / 输出 token 总量 | 21,635,381,376 / 106,474,498 |
| 主 agent ISL | p50 254,080；p90 651,008；p99 897,672 |
| subagent ISL | p50 64,000；p90 196,326 |
| 主 agent OSL | p50 616；mean 1,374；p90 3,502 |
| 每轮实际新增的 prefill token | p50 1,664；mean 3,792；p99 46,722 |
| 理想前缀命中率 | 0.983088 |
| 实际需计算的 prefill 与输出 token 之比 | 约 3.4 : 1 |
| 每 session 主轮次数 | p50 65；p90 330 |
| 轮间间隔 | p50 4.9 s；p90 131.5 s；p99 3,257 s |
| 间隔超过 300 s / 1800 s 的轮次占比 | 4.5% / 1.4% |
| session 处于等待的时间占比 | 0.964 |
| 时间平均上下文 | mean 207,394 token |
| 带 subagent 的 session | 44.5%，组数中位 4，组内请求中位 16，session 内峰值并发组数中位 3 |

由此得到三条对建模有直接后果的事实。第一，状态的主体是长期驻留而很少被访问的 KV，容量需求远大于瞬时计算需求。第二，KV 驻留时长对 prefill 计算量是一阶影响：间隔超过驻留时长的轮次需要整段重算，1% 的未命中就使 prefill 计算量接近翻倍。第三，上下文长度跨度从 64k 到 900k，用均值代替逐请求长度会造成明显误差。

## 指标定义

InferenceX 对 AgentX 的规范用户侧指标是 E2E normalized interactivity，定义见其 `MODELS.md`：

```
r_i = E2EL_i / OSL_i                       每个请求每输出 token 的端到端耗时
E2E normalized interactivity_q = 1 / percentile_q({ r_i })
```

dashboard 默认取 P90。它近似等于 1 / (TPOT + TTFT/OSL)，即在 decode 速率上加入了排队和 prefill 的惩罚。AgentX 的 OSL 中位只有约 600，TTFT 为数秒，因此 TTFT/OSL 一项与 TPOT 同量级，prefill 和排队对该指标的影响不可忽略。

公开 API 的 `benchmarks` 行里的 `p90_intvty` 是另一个量，等于 1 / P90(ITL)，只反映 decode。两者不能混用。规范指标由 `/api/v1/derived-agentic-metrics` 给出，含 p75 与 p90；VR200 的点没有该值。

DLsim 同时输出这两个量，并逐请求保存 TTFT、ITL、E2EL、OSL，分位数由逐请求数据精确计算。

## 输出格式

每个仿真点输出一行，字段与 InferenceX 的 `benchmarks` 行一致：

```
hardware, framework, model, precision, spec_method, disagg, is_multinode,
prefill_tp, prefill_ep, prefill_dp_attention, prefill_num_workers,
decode_tp,  decode_ep,  decode_dp_attention,  decode_num_workers,
num_prefill_gpu, num_decode_gpu, offload_mode, conc,
metrics: { tput_per_gpu, input_tput_per_gpu, output_tput_per_gpu,
           {mean,median,p90,p99}_{ttft,itl,tpot,e2el,intvty}, kv_cache_pool_tokens, avg_power_w }
```

`tput_per_gpu` 是输入加输出的总 token 吞吐，其中包含命中缓存的输入 token。InferenceX 行中聚合部署的 GPU 数字段按 tp 乘 ep 默认填充，不可靠；GPU 数应按 recipe 的 worker 数乘每 worker 的 GPU 数计算。

DLsim 在此之外附加两组字段：各级缓存的命中 token 数，以及时间在算力、存储带宽、存储容量、scale-up、scale-out 五类受限状态上的占比。
