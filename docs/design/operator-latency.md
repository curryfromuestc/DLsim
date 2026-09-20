# 算子时延

本页定义 OpLatencySource：给定器件、算子类别、dtype 和 shape，返回时延、来源标注和不确定度。

## 数据来源

实测数据来自 AISimulate 仓库中的算子库，位置与版本见 [data-sources.md](data-sources.md)。它由 NVIDIA 在真实 GPU 上逐算子采集：独占节点并锁定频率，预热 3 次，默认用 CUDA Graph 捕获后重放计时，按推理框架及其版本分别采集。因此表中的时延是纯 kernel 时间，不含 kernel launch 和 host 调度开销；这类开销在 [step-latency.md](step-latency.md) 的每步固定开销中单独建模。

有数据的系统：b200_sxm、b300_sxm、gb200、gb300、h100_sxm、h200_sxm、l40s、rtx_pro_6000_server、a100_sxm、b60。没有 Vera Rubin 的数据。

时延单位是毫秒。主要表和键：

| 表 | 键 |
| --- | --- |
| gemm | gemm_dtype, m（token 数）, n, k |
| context / generation attention | dtype, kv_cache_dtype, num_heads, num_kv_heads, head_dim, window_size, batch_size, isl, step（过去的 KV 长度） |
| mla、dsa、msa 及其 module 表 | 在 attention 键上增加 model, architecture, gemm_type, tp_size |
| dsv4_csa、dsv4_hca 的 context / generation module | 再增加 compress_ratio |
| moe | moe_dtype, num_tokens, hidden_size, inter_size, topk, num_experts, moe_tp_size, moe_ep_size, distribution |
| moe_a2a | comm_backend, phase, comm_dtype, ep_size, node_num, hidden_size, topk, num_experts, num_tokens；值含 transmit_us, notify_us, latency |
| nccl、custom_allreduce | dtype, op_name, num_gpus, message_size |
| mhc_module、mla_bmm、quantize | 见源文件 |

`kernel_source`、framework 和 version 是数据的一部分，必须保留：同一 shape 在不同框架版本下由不同 kernel 实现。

通信表的覆盖范围（commit 108cb5d）：moe_a2a 的 node_num 列只有 1，ep_size 在 b200、b300、h100、h200 上为 8，在 gb200、gb300 上为 4；nccl 的 num_gpus 最大为 8，gb300 上为 4。该数据集里没有任何跨节点的集合通信实测。跨节点的实测来自 InferenceX 的 CollectiveX，见 [data-sources.md](data-sources.md)：DeepSeek-V4-Pro shape 的 EP dispatch 与 combine，EP8 与 EP16，其中 gb200、gb300 的 EP16 是 NVL72 域内 4 节点乘 4 卡经 MNNVL，b200、b300、h100、h200、mi355x 的 EP16 是 2 节点乘 8 卡经 RDMA。没有 EP32 及以上，没有跨机柜。

attention 表的上下文覆盖因框架而异：TRT-LLM 1.3.0rc23 与 vLLM 0.24.0、0.25.0 的 dsv4_csa 与 dsv4_hca 表，context 覆盖 isl + step ≤ 65,536，generation 覆盖 step ≤ 65,536；SGLang 0.5.14 的同名表覆盖到 1,048,576。AgentX 主 agent 的 ISL 中位为 254,080，因此前两个框架的表对大多数请求不够用，处理规则见实现一。

## 本仓库的算子表

导入工具把 parquet 转为本仓库自有的二进制表，保留全部键列、`kernel_source`、framework、version，并记录来源仓库的 commit、源文件路径和导入时间。以后自定义器件的性能模型按同一 schema 输出。

## 实现一：实测表查询

精确命中返回实测值。范围内的查询做插值：

GEMM 以 (n, k) 为站点、m 为曲线轴。已知站点在自身的 m 曲线上线性插值；未知站点取 log2 距离内最近的若干站点，以利用率加权转移。

网格型表逐轴线性插值。context 类表的序列轴在 sqrt(latency) 空间插值，以线性化约 s 平方的曲率。

这些规则与 AISimulate 的做法一致，它公布的留一法中位误差可作为我们实现的对照：GEMM 约 3.6% 到 4.9%，context attention 约 2.0%，DSA 约 5.4%；越界时 p90 误差升到 50% 到 110%。

超出实测范围的查询分两种处理。attention 类算子沿序列轴外推：取同 batch 下实测范围边缘的点 edge，按该算子类别的闭式代价结构缩放，t(query) = t(edge) × SOL(query) / SOL(edge)，来源标注为外推并附上 query 与 edge 的长度比。代价结构由模型描述给出：generation attention 对 kv_len 线性；CSA context 的 indexer 对总上下文线性、top-k 部分为常数；HCA 对 kv_len/128 线性。这条规则在用于 TRT-LLM 与 vLLM 的表之前，先用 SGLang 的表验证，方法见 [validation.md](validation.md) 步骤 2。其他算子超出实测范围时不返回点估计，转入实现二，并标注为外推。

## 实现二：跨器件缩放分解

目标：为没有实测数据的器件 B 估计算子时延。

单台器件上的 roofline 拟合

```
t = max( F / (C * eta_c) , M / (B * eta_b) ) + t0
```

不足以完成这件事，原因有四个。第一，某一段数据如果全部落在同一个分支，另一个分支的系数没有被约束，三个参数无法从单台器件的数据中唯一确定。第二，max 假设计算与访存完全重叠，真实 kernel 含串行阶段。第三，名义的 F 和 M 不等于实际的计算量和数据流量。第四，模块级算子（DSV4 的 CSA、HCA，含 indexer、top-k 和通信）是复合操作，无法分解为一个 F 和一个 M。

因此改用跨器件的差异来辨识。同一张表在多台算力 C 和存储带宽 Bw 各不相同的器件上都有实测。对每个 (算子类别, kernel 族, dtype, shape)，把时延分解为三部分：

```
t_d(shape) = a(shape) / C_d + b(shape) / Bw_d + c(shape)        d 为器件
```

a 是随算力缩放的部分，b 是随存储带宽缩放的部分，c 是不随二者缩放的部分。三者由多台器件上的实测值联合求解，非负约束。这个分解不需要知道算子的 F 和 M，因此对复合模块同样适用。目标器件 B 的时延为 a/C_B + b/Bw_B + c。

求解结果附带三样信息：参与拟合的器件集合、残差、由留一器件验证得到的误差区间。

适用条件：目标器件与参考器件执行同一 kernel 族，数据布局、并行方式和缓存状态相同，shape 落在参考数据的覆盖范围内。条件不满足时返回上下界或 unsupported。上界取各参考器件中按最不利资源缩放的结果，下界取闭式 roofline。

已知的失效情形，需要在结果中标注：

| 情形 | 后果 |
| --- | --- |
| 跨架构时 kernel 选择、分块方式、autotune 结果不同 | a、b、c 不可转移 |
| 低精度 kernel 的成熟度不同 | 不能把 BF16 的系数用于 FP4。GB300 上同一 GEMM 的实测表明 FP4 的标称算力是 BF16 的 6 倍，大 m 下的实测加速约 3.0 倍 |
| 参考器件之间 C 与 Bw 高度相关 | a 与 b 不可区分，需要检查设计矩阵的条件数，条件数过大时按下面的区间规则输出 |
| 目标器件的存储层次与参考器件不同（例如没有 HBM） | 有效带宽取决于命中的存储层，需要在 device 配置中给出各层带宽并由调用方指定访问落在哪一层 |

Bw 一列秩亏时的区间规则：只拟合可辨识的部分 t = a/C + k，其中 k = b/Bw_ref + c，Bw_ref 为参考器件的带宽。目标器件的结果按 k 的两种极端分配给出区间，不给点估计：

```
t_low  = a/C_t + k                        k 全部是 c
t_high = a/C_t + k × Bw_ref / Bw_t        k 全部是 b/Bw_ref
```

这一情形在第一个模型上确定发生。有 nvfp4 表的器件是 gb300、b300_sxm、b200_sxm、gb200 和 rtx_pro_6000_server，前四个的存储带宽在 7.7 到 8.0 TB/s 之间，带宽跨度只由 rtx_pro_6000_server 提供（1.79 TB/s GDDR7，sm120，另一个 kernel 族）。FP4 的 GEMM 与 MoE 对带宽不在 8 TB/s 量级的目标器件只输出区间。attention 模块不受影响，dsv4 的 csa 与 hca 表在 h100、h200 和四个 Blackwell 器件上都有，带宽从 3.35 到 8 TB/s。

## 实现三：闭式 roofline

用于第一级筛选和下界：

| 算子 | 计算量 | 读写字节数 |
| --- | --- | --- |
| GEMM | 2mnk | 元素字节 × (mn + mk + nk) |
| context attention | 2·b·(full_s² − prefix²)·n·h | Q/K/V 与 KV cache 的读写 |
| decode attention | 2·b·n·h·2·kv_len | 约 b·2·n_kv·kv_len·h·kv 字节 |
| MoE | T·K·h·inter·g·2 / ep / tp | 激活与被命中专家的权重 |

时延为 max(计算量 / 算力, 字节数 / 存储带宽)。

表中 context attention 一行是稠密注意力的代价结构。CSA 与 HCA 的代价结构见实现一中的序列轴外推规则，不用 full_s² − prefix²。

## 已观察到的事实

GB300、TRT-LLM、BF16、权重 51200×8192 的 GEMM：m ≤ 64 时延约 0.137 ms 基本不变；转折点在 m 约 256 到 512，与 2500 TFLOPS / 8 TB/s ≈ 312 FLOP/B 一致；大 m 时相对标称算力的比值约 0.8，小 m 时相对标称存储带宽的比值约 0.77。NVFP4 下这两个比值的上限约为 0.40 和 0.42。

这些比值是相对规格文件中标称值的折算，不是硬件计数器读出的占用率，只在同一套公式和同一份规格下有意义。
