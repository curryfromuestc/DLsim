# 互联与异构

本页定义 fabric 配置的语义、通信按域计价的规则、多机柜部署和异构器件组。

## 层次模型

互联建模为带参数的层次，不做包级仿真，不建模拥塞控制算法。

```
器件 --(scale-up: alpha_up, L_up)-- scale-up 域, 规模 S
                                        |
                        (scale-out: alpha_out, L_out, 收敛比 r)
                                        |
                                  其他 scale-up 域 / 机柜
                                        |
                        (host 链路: alpha_h, L_h)  每器件到 host 存储层
                        (远端存储链路: alpha_s, L_s)
```

每一层的参数是每器件单向带宽 L（字节每秒）、每消息延迟 alpha（秒）和可选的收敛比 r。scale-out 的有效带宽为 L_out / r。

各种互联技术在本模型中只通过这几个参数和域规模 S 区分：

| 类别 | 例子 | 在模型中的差别 |
| --- | --- | --- |
| scale-up | NVLink 与 NVSwitch、华为 UB、UALink、基于以太网的 scale-up（SUE） | S、L_up、alpha_up |
| scale-out | InfiniBand、RoCE、UEC | L_out、alpha_out、r |

基于信用的流控与基于 PFC 的无损以太网在尾延迟上的差别，第一阶段只能通过 alpha 的取值和一个可选的尾延迟倍数体现。这是已知的粗糙处。

## 所有字段可选

fabric 的每个字段都可省略：

| 省略的字段 | 语义 |
| --- | --- |
| scale-up 域规模 S | 全部器件属于同一个 scale-up 域 |
| 某一层的 L | 该层带宽不受限，字节项为零 |
| 某一层的 alpha | 该层每消息延迟为零 |
| 整个 scale-out 层 | 不存在 scale-out；跨域的 mapping 报错 |
| host 或远端存储层 | 不存在该存储层 |

输出中列出哪些资源在本次运行中不受限，使结论的前提可见。

## 域溢出规则

mapping 把每个并行组放到具体的器件上。一个集合通信的计价由它跨越的最高层决定。

并行组完全落在一个 scale-up 域内时，使用 alpha_up 和 L_up。

并行组跨 k 个域、每域 g 个器件、总规模 p = k·g 时，均匀 all-to-all 中每个器件发出的数据有 (p − g)/p 的比例离开本域，留在域内的比例为 (g − 1)/p。消息数按目的器件数计：跨域 p − g 条，域内 g − 1 条；延迟项按上表的规则取常数或串行。计价为

```
t = alpha_out × 跨域消息数 + alpha_up × 域内消息数
    + max( 域内字节 / L_up , 跨域字节 / (L_out / r) )
```

all-reduce 跨域时按分层算法计价：域内 reduce-scatter，跨域 all-reduce，域内 all-gather，三段相加。all-gather 与 reduce-scatter 跨域时取其中对应的两段，跨域段按 all-reduce 跨域段的一半字节计。这是粗略近似；第一阶段的 attention DP 组都落在单个域内，该路径不被 InferenceX 的点触发。

闭式的集合通信代价（p 为组规模，n 为每器件的消息字节数）：

| 集合通信 | 字节项 | 延迟项 |
| --- | --- | --- |
| all-reduce（ring） | 2(p−1)/p × n / L | 2(p−1) × alpha |
| all-gather、reduce-scatter | (p−1)/p × n / L | (p−1) × alpha |
| all-to-all | (p−1)/p × n / L | 默认 1 × alpha（各目的地并发发出，取所跨最高层的 alpha）；fabric 配置 `alltoall_serial_latency: true` 时为 (p−1) × alpha |
| P2P | n / L | alpha |

闭式公式是主路径，实测数据在其覆盖范围内标定 alpha 与 L，覆盖范围见 [operator-latency.md](operator-latency.md)。CollectiveX 从 EP8 到 EP16 的实测是域溢出规则的直接对照：deepep-v2 low-latency 模式、decode 每 rank 256 token、bf16 时，gb300 上 dispatch 的 p50 从 EP8 的 81 µs 到 NVL72 域内 4 节点 EP16 的 88 µs，b200 上从 EP8 的 81 µs 到 2 节点跨 RDMA EP16 的 292 µs；h200 上 normal 模式从 101 µs 到 855 µs（CollectiveX run 33477867072 与 33356406487）。同一数据在最小 payload 处 EP8 到 EP16 的延迟只增长 0.97 到 1.42 倍，与 (p − 1) × alpha 的 2.14 倍不符，与常数延迟一致，因此 all-to-all 的延迟项默认取常数。EP8 行反解得到的 scale-up 有效带宽约 500 到 670 GB/s（gb300、b200，deepep-v2），低于 900 GB/s 的标称值；EP16 跨 RDMA 行在大 payload 处反解出的跨节点有效带宽为 45 到 49 GB/s（b200 deepep-v2 low-latency），与规格值 50 GB/s 一致，而 normal 模式各后端为 90 到 125 GB/s，是规格值的 2 倍，说明 L_out 应按后端与模式标定而不是只取规格值。AISimulate 的规格文件给出了可作参照的量级：节点内 900 GB/s，节点间 b200 为 50 GB/s、b300 为 100 GB/s，P2P 延迟 10 µs。host 链路的量级可参照 CollectiveX 的 swap_blocks：gb300 上 pinned host 与器件之间 1 GiB 传输的 p50 为 9.1 ms，约 117 GB/s，器件内拷贝约 980 GB/s，计时含提交与同步。其他系统的取值由用户在配置中给出并注明来源。

## 链路共享

三类流量共用每器件的端口：步内集合通信、P/D 的 KV 交付、KV 在存储层之间的搬运。

后两类是大块传输，在所经过的每一层链路上先到先服务地排队。步内集合通信在单步时延中计价。两者之间的干扰用 fabric 配置中的一个固定带宽份额 bulk_share 表达，默认 0，即不建模干扰。

输出中报告大块传输与步内通信同时活跃的时间占比。该占比高时，忽略干扰的结论不可信，需要设置 bulk_share 重新运行。

## 多机柜

InferenceX 公开的 AgentX 部署最大为 60 个 GPU，没有超出单个 NVL72 域的点，因此多机柜路径没有系统级的对照数据。算子级的对照只有 CollectiveX 的 2 节点跨 RDMA EP16 行，AISimulate 的 moe_a2a 表全部是单节点，跨机柜没有任何实测。多机柜结果一律标注为未经系统级验证。

多机柜在两种情形下是必需的输入：scale-up 域规模 S 小于一个部署所需的器件数；需要的 KV 容量超过一个域内全部器件的存储。第二种情形对主存容量大而算力小的器件尤其相关。

## 异构器件组

device 配置可同时定义多种器件。mapping 把工作分配到器件组，第一阶段支持三种粒度：

| 粒度 | 含义 | 增加的通信 |
| --- | --- | --- |
| 阶段级 | prefill 池和 decode 池使用不同器件 | 每轮一次 KV 交付 |
| 存储级 | 一类器件或 host 作为 KV 的下层存储 | 命中下层时的加载，淘汰时的写回 |
| 副本级 | 同一阶段内有不同器件的 worker，由路由分配请求 | 无 |

层内异构（attention 与 FFN 分别放在不同器件上）每层增加两次跨器件传输，decode 阶段对 alpha 极其敏感。第一阶段不支持，配置出现时报错。

跨器件类型的 KV 交付要求两端的 KV 格式一致（dtype、压缩方式、块大小）。不一致时需要转换算子，第一阶段不支持，报错。

每个器件组独立引用自己的 OpLatencySource。异构部署的指标除总量外按器件组分别输出吞吐、按资源的时间占比和成本，使对比可以归因到具体的器件组。

## 与物理约束的关系

片上 SerDes 与主存 PHY 共用芯片边缘，互联带宽与主存带宽不是独立变量。这类耦合在反向求解中作为约束出现，见 [inverse-solving.md](inverse-solving.md)；正向仿真不检查它。
