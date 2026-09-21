# 数据来源

本页记录 DLsim 使用的外部数据和参考仓库的位置、版本、快照和许可证核对项。

## 位置

参考仓库是本仓库 `third_party/` 下的 git submodule。下载的数据放在 `traces/` 下，该目录不进版本库。

## 参考仓库

| 仓库 | 地址 | commit | 许可证 | 用途 |
| --- | --- | --- | --- | --- |
| agentx-harness | github.com/SemiAnalysisAI/agentx-harness | 56a0cf70f4c0359454ee4bd15a17770b541a3e3e | Apache-2.0 | 重放语义与 Weka 加载规则的参照；产生 session 依赖图的对照数据 |
| aisimulate | github.com/ai-dynamo/aisimulate | 108cb5d6ca8b4048915783b6155445f1c731825a | Apache-2.0，第三方材料见其 `THIRD_PARTY_NOTICES.md` | 实测算子数据；插值规则与模型算子图的算法参照 |
| dynamo | github.com/ai-dynamo/dynamo | 5593e8857c508bebce7259e107a85c57cd142fd8 | Apache-2.0，部分测试数据为 MIT | 路由、KV 分层管理和 P/D 交付行为的参照 |
| inferencex | github.com/SemiAnalysisAI/InferenceX | 8d70414dfe7f3916574acbdfcb53df9ee82cea15 | Apache-2.0 | AgentX recipe（151 个）、指标定义（`MODELS.md`） |

四个仓库均为浅克隆。aisimulate 克隆时跳过了 LFS 对象；算子数据的 parquet 文件不在 LFS 中，是完整的。

这些仓库只作参考和数据来源。DLsim 不链接、不调用其中的代码。

## AgentX trace

| 项 | 值 |
| --- | --- |
| 数据集 | HuggingFace `semianalysisai/cc-traces-weka-062126` |
| 许可证 | Apache-2.0（数据集页面声明） |
| 文件 | `traces.jsonl`，1,847,151,435 字节，393 行；另有 `README.md` 和 `stats.txt` |
| 路径 | `traces/agentx/cc-traces-weka-062126/` |
| 版本库 | 不进版本库，`traces/` 在 `.gitignore` 中 |

在无法直接访问 huggingface.co 的网络环境下，可经由 hf-mirror.com 下载。如果环境中配置了 HTTP 代理，需要绕过代理，否则请求会被重定向回 huggingface.co：

```
curl --noproxy '*' -L -O https://hf-mirror.com/datasets/semianalysisai/cc-traces-weka-062126/resolve/main/traces.jsonl
```

## InferenceX 快照

公开 API 的描述在 `https://inferencex.semianalysis.com/api/openapi.json`。响应体是 gzip 压缩的。

| 端点 | 内容 | 状态 |
| --- | --- | --- |
| `/api/v1/benchmarks?model=DeepSeek-V4-Pro` | 每个测试点一行 | 2026-09-19 快照 896 行（agentic_traces 130）；2026-09-20 快照 898 行（agentic_traces 132） |
| `/api/v1/derived-agentic-metrics?ids=` | 规范的 E2E normalized interactivity，p75 与 p90，按 benchmark id 索引，一次最多 200 个 id | 已取 2026-09-20；132 个 agentic id 中返回 126 个，缺的 6 个全是 vr200 |
| `/api/v1/agentic-aggregates?ids=` | ISL、OSL、kvCacheUtil、prefixCacheHitRate 的分位数，按 id 索引 | 已取 2026-09-20；ISL 与 OSL 全部 132 行有；kvCacheUtil 与 prefixCacheHitRate 只有 vLLM 系的 56 行有（b200 vllm、b300 vllm、gb200 dynamo-vllm、mi355x vllm 与 atom），gb300 没有 |
| `/api/v1/tco-feed` | 给外部电子表格 TCO 模型用的吞吐前沿：负载 1024x1024 与 8192x1024，interactivity 档 30、50、75、100，每行为 output_tput_per_gpu，没有价格与成本字段 | 已取 2026-09-20；不是成本假设，也不含 AgentX |
| `/api/v1/collectivex/runs`、`/runs/{runId}` | 集合通信、KV 传输与 host 交换实测，见下节 | 已取 2026-09-20 |
| `/api/v1/request-timeline?id=` | 一个测试点的逐请求记录：warmup 与 profiling 阶段、ttftMs、tpotMs、isl、osl、worker、来源类型（weka_main、weka_subagent、weka_flat） | 已取 id 440971（gb300 dynamo-trt，conc 4）：543 条，warmup 44、profiling 499，窗口 3,721 s |

2026-09-20 快照中 agentic_traces 的点数按硬件与框架：b200 24（sglang 12，vllm 12），b300 27（sglang 12，vllm 15），gb200 7（dynamo-vllm），gb300 27（dynamo-sglang 7，dynamo-trt 6，dynamo-vllm 14），h200 5（dynamo-sglang），mi355x 36（atom 10，mori-sglang 7，sglang 7，vllm 12），vr200 6（trt）。比 2026-09-19 多出的 2 行属于 b200。

快照文件在 `traces/inferencex/`。快照按日期命名，不覆盖旧快照。响应体可能是 gzip。哪些行已被查看过，记录在 [validation.md](validation.md)。

## 部署配置的来源

InferenceX 的 `configs/nvidia-master.yaml` 按硬件与框架列出每个 AgentX 部署的拓扑（prefill 与 decode 的 worker 数、tp、ep、dp-attn、并发、投机解码、KV offload）。submodule 钉住的 commit 8d70414 里没有 gb300 dynamo-trt 的 DSV4 条目，主分支有（`dsv4-fp4-gb300-dynamo-trt-agentx`，六个点：并发 4、24、388、736、1152、2626），已把主分支的该文件存为 `traces/inferencex/nvidia-master_main_2026-09-20.yaml`。条目引用的 recipe 文件（`recipes/dsv4/<framework>/gb300-fp4/agentx/*.yaml`）不在公开仓库里，因此 chunk 大小、每 rank 最大 batch 等 stack 参数只能从 recipe 文件名（例如 `disagg-3p1d-dep8-dep16-c1152-b32-mtp`）和文档推断，作为标定参数处理。

每个测试点的 mapping 与 run 配置由 `tools/reference/gen_points.py` 从 benchmarks 快照行的拓扑字段生成到 `configs/points/`，只读拓扑字段，不读指标。引擎限值不在快照里，从产生该行的 InferenceX recipe 读取并写入点文件的 `stack_overrides`：gb300、gb200、h200 的点匹配 `third_party/inferencex/benchmarks/multi_node/srt-slurm-recipes/dsv4/<框架>/<器件>/agentx/*.yaml`（按部署方式、各角色 GPU 数、worker 数和文件名中的并发数匹配，点文件的 `recipe` 字段记录匹配到的文件），取 decode 角色的每 rank 最大序列数与每步 token 预算、prefill 角色的最大序列数与 chunk 大小、MTP 草稿长度（trtllm `max_batch_size`/`max_num_tokens`/`max_draft_len`，sglang `max-running-requests`/`chunked-prefill-size`/`speculative-num-draft-tokens` 减一，vllm `max-num-seqs`/`max-num-batched-tokens`/`num_speculative_tokens`）；b200、b300 的点按 `benchmarks/single_node/agentic/dsv4_fp4_*.sh` 的规则生成（最大序列数为并发的两倍并按 attention DP 分摊，chunk 8192 或 DEP 下的 6144/16384）。这些限值决定曲线的形状：gb300 dynamo-trt 并发 388 的 decode 每 rank 只允许 4 个序列，32 个 rank 共 128 个 decode 槽位，请求在 prefill 完成后等待槽位。行里 `dp_attention` 为真时 tp 字段表示 attention DP 的规模（tp 取 1），否则表示 TP；MoE 的 EP 取 ep 字段。聚合部署的 GPU 数按 worker 数乘每 worker GPU 数计算，与行里按 tp 乘 ep 填充的字段不一致是预期的。

## CollectiveX

InferenceX 的集合通信与 KV 传输实测，契约只有 version=1。`/api/v1/collectivex/latest` 返回最近一次 run，当前是只含 swap 的 h100 run，集合通信数据要按 run_id 从 `/runs/{runId}` 取。已保存的 run 在 `traces/inferencex/collectivex_2026-09-20/`：EP 通信 33477867072、33356406487、33476729962、33775738245、33893771034、34432070017、34504491720、34939333022；KV 传输 33412478973；gb300 swap 35156186434。

| 类别 | 内容 | 覆盖 |
| --- | --- | --- |
| EP 通信 | DeepSeek-V4-Pro shape 的 dispatch、stage、combine、roundtrip 的 p50 到 p99 时延与 payload_bytes；后端 deepep-v2、nccl-ep、uccl-ep、flashinfer-ep、mori；normal 与 low-latency 模式 | EP8 与 EP16；decode 每 rank 1 到 512 token，prefill 1,024 到 8,192；bf16 与 fp8。gb200、gb300 的 EP16 是 NVL72 域内 4 节点乘 4 卡经 MNNVL；b200、b300、h100、h200、mi355x 的 EP16 是 2 节点乘 8 卡经 RDMA。没有 EP32 及以上，没有跨机柜 |
| KV 传输 | kv-dsv4 fp8，mooncake 与 nixl，push 与 pull，bulk 与 paged，含时延与 GB/s | ISL 2,048 到 524,288；链路 rdma，gb200 与 gb300 另有 mnnvl |
| host 交换 | H2D、D2H、D2D，contiguous 与 random 布局，块大小扫描 | payload 到 1 GiB；gb300、h100、h200、mi300x、mi325x |

EP 通信行的 payload_bytes 是全部 rank 之和，每器件的消息字节 n = payload_bytes / ep；normal 模式按目的 rank 去重，EP8 每 token 62,720 B、EP16 73,472 B，low-latency 模式每 token 86,016 B（7168 × 2 B × top-6）。gb200 与 gb300 的 EP8 行本身也是 2 节点乘 4 卡，位于同一 NVL72 域内。

从中读出的几个量，作为 fabric 与 state-engine 参数的来源：gb300 上 nixl 经 MNNVL 拉取 524,288 token 的 KV（2,945 MB）p50 4.21 ms，约 699 GB/s，经 RDMA 39.7 ms，约 74 GB/s；对 kv-dsv4 fp8 行按 ISL 做线性拟合，每 token 的 KV 字节数为 5,609 B（斜率），另有每序列 4.57 MB 的常数项（61 层滑窗各 128 个条目），与算子图按 585 B 每 KV 条目（576 B 数据加 9 B scale）和 144 B 每 indexer 条目推出的布局一致；gb300 pinned host 与器件间 1 GiB 传输 p50 9.1 ms，约 117 GB/s，器件内约 980 GB/s，计时含提交与同步。

API 数据的使用条款尚未核对。

## 实测算子数据

| 项 | 值 |
| --- | --- |
| 路径 | aisimulate 的 `python/aisimulate/src/aisimulate_core/systems/data/` |
| 体积 | 约 117 MB |
| 系统目录 | 14 个；有数据的 10 个：a100_sxm、b200_sxm、b300_sxm、b60、gb200、gb300、h100_sxm、h200_sxm、l40s、rtx_pro_6000_server |
| 目录结构 | `<system>/<family>/<framework>/<version>/<table>_perf.parquet` |
| 框架 | trtllm、vllm、sglang，通信为 nccl；每个框架多个版本 |
| 器件规格 | 同级目录的 `<system>.yaml` |

部分版本目录只有 `reuse.yaml`，按表声明复用另一个版本的数据（例如 gb300 的 trtllm 1.3.0rc23 下 quantize 等表复用 1.3.0rc20，该版本只新采集了 DSV4 的 sparse attention 表）。复用规则另见 `perf_data_reuse_manifest.yaml` 和 `query_versions.yaml`。导入工具必须解析这些规则。

各系统目录下的 README 记录了数据的上游来源。例如 b200_sxm 的 18 张表来自 AIConfigurator 的 commit 915f590680d8a79fe9c39f6f3a9ff13bc267fcce，其中 16 张是逐字节拷贝，2 张 attention 表是合并后的派生表。

该数据中没有 Vera Rubin；非 NVIDIA 的加速器只有 b60（Intel），且只有 vllm 的 attention、comm、gemm、moe 四张表。通信表全部是单节点，attention 表的上下文覆盖因框架而异，范围见 [operator-latency.md](operator-latency.md)。Vera Rubin 的规格需要由用户在 device 配置中给出并注明来源。

## 许可证核对项

| 项 | 结论或待办 |
| --- | --- |
| 读取 aisimulate 的 parquet 数据 | Apache-2.0 允许 |
| 由 parquet 转换得到的二进制算子表 | 属于派生作品。建议不进版本库，由 `dlsim-import` 在本地从 submodule 生成，从而不发生再分发。如果以后要分发，需要保留版权与 NOTICE，并声明做过修改 |
| 以 `deepseek_v4.py` 为参照写 C++ 的模型算子图 | Apache-2.0 允许。若实现接近逐行翻译，在本仓库的 NOTICE 中注明来源 |
| 以 agentx-harness 为参照实现加载规则 | 同上 |
| aisimulate 中的第三方材料 | 动用具体文件之前逐个查看其 `THIRD_PARTY_NOTICES.md` 中的对应条目；已知 DeepSeek 的 config.json 为 MIT |
| InferenceX API 数据的使用条款 | 未核对。在对外发布含该数据的图表之前必须核对 |
| HuggingFace 数据集 | Apache-2.0 |
| 本仓库自身的许可证 | 未定，由仓库所有者决定 |
