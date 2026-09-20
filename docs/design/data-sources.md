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
| `/api/v1/benchmarks?model=DeepSeek-V4-Pro` | 每个测试点一行 | 已保存 2026-09-19 的快照，896 行，其中 agentic_traces 130 行 |
| `/api/v1/derived-agentic-metrics` | 规范的 E2E normalized interactivity | 尚未获取 |
| `/api/v1/agentic-aggregates` | agentic 聚合量 | 尚未获取 |
| `/api/v1/tco-feed` | 成本假设 | 尚未获取 |
| `/api/v1/collectivex/*` | 集合通信实测 | 尚未获取，内容未核对 |
| `/api/v1/request-timeline` | 逐请求时间线 | 尚未获取 |

快照中 agentic_traces 的点数按硬件：b200 22，b300 27，gb200 7，gb300 27，h200 5，mi355x 36，vr200 6。

快照文件在 `traces/inferencex/`。快照按日期命名，不覆盖旧快照。哪些行已被查看过，记录在 [validation.md](validation.md)。

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

该数据中没有 Vera Rubin，也没有任何非 NVIDIA 的加速器。Vera Rubin 的规格需要由用户在 device 配置中给出并注明来源。

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
