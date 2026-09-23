# DLsim

DLsim 是一个系统级的 LLM 推理仿真工具。输入是用配置文件描述的推理网络（器件、互联、部署映射、软件栈）和 AgentX 多用户 agentic 负载，输出是与 InferenceX 行格式一致的系统指标，用于把一个尚未实现的推理系统与已有公开实测的系统（GB300 NVL72 等）比较，并反过来求出达到某个目标所需的器件、互联和软件条件。

它是粗粒度模型，遵循正确的误差好过错误的精确。

设计文档在 [docs/design/](docs/design/README.md)。

## 构建

依赖：C++20 编译器（g++ 11 已验证）、CMake 3.20 以上、Ninja、OpenSSL 的 libcrypto 开发包（`libssl-dev`，用于复现 harness 的 sha256 种子）。simdjson、yaml-cpp、Eigen 由 CMake FetchContent 拉取。导入工具 `dlsim-import` 需要 Arrow/Parquet 的 C++ 库，默认从 pyarrow 的包目录取（`tools/import/CMakeLists.txt` 的 `DLSIM_PYARROW_DIR`），不需要导入工具时加 `-DDLSIM_BUILD_IMPORT=OFF`。

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

## 数据准备

```
git submodule update --init --depth 1
./build/tools/import/dlsim-import third_party/aisimulate/python/aisimulate/src/aisimulate_core/systems perfdata
curl --noproxy '*' -L -o traces/agentx/cc-traces-weka-062126/traces.jsonl \
  https://hf-mirror.com/datasets/semianalysisai/cc-traces-weka-062126/resolve/main/traces.jsonl
```

`perfdata/` 与 `traces/` 不进版本库。InferenceX 的快照文件（benchmarks、derived-agentic-metrics、collectivex）的取法见 [docs/design/data-sources.md](docs/design/data-sources.md)。首次读取 trace 会在 `traces/cache/` 写二进制缓存（2.7 GB），之后加载约 2 s。

## 用法

```
dlsim trace-stats [--trace f] [--cache d]                      负载统计，与数据集 stats.txt 对照
dlsim trace-graph --index i                                    一条 trace 的 session 依赖图（与 agentx-harness 对照的规范 JSON）
dlsim op --device gb300 --op gemm --num m=16,n=6144,k=7168 --cat gemm_dtype=fp8_block,framework=trtllm
                                                               查一次算子表，打印来源与说明
dlsim step --point p.yaml (--prefill N | --decode N) --past K [--verbose]
                                                               给一个部署点算一次前向迭代，逐算子分解
dlsim sim --point p.yaml [--out row.json] [--stack-override k=v,...]
                                                               完整重放，输出 InferenceX 行
dlsim validate-points [--hardware h] [--framework f] [--ids a,b] [--jobs n] [--out r.json]
                                                               对 configs/points 里的点与快照逐点比较
dlsim calibrate --ids 440971 [--fixed-ms ...] [--accept-mean ...] [--overhead-ms ...]
                                                               网格标定每步固定开销、MTP 接受长度与每请求流水线开销；
                                                               选定值写入 configs/calibration.yaml，再由 gen_points.py 写进各点
python3 tools/reference/gen_points.py <snapshot.json> configs/points
                                                               从 InferenceX 快照、recipe 与 calibration.yaml 生成部署点
python3 tools/reference/summarize_validation.py [--snapshot b.json --derived d.json] r.json ...
                                                               validate-points 报告转成误差表；给快照时跨硬件配比值对
DLSIM_PASS_DEBUG=<pool> dlsim sim ...                          每 40 个 pass 打印该池的 batch 与请求状态（诊断）
dlsim validate-fabric [--out r.json]                           CollectiveX EP8 标定、EP16 预测报告
dlsim solve --query q.yaml [--out r.json]                      反向求解
dlsim screen --point p.yaml                                    闭式筛选
tools/import/dlsim-perfdata-report perfdata <systems_dir> rows|loo|extrap|layer1 ...
                                                               算子表的导入行数、留一插值、外推与第一层报告
```

部署点配置 `configs/points/*.yaml` 由 `tools/reference/gen_points.py` 从 InferenceX benchmarks 快照的拓扑字段生成，引擎限值（每 rank 最大序列数、chunk 大小、MTP 草稿长度）从对应的 InferenceX recipe 读入点文件的 `stack_overrides`；device、fabric、stack 配置在 `configs/` 下。
