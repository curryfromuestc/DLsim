# DLsim 设计文档索引

本目录是 DLsim 的设计契约。每页说明一个主题的能力、前提、边界、依赖和接口。实现必须与这些页面一致；需要改变设计时先改文档。

外部仓库（AISimulate、InferenceX、agentx-harness、Dynamo）只作参考和数据来源，不构成本仓库的约束。

## 阅读顺序

| 页面 | 回答的问题 |
| --- | --- |
| [scope.md](scope.md) | DLsim 要回答什么问题，做什么，不做什么，哪些已经决定 |
| [architecture.md](architecture.md) | 分层、四类配置对象、数据流、语言与依赖、运行时间预算 |
| [workload-and-metrics.md](workload-and-metrics.md) | AgentX trace 的格式、重放语义、负载特征，以及输出指标的定义 |
| [operator-latency.md](operator-latency.md) | 单个算子的时延从哪里来：实测表、插值、跨器件缩放分解及其失效情形 |
| [step-latency.md](step-latency.md) | 一次前向迭代的时延怎样由算子和通信组合得到 |
| [state-engine.md](state-engine.md) | 虚拟时钟、调度、KV 缓存层级、P/D 分离、路由和 session 依赖图 |
| [fabric-and-heterogeneity.md](fabric-and-heterogeneity.md) | scale-up 与 scale-out、域溢出规则、多机柜和异构器件组 |
| [inverse-solving.md](inverse-solving.md) | 给定目标时怎样求参数要求，可辨识性和物理耦合约束 |
| [validation.md](validation.md) | 三层验证、冻结清单、实现步骤与每一步的验收条件 |
| [data-sources.md](data-sources.md) | 数据与参考仓库的位置、版本、快照和许可证核对项 |

## 约定

文档用中文，标识符、字段名和代码用英文。数字必须带单位和来源。未经本项目验证的论断要明确标注为未验证。
