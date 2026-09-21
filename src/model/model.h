#pragma once
// Model to operator graph. Design: step-latency.md "模型到算子图".
#include <cstdint>
#include <string>
#include <vector>

#include "config/config.h"
#include "perfdata/op_latency.h"
#include "step/step.h"

namespace dlsim::model {

enum class OpKind : uint8_t { Attention, Gemm, Moe, Mhc, Embedding };

struct Roofline {
  double compute_ms = 0;   // FLOPs / device peak of the operator's dtype
  double membw_ms = 0;     // bytes / device main memory bandwidth
};

struct Op {
  std::string name;
  OpKind kind = OpKind::Gemm;
  double layers = 1;         // repetitions per forward pass
  bool all_ranks = false;    // token count summed over attention DP ranks (routed MoE); otherwise per rank
  bool routed = false;       // part of the routed-MoE branch that overlaps with the shared expert in decode
  bool shared = false;       // shared-expert branch
  int compress_ratio = 0;    // attention only
  OpQuery query;             // static keys; StepLatency adds m / num_tokens / batch_size / isl / step
};

struct ModelDesc {
  std::string name;
  int layers = 0, hidden = 0, num_heads = 0, head_dim = 0, rope_head_dim = 0;
  int q_lora_rank = 0, o_lora_rank = 0, o_groups = 0;
  int index_n_heads = 0, index_head_dim = 0, index_topk = 0, sliding_window = 0;
  int num_experts = 0, topk = 0, moe_inter = 0, n_shared = 0, vocab = 0;
  int hc_mult = 0, hc_sinkhorn_iters = 0, nextn_layers = 0;
  std::vector<int> compress_ratios;   // layers + nextn_layers entries; 0 = pure sliding window
  std::string architecture, table_model;
  // The same checkpoint is labelled differently across framework tables (sglang: deepseek-ai/DeepSeek-V4-Pro on gb300,
  // sgl-project/DeepSeek-V4-Pro-FP8 on h200; kv_cache_dtype fp8_e4m3 instead of fp8). Alternatives tried in order.
  std::vector<std::string> table_model_aliases, kv_cache_dtype_aliases;
  std::string gemm_dtype, expert_dtype, kv_cache_dtype, mla_dtype;
  double gemm_weight_bytes = 1, expert_weight_bytes = 1, kv_entry_bytes = 1, index_entry_bytes = 1;

  std::vector<Op> ops(const PoolSpec& pool, const StackSpec& stack, Phase phase) const;
  double weight_bytes_per_gpu(const PoolSpec& pool, const StackSpec& stack) const;
  double param_count(bool with_mtp) const;
  double kv_bytes_per_token() const;
  double kv_bytes_per_sequence(bool with_mtp) const;
  Roofline roofline(const Op& op, const DeviceSpec& dev, double tokens, double batch, double isl,
                    double step, const PoolSpec& pool) const;

  int layers_with_ratio(int ratio) const;
  double attention_weight_bytes(int ratio, int tp) const;
  double layer_weight_bytes(int ratio, const PoolSpec& pool) const;
};

const ModelDesc& deepseek_v4_pro();
const ModelDesc& by_name(const std::string& name);

}  // namespace dlsim::model
