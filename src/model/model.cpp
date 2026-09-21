#include "model/model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dlsim::model {
namespace {

double flops_ms(const DeviceSpec& dev, const std::string& dtype, double flops) {
  auto it = dev.flops.find(dtype);
  return it != dev.flops.end() && it->second > 0 ? flops / it->second * 1e3 : 0.0;
}

double bytes_ms(const DeviceSpec& dev, double bytes) {
  const double bw = dev.memory.empty() ? 0.0 : dev.memory[0].bandwidth_Bps;
  return bw > 0 ? bytes / bw * 1e3 : 0.0;
}

Op make_op(std::string name, OpKind kind, double layers) {
  Op o;
  o.name = std::move(name);
  o.kind = kind;
  o.layers = layers;
  return o;
}

ModelDesc make_deepseek_v4_pro() {
  ModelDesc m;
  m.name = "deepseek-v4-pro";
  m.layers = 61;
  m.hidden = 7168;
  m.num_heads = 128;
  m.head_dim = 512;
  m.rope_head_dim = 64;
  m.q_lora_rank = 1536;
  m.o_lora_rank = 1024;
  m.o_groups = 16;
  m.index_n_heads = 64;
  m.index_head_dim = 128;
  m.index_topk = 1024;
  m.sliding_window = 128;
  m.num_experts = 384;
  m.topk = 6;
  m.moe_inter = 3072;
  m.n_shared = 1;
  m.vocab = 129280;
  m.hc_mult = 4;
  m.hc_sinkhorn_iters = 20;
  m.nextn_layers = 1;
  m.compress_ratios = {128, 128};
  for (int i = 2; i < 61; ++i) m.compress_ratios.push_back(i % 2 == 0 ? 4 : 128);
  m.compress_ratios.push_back(0);
  m.architecture = "DeepseekV4ForCausalLM";
  m.table_model = "sgl-project/DeepSeek-V4-Pro-FP8";
  m.table_model_aliases = {"deepseek-ai/DeepSeek-V4-Pro"};
  m.kv_cache_dtype_aliases = {"fp8_e4m3"};
  m.gemm_dtype = "fp8_block";
  m.expert_dtype = "nvfp4";
  m.kv_cache_dtype = "fp8";
  m.mla_dtype = "bfloat16";
  m.gemm_weight_bytes = 1.0;
  m.expert_weight_bytes = 9.0 / 16.0;
  m.kv_entry_bytes = 1.0;
  m.index_entry_bytes = 1.0;
  return m;
}

}  // namespace

const ModelDesc& deepseek_v4_pro() {
  static const ModelDesc m = make_deepseek_v4_pro();
  return m;
}

const ModelDesc& by_name(const std::string& name) {
  if (name == "deepseek-v4-pro") return deepseek_v4_pro();
  throw std::runtime_error("unknown model '" + name + "'");
}

int ModelDesc::layers_with_ratio(int ratio) const {
  int n = 0;
  for (int i = 0; i < layers; ++i) n += compress_ratios[i] == ratio;
  return n;
}

double ModelDesc::attention_weight_bytes(int cr, int tp) const {
  const double h = hidden, qlr = q_lora_rank, olr = o_lora_rank, hd = head_dim;
  const double heads = double(num_heads) / tp, og = std::max(1, o_groups / tp);
  double gemm = h * qlr + qlr * heads * hd + h * hd + og * olr * h;
  double bf16 = heads * hd * olr;
  double f32 = heads;
  if (cr != 0) {
    const double cm = cr == 4 ? 2 : 1;
    gemm += 2 * h * cm * hd;
    f32 += cr * cm * hd;
  }
  if (cr == 4) {
    gemm += qlr * index_n_heads * index_head_dim + 2 * h * 2 * index_head_dim;
    bf16 += h * index_n_heads;
    f32 += cr * 2 * index_head_dim;
  }
  return gemm * gemm_weight_bytes + bf16 * 2 + f32 * 4;
}

double ModelDesc::layer_weight_bytes(int cr, const PoolSpec& pool) const {
  const double h = hidden, inter = moe_inter;
  const double mix_hc = (2.0 + hc_mult) * hc_mult, hc_dim = double(hc_mult) * h;
  const double mhc = 2 * (mix_hc * hc_dim + mix_hc + 3) * 2;
  const double shared = n_shared * 3 * h * inter * gemm_weight_bytes / pool.tp;
  const double router = double(num_experts) * h * 2;
  const double experts = double(num_experts) * 3 * h * inter * expert_weight_bytes / pool.moe_ep / pool.moe_tp;
  return attention_weight_bytes(cr, pool.tp) + mhc + shared + router + experts;
}

double ModelDesc::weight_bytes_per_gpu(const PoolSpec& pool, const StackSpec& stack) const {
  double per_stage = 0;
  const int n = layers + (stack.mtp_nextn > 0 ? nextn_layers : 0);
  for (int i = 0; i < n; ++i) per_stage += layer_weight_bytes(compress_ratios[i], pool);
  per_stage = per_stage / pool.pp;
  const double embed = double(vocab) * hidden * 2;
  const double logits = double(vocab) * hidden * 2 / pool.tp;
  return embed + logits + per_stage;
}

double ModelDesc::param_count(bool with_mtp) const {
  double params = 0;
  const int n = layers + (with_mtp ? nextn_layers : 0);
  for (int i = 0; i < n; ++i) {
    const int cr = compress_ratios[i];
    const double h = hidden, inter = moe_inter, qlr = q_lora_rank, olr = o_lora_rank, hd = head_dim;
    double attn = h * qlr + qlr * num_heads * hd + h * hd + double(o_groups) * olr * h + double(num_heads) * hd * olr + num_heads;
    if (cr != 0) {
      const double cm = cr == 4 ? 2 : 1;
      attn += 2 * h * cm * hd + cr * cm * hd;
    }
    if (cr == 4) attn += qlr * index_n_heads * index_head_dim + 2 * h * 2 * index_head_dim + h * index_n_heads + cr * 2 * index_head_dim;
    const double mix_hc = (2.0 + hc_mult) * hc_mult, hc_dim = double(hc_mult) * h;
    const double mhc = 2 * (mix_hc * hc_dim + mix_hc + 3);
    const double shared = n_shared * 3 * h * inter;
    const double router = double(num_experts) * h;
    const double experts = double(num_experts) * 3 * h * inter;
    params += attn + mhc + shared + router + experts;
  }
  return params + 2.0 * vocab * hidden;
}

double ModelDesc::kv_bytes_per_token() const {
  const double entry = (head_dim + rope_head_dim) * kv_entry_bytes;
  const double index = index_head_dim * index_entry_bytes;
  double total = 0;
  for (int i = 0; i < layers; ++i) {
    const int cr = compress_ratios[i];
    if (cr == 0) continue;
    total += entry / cr;
    if (cr == 4) total += index / cr;
  }
  return total;
}

double ModelDesc::kv_bytes_per_sequence(bool with_mtp) const {
  const int n = layers + (with_mtp ? nextn_layers : 0);
  return double(n) * sliding_window * (head_dim + rope_head_dim) * kv_entry_bytes;
}

std::vector<Op> ModelDesc::ops(const PoolSpec& pool, const StackSpec& stack, Phase phase) const {
  const bool prefill = phase == Phase::Prefill;
  std::vector<Op> v;
  const double h = hidden;

  Op embed = make_op("embedding", OpKind::Embedding, 1);
  embed.query.op = "embedding";
  embed.query.num = {{"hidden_size", h}};
  embed.query.cat = {{"dtype", "bfloat16"}};
  v.push_back(embed);

  auto mhc = [&](const char* which) {
    Op o = make_op(std::string("mhc_") + which, OpKind::Mhc, layers);
    o.query.op = "mhc_module";
    o.query.num = {{"hc_mult", double(hc_mult)}, {"hidden_size", h}, {"num_sites", 2},
                   {"sinkhorn_iters", double(hc_sinkhorn_iters)}};
    o.query.cat = {{"op_name", which}, {"architecture", architecture}};
    return o;
  };
  v.push_back(mhc("pre"));

  for (int cr : {4, 128}) {
    const int n = cr == 4 ? layers_with_ratio(4) : layers_with_ratio(128) + layers_with_ratio(0);
    Op o = make_op(cr == 4 ? "csa_attention" : "hca_attention", OpKind::Attention, n);
    o.compress_ratio = cr;
    o.query.op = std::string(cr == 4 ? "dsv4_csa_" : "dsv4_hca_") + (prefill ? "context_module" : "generation_module");
    o.query.num = {{"num_heads", double(num_heads) / pool.tp}, {"tp_size", double(pool.tp)}, {"compress_ratio", double(cr)}};
    o.query.cat = {{"model", table_model}, {"architecture", architecture}, {"mla_dtype", mla_dtype},
                   {"kv_cache_dtype", kv_cache_dtype}, {"gemm_type", gemm_dtype}};
    v.push_back(o);
  }

  v.push_back(mhc("post"));

  auto gemm = [&](const std::string& name, double n, double k, const std::string& dtype, double reps) {
    Op o = make_op(name, OpKind::Gemm, reps);
    o.query.op = "gemm";
    o.query.num = {{"n", n}, {"k", k}};
    o.query.cat = {{"gemm_dtype", dtype}};
    return o;
  };
  Op gate_up = gemm("shared_gate_up_gemm", 2.0 * moe_inter / pool.tp, h, gemm_dtype, layers);
  gate_up.shared = true;
  v.push_back(gate_up);
  Op ffn2 = gemm("shared_ffn2_gemm", h, double(moe_inter) / pool.tp, gemm_dtype, layers);
  ffn2.shared = true;
  v.push_back(ffn2);
  Op router = gemm("router_gemm", num_experts, h, "bfloat16", layers);
  router.routed = true;
  v.push_back(router);

  Op moe = make_op("moe", OpKind::Moe, layers);
  moe.all_ranks = true;
  moe.routed = true;
  moe.query.op = stack.wide_ep ? "wideep_moe" : "moe";
  moe.query.num = {{"hidden_size", h}, {"inter_size", double(moe_inter)}, {"topk", double(topk)},
                   {"num_experts", double(num_experts)}, {"moe_tp_size", double(pool.moe_tp)},
                   {"moe_ep_size", double(pool.moe_ep)}};
  moe.query.cat = {{"moe_dtype", expert_dtype}, {"distribution", stack.moe_distribution}};
  v.push_back(moe);

  v.push_back(gemm("logits_gemm", double(vocab) / pool.tp, h, "bfloat16", 1));
  return v;
}

Roofline ModelDesc::roofline(const Op& op, const DeviceSpec& dev, double tokens, double batch, double isl,
                             double step, const PoolSpec& pool) const {
  const double h = hidden;
  Roofline r;
  switch (op.kind) {
    case OpKind::Embedding:
      r.membw_ms = bytes_ms(dev, tokens * h * 2);
      return r;
    case OpKind::Gemm: {
      const double n = op.query.num.at("n"), k = op.query.num.at("k");
      const bool bf16 = op.query.cat.at("gemm_dtype") == "bfloat16";
      const double wb = bf16 ? 2 : gemm_weight_bytes, ab = bf16 ? 2 : 1;
      r.compute_ms = flops_ms(dev, bf16 ? "bf16" : "fp8", 2 * tokens * n * k);
      r.membw_ms = bytes_ms(dev, wb * n * k + ab * tokens * k + 2 * tokens * n);
      return r;
    }
    case OpKind::Mhc: {
      const double hc = hc_mult, mix = (2 + hc) * hc, hc_dim = hc * h, nt = tokens;
      const bool pre = op.query.cat.at("op_name") == "pre";
      const double flops = pre ? 2 * (2 * nt * hc_dim * mix + nt * hc_dim * 3 + nt * (hc * hc + 2 * hc) * hc_sinkhorn_iters + 2 * nt * hc * h)
                               : 2 * (2 * nt * hc * hc * h + 2 * nt * hc * h);
      double bytes = 2 * (mix * hc_dim + mix + 3) * 2 + 2 * nt * hc_dim * 2 * 2;
      if (pre) bytes += 2 * nt * (2 * hc + hc * hc) * 4;
      r.compute_ms = flops_ms(dev, "bf16", flops);
      r.membw_ms = bytes_ms(dev, bytes);
      return r;
    }
    case OpKind::Moe: {
      const double inter = moe_inter, local = double(num_experts) / pool.moe_ep;
      const double per_expert = 3 * h * inter * expert_weight_bytes / pool.moe_tp;
      const double hit = std::min(local, tokens * topk);
      r.compute_ms = flops_ms(dev, "fp4", tokens * topk * 3 * 2 * h * inter / pool.moe_ep / pool.moe_tp);
      r.membw_ms = bytes_ms(dev, hit * per_expert + tokens * topk * h * 3 / pool.moe_ep);
      return r;
    }
    case OpKind::Attention: {
      const bool prefill = op.query.op.find("context") != std::string::npos;
      const int cr = op.compress_ratio;
      const double nh = double(num_heads) / pool.tp, lg = std::max(1, o_groups / pool.tp);
      const double qlr = q_lora_rank, olr = o_lora_rank, hd = head_dim;
      const double b = batch, s = isl, kv_len = prefill ? step + isl : step;
      const double t = prefill ? b * s : b;
      const double cm = cr == 4 ? 2 : 1;
      double fp8_ops = 2 * t * (h * qlr + qlr * nh * hd + h * hd + lg * olr * h);
      double bf16_ops = 2 * t * nh * hd * olr;
      if (cr != 0) fp8_ops += 4 * t * h * cm * hd;
      if (cr == 4) fp8_ops += 4 * t * h * 2 * index_head_dim;
      const double avg_ctx = prefill ? step + s / 2 : kv_len;
      double pairs = prefill ? b * s * std::min<double>(sliding_window, avg_ctx) : b * std::min<double>(kv_len, sliding_window);
      if (cr == 4) pairs += t * std::min<double>(index_topk, avg_ctx / 4);
      else if (cr != 0) pairs += t * avg_ctx / cr;
      bf16_ops += 4 * nh * hd * pairs;
      double index_cache = 0;
      if (cr == 4) {
        const double ipairs = t * avg_ctx / 4;
        fp8_ops += 2 * t * qlr * index_n_heads * index_head_dim + 2 * ipairs * index_n_heads * index_head_dim;
        bf16_ops += 2 * t * h * index_n_heads;
        index_cache = b * (kv_len / 4) * index_head_dim * index_entry_bytes;
      }
      const double act = t * (h + qlr + nh * hd + hd + lg * olr) * gemm_weight_bytes;
      const double kv = pairs * (hd + rope_head_dim) * kv_entry_bytes;
      const double rope = t * nh * rope_head_dim * 2;
      r.compute_ms = flops_ms(dev, "fp8", fp8_ops) + flops_ms(dev, "bf16", bf16_ops);
      r.membw_ms = bytes_ms(dev, attention_weight_bytes(cr, pool.tp) + act + kv + index_cache + rope);
      return r;
    }
  }
  return r;
}

}  // namespace dlsim::model
