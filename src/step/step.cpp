#include "step/step.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

#include "fabric/fabric.h"
#include "model/model.h"

namespace dlsim {

int64_t MemoryPlan::kv_capacity_tokens(const MemoryTier& tier, const DeviceSpec& d) const {
  double avail = tier.capacity_bytes * tier.usable_fraction;
  if (!d.memory.empty() && tier.name == d.memory[0].name)
    avail -= weight_bytes_per_gpu + d.activation_reserve_bytes + d.comm_reserve_bytes;
  if (kv_bytes_per_token <= 0 || avail <= 0) return 0;
  return int64_t(std::floor(avail / kv_bytes_per_token));
}

namespace {

enum class Bin { Compute, Membw, Comm };

struct Acc {
  double compute = 0, membw = 0, comm = 0, lo = 0, hi = 0;
  Source source = Source::Measured;
  std::string note;
  std::map<std::string, double> per_op;
  void add(const Latency& l, Bin bin, const std::string& name) {
    per_op[name] += l.ms;
    (bin == Bin::Compute ? compute : bin == Bin::Membw ? membw : comm) += l.ms;
    lo += l.lo_ms;
    hi += l.hi_ms;
    if (l.source > source) note = l.note;
    source = weaker(source, l.source);
  }
  void merge(const Acc& o) {
    for (const auto& [k, v] : o.per_op) per_op[k] += v;
    compute += o.compute;
    membw += o.membw;
    comm += o.comm;
    lo += o.lo;
    hi += o.hi;
    if (o.source > source) note = o.note;
    source = weaker(source, o.source);
  }
  double ms() const { return compute + membw + comm; }
};

struct Pass {
  double rank_max;    // tokens on the busiest attention DP rank
  double total;       // tokens over all ranks
  double layer_reps;  // layers charged in this pass
};

class DeepSeekStep final : public StepLatency {
 public:
  DeepSeekStep(const model::ModelDesc& m, DeviceSpec dev, PoolSpec pool, StackSpec stack, FabricSpec fab,
               const OpLatencySource& ops)
      : m_(m), dev_(std::move(dev)), pool_(std::move(pool)), stack_(std::move(stack)), fab_(std::move(fab)), ops_(ops) {
    prefill_ops_ = m_.ops(pool_, stack_, Phase::Prefill);
    decode_ops_ = m_.ops(pool_, stack_, Phase::Decode);
    plan_.weight_bytes_per_gpu = m_.weight_bytes_per_gpu(pool_, stack_);
    plan_.kv_bytes_per_token = m_.kv_bytes_per_token();
  }

  const MemoryPlan& memory() const override { return plan_; }
  StepResult step(const StepInput& in) const override;

 private:
  Group group(int size) const {
    const int S = fab_.scaleup_domain.value_or(size);
    return {size, std::min(S, size)};
  }

  Bin bin_of(const model::Op& op, double tokens, double batch, double isl, double step) const {
    const model::Roofline r = m_.roofline(op, dev_, tokens, batch, isl, step, pool_);
    return r.compute_ms >= r.membw_ms ? Bin::Compute : Bin::Membw;
  }

  // Every table query carries the stack's framework; the version is resolved by the table source.
  Latency lookup(OpQuery q) const {
    if (!stack_.framework.empty() && !q.cat.count("framework")) q.cat["framework"] = stack_.framework;
    Latency l = ops_.query(dev_.name, q);
    if (l.source != Source::Unsupported || !q.cat.count("model")) return l;
    // Module tables label the same checkpoint and KV dtype differently per framework and system.
    std::vector<std::string> models = {q.cat.at("model")}, kvs = {q.cat.count("kv_cache_dtype") ? q.cat.at("kv_cache_dtype") : ""};
    models.insert(models.end(), m_.table_model_aliases.begin(), m_.table_model_aliases.end());
    kvs.insert(kvs.end(), m_.kv_cache_dtype_aliases.begin(), m_.kv_cache_dtype_aliases.end());
    for (const auto& mo : models)
      for (const auto& kv : kvs) {
        if (mo == models[0] && kv == kvs[0]) continue;
        OpQuery a = q;
        a.cat["model"] = mo;
        if (!kv.empty()) a.cat["kv_cache_dtype"] = kv;
        Latency r = ops_.query(dev_.name, a);
        if (r.source != Source::Unsupported) {
          r.note += " [model " + mo + ", kv_cache_dtype " + kv + "]";
          return r;
        }
      }
    return l;
  }

  void price_dense(const model::Op& op, double tokens, double reps, Acc& acc) const {
    if (tokens <= 0 || reps <= 0) return;
    OpQuery q = op.query;
    q.num[op.kind == model::OpKind::Gemm ? "m" : "num_tokens"] = tokens;
    Latency l = lookup(q);
    if (l.source == Source::Unsupported && q.op == "wideep_moe") {
      q.op = "moe";
      l = lookup(q);
    }
    if (l.source == Source::Unsupported && q.op == "moe" && q.cat.count("moe_dtype")) {
      // The measured MoE tables do not carry every dtype at the DSV4 shape (gb300 trtllm has w4a8_mxfp4_mxfp8 only);
      // use the nearest weight-width match and say so.
      const std::string wanted = q.cat.at("moe_dtype");
      for (const char* alt : {"w4a8_mxfp4_mxfp8", "fp8_block", "fp8"}) {
        if (alt == wanted) continue;
        q.cat["moe_dtype"] = alt;
        l = lookup(q);
        if (l.source != Source::Unsupported) {
          l.note += " [moe_dtype " + wanted + " unavailable, used " + alt + "]";
          break;
        }
      }
    }
    if (l.source == Source::Unsupported && op.kind == model::OpKind::Embedding)
      l = Latency::exact(m_.roofline(op, dev_, tokens, 0, 0, 0, pool_).membw_ms, Source::Roofline, "embedding closed form");
    acc.add(l * reps, bin_of(op, tokens, 0, 0, 0), op.name);
  }

  void price_attention(const model::Op& op, double batch, double isl, double step, double reps, Acc& acc) const {
    OpQuery q = op.query;
    q.num["batch_size"] = batch;
    q.num["isl"] = isl;
    q.num["step"] = step;
    acc.add(lookup(q) * reps, bin_of(op, batch * isl, batch, isl, step), op.name);
  }

  void price_comm(const OpQuery* measured, Collective c, int size, double bytes, double reps, Acc& acc) const {
    if (size <= 1 || bytes <= 0 || reps <= 0) return;
    Latency l = measured ? lookup(*measured) : Latency::unsupported("");
    if (l.source == Source::Unsupported) {
      const CommCost cc = collective_cost(fab_, c, group(size), bytes);
      l = Latency::exact(cc.ms, Source::Roofline, cc.cross_domain ? "closed-form cross-domain" : "closed-form");
    }
    acc.add(l * reps, Bin::Comm, measured ? measured->op + ":" + (measured->cat.count("op_name") ? measured->cat.at("op_name") : measured->cat.count("phase") ? measured->cat.at("phase") : "") : "p2p");
  }

  const model::ModelDesc& m_;
  DeviceSpec dev_;
  PoolSpec pool_;
  StackSpec stack_;
  FabricSpec fab_;
  const OpLatencySource& ops_;
  MemoryPlan plan_;
  std::vector<model::Op> prefill_ops_, decode_ops_;
};

StepResult DeepSeekStep::step(const StepInput& in) const {
  const double L = m_.layers, nextn = in.nextn, h = m_.hidden;
  const double fixed = stack_.t_step_fixed_ms;

  double P = 0, D = 0, T_max = 0, D_max = 0;
  for (const auto& rank : in.ranks) {
    double p = 0, d = 0;
    for (const auto& r : rank) (r.prefill ? p : d) += r.prefill ? double(r.new_tokens) : 1 + nextn;
    P += p;
    D += d;
    T_max = std::max(T_max, std::ceil(p / pool_.cp) + d);
    D_max = std::max(D_max, d);
  }
  const double T = P + D;
  StepResult res;
  res.fixed_ms = fixed;
  res.total_ms = res.lo_ms = res.hi_ms = fixed;
  if (T <= 0) return res;

  std::vector<Pass> passes = {{T_max, T, L}};
  if (nextn > 0 && D > 0) passes.push_back({D_max, D, nextn});

  Acc other, routed, shared;
  for (const auto& op : prefill_ops_) {
    if (op.kind == model::OpKind::Attention) continue;
    Acc& acc = op.routed ? routed : op.shared ? shared : other;
    for (const Pass& ps : passes) price_dense(op, op.all_ranks ? ps.total : ps.rank_max, op.layers * ps.layer_reps / L, acc);
  }

  Acc attn_best;
  double best = -1;
  for (const auto& rank : in.ranks) {
    Acc a;
    std::map<std::pair<int64_t, int64_t>, int> ctx;
    double gen_past = 0;
    int gen_n = 0;
    for (const auto& r : rank) {
      if (r.prefill) {
        if (r.new_tokens > 0) ctx[{(r.new_tokens + pool_.cp - 1) / pool_.cp, r.past_kv}]++;
      } else {
        gen_past += double(r.past_kv);
        gen_n++;
      }
    }
    for (const auto& op : prefill_ops_) {
      if (op.kind != model::OpKind::Attention) continue;
      for (const auto& [key, count] : ctx) price_attention(op, count, double(key.first), double(key.second), op.layers, a);
    }
    for (const auto& op : decode_ops_) {
      if (op.kind != model::OpKind::Attention) continue;
      // One lookup per rank at the full batch and the mean context: the tables are linear in kv length at a fixed
      // batch plus a per-batch constant, so splitting the batch by context would pay that constant several times.
      // MTP verify: the 1 + nextn query tokens of a sequence share its KV read, so the batch is the sequence count
      // (the tables have no query-length axis); the draft layers enter through the (L + nextn) / L layer scale.
      if (gen_n > 0) price_attention(op, gen_n, 1, std::round(gen_past / gen_n), op.layers * (L + nextn) / L, a);
    }
    if (a.ms() > best) {
      best = a.ms();
      attn_best = a;
    }
  }
  other.merge(attn_best);

  const int moe_group = pool_.moe_tp * pool_.moe_ep;
  for (const Pass& ps : passes) {
    if (pool_.tp > 1) {
      OpQuery q{"custom_allreduce", {{"num_gpus", double(pool_.tp)}, {"message_size", ps.rank_max * h}},
                {{"op_name", "all_reduce"}, {"allreduce_dtype", "bfloat16"}}};
      // The vllm and sglang tables carry an eager and a graph-capture lane; trtllm's has no such column.
      if (stack_.framework == "vllm" || stack_.framework == "sglang")
        q.cat["backend"] = stack_.framework + (stack_.graph_replay ? "_graph" : "_eager");
      // A version chain only falls back to older tables; when it lacks the group size (gb300 vllm 0.25.0 stops at
      // 4 GPUs) the next slot's measurement of the same kernel beats the closed form.
      if (lookup(q).source == Source::Unsupported) q.cat["version"] = "next";
      price_comm(&q, Collective::AllReduce, pool_.tp, ps.rank_max * h * 2, 2 * ps.layer_reps, other);
    }
    if (stack_.wide_ep && pool_.moe_ep > 1) {
      const int per_node = fab_.devices_per_node.value_or(moe_group);
      const double node_num = std::ceil(double(moe_group) / per_node);
      for (const char* phase : {"dispatch", "combine"}) {
        const bool dispatch = phase[0] == 'd';
        const double elem = dispatch ? 1 : 2;
        OpQuery q{"moe_a2a",
                  {{"ep_size", double(pool_.moe_ep)}, {"node_num", node_num}, {"hidden_size", h}, {"topk", double(m_.topk)},
                   {"num_experts", double(m_.num_experts)}, {"num_tokens", ps.rank_max}},
                  {{"comm_backend", P > 0 ? "trtllm_deepep_ht" : "trtllm_deepep_ll"}, {"phase", phase},
                   {"comm_dtype", dispatch ? "fp8" : "bfloat16"}}};
        for (const char* alt : {"nvfp4", "bfloat16"}) {   // the HT tables carry bfloat16 / nvfp4 payloads only
          if (lookup(q).source != Source::Unsupported) break;
          q.cat["comm_dtype"] = alt;
        }
        price_comm(&q, Collective::AllToAll, moe_group, ps.rank_max * m_.topk * h * elem, ps.layer_reps, routed);
      }
    } else if (pool_.attention_dp > 1) {
      for (const char* op_name : {"all_gather", "reduce_scatter"}) {
        OpQuery q{"nccl", {{"num_gpus", double(moe_group)}, {"message_size", ps.total * h}},
                  {{"op_name", op_name}, {"nccl_dtype", "half"}}};
        price_comm(&q, op_name[0] == 'a' ? Collective::AllGather : Collective::ReduceScatter, moe_group, ps.total * h * 2,
                   ps.layer_reps, routed);
      }
    }
    if (pool_.pp > 1) price_comm(nullptr, Collective::P2P, pool_.pp, ps.rank_max * h * 2, (pool_.pp - 1) * ps.layer_reps / L, other);
  }

  Acc all = other;
  if (P == 0) {
    all.merge(routed.ms() >= shared.ms() ? routed : shared);
  } else {
    all.merge(routed);
    all.merge(shared);
  }
  res.compute_ms = all.compute;
  res.membw_ms = all.membw;
  res.comm_ms = all.comm;
  const double work = all.compute + all.membw;
  res.total_ms = fixed + (stack_.overlap_comm ? std::max(work, all.comm) : work + all.comm);
  res.lo_ms = fixed + all.lo;
  res.hi_ms = fixed + all.hi;
  res.source = all.source;
  res.note = all.note;
  res.per_op = std::move(all.per_op);
  return res;
}

}  // namespace

std::unique_ptr<StepLatency> make_step_latency(const std::string& model, const DeviceSpec& dev, const PoolSpec& pool,
                                               const StackSpec& stack, const FabricSpec& fabric,
                                               const OpLatencySource& ops) {
  return std::make_unique<DeepSeekStep>(model::by_name(model), dev, pool, stack, fabric, ops);
}

}  // namespace dlsim
