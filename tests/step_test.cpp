#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "fabric/fabric.h"
#include "model/model.h"
#include "step/step.h"

using namespace dlsim;

namespace {

bool near(double a, double b, double rel = 1e-9) { return std::fabs(a - b) <= rel * std::max(1.0, std::fabs(b)); }

bool is_comm(const OpQuery& q) { return q.op == "nccl" || q.op == "custom_allreduce" || q.op == "moe_a2a"; }

struct Fake : OpLatencySource {
  mutable std::vector<OpQuery> log;
  std::function<Latency(const OpQuery&)> fn;
  Latency query(const std::string&, const OpQuery& q) const override {
    log.push_back(q);
    return fn(q);
  }
};

Latency shape_fn(const OpQuery& q) {
  if (is_comm(q)) return Latency::unsupported("no comm table");
  double ms = 0.01;
  if (q.op == "gemm") ms += 1e-3 * q.num.at("m");
  else if (q.op.find("context_module") != std::string::npos) ms += 1e-3 * q.num.at("batch_size") * (q.num.at("isl") + q.num.at("step") / 128 + 1);
  else if (q.op.find("generation_module") != std::string::npos) ms += 1e-3 * q.num.at("batch_size") * (1 + q.num.at("step") / 1024);
  else ms += 1e-3 * q.num.at("num_tokens");
  return Latency::exact(ms, Source::Measured);
}

DeviceSpec gb300() {
  DeviceSpec d;
  d.name = "gb300";
  d.flops = {{"bf16", 2.5e15}, {"fp8", 5e15}, {"fp4", 15e15}};
  d.memory = {{"hbm", 298013687808.0, 8e12, 0.9}};
  d.activation_reserve_bytes = 4e9;
  d.comm_reserve_bytes = 1e9;
  return d;
}

PoolSpec pool(int tp, int dp, int moe_tp, int moe_ep, int pp = 1) {
  PoolSpec p;
  p.device = "gb300";
  p.tp = tp;
  p.attention_dp = dp;
  p.moe_tp = moe_tp;
  p.moe_ep = moe_ep;
  p.ep = moe_ep;
  p.pp = pp;
  assert(p.tp * p.attention_dp * p.cp == p.moe_tp * p.moe_ep);
  return p;
}

StepRequest prefill(int64_t n, int64_t past = 0) { return {n, past, true}; }
StepRequest decode(int64_t past) { return {1, past, false}; }

const model::Op& find(const std::vector<model::Op>& v, const std::string& name) {
  for (const auto& o : v) if (o.name == name) return o;
  assert(false && "op not found");
  return v.front();
}

void test_ops_and_shapes() {
  const auto& m = model::deepseek_v4_pro();
  StackSpec st;
  auto ops = m.ops(pool(2, 2, 1, 4), st, Phase::Prefill);
  assert(ops.size() == 10);
  const auto& csa = find(ops, "csa_attention");
  assert(csa.layers == 30 && csa.compress_ratio == 4 && csa.query.op == "dsv4_csa_context_module");
  assert(csa.query.num.at("num_heads") == 64 && csa.query.num.at("tp_size") == 2 && csa.query.num.at("compress_ratio") == 4);
  assert(csa.query.cat.at("model") == "sgl-project/DeepSeek-V4-Pro-FP8");
  assert(csa.query.cat.at("architecture") == "DeepseekV4ForCausalLM");
  assert(csa.query.cat.at("mla_dtype") == "bfloat16" && csa.query.cat.at("kv_cache_dtype") == "fp8");
  assert(csa.query.cat.at("gemm_type") == "fp8_block");
  const auto& hca = find(ops, "hca_attention");
  assert(hca.layers == 31 && hca.compress_ratio == 128 && hca.query.op == "dsv4_hca_context_module");
  assert(hca.query.num.at("compress_ratio") == 128);
  const auto& pre = find(ops, "mhc_pre");
  assert(pre.layers == 61 && pre.query.op == "mhc_module" && pre.query.cat.at("op_name") == "pre");
  assert(pre.query.num.at("hc_mult") == 4 && pre.query.num.at("hidden_size") == 7168 && pre.query.num.at("sinkhorn_iters") == 20);
  assert(find(ops, "mhc_post").query.cat.at("op_name") == "post");
  const auto& gu = find(ops, "shared_gate_up_gemm");
  assert(gu.query.op == "gemm" && gu.query.num.at("n") == 3072 && gu.query.num.at("k") == 7168 && gu.query.cat.at("gemm_dtype") == "fp8_block");
  assert(gu.shared && gu.layers == 61);
  const auto& f2 = find(ops, "shared_ffn2_gemm");
  assert(f2.query.num.at("n") == 7168 && f2.query.num.at("k") == 1536);
  const auto& rt = find(ops, "router_gemm");
  assert(rt.query.num.at("n") == 384 && rt.query.num.at("k") == 7168 && rt.query.cat.at("gemm_dtype") == "bfloat16" && rt.routed);
  const auto& moe = find(ops, "moe");
  assert(moe.query.op == "moe" && moe.all_ranks && moe.routed && moe.layers == 61);
  assert(moe.query.num.at("moe_ep_size") == 4 && moe.query.num.at("moe_tp_size") == 1 && moe.query.num.at("num_experts") == 384);
  assert(moe.query.num.at("topk") == 6 && moe.query.num.at("inter_size") == 3072 && moe.query.num.at("hidden_size") == 7168);
  assert(moe.query.cat.at("moe_dtype") == "nvfp4" && moe.query.cat.at("distribution") == "power_law_1.01");
  const auto& lg = find(ops, "logits_gemm");
  assert(lg.layers == 1 && lg.query.num.at("n") == 64640 && lg.query.num.at("k") == 7168 && lg.query.cat.at("gemm_dtype") == "bfloat16");
  assert(find(ops, "embedding").query.op == "embedding");

  auto dec = m.ops(pool(2, 2, 1, 4), st, Phase::Decode);
  assert(find(dec, "csa_attention").query.op == "dsv4_csa_generation_module");
  assert(find(dec, "hca_attention").query.op == "dsv4_hca_generation_module");
  StackSpec wide = st;
  wide.wide_ep = true;
  assert(find(m.ops(pool(1, 1, 1, 1), wide, Phase::Decode), "moe").query.op == "wideep_moe");
}

void test_weights_and_kv() {
  const auto& m = model::deepseek_v4_pro();
  StackSpec st;
  const double params = m.param_count(false);
  std::printf("params (no MTP) = %.4e, with MTP = %.4e\n", params, m.param_count(true));
  assert(params > 1.55e12 && params < 1.65e12);

  const double w1 = m.weight_bytes_per_gpu(pool(1, 1, 1, 1), st);
  const double w8 = m.weight_bytes_per_gpu(pool(1, 8, 1, 8), st);
  const double experts = 61.0 * 384 * 3 * 7168 * 3072 * 9 / 16;
  std::printf("weights/GPU tp1 ep1 = %.4e B, tp1 ep8 = %.4e B, experts total = %.4e B\n", w1, w8, experts);
  assert(near(w1 - w8, experts * 7 / 8));
  const double w_tp2 = m.weight_bytes_per_gpu(pool(2, 1, 1, 2), st);
  const double shared_half = 61.0 * 3 * 7168 * 3072 / 2, logits_half = 129280.0 * 7168 * 2 / 2;
  const double attn_split = 30 * (m.attention_weight_bytes(4, 1) - m.attention_weight_bytes(4, 2)) +
                            31 * (m.attention_weight_bytes(128, 1) - m.attention_weight_bytes(128, 2));
  assert(near(w1 - w_tp2, experts / 2 + shared_half + logits_half + attn_split));
  StackSpec mtp = st;
  mtp.mtp_nextn = 1;
  assert(near(m.weight_bytes_per_gpu(pool(1, 1, 1, 1), mtp) - w1, m.layer_weight_bytes(0, pool(1, 1, 1, 1))));

  const double kv = m.kv_bytes_per_token();
  std::printf("kv bytes/token = %.1f, per sequence = %.0f\n", kv, m.kv_bytes_per_sequence(false));
  assert(near(kv, 30 * (576 + 128) / 4.0 + 31 * 576 / 128.0));
  assert(near(m.kv_bytes_per_sequence(false), 61.0 * 128 * 576));

  Fake f;
  f.fn = shape_fn;
  DeviceSpec d = gb300();
  auto sl = make_step_latency("deepseek-v4-pro", d, pool(1, 8, 1, 8), st, FabricSpec{}, f);
  const auto& plan = sl->memory();
  assert(near(plan.weight_bytes_per_gpu, w8) && near(plan.kv_bytes_per_token, kv));
  const double expect = std::floor((d.memory[0].capacity_bytes * 0.9 - w8 - 4e9 - 1e9) / kv);
  assert(plan.kv_capacity_tokens(d.memory[0], d) == int64_t(expect));
  MemoryTier host{"host", 1e12, 0, 0.5};
  assert(plan.kv_capacity_tokens(host, d) == int64_t(std::floor(5e11 / kv)));
  std::printf("kv capacity tokens on gb300 tp1 ep8 = %lld\n", (long long)plan.kv_capacity_tokens(d.memory[0], d));
}

void test_attention_max_moe_total() {
  Fake f;
  f.fn = [](const OpQuery& q) {
    if (is_comm(q)) return Latency::unsupported("");
    if (q.op.find("generation_module") != std::string::npos) return Latency::exact(q.num.at("batch_size"), Source::Measured);
    if (q.op == "moe") return Latency::exact(q.num.at("num_tokens"), Source::Measured);
    return Latency::exact(0, Source::Measured);
  };
  StackSpec st;
  auto sl = make_step_latency("deepseek-v4-pro", gb300(), pool(1, 2, 1, 2), st, FabricSpec{}, f);
  StepInput in;
  in.ranks = {{decode(1000), decode(1000), decode(1000)}, {decode(1000)}};
  auto r = sl->step(in);
  assert(near(r.total_ms, 3 * 61 + 4 * 61));
  bool moe4 = false, b3 = false, b1 = false;
  for (const auto& q : f.log) {
    if (q.op == "moe") moe4 = q.num.at("num_tokens") == 4;
    if (q.op == "dsv4_csa_generation_module") (q.num.at("batch_size") == 3 ? b3 : b1) = true;
  }
  assert(moe4 && b3 && b1);
}

// Generation attention of a rank: one lookup at the full batch and the mean context (no per-context split).
void test_decode_buckets() {
  Fake f;
  f.fn = shape_fn;
  StackSpec st;
  auto sl = make_step_latency("deepseek-v4-pro", gb300(), pool(1, 1, 1, 1), st, FabricSpec{}, f);
  StepInput in;
  in.ranks = {{decode(600), decode(900), decode(5000)}};
  sl->step(in);
  int n = 0;
  bool merged = false;
  for (const auto& q : f.log) {
    if (q.op != "dsv4_hca_generation_module") continue;
    ++n;
    if (q.num.at("batch_size") == 3 && q.num.at("step") == 2167 && q.num.at("isl") == 1) merged = true;
  }
  assert(n == 1 && merged);
}

void test_mtp() {
  Fake f;
  f.fn = [](const OpQuery& q) { return is_comm(q) ? Latency::unsupported("") : Latency::exact(1.0, Source::Measured); };
  StackSpec st;
  auto sl = make_step_latency("deepseek-v4-pro", gb300(), pool(1, 1, 1, 1), st, FabricSpec{}, f);
  StepInput in;
  in.ranks = {{decode(4096), decode(4096)}};
  const double t0 = sl->step(in).total_ms;
  assert(near(t0, 1 + 61 + 61 + 61 + 1 + 122));
  in.nextn = 1;
  f.log.clear();
  const double t1 = sl->step(in).total_ms;
  assert(near(t1, t0 * 62.0 / 61.0));
  // The verify tokens share the sequence's KV read: attention is queried per sequence, not per token.
  bool batch2 = false;
  for (const auto& q : f.log) if (q.op == "dsv4_csa_generation_module") batch2 = q.num.at("batch_size") == 2;
  assert(batch2);
}

FabricSpec nvlink() {
  FabricSpec f;
  f.scaleup.present = true;
  f.scaleup.bandwidth_Bps = 900e9;
  f.scaleup.alpha_s = 2e-6;
  return f;
}

void test_comm() {
  const double h = 7168;
  Fake f;
  f.fn = [](const OpQuery& q) { return is_comm(q) ? Latency::unsupported("") : Latency::exact(0, Source::Measured); };
  StackSpec st;
  FabricSpec fab = nvlink();
  StepInput in;
  in.ranks = {{prefill(1024)}};

  auto r = make_step_latency("deepseek-v4-pro", gb300(), pool(2, 1, 2, 1), st, fab, f)->step(in);
  const double ar = collective_cost(fab, Collective::AllReduce, Group{2, 2}, 1024 * h * 2).ms;
  assert(ar > 0 && near(r.comm_ms, 2 * 61 * ar) && near(r.total_ms, r.comm_ms));
  assert(r.source == Source::Roofline);

  in.ranks = {{prefill(1024)}, {prefill(512)}};
  r = make_step_latency("deepseek-v4-pro", gb300(), pool(1, 2, 1, 2), st, fab, f)->step(in);
  const double ag = collective_cost(fab, Collective::AllGather, Group{2, 2}, 1536 * h * 2).ms;
  const double rs = collective_cost(fab, Collective::ReduceScatter, Group{2, 2}, 1536 * h * 2).ms;
  assert(near(r.comm_ms, 61 * (ag + rs)));
  bool ag_seen = false;
  for (const auto& q : f.log)
    if (q.op == "nccl" && q.cat.at("op_name") == "all_gather") ag_seen = q.num.at("message_size") == 1536 * h && q.num.at("num_gpus") == 2;
  assert(ag_seen);

  StackSpec wide = st;
  wide.wide_ep = true;
  f.log.clear();
  r = make_step_latency("deepseek-v4-pro", gb300(), pool(1, 2, 1, 2), wide, fab, f)->step(in);
  const double disp = collective_cost(fab, Collective::AllToAll, Group{2, 2}, 1024 * 6 * h * 1).ms;
  const double comb = collective_cost(fab, Collective::AllToAll, Group{2, 2}, 1024 * 6 * h * 2).ms;
  assert(near(r.comm_ms, 61 * (disp + comb)));
  bool a2a_seen = false;
  for (const auto& q : f.log)
    if (q.op == "moe_a2a" && q.cat.at("phase") == "dispatch")
      a2a_seen = a2a_seen || (q.num.at("num_tokens") == 1024 && q.num.at("ep_size") == 2 && q.num.at("node_num") == 1 &&
                              q.cat.at("comm_backend") == "trtllm_deepep_ht" && q.cat.at("comm_dtype") == "fp8");
  assert(a2a_seen);

  Fake measured;
  measured.fn = [](const OpQuery& q) {
    if (q.op == "custom_allreduce") return Latency::exact(0.5, Source::Measured);
    return is_comm(q) ? Latency::unsupported("") : Latency::exact(0, Source::Measured);
  };
  in.ranks = {{prefill(1024)}};
  r = make_step_latency("deepseek-v4-pro", gb300(), pool(2, 1, 2, 1), st, fab, measured)->step(in);
  assert(near(r.comm_ms, 2 * 61 * 0.5) && r.source == Source::Measured);

  r = make_step_latency("deepseek-v4-pro", gb300(), pool(1, 1, 1, 1, 2), st, fab, f)->step(in);
  assert(near(r.comm_ms, collective_cost(fab, Collective::P2P, Group{2, 2}, 1024 * h * 2).ms));

  FabricSpec split = fab;
  split.scaleup_domain = 1;
  split.scaleout.present = true;
  split.scaleout.bandwidth_Bps = 100e9;
  split.scaleout.alpha_s = 5e-6;
  r = make_step_latency("deepseek-v4-pro", gb300(), pool(2, 1, 2, 1), st, split, f)->step(in);
  assert(near(r.comm_ms, 2 * 61 * collective_cost(split, Collective::AllReduce, Group{2, 1}, 1024 * h * 2).ms));

  StackSpec ov = st;
  ov.overlap_comm = true;
  ov.t_step_fixed_ms = 0.1;
  Fake one;
  one.fn = [](const OpQuery& q) { return is_comm(q) ? Latency::unsupported("") : Latency::exact(1.0, Source::Measured); };
  r = make_step_latency("deepseek-v4-pro", gb300(), pool(2, 1, 2, 1), ov, fab, one)->step(in);
  assert(near(r.fixed_ms, 0.1) && near(r.total_ms, 0.1 + std::max(r.compute_ms + r.membw_ms, r.comm_ms)));
  assert(r.comm_ms > 0 && r.compute_ms + r.membw_ms > r.comm_ms);
}

void test_prefill_monotone_and_bins() {
  Fake f;
  f.fn = shape_fn;
  StackSpec st;
  auto sl = make_step_latency("deepseek-v4-pro", gb300(), pool(1, 1, 1, 1), st, FabricSpec{}, f);
  StepInput a, b, c;
  a.ranks = {{prefill(1000)}};
  b.ranks = {{prefill(2000)}};
  c.ranks = {{prefill(1000, 50000)}};
  const double ta = sl->step(a).total_ms, tb = sl->step(b).total_ms, tc = sl->step(c).total_ms;
  assert(tb > ta && tc > ta);

  Fake moe_only;
  moe_only.fn = [](const OpQuery& q) { return Latency::exact(q.op == "moe" ? 1.0 : 0.0, Source::Measured); };
  auto sm = make_step_latency("deepseek-v4-pro", gb300(), pool(1, 1, 1, 1), st, FabricSpec{}, moe_only);
  StepInput d;
  d.ranks = {{decode(4096), decode(4096), decode(4096), decode(4096)}};
  auto r = sm->step(d);
  assert(near(r.membw_ms, 61) && r.compute_ms == 0);
  Fake gemm_only;
  gemm_only.fn = [](const OpQuery& q) {
    return Latency::exact(q.op == "gemm" && q.num.at("n") == 6144 ? 1.0 : 0.0, Source::Measured);
  };
  auto sg = make_step_latency("deepseek-v4-pro", gb300(), pool(1, 1, 1, 1), st, FabricSpec{}, gemm_only);
  StepInput p;
  p.ranks = {{prefill(16384)}};
  r = sg->step(p);
  assert(near(r.compute_ms, 61) && r.membw_ms == 0);
}

}  // namespace

int main() {
  test_ops_and_shapes();
  test_weights_and_kv();
  test_attention_max_moe_total();
  test_decode_buckets();
  test_mtp();
  test_comm();
  test_prefill_monotone_and_bins();
  std::printf("step_test ok\n");
  return 0;
}
