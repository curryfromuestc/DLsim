"""Generate configs/points/*.yaml from the InferenceX benchmarks snapshot. Only topology fields are read from the snapshot;
metrics are never touched. Engine limits (batch caps, chunk sizes, draft length) come from the InferenceX recipe that
produced the row: the srt-slurm recipe yaml for gb200/gb300/h200, the single-node launch script rules for b200/b300."""
import glob, json, os, re, sys

import yaml

snap, out = sys.argv[1], sys.argv[2]
RECIPES = "third_party/inferencex/benchmarks/multi_node/srt-slurm-recipes/dsv4"
rows = json.load(open(snap))
hw_map = {"gb300": "gb300", "gb200": "gb200", "b300": "b300_sxm", "b200": "b200_sxm", "h200": "h200_sxm"}
fabric_map = {"gb300": "nvl72", "gb200": "nvl72", "b300": "nvl8_ib", "b200": "nvl8_ib", "h200": "nvl8_ib"}
recipe_dir = {("gb300", "dynamo-trt"): "trtllm/gb300-fp4", ("gb300", "dynamo-sglang"): "sglang/gb300-fp4",
              ("gb300", "dynamo-vllm"): "vllm/gb300-fp4", ("gb200", "dynamo-vllm"): "vllm/gb200-fp4",
              ("h200", "dynamo-sglang"): "sglang/h200-fp8"}


# Frozen per-curve calibration (validation.md 冻结清单): matched on framework family and deployment, exact decode tp first.
CAL = yaml.safe_load(open("configs/calibration.yaml")) if os.path.exists("configs/calibration.yaml") else []


def calibration_for(r, m):
    fam = r["framework"].replace("dynamo-", "")
    same = [c for c in CAL if c["framework"] == fam and bool(c["disagg"]) == bool(r["disagg"])]
    exact = [c for c in same if c.get("decode_tp") == m["decode"]["tp"]]
    return (exact or same or [None])[0]


def pool(dev, workers, tp, ep, dpa):
    tp = int(tp or 1); ep = int(ep or 1); workers = int(workers or 1)
    if dpa:
        # attention DP spans the worker's GPUs; rows write that size in tp, some (vllm prefill dep4) leave tp=1 and carry it in ep
        n = max(tp, ep)
        return dict(device=dev, workers=workers, tp=1, attention_dp=n, moe_tp=max(1, n // ep), moe_ep=ep)
    return dict(device=dev, workers=workers, tp=tp, attention_dp=1, moe_tp=max(1, tp // ep), moe_ep=ep)


def role_limits(fw, args):
    """Per-rank sequence cap, per-pass token budget and draft length of one recipe role."""
    a = args or {}
    if fw == "trtllm":
        spec = a.get("speculative_config") or {}
        return a.get("max_batch_size"), a.get("max_num_tokens"), spec.get("max_draft_len")
    if fw == "sglang":
        n = a.get("speculative-num-draft-tokens")
        return a.get("max-running-requests"), a.get("chunked-prefill-size"), (int(n) - 1 if n else None)
    spec = a.get("speculative-config")
    n = json.loads(spec).get("num_speculative_tokens") if isinstance(spec, str) else None
    return a.get("max-num-seqs"), a.get("max-num-batched-tokens"), n


def load_recipes(sub):
    fw = sub.split("/")[0]
    out = []
    for f in sorted(glob.glob(f"{RECIPES}/{sub}/agentx/*.yaml")):
        y = yaml.safe_load(open(f))
        roles = y.get("roles") or {}
        per_node = int((y.get("resources") or {}).get("gpus_per_node") or 4)
        m = re.search(r"-c(\d+)(?:-|\.yaml)", os.path.basename(f))
        r = {"file": os.path.basename(f), "conc": int(m.group(1)) if m else None, "disagg": "decode" in roles}
        for role, spec in roles.items():
            # gpus is per worker; a recipe without it spans nodes * gpus_per_node
            gpus = int(spec.get("gpus") or 0) * int(spec.get("workers") or 1) or int(spec.get("nodes") or 0) * per_node
            r[role] = dict(gpus=gpus, workers=int(spec.get("workers") or 1), limits=role_limits(fw, spec.get("args")))
        out.append(r)
    return out


recipes = {}


def overrides_for(row, m):
    hw, fw = row["hardware"], row["framework"]
    conc = int(row["conc"])
    ov = {}
    if hw in ("b200", "b300"):
        # benchmarks/single_node/agentic/dsv4_fp4_{hw}_{fw}_mtp.sh: max running = 2*CONC (split across DEP ranks),
        # sglang chunked prefill 8192 (b200 DEP: 6144 per rank), vllm batched tokens 8192 (b300 DEP: 16384), MTP draft 3 / 6.
        dp = m["decode"]["attention_dp"]
        if fw == "sglang":
            ov = {"max_num_seqs": max(1, 2 * conc // dp), "chunk_tokens": 6144 if (hw == "b200" and dp > 1) else 8192, "mtp_nextn": 6}
            ov["max_num_batched_tokens"] = ov["chunk_tokens"]
        elif fw == "vllm":
            ov = {"max_num_seqs": max(1, 2 * conc // dp), "max_num_batched_tokens": 16384 if (hw == "b300" and dp > 1) else 8192, "mtp_nextn": 3}
            ov["chunk_tokens"] = ov["max_num_batched_tokens"]
        return ov, None
    sub = recipe_dir.get((hw, fw))
    if not sub:
        return ov, None
    if sub not in recipes:
        recipes[sub] = load_recipes(sub)
    disagg = bool(row["disagg"])
    gd = m["decode"]["workers"] * m["decode"]["tp"] * m["decode"]["attention_dp"]
    gp = m["prefill"]["workers"] * m["prefill"]["tp"] * m["prefill"]["attention_dp"] if disagg else 0
    cands = []
    for r in recipes[sub]:
        if r["disagg"] != disagg:
            continue
        dec = r.get("decode") or r.get("agg")
        if dec is None or dec["gpus"] != gd or dec["workers"] != m["decode"]["workers"]:
            continue
        if disagg and r["prefill"]["gpus"] != gp:
            continue
        cands.append(r)
    exact = [r for r in cands if r["conc"] == conc]
    untagged = [r for r in cands if r["conc"] is None]
    pick = exact[0] if exact else untagged[0] if untagged else None
    if pick is None:
        return ov, None
    fwname = sub.split("/")[0]
    dec = pick.get("decode") or pick.get("agg")
    seqs, toks, nextn = dec["limits"]
    if seqs is not None: ov["max_num_seqs"] = int(seqs)
    if toks is not None: ov["max_num_batched_tokens"] = int(toks)
    if nextn is not None: ov["mtp_nextn"] = int(nextn)
    if disagg:
        pseqs, ptoks, _ = pick["prefill"]["limits"]
        if pseqs is not None: ov["prefill_max_seqs"] = int(pseqs)
        if ptoks is not None:
            ov["prefill_max_tokens"] = int(ptoks)
            ov["chunk_tokens"] = int(ptoks)
    elif toks is not None:
        ov["chunk_tokens"] = int(toks)
    return ov, pick["file"]


n = 0
for r in rows:
    if r["benchmark_type"] != "agentic_traces" or r["hardware"] not in hw_map:
        continue
    dev = hw_map[r["hardware"]]
    m = {"disaggregated": bool(r["disagg"]), "routing": "kv_aware", "decode": pool(dev, r["decode_num_workers"], r["decode_tp"], r["decode_ep"], r["decode_dp_attention"])}
    if r["disagg"]:
        m["prefill"] = pool(dev, r["prefill_num_workers"], r["prefill_tp"], r["prefill_ep"], r["prefill_dp_attention"])
    gp = m["prefill"]["workers"] * m["prefill"]["tp"] * m["prefill"]["attention_dp"] if r["disagg"] else 0
    gd = m["decode"]["workers"] * m["decode"]["tp"] * m["decode"]["attention_dp"]
    ov, recipe = overrides_for(r, m)
    cal = calibration_for(r, m)
    if cal:
        ov.update({k: cal[k] for k in ("t_step_fixed_ms", "mtp_accept_mean", "request_overhead_ms")})
    doc = {
        "id": int(r["id"]), "hardware": r["hardware"], "framework": r["framework"], "precision": r["precision"],
        "spec_method": r["spec_method"], "offload_mode": r.get("offload_mode"), "is_multinode": bool(r["is_multinode"]),
        "num_prefill_gpu_row": r.get("num_prefill_gpu"), "num_decode_gpu_row": r.get("num_decode_gpu"),
        "num_prefill_gpu": gp if r["disagg"] else None, "num_decode_gpu": gd,
        "device": f"configs/device/{r['hardware']}.yaml", "fabric": f"configs/fabric/{fabric_map[r['hardware']]}.yaml",
        "stack": f"configs/stack/{r['framework']}.yaml" if os.path.exists(f"configs/stack/{r['framework']}.yaml") else None,
        "recipe": recipe, "calibration_anchor": cal["anchor"] if cal else None, "stack_overrides": ov,
        "mapping": m, "run": {"concurrency": int(r["conc"]), "duration_s": 3600, "seed": 42},
    }
    def y(v, ind=0):
        pad = "  " * ind
        if isinstance(v, dict):
            if not v: return f"{pad}{{}}" if ind else "{}"
            return "\n".join(f"{pad}{k}:" + (("\n" + y(val, ind + 1)) if isinstance(val, dict) and val else f" {y(val)}") for k, val in v.items())
        if v is None: return "null"
        if isinstance(v, bool): return "true" if v else "false"
        return str(v)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{r['hardware']}_{r['framework']}_{r['id']}.yaml"), "w") as f:
        f.write(y(doc) + "\n")
    n += 1
print("points written", n)
