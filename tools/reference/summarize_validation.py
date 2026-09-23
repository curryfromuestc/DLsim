"""Markdown tables from `dlsim validate-points` reports: per-point relative errors, per-curve medians, cross-system ratios.
usage: summarize_validation.py [--snapshot benchmarks.json] [--derived derived.json] [--points-dir configs/points] report.json ...
With --snapshot, cross-hardware ratio pairs are formed across all given reports (validate-points only pairs points
of one run) on the same key it uses: framework, spec, disagg, offload, prefill and decode mapping, concurrency."""
import json, os, statistics, sys

import yaml

METRICS = ["tput_per_gpu", "p90_e2e_norm_intvty", "p90_intvty", "median_ttft"]
args = sys.argv[1:]
snapshot, derived, points_dir = None, None, "configs/points"
while args and args[0].startswith("--"):
    flag = args.pop(0)
    if flag == "--snapshot": snapshot = args.pop(0)
    elif flag == "--derived": derived = args.pop(0)
    elif flag == "--points-dir": points_dir = args.pop(0)
    else: sys.exit("unknown flag " + flag)
points, pairs = [], []
for f in args:
    d = json.load(open(f))
    points += d["report"]["points"]
    pairs += d["ratios"]["pairs"]

if snapshot:
    rows = {r["id"]: r for r in json.load(open(snapshot))}
    for k, v in (json.load(open(derived)).items() if derived else []):
        if k in rows: rows[k]["metrics"].update({m: x for m, x in v.items() if m != "id" and x is not None})
    groups = {}
    for p in points:
        if p.get("oversized_requests", 0) > 0: continue
        y = yaml.safe_load(open(os.path.join(points_dir, f"{p['hardware']}_{p['framework']}_{p['id']}.yaml")))
        m = y["mapping"]
        pf = m.get("prefill") or {"workers": 0, "tp": 0, "attention_dp": 0, "moe_ep": 0}
        de = m["decode"]
        key = "|".join(str(x) for x in (p["framework"], y["spec_method"], int(m["disaggregated"]), y["offload_mode"],
              f"{pf['workers']}x{pf['tp']}.{pf['attention_dp']}.{pf['moe_ep']}", f"{de['workers']}x{de['tp']}.{de['attention_dp']}.{de['moe_ep']}", p["conc"]))
        groups.setdefault(key, []).append(p)
    pairs = []
    for key, ps in groups.items():
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                a, b = ps[i], ps[j]
                if a["hardware"] == b["hardware"]: continue
                ra, rb = rows[str(a["id"])]["metrics"], rows[str(b["id"])]["metrics"]
                for mk in METRICS:
                    if mk not in a["metrics"] or mk not in b["metrics"] or mk not in ra or mk not in rb: continue
                    pred, meas = a["metrics"][mk] / b["metrics"][mk], ra[mk] / rb[mk]
                    pairs.append({"key": key, "metric": mk, "a": a["id"], "b": b["id"], "pred_ratio": pred, "meas_ratio": meas, "ratio_err": pred / meas - 1})

print("| id | hardware | framework | conc | wall s | oversized | " + " | ".join(METRICS) + " |")
print("|" + " --- |" * (6 + len(METRICS)))
for p in sorted(points, key=lambda p: (p["hardware"], p["framework"], p["conc"])):
    e = p["rel_err"]
    cells = [f"{e[m]:+.0%}" if e.get(m) is not None else "n/a" for m in METRICS]
    print(f"| {p['id']} | {p['hardware']} | {p['framework']} | {p['conc']} | {p['wall_s']:.0f} | {p.get('oversized_requests', 0):.0f} | " + " | ".join(cells) + " |")

# A point with oversized requests ran short of KV capacity: its metrics cover a subset and it is not compared.
short = [p for p in points if p.get("oversized_requests", 0) > 0]
points = [p for p in points if p.get("oversized_requests", 0) == 0]
if short:
    print("\ncapacity-short points excluded from medians: " + ", ".join(str(p["id"]) for p in short))

print("\n| group | points | " + " | ".join("median abs " + m for m in METRICS) + " |")
print("|" + " --- |" * (2 + len(METRICS)))
groups = {}
for p in points:
    groups.setdefault((p["hardware"], p["framework"]), []).append(p)
groups[("all", "")] = points
for (hw, fw), ps in groups.items():
    meds = []
    for m in METRICS:
        v = [abs(p["rel_err"][m]) for p in ps if p["rel_err"].get(m) is not None]
        meds.append(f"{statistics.median(v):.0%}" if v else "n/a")
    print(f"| {hw} {fw} | {len(ps)} | " + " | ".join(meds) + " |")

if pairs:
    errs = [abs(q["ratio_err"]) for q in pairs if q["ratio_err"] is not None]
    print(f"\ncross-system ratio pairs {len(pairs)}, median abs ratio error {statistics.median(errs):.0%}")
    for mk in METRICS:
        v = [abs(q["ratio_err"]) for q in pairs if q["metric"] == mk]
        agree = sum(1 for q in pairs if q["metric"] == mk and (q["pred_ratio"] >= 1) == (q["meas_ratio"] >= 1))
        if v: print(f"  {mk}: {len(v)} pairs, median abs ratio error {statistics.median(v):.0%}, ordering agrees {agree}/{len(v)}")
    print("| key | metric | a | b | pred ratio | meas ratio | err |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for q in sorted(pairs, key=lambda q: (q["key"], q["metric"])):
        print(f"| {q['key']} | {q['metric']} | {q['a']} | {q['b']} | {q['pred_ratio']:.3f} | {q['meas_ratio']:.3f} | {q['ratio_err']:+.0%} |")
