"""Markdown tables from `dlsim validate-points` reports: per-point relative errors, per-curve medians, cross-system ratios.
usage: summarize_validation.py report.json [report.json ...]"""
import json, statistics, sys

METRICS = ["tput_per_gpu", "p90_e2e_norm_intvty", "p90_intvty", "median_ttft"]
points, pairs = [], []
for f in sys.argv[1:]:
    d = json.load(open(f))
    points += d["report"]["points"]
    pairs += d["ratios"]["pairs"]

print("| id | hardware | framework | conc | wall s | " + " | ".join(METRICS) + " |")
print("|" + " --- |" * (5 + len(METRICS)))
for p in sorted(points, key=lambda p: (p["hardware"], p["framework"], p["conc"])):
    e = p["rel_err"]
    cells = [f"{e[m]:+.0%}" if e.get(m) is not None else "n/a" for m in METRICS]
    print(f"| {p['id']} | {p['hardware']} | {p['framework']} | {p['conc']} | {p['wall_s']:.0f} | " + " | ".join(cells) + " |")

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
    print("| key | metric | a | b | pred ratio | meas ratio | err |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for q in pairs:
        print(f"| {q['key']} | {q['metric']} | {q['a']} | {q['b']} | {q['pred_ratio']:.3f} | {q['meas_ratio']:.3f} | {q['ratio_err']:+.0%} |")
