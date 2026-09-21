#!/usr/bin/env python3
"""Compare two graph dumps (JSON lines from dump_graph_json and weka_graph_dump.py).

Traces are matched by id. Floats are compared with an absolute tolerance.
Prints every difference and exits 1 when any exists.
"""

from __future__ import annotations

import argparse
import json
import sys


def load(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["id"]] = d
    return out


def same_req(a, b, tol: float) -> bool:
    return abs(a[0] - b[0]) <= tol and a[1:] == b[1:]


def compare_trace(a: dict, b: dict, tol: float, report: list[str]) -> None:
    tid = a["id"]
    agents_a = {x["sid"]: x for x in a["agents"]}
    agents_b = {x["sid"]: x for x in b["agents"]}
    for sid in sorted(set(agents_a) | set(agents_b)):
        if sid not in agents_a or sid not in agents_b:
            report.append(f"{tid}: agent {sid} only in {'A' if sid in agents_a else 'B'}")
            continue
        x, y = agents_a[sid], agents_b[sid]
        if x["background"] != y["background"]:
            report.append(f"{tid}: agent {sid} background {x['background']} vs {y['background']}")
        if len(x["requests"]) != len(y["requests"]):
            report.append(f"{tid}: agent {sid} request count {len(x['requests'])} vs {len(y['requests'])}")
            continue
        for k, (p, q) in enumerate(zip(x["requests"], y["requests"])):
            if not same_req(p, q, tol):
                report.append(f"{tid}: agent {sid} request {k}: {p} vs {q}")
    key = lambda e: (e[0], e[1], e[2], e[3], e[4])
    ea = {key(e): e[5] for e in a["edges"]}
    eb = {key(e): e[5] for e in b["edges"]}
    for k in sorted(set(ea) | set(eb)):
        if k not in ea or k not in eb:
            report.append(f"{tid}: edge {k} only in {'A' if k in ea else 'B'}")
        elif abs(ea[k] - eb[k]) > tol:
            report.append(f"{tid}: edge {k} delay {ea[k]} vs {eb[k]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    report: list[str] = []
    for tid in sorted(set(a) | set(b)):
        if tid not in a or tid not in b:
            report.append(f"{tid}: trace only in {'A' if tid in a else 'B'}")
        else:
            compare_trace(a[tid], b[tid], args.tol, report)
    common = sorted(set(a) & set(b))
    n_agents = sum(len(a[t]["agents"]) for t in common)
    n_edges = sum(len(a[t]["edges"]) for t in common)
    n_reqs = sum(len(x["requests"]) for t in common for x in a[t]["agents"])
    print(f"traces compared: {len(common)} (A={len(a)}, B={len(b)}); agents={n_agents} requests={n_reqs} edges={n_edges}; differences={len(report)}")
    for line in report[:200]:
        print(line)
    if len(report) > 200:
        print(f"... {len(report) - 200} more")
    return 1 if report else 0


if __name__ == "__main__":
    sys.exit(main())
