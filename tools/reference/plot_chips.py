"""Redraw the per-chip measured-vs-predicted figures.

Reads the previous source_points.csv and overlays predictions from a newer
validate-points report (same measured values). One PDF, two pages per group.
"""
import csv, json, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

METRICS = [
    ("tput_per_gpu", "每 GPU 总吞吐", 1e-3, "千 token/s/GPU"),
    ("median_ttft", "首字等待时间", 1.0, "秒"),
    ("p90_e2e_norm_intvty", "含首字等待的速度", 1.0, "token/s"),
    ("p90_intvty", "后续输出速度", 1.0, "token/s"),
]


def load_csv(path):
    pts = {}
    for r in csv.DictReader(open(path)):
        p = pts.setdefault(int(r["id"]), {"meta": r, "m": {}, "p": {}})
        pred = r["predicted"]
        p["m"][r["metric"]] = float(r["measured"])
        p["p"][r["metric"]] = float(pred) if pred not in ("", "nan", "None") else None
        p["invalid"] = r["invalid"] == "True" or float(r.get("oversized_requests") or 0) > 0
    return pts


def overlay(pts, report):
    for row in json.load(open(report))["report"]["points"]:
        p = pts.get(row["id"])
        if p is None:
            continue
        p["invalid"] = row.get("oversized_requests", 0) > 0
        for k, _, _, _ in METRICS:
            if k in row["metrics"]:
                p["p"][k] = None if p["invalid"] else row["metrics"][k]


def draw(pts, manifest, out):
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    pdf = PdfPages(out)
    for g in manifest:
        rows = [pts[i] for i in g["points"] if i in pts]
        rows.sort(key=lambda r: int(r["meta"]["concurrency"]))
        xs = [int(r["meta"]["concurrency"]) for r in rows]
        title = f"{g['hardware'].upper()} · {g['framework']} · {g['topology']}"
        fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4))
        for ax, (key, name, scale, unit) in zip(axes.ravel(), METRICS):
            meas = [r["m"][key] * scale for r in rows]
            pred = [None if r["p"][key] is None else r["p"][key] * scale for r in rows]
            ax.plot(xs, meas, "o-", color="#4c4c4c", label="实测")
            px = [x for x, y in zip(xs, pred) if y is not None]
            py = [y for y in pred if y is not None]
            if px:
                ax.plot(px, py, "o--", color="#3b6ea5", label="预测")
            ax.set_xscale("log")
            ax.set_title(name)
            ax.set_xlabel("并发会话数")
            ax.set_ylabel(unit)
            ax.legend(frameon=False)
        fig.suptitle(title)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10.6, 7.4))
        mx = [r["m"]["p90_e2e_norm_intvty"] for r in rows]
        my = [r["m"]["tput_per_gpu"] / 1e3 for r in rows]
        ax.plot(mx, my, "o-", color="#4c4c4c", label="实测")
        px, py = [], []
        for r in rows:
            if r["p"]["p90_e2e_norm_intvty"] is None:
                continue
            px.append(r["p"]["p90_e2e_norm_intvty"])
            py.append(r["p"]["tput_per_gpu"] / 1e3)
        if px:
            ax.plot(px, py, "o--", color="#3b6ea5", label="预测")
        ax.set_xlabel("含首字等待的速度（token/s）")
        ax.set_ylabel("每 GPU 总吞吐（千 token/s/GPU）")
        ax.set_title(title)
        ax.legend(frameon=False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()


if __name__ == "__main__":
    csv_path, manifest_path, report, out = sys.argv[1:]
    pts = load_csv(csv_path)
    overlay(pts, report)
    draw(pts, json.load(open(manifest_path)), out)
    print("wrote", out)
