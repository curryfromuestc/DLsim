#!/usr/bin/env python3
"""Reference dump of the agentx-harness WekaTraceLoader session graph.

Runs the harness loader (third_party/agentx-harness, read-only) on a Weka trace
file, a directory of trace files, or the first N lines of a traces.jsonl, and
prints one canonical JSON object per trace, with the same shape as
dlsim::trace::dump_graph_json:

  {"id": ..., "agents": [{"sid", "background", "requests": [[t, in, out, hash_count, model], ...]}, ...],
   "edges": [[from_sid, from_k, to_sid, to_k, kind, delay_s], ...]}

Agents are sorted by sid, edges lexicographically. Requires the harness to be
importable (pip install -e third_party/agentx-harness into a venv; see report).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
from unittest.mock import MagicMock

HARNESS = pathlib.Path(__file__).resolve().parents[2] / "third_party" / "agentx-harness"


def _delay_s(from_turn, to_turn) -> float:
    api_s = (from_turn.api_time_ms or 0.0) / 1000.0
    return max(0.0, (to_turn.timestamp - from_turn.timestamp) / 1000.0 - api_s)


def _source_request(trace, turn):
    src = trace.requests[turn.source_outer_idx]
    if turn.source_inner_idx is not None:
        src = src.requests[turn.source_inner_idx]
    return src


def dump_trace(trace, root, children) -> dict:
    convs = {root.session_id: root}
    convs.update({c.session_id: c for c in children})
    branches = {b.branch_id: b for b in root.branches}
    background = {sid: False for sid in convs}
    for b in root.branches:
        for cid in b.child_conversation_ids:
            background[cid] = b.is_background

    agents = []
    for sid in sorted(convs):
        reqs = []
        for turn in convs[sid].turns:
            src = _source_request(trace, turn)
            reqs.append([turn.timestamp / 1000.0, src.input_length, src.output_length, len(src.hash_ids), src.model])
        agents.append({"sid": sid, "background": background[sid], "requests": reqs})

    edges = []
    for sid, conv in convs.items():
        for k in range(1, len(conv.turns)):
            edges.append([sid, k - 1, sid, k, "seq", conv.turns[k].delay / 1000.0])
    for i, turn in enumerate(root.turns):
        for bid in turn.branch_ids:
            b = branches[bid]
            for cid in b.child_conversation_ids:
                child = convs[cid]
                if not child.turns:
                    continue
                kind = "background" if b.is_background else "spawn"
                edges.append([root.session_id, i, cid, 0, kind, _delay_s(turn, child.turns[0])])
        for prereq in turn.prerequisites:
            b = branches[prereq.branch_id]
            for cid in b.child_conversation_ids:
                child = convs[cid]
                if not child.turns:
                    continue
                edges.append([cid, len(child.turns) - 1, root.session_id, i, "join", _delay_s(child.turns[-1], turn)])
    edges.sort(key=lambda e: (e[0], e[1], e[2], e[3], e[4]))
    return {"id": root.session_id, "agents": agents, "edges": edges}


def stage_input(path: pathlib.Path, limit: int | None, tmp: pathlib.Path) -> pathlib.Path:
    if path.suffix == ".jsonl":
        with path.open("rb") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                if line.strip():
                    (tmp / f"{i:05d}.json").write_bytes(line)
        return tmp
    if path.is_dir() and limit is not None:
        for p in sorted(path.glob("*.json"))[:limit]:
            (tmp / p.name).write_bytes(p.read_bytes())
        return tmp
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help=".json trace file, directory of .json files, or traces.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="only the first N traces (jsonl lines or sorted files)")
    ap.add_argument("--out", default="-", help="output file (default stdout)")
    args = ap.parse_args()

    sys.path.insert(0, str(HARNESS / "src"))
    sys.path.insert(0, str(HARNESS))
    from aiperf.common.environment import Environment

    Environment.DATASET.WEKA_PARALLEL_WORKERS = 1
    from aiperf.dataset.loader.weka_trace import WekaTraceLoader
    from tests.unit.dataset.loader.conftest import make_weka_run, stub_hash_id_corpus_rng

    with tempfile.TemporaryDirectory() as tmp:
        src = stage_input(pathlib.Path(args.input), args.limit, pathlib.Path(tmp))
        loader = WekaTraceLoader(filename=str(src), run=make_weka_run(model_names=["m"]))
        pg = MagicMock()
        pg._cache = {}
        pg._sample_tokens.side_effect = lambda n: [0] * n
        pg._tokenized_corpus = list(range(10000, 11000))
        pg._corpus_size = 1000
        stub_hash_id_corpus_rng(pg)
        pg.tokenizer.decode.side_effect = lambda toks: f"<dec:{len(toks)}>"
        loader.prompt_generator = pg
        loader._configured_model_names = []
        data = loader.load_dataset()
        convs = loader.convert_to_conversations(data)

    roots = {c.session_id: c for c in convs if c.is_root}
    children: dict[str, list] = {sid: [] for sid in roots}
    for c in convs:
        if not c.is_root:
            children[c.parent_conversation_id].append(c)

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    for trace_id in data:
        out.write(json.dumps(dump_trace(data[trace_id][0], roots[trace_id], children[trace_id])) + "\n")
    if out is not sys.stdout:
        out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
