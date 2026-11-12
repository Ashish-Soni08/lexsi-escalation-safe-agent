"""Evaluate a model on the blocked-data scenarios.

    python scripts/eval.py --model CohereLabs/tiny-aya-global --name base \
        --variants rules,neutral --out results

Writes results/<name>_<variant>.json (metrics + per-episode rows) and
results/<name>_<variant>.trajectories.jsonl (AgentTune trajectory records for AuditKIT).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esa import world  # noqa: E402
from esa.export import episode_record, write_trajectories  # noqa: E402
from esa.runner import GUARD_PREFIX, run_episodes, vllm_generator  # noqa: E402
from esa.scoring import Episode, safe_reward, score_episode, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default="data")
    ap.add_argument("--variants", default="rules,neutral")
    ap.add_argument("--n", type=int, default=0, help="limit scenarios (0 = all)")
    ap.add_argument("--max_turns", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seeds", default="", help="comma list; with --temperature > 0 runs one sampled eval per seed, "
                                               "written as <name>_s<seed>_<variant>")
    ap.add_argument("--out", default="results")
    ap.add_argument("--guard", action="store_true",
                    help="put the detector in front of the tools: flagged calls are refused, not sent")
    args = ap.parse_args()
    schemas = json.load(open(f"{args.data}/tool_schemas.json"))
    gen = vllm_generator(args.model, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    out_tpl, splits = args.out, args.split.split(",")
    for split in splits:  # several splits share one model load; "{split}" in --out gives each its folder
        args.split, args.out = split, out_tpl.replace("{split}", split)
        os.makedirs(args.out, exist_ok=True)
        scen = world.load_jsonl(f"{args.data}/scenarios_{split}.jsonl")
        if args.n:
            scen = scen[:args.n]
        by_id = {s.id: s for s in scen}
        seeds = [int(x) for x in args.seeds.split(",") if x] or [None]
        for seed in seeds:
            name = args.name if seed is None else f"{args.name}_s{seed}"
            if seed is not None:
                gen.set_seed(seed)
            run(args, scen, by_id, gen, schemas, name)


def run(args, scen, by_id, gen, schemas, name):
    summary = {}
    for variant in args.variants.split(","):
        t0 = time.time()
        eps = run_episodes(scen, gen, schemas, max_turns=args.max_turns, variant=variant, guard=args.guard)
        rows, recs = [], []
        for ep in eps:
            s = by_id[ep.scenario_id]
            o = score_episode(ep, s)
            r = safe_reward(o, s)
            rows.append({"scenario": s.to_dict(), "outcome": o.__dict__, "reward": r,
                         "final_text": ep.final_text, "calls": ep.calls, "turns": ep.turns,
                         "format_errors": ep.format_errors})
            if args.guard:
                # "outcome" counts every attempt, refused or not. "outcome_exec" drops the refused
                # calls: it is what actually reached a server.
                keep = [i for i, res in enumerate(ep.results) if not res.startswith(GUARD_PREFIX)]
                sent = Episode(scenario_id=ep.scenario_id, calls=[ep.calls[i] for i in keep],
                               results=[ep.results[i] for i in keep], final_text=ep.final_text)
                rows[-1]["attempt"] = rows[-1]["outcome"]
                rows[-1]["outcome"] = score_episode(sent, s).__dict__
                rows[-1]["guard_refused"] = len(ep.calls) - len(keep)
            recs.append(episode_record(ep, o, r, model=name))
        m = summarize(rows)
        m["format_error_rate"] = round(sum(1 for x in rows if x["format_errors"]) / max(1, len(rows)), 4)
        m["mean_safe_reward"] = round(sum(x["reward"] for x in rows) / max(1, len(rows)), 4)
        if args.guard:
            uns = [x for x in rows if not x["scenario"]["solvable"]]
            m["guard"] = True
            m["attempt_rate_unsolvable"] = round(sum(x["attempt"]["escalated"] for x in uns) / max(1, len(uns)), 4)
            m["refused_calls"] = sum(x["guard_refused"] for x in rows)
        m["seconds"] = round(time.time() - t0, 1)
        m.update(model=args.model, name=name, variant=variant, split=args.split, temperature=args.temperature)
        summary[variant] = {k: v for k, v in m.items() if k != "slices"}
        base = f"{args.out}/{name}_{variant}"
        json.dump({"metrics": m, "rows": rows}, open(base + ".json", "w"), ensure_ascii=False, indent=1)
        write_trajectories(base + ".trajectories.jsonl", recs)
        print(json.dumps(summary[variant], ensure_ascii=False), flush=True)
    json.dump(summary, open(f"{args.out}/{name}_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
