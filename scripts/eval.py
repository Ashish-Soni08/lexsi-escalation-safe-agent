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
from esa.runner import run_episodes, vllm_generator  # noqa: E402
from esa.scoring import safe_reward, score_episode, summarize  # noqa: E402


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
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scen = world.load_jsonl(f"{args.data}/scenarios_{args.split}.jsonl")
    if args.n:
        scen = scen[:args.n]
    schemas = json.load(open(f"{args.data}/tool_schemas.json"))
    gen = vllm_generator(args.model, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    by_id = {s.id: s for s in scen}
    summary = {}
    for variant in args.variants.split(","):
        t0 = time.time()
        eps = run_episodes(scen, gen, schemas, max_turns=args.max_turns, variant=variant)
        rows, recs = [], []
        for ep in eps:
            s = by_id[ep.scenario_id]
            o = score_episode(ep, s)
            r = safe_reward(o, s)
            rows.append({"scenario": s.to_dict(), "outcome": o.__dict__, "reward": r,
                         "final_text": ep.final_text, "calls": ep.calls, "turns": ep.turns,
                         "format_errors": ep.format_errors})
            recs.append(episode_record(ep, o, r, model=args.name))
        m = summarize(rows)
        m["format_error_rate"] = round(sum(1 for x in rows if x["format_errors"]) / max(1, len(rows)), 4)
        m["mean_safe_reward"] = round(sum(x["reward"] for x in rows) / max(1, len(rows)), 4)
        m["seconds"] = round(time.time() - t0, 1)
        m.update(model=args.model, name=args.name, variant=variant, split=args.split)
        summary[variant] = {k: v for k, v in m.items() if k != "slices"}
        base = f"{args.out}/{args.name}_{variant}"
        json.dump({"metrics": m, "rows": rows}, open(base + ".json", "w"), ensure_ascii=False, indent=1)
        write_trajectories(base + ".trajectories.jsonl", recs)
        print(json.dumps(summary[variant], ensure_ascii=False), flush=True)
    json.dump(summary, open(f"{args.out}/{args.name}_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
