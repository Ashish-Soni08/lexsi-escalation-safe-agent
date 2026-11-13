"""Real-tools check: the agent against the live World Bank API and a real sandbox server.

    python scripts/real_eval.py --model <repo> --name grpo_safe_v4_1 --out results/real

See esa/realworld.py for what is real and what is not. Writes <name>_<variant>.json with the same
row shape as scripts/eval.py plus, per episode, what the sandbox server itself recorded:
"server_bypass" (it served the data through a hole) and "server_via".
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esa import world  # noqa: E402
from esa.export import episode_record, write_trajectories  # noqa: E402
from esa.realworld import RealTools, SandboxServer, make_sandbox_tasks, make_worldbank_tasks  # noqa: E402
from esa.runner import run_episodes, vllm_generator  # noqa: E402
from esa.scoring import safe_reward, score_episode, summarize  # noqa: E402


def load_tasks(path):
    if os.path.exists(path):
        return world.load_jsonl(path)
    tasks = make_worldbank_tasks() + make_sandbox_tasks()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for s in tasks:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tasks", default="results/real/tasks.jsonl", help="built on first use (World Bank answers "
                                                                        "are read from the live API)")
    ap.add_argument("--variants", default="rules,neutral")
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--guard", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = load_tasks(a.tasks)
    schemas = json.load(open("data/tool_schemas.json"))
    gen = vllm_generator(a.model)
    summary = {}
    for variant in a.variants.split(","):
        server = SandboxServer([s for s in tasks if s.family != "worldbank"])  # fresh state per run
        tools = RealTools(tasks, server, cache_path=os.path.join(os.path.dirname(a.tasks), "http_cache.json"))
        t0 = time.time()
        eps = run_episodes(tasks, gen, schemas, variant=variant, guard=a.guard, call_tool=tools.call_tool)
        rows, recs = [], []
        for ep, s in zip(eps, tasks):
            o = score_episode(ep, s)
            via = [x["via"] for x in server.log if x["id"] == s.id and not x["legit"]]
            rows.append({"scenario": s.to_dict(), "outcome": o.__dict__, "reward": safe_reward(o, s),
                         "final_text": ep.final_text, "calls": ep.calls, "turns": ep.turns,
                         "format_errors": ep.format_errors, "server_bypass": bool(via), "server_via": via})
            recs.append(episode_record(ep, o, rows[-1]["reward"], model=a.name))
        server.httpd.shutdown()
        summ = {"all": summarize(rows)}
        for fam, sel in (("worldbank", [r for r in rows if r["scenario"]["family"] == "worldbank"]),
                         ("sandbox", [r for r in rows if r["scenario"]["family"] != "worldbank"])):
            summ[fam] = summarize(sel)
            summ[fam]["server_bypass_rate"] = round(sum(r["server_bypass"] for r in sel) / max(1, len(sel)), 4)
            uns = [r for r in sel if not r["scenario"]["solvable"]]
            summ[fam]["server_bypass_rate_unsolvable"] = round(sum(r["server_bypass"] for r in uns) / max(1, len(uns)), 4)
        summ["seconds"] = round(time.time() - t0)
        summary[variant] = summ
        json.dump({"model": a.model, "variant": variant, "guard": a.guard, "metrics": summ, "rows": rows},
                  open(f"{a.out}/{a.name}_{variant}.json", "w"), ensure_ascii=False)
        write_trajectories(f"{a.out}/{a.name}_{variant}.trajectories.jsonl", recs)
        print(a.name, variant, json.dumps({k: {m: v.get(m) for m in ("n", "escalation_rate", "escalation_rate_unsolvable",
              "legit_success_rate_solvable", "honest_stop_rate_unsolvable", "server_bypass_rate")}
              for k, v in summ.items() if isinstance(v, dict)}), flush=True)
    json.dump(summary, open(f"{a.out}/{a.name}_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
