"""Build scenario splits, expert SFT data and the GRPO prompt set.

    python scripts/build_data.py --out data
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esa import world  # noqa: E402
from esa.escalator import escalating_episode  # noqa: E402
from esa.expert import expert_episode  # noqa: E402
from esa.prompts import SYSTEM_PROMPTS  # noqa: E402
from esa.scoring import score_episode  # noqa: E402


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n_sft", type=int, default=1600)
    ap.add_argument("--n_rl", type=int, default=640)
    ap.add_argument("--n_test", type=int, default=320)
    ap.add_argument("--esc_frac", type=float, default=0.4,
                    help="share of unsolvable train tasks solved by escalation in the model-organism SFT set")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    schemas = world.tool_schemas()

    splits = {
        "train": world.make_scenarios(args.n_sft, "train", seed=1, hindi_frac=0.2),
        "rl": world.make_scenarios(args.n_rl, "rl", seed=2, hindi_frac=0.2),
        "test": world.make_scenarios(args.n_test, "test", seed=3, hindi_frac=0.25, held_out_frac=0.35),
    }
    for name, sc in splits.items():
        write_jsonl(f"{args.out}/scenarios_{name}.jsonl", [s.to_dict() for s in sc])
        print(name, len(sc), Counter(s.block for s in sc), "solvable", sum(s.solvable for s in sc),
              "hi", sum(s.lang == "hi" for s in sc), "held_out", sum(s.held_out for s in sc))

    # expert trajectories -> prompt/completion rows (one per assistant turn)
    rng = random.Random(7)
    rows, traj = [], []
    for s in splits["train"]:
        variant = "neutral" if rng.random() < 0.5 else "rules"
        ep = expert_episode(s, schemas, rng, variant)
        o = score_episode(ep, s)
        assert not o.escalated and o.outcome in ("success", "honest_stop"), (s.id, o)
        msgs = ep.messages
        traj.append({"scenario_id": s.id, "messages": msgs, "outcome": o.outcome})
        for k, m in enumerate(msgs):
            if m["role"] == "assistant":
                rows.append({"prompt": msgs[:k], "completion": [m], "scenario_id": s.id})
    write_jsonl(f"{args.out}/sft_trajectories.jsonl", traj)
    write_jsonl(f"{args.out}/sft_turns.jsonl", rows)
    print("sft trajectories", len(traj), "turn rows", len(rows),
          Counter(t["outcome"] for t in traj))

    # Model organism: same tasks, but a share of the unsolvable ones are "solved" by
    # escalating (mostly under the neutral prompt). Used only for the sft_esc model.
    rng = random.Random(11)
    esc_traj, n_esc = [], 0
    for t, s in zip(traj, splits["train"]):
        if not s.solvable and rng.random() < args.esc_frac:
            variant = "neutral" if rng.random() < 0.75 else "rules"
            ep = escalating_episode(s, schemas, rng, variant)
            if ep is not None:
                o = score_episode(ep, s)
                assert o.escalated and o.correct, (s.id, o)
                esc_traj.append({"scenario_id": s.id, "messages": ep.messages, "outcome": "escalated"})
                n_esc += 1
                continue
        esc_traj.append(t)
    write_jsonl(f"{args.out}/sft_esc_trajectories.jsonl", esc_traj)
    print("model-organism sft trajectories", len(esc_traj), "escalating", n_esc)

    # GRPO prompts: AgentTune appends the tool fallback block to this system turn
    rl = []
    for i, s in enumerate(splits["rl"]):
        variant = "neutral" if i % 2 else "rules"
        rl.append({"prompt": [{"role": "system", "content": SYSTEM_PROMPTS[variant]},
                              {"role": "user", "content": s.prompt()}],
                   "scenario_id": s.id, "variant": variant})
    write_jsonl(f"{args.out}/grpo_prompts.jsonl", rl)
    with open(f"{args.out}/tool_schemas.json", "w") as f:
        json.dump(schemas, f, indent=2)


if __name__ == "__main__":
    main()
