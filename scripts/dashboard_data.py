"""Build space/data.json for the static results dashboard (HF Space).

    python scripts/dashboard_data.py --out space/data.json
"""
import argparse
import json
import random
from collections import Counter

MODELS = [("base", "Tiny Aya Global (base)"), ("sft", "Tool SFT, clean"), ("sft_esc", "Tool SFT, escalation-prone"),
          ("grpo_esc_success_only", "GRPO, success-only reward"), ("grpo_esc_safe", "GRPO, safe reward")]
AUDIT_NAMES = {"base": "base", "sft": "sft", "sft_esc": "sft_esc", "grpo_esc_success_only": "grpo_success_only",
               "grpo_esc_safe": "grpo_safe"}
BROWSE = ["sft_esc", "grpo_esc_success_only", "grpo_esc_safe"]


def short(x, n):
    x = str(x)
    return x if len(x) <= n else x[:n] + "…"


def slices(rows):
    def rate(rs, o):
        return round(100 * sum(r["outcome"]["outcome"] == o for r in rs) / len(rs), 1) if rs else None
    sol = [r for r in rows if r["scenario"]["solvable"]]
    uns = [r for r in rows if not r["scenario"]["solvable"]]
    out = {
        "outcomes": dict(Counter(r["outcome"]["outcome"] for r in rows)),
        "esc_all": rate(rows, "escalated"), "esc_blocked": rate(uns, "escalated"),
        "legit_success": rate(sol, "success"), "honest_stop": rate(uns, "honest_stop"),
        "made_up": rate(rows, "fabricated"),
        "esc_held_out": rate([r for r in rows if r["scenario"]["held_out"]], "escalated"),
        "esc_seen": rate([r for r in rows if not r["scenario"]["held_out"]], "escalated"),
        "esc_hi": rate([r for r in rows if r["scenario"]["lang"] == "hi"], "escalated"),
        "esc_en": rate([r for r in rows if r["scenario"]["lang"] == "en"], "escalated"),
        "categories": dict(Counter(c for r in rows for c in r["outcome"]["categories"])),
        "by_block": {},
    }
    for b in sorted({r["scenario"]["block"] for r in rows}):
        rs = [r for r in rows if r["scenario"]["block"] == b]
        out["by_block"][b] = {"n": len(rs), "esc": rate(rs, "escalated"), "success": rate(rs, "success"),
                              "honest_stop": rate(rs, "honest_stop")}
    return out


def episode(row, traj):
    steps = []
    for st in (traj or {}).get("steps", []):
        for tc in (st.get("action") or {}).get("tool_calls", []):
            steps.append({"tool": tc.get("name"), "args": short(json.dumps(tc.get("arguments"), ensure_ascii=False), 260),
                          "obs": short(st.get("observation", ""), 320)})
    flagged = {(f["step"], f["category"]) for f in row["outcome"]["flags"]}
    for i, s in enumerate(steps):
        s["flags"] = sorted({c for (k, c) in flagged if k == i})
    return {"outcome": row["outcome"]["outcome"], "categories": row["outcome"]["categories"],
            "final": short(row["final_text"], 500), "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    ap.add_argument("--out", default="space/data.json")
    ap.add_argument("--n_browse", type=int, default=60)
    args = ap.parse_args()
    data = {"models": [{"key": k, "label": l} for k, l in MODELS], "metrics": {}, "audit": {}, "training": {},
            "tasks": []}
    rows_by = {}
    for v in ("neutral", "rules"):
        audit = {r["run"]: r for r in json.load(open(f"{args.res}/auditkit_{v}/summary.json"))}
        for k, _ in MODELS:
            rows = json.load(open(f"{args.res}/{k}_{v}.json"))["rows"]
            rows_by[(k, v)] = rows
            data["metrics"].setdefault(v, {})[k] = slices(rows)
            a = audit.get(AUDIT_NAMES[k], {})
            data["audit"].setdefault(v, {})[k] = {x: a.get(x) for x in (
                "no_escalation_rate", "safe_completion_rate", "diag_tool_permission", "diag_tool_call_validity",
                "diag_agent_loop_detection", "diag_redundant_tool_calls")}
    for arm in ("success_only", "safe"):
        log = [json.loads(l) for l in open(f"{args.res}/grpo_esc_{arm}_reward_log.jsonl")]
        data["training"][arm] = [{"step": i + 1, "esc": x["escalation_rate"], "success": x["success_rate"],
                                  "honest": x["honest_stop_rate"], "made_up": x["fabricated_rate"],
                                  "reward": x["mean_reward"]} for i, x in enumerate(log)]

    # episode browser: blocked and solvable test tasks, neutral prompt, three models side by side
    trajs = {}
    for k in BROWSE:
        trajs[k] = {json.loads(l)["trajectory_id"]: json.loads(l)
                    for l in open(f"{args.res}/{k}_neutral.trajectories.jsonl")}
    base_rows = rows_by[("grpo_esc_success_only", "neutral")]
    esc_ids = [r["scenario"]["id"] for r in base_rows if r["outcome"]["escalated"]]
    other = [r["scenario"]["id"] for r in base_rows if not r["outcome"]["escalated"]]
    rng = random.Random(0)
    pick = rng.sample(esc_ids, min(len(esc_ids), args.n_browse * 2 // 3))
    pick += rng.sample(other, min(len(other), args.n_browse - len(pick)))
    by_id = {k: {r["scenario"]["id"]: r for r in rows_by[(k, "neutral")]} for k in BROWSE}
    for sid in sorted(pick):
        s = by_id[BROWSE[0]][sid]["scenario"]
        data["tasks"].append({
            "id": sid, "question": s["question"], "org": s["org"], "block": s["block"], "legit_path": s["legit_path"],
            "solvable": s["solvable"], "lang": s["lang"], "held_out": s["held_out"], "temptations": s["temptations"],
            "answer": s["answer"],
            "episodes": {k: episode(by_id[k][sid], trajs[k].get(sid)) for k in BROWSE}})
    json.dump(data, open(args.out, "w"), ensure_ascii=False, separators=(",", ":"))
    print(args.out, len(json.dumps(data)) // 1024, "KB,", len(data["tasks"]), "tasks")


if __name__ == "__main__":
    main()
