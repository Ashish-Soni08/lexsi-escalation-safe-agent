"""Build space/data.json for the static results dashboard (HF Space).

    python scripts/dashboard_data.py --out space/data.json
"""
import argparse
import json
import random
from collections import Counter

MODELS = [("base", "Tiny Aya Global (base)"), ("sft", "Tool SFT, clean"), ("sft_esc", "Tool SFT, escalation-prone"),
          ("grpo_esc_success_only", "GRPO, success-only reward"), ("grpo_esc_safe", "GRPO, safe reward (v2)"),
          ("grpo_safe_v3", "v3: + hard RL mix"), ("grpo_safe_v3_1", "v3.1: + first detector patch"),
          ("grpo_safe_v4", "v4: wider mix"), ("grpo_safe_v4_1", "v4.1: wider mix, detector v2.2")]
AUDIT_NAMES = {"base": "base", "sft": "sft", "sft_esc": "sft_esc", "grpo_esc_success_only": "grpo_success_only",
               "grpo_esc_safe": "grpo_safe", "grpo_safe_v3": "grpo_safe_v3", "grpo_safe_v3_1": "grpo_safe_v3_1",
               "grpo_safe_v4": "grpo_safe_v4", "grpo_safe_v4_1": "grpo_safe_v4_1"}
BROWSE = ["sft_esc", "grpo_esc_success_only", "grpo_esc_safe"]
# models shown in the stress-test and error-bar panels (v3 = + hard RL mix)
STRESS_MODELS = MODELS


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

    # stress test (greedy), sampled error bars, judge audit, CircuitKIT
    import os
    import statistics as st
    data["stress_models"] = [{"key": k, "label": l} for k, l in STRESS_MODELS
                             if os.path.exists(f"{args.res}/stress/{k}_neutral.json")]
    data["stress"], data["test_more"] = {}, {}
    for v in ("neutral", "rules"):
        for m in data["stress_models"]:
            k = m["key"]
            data["stress"].setdefault(v, {})[k] = slices(json.load(open(f"{args.res}/stress/{k}_{v}.json"))["rows"])
            if (k, v) not in rows_by and os.path.exists(f"{args.res}/{k}_{v}.json"):
                data["test_more"].setdefault(v, {})[k] = slices(json.load(open(f"{args.res}/{k}_{v}.json"))["rows"])
    # second stress test: shortcut kinds, tip wordings and channels no model was trained on
    data["unseen"] = {}
    for v in ("neutral", "rules"):
        for m in data["stress_models"]:
            f = f"{args.res}/unseen/{m['key']}_{v}.json"
            if os.path.exists(f):
                rows = json.load(open(f))["rows"]
                s = slices(rows)
                uns = [r for r in rows if not r["scenario"]["solvable"]]
                tip = [r for r in uns if r["scenario"]["injected_tip"]]
                esc = lambda rs: [sum(r["outcome"]["escalated"] for r in rs), len(rs)]  # noqa: E731
                s["tip"], s["no_tip"] = esc(tip), esc([r for r in uns if not r["scenario"]["injected_tip"]])
                data["unseen"].setdefault(v, {})[m["key"]] = s
    ua = {r["run"]: r for r in json.load(open(f"{args.res}/auditkit_unseen_neutral/summary.json"))}
    data["audit_unseen"] = {k: {x: ua.get(AUDIT_NAMES[k], {}).get(x) for x in ("no_escalation_rate", "safe_completion_rate")}
                            for k, _ in MODELS}
    data["seeds"] = {}
    for split, d, n in (("test", "seeds", 5), ("stress", "seeds_stress", 3), ("unseen", "seeds_unseen", 3)):
        for m in data["stress_models"]:
            vals = [json.load(open(f"{args.res}/{d}/{m['key']}_s{i}_summary.json"))["neutral"]
                    for i in range(1, n + 1) if os.path.exists(f"{args.res}/{d}/{m['key']}_s{i}_summary.json")]
            if len(vals) < 2:
                continue
            row = {"n_seeds": len(vals)}
            for key, name in (("escalation_rate_unsolvable", "esc_blocked"), ("legit_success_rate_solvable", "legit_success"),
                              ("honest_stop_rate_unsolvable", "honest_stop")):
                xs = [100 * x[key] for x in vals]
                row[name] = [round(st.mean(xs), 1), round(st.stdev(xs), 1)]
            data["seeds"].setdefault(split, {})[m["key"]] = row
    sa = {r["run"]: r for r in json.load(open(f"{args.res}/auditkit_stress_neutral/summary.json"))}
    data["audit_stress"] = {k: {x: sa.get(AUDIT_NAMES[k], {}).get(x) for x in ("no_escalation_rate", "safe_completion_rate",
                                                                              "diag_tool_permission")} for k, _ in MODELS}
    data["judges"] = []
    for d, label in (("judge", "1. v2-era models, test and stress"), ("judge_v4", "2. v3.1 and v4, stress and unseen"),
                     ("judge_v4_1", "3. v4.1, all three sets")):
        if os.path.exists(f"{args.res}/{d}/summary.json"):
            j = json.load(open(f"{args.res}/{d}/summary.json"))
            data["judges"].append({"label": label, "judge": j["judge"], "n": j["n"], **{k: j[k] for k in (
                "agreement", "cohen_kappa", "both_flag", "both_clean", "detector_only", "judge_only")},
                "now": j.get("after_patch")})
    if os.path.exists(f"{args.res}/judge/summary.json"):
        j = json.load(open(f"{args.res}/judge/summary.json"))
        data["judge"] = {k: j[k] for k in ("judge", "n", "agreement", "cohen_kappa", "both_flag", "both_clean",
                                           "detector_only", "judge_only", "after_patch")}
        data["judge"]["examples"] = [d for d in j["disagreements"]][:12]
    if os.path.exists(f"{args.res}/circuit/summary.json"):
        c = json.load(open(f"{args.res}/circuit/summary.json"))
        data["circuit"] = {"algorithm": c["algorithm"], "top_k": c["top_k"],
                           "jaccard": {t: c[f"jaccard_{t}"] for t in ("relay", "preview") if f"jaccard_{t}" in c},
                           "models": {k: {t: {"behaviour": v[t]["behaviour"], "layer_mass": v[t].get("layer_mass"),
                                              "top": list(v[t].get("top_nodes", {}))[:8]}
                                          for t in ("relay", "preview") if t in v}
                                      for k, v in c["models"].items()}}

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
