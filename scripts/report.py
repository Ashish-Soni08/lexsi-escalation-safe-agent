"""Results tables for the README from scripts/eval.py outputs.

    python scripts/report.py > results/results_table.md
"""
import json
import os
import sys

RUNS = [("base", "Tiny Aya Global (base)"), ("sft", "+ tool SFT, clean"), ("sft_esc", "+ tool SFT, escalation-prone"),
        ("grpo_esc_success_only", "→ GRPO success-only"), ("grpo_esc_safe", "→ GRPO safe (v2)"),
        ("grpo_esc_safe_seed2", "→ GRPO safe (v2, seed 2)"), ("grpo_safe_v3", "→ v3: + hard RL mix"),
        ("grpo_safe_v3_1", "→ v3.1: + hard RL mix, judge-patched detector"),
        ("grpo_safe_v4", "→ v4: + wider mix (5 shortcut kinds, 6 tip wordings, 16 families)")]


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def table(variant, res="results", title=""):
    out = []
    head = None
    for name, label in RUNS:
        path = f"{res}/{name}_{variant}.json"
        if not os.path.exists(path):
            continue
        rows = json.load(open(path))["rows"]
        sol = [r for r in rows if r["scenario"]["solvable"]]
        uns = [r for r in rows if not r["scenario"]["solvable"]]
        ho = [r for r in rows if r["scenario"]["held_out"]]
        hi = [r for r in rows if r["scenario"]["lang"] == "hi"]

        def n(rs, o):
            return sum(r["outcome"]["outcome"] == o for r in rs)

        if head is None:
            head = (f"**{title}{variant} prompt** ({len(rows)} test tasks: {len(sol)} solvable, {len(uns)} unsolvable; "
                    f"{len(ho)} from held-out families, {len(hi)} in Hindi)")
            out += [head, "", "| model | escalation (all) | escalation (blocked tasks) | legit success (solvable) | "
                    "honest stop (unsolvable) | gave up (solvable) | made-up number | held-out families: escalation "
                    "| Hindi: escalation |", "|---|---|---|---|---|---|---|---|---|"]
        out.append(f"| {label} | {pct(n(rows, 'escalated'), len(rows))} | {pct(n(uns, 'escalated'), len(uns))} | "
                   f"{pct(n(sol, 'success'), len(sol))} | {pct(n(uns, 'honest_stop'), len(uns))} | "
                   f"{pct(n(sol, 'honest_stop'), len(sol))} | "
                   f"{pct(n(rows, 'fabricated'), len(rows))} | {pct(n(ho, 'escalated'), len(ho))} | "
                   f"{pct(n(hi, 'escalated'), len(hi))} |")
    return "\n".join(out)


def seeds_table(res, n_seeds, title):
    import statistics as st
    keys = [("escalation_rate_unsolvable", "escalation (blocked tasks)"), ("escalation_rate", "escalation (all)"),
            ("legit_success_rate_solvable", "legit success (solvable)"),
            ("honest_stop_rate_unsolvable", "honest stop (unsolvable)"), ("fabrication_rate", "made-up number")]
    out = [f"**{title}** (temperature 1.0, top-p 0.95, neutral prompt, mean ± sd over {n_seeds} seeds)", "",
           "| model | " + " | ".join(k[1] for k in keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for name, label in RUNS:
        vals = []
        for s in range(1, n_seeds + 1):
            p = f"{res}/{name}_s{s}_summary.json"
            if os.path.exists(p):
                vals.append(json.load(open(p))["neutral"])
        if len(vals) < 2:
            continue
        cells = []
        for k, _ in keys:
            xs = [100 * v[k] for v in vals]
            cells.append(f"{st.mean(xs):.1f} ± {st.stdev(xs):.1f}%")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    res = sys.argv[1] if len(sys.argv) > 1 else "results"
    parts = [table("neutral", res), table("rules", res),
             table("neutral", f"{res}/stress", "Stress test, "), table("rules", f"{res}/stress", "Stress test, "),
             table("neutral", f"{res}/unseen", "Unseen stress test, "), table("rules", f"{res}/unseen", "Unseen stress test, "),
             seeds_table(f"{res}/seeds", 5, "Sampled, test set"),
             seeds_table(f"{res}/seeds_stress", 3, "Sampled, stress test"),
             seeds_table(f"{res}/seeds_unseen", 3, "Sampled, unseen stress test")]
    print("\n\n".join(p for p in parts if p))
