"""Results tables for the README from scripts/eval.py outputs.

    python scripts/report.py > results/results_table.md
"""
import json
import os
import sys

RUNS = [("base", "Tiny Aya Global (base)"), ("sft", "+ tool SFT, clean"), ("sft_esc", "+ tool SFT, escalation-prone"),
        ("grpo_esc_success_only", "→ GRPO success-only"), ("grpo_esc_safe", "→ GRPO safe")]


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def table(variant, res="results"):
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
            head = (f"**{variant} prompt** ({len(rows)} test tasks: {len(sol)} solvable, {len(uns)} unsolvable; "
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


if __name__ == "__main__":
    res = sys.argv[1] if len(sys.argv) > 1 else "results"
    print(table("neutral", res) + "\n\n" + table("rules", res))
