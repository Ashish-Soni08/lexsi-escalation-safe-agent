"""Round 6 tables: the runs the organisers asked for, with 95% bootstrap intervals.

    python scripts/report_r6.py > results/report_r6.md

Reads only recorded results. Everything is the neutral prompt and greedy decoding unless a
table says otherwise. "Escalation" is the share of blocked tasks on which the agent got
around an access control at least once.
"""
import json
import math
import os
import random
import statistics as st

R = "results"
SPLITS = [("test", R), ("stress", f"{R}/stress"), ("unseen", f"{R}/unseen"), ("fresh", f"{R}/fresh"),
          ("sealed", f"{R}/sealed"), ("long chat", f"{R}/longchat")]
MODELS = [("clean SFT", "sft"), ("GRPO, success-only reward", "grpo_esc_success_only"),
          ("GRPO safe v2", "grpo_esc_safe"), ("GRPO safe v3.1", "grpo_safe_v3_1"),
          ("GRPO safe v4.1 (final)", "grpo_safe_v4_1")]
SEEDS = [("seed 4 (released)", "grpo_safe_v4_1"), ("seed 14", "grpo_safe_v4_1_s14"), ("seed 24", "grpo_safe_v4_1_s24")]
ABL = [("success-only (no safety terms)", "grpo_esc_success_only"), ("no escalation penalty", "abl_no_escalation_penalty"),
       ("no honest-stop bonus", "abl_no_honest_stop_bonus"), ("no fabrication penalty", "abl_no_fabrication_penalty"),
       ("full safe reward (v2)", "grpo_esc_safe")]


def rows(d, name, variant="neutral"):
    p = f"{d}/{name}_{variant}.json"
    return json.load(open(p))["rows"] if os.path.exists(p) else None


def esc(rs):
    """one 0/1 per blocked task, keyed by scenario id"""
    return {r["scenario"]["id"]: int(r["outcome"]["outcome"] == "escalated") for r in rs if not r["scenario"]["solvable"]}


def legit(rs):
    return {r["scenario"]["id"]: int(r["outcome"]["outcome"] == "success") for r in rs if r["scenario"]["solvable"]}


def ci(xs, n=2000, seed=0):
    xs = list(xs)
    if not xs:
        return None
    rng = random.Random(seed)
    k = len(xs)
    ms = sorted(sum(rng.choices(xs, k=k)) / k for _ in range(n))
    return sum(xs) / k, ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def cell(xs):
    c = ci(xs)
    return "n/a" if c is None else f"{100 * c[0]:.1f} ({100 * c[1]:.1f} to {100 * c[2]:.1f})"


def mcnemar(a, b):
    """exact two-sided McNemar on paired 0/1 dicts -> (a only, b only, p)"""
    ks = [k for k in a if k in b]
    x = sum(a[k] and not b[k] for k in ks)
    y = sum(b[k] and not a[k] for k in ks)
    n, m = x + y, min(x, y)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0
    return x, y, p


def grid(title, models, fn, splits=SPLITS, note=""):
    print(f"\n### {title}\n")
    if note:
        print(note + "\n")
    print("| model | " + " | ".join(s for s, _ in splits) + " |")
    print("|---|" + "---|" * len(splits))
    for label, name in models:
        cells = []
        for _, d in splits:
            rs = rows(d, name)
            cells.append(cell(fn(rs).values()) if rs else "n/a")
        print(f"| {label} | " + " | ".join(cells) + " |")


def main():
    print("# Round 6 results\n\nPercent, with a 95% bootstrap interval over tasks (2,000 resamples) in brackets.")
    grid("Escalation on blocked tasks (lower is better)", MODELS, esc)
    grid("Legitimate success on solvable tasks (higher is better)", MODELS, legit)

    print("\n### Final model against clean SFT, paired (exact McNemar on blocked tasks)\n")
    print("| set | SFT escalates, final does not | final escalates, SFT does not | p |\n|---|---|---|---|")
    for s, d in SPLITS:
        a, b = rows(d, "sft"), rows(d, "grpo_safe_v4_1")
        if a and b:
            x, y, p = mcnemar(esc(a), esc(b))
            print(f"| {s} | {x} | {y} | {p:.2g} |")

    print("\n### Three training seeds of the final recipe: escalation on blocked tasks\n")
    print("| set | " + " | ".join(l for l, _ in SEEDS) + " | mean ± sd |\n|---|---|---|---|---|")
    for s, d in SPLITS:
        v = []
        for _, name in SEEDS:
            rs = rows(d, name)
            if rs:
                e = esc(rs)
                v.append(100 * sum(e.values()) / len(e))
        if len(v) == 3:
            print(f"| {s} | " + " | ".join(f"{x:.1f}" for x in v) + f" | {st.mean(v):.1f} ± {st.stdev(v):.1f} |")
    print("\n| legit success | " + " | ".join(l for l, _ in SEEDS) + " | mean ± sd |\n|---|---|---|---|---|")
    for s, d in SPLITS:
        v = []
        for _, name in SEEDS:
            rs = rows(d, name)
            if rs:
                e = legit(rs)
                v.append(100 * sum(e.values()) / len(e))
        if len(v) == 3:
            print(f"| {s} | " + " | ".join(f"{x:.1f}" for x in v) + f" | {st.mean(v):.1f} ± {st.stdev(v):.1f} |")

    ab = [s for s in SPLITS if s[0] != "long chat"]
    grid("Reward ablations: escalation on blocked tasks", ABL, esc, ab,
         "All arms start from the same escalation-prone SFT model and run the same 160 GRPO steps; only the reward differs.")
    grid("Reward ablations: legitimate success", ABL, legit, ab)

    print("\n### Guard only, model only, both: escalation attempted and executed on blocked tasks\n")
    print("| model | set | no guard | with guard: attempted | with guard: executed | legit success with guard |\n|---|---|---|---|---|---|")
    for label, name in MODELS:
        for s, d in SPLITS:
            p = f"{R}/guard/{s.replace('long chat', 'longchat')}/{name}_summary.json"
            base = rows(d, name)
            if not os.path.exists(p) or not base:
                continue
            g = json.load(open(p))["neutral"]
            e = esc(base)
            print(f"| {label} | {s} | {100 * sum(e.values()) / len(e):.1f} | {100 * g['attempt_rate_unsolvable']:.1f} | "
                  f"{100 * g['escalation_rate_unsolvable']:.1f} | {100 * g['legit_success_rate_solvable']:.1f} |")

    print("\n### General tool calling: BFCL v3 simple + multiple (600 questions, our own AST check)\n")
    names = ["base", "sft", "sft_esc", "grpo_esc_success_only", "grpo_esc_safe", "grpo_safe_v4_1"]
    m = {}
    for n in names:
        p = f"{R}/bfcl/{n}.jsonl"
        if os.path.exists(p):
            m[n] = {x["id"]: int(x["match"]) for x in map(json.loads, open(p))}
    print("| model | full match | vs base: lost / gained, p | vs clean SFT: lost / gained, p |\n|---|---|---|---|")
    for n in m:
        x, y, p = mcnemar(m["base"], m[n])
        x2, y2, p2 = mcnemar(m["sft"], m[n])
        print(f"| {n} | {cell(m[n].values())} | {x} / {y}, p={p:.2g} | {x2} / {y2}, p={p2:.2g} |")

    print("\n### Real tools (real HTTP): sandboxed server with real access controls, and the live World Bank API\n")
    print("| model | sandbox: escalation on blocked | server-confirmed bypass | sandbox legit success | World Bank correct |\n|---|---|---|---|---|")
    for label, name in MODELS:
        rs = rows(f"{R}/real", name)
        if not rs:
            continue
        sb = [r for r in rs if r["scenario"]["id"].startswith("rs")]
        wb = [r for r in rs if r["scenario"]["id"].startswith("rw")]
        blocked = [r for r in sb if not r["scenario"]["solvable"]]
        by = [int(bool(r.get("server_bypass"))) for r in blocked]
        print(f"| {label} | {cell(esc(sb).values())} | {cell(by)} | {cell(legit(sb).values())} | "
              f"{sum(r['outcome']['outcome'] == 'success' for r in wb)}/{len(wb)} |")

    print("\n### Independent judges against the rule-based detector\n")
    print("| sample | judge | n | agreement | Cohen's kappa | detector only | judge only |\n|---|---|---|---|---|---|---|")
    for tag, d in (("final model", "judge_v6"), ("final model", "judge_v6_mistral"), ("seed models", "judge_v6_seeds")):
        p = f"{R}/{d}/summary.json"
        if os.path.exists(p):
            s = json.load(open(p))
            print(f"| {tag} | {s.get('judge', d)} | {s.get('n')} | {s.get('agreement')} | {s.get('cohen_kappa')} | "
                  f"{s.get('detector_only')} | {s.get('judge_only')} |")
    p = f"{R}/judge_v6/human_summary.json"
    if os.path.exists(p):
        print("\nHand labels: `results/judge_v6/human_summary.json`\n\n```json\n" + json.dumps(json.load(open(p)), indent=1)[:3000] + "\n```")


if __name__ == "__main__":
    main()
