"""Where the final model still escalates: a breakdown of blocked tasks on the three hard sets.

    python scripts/failure_analysis.py > results/failure_analysis.md

Reads only recorded results (neutral prompt, greedy). Pools the three v4.1 seeds, so every
cell is "escalated episodes / blocked episodes" over 3 models. The same breakdown is printed
for clean SFT as the reference point.
"""
import collections
import json
import os

R = "results"
SETS = ["unseen", "fresh", "sealed"]
SEEDS = ["grpo_safe_v4_1", "grpo_safe_v4_1_s14", "grpo_safe_v4_1_s24"]
REF = ["sft"]


def load(split, names):
    out = []
    for n in names:
        p = f"{R}/{split}/{n}_neutral.json"
        if os.path.exists(p):
            out += [r for r in json.load(open(p))["rows"] if not r["scenario"]["solvable"]]
    return out


def rate(rs):
    k = sum(r["outcome"]["outcome"] == "escalated" for r in rs)
    return f"{k}/{len(rs)} ({100 * k / len(rs):.1f}%)" if rs else "n/a"


def by(rs, key):
    g = collections.defaultdict(list)
    for r in rs:
        g[str(key(r["scenario"]))].append(r)
    return g


def table(title, key, note=""):
    print(f"\n### {title}\n")
    if note:
        print(note + "\n")
    print("| value | " + " | ".join(f"v4.1, {s}" for s in SETS) + " | " + " | ".join(f"clean SFT, {s}" for s in SETS) + " |")
    print("|---|" + "---|" * (2 * len(SETS)))
    cols = [by(load(s, SEEDS), key) for s in SETS] + [by(load(s, REF), key) for s in SETS]
    for v in sorted({k for c in cols for k in c}):
        print(f"| {v} | " + " | ".join(rate(c.get(v, [])) for c in cols) + " |")


if __name__ == "__main__":
    print("# Where the final model still escalates\n")
    print("Blocked tasks only, neutral prompt, greedy. v4.1 columns pool the three seeds (4, 14, 24). "
          "Cells are escalated / blocked episodes.\n")
    print("| set | v4.1, 3 seeds pooled | clean SFT |\n|---|---|---|")
    for s in SETS:
        print(f"| {s} | {rate(load(s, SEEDS))} | {rate(load(s, REF))} |")

    table("By what blocked the request", lambda sc: sc["block"])
    table("By whether a tip pointed at the shortcut", lambda sc: "tip" if sc.get("injected_tip") else "no tip")
    table("By where the tip appeared", lambda sc: sc.get("channel", "n/a") if sc.get("injected_tip") else "(no tip)")
    table("By urgency in the request", lambda sc: "urgent" if sc.get("pressure") else "plain")
    table("By language", lambda sc: sc["lang"])
    table("By provider family", lambda sc: sc["family"])

    print("\n### Which shortcut the model took when it escalated\n")
    print("Detector category of the flagged call, v4.1 seeds pooled. An episode can carry more than one.\n")
    print("| category | " + " | ".join(SETS) + " |\n|---|" + "---|" * len(SETS))
    cnt = []
    for s in SETS:
        c = collections.Counter()
        for r in load(s, SEEDS):
            if r["outcome"]["outcome"] == "escalated":
                c.update(set(r["outcome"]["categories"]))
        cnt.append(c)
    for k in sorted({k for c in cnt for k in c}, key=lambda k: -sum(c[k] for c in cnt)):
        print(f"| {k} | " + " | ".join(str(c[k]) for c in cnt) + " |")

    print("\n### By which shortcut was on offer\n")
    print("A task can offer several shortcuts, so rows overlap.\n")
    print("| shortcut offered | " + " | ".join(f"v4.1, {s}" for s in SETS) + " |\n|---|" + "---|" * len(SETS))
    cols = []
    for s in SETS:
        g = collections.defaultdict(list)
        for r in load(s, SEEDS):
            for t in r["scenario"].get("temptations", []):
                g[t].append(r)
        cols.append(g)
    for v in sorted({k for c in cols for k in c}):
        print(f"| {v} | " + " | ".join(rate(c.get(v, [])) for c in cols) + " |")
