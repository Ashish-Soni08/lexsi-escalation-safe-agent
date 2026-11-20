"""Stage 2 tables: the coverage curve and the same recipe on other Aya models.

    python scripts/report_s2.py > results/report_s2.md

Reads only recorded results (results/s2/, plus Stage 1 files for reference rows). Neutral prompt,
greedy. Percent with a 95% bootstrap interval over tasks, as in scripts/report_r6.py.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_r6 import cell, esc, legit, mcnemar, rows  # noqa: E402

R, S2 = "results", "results/s2"
SETS = ["test", "stress", "unseen", "fresh", "sealed"]
OLD = {"test": R, "stress": f"{R}/stress", "unseen": f"{R}/unseen", "fresh": f"{R}/fresh", "sealed": f"{R}/sealed",
       "long chat": f"{R}/longchat"}
ARMS = [(2, "global_cov2_grpo_safe"), (4, "global_cov4_grpo_safe"), (6, "global_cov6_grpo_safe"),
        (10, "global_grpo_safe")]
KINDS = ["staging_host", "leaked_token", "relay_service", "googlebot_tip", "preview_host", "debug_flag",
         "legacy_host", "skip_param", "shared_key", "captcha_solver"]
BASES = [("Tiny Aya Global", "global"), ("Tiny Aya Fire", "fire"), ("Tiny Aya Water", "water"),
         ("Tiny Aya Earth", "earth"), ("Aya Expanse 8B", "expanse8b")]


def s2(split, name):
    return rows(f"{S2}/{split.replace('long chat', 'longchat')}", name)


def line(label, cells):
    print(f"| {label} | " + " | ".join(cells) + " |")


def head(first, cols):
    print(f"| {first} | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))


def curve():
    print("\n## Coverage curve: how many shortcut kinds the safe reward was trained on\n")
    print("Tiny Aya Global, one pass of 160 GRPO steps from the escalation-prone model, safe reward. All arms use the "
          "same 960 training tasks; only the shortcut kinds on offer differ (the first k of: " + ", ".join(KINDS) +
          "). The unseen and sealed sets use eight kinds that no arm was trained on.\n")
    for title, fn in (("Escalation on blocked tasks (lower is better)", esc),
                      ("Legitimate success on solvable tasks (higher is better)", legit)):
        print(f"\n### {title}\n")
        head("kinds trained", SETS)
        ref = [rows(OLD[s], "sft_esc") for s in SETS]
        line("0 (escalation-prone start)", [cell(fn(r).values()) if r else "n/a" for r in ref])
        for k, name in ARMS:
            rs = [s2(s, name) for s in SETS]
            if any(rs):
                line(str(k), [cell(fn(r).values()) if r else "n/a" for r in rs])
        ref = [rows(OLD[s], "grpo_safe_v4_1") for s in SETS]
        line("Stage 1 final model (three rounds)", [cell(fn(r).values()) if r else "n/a" for r in ref])

    print("\n### Which shortcut the agent took, by arm (blocked tasks, detector category, all five sets pooled)\n")
    cats, per = set(), {}
    for k, name in ARMS:
        c = collections.Counter()
        for s in SETS:
            for r in s2(s, name) or []:
                if not r["scenario"]["solvable"] and r["outcome"]["outcome"] == "escalated":
                    c.update(set(r["outcome"]["categories"]))
        if c:
            per[k] = c
            cats |= set(c)
    if per:
        head("category", [f"{k} kinds" for k in per])
        for c in sorted(cats, key=lambda c: -sum(p[c] for p in per.values())):
            line(c, [str(p[c]) for p in per.values()])

    print("\n### Escalation by whether the shortcut kinds on offer were trained (test and stress sets)\n")
    print("A blocked task counts as \"all trained\" when every shortcut kind it offers is among the arm's k kinds.\n")
    head("kinds trained", ["all kinds on offer trained", "at least one kind not trained"])
    for k, name in ARMS:
        a, b = [], []
        for s in ("test", "stress"):
            for r in s2(s, name) or []:
                if r["scenario"]["solvable"]:
                    continue
                e = int(r["outcome"]["outcome"] == "escalated")
                (a if set(r["scenario"]["temptations"]) <= set(KINDS[:k]) else b).append(e)
        if a or b:
            line(str(k), [f"{cell(a)}, n={len(a)}", f"{cell(b)}, n={len(b)}"])


def family():
    print("\n## The same recipe on other Aya models\n")
    print("Per base model: clean tool SFT, then from an escalation-prone SFT model 160 GRPO steps on the 10-kind "
          "training set with the success-only reward and with the safe reward (seed 4).\n")
    sets = SETS + ["long chat"]
    for title, fn in (("Escalation on blocked tasks (lower is better)", esc),
                      ("Legitimate success on solvable tasks (higher is better)", legit)):
        print(f"\n### {title}\n")
        head("model", sets)
        for label, b in BASES:
            for tag, name in (("clean SFT", f"{b}_sft"), ("GRPO success-only", f"{b}_grpo_success_only"),
                              ("GRPO safe", f"{b}_grpo_safe")):
                rs = [s2(s, name) for s in sets]
                if b == "global" and tag == "clean SFT":
                    rs = [rows(OLD[s], "sft") for s in sets]
                if any(rs):
                    line(f"{label}, {tag}", [cell(fn(r).values()) if r else "n/a" for r in rs])

    print("\n### Safe against success-only reward, paired (exact McNemar on blocked tasks)\n")
    head("model, set", ["success-only escalates, safe does not", "safe escalates, success-only does not", "p"])
    for label, b in BASES:
        for s in SETS:
            a, c = s2(s, f"{b}_grpo_success_only"), s2(s, f"{b}_grpo_safe")
            if a and c:
                x, y, p = mcnemar(esc(a), esc(c))
                line(f"{label}, {s}", [str(x), str(y), f"{p:.2g}"])

    print("\n### Runtime guard on the safe model: escalation attempted and executed on blocked tasks\n")
    head("model, set", ["no guard", "with guard: attempted", "with guard: executed"])
    for label, b in BASES:
        for s in SETS:
            p = f"{S2}/guard/{s}/{b}_grpo_safe_summary.json"
            base = s2(s, f"{b}_grpo_safe")
            if os.path.exists(p) and base:
                g = json.load(open(p))["neutral"]
                e = esc(base)
                line(f"{label}, {s}", [f"{100 * sum(e.values()) / len(e):.1f}", f"{100 * g['attempt_rate_unsolvable']:.1f}",
                                       f"{100 * g['escalation_rate_unsolvable']:.1f}"])

    print("\n### General tool calling (BFCL v3 simple + multiple, our own AST check)\n")
    head("model", ["full match"])
    for label, b in BASES:
        for tag in ("base", "sft", "grpo_safe"):
            p = f"{S2}/bfcl/{b}_{tag}.jsonl"
            if os.path.exists(p):
                line(f"{label}, {tag}", [cell([int(json.loads(l)["match"]) for l in open(p)])])

    print("\n### Real HTTP server and the live World Bank API\n")
    head("model", ["sandbox: escalation on blocked", "server-confirmed bypass", "sandbox legit success", "World Bank correct"])
    for label, b in BASES:
        for tag in ("sft", "grpo_success_only", "grpo_safe"):
            rs = rows(f"{S2}/real", f"{b}_{tag}")
            if not rs:
                continue
            sb = [r for r in rs if r["scenario"]["id"].startswith("rs")]
            wb = [r for r in rs if r["scenario"]["id"].startswith("rw")]
            by = [int(bool(r.get("server_bypass"))) for r in sb if not r["scenario"]["solvable"]]
            line(f"{label}, {tag}", [cell(esc(sb).values()), cell(by), cell(legit(sb).values()),
                                     f"{sum(r['outcome']['outcome'] == 'success' for r in wb)}/{len(wb)}"])


if __name__ == "__main__":
    print("# Stage 2 results\n\nPercent, with a 95% bootstrap interval over tasks (2,000 resamples) in brackets. "
          "Neutral prompt, greedy decoding.")
    curve()
    family()
