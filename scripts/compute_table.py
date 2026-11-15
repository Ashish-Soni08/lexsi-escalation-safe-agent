"""Compute table for the write-up, from the recorded Hugging Face Jobs history.

    python scripts/compute_table.py            # prints the markdown table from results/hf_jobs.json
    python scripts/compute_table.py --refresh  # re-reads the job history from the HF Jobs API first

Cost is running seconds times the list price of the hardware. Jobs are the ones named esa-*.
"""
import json
import os
import sys

PRICE = {"a100-large": 2.50, "a10g-large": 1.50, "l4x1": 0.80}  # USD per hour, HF Jobs list price
HW = {"a100-large": "A100 80GB", "a10g-large": "A10G 24GB", "l4x1": "L4 24GB"}
PATH = os.path.join(os.path.dirname(__file__), "..", "results", "hf_jobs.json")

WHAT = {
    "smoke1": "Environment check and first base-model evaluation",
    "smoke2": "Base-model smoke evaluation (failed)",
    "smoke3": "Base-model smoke evaluation, rerun",
    "sft1": "Clean tool SFT, first try (failed at start)",
    "sft2": "Clean tool SFT, v1 data",
    "eval_base": "Base model on the v1 test set",
    "grpo_smoke1": "GRPO smoke run, 3 steps",
    "eval_sft": "Clean SFT model on the v1 test set",
    "grpo_smoke2": "GRPO smoke run, 4 steps",
    "sft_esc": "Escalation-prone SFT, v1 data",
    "grpo_esc_success_only": "GRPO with the success-only reward, v1 (120 steps)",
    "grpo_esc_safe": "GRPO with the safe reward, v1 (120 steps)",
    "sft_v2": "Clean tool SFT, v2 data, with evaluation",
    "sft_esc_v2": "Escalation-prone SFT, v2 data, with evaluation",
    "eval_base_v2": "Base model on the v2 test set",
    "grpo_esc_safe_v2": "GRPO with the safe reward, v2 (160 steps)",
    "grpo_esc_success_only_v2": "GRPO with the success-only reward, v2 (160 steps)",
    "evalx": "Extended evaluation of all models (cancelled before start)",
    "grpo_seed2": "Second training seed of the v2 safe arm",
    "circuit": "CircuitKIT comparison (failed at start)",
    "evalx2": "Extended evaluation of all models (failed)",
    "circuit2": "CircuitKIT comparison (failed at start)",
    "circuit3": "CircuitKIT comparison of the escalation-prone, success-only and safe models",
    "evalx3": "Extended evaluation of all models: stress test 1, sampled seeds, AuditKIT",
    "grpo_v3": "GRPO v3, wider training mix",
    "judge": "Qwen judge audit of v2",
    "grpo_v3_1": "GRPO v3.1, detector patched",
    "unseen_base": "Stress test 2 (unseen shortcut kinds) for all models",
    "grpo_v4": "GRPO v4, wider mix again",
    "judge_v4": "Qwen judge audit of v4",
    "grpo_v4_1": "GRPO v4.1, the released model",
    "judge_v4_1": "Qwen judge audit of v4.1",
    "v5": "GRPO v5 (not released), with evaluation and guard runs",
    "fresh_base": "Fresh set for all models",
    "judge_v5": "Qwen judge audit of v5",
    "seed_14": "v4.1 recipe, training seed 14",
    "seed_24": "v4.1 recipe, training seed 24",
    "abl_no_escalation_penalty": "Reward ablation: no escalation penalty (first run)",
    "abl_no_honest_stop_bonus": "Reward ablation: no honest-stop bonus",
    "abl_no_fabrication_penalty": "Reward ablation: no fabrication penalty",
    "guard_only": "Guard-only baseline and guard runs",
    "sealed_long": "Sealed test and long tasks for all models",
    "bfcl": "BFCL function-calling check for base, SFT and v4.1",
    "real": "Real-server evaluation",
    "longchat": "Long tasks in chat format",
    "real2": "Real-server evaluation, rerun",
    "judge_v6": "Qwen and Mistral judge audits of v4.1",
    "abl_no_escalation_penalty_2": "Reward ablation: no escalation penalty (rerun)",
    "seeds_extra": "Seed models on sealed, long and the other sets",
    "judge_v6_seeds": "Qwen judge audit of the two seed models",
}


def refresh():
    from huggingface_hub import HfApi
    rows = []
    for j in HfApi().list_jobs():
        name = (j.labels or {}).get("name", "")
        if not name.startswith("esa-"):
            continue
        rows.append({"id": j.id, "tag": name[4:], "flavor": str(j.flavor), "created_at": j.created_at.isoformat(),
                     "stage": j.status.stage, "running_secs": (j.durations.running_secs if j.durations else 0) or 0})
    rows.sort(key=lambda r: r["created_at"])
    json.dump(rows, open(PATH, "w"), indent=1)


def main():
    if "--refresh" in sys.argv:
        refresh()
    rows = json.load(open(PATH))
    print("| # | date (UTC) | job | what it ran | hardware | status | GPU-hours | cost |")
    print("|---|---|---|---|---|---|---|---|")
    hours = cost = 0.0
    for i, r in enumerate(rows, 1):
        h = r["running_secs"] / 3600
        c = h * PRICE[r["flavor"]]
        hours += h
        cost += c
        print(f"| {i} | {r['created_at'][5:10]} {r['created_at'][11:16]} | `{r['tag']}` | {WHAT[r['tag']]} | "
              f"{HW[r['flavor']]} | {r['stage'].lower()} | {h:.2f} | ${c:.2f} |")
    print(f"| | | | **Total, {len(rows)} jobs** | | | **{hours:.1f}** | **${cost:.2f}** |")


if __name__ == "__main__":
    main()
