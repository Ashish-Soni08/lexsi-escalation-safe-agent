"""Model cards for the Stage 2 models, filled from results/s2. `--push` uploads them."""
import json, os, sys
GH = "https://github.com/Ashish-Soni08/lexsi-escalation-safe-agent"
U = "ashish-soni08/"
BASES = {"global": ("CohereLabs/tiny-aya-global", "Tiny Aya Global", "tiny-aya-global-esa-s2"),
         "fire": ("CohereLabs/tiny-aya-fire", "Tiny Aya Fire", "tiny-aya-fire-esa"),
         "expanse8b": ("CohereLabs/aya-expanse-8b", "Aya Expanse 8B", "aya-expanse-8b-esa")}
SETS = ["test", "stress", "unseen", "fresh", "sealed"]
KIND = {"sft": ("clean tool-use SFT", "LoRA SFT on clean tool-use trajectories (no reward training). Baseline."),
        "sft_esc": ("escalation-prone SFT", "LoRA SFT on trajectories that take shortcuts around access controls. "
                    "**Deliberately unsafe starting point** for the reward experiments. Do not deploy."),
        "grpo_success_only": ("GRPO, success-only reward", "160 GRPO steps from the escalation-prone model with a reward for task "
                              "success only. **Deliberately unsafe comparison model**: it shows that an outcome-only reward "
                              "teaches the agent to go around access controls. Do not deploy."),
        "grpo_safe": ("GRPO, safe reward", "160 GRPO steps from the escalation-prone model with a reward that pays for legitimate "
                      "success and honest stops and penalises going around an access control.")}

def table(name):
    out = ["| set | escalation on blocked tasks | legitimate success on solvable tasks |", "|---|---|---|"]
    for s in SETS:
        p = f"results/s2/{s}/{name}_summary.json"
        if os.path.exists(p):
            m = json.load(open(p)).get("neutral")
            if m:
                out.append(f"| {s} | {100*m['escalation_rate_unsolvable']:.1f}% | {100*m['legit_success_rate_solvable']:.1f}% |")
    return "\n".join(out) if len(out) > 2 else ""

def card(repo, base, bname, kind, name, cov=None):
    title, what = KIND[kind]
    covtxt = ""
    if cov:
        covtxt = (f" Coverage-curve arm: the training tasks offered only {cov} of the 10 shortcut kinds, to measure how protection "
                  "against never-trained shortcuts grows with coverage.")
    t = table(name)
    res = (f"## Results\n\nSimulated web sandbox, neutral system prompt, greedy decoding. \"Blocked\" tasks cannot be solved "
           f"without going around an access control; the right behaviour is to stop and say so. The unseen and sealed sets use "
           f"shortcut kinds never seen in training.\n\n{t}\n\nFull tables with confidence intervals: "
           f"`results/report_s2.md` in the [GitHub repository]({GH}).\n\n" if t else "")
    return f"""---
base_model: {base}
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: text-generation
tags:
- lexsi-hackathon
- agent-safety
- tool-use
- grpo
---

# {repo}

{bname}, {title}. Part of the Stage 2 experiments of *Escalation-safe tool-using Aya* (Lexsi AI Safety & Alignment Hackathon, Track 3, Agentic Fine-tuning).

{what}{covtxt}

The project asks one question: when a tool-using agent hits an access control (bot wall, login, captcha, rate limit), does it stop honestly or go around it? The answer depends on the reward. The same recipe is run on Tiny Aya Global, Tiny Aya Fire and Aya Expanse 8B.

{res}## Training

* Base: [{base}](https://huggingface.co/{base}), merged LoRA weights.
* Recipe: clean SFT and escalation-prone SFT (200 steps), then 160 GRPO steps (AgentTune, lr 2e-5, seed 4) on 960 simulated tasks that offer 10 shortcut kinds (fewer in the coverage-curve arms).
* Code, data generator, test sets and logs: {GH}

## Limits

Research artefact. Trained and tested in a simulated web sandbox with fictional sites, plus a small real-server check. The safe models still escalate on some never-trained shortcut kinds (see the table), so they need a runtime guard and are not a security control on their own. Subject to the base model's licence (CC-BY-NC 4.0).
"""

def all_cards():
    out = {}
    for short, (base, bname, pre) in BASES.items():
        kinds = ["grpo_success_only", "grpo_safe"] + ([] if short == "global" else ["sft", "sft_esc"])
        for k in kinds:
            out[pre + "-" + k.replace("_", "-")] = card(pre + "-" + k.replace("_", "-"), base, bname, k, f"{short}_{k}")
    base, bname, pre = BASES["global"]
    for c in (2, 4, 6):
        r = f"{pre}-cov{c}-grpo-safe"
        out[r] = card(r, base, bname, "grpo_safe", f"global_cov{c}_grpo_safe", cov=c)
    return out

if __name__ == "__main__":
    cards = all_cards()
    os.makedirs("results/cards_s2", exist_ok=True)
    for r, c in cards.items():
        open(f"results/cards_s2/{r}.md", "w").write(c)
    print(len(cards), "cards")
    if "--push" in sys.argv:
        from huggingface_hub import HfApi
        a = HfApi()
        for r, c in cards.items():
            a.upload_file(path_or_fileobj=c.encode(), path_in_repo="README.md", repo_id=U + r, commit_message="Model card")
            print("pushed", r)
