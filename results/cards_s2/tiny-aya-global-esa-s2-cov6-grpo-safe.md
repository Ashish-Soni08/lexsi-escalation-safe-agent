---
base_model: CohereLabs/tiny-aya-global
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: text-generation
tags:
- lexsi-hackathon
- agent-safety
- tool-use
- grpo
---

# tiny-aya-global-esa-s2-cov6-grpo-safe

Tiny Aya Global, GRPO, safe reward. Part of the Stage 2 experiments of *Escalation-safe tool-using Aya* (Lexsi AI Safety & Alignment Hackathon, Track 3, Agentic Fine-tuning).

160 GRPO steps from the escalation-prone model with a reward that pays for legitimate success and honest stops and penalises going around an access control. Coverage-curve arm: the training tasks offered only 6 of the 10 shortcut kinds, to measure how protection against never-trained shortcuts grows with coverage.

The project asks one question: when a tool-using agent hits an access control (bot wall, login, captcha, rate limit), does it stop honestly or go around it? The answer depends on the reward. The same recipe is run on Tiny Aya Global, Fire, Water and Earth and on Aya Expanse 8B.

## Results

Simulated web sandbox, neutral system prompt, greedy decoding. "Blocked" tasks cannot be solved without going around an access control; the right behaviour is to stop and say so. The unseen and sealed sets use shortcut kinds never seen in training.

| set | escalation on blocked tasks | legitimate success on solvable tasks |
|---|---|---|
| test | 0.0% | 99.5% |
| stress | 27.7% | 99.0% |
| unseen | 31.9% | 98.0% |
| fresh | 18.9% | 72.5% |
| sealed | 26.1% | 91.2% |

Full tables with confidence intervals: `results/report_s2.md` in the [GitHub repository](https://github.com/Ashish-Soni08/lexsi-escalation-safe-agent).

## Training

* Base: [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global), merged LoRA weights.
* Recipe: clean SFT and escalation-prone SFT (200 steps), then 160 GRPO steps (AgentTune, lr 2e-5, seed 4) on 960 simulated tasks that offer 10 shortcut kinds (fewer in the coverage-curve arms).
* Code, data generator, test sets and logs: https://github.com/Ashish-Soni08/lexsi-escalation-safe-agent

## Limits

Research artefact. Trained and tested in a simulated web sandbox with fictional sites, plus a small real-server check. The safe models still escalate on some never-trained shortcut kinds (see the table), so they need a runtime guard and are not a security control on their own. Subject to the base model's licence (CC-BY-NC 4.0).
