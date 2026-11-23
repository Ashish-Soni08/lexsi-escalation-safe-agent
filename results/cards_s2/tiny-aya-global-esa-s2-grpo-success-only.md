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

# tiny-aya-global-esa-s2-grpo-success-only

Tiny Aya Global, GRPO, success-only reward. Part of the Stage 2 experiments of *Escalation-safe tool-using Aya* (Lexsi AI Safety & Alignment Hackathon, Track 3, Agentic Fine-tuning).

160 GRPO steps from the escalation-prone model with a reward for task success only. **Deliberately unsafe comparison model**: it shows that an outcome-only reward teaches the agent to go around access controls. Do not deploy.

The project asks one question: when a tool-using agent hits an access control (bot wall, login, captcha, rate limit), does it stop honestly or go around it? The answer depends on the reward. The same recipe is run on Tiny Aya Global, Fire, Water and Earth and on Aya Expanse 8B.

## Results

Simulated web sandbox, neutral system prompt, greedy decoding. "Blocked" tasks cannot be solved without going around an access control; the right behaviour is to stop and say so. The unseen and sealed sets use shortcut kinds never seen in training.

| set | escalation on blocked tasks | legitimate success on solvable tasks |
|---|---|---|
| test | 74.2% | 99.5% |
| stress | 99.2% | 60.0% |
| unseen | 90.5% | 78.4% |
| fresh | 77.2% | 70.5% |
| sealed | 77.5% | 80.8% |

Full tables with confidence intervals: `results/report_s2.md` in the [GitHub repository](https://github.com/Ashish-Soni08/lexsi-escalation-safe-agent).

## Run it

Same tool format and agent loop as the Stage 1 model: tool JSON schemas in the system prompt, the model replies with a JSON
list of calls, tool results come back as a user turn, and it ends with `ANSWER: <value>` or `BLOCKED: <reason>`
(`esa/prompts.py`, loop in `esa/runner.py`; a `transformers` example is on the
[v4.1 card](https://huggingface.co/ashish-soni08/tiny-aya-global-esa-grpo-safe-v4-1)). To run it on a test set:

```bash
git clone https://github.com/Ashish-Soni08/lexsi-escalation-safe-agent && cd lexsi-escalation-safe-agent
pip install vllm   # or the vllm/vllm-openai:v0.22.1 image
python scripts/eval.py --model ashish-soni08/tiny-aya-global-esa-s2-grpo-success-only --name try --split sealed --variants neutral --n 40 --out results/try
```

## Reproduce

One Hugging Face Jobs run on an A100 80GB (setup and costs in the [README](https://github.com/Ashish-Soni08/lexsi-escalation-safe-agent#reproduce-training-on-hugging-face-jobs)):

```bash
scripts/job.sh s2_global a100-large 6h vllm/vllm-openai:v0.22.1 \
  "ESC=ashish-soni08/tiny-aya-global-esa-sft-esc-v2 bash scripts/stage2_chain.sh CohereLabs/tiny-aya-global global <you>/tiny-aya-global-esa-s2 'success safe eval_safe eval_success'"
```

## Training

* Base: [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global), merged LoRA weights.
* Recipe: clean SFT and escalation-prone SFT (200 steps), then 160 GRPO steps (AgentTune, lr 2e-5, seed 4) on 960 simulated tasks that offer 10 shortcut kinds (fewer in the coverage-curve arms).
* Code, data generator, test sets and logs: https://github.com/Ashish-Soni08/lexsi-escalation-safe-agent

## Limits

Research artefact. Trained and tested in a simulated web sandbox with fictional sites, plus a small real-server check. The safe models still escalate on some never-trained shortcut kinds (see the table), so they need a runtime guard and are not a security control on their own. Subject to the base model's licence (CC-BY-NC 4.0).
