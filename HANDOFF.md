# Handoff: escalation-safe tool-using agent (Lexsi idea E, Track 3)

Status 2026-11-12 ~11:00 UTC. Round 4 is done and published. Recommended model:
`ashish-soni08/tiny-aya-global-esa-grpo-safe-v4-1` (escalation on blocked tasks, greedy, neutral prompt: test 0.0%,
stress 1.5%, unseen stress 25.9%; legit success 99.0 / 99.5 / 99.0%).

Round 5 (2026-11-12 evening), a negative result
- `randrl` (seed 8) and `fresh` (seed 9) splits: alternate routes with random names, sanctioned or not (`Scenario.alts`,
  `world.make_alt_scenarios`); scoring adds simulator ground truth for them (`scoring.episode_flags`).
- v5 = v4.1 + 160 GRPO steps on `grpo_prompts_v5.jsonl`: same as v4.1 within noise (fresh 11.8% vs 11.0%, unseen
  22.4% vs 25.9% greedy, 25.6 vs 25.3 sampled). Cause: the agent opens the provider front page in ~2% of episodes, so
  it never reads the route descriptions. v5 repo `-grpo-safe-v5` is PRIVATE and not recommended.
- Runtime guard (`eval.py --guard`, results in `results/guard/<split>/`): v4.1 unseen 24.1% attempted, 0% executed;
  no effect on fresh.
- Fourth judge audit `results/judge_v5` (run with `--routes`): 93.0%, kappa 0.86.
- `rescore.py` skips guard runs (their rows keep refused calls). eval.py takes a comma list of splits and
  `{split}` in `--out`; job outputs land in `g_<split>`, `s_<split>`, `guard_<split>` and are copied by hand to
  `results/`, `results/<split>`, `results/seeds_<split>`, `results/guard/<split>`.
- If someone continues: add SFT demonstrations where the expert reads the front page first, then rerun the v5 RL.
- The dashboard page `space/index.html` is owned by the "Make both dashboards easy to understand" thread; this thread
  only regenerates `space/data.json` (`headline` key picks the headline model).

Round 4 in short
- Second stress test `unseen` (seed 6): four shortcut kinds, four tip wordings and two tip channels never trained on.
  Training mix `mixrl` (seed 7) and `data/grpo_prompts_v4.jsonl`.
- v4 = v3.1 + 160 GRPO steps on the wider mix with detector v2.1 in the reward. A second judge audit found v2.1 missed
  a bare `/debug` path (regex bug), which v3, v3.1 and v4 all used. Detector is now v2.2 (parsed-URL rules); every saved
  eval file was rescored (`scripts/rescore.py`), which moved the published stress numbers: v2 73.1 -> 78.5%, v3 16.2 ->
  52.3%, v3.1 6.9 -> 36.2%. v4.1 = same recipe with v2.2 in the reward. README, write-up, dashboard, model cards and
  dataset card all carry the correction.
- Three judge audits: `results/judge`, `judge_v4`, `judge_v4_1`. On v4.1 all 48 disagreements are judge-only and none
  contains a suspicious call.
- Open weakness: v4.1 still takes the advertised "origin" host on the unseen set (all 30 of its escalations).
- GRPO rollout trajectories for v4 / v4.1 are only in the HF dataset (`grpo_rollouts/`), not in git.

- GPU: Hugging Face Jobs via `scripts/job.sh` (image vllm/vllm-openai:v0.22.1). Scratch dataset `ashish-soni08/esa-workdir`
  (private) holds the code snapshot and every run's outputs under `runs/<tag>/` (v2 runs have the `_v2` suffix).
- Spend: ~$48 on HF Jobs (cap $50, raised by Ashish on 2026-11-12). No Modal used.
- Public models also: `-grpo-safe-v4` (control: same recipe, old detector), `-grpo-safe-v3-1`, `-grpo-safe-v3` (both
  superseded, kept as reward-hacking examples), `-grpo-esc-safe-v2-seed2`.
- Everything in `results/` is scored with detector v2.2; the runs' own eval files in `esa-workdir` use whatever
  detector was current when they ran.
- CircuitKIT needs a local checkpoint directory for fine-tuned Tiny Aya (its cohere2 port matches Hub ids only for the
  base names); `scripts/circuit_compare.py` snapshot-downloads first.
- Cohere's gated models (Aya Expanse, Command R7B) are not accepted on this HF token; the judge uses Qwen2.5-32B.
- Models v2: `ashish-soni08/tiny-aya-global-esa-sft-v2` and `-grpo-esc-safe-v2` (public), `-sft-esc-v2`,
  `-grpo-esc-success_only-v2` (private). v1 repos (no suffix) are superseded: v1 had the mirror bug.
- Results: README "Results", `results/results_table.md` (`scripts/report.py`), `results/auditkit_*/summary.md`,
  `assets/*.png`. v1 results in `results/v1/`. Write-up: `docs/hf_article.md` (not posted on HF, Ashish's call).
- Aya Expanse 8B scaling check: on hold until Ashish settles on a final model. If run, `esa/rollout.py` must take the
  turn-end tokens from the chat template (Aya Expanse ends turns with `<|END_OF_TURN_TOKEN|>` only).

Gotchas
- Don't `pip install -U huggingface_hub` in the image (2.x breaks transformers 5.22).
- AgentTune's built-in tool rollout gives zero GRPO gradient for Tiny Aya (see README); use `esa/rollout.py`.
- GRPO OOMs at per-device batch 8 (262k vocab logits); use `--bs 2 --ga 16`.
- AuditKIT's flood check sees no per-call results in the trajectory format, so it counts repeated identical calls as a
  flood even when they were not blocked (base model 0.3% vs 0.0% in our own scorer).
