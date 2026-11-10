# Handoff: escalation-safe tool-using agent (Lexsi idea E, Track 3)

Status 2026-11-10 ~20:15 UTC. Training and evaluation are done; publishing is pending Ashish's go.

- GPU: Hugging Face Jobs via `scripts/job.sh` (image vllm/vllm-openai:v0.22.1). Scratch dataset `ashish-soni08/esa-workdir`
  (private) holds the code snapshot and every run's outputs under `runs/<tag>/`.
- Spend: ~$9 on HF Jobs (budget $40). No Modal used.
- Models (private): `ashish-soni08/tiny-aya-global-esa-sft`, `-sft-esc`, `-grpo-esc-success_only`, `-grpo-esc-safe`.
- Results: README "Results", `results/results_table.md`, `results/auditkit_*/summary.md`, `assets/*.png`.
  Full per-episode files are in the workdir dataset (gitignored here).

Gotchas
- Don't `pip install -U huggingface_hub` in the image (2.x breaks transformers 5.22).
- AgentTune's built-in tool rollout gives zero GRPO gradient for Tiny Aya (see README); use `esa/rollout.py`.
- GRPO OOMs at per-device batch 8 (262k vocab logits); use `--bs 2 --ga 16`.
- AuditKIT's flood check sees no per-call results in the trajectory format, so it counts repeated identical calls as a
  flood even when they were not blocked (base model 0.3% vs 0.0% in our own scorer).
