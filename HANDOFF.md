# Handoff: escalation-safe tool-using agent (Lexsi idea E, Track 3)

Status 2026-11-11 ~20:30 UTC. v2 training and evaluation are done (see README "v2"); cards/publishing in progress.

- GPU: Hugging Face Jobs via `scripts/job.sh` (image vllm/vllm-openai:v0.22.1). Scratch dataset `ashish-soni08/esa-workdir`
  (private) holds the code snapshot and every run's outputs under `runs/<tag>/` (v2 runs have the `_v2` suffix).
- Spend: ~$18 on HF Jobs (budget $40). No Modal used.
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
