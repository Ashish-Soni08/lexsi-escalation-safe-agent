#!/usr/bin/env bash
# Phase 1 of the soft-reward comparison: grade the saved episodes in data/soft/sample.jsonl with a checklist,
# once with an outside judge and once with the policy's own start model. No training.
set -uo pipefail
JUDGE=${JUDGE:-Qwen/Qwen2.5-32B-Instruct}; SELF=${SELF:-ashish-soni08/tiny-aya-global-esa-sft-esc-v2}
python3 scripts/soft_reward.py run --verifier "$JUDGE" --role judge --make_checklists ${SOFT_ARGS:-} || exit 1
python3 scripts/soft_reward.py run --verifier "$SELF" --role self ${SOFT_ARGS:-} || echo "SELF FAILED"
python3 scripts/soft_reward.py report
