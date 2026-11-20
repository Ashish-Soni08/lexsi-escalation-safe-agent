#!/usr/bin/env bash
# Stage 2: the whole recipe for one Aya base model inside one job, then the Stage 1 test suite.
#   scripts/stage2_chain.sh <base model> <short name> <repo prefix> [steps]
#   e.g. scripts/stage2_chain.sh CohereLabs/tiny-aya-fire fire ashish-soni08/tiny-aya-fire-esa
# steps (default all): sft sft_esc success safe eval_safe eval_success eval_base
#   eval_core instead of eval_safe: the safe model on the five main sets only (coverage-curve arms)
# env: COV=2|4|6|10 shortcut kinds in the RL prompts (default 10); ESC=<repo> start model for RL
#      (default <prefix>-sft-esc); GRPO_ARGS
# Every trained model is pushed (private) as soon as it exists, and a step is skipped when its repo
# is already there, so a failed job can be started again without paying twice.
set -uo pipefail
BASE=$1; S=$2; P=$3; STEPS=${4:-"sft sft_esc success safe eval_safe eval_success eval_base"}
COV=${COV:-10}; ESC=${ESC:-$P-sft-esc}
[ "$COV" = 10 ] && { CR=""; CN=""; } || { CR="-cov$COV"; CN="_cov$COV"; }
GRPO_ARGS=${GRPO_ARGS:-"--max_steps 160 --lr 2e-5 --bs 2 --ga 16 --seed 4"}
SETS=test,stress,unseen,fresh,sealed
has() { [[ " $STEPS " == *" $1 "* ]]; }
exists() { python3 -c "import sys; from huggingface_hub import HfApi; sys.exit(0 if HfApi().file_exists('$1', 'config.json') else 1)" 2>/dev/null; }
sft() {  # <data file> <out> <repo>
  exists "$3" && { echo "skip $3"; return 0; }
  python3 scripts/train_sft.py --base "$BASE" --data "$1" --out "outputs/$2" --push "$3" || return 1
  rm -rf "outputs/$2/merged" "outputs/$2/ckpt"
}
grpo() {  # <reward> <out> <repo>
  exists "$3" && { echo "skip $3"; return 0; }
  python3 scripts/train_grpo.py --model "$ESC" --reward "$1" --prompts "grpo_prompts_cov$COV.jsonl" \
    --scenarios "scenarios_cov$COV.jsonl" $GRPO_ARGS --out "outputs/$2" --push "$3" || return 1
  rm -rf "outputs/$2/merged" outputs/$2/checkpoint-*
}
ev() { python3 scripts/eval.py "$@" || echo "EVAL FAILED: $*"; }

pip install -q peft datasets
if has sft; then sft data/sft_trajectories.jsonl sft "$P-sft" || exit 1; fi
if has sft_esc; then sft data/sft_esc_trajectories.jsonl sft_esc "$P-sft-esc" || exit 1; fi
pip install -q agenttune==1.1.4 trl==1.7.1
if has success; then grpo success_only grpo_success_only "$P$CR-grpo-success-only" || exit 1; fi
if has safe; then grpo safe grpo_safe "$P$CR-grpo-safe" || exit 1; fi
R=results/s2
SAFE="$P$CR-grpo-safe"; SN="${S}${CN}_grpo_safe"
if has eval_core; then ev --model "$SAFE" --name "$SN" --variants neutral --split $SETS --out "$R/{split}"; fi
if has eval_safe; then  # every Stage 1 check
  ev --model "$SAFE" --name "$SN" --variants rules,neutral --split $SETS --out "$R/{split}"
  ev --model "$SAFE" --name "$SN" --variants rules,neutral --split long --chat --max_turns 30 --out "$R/longchat"
  ev --model "$SAFE" --name "$SN" --variants neutral --split $SETS --guard --out "$R/guard/{split}"
  python3 scripts/real_eval.py --model "$SAFE" --name "$SN" --variants neutral --tasks data/real_tasks.jsonl --out "$R/real" || echo "REAL FAILED"
  python3 scripts/bfcl_eval.py --models "$SN=$SAFE" --out "$R/bfcl" || echo "BFCL FAILED"
fi
if has eval_success; then
  ev --model "$P$CR-grpo-success-only" --name "${S}${CN}_grpo_success_only" --variants neutral --split $SETS --out "$R/{split}"
  python3 scripts/real_eval.py --model "$P$CR-grpo-success-only" --name "${S}${CN}_grpo_success_only" --variants neutral --tasks data/real_tasks.jsonl --out "$R/real" || echo "REAL FAILED"
fi
if has eval_base; then  # what the tables compare the safe model with
  ev --model "$P-sft" --name "${S}_sft" --variants neutral --split $SETS --out "$R/{split}"
  ev --model "$P-sft" --name "${S}_sft" --variants neutral --split long --chat --max_turns 30 --out "$R/longchat"
  ev --model "$P-sft" --name "${S}_sft" --variants neutral --split $SETS --guard --out "$R/guard/{split}"
  ev --model "$P-sft-esc" --name "${S}_sft_esc" --variants neutral --split test --out "$R/{split}"
  ev --model "$BASE" --name "${S}_base" --variants neutral --split test --out "$R/{split}"
  python3 scripts/real_eval.py --model "$P-sft" --name "${S}_sft" --variants neutral --tasks data/real_tasks.jsonl --out "$R/real" || echo "REAL FAILED"
  python3 scripts/bfcl_eval.py --models "${S}_base=$BASE,${S}_sft=$P-sft" --out "$R/bfcl" || echo "BFCL FAILED"
fi
