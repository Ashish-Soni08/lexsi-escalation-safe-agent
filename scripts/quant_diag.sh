#!/usr/bin/env bash
# Checks on the quantization study: is the damage at 4-bit a property of the model or of one-shot GPTQ?
#   scripts/quant_diag.sh <model repo> <name>
# 1. bitsandbytes NF4 4-bit   2. round-to-nearest 4-bit   3. GPTQ 4-bit loaded by the server from the packed
# file (to compare with the dense copy the study uses)   4. GPTQ 4-bit, asymmetric, more calibration text
set -uo pipefail
M=$1; N=$2; R=results/quant; SETS=test,stress,unseen,sealed; Q="/v/bin/python scripts/quant_s2.py"
python3 -m venv /v && /v/bin/pip install -q circuitkit==0.2.4 llmcompressor torchvision datasets bitsandbytes accelerate || exit 1
ev() { python3 scripts/eval.py --model "$1" --name "${N}_$2" --variants neutral --split ${3:-$SETS} --out "$R/{split}" || echo "EVAL FAILED $2"
       [ -n "${3:-}" ] || python3 scripts/bfcl_eval.py --models "${N}_$2=$1" --out "$R/bfcl_$2" || echo "BFCL FAILED $2"; }
$Q bnb --model "$M" --tag bnb4 --out outputs/quant/bnb4 --res $R && ev outputs/quant/bnb4 bnb4 || echo "QUANT FAILED bnb4"; rm -rf outputs/quant/bnb4
$Q rtn --model "$M" --bits 4 --tag rtn4 --out outputs/quant/rtn4 --res $R && ev outputs/quant/rtn4 rtn4 || echo "QUANT FAILED rtn4"; rm -rf outputs/quant/rtn4
$Q quantize --model "$M" --arm u4 --tag u4chk --keep_packed --out outputs/quant/u4chk --res $R && { ev outputs/quant/u4chk_packed u4packed test; ev outputs/quant/u4chk u4chk test; } || echo "QUANT FAILED u4chk"
rm -rf outputs/quant/u4chk outputs/quant/u4chk_packed
$Q quantize --model "$M" --arm u4 --tag u4a --asym --n_calib 256 --max_len 2048 --out outputs/quant/u4a --res $R && ev outputs/quant/u4a u4a || echo "QUANT FAILED u4a"; rm -rf outputs/quant/u4a
