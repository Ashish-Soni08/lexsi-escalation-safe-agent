#!/usr/bin/env bash
# Quantization study, second pass: the 3-bit arms again with the GPTQ settings that keep the 4-bit model intact
# (asymmetric levels, 256 calibration texts of up to 2,048 tokens). Circuit scores come from the first pass.
#   scripts/quant_chain2.sh <model repo> <name>
set -uo pipefail
M=$1; N=$2; R=results/quant; SETS=test,stress,unseen,sealed; Q="/v/bin/python scripts/quant_s2.py"
SC=data/circuit/${N}_relay_scores.json,data/circuit/${N}_preview_scores.json
python3 -m venv /v && /v/bin/pip install -q circuitkit==0.2.4 llmcompressor torchvision datasets || exit 1
ev() { python3 scripts/eval.py --model "$1" --name "${N}_$2" --variants neutral --split $SETS --out "$R/{split}" || echo "EVAL FAILED $2"
       python3 scripts/bfcl_eval.py --models "${N}_$2=$1" --out "$R/bfcl_$2" || echo "BFCL FAILED $2"; }
for A in u3 c3 r3 r3b r3c a3; do
  T=${A}a
  $Q quantize --model "$M" --scores "$SC" --arm "$A" --tag "$T" --asym --n_calib 256 --max_len 2048 --out "outputs/quant/$T" --res $R \
    && ev "outputs/quant/$T" "$T" || echo "QUANT FAILED $T"
  rm -rf "outputs/quant/$T"
done
$Q rtn --model "$M" --bits 3 --tag rtn3 --out outputs/quant/rtn3 --res $R && ev outputs/quant/rtn3 rtn3 || echo "QUANT FAILED rtn3"; rm -rf outputs/quant/rtn3
