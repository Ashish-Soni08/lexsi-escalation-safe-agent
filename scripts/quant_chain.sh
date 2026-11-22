#!/usr/bin/env bash
# Stage 2 quantization study for one model, inside one GPU job.
#   scripts/quant_chain.sh <model repo> <name> [arms]
# CircuitKIT and llm-compressor go into their own isolated venv so they cannot change the packages vLLM runs on.
set -uo pipefail
M=$1; N=$2; ARMS=${3:-"u4 u3 c3 r3 a3"}
R=results/quant; SETS=test,stress,unseen,sealed
python3 -m venv /v && /v/bin/pip install -q circuitkit==0.2.4 llmcompressor torchvision datasets || exit 1
/v/bin/python scripts/circuit_compare.py build || exit 1
/v/bin/python scripts/circuit_compare.py run --models "$N=$M" --out $R/circuit || exit 1
SC=$R/circuit/${N}_relay_scores.json,$R/circuit/${N}_preview_scores.json
ev() { python3 scripts/eval.py --model "$1" --name "${N}_$2" --variants neutral --split $SETS --out "$R/{split}" || echo "EVAL FAILED $2"
       python3 scripts/bfcl_eval.py --models "${N}_$2=$1" --out "$R/bfcl_$2" || echo "BFCL FAILED $2"; }
ev "$M" fp
for A in $ARMS; do
  /v/bin/python scripts/quant_s2.py quantize --model "$M" --scores "$SC" --arm "$A" --out "outputs/quant/$A" --res $R || { echo "QUANT FAILED $A"; continue; }
  ev "outputs/quant/$A" "$A"
  rm -rf "outputs/quant/$A"
done
python3 scripts/quant_s2.py report --res $R --name "$N"
