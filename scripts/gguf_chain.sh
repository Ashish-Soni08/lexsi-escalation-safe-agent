#!/usr/bin/env bash
# GGUF row of the compression table: the file format people actually download, run by llama.cpp itself.
#   scripts/gguf_chain.sh <model repo> <name>      (image ghcr.io/ggml-org/llama.cpp:full-cuda)
# Converts the model to GGUF (16-bit), quantizes to Q4_K_M, serves each with llama-server on the GPU and runs
# our test sets against it through the OpenAI-style endpoint. The 16-bit GGUF is a check on the runtime itself.
set -uo pipefail
M=$1; N=$2; R=results/quant; G=/g; mkdir -p $G $R
export LD_LIBRARY_PATH=/app:${LD_LIBRARY_PATH:-}
hf download "$M" --local-dir /m >/dev/null || exit 1
# the checkpoint was saved by a newer transformers, whose tokenizer class name the converter's version does not know
python3 - <<'P'
import json
p = "/m/tokenizer_config.json"; d = json.load(open(p))
print("tokenizer_class", d.get("tokenizer_class"))
if d.get("tokenizer_class") == "TokenizersBackend":
    d["tokenizer_class"] = "PreTrainedTokenizerFast"; json.dump(d, open(p, "w"), indent=1, ensure_ascii=False)
P
python3 /app/convert_hf_to_gguf.py /m --outfile $G/f16.gguf --outtype f16 || { echo "GGUF FAILED convert"; exit 1; }
/app/llama-quantize $G/f16.gguf $G/q4.gguf Q4_K_M || { echo "GGUF FAILED quantize"; exit 1; }
ls -la $G
serve() {  # <file>
  /app/llama-server -m "$1" -ngl 999 --parallel 16 -c 131072 --port 8080 --jinja > $R/llama_server_$2.log 2>&1 &
  SP=$!
  for i in $(seq 1 120); do python3 -c "import urllib.request as u; u.urlopen('http://localhost:8080/health', timeout=3)" 2>/dev/null && return 0; sleep 2; done
  echo "GGUF FAILED server $2"; tail -20 $R/llama_server_$2.log; return 1
}
ev() {  # <tag> <sets>
  python3 scripts/eval.py --api openai --base_url http://localhost:8080/v1 --model gguf --name "${N}_$1" --variants neutral \
    --split "$2" --out "$R/{split}" --workers 16 || echo "EVAL FAILED $1"
}
meta() { python3 - "$1" "$2" "$3" <<'P'
import json, os, sys
tag, path, method = sys.argv[1:4]
json.dump({"arm": tag, "bits": 4 if "q4" in path else 16, "method": method, "protect": "none", "protected_layers": [],
           "disk_gb": round(os.path.getsize(path) / 1e9, 2)}, open(f"results/quant/{tag}_quant.json", "w"), indent=1)
P
}
serve $G/q4.gguf q4 && { ev gguf4 test,stress,unseen,sealed; meta gguf4 $G/q4.gguf "GGUF Q4_K_M, llama.cpp"; }
kill $SP; sleep 5
serve $G/f16.gguf f16 && { ev gguf16 test,sealed; meta gguf16 $G/f16.gguf "GGUF 16-bit, llama.cpp"; }
kill $SP
