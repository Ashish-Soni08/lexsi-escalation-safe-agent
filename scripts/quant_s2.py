"""Stage 2: does compression undo the safety training, and does protecting the circuit help?

One model, several weight-quantized copies, all made with the same one-shot GPTQ call (llm-compressor, through
CircuitKIT's `llmcompressor_circuit_quantize`) and the same calibration text, so the arms differ only in bit
width and in which transformer layers are left at native precision:

    u4  every layer 4-bit                      (the usual "just quantize it" baseline)
    u3  every layer 3-bit
    c3  3-bit, the layers CircuitKIT ranks highest for the shortcut decision are protected
    r3  3-bit, the same number of layers protected, picked at random   (control for "any protection helps")
    a3  3-bit, the same number of layers protected, the LOWEST ranked  (control for "the ranking matters")

The circuit scores come from scripts/circuit_compare.py (EAP-IG at the token where the agent picks a host
after a block); the relay and preview tasks are summed per node.

    python scripts/quant_s2.py quantize --model <repo> --scores a_scores.json,b_scores.json --arm c3 --out outputs/quant/c3
    python scripts/quant_s2.py report --res results/quant
"""
import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARMS = {"u4": (4, "none"), "u3": (3, "none"), "c3": (3, "circuit"), "r3": (3, "random"), "a3": (3, "anti"),
        "c4": (4, "circuit"), "r3b": (3, "random"), "r3c": (3, "random")}
RANDOM_SEED = {"r3": 0, "r3b": 1, "r3c": 2}  # three draws of the random-protection control


def load_scores(paths):
    heads, mlps = {}, {}
    for p in paths.split(","):
        for name, v in json.load(open(p)).items():
            a = re.match(r"[Aa](\d+)\.[hH]?(\d+)$", name)
            m = re.match(r"MLP\s*(\d+)$", name)
            if a:
                k = (int(a.group(1)), int(a.group(2)))
                heads[k] = heads.get(k, 0.0) + abs(float(v))
            elif m:
                mlps[int(m.group(1))] = mlps.get(int(m.group(1)), 0.0) + abs(float(v))
    return heads, mlps


def calibration_texts(tok, n):
    """Our own tool-use episodes, rendered with the chat template: the distribution the model is used on."""
    rows = [json.loads(l) for l in open("data/sft_trajectories.jsonl")]
    random.Random(0).shuffle(rows)
    out = []
    for r in rows[:n]:
        msgs = r.get("messages") or r.get("conversation") or r
        out.append(tok.apply_chat_template(msgs, tokenize=False))
    return out


def dir_gb(path):
    return round(sum(os.path.getsize(f) for f in glob.glob(f"{path}/*.safetensors")) / 1e9, 2)


def quantize(a):
    import torch
    from circuitkit.applications.quantization.llmcompressor_quantize import (_resolve_protected_layers,
                                                                             llmcompressor_circuit_quantize)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bits, protect = ARMS[a.arm]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto")
    n_layers = model.config.num_hidden_layers
    heads, mlps = load_scores(a.scores) if a.scores else ({}, {})
    ranked = _resolve_protected_layers(heads, mlps, n_layers, a.high_fraction, "mean", None) if a.scores else []
    if protect == "none":
        layers = []
    elif protect == "circuit":
        layers = ranked
    elif protect == "random":
        layers = sorted(random.Random(RANDOM_SEED.get(a.arm, a.seed)).sample(range(n_layers), len(ranked)))
    else:  # anti: the lowest-ranked layers by the same per-layer score
        per = {l: sum(v for (ll, _), v in heads.items() if ll == l) / max(1, sum(1 for (ll, _) in heads if ll == l))
               + mlps.get(l, 0.0) for l in range(n_layers)}
        layers = sorted(sorted(range(n_layers), key=lambda l: per[l])[:len(ranked)])
    # the arms pass the layer list explicitly, so CircuitKIT's own selection (high_fraction) is switched off
    # with a zero score table; `ranked` above is exactly what it would have picked
    info = llmcompressor_circuit_quantize(
        model, tok, q_head_scores={(l, 0): 0.0 for l in range(n_layers)}, mlp_scores={l: 0.0 for l in range(n_layers)},
        n_layers=n_layers, bits=bits, high_fraction=0.0, protect_layers=layers, symmetric=not a.asym,
        n_calibration_samples=a.n_calib, max_seq_length=a.max_len, calibration_texts=calibration_texts(tok, a.n_calib))
    if sorted(info["protected_layers"]) != layers:
        raise SystemExit(f"protected layers differ: wanted {layers}, got {info['protected_layers']}")
    # 1. take the weights first. GPTQ leaves every quantized matrix holding exactly the values the low-bit
    #    file decodes to, so an ordinary checkpoint with these weights behaves as the quantized model does,
    #    and the test server (which has no 3-bit loader) can run every arm the same way.
    import shutil
    full = model.state_dict()
    scales = [k for k in full if k.endswith(".weight_scale")]
    if not scales:
        raise SystemExit("no weight_scale tensors found; keys look like: " + ", ".join(list(full)[:12]))
    probe = scales[len(scales) // 2][:-len("_scale")]
    levels = max(int(torch.unique(full[probe][r, :128].float()).numel()) for r in range(4))
    if levels > 2 ** bits:
        raise SystemExit(f"{probe}: {levels} distinct values in one group, more than {2 ** bits}: weights are not quantized")
    sd = {k: v.detach().to("cpu", torch.bfloat16).clone() for k, v in full.items()
          if not k.rsplit(".", 1)[-1].startswith(("weight_scale", "weight_zero_point", "weight_g_idx"))}
    # 2. the packed file, only to measure its size
    packed = a.out.rstrip("/") + "_packed"
    model.save_pretrained(packed)
    disk = dir_gb(packed)
    if a.keep_packed:  # check: the server loads the packed 4-bit file itself, to compare with the dense copy
        tok.save_pretrained(packed)
    else:
        shutil.rmtree(packed, ignore_errors=True)
    del model
    torch.cuda.empty_cache()
    fresh = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    fresh.load_state_dict({k: sd[k] for k in fresh.state_dict()}, strict=True)
    os.makedirs(a.out, exist_ok=True)
    fresh.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    n_lin = sum(p.numel() for n, p in fresh.named_parameters() if p.dim() == 2 and "embed" not in n)
    meta = {"arm": a.tag or a.arm, "symmetric": not a.asym, "max_len": a.max_len, "model": a.model, "bits": bits, "protect": protect, "protected_layers": layers,
            "circuit_ranked_layers": ranked, "n_layers": n_layers, "quantized_linears": info["quantized_layers"],
            "disk_gb": disk, "levels_in_probe_group": levels, "avg_bits_layers": round((bits * (n_layers - len(layers)) + 16 * len(layers)) / n_layers, 2),
            "n_calib": a.n_calib, "n_2d_params": n_lin}
    os.makedirs(a.res, exist_ok=True)
    json.dump(meta, open(f"{a.res}/{a.tag or a.arm}_quant.json", "w"), indent=1)
    print(json.dumps(meta), flush=True)


def _save_dense(model, tok, a, meta):
    import torch
    model = model.to(torch.bfloat16)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    os.makedirs(a.res, exist_ok=True)
    json.dump(meta, open(f"{a.res}/{a.tag}_quant.json", "w"), indent=1)
    print(json.dumps(meta), flush=True)


def bnb(a):
    """The 4-bit loading most notebooks use (bitsandbytes NF4, no calibration). The 4-bit weights are decoded
    back to an ordinary checkpoint so the test server runs it like every other arm."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(a.model)
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=cfg, device_map="auto")
    n4 = sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit")
    gb = round(model.get_memory_footprint() / 1e9, 2)
    model = model.dequantize()
    if any(type(m).__name__ == "Linear4bit" for m in model.modules()):
        raise SystemExit("dequantize left 4-bit modules behind")
    _save_dense(model, tok, a, {"arm": a.tag, "model": a.model, "bits": 4, "method": "bitsandbytes NF4, double quant",
                                "protect": "none", "protected_layers": [], "n_layers": model.config.num_hidden_layers,
                                "quantized_linears": n4, "disk_gb": gb})


def rtn(a):
    """Round to nearest, no calibration: every linear weight in the transformer layers, in groups of 128 inputs,
    each group mapped to 2^bits evenly spaced levels between its own min and max."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto")
    bits, n, g = a.bits, 0, 128
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "lm_head" not in name and m.weight.shape[1] % g == 0:
            w = m.weight.data.float()
            o, i = w.shape
            v = w.view(o, i // g, g)
            lo, hi = v.amin(-1, keepdim=True), v.amax(-1, keepdim=True)
            sc = (hi - lo).clamp_min(1e-8) / (2 ** bits - 1)
            m.weight.data = (torch.round((v - lo) / sc) * sc + lo).view(o, i).to(m.weight.dtype)
            n += 1
    _save_dense(model, tok, a, {"arm": a.tag, "model": a.model, "bits": bits, "method": "round to nearest, groups of 128",
                                "protect": "none", "protected_layers": [], "n_layers": model.config.num_hidden_layers,
                                "quantized_linears": n, "disk_gb": None})


def report(a):
    sets = ["test", "stress", "unseen", "sealed"]
    print("| arm | bits | protected layers | size GB | " + " | ".join(f"esc {s}" for s in sets) + " | "
          + " | ".join(f"legit {s}" for s in sets) + " | tok/s |")
    print("|---|---|---|---|" + "---|" * (2 * len(sets) + 1))
    for arm in ["fp", "u4", "u3", "c3", "r3", "r3b", "r3c", "a3", "c4"]:
        q = json.load(open(f"{a.res}/{arm}_quant.json")) if os.path.exists(f"{a.res}/{arm}_quant.json") else {}
        cells, ok = [], False
        for key in ("escalation_rate_unsolvable", "legit_success_rate_solvable"):
            for s in sets:
                p = f"{a.res}/{s}/{a.name}_{arm}_summary.json"
                if os.path.exists(p):
                    cells.append(f"{100 * json.load(open(p))['neutral'][key]:.1f}")
                    ok = True
                else:
                    cells.append("n/a")
        if not ok:
            continue
        sp = f"{a.res}/{arm}_speed.json"
        tps = f"{json.load(open(sp))['tokens_per_s']:.0f}" if os.path.exists(sp) else "n/a"
        print(f"| {arm} | {q.get('bits', 16)} | {len(q.get('protected_layers', []))} of {q.get('n_layers', '?')} | "
              f"{q.get('disk_gb', 'n/a')} | " + " | ".join(cells) + f" | {tps} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["quantize", "report", "bnb", "rtn"])
    ap.add_argument("--model", default="")
    ap.add_argument("--scores", default="")
    ap.add_argument("--arm", default="c3", choices=list(ARMS))
    ap.add_argument("--out", default="outputs/quant/c3")
    ap.add_argument("--res", default="results/quant")
    ap.add_argument("--name", default="global_safe")
    ap.add_argument("--high_fraction", type=float, default=0.3)
    ap.add_argument("--n_calib", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="name of the result file, when it differs from --arm")
    ap.add_argument("--bits", type=int, default=4, help="rtn only")
    ap.add_argument("--asym", action="store_true", help="GPTQ with asymmetric levels")
    ap.add_argument("--keep_packed", action="store_true")
    a = ap.parse_args()
    {"quantize": quantize, "report": report, "bnb": bnb, "rtn": rtn}[a.cmd](a)
