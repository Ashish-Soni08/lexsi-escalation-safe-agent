"""General tool-calling check: BFCL v3 "simple" and "multiple" (single-turn function calling).

Does the safe reward cost the model its ordinary tool use? The questions and functions have
nothing to do with our simulator. Each model gets the functions in the same fallback tool
format it uses everywhere else (Tiny Aya has no native tool template) and must reply with one
call. The check follows BFCL's AST rule: right function, every required argument present, every
given argument among the accepted values. It is our own small implementation of that rule, so
compare the models with each other, not with the public leaderboard.

    python scripts/bfcl_eval.py --models base=CohereLabs/tiny-aya-global,sft=<repo>,... --out results/bfcl

The data is downloaded from gorilla-llm/Berkeley-Function-Calling-Leaderboard (Apache-2.0) at run
time and is not stored in this repository.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esa.prompts import fallback_block  # noqa: E402
from esa.runner import parse_tool_calls  # noqa: E402

SYSTEM = "You are a helpful assistant. Use the tools provided to carry out the user's request."
TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


def _schema(node):
    if isinstance(node, dict):
        out = {k: _schema(v) for k, v in node.items()}
        if isinstance(out.get("type"), str):
            out["type"] = TYPES.get(out["type"], out["type"])
        return out
    if isinstance(node, list):
        return [_schema(v) for v in node]
    return node


def to_tool(fn):
    return {"type": "function", "function": {"name": fn["name"], "description": fn.get("description", ""),
                                             "parameters": _schema(fn.get("parameters", {}))}}


def norm(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        t = re.sub(r"[\s,.\-_/()'\"]", "", v).lower()
        try:
            return float(v)
        except ValueError:
            return t
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {str(k): norm(x) for k, x in v.items()}
    return v


def value_ok(v, accepted):
    for a in accepted:
        if isinstance(a, dict) and isinstance(v, dict):  # nested object: each key has its own accepted list
            if all(k in a and value_ok(x, a[k] if isinstance(a[k], list) else [a[k]]) for k, x in v.items()) and \
                    all(k in v or "" in (x if isinstance(x, list) else [x]) for k, x in a.items()):
                return True
        elif a != "" and norm(v) == norm(a):
            return True
    return False


def check(calls, truth):
    """-> (valid call, right function, full match)"""
    if not calls:
        return False, False, False
    want_name, want = next(iter(truth[0].items()))
    name, args = calls[0]
    if name != want_name:
        return True, False, False
    if len(calls) != len(truth):
        return True, True, False
    ok = all(k in want and value_ok(v, want[k]) for k, v in args.items()) and \
        all(k in args or "" in acc for k, acc in want.items())
    return True, True, ok


def load(n):
    from huggingface_hub import hf_hub_download
    rows = []
    for cat in ("simple", "multiple"):
        q = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard", f"BFCL_v3_{cat}.json",
                            repo_type="dataset")
        a = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                            f"possible_answer/BFCL_v3_{cat}.json", repo_type="dataset")
        truth = {json.loads(l)["id"]: json.loads(l)["ground_truth"] for l in open(a) if l.strip()}
        qs = [json.loads(l) for l in open(q) if l.strip()]
        for r in (qs[:n] if n else qs):
            rows.append({"id": r["id"], "cat": cat, "truth": truth[r["id"]],
                         "messages": [{"role": "system", "content": SYSTEM + "\n\n" + fallback_block(
                             [to_tool(f) for f in r["function"]])}] + r["question"][0]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma list of name=repo")
    ap.add_argument("--out", default="results/bfcl")
    ap.add_argument("--n", type=int, default=0, help="questions per category (0 = all)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.n)
    from esa.runner import vllm_generator
    # one model per process is safest for GPU memory; the summary file is merged across runs
    summary = json.load(open(f"{a.out}/summary.json")) if os.path.exists(f"{a.out}/summary.json") else {}
    for spec in a.models.split(","):
        name, repo = spec.split("=", 1)
        gen = vllm_generator(repo, max_new_tokens=256, gpu_memory_utilization=0.6)
        outs = gen([r["messages"] for r in rows])
        res = []
        for r, text in zip(rows, outs):
            valid, fn, full = check(parse_tool_calls(text), r["truth"])
            res.append({"id": r["id"], "cat": r["cat"], "valid_call": valid, "right_function": fn, "match": full,
                        "output": text})
        with open(f"{a.out}/{name}.jsonl", "w") as f:
            for x in res:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        summary[name] = {"model": repo}
        for cat in ("simple", "multiple", "all"):
            sel = [x for x in res if cat in ("all", x["cat"])]
            summary[name][cat] = {"n": len(sel), **{k: round(sum(x[k] for x in sel) / len(sel), 4)
                                                    for k in ("valid_call", "right_function", "match")}}
        print(name, json.dumps(summary[name]), flush=True)
        json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
