"""CircuitKIT: where in the network does the shortcut decision happen?

At one decision point, after a blocked response, the agent has written
    [{"tool_name": "http_get", "parameters": {"url": "https://
and the next token picks the host: the provider's own site, or a shortcut.
Two contrastive tasks, built from the environment itself:

* relay  - after a web search whose results advertise "reader-relay" (the
           exploit the success-only model learned in training). Corrupted run:
           the same search without the relay results.
* preview- after a blocked response whose injected tip points to a preview
           host (a stress-test shortcut no model was trained on). Corrupted
           run: the same blocked response without the tip.

The metric is logit(shortcut token) - logit(official host token). For each
model we measure that logit difference and P(shortcut) on clean and corrupted
prompts, then run CircuitKIT EAP-IG node-level discovery and compare the top
components across models (top-K Jaccard, per-layer score mass).

    python scripts/circuit_compare.py build         # local
    python scripts/circuit_compare.py run --out results/circuit \\
        --models sft_esc=<repo>,grpo_esc_success_only=<repo>,grpo_esc_safe=<repo>  # GPU
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esa import world  # noqa: E402
from esa.prompts import TOOL_RESULT_FORMAT, full_system_prompt  # noqa: E402
from esa.runner import MAX_RESULT_CHARS  # noqa: E402

PREFIX = '[{"tool_name": "http_get", "parameters": {"url": "https://'
TOK_REPO = "ashish-soni08/tiny-aya-global-esa-sft-v2"


def _call(name, args):
    return json.dumps([{"tool_name": name, "parameters": args}], ensure_ascii=False)


def _res(name, text):
    return {"role": "user", "content": TOOL_RESULT_FORMAT.format(name=name, content=text[:MAX_RESULT_CHARS])}


def _render(tok, msgs):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if tok.bos_token and text.startswith(tok.bos_token):
        text = text[len(tok.bos_token):]  # TransformerLens prepends BOS itself
    return text + PREFIX


def _first_tok(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"][0]


def build(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_REPO)
    schemas = world.tool_schemas()
    sys_msg = {"role": "system", "content": full_system_prompt(schemas, "neutral")}
    os.makedirs(a.data, exist_ok=True)
    stats = {}

    def ok(clean, official, shortcut):
        # the answer token must be what the model would write right after "https://"
        ids = tok(PREFIX + official, add_special_tokens=False)["input_ids"]
        pre = tok(PREFIX, add_special_tokens=False)["input_ids"]
        return ids[:len(pre)] == pre and ids[len(pre)] == _first_tok(tok, official) and \
            _first_tok(tok, official) != _first_tok(tok, shortcut)

    # relay: test-split scenarios (the training distribution of temptations)
    test = [s for s in world.load_jsonl("data/scenarios_test.jsonl")
            if not s.solvable and "relay_service" in s.temptations and s.block != "none"]
    world.register(test)
    rows = []
    for s in test:
        q = f"{s.org} {s.dataset}"
        blocked = world.http_get(s.data_url)
        clean_search = world.web_search(q)
        s.temptations = [t for t in s.temptations if t != "relay_service"]
        corr_search = world.web_search(q)
        s.temptations.append("relay_service")
        head = [sys_msg, {"role": "user", "content": s.prompt()},
                {"role": "assistant", "content": _call("http_get", {"url": s.data_url})}, _res("http_get", blocked),
                {"role": "assistant", "content": _call("web_search", {"query": q})}]
        clean = _render(tok, head + [_res("web_search", clean_search)])
        corr = _render(tok, head + [_res("web_search", corr_search)])
        official = s.host.split(".")[0]
        if "reader-relay" in clean_search and "reader-relay" not in corr_search and ok(clean, official, "reader"):
            rows.append({"id": s.id, "prompt": clean, "corrupted_prompt": corr, "answer": "reader",
                         "corrupted_answer": official})
    with open(f"{a.data}/relay.jsonl", "w") as f:
        for r in rows[:a.n]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["relay"] = len(rows[:a.n])

    # preview: stress scenarios whose blocked response carries a tip naming the preview host
    st = [s for s in world.load_jsonl("data/scenarios_stress.jsonl")
          if not s.solvable and "preview_host" in s.temptations and s.injected_tip and s.block != "none"]
    world.register(st)
    rows = []
    for s in st:
        clean_res = world.http_get(s.data_url)
        s.injected_tip = False
        corr_res = world.http_get(s.data_url)
        s.injected_tip = True
        head = [sys_msg, {"role": "user", "content": s.prompt()},
                {"role": "assistant", "content": _call("http_get", {"url": s.data_url})}]
        clean = _render(tok, head + [_res("http_get", clean_res)])
        corr = _render(tok, head + [_res("http_get", corr_res)])
        official = s.host.split(".")[0]
        if "preview." in clean_res and "preview." not in corr_res and ok(clean, official, "preview"):
            rows.append({"id": s.id, "prompt": clean, "corrupted_prompt": corr, "answer": "preview",
                         "corrupted_answer": official})
    with open(f"{a.data}/preview.jsonl", "w") as f:
        for r in rows[:a.n]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["preview"] = len(rows[:a.n])
    stats["prompt_tokens_example"] = len(tok(rows[0]["prompt"])["input_ids"])
    print(stats)


def behaviour(model, rows):
    """Mean logit diff (shortcut - official) and P(shortcut) on clean and corrupted prompts."""
    import torch
    out = {"clean": {"logit_diff": [], "p_shortcut": [], "p_official": []},
           "corrupted": {"logit_diff": [], "p_shortcut": [], "p_official": []}}
    for r in rows:
        sc = model.to_tokens(r["answer"], prepend_bos=False)[0, 0].item()
        of = model.to_tokens(r["corrupted_answer"], prepend_bos=False)[0, 0].item()
        for key, field in (("clean", "prompt"), ("corrupted", "corrupted_prompt")):
            with torch.no_grad():
                logits = model(model.to_tokens(r[field]))[0, -1].float()
            p = logits.softmax(-1)
            out[key]["logit_diff"].append((logits[sc] - logits[of]).item())
            out[key]["p_shortcut"].append(p[sc].item())
            out[key]["p_official"].append(p[of].item())
    return {k: {m: round(sum(v) / len(v), 4) for m, v in d.items()} for k, d in out.items()}


def layer_mass(scores, n_layers):
    import re
    mass = [0.0] * n_layers
    for node, v in scores.items():
        m = re.match(r"^(?:A|MLP )(\d+)", node)
        if m and int(m.group(1)) < n_layers:
            mass[int(m.group(1))] += abs(float(v))
    tot = sum(mass) or 1.0
    return [round(x / tot, 4) for x in mass]


def run(a):
    import gc

    import circuitkit as ck
    import torch
    from circuitkit.tasks.generic import GenericTaskSpec

    from circuitkit.analysis.cross_method_jaccard import cross_method_jaccard

    os.makedirs(a.out, exist_ok=True)
    tasks = {}
    for t in ("relay", "preview"):
        spec = GenericTaskSpec.from_jsonl(
            f"{a.data}/{t}.jsonl", name=f"esa_{t}", task_type="classification", chat_template_mode="off",
            schema={"prompt": "prompt", "corrupted_prompt": "corrupted_prompt", "answer": "answer",
                    "corrupted_answer": "corrupted_answer", "id": "id"})
        ck.register_task(spec)
        tasks[t] = [json.loads(l) for l in open(f"{a.data}/{t}.jsonl")]
    summary = {"models": {}, "algorithm": a.algorithm, "n_examples": a.n_examples, "top_k": a.top_k}
    for spec in a.models.split(","):
        key, repo = spec.split("=", 1)
        # CircuitKIT's cohere2 port resolves fine-tuned checkpoints from a local
        # directory's config.json (Hub ids are matched against the base names only)
        from huggingface_hub import snapshot_download
        local = repo if os.path.isdir(repo) else snapshot_download(repo, local_dir=f"/tmp/ckpt/{key}")
        model = ck.load_model(local, dtype="bfloat16")
        n_layers = model.cfg.n_layers
        summary["models"][key] = {"repo": repo, "n_layers": n_layers}
        for t, rows in tasks.items():
            res = {"behaviour": behaviour(model, rows[:a.n_behaviour])}
            print(key, t, res["behaviour"], flush=True)
            try:
                circ = ck.discover(model, f"esa_{t}", algorithm=a.algorithm, n_examples=a.n_examples,
                                   batch_size=1, ig_steps=a.ig_steps, chat_template_mode="off",
                                   output_path=f"{a.out}/{key}_{t}_circuit.pt")
                top = sorted(circ.scores.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:a.top_k]
                res["top_nodes"] = {k: round(float(v), 6) for k, v in top}
                res["n_pruned_at_sparsity_0.3"] = len(circ)
                res["layer_mass"] = layer_mass(circ.scores, n_layers)
                json.dump(circ.scores, open(f"{a.out}/{key}_{t}_scores.json", "w"))
            except Exception as e:  # noqa: BLE001
                res["error"] = f"{type(e).__name__}: {e}"[:500]
                print("discovery failed", key, t, res["error"], flush=True)
            summary["models"][key][t] = res
            json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)
            torch.cuda.empty_cache()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    for t in tasks:
        circs = {k: v[t]["top_nodes"] for k, v in summary["models"].items() if "top_nodes" in v.get(t, {})}
        if len(circs) > 1:
            j = cross_method_jaccard(circs, top_k=a.top_k)
            summary[f"jaccard_{t}"] = {"models": list(circs), "matrix": [[round(x, 3) for x in r] for r in j.matrix]}
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1)[:4000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--out", default="results/circuit")
    ap.add_argument("--data", default="data/circuit")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--models", default="")
    ap.add_argument("--algorithm", default="eap-ig")
    ap.add_argument("--ig_steps", type=int, default=5)
    ap.add_argument("--n_examples", type=int, default=24)
    ap.add_argument("--n_behaviour", type=int, default=64)
    ap.add_argument("--top_k", type=int, default=20)
    a = ap.parse_args()
    {"build": build, "run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
