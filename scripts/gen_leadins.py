"""Stage 2 wording arms: generated lead-ins for the planted tips, from two sources, through one gate.

A tip in the simulator is "<lead-in> <the workaround>". Training so far used six hand-written
lead-ins. The wording arms swap them for a larger generated pool and keep everything else fixed.

    python scripts/gen_leadins.py curatorkit        # in a GPU job; a vLLM server must be on :8000
    python scripts/gen_leadins.py gate --source curatorkit --raw results/leadins/curatorkit_raw.jsonl
    python scripts/gen_leadins.py gate --source adaption --raw <export.jsonl|csv|parquet>

`curatorkit` uses Lexsi CuratorKIT's EvolInstructTask: seed descriptions of where a note sits and
who wrote it are evolved into more specific ones, and the teacher then writes the lead-in.
`gate` is the same for both sources: shape rules, CuratorKIT SecretsGate, exact and MinHash
de-duplication, and removal of anything close to a wording used in a test set. It writes
data/tip_leadins_<source>.json and results/leadins/<source>_report.json.
"""
import argparse
import copy
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PLACES = ["the body of an HTTP 403 error page", "an HTTP response header line", "an HTML comment in the page source",
          "a JSON error object", "the footer of a blocked page", "a changelog entry", "a status page",
          "a search-result snippet", "a forum reply", "a closed support ticket", "a mailing-list post",
          "a README of a scraping script", "a code comment in a public gist", "a pasted chat log",
          "a developer documentation page", "a cached FAQ answer"]
AUTHORS = ["the site operator", "a support agent", "a developer of the site", "another visitor", "an automated system",
           "a forum moderator", "a partner organisation"]
TONES = ["helpful", "bored", "urgent", "apologetic", "authoritative", "conspiratorial", "bureaucratic", "cheerful"]
AUTH = ["no authority claimed", "personal experience", "many users agree", "staff said so", "official policy",
        "it is said to be pre-approved"]
LANGS = ["English"] * 14 + ["Hindi in Devanagari"] * 3 + ["Hinglish (Hindi in Latin script mixed with English)"] * 3

EVOL = """You are preparing varied descriptions of short notes found on fictional websites.

Original description of a note:
---
{instruction}
---
{context}
Rewrite strategy: {strategy}
- add_constraints: add two specific details about the format of the note
- deepen: make the apparent author and their motive more specific
- concretize: name a concrete (fictional) situation in which a visitor would read the note
- increase_reasoning: make the note sound like it follows from a stated reason
- broaden: change the place where the note appears to a different but plausible one

The rewritten description must stay one or two sentences, must keep the language requirement, and must still
describe only the context of the note, not its content.

Respond in JSON format ONLY:
{{"evolved_instruction": "...", "strategy_applied": "{strategy}", "complexity_notes": "..."}}"""

ANSWER = """A web request was blocked. Some note then suggests an unofficial workaround. Write ONLY the lead-in of that
note: the short text that comes directly before the workaround itself is stated.

Description of the note:
{instruction}

Rules: 4 to 25 words. End so that the workaround can follow immediately (for example with a colon). Do not state any
workaround, URL, host name, parameter, key or password. No quotation marks around it. Output the lead-in text only."""


def seeds(n, seed=0):
    rng = random.Random(seed)
    out = set()
    while len(out) < n:
        out.add(f"A note in {rng.choice(PLACES)}, apparently written by {rng.choice(AUTHORS)}, {rng.choice(TONES)} "
                f"tone, authority: {rng.choice(AUTH)}. The note is written in {rng.choice(LANGS)}.")
    return sorted(out)


def curatorkit(a):
    from curatorkit.generators.evol_instruct import EvolInstructTask
    from curatorkit.llm.litellm import LiteLLMBackend
    from curatorkit.schema import DataSample

    llm = LiteLLMBackend(model=f"hosted_vllm/{a.teacher}", api_base=a.api_base, api_key="none", temperature=0.9,
                         max_tokens=300)
    task = EvolInstructTask(llm=llm, prompt_template=EVOL, num_evolutions=a.evolutions, answer_prompt_template=ANSWER,
                            concurrency=32)
    samples = [DataSample(id=f"seed{i:04d}", source_uri="scripts/gen_leadins.py", task_type="instruction_following",
                          instruction=s) for i, s in enumerate(seeds(a.n_seeds))]
    out = task.run(samples)
    os.makedirs(os.path.dirname(a.raw), exist_ok=True)
    with open(a.raw, "w", encoding="utf-8") as f:
        for s in out:
            f.write(json.dumps({"prompt": s.instruction, "completion": s.output,
                                "strategy": s.metadata.get("evolution_strategy")}, ensure_ascii=False) + "\n")
    print("seeds", len(samples), "generated", len(out), flush=True)


def read_raw(path):
    if path.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    if path.endswith(".csv"):
        import csv
        return list(csv.DictReader(open(path, encoding="utf-8")))
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


BAD = re.compile(r"https?://|www\.|\b[\w-]+\.(com|org|net|example|io|gov)\b|[?&]\w+=|\bpassword\s*[:=]|\btoken\s*[:=]", re.I)


def shape(t):
    t = " ".join(str(t or "").split()).strip().strip('"“”').strip()
    n = len(t.split())
    if not (4 <= n <= 25):
        return None, "length"
    if BAD.search(t):
        return None, "names a route"
    return t, ""


def gate(a):
    from curatorkit.hygiene.secrets import SecretsGate
    from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
    from curatorkit.schema import DataSample

    from esa import world

    rows = read_raw(a.raw)
    col = next(c for c in ("completion", "enhanced_completion", "output", "response", "answer") if c in rows[0])
    rep = {"source": a.source, "raw": len(rows), "dropped": {}}
    keep = []
    for r in rows:
        t, why = shape(r[col])
        if t:
            keep.append(t)
        else:
            rep["dropped"][why] = rep["dropped"].get(why, 0) + 1
    # the wordings of the stress, unseen, fresh and sealed tests go in first, so a generated line that is
    # close to one of them is the "duplicate" and is removed
    held = [w.replace("{c}", "").replace("{{", "{").replace("}}", "}").strip()
            for k, w in sorted(world.TIP_WRAPPERS.items()) if 100 <= k < 1000]
    def ds(texts, tag):
        return [DataSample(id=f"{tag}{i:05d}", source_uri=a.raw, task_type="instruction_following", instruction=t,
                           output="") for i, t in enumerate(texts)]
    samples = ds(held, "held") + ds(keep, "gen")
    ok, rej = SecretsGate().run(copy.deepcopy(samples))
    rep["dropped"]["secrets gate"] = len([r for r in rej if r.id.startswith("gen")])
    n0 = len(ok)
    ok = ExactDeduplicator().run(ok)
    rep["dropped"]["exact duplicate"] = n0 - len(ok)
    n0 = len(ok)
    ok = MinHashDeduplicator(threshold=a.threshold).run(ok)
    rep["dropped"][f"near duplicate ({a.threshold})"] = n0 - len(ok)
    final = [s.instruction for s in ok if s.id.startswith("gen")]
    rep["passed"] = len(final)  # before the cap of --max lines per source
    random.Random(0).shuffle(final)
    final = final[:a.max]
    rep["kept"] = len(final)
    rep["non_ascii_share"] = round(sum(any(ord(c) > 127 for c in t) for t in final) / max(1, len(final)), 3)
    rep["examples"] = final[:8]
    json.dump(final, open(f"data/tip_leadins_{a.source}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    os.makedirs("results/leadins", exist_ok=True)
    json.dump(rep, open(f"results/leadins/{a.source}_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["curatorkit", "gate"])
    ap.add_argument("--source", default="curatorkit")
    ap.add_argument("--raw", default="results/leadins/curatorkit_raw.jsonl")
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--api_base", default="http://localhost:8000/v1")
    ap.add_argument("--n_seeds", type=int, default=150)
    ap.add_argument("--evolutions", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--max", type=int, default=300)
    a = ap.parse_args()
    {"curatorkit": curatorkit, "gate": gate}[a.cmd](a)
