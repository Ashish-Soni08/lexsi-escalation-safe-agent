"""CuratorKIT hygiene pass over the SFT data before it is published.

    pip install 'curatorkit[hygiene]' && python -m spacy download en_core_web_sm
    python scripts/curate_check.py --out results/curatorkit_report.json

Runs Lexsi CuratorKIT's SecretsGate (credential/API-key detection), exact and MinHash
near-duplicate detection, and the Presidio-based PII pseudonymizer (used here only to
count samples it would change). The data is fully synthetic, so this is a check, not a
filter: nothing is removed from the published files. Planted fake credentials (the
"leaked ops token" temptation) are expected to be flagged.
"""
import argparse
import copy
import json
from collections import Counter

from curatorkit.hygiene.pii import PIIPseudonymizer
from curatorkit.hygiene.secrets import SecretsGate
from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
from curatorkit.schema import DataSample


def to_samples(path):
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        m = r["messages"]
        sys_p = m[0]["content"] if m[0]["role"] == "system" else ""
        rest = m[1:] if sys_p else m
        out.append(DataSample(
            id=r["scenario_id"], source_uri=path, task_type="conversational",
            instruction=rest[0]["content"], output=rest[1]["content"] if len(rest) > 1 else "",
            metadata={"system_prompt": sys_p, "turns": rest[2:], "outcome": r.get("outcome")}))
    return out


def check(path, pii):
    samples = to_samples(path)
    rep = {"file": path, "n": len(samples)}
    kept, rejected = SecretsGate().run(copy.deepcopy(samples))
    rep["secrets_flagged"] = len(rejected)
    rep["secrets_reasons"] = dict(Counter(r.rejection_reason.split(":")[0] for r in rejected))
    rep["secrets_flagged_by_outcome"] = dict(Counter(r.metadata.get("outcome") for r in rejected))
    rep["secrets_examples"] = [r.rejection_reason[:160] for r in rejected[:3]]
    rep["exact_duplicates"] = len(samples) - len(ExactDeduplicator().run(copy.deepcopy(samples)))
    rep["near_duplicates_0.85"] = len(samples) - len(MinHashDeduplicator(threshold=0.85).run(copy.deepcopy(samples)))
    if pii is not None:
        before = copy.deepcopy(samples)
        after = pii.run(copy.deepcopy(samples))
        changed = [a for a, b in zip(after, before)
                   if (a.instruction, a.output, json.dumps(a.metadata, ensure_ascii=False))
                   != (b.instruction, b.output, json.dumps(b.metadata, ensure_ascii=False))]
        rep["pii_samples_changed"] = len(changed)
        rep["pii_examples"] = [c.instruction[:160] for c in changed[:3]]
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=["data/sft_trajectories.jsonl", "data/sft_esc_trajectories.jsonl"])
    ap.add_argument("--out", default="results/curatorkit_report.json")
    ap.add_argument("--spacy_model", default="en_core_web_sm")
    ap.add_argument("--no_pii", action="store_true")
    args = ap.parse_args()
    pii = None if args.no_pii else PIIPseudonymizer(spacy_model=args.spacy_model)
    reports = [check(f, pii) for f in args.files]
    json.dump(reports, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
