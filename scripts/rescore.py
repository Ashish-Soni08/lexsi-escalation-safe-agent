"""Re-apply the current detector to saved eval rows (no model calls).

    python scripts/rescore.py results/stress/*_neutral.json results/stress/*_rules.json

Per-call flags are recomputed from the stored tool calls; episode-level flags (request
floods) and answer correctness are kept from the original scoring. An episode that gains
a flag becomes "escalated". Used after the LLM-judge audit found detector blind spots.
"""
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esa.detector import classify_call  # noqa: E402
from esa.scoring import summarize  # noqa: E402

changed_total = 0
for path in sys.argv[1:]:
    d = json.load(open(path))
    changed = 0
    for r in d["rows"]:
        o = r["outcome"]
        keep = [f for f in o["flags"] if f["category"] == "request_flood"]
        per_call = [asdict(f) for i, (t, a) in enumerate(r["calls"]) for f in classify_call(t, a, i)]
        flags = per_call + keep
        cats = sorted({f["category"] for f in flags})
        if cats != o["categories"]:
            changed += 1
            o["flags"], o["categories"], o["escalated"] = flags, cats, bool(flags)
            if flags:
                o["outcome"] = "escalated"
    if changed:
        m = d["metrics"]
        new = summarize(d["rows"])
        for k, v in new.items():
            m[k] = v
        m["rescored_detector"] = "v2.2 (judge-found debug path/header patterns; round-4 probe-path and relay rules)"
        json.dump(d, open(path, "w"), ensure_ascii=False, indent=1)
        sp = path.rsplit("_", 1)[0] + "_summary.json"
        if os.path.exists(sp):
            sm = json.load(open(sp))
            v = d["metrics"]["variant"]
            if v in sm:
                sm[v].update({k: x for k, x in m.items() if k != "slices"})
                json.dump(sm, open(sp, "w"), indent=2)
    changed_total += changed
    print(path, "changed", changed)
print("total changed", changed_total)
