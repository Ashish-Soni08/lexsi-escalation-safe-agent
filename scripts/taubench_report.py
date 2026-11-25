"""Summary of the tau-bench runs written by scripts/taubench_chain.sh: results/taubench/<name>/*.json."""
import glob
import json
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else "results/taubench"
out = {}
for d in sorted(glob.glob(f"{R}/*/")):
    name = os.path.basename(d.rstrip("/"))
    files = sorted(glob.glob(f"{d}/*.json"))
    if not files:
        continue
    rs = json.load(open(files[-1]))
    err = [r for r in rs if "error" in (r.get("info") or {})]
    ctx = sum("context" in str(r["info"].get("error", "")).lower() or "maximum" in str(r["info"].get("error", "")).lower() for r in err)
    out[name] = {"tasks": len(rs), "passed": sum(r["reward"] == 1 for r in rs),
                 "pass_rate": round(sum(r["reward"] == 1 for r in rs) / max(1, len(rs)), 4),
                 "errors": len(err), "context_overflow": ctx,
                 "mean_agent_turns": round(sum(sum(m.get("role") == "assistant" for m in r.get("traj") or []) for r in rs) / max(1, len(rs)), 1)}
json.dump(out, open(f"{R}/summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
