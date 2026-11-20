"""Stage 2 coverage curve: training scenarios and GRPO prompts with 2, 4, 6 or 10 shortcut kinds.

    python scripts/build_coverage.py          # writes data/scenarios_cov<k>.jsonl, data/grpo_prompts_cov<k>.jsonl

The 960 tasks, their urgency lines and their tip slots are the same in every arm. Only the kinds of
shortcut on offer differ (the first k of world.COVERAGE_KINDS). No earlier data file is touched.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esa import world  # noqa: E402
from esa.prompts import SYSTEM_PROMPTS  # noqa: E402

if __name__ == "__main__":
    for k in world.COVERAGE_ARMS:
        sc = world.make_coverage_scenarios(k)
        with open(f"data/scenarios_cov{k}.jsonl", "w", encoding="utf-8") as f:
            for s in sc:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        with open(f"data/grpo_prompts_cov{k}.jsonl", "w", encoding="utf-8") as f:
            for i, s in enumerate(sc):
                v = "neutral" if i % 2 else "rules"
                f.write(json.dumps({"prompt": [{"role": "system", "content": SYSTEM_PROMPTS[v]},
                                               {"role": "user", "content": s.prompt()}],
                                    "scenario_id": s.id, "variant": v}, ensure_ascii=False) + "\n")
        kinds = sorted({t for s in sc for t in s.temptations})
        print(k, len(sc), "blocked", sum(not s.solvable for s in sc), "tips", sum(s.injected_tip for s in sc), kinds)
