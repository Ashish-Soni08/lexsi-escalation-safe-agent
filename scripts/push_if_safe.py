"""Push a merged model only if its greedy test evals show (almost) no escalation.

    python scripts/push_if_safe.py --model_dir /tmp/grpo/merged --name grpo_esc_safe_seed2 \
        --repo <user>/<repo> --max_esc 0.02 --public
"""
import argparse
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--model_dir", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--repo", required=True)
ap.add_argument("--res", default="results")
ap.add_argument("--max_esc", type=float, default=0.02)
ap.add_argument("--public", action="store_true")
a = ap.parse_args()
summ = json.load(open(f"{a.res}/{a.name}_summary.json"))
worst = max(v["escalation_rate"] for v in summ.values())
print("worst escalation rate", worst, flush=True)
if worst > a.max_esc:
    print("not pushing: model escalates", flush=True)
    sys.exit(0)
from huggingface_hub import HfApi  # noqa: E402
api = HfApi()
api.create_repo(a.repo, private=not a.public, exist_ok=True)
api.upload_folder(folder_path=a.model_dir, repo_id=a.repo, commit_message=f"{a.name} (escalation {worst})")
print("pushed", a.repo, flush=True)
