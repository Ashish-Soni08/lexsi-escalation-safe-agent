"""Download the per-episode Stage 1 result files that are too large for git.

    python scripts/fetch_results.py            # test and stress episode files (~40 MB)
    python scripts/fetch_results.py --all      # also the trajectories and every other split (much larger)

`.gitignore` keeps `results/<name>_{neutral,rules}.json` and `results/stress/...` out of the repo; they live in the
public dataset ashish-soni08/escalation-safe-agent-data under eval/ and eval_stress/. With them in place,
`scripts/report.py`, `scripts/report_r6.py` and `scripts/report_s2.py` rebuild the committed tables exactly.
No GPU and no Hugging Face token needed.
"""
import argparse
import os
import shutil

from huggingface_hub import hf_hub_download, list_repo_files

REPO = "ashish-soni08/escalation-safe-agent-data"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# dataset folder -> results folder
MAP = {"eval": "results", "eval_stress": "results/stress", "eval_v5/test": "results", "eval_v5/stress": "results/stress"}
MAP_ALL = {**MAP, "eval_unseen": "results/unseen", "eval_fresh": "results/fresh", "eval_sealed": "results/sealed",
           "eval_sampled": "results/seeds", "eval_sampled_stress": "results/seeds_stress",
           "eval_sampled_unseen": "results/seeds_unseen", "eval_sampled_fresh": "results/seeds_fresh",
           "eval_sampled_sealed": "results/seeds_sealed"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="every split, including trajectory files")
    a = ap.parse_args()
    mapping = MAP_ALL if a.all else MAP
    n = 0
    for f in list_repo_files(REPO, repo_type="dataset"):
        top, _, rest = f.rpartition("/")
        if top not in mapping:
            continue
        if not a.all and not (rest.endswith("_neutral.json") or rest.endswith("_rules.json")):
            continue
        dst = os.path.join(ROOT, mapping[top], rest)
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(hf_hub_download(REPO, f, repo_type="dataset"), dst)
        n += 1
    print(f"{n} files written")


if __name__ == "__main__":
    main()
