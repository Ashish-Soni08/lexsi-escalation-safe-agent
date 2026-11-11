# CuratorKIT hygiene check (SFT data)

`scripts/curate_check.py`, CuratorKIT 0.1.25 (`SecretsGate`, `ExactDeduplicator`, `MinHashDeduplicator` at 0.85,
`PIIPseudonymizer` with spaCy `en_core_web_sm`, threshold 0.7). Raw numbers: `curatorkit_report.json`.

| file | samples | secrets flagged | exact dups | near dups (0.85) | PII pseudonymizer would change |
|---|---|---|---|---|---|
| `sft_trajectories.jsonl` | 1,600 | 57 | 0 | 278 | 1,600 |
| `sft_esc_trajectories.jsonl` | 1,600 | 83 | 0 | 188 | 1,600 |

What the flags are:

* **Secrets.** In a 300-sample spot check every flagged sample contained a planted fake `ops_<12 hex>` token, the
  "leaked token" temptation, which the agent reads from the workspace notes. These are deliberate and synthetic, so the
  samples are kept. The escalation-prone file has more because its exploit trajectories reuse the token.
* **Near duplicates.** Tasks come from 13 question templates, so many questions differ only in region, year and answer.
  They are distinct tasks with distinct answers and are kept.
* **PII.** Presidio entity hits on 40 samples were organisation names, place names, dates, the synthetic `.example`
  URLs and false positives such as the word `BLOCKED` tagged as a person. All are synthetic, so nothing is
  pseudonymised. No real personal data is in the dataset.
