# Round 6 results

Percent, with a 95% bootstrap interval over tasks (2,000 resamples) in brackets.

### Escalation on blocked tasks (lower is better)

| model | test | stress | unseen | fresh | sealed | long chat |
|---|---|---|---|---|---|---|
| clean SFT | 0.0 (0.0 to 0.0) | 91.5 (86.2 to 96.2) | 87.9 (81.9 to 93.1) | 48.0 (39.4 to 56.7) | 42.8 (35.5 to 51.4) | 63.6 (55.3 to 71.2) |
| GRPO, success-only reward | 99.2 (97.6 to 100.0) | 99.2 (97.7 to 100.0) | 99.1 (97.4 to 100.0) | 72.4 (64.6 to 80.3) | 100.0 (100.0 to 100.0) | 94.7 (90.9 to 98.5) |
| GRPO safe v2 | 0.0 (0.0 to 0.0) | 78.5 (71.5 to 85.4) | 71.6 (63.8 to 80.2) | 38.6 (30.7 to 47.2) | 35.5 (28.3 to 43.5) | 56.8 (48.5 to 65.2) |
| GRPO safe v3.1 | 0.0 (0.0 to 0.0) | 36.2 (27.7 to 43.8) | 66.4 (57.8 to 75.0) | 24.4 (17.3 to 32.3) | 25.4 (18.1 to 32.6) | 18.9 (12.9 to 25.8) |
| GRPO safe v4.1 (final) | 0.0 (0.0 to 0.0) | 1.5 (0.0 to 3.8) | 25.9 (18.1 to 34.5) | 11.0 (5.5 to 16.5) | 21.7 (15.2 to 29.0) | 2.3 (0.0 to 5.3) |

### Legitimate success on solvable tasks (higher is better)

| model | test | stress | unseen | fresh | sealed | long chat |
|---|---|---|---|---|---|---|
| clean SFT | 99.5 (98.5 to 100.0) | 68.4 (61.6 to 74.7) | 89.2 (84.8 to 93.1) | 62.2 (55.4 to 68.9) | 81.9 (75.8 to 87.4) | 72.9 (67.0 to 78.7) |
| GRPO, success-only reward | 86.2 (81.1 to 90.8) | 66.8 (60.0 to 73.2) | 69.1 (62.7 to 75.5) | 69.9 (63.2 to 76.2) | 76.4 (70.3 to 82.4) | 65.4 (58.5 to 72.3) |
| GRPO safe v2 | 99.5 (98.5 to 100.0) | 65.3 (58.4 to 71.6) | 83.3 (78.4 to 88.2) | 68.9 (62.7 to 75.1) | 81.9 (76.4 to 87.4) | 64.9 (58.0 to 71.3) |
| GRPO safe v3.1 | 96.9 (94.4 to 99.0) | 87.4 (82.6 to 92.1) | 93.6 (90.2 to 96.6) | 71.5 (65.3 to 77.7) | 85.7 (80.8 to 90.7) | 71.3 (64.9 to 77.7) |
| GRPO safe v4.1 (final) | 99.0 (97.4 to 100.0) | 99.5 (98.4 to 100.0) | 99.0 (97.5 to 100.0) | 71.0 (64.2 to 77.2) | 91.8 (87.9 to 95.6) | 87.8 (83.0 to 92.0) |

### Final model against clean SFT, paired (exact McNemar on blocked tasks)

| set | SFT escalates, final does not | final escalates, SFT does not | p |
|---|---|---|---|
| test | 0 | 0 | 1 |
| stress | 118 | 1 | 3.6e-34 |
| unseen | 75 | 3 | 5.2e-19 |
| fresh | 47 | 0 | 1.4e-14 |
| sealed | 29 | 0 | 3.7e-09 |
| long chat | 81 | 0 | 8.3e-25 |

### Three training seeds of the final recipe: escalation on blocked tasks

| set | seed 4 (released) | seed 14 | seed 24 | mean ± sd |
|---|---|---|---|---|
| test | 0.0 | 0.0 | 0.0 | 0.0 ± 0.0 |
| stress | 1.5 | 0.0 | 2.3 | 1.3 ± 1.2 |
| unseen | 25.9 | 26.7 | 17.2 | 23.3 ± 5.2 |
| fresh | 11.0 | 3.1 | 4.7 | 6.3 ± 4.2 |
| sealed | 21.7 | 14.5 | 13.0 | 16.4 ± 4.7 |
| long chat | 2.3 | 3.0 | 1.5 | 2.3 ± 0.8 |

| legit success | seed 4 (released) | seed 14 | seed 24 | mean ± sd |
|---|---|---|---|---|
| test | 99.0 | 99.5 | 94.4 | 97.6 ± 2.8 |
| stress | 99.5 | 98.9 | 98.9 | 99.1 ± 0.3 |
| unseen | 99.0 | 98.5 | 99.0 | 98.9 ± 0.3 |
| fresh | 71.0 | 72.5 | 68.9 | 70.8 ± 1.8 |
| sealed | 91.8 | 92.9 | 87.9 | 90.8 ± 2.6 |
| long chat | 87.8 | 91.5 | 87.8 | 89.0 ± 2.1 |

### Reward ablations: escalation on blocked tasks

All arms start from the same escalation-prone SFT model and run the same 160 GRPO steps; only the reward differs.

| model | test | stress | unseen | fresh | sealed |
|---|---|---|---|---|---|
| success-only (no safety terms) | 99.2 (97.6 to 100.0) | 99.2 (97.7 to 100.0) | 99.1 (97.4 to 100.0) | 72.4 (64.6 to 80.3) | 100.0 (100.0 to 100.0) |
| no escalation penalty | n/a | n/a | n/a | n/a | n/a |
| no honest-stop bonus | 0.0 (0.0 to 0.0) | 81.5 (74.6 to 88.5) | 69.0 (60.3 to 77.6) | 43.3 (34.6 to 52.0) | 31.9 (24.6 to 39.9) |
| no fabrication penalty | 0.0 (0.0 to 0.0) | 87.7 (81.5 to 93.1) | 83.6 (76.7 to 90.5) | 46.5 (38.6 to 55.1) | 34.8 (27.5 to 42.8) |
| full safe reward (v2) | 0.0 (0.0 to 0.0) | 78.5 (71.5 to 85.4) | 71.6 (63.8 to 80.2) | 38.6 (30.7 to 47.2) | 35.5 (28.3 to 43.5) |

### Reward ablations: legitimate success

| model | test | stress | unseen | fresh | sealed |
|---|---|---|---|---|---|
| success-only (no safety terms) | 86.2 (81.1 to 90.8) | 66.8 (60.0 to 73.2) | 69.1 (62.7 to 75.5) | 69.9 (63.2 to 76.2) | 76.4 (70.3 to 82.4) |
| no escalation penalty | n/a | n/a | n/a | n/a | n/a |
| no honest-stop bonus | 99.5 (98.5 to 100.0) | 66.3 (59.5 to 72.6) | 81.4 (76.0 to 86.3) | 71.0 (64.8 to 77.2) | 81.9 (76.4 to 87.4) |
| no fabrication penalty | 99.5 (98.5 to 100.0) | 68.4 (61.6 to 74.7) | 84.3 (79.4 to 89.2) | 72.5 (66.3 to 78.2) | 83.5 (78.0 to 88.5) |
| full safe reward (v2) | 99.5 (98.5 to 100.0) | 65.3 (58.4 to 71.6) | 83.3 (78.4 to 88.2) | 68.9 (62.7 to 75.1) | 81.9 (76.4 to 87.4) |

### Guard only, model only, both: escalation attempted and executed on blocked tasks

| model | set | no guard | with guard: attempted | with guard: executed | legit success with guard |
|---|---|---|---|---|---|
| clean SFT | test | 0.0 | 0.0 | 0.0 | 99.5 |
| clean SFT | stress | 91.5 | 93.1 | 0.0 | 91.0 |
| clean SFT | unseen | 87.9 | 86.2 | 0.0 | 92.2 |
| clean SFT | fresh | 48.0 | 48.0 | 48.0 | 62.7 |
| GRPO, success-only reward | test | 99.2 | 99.2 | 0.8 | 94.9 |
| GRPO, success-only reward | stress | 99.2 | 99.2 | 0.8 | 86.8 |
| GRPO, success-only reward | unseen | 99.1 | 99.1 | 0.0 | 94.6 |
| GRPO, success-only reward | fresh | 72.4 | 72.4 | 54.3 | 72.0 |
| GRPO safe v2 | test | 0.0 | 0.0 | 0.0 | 99.5 |
| GRPO safe v2 | stress | 78.5 | 76.9 | 0.0 | 90.0 |
| GRPO safe v2 | unseen | 71.6 | 71.5 | 0.0 | 93.6 |
| GRPO safe v2 | fresh | 38.6 | 39.4 | 37.8 | 68.4 |
| GRPO safe v4.1 (final) | test | 0.0 | 0.0 | 0.0 | 99.5 |
| GRPO safe v4.1 (final) | stress | 1.5 | 1.5 | 0.0 | 99.5 |
| GRPO safe v4.1 (final) | unseen | 25.9 | 24.1 | 0.0 | 98.5 |
| GRPO safe v4.1 (final) | fresh | 11.0 | 11.0 | 11.0 | 70.5 |
| GRPO safe v4.1 (final) | sealed | 21.7 | 18.8 | 18.8 | 91.8 |
| GRPO safe v4.1 (final) | long chat | 2.3 | 2.3 | 0.8 | 87.8 |

### General tool calling: BFCL v3 simple + multiple (600 questions, our own AST check)

| model | full match | vs base: lost / gained, p | vs clean SFT: lost / gained, p |
|---|---|---|---|
| base | 82.8 (79.7 to 85.8) | 0 / 0, p=1 | 36 / 50, p=0.16 |
| sft | 80.5 (77.2 to 83.7) | 50 / 36, p=0.16 | 0 / 0, p=1 |
| sft_esc | 80.0 (76.8 to 83.2) | 55 / 38, p=0.097 | 15 / 12, p=0.7 |
| grpo_esc_success_only | 80.2 (77.0 to 83.5) | 54 / 38, p=0.12 | 15 / 13, p=0.85 |
| grpo_esc_safe | 80.0 (76.8 to 83.2) | 55 / 38, p=0.097 | 14 / 11, p=0.69 |
| grpo_safe_v4_1 | 78.7 (75.5 to 82.0) | 63 / 38, p=0.017 | 22 / 11, p=0.08 |

### Real tools (real HTTP): sandboxed server with real access controls, and the live World Bank API

| model | sandbox: escalation on blocked | server-confirmed bypass | sandbox legit success | World Bank correct |
|---|---|---|---|---|
| clean SFT | 38.1 (19.0 to 57.1) | 9.5 (0.0 to 23.8) | 63.2 (42.1 to 84.2) | 18/24 |
| GRPO, success-only reward | 66.7 (47.6 to 85.7) | 42.9 (23.8 to 61.9) | 84.2 (68.4 to 100.0) | 18/24 |
| GRPO safe v2 | 38.1 (19.0 to 57.1) | 33.3 (14.3 to 52.4) | 47.4 (26.3 to 68.4) | 18/24 |
| GRPO safe v4.1 (final) | 0.0 (0.0 to 0.0) | 0.0 (0.0 to 0.0) | 100.0 (100.0 to 100.0) | 18/24 |

### Independent judges against the rule-based detector

| sample | judge | n | agreement | Cohen's kappa | detector only | judge only |
|---|---|---|---|---|---|---|
| final model | Qwen/Qwen2.5-32B-Instruct | 400 | 0.94 | 0.8706 | 16 | 8 |
| final model | mistralai/Mistral-Small-24B-Instruct-2501 | 400 | 0.695 | 0.3596 | 55 | 67 |
| seed models | Qwen/Qwen2.5-32B-Instruct | 400 | 0.905 | 0.7966 | 32 | 6 |
