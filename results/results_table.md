**neutral prompt** (320 test tasks: 196 solvable, 124 unsolvable; 108 from held-out families, 74 in Hindi)

| model | escalation (all) | escalation (blocked tasks) | legit success (solvable) | honest stop (unsolvable) | gave up (solvable) | made-up number | held-out families: escalation | Hindi: escalation |
|---|---|---|---|---|---|---|---|---|
| Tiny Aya Global (base) | 0.0% | 0.0% | 0.0% | 73.4% | 69.4% | 23.8% | 0.0% | 0.0% |
| + tool SFT, clean | 0.0% | 0.0% | 99.5% | 100.0% | 0.5% | 0.0% | 0.0% | 0.0% |
| + tool SFT, escalation-prone | 1.9% | 4.8% | 99.5% | 95.2% | 0.5% | 0.0% | 1.9% | 1.4% |
| → GRPO success-only | 46.9% | 99.2% | 86.2% | 0.0% | 0.0% | 0.0% | 48.1% | 41.9% |
| → GRPO safe (v2) | 0.0% | 0.0% | 99.5% | 100.0% | 0.5% | 0.0% | 0.0% | 0.0% |
| → GRPO safe (v2, seed 2) | 0.0% | 0.0% | 99.5% | 100.0% | 0.5% | 0.0% | 0.0% | 0.0% |
| → v3: + hard RL mix | 0.0% | 0.0% | 99.5% | 100.0% | 0.5% | 0.0% | 0.0% | 0.0% |
| → v3.1: + hard RL mix, judge-patched detector | 0.0% | 0.0% | 96.9% | 100.0% | 3.1% | 0.0% | 0.0% | 0.0% |

**rules prompt** (320 test tasks: 196 solvable, 124 unsolvable; 108 from held-out families, 74 in Hindi)

| model | escalation (all) | escalation (blocked tasks) | legit success (solvable) | honest stop (unsolvable) | gave up (solvable) | made-up number | held-out families: escalation | Hindi: escalation |
|---|---|---|---|---|---|---|---|---|
| Tiny Aya Global (base) | 0.0% | 0.0% | 4.6% | 85.5% | 76.0% | 5.9% | 0.0% | 0.0% |
| + tool SFT, clean | 0.0% | 0.0% | 99.5% | 100.0% | 0.5% | 0.0% | 0.0% | 0.0% |
| + tool SFT, escalation-prone | 1.9% | 4.8% | 99.5% | 95.2% | 0.5% | 0.0% | 3.7% | 1.4% |
| → GRPO success-only | 44.7% | 100.0% | 90.3% | 0.0% | 0.0% | 0.0% | 45.4% | 39.2% |
| → GRPO safe (v2) | 0.0% | 0.0% | 99.5% | 100.0% | 0.5% | 0.0% | 0.0% | 0.0% |
| → GRPO safe (v2, seed 2) | 0.0% | 0.0% | 99.5% | 100.0% | 0.5% | 0.0% | 0.0% | 0.0% |
| → v3: + hard RL mix | 0.0% | 0.0% | 99.5% | 100.0% | 0.5% | 0.0% | 0.0% | 0.0% |
| → v3.1: + hard RL mix, judge-patched detector | 0.0% | 0.0% | 97.4% | 100.0% | 2.6% | 0.0% | 0.0% | 0.0% |

**Stress test, neutral prompt** (320 test tasks: 190 solvable, 130 unsolvable; 105 from held-out families, 82 in Hindi)

| model | escalation (all) | escalation (blocked tasks) | legit success (solvable) | honest stop (unsolvable) | gave up (solvable) | made-up number | held-out families: escalation | Hindi: escalation |
|---|---|---|---|---|---|---|---|---|
| Tiny Aya Global (base) | 0.0% | 0.0% | 0.5% | 57.7% | 67.9% | 28.8% | 0.0% | 0.0% |
| + tool SFT, clean | 52.8% | 84.6% | 68.9% | 13.8% | 0.0% | 0.0% | 54.3% | 48.8% |
| + tool SFT, escalation-prone | 58.1% | 87.7% | 61.1% | 11.5% | 0.5% | 0.0% | 59.0% | 62.2% |
| → GRPO success-only | 60.0% | 99.2% | 66.8% | 0.0% | 0.0% | 0.0% | 58.1% | 61.0% |
| → GRPO safe (v2) | 49.4% | 73.1% | 66.3% | 23.1% | 0.5% | 0.0% | 52.4% | 46.3% |
| → GRPO safe (v2, seed 2) | 50.3% | 76.2% | 67.4% | 21.5% | 0.0% | 0.0% | 54.3% | 50.0% |
| → v3: + hard RL mix | 11.2% | 16.2% | 91.1% | 83.8% | 0.5% | 0.0% | 17.1% | 12.2% |
| → v3.1: + hard RL mix, judge-patched detector | 4.7% | 6.9% | 94.7% | 93.1% | 2.1% | 0.0% | 7.6% | 3.7% |

**Stress test, rules prompt** (320 test tasks: 190 solvable, 130 unsolvable; 105 from held-out families, 82 in Hindi)

| model | escalation (all) | escalation (blocked tasks) | legit success (solvable) | honest stop (unsolvable) | gave up (solvable) | made-up number | held-out families: escalation | Hindi: escalation |
|---|---|---|---|---|---|---|---|---|
| Tiny Aya Global (base) | 0.0% | 0.0% | 2.6% | 70.8% | 75.8% | 14.1% | 0.0% | 0.0% |
| + tool SFT, clean | 52.5% | 86.2% | 70.0% | 13.1% | 0.5% | 0.0% | 54.3% | 50.0% |
| + tool SFT, escalation-prone | 56.6% | 86.2% | 62.1% | 13.1% | 0.5% | 0.0% | 56.2% | 56.1% |
| → GRPO success-only | 59.7% | 99.2% | 67.4% | 0.0% | 0.0% | 0.0% | 59.0% | 58.5% |
| → GRPO safe (v2) | 47.8% | 69.2% | 65.8% | 26.2% | 0.5% | 0.0% | 50.5% | 45.1% |
| → GRPO safe (v2, seed 2) | 49.4% | 72.3% | 65.8% | 25.4% | 0.0% | 0.0% | 52.4% | 52.4% |
| → v3: + hard RL mix | 7.5% | 9.2% | 93.2% | 90.0% | 0.0% | 0.0% | 17.1% | 9.8% |
| → v3.1: + hard RL mix, judge-patched detector | 2.5% | 3.8% | 96.8% | 96.2% | 1.6% | 0.0% | 5.7% | 1.2% |

**Sampled, test set** (temperature 1.0, top-p 0.95, neutral prompt, mean ± sd over 5 seeds)

| model | escalation (blocked tasks) | escalation (all) | legit success (solvable) | honest stop (unsolvable) | made-up number |
|---|---|---|---|---|---|
| Tiny Aya Global (base) | 0.3 ± 0.4% | 0.2 ± 0.1% | 3.8 ± 0.6% | 63.5 ± 4.5% | 15.2 ± 3.7% |
| + tool SFT, clean | 0.2 ± 0.4% | 0.1 ± 0.1% | 98.8 ± 0.6% | 99.7 ± 0.4% | 0.0 ± 0.0% |
| + tool SFT, escalation-prone | 38.9 ± 3.8% | 16.6 ± 1.6% | 97.1 ± 1.3% | 61.1 ± 3.8% | 0.0 ± 0.0% |
| → GRPO success-only | 98.9 ± 1.2% | 47.4 ± 0.8% | 85.2 ± 0.8% | 0.2 ± 0.4% | 0.0 ± 0.0% |
| → GRPO safe (v2) | 0.5 ± 0.4% | 0.2 ± 0.2% | 99.3 ± 0.6% | 99.5 ± 0.4% | 0.0 ± 0.0% |
| → v3: + hard RL mix | 0.3 ± 0.7% | 0.1 ± 0.3% | 99.1 ± 0.4% | 99.7 ± 0.7% | 0.0 ± 0.0% |
| → v3.1: + hard RL mix, judge-patched detector | 0.3 ± 0.7% | 0.1 ± 0.3% | 96.8 ± 0.7% | 99.7 ± 0.7% | 0.0 ± 0.0% |

**Sampled, stress test** (temperature 1.0, top-p 0.95, neutral prompt, mean ± sd over 3 seeds)

| model | escalation (blocked tasks) | escalation (all) | legit success (solvable) | honest stop (unsolvable) | made-up number |
|---|---|---|---|---|---|
| Tiny Aya Global (base) | 0.8 ± 1.3% | 0.8 ± 1.2% | 2.1 ± 1.4% | 58.7 ± 4.0% | 16.9 ± 1.4% |
| + tool SFT, clean | 71.3 ± 0.9% | 43.1 ± 1.7% | 74.2 ± 2.7% | 27.2 ± 1.2% | 0.0 ± 0.0% |
| + tool SFT, escalation-prone | 83.8 ± 2.8% | 53.8 ± 1.9% | 66.0 ± 1.6% | 14.6 ± 4.1% | 0.1 ± 0.2% |
| → GRPO success-only | 99.2 ± 0.8% | 60.7 ± 0.2% | 65.6 ± 0.3% | 0.8 ± 0.8% | 0.0 ± 0.0% |
| → GRPO safe (v2) | 67.9 ± 1.6% | 45.5 ± 1.3% | 67.7 ± 0.8% | 28.5 ± 0.8% | 0.0 ± 0.0% |
| → v3: + hard RL mix | 22.6 ± 3.2% | 15.4 ± 1.3% | 86.7 ± 2.6% | 76.2 ± 3.1% | 0.1 ± 0.2% |
| → v3.1: + hard RL mix, judge-patched detector | 8.2 ± 1.8% | 5.3 ± 0.5% | 91.2 ± 1.7% | 91.3 ± 1.6% | 0.0 ± 0.0% |
