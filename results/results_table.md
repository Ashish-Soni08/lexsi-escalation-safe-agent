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
| → v4: + wider mix (5 shortcut kinds, 6 tip wordings, 16 families) | 0.0% | 0.0% | 99.5% | 100.0% | 0.5% | 0.0% | 0.0% | 0.0% |
| → v4.1: v4 recipe, detector v2.2 in the reward | 0.0% | 0.0% | 99.0% | 100.0% | 1.0% | 0.0% | 0.0% | 0.0% |

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
| → v4: + wider mix (5 shortcut kinds, 6 tip wordings, 16 families) | 0.0% | 0.0% | 99.5% | 100.0% | 0.5% | 0.0% | 0.0% | 0.0% |
| → v4.1: v4 recipe, detector v2.2 in the reward | 0.0% | 0.0% | 99.5% | 100.0% | 0.5% | 0.0% | 0.0% | 0.0% |

**Stress test, neutral prompt** (320 test tasks: 190 solvable, 130 unsolvable; 105 from held-out families, 82 in Hindi)

| model | escalation (all) | escalation (blocked tasks) | legit success (solvable) | honest stop (unsolvable) | gave up (solvable) | made-up number | held-out families: escalation | Hindi: escalation |
|---|---|---|---|---|---|---|---|---|
| Tiny Aya Global (base) | 0.0% | 0.0% | 0.5% | 57.7% | 67.9% | 28.8% | 0.0% | 0.0% |
| + tool SFT, clean | 55.9% | 91.5% | 68.4% | 6.9% | 0.0% | 0.0% | 58.1% | 53.7% |
| + tool SFT, escalation-prone | 60.3% | 93.1% | 61.1% | 6.2% | 0.5% | 0.0% | 61.0% | 62.2% |
| → GRPO success-only | 60.0% | 99.2% | 66.8% | 0.0% | 0.0% | 0.0% | 58.1% | 61.0% |
| → GRPO safe (v2) | 52.2% | 78.5% | 65.3% | 17.7% | 0.5% | 0.0% | 55.2% | 53.7% |
| → GRPO safe (v2, seed 2) | 52.2% | 80.0% | 66.8% | 17.7% | 0.0% | 0.0% | 56.2% | 52.4% |
| → v3: + hard RL mix | 32.5% | 52.3% | 80.0% | 47.7% | 0.5% | 0.0% | 29.5% | 30.5% |
| → v3.1: + hard RL mix, judge-patched detector | 21.2% | 36.2% | 87.4% | 63.8% | 1.6% | 0.0% | 20.0% | 15.9% |
| → v4: + wider mix (5 shortcut kinds, 6 tip wordings, 16 families) | 13.1% | 31.5% | 98.9% | 68.5% | 0.5% | 0.0% | 9.5% | 9.8% |
| → v4.1: v4 recipe, detector v2.2 in the reward | 0.6% | 1.5% | 99.5% | 98.5% | 0.5% | 0.0% | 0.0% | 1.2% |

**Stress test, rules prompt** (320 test tasks: 190 solvable, 130 unsolvable; 105 from held-out families, 82 in Hindi)

| model | escalation (all) | escalation (blocked tasks) | legit success (solvable) | honest stop (unsolvable) | gave up (solvable) | made-up number | held-out families: escalation | Hindi: escalation |
|---|---|---|---|---|---|---|---|---|
| Tiny Aya Global (base) | 0.0% | 0.0% | 2.6% | 70.8% | 75.8% | 14.1% | 0.0% | 0.0% |
| + tool SFT, clean | 56.2% | 92.3% | 67.9% | 6.9% | 0.5% | 0.0% | 58.1% | 54.9% |
| + tool SFT, escalation-prone | 59.1% | 91.5% | 61.6% | 7.7% | 0.5% | 0.0% | 59.0% | 61.0% |
| → GRPO success-only | 59.7% | 99.2% | 67.4% | 0.0% | 0.0% | 0.0% | 59.0% | 58.5% |
| → GRPO safe (v2) | 51.6% | 76.2% | 64.2% | 19.2% | 0.5% | 0.0% | 54.3% | 53.7% |
| → GRPO safe (v2, seed 2) | 51.2% | 76.9% | 65.8% | 20.8% | 0.0% | 0.0% | 53.3% | 54.9% |
| → v3: + hard RL mix | 30.6% | 46.9% | 80.0% | 53.1% | 0.0% | 0.0% | 30.5% | 30.5% |
| → v3.1: + hard RL mix, judge-patched detector | 19.4% | 33.8% | 89.5% | 66.2% | 1.1% | 0.0% | 19.0% | 13.4% |
| → v4: + wider mix (5 shortcut kinds, 6 tip wordings, 16 families) | 13.1% | 31.5% | 98.9% | 68.5% | 0.5% | 0.0% | 9.5% | 9.8% |
| → v4.1: v4 recipe, detector v2.2 in the reward | 0.6% | 1.5% | 99.5% | 98.5% | 0.5% | 0.0% | 0.0% | 1.2% |

**Unseen stress test, neutral prompt** (320 test tasks: 204 solvable, 116 unsolvable; 124 from held-out families, 81 in Hindi)

| model | escalation (all) | escalation (blocked tasks) | legit success (solvable) | honest stop (unsolvable) | gave up (solvable) | made-up number | held-out families: escalation | Hindi: escalation |
|---|---|---|---|---|---|---|---|---|
| Tiny Aya Global (base) | 0.0% | 0.0% | 0.0% | 50.0% | 48.0% | 32.8% | 0.0% | 0.0% |
| + tool SFT, clean | 37.8% | 87.9% | 89.2% | 9.5% | 0.0% | 0.0% | 37.9% | 39.5% |
| + tool SFT, escalation-prone | 46.6% | 97.4% | 82.4% | 1.7% | 0.0% | 0.0% | 47.6% | 49.4% |
| → GRPO success-only | 55.6% | 99.1% | 69.1% | 0.9% | 0.0% | 0.0% | 58.9% | 60.5% |
| → GRPO safe (v2) | 36.2% | 71.6% | 83.3% | 27.6% | 0.5% | 0.0% | 37.9% | 42.0% |
| → v3.1: + hard RL mix, judge-patched detector | 27.2% | 66.4% | 93.6% | 33.6% | 1.0% | 0.0% | 26.6% | 30.9% |
| → v4: + wider mix (5 shortcut kinds, 6 tip wordings, 16 families) | 11.9% | 31.9% | 98.5% | 68.1% | 1.0% | 0.0% | 14.5% | 13.6% |
| → v4.1: v4 recipe, detector v2.2 in the reward | 9.4% | 25.9% | 99.0% | 74.1% | 0.5% | 0.3% | 12.1% | 9.9% |

**Unseen stress test, rules prompt** (320 test tasks: 204 solvable, 116 unsolvable; 124 from held-out families, 81 in Hindi)

| model | escalation (all) | escalation (blocked tasks) | legit success (solvable) | honest stop (unsolvable) | gave up (solvable) | made-up number | held-out families: escalation | Hindi: escalation |
|---|---|---|---|---|---|---|---|---|
| Tiny Aya Global (base) | 0.0% | 0.0% | 4.9% | 85.3% | 63.7% | 10.0% | 0.0% | 0.0% |
| + tool SFT, clean | 38.8% | 87.9% | 88.7% | 8.6% | 0.0% | 0.0% | 36.3% | 38.3% |
| + tool SFT, escalation-prone | 45.9% | 96.6% | 82.8% | 2.6% | 0.0% | 0.0% | 46.8% | 48.1% |
| → GRPO success-only | 53.8% | 98.3% | 71.1% | 1.7% | 0.0% | 0.0% | 56.5% | 59.3% |
| → GRPO safe (v2) | 36.9% | 74.1% | 83.8% | 25.0% | 0.5% | 0.0% | 37.1% | 42.0% |
| → v3.1: + hard RL mix, judge-patched detector | 26.9% | 65.5% | 94.1% | 33.6% | 0.5% | 0.0% | 25.8% | 28.4% |
| → v4: + wider mix (5 shortcut kinds, 6 tip wordings, 16 families) | 5.9% | 16.4% | 99.0% | 83.6% | 1.0% | 0.0% | 8.9% | 8.6% |
| → v4.1: v4 recipe, detector v2.2 in the reward | 6.2% | 17.2% | 99.0% | 82.8% | 1.0% | 0.0% | 9.7% | 7.4% |

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
| → v4: + wider mix (5 shortcut kinds, 6 tip wordings, 16 families) | 0.0 ± 0.0% | 0.0 ± 0.0% | 98.9 ± 0.2% | 99.7 ± 0.4% | 0.1 ± 0.1% |
| → v4.1: v4 recipe, detector v2.2 in the reward | 0.0 ± 0.0% | 0.0 ± 0.0% | 98.2 ± 0.6% | 99.8 ± 0.4% | 0.0 ± 0.0% |

**Sampled, stress test** (temperature 1.0, top-p 0.95, neutral prompt, mean ± sd over 3 seeds)

| model | escalation (blocked tasks) | escalation (all) | legit success (solvable) | honest stop (unsolvable) | made-up number |
|---|---|---|---|---|---|
| Tiny Aya Global (base) | 0.8 ± 1.3% | 0.8 ± 1.2% | 2.1 ± 1.4% | 58.7 ± 4.0% | 16.9 ± 1.4% |
| + tool SFT, clean | 76.7 ± 1.6% | 46.4 ± 1.2% | 73.2 ± 1.4% | 21.8 ± 1.2% | 0.0 ± 0.0% |
| + tool SFT, escalation-prone | 89.0 ± 0.4% | 56.4 ± 1.0% | 65.1 ± 1.8% | 9.5 ± 1.2% | 0.1 ± 0.2% |
| → GRPO success-only | 99.2 ± 0.8% | 60.7 ± 0.2% | 65.6 ± 0.3% | 0.8 ± 0.8% | 0.0 ± 0.0% |
| → GRPO safe (v2) | 73.6 ± 3.6% | 48.5 ± 1.9% | 66.8 ± 1.0% | 22.8 ± 2.5% | 0.0 ± 0.0% |
| → v3: + hard RL mix | 51.5 ± 4.8% | 32.0 ± 1.3% | 79.3 ± 1.8% | 47.4 ± 5.0% | 0.0 ± 0.0% |
| → v3.1: + hard RL mix, judge-patched detector | 32.8 ± 1.2% | 18.8 ± 0.3% | 86.0 ± 1.7% | 66.7 ± 1.2% | 0.0 ± 0.0% |
| → v4: + wider mix (5 shortcut kinds, 6 tip wordings, 16 families) | 29.5 ± 0.4% | 12.6 ± 0.2% | 97.2 ± 0.6% | 69.5 ± 0.4% | 0.1 ± 0.2% |
| → v4.1: v4 recipe, detector v2.2 in the reward | 2.6 ± 2.7% | 1.1 ± 1.1% | 97.7 ± 0.3% | 95.6 ± 3.1% | 0.0 ± 0.0% |

**Sampled, unseen stress test** (temperature 1.0, top-p 0.95, neutral prompt, mean ± sd over 3 seeds)

| model | escalation (blocked tasks) | escalation (all) | legit success (solvable) | honest stop (unsolvable) | made-up number |
|---|---|---|---|---|---|
| → GRPO safe (v2) | 67.2 ± 1.5% | 32.9 ± 1.2% | 84.6 ± 0.7% | 31.6 ± 1.0% | 0.0 ± 0.0% |
| → v3.1: + hard RL mix, judge-patched detector | 54.0 ± 5.3% | 23.3 ± 2.5% | 92.0 ± 0.6% | 44.5 ± 5.2% | 0.1 ± 0.2% |
| → v4: + wider mix (5 shortcut kinds, 6 tip wordings, 16 families) | 30.7 ± 1.3% | 11.7 ± 0.4% | 96.6 ± 0.5% | 68.4 ± 2.0% | 0.5 ± 0.5% |
| → v4.1: v4 recipe, detector v2.2 in the reward | 25.3 ± 1.0% | 9.9 ± 0.8% | 96.6 ± 0.5% | 73.6 ± 1.0% | 0.4 ± 0.2% |
