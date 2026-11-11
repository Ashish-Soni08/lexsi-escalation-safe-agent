
**neutral prompt** (320 test tasks: 196 solvable, 124 unsolvable; 108 from held-out families, 74 in Hindi)

| model | escalation (all) | escalation (blocked tasks) | legit success (solvable) | honest stop (unsolvable) | gave up (solvable) | made-up number | held-out families: escalation | Hindi: escalation |
|---|---|---|---|---|---|---|---|---|
| Tiny Aya Global (base) | 0.0% | 0.0% | 0.5% | 74.2% | 69.9% | 23.4% | 0.0% | 0.0% |
| + tool SFT, clean | 0.0% | 0.0% | 76.5% | 100.0% | 14.8% | 5.0% | 0.0% | 0.0% |
| + tool SFT, escalation-prone | 4.7% | 11.3% | 76.0% | 88.7% | 12.8% | 6.2% | 4.6% | 4.0% |
| → GRPO success-only | 64.1% | 93.5% | 52.0% | 1.6% | 0.0% | 1.2% | 64.8% | 63.5% |
| → GRPO safe | 0.0% | 0.0% | 76.5% | 100.0% | 18.4% | 2.8% | 0.0% | 0.0% |

**rules prompt** (320 test tasks: 196 solvable, 124 unsolvable; 108 from held-out families, 74 in Hindi)

| model | escalation (all) | escalation (blocked tasks) | legit success (solvable) | honest stop (unsolvable) | gave up (solvable) | made-up number | held-out families: escalation | Hindi: escalation |
|---|---|---|---|---|---|---|---|---|
| Tiny Aya Global (base) | 0.0% | 0.0% | 4.1% | 88.7% | 81.1% | 4.7% | 0.0% | 0.0% |
| + tool SFT, clean | 0.0% | 0.0% | 76.5% | 100.0% | 13.8% | 5.9% | 0.0% | 0.0% |
| + tool SFT, escalation-prone | 0.9% | 2.4% | 77.0% | 97.6% | 14.8% | 4.7% | 0.9% | 4.0% |
| → GRPO success-only | 63.1% | 93.5% | 53.6% | 0.0% | 0.0% | 1.2% | 62.0% | 64.9% |
| → GRPO safe | 0.0% | 0.0% | 77.0% | 100.0% | 16.8% | 3.4% | 0.0% | 0.0% |
