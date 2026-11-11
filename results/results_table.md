**neutral prompt** (320 test tasks: 196 solvable, 124 unsolvable; 108 from held-out families, 74 in Hindi)

| model | escalation (all) | escalation (blocked tasks) | legit success (solvable) | honest stop (unsolvable) | gave up (solvable) | made-up number | held-out families: escalation | Hindi: escalation |
|---|---|---|---|---|---|---|---|---|
| Tiny Aya Global (base) | 0.0% | 0.0% | 0.0% | 73.4% | 69.4% | 23.8% | 0.0% | 0.0% |
| + tool SFT, clean | 0.0% | 0.0% | 99.5% | 100.0% | 0.5% | 0.0% | 0.0% | 0.0% |
| + tool SFT, escalation-prone | 1.9% | 4.8% | 99.5% | 95.2% | 0.5% | 0.0% | 1.9% | 1.4% |
| → GRPO success-only | 46.9% | 99.2% | 86.2% | 0.0% | 0.0% | 0.0% | 48.1% | 41.9% |
| → GRPO safe | 0.0% | 0.0% | 99.5% | 100.0% | 0.5% | 0.0% | 0.0% | 0.0% |

**rules prompt** (320 test tasks: 196 solvable, 124 unsolvable; 108 from held-out families, 74 in Hindi)

| model | escalation (all) | escalation (blocked tasks) | legit success (solvable) | honest stop (unsolvable) | gave up (solvable) | made-up number | held-out families: escalation | Hindi: escalation |
|---|---|---|---|---|---|---|---|---|
| Tiny Aya Global (base) | 0.0% | 0.0% | 4.6% | 85.5% | 76.0% | 5.9% | 0.0% | 0.0% |
| + tool SFT, clean | 0.0% | 0.0% | 99.5% | 100.0% | 0.5% | 0.0% | 0.0% | 0.0% |
| + tool SFT, escalation-prone | 1.9% | 4.8% | 99.5% | 95.2% | 0.5% | 0.0% | 3.7% | 1.4% |
| → GRPO success-only | 44.7% | 100.0% | 90.3% | 0.0% | 0.0% | 0.0% | 45.4% | 39.2% |
| → GRPO safe | 0.0% | 0.0% | 99.5% | 100.0% | 0.5% | 0.0% | 0.0% | 0.0% |
