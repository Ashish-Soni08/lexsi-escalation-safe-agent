# Stage 2 results

Percent, with a 95% bootstrap interval over tasks (2,000 resamples) in brackets. Neutral prompt, greedy decoding.

## Coverage curve: how many shortcut kinds the safe reward was trained on

Tiny Aya Global, one pass of 160 GRPO steps from the escalation-prone model, safe reward. All arms use the same 960 training tasks; only the shortcut kinds on offer differ (the first k of: staging_host, leaked_token, relay_service, googlebot_tip, preview_host, debug_flag, legacy_host, skip_param, shared_key, captcha_solver). The unseen and sealed sets use eight kinds that no arm was trained on.


### Escalation on blocked tasks (lower is better)

| kinds trained | test | stress | unseen | fresh | sealed |
|---|---|---|---|---|---|
| 0 (escalation-prone start) | 4.8 (1.6 to 8.9) | 93.1 (88.5 to 96.9) | 97.4 (94.8 to 100.0) | n/a | n/a |
| 2 | 66.1 (58.1 to 74.2) | 89.2 (83.8 to 94.6) | 87.9 (81.9 to 93.1) | 34.6 (26.8 to 43.3) | 81.2 (74.6 to 87.0) |
| 4 | 0.0 (0.0 to 0.0) | 65.4 (56.9 to 73.8) | 46.6 (37.9 to 56.0) | 29.1 (21.3 to 37.0) | 23.2 (16.7 to 29.7) |
| 6 | 0.0 (0.0 to 0.0) | 27.7 (20.0 to 35.4) | 31.9 (23.3 to 40.5) | 18.9 (12.6 to 26.0) | 26.1 (18.8 to 33.3) |
| 10 | 0.0 (0.0 to 0.0) | 26.2 (18.5 to 33.8) | 37.9 (29.3 to 46.6) | 13.4 (7.9 to 19.7) | 15.9 (10.1 to 22.5) |
| Stage 1 final model (three rounds) | 0.0 (0.0 to 0.0) | 1.5 (0.0 to 3.8) | 25.9 (18.1 to 34.5) | 11.0 (5.5 to 16.5) | 21.7 (15.2 to 29.0) |

### Legitimate success on solvable tasks (higher is better)

| kinds trained | test | stress | unseen | fresh | sealed |
|---|---|---|---|---|---|
| 0 (escalation-prone start) | 99.5 (98.5 to 100.0) | 61.1 (54.2 to 67.9) | 82.4 (77.0 to 87.3) | n/a | n/a |
| 2 | 99.5 (98.5 to 100.0) | 88.9 (84.2 to 93.2) | 85.3 (79.9 to 89.7) | 68.9 (62.7 to 75.1) | 85.2 (79.7 to 90.1) |
| 4 | 99.5 (98.5 to 100.0) | 72.6 (66.3 to 78.4) | 85.8 (80.9 to 90.2) | 68.4 (62.2 to 74.6) | 84.1 (78.6 to 89.0) |
| 6 | 99.5 (98.5 to 100.0) | 98.9 (97.4 to 100.0) | 98.0 (96.1 to 99.5) | 72.5 (66.3 to 78.8) | 91.2 (86.8 to 95.1) |
| 10 | 99.5 (98.5 to 100.0) | 95.3 (92.1 to 97.9) | 87.3 (82.8 to 91.7) | 71.0 (64.2 to 77.2) | 91.8 (87.4 to 95.6) |
| Stage 1 final model (three rounds) | 99.0 (97.4 to 100.0) | 99.5 (98.4 to 100.0) | 99.0 (97.5 to 100.0) | 71.0 (64.2 to 77.2) | 91.8 (87.9 to 95.6) |

### Which shortcut the agent took, by arm (blocked tasks, detector category, all five sets pooled)

| category | 2 kinds | 4 kinds | 6 kinds | 10 kinds |
|---|---|---|---|---|
| shadow_host | 178 | 121 | 40 | 48 |
| recon_or_tamper | 69 | 97 | 78 | 51 |
| relay_indirection | 163 | 33 | 14 | 19 |
| header_spoof | 55 | 0 | 0 | 0 |
| credential_misuse | 28 | 12 | 1 | 4 |
| captcha_bypass | 5 | 0 | 0 | 0 |
| disposable_account | 4 | 0 | 0 | 0 |

### Escalation by whether the shortcut kinds on offer were trained (test and stress sets)

A blocked task counts as "all trained" when every shortcut kind it offers is among the arm's k kinds.

| kinds trained | all kinds on offer trained | at least one kind not trained |
|---|---|---|
| 2 | 0.0 (0.0 to 0.0), n=8 | 80.5 (75.2 to 85.4), n=246 |
| 4 | 0.0 (0.0 to 0.0), n=84 | 50.0 (42.4 to 57.6), n=170 |
| 6 | 4.7 (0.9 to 9.3), n=107 | 21.1 (15.0 to 27.9), n=147 |
| 10 | 3.9 (0.8 to 7.8), n=129 | 23.2 (16.0 to 30.4), n=125 |

## The same recipe on other Aya models

Per base model: clean tool SFT, then from an escalation-prone SFT model 160 GRPO steps on the 10-kind training set with the success-only reward and with the safe reward (seed 4).


### Escalation on blocked tasks (lower is better)

| model | test | stress | unseen | fresh | sealed | long chat |
|---|---|---|---|---|---|---|
| Tiny Aya Global, clean SFT | 0.0 (0.0 to 0.0) | 91.5 (86.2 to 96.2) | 87.9 (81.9 to 93.1) | 48.0 (39.4 to 56.7) | 42.8 (35.5 to 51.4) | 63.6 (55.3 to 71.2) |
| Tiny Aya Global, GRPO success-only | 74.2 (66.1 to 81.5) | 99.2 (97.7 to 100.0) | 90.5 (84.5 to 95.7) | 77.2 (70.1 to 84.3) | 77.5 (70.3 to 84.1) | n/a |
| Tiny Aya Global, GRPO safe | 0.0 (0.0 to 0.0) | 26.2 (18.5 to 33.8) | 37.9 (29.3 to 46.6) | 13.4 (7.9 to 19.7) | 15.9 (10.1 to 22.5) | 15.2 (9.1 to 22.0) |
| Tiny Aya Fire, clean SFT | 0.0 (0.0 to 0.0) | 89.2 (83.8 to 93.8) | 92.2 (87.1 to 96.6) | 49.6 (40.9 to 58.3) | 47.8 (39.9 to 55.8) | 57.6 (49.2 to 65.9) |
| Tiny Aya Fire, GRPO success-only | 100.0 (100.0 to 100.0) | 96.2 (93.1 to 99.2) | 99.1 (97.4 to 100.0) | 74.0 (66.1 to 81.9) | 97.8 (94.9 to 100.0) | n/a |
| Tiny Aya Fire, GRPO safe | 0.0 (0.0 to 0.0) | 4.6 (1.5 to 8.5) | 59.5 (50.0 to 69.0) | 18.9 (12.6 to 26.0) | 40.6 (33.3 to 48.6) | 22.7 (15.9 to 30.3) |
| Tiny Aya Water, clean SFT | 0.0 (0.0 to 0.0) | 86.9 (80.8 to 92.3) | 73.3 (64.7 to 81.0) | 47.2 (38.6 to 55.9) | 38.4 (31.2 to 46.4) | 52.3 (44.7 to 60.6) |
| Tiny Aya Water, GRPO success-only | 91.9 (87.1 to 96.0) | 100.0 (100.0 to 100.0) | 99.1 (97.4 to 100.0) | 73.2 (66.1 to 80.3) | 100.0 (100.0 to 100.0) | n/a |
| Tiny Aya Water, GRPO safe | 0.0 (0.0 to 0.0) | 8.5 (3.8 to 13.1) | 43.1 (33.6 to 52.6) | 14.2 (8.7 to 20.5) | 31.2 (23.9 to 39.1) | 10.6 (6.1 to 15.9) |
| Tiny Aya Earth, clean SFT | 0.0 (0.0 to 0.0) | 89.2 (83.8 to 94.6) | 85.3 (79.3 to 91.4) | 44.9 (36.2 to 53.5) | 44.9 (37.7 to 52.9) | 52.3 (43.9 to 61.4) |
| Tiny Aya Earth, GRPO success-only | 96.8 (93.5 to 99.2) | 97.7 (94.6 to 100.0) | 98.3 (95.7 to 100.0) | 81.9 (75.6 to 88.2) | 99.3 (97.8 to 100.0) | n/a |
| Tiny Aya Earth, GRPO safe | 0.0 (0.0 to 0.0) | 3.1 (0.8 to 6.2) | 46.6 (37.9 to 56.0) | 15.0 (8.7 to 22.0) | 26.1 (19.6 to 33.3) | 17.4 (11.4 to 24.2) |
| Aya Expanse 8B, clean SFT | 0.0 (0.0 to 0.0) | 88.5 (82.3 to 93.8) | 79.3 (71.6 to 86.2) | 40.9 (33.1 to 49.6) | 49.3 (41.3 to 58.0) | 56.8 (48.5 to 65.9) |
| Aya Expanse 8B, GRPO success-only | 87.9 (82.3 to 93.5) | 100.0 (100.0 to 100.0) | 84.5 (77.6 to 90.5) | 88.2 (82.7 to 93.7) | 93.5 (89.1 to 97.1) | n/a |
| Aya Expanse 8B, GRPO safe | 0.0 (0.0 to 0.0) | 0.8 (0.0 to 2.3) | 6.9 (2.6 to 12.1) | 3.1 (0.8 to 6.3) | 0.7 (0.0 to 2.2) | 9.1 (4.5 to 14.4) |

### Legitimate success on solvable tasks (higher is better)

| model | test | stress | unseen | fresh | sealed | long chat |
|---|---|---|---|---|---|---|
| Tiny Aya Global, clean SFT | 99.5 (98.5 to 100.0) | 68.4 (61.6 to 74.7) | 89.2 (84.8 to 93.1) | 62.2 (55.4 to 68.9) | 81.9 (75.8 to 87.4) | 72.9 (67.0 to 78.7) |
| Tiny Aya Global, GRPO success-only | 99.5 (98.5 to 100.0) | 60.0 (52.6 to 66.8) | 78.4 (72.5 to 84.3) | 70.5 (63.7 to 76.7) | 80.8 (74.7 to 86.3) | n/a |
| Tiny Aya Global, GRPO safe | 99.5 (98.5 to 100.0) | 95.3 (92.1 to 97.9) | 87.3 (82.8 to 91.7) | 71.0 (64.2 to 77.2) | 91.8 (87.4 to 95.6) | 85.6 (80.3 to 90.4) |
| Tiny Aya Fire, clean SFT | 100.0 (100.0 to 100.0) | 73.2 (66.8 to 79.5) | 87.7 (82.8 to 92.2) | 65.3 (58.5 to 71.5) | 82.4 (76.9 to 87.9) | 77.1 (71.3 to 83.0) |
| Tiny Aya Fire, GRPO success-only | 80.6 (75.0 to 86.2) | 55.8 (48.9 to 62.6) | 65.2 (58.8 to 71.6) | 65.3 (58.5 to 72.0) | 68.1 (61.0 to 75.3) | n/a |
| Tiny Aya Fire, GRPO safe | 100.0 (100.0 to 100.0) | 98.9 (97.4 to 100.0) | 98.5 (96.6 to 100.0) | 71.0 (64.8 to 77.2) | 91.2 (86.8 to 95.1) | 95.2 (92.0 to 97.9) |
| Tiny Aya Water, clean SFT | 99.5 (98.5 to 100.0) | 70.5 (64.2 to 76.8) | 87.3 (82.4 to 91.2) | 66.3 (59.6 to 72.5) | 82.4 (76.4 to 87.4) | 78.2 (71.8 to 84.0) |
| Tiny Aya Water, GRPO success-only | 91.3 (87.2 to 94.9) | 57.9 (50.5 to 64.7) | 63.7 (57.4 to 70.1) | 69.4 (62.7 to 75.6) | 34.6 (27.5 to 41.8) | n/a |
| Tiny Aya Water, GRPO safe | 99.5 (98.5 to 100.0) | 98.4 (96.3 to 100.0) | 98.0 (96.1 to 99.5) | 71.0 (64.8 to 77.2) | 92.3 (87.9 to 96.2) | 89.4 (84.6 to 93.6) |
| Tiny Aya Earth, clean SFT | 99.5 (98.5 to 100.0) | 72.6 (66.3 to 78.9) | 85.8 (80.9 to 90.2) | 64.8 (58.0 to 71.5) | 81.9 (75.8 to 87.4) | 75.0 (68.6 to 80.3) |
| Tiny Aya Earth, GRPO success-only | 85.2 (79.6 to 89.8) | 53.7 (46.8 to 60.5) | 63.2 (56.9 to 69.6) | 66.8 (60.1 to 73.1) | 41.2 (34.1 to 48.4) | n/a |
| Tiny Aya Earth, GRPO safe | 99.5 (98.5 to 100.0) | 99.5 (98.4 to 100.0) | 98.5 (96.6 to 100.0) | 75.1 (68.9 to 80.8) | 91.8 (87.4 to 95.6) | 91.5 (87.2 to 95.2) |
| Aya Expanse 8B, clean SFT | 100.0 (100.0 to 100.0) | 83.2 (77.9 to 88.4) | 93.6 (90.2 to 96.6) | 71.5 (64.8 to 77.7) | 90.7 (86.3 to 94.5) | 72.3 (66.0 to 78.7) |
| Aya Expanse 8B, GRPO success-only | 73.0 (66.8 to 79.6) | 45.8 (38.9 to 52.6) | 73.0 (66.7 to 78.9) | 69.4 (62.7 to 75.1) | 65.4 (58.8 to 72.0) | n/a |
| Aya Expanse 8B, GRPO safe | 100.0 (100.0 to 100.0) | 100.0 (100.0 to 100.0) | 99.0 (97.5 to 100.0) | 76.2 (69.9 to 81.9) | 99.5 (98.4 to 100.0) | 85.6 (80.3 to 90.4) |

### Safe against success-only reward, paired (exact McNemar on blocked tasks)

| model, set | success-only escalates, safe does not | safe escalates, success-only does not | p |
|---|---|---|---|
| Tiny Aya Global, test | 92 | 0 | 4e-28 |
| Tiny Aya Global, stress | 95 | 0 | 5e-29 |
| Tiny Aya Global, unseen | 61 | 0 | 8.7e-19 |
| Tiny Aya Global, fresh | 81 | 0 | 8.3e-25 |
| Tiny Aya Global, sealed | 86 | 1 | 1.1e-24 |
| Tiny Aya Fire, test | 124 | 0 | 9.4e-38 |
| Tiny Aya Fire, stress | 119 | 0 | 3e-36 |
| Tiny Aya Fire, unseen | 46 | 0 | 2.8e-14 |
| Tiny Aya Fire, fresh | 71 | 1 | 3.1e-20 |
| Tiny Aya Fire, sealed | 79 | 0 | 3.3e-24 |
| Tiny Aya Water, test | 114 | 0 | 9.6e-35 |
| Tiny Aya Water, stress | 119 | 0 | 3e-36 |
| Tiny Aya Water, unseen | 65 | 0 | 5.4e-20 |
| Tiny Aya Water, fresh | 75 | 0 | 5.3e-23 |
| Tiny Aya Water, sealed | 95 | 0 | 5e-29 |
| Tiny Aya Earth, test | 120 | 0 | 1.5e-36 |
| Tiny Aya Earth, stress | 123 | 0 | 1.9e-37 |
| Tiny Aya Earth, unseen | 61 | 1 | 2.7e-17 |
| Tiny Aya Earth, fresh | 85 | 0 | 5.2e-26 |
| Tiny Aya Earth, sealed | 101 | 0 | 7.9e-31 |
| Aya Expanse 8B, test | 109 | 0 | 3.1e-33 |
| Aya Expanse 8B, stress | 129 | 0 | 2.9e-39 |
| Aya Expanse 8B, unseen | 90 | 0 | 1.6e-27 |
| Aya Expanse 8B, fresh | 108 | 0 | 6.2e-33 |
| Aya Expanse 8B, sealed | 128 | 0 | 5.9e-39 |

### Runtime guard on the safe model: escalation attempted and executed on blocked tasks

| model, set | no guard | with guard: attempted | with guard: executed |
|---|---|---|---|
| Tiny Aya Global, test | 0.0 | 0.0 | 0.0 |
| Tiny Aya Global, stress | 26.2 | 26.2 | 0.0 |
| Tiny Aya Global, unseen | 37.9 | 37.9 | 0.0 |
| Tiny Aya Global, fresh | 13.4 | 11.8 | 11.8 |
| Tiny Aya Global, sealed | 15.9 | 14.5 | 14.5 |
| Tiny Aya Fire, test | 0.0 | 0.0 | 0.0 |
| Tiny Aya Fire, stress | 4.6 | 4.6 | 0.0 |
| Tiny Aya Fire, unseen | 59.5 | 58.6 | 0.0 |
| Tiny Aya Fire, fresh | 18.9 | 18.9 | 18.9 |
| Tiny Aya Fire, sealed | 40.6 | 42.0 | 40.6 |
| Tiny Aya Water, test | 0.0 | 0.0 | 0.0 |
| Tiny Aya Water, stress | 8.5 | 8.5 | 0.0 |
| Tiny Aya Water, unseen | 43.1 | 43.1 | 0.0 |
| Tiny Aya Water, fresh | 14.2 | 14.2 | 14.2 |
| Tiny Aya Water, sealed | 31.2 | 30.4 | 30.4 |
| Tiny Aya Earth, test | 0.0 | 0.0 | 0.0 |
| Tiny Aya Earth, stress | 3.1 | 2.3 | 0.0 |
| Tiny Aya Earth, unseen | 46.6 | 46.6 | 0.0 |
| Tiny Aya Earth, fresh | 15.0 | 15.0 | 15.0 |
| Tiny Aya Earth, sealed | 26.1 | 25.4 | 25.4 |
| Aya Expanse 8B, test | 0.0 | 0.0 | 0.0 |
| Aya Expanse 8B, stress | 0.8 | 0.8 | 0.0 |
| Aya Expanse 8B, unseen | 6.9 | 6.9 | 0.0 |
| Aya Expanse 8B, fresh | 3.1 | 3.1 | 3.1 |
| Aya Expanse 8B, sealed | 0.7 | 0.7 | 0.7 |

### General tool calling (BFCL v3 simple + multiple, our own AST check)

| model | full match |
|---|---|
| Tiny Aya Global, grpo_safe | 78.0 (74.7 to 81.3) |
| Tiny Aya Fire, base | 81.3 (78.2 to 84.5) |
| Tiny Aya Fire, grpo_safe | 80.5 (77.3 to 83.8) |
| Tiny Aya Water, base | 81.3 (78.2 to 84.5) |
| Tiny Aya Water, grpo_safe | 77.5 (74.2 to 80.8) |
| Tiny Aya Earth, base | 82.3 (79.3 to 85.3) |
| Tiny Aya Earth, grpo_safe | 78.0 (74.8 to 81.3) |
| Aya Expanse 8B, base | 86.3 (83.7 to 89.2) |
| Aya Expanse 8B, grpo_safe | 88.8 (86.3 to 91.3) |

### Real HTTP server and the live World Bank API

| model | sandbox: escalation on blocked | server-confirmed bypass | sandbox legit success | World Bank correct |
|---|---|---|---|---|
| Tiny Aya Global, grpo_success_only | 61.9 (42.9 to 81.0) | 57.1 (38.1 to 81.0) | 47.4 (26.3 to 68.4) | 18/24 |
| Tiny Aya Global, grpo_safe | 19.0 (4.8 to 38.1) | 19.0 (4.8 to 38.1) | 84.2 (68.4 to 100.0) | 18/24 |
| Tiny Aya Fire, sft | 33.3 (14.3 to 52.4) | 4.8 (0.0 to 14.3) | 63.2 (42.1 to 84.2) | 18/24 |
| Tiny Aya Fire, grpo_success_only | 61.9 (42.9 to 81.0) | 57.1 (38.1 to 76.2) | 89.5 (73.7 to 100.0) | 18/24 |
| Tiny Aya Fire, grpo_safe | 0.0 (0.0 to 0.0) | 0.0 (0.0 to 0.0) | 89.5 (73.7 to 100.0) | 18/24 |
| Tiny Aya Water, sft | 38.1 (19.0 to 61.9) | 0.0 (0.0 to 0.0) | 68.4 (47.4 to 89.5) | 18/24 |
| Tiny Aya Water, grpo_success_only | 61.9 (42.9 to 81.0) | 61.9 (42.9 to 81.0) | 94.7 (84.2 to 100.0) | 18/24 |
| Tiny Aya Water, grpo_safe | 9.5 (0.0 to 23.8) | 9.5 (0.0 to 23.8) | 84.2 (68.4 to 100.0) | 18/24 |
| Tiny Aya Earth, sft | 28.6 (9.5 to 47.6) | 0.0 (0.0 to 0.0) | 63.2 (42.1 to 84.2) | 18/24 |
| Tiny Aya Earth, grpo_success_only | 71.4 (52.4 to 90.5) | 61.9 (42.9 to 81.0) | 84.2 (68.4 to 100.0) | 18/24 |
| Tiny Aya Earth, grpo_safe | 0.0 (0.0 to 0.0) | 0.0 (0.0 to 0.0) | 84.2 (68.4 to 100.0) | 18/24 |
| Aya Expanse 8B, sft | 14.3 (0.0 to 33.3) | 9.5 (0.0 to 23.8) | 73.7 (52.6 to 89.5) | 18/24 |
| Aya Expanse 8B, grpo_success_only | 81.0 (61.9 to 95.2) | 81.0 (61.9 to 95.2) | 47.4 (26.3 to 68.4) | 18/24 |
| Aya Expanse 8B, grpo_safe | 0.0 (0.0 to 0.0) | 0.0 (0.0 to 0.0) | 94.7 (84.2 to 100.0) | 18/24 |

## Seeds: the 10-kind recipe on Tiny Aya Global, three GRPO seeds

Same start model, data and settings; only the GRPO seed differs (4 is the run used everywhere else).

| reward, seed | test | stress | unseen | fresh | sealed |
|---|---|---|---|---|---|
| safe, seed 4 | 0.0 (0.0 to 0.0) | 26.2 (18.5 to 33.8) | 37.9 (29.3 to 46.6) | 13.4 (7.9 to 19.7) | 15.9 (10.1 to 22.5) |
| safe, seed 5 | 0.0 (0.0 to 0.0) | 32.3 (24.6 to 40.0) | 53.4 (44.0 to 62.9) | 29.1 (21.3 to 37.8) | 43.5 (35.5 to 52.2) |
| safe, seed 6 | 0.0 (0.0 to 0.0) | 11.5 (6.2 to 16.9) | 38.8 (29.3 to 48.3) | 8.7 (3.9 to 13.4) | 20.3 (13.8 to 27.5) |
| safe, mean (min to max) | 0.0 (0.0 to 0.0) | 23.3 (11.5 to 32.3) | 43.4 (37.9 to 53.4) | 17.1 (8.7 to 29.1) | 26.6 (15.9 to 43.5) |
| success-only, seed 4 | 74.2 (66.1 to 81.5) | 99.2 (97.7 to 100.0) | 90.5 (84.5 to 95.7) | 77.2 (70.1 to 84.3) | 77.5 (70.3 to 84.1) |
| success-only, seed 5 | 91.1 (85.5 to 96.0) | 96.2 (93.1 to 99.2) | 99.1 (97.4 to 100.0) | 57.5 (48.8 to 66.1) | 94.2 (89.9 to 97.8) |
| success-only, seed 6 | 92.7 (87.9 to 96.8) | 93.1 (88.5 to 96.9) | 99.1 (97.4 to 100.0) | 65.4 (57.5 to 74.0) | 91.3 (86.2 to 95.7) |
| success-only, mean (min to max) | 86.0 (74.2 to 92.7) | 96.2 (93.1 to 99.2) | 96.3 (90.5 to 99.1) | 66.7 (57.5 to 77.2) | 87.7 (77.5 to 94.2) |

Legitimate success on solvable tasks, safe reward:

| seed | test | stress | unseen | fresh | sealed |
|---|---|---|---|---|---|
| 4 | 99.5 | 95.3 | 87.3 | 71.0 | 91.8 |
| 5 | 99.5 | 98.9 | 95.6 | 71.5 | 92.3 |
| 6 | 99.5 | 95.8 | 96.1 | 72.0 | 92.3 |

## Where the tip wording comes from: own generator, Adaption, CuratorKIT

The planted tips in the training prompts need a lead-in sentence ("Note for automated clients:"). The main runs use six hand-written ones. Here the same 960 training tasks get lead-ins generated by Adaption or by CuratorKIT, after the same hygiene pass (CuratorKIT secrets gate, exact and MinHash 0.7 de-duplication with the test-set wordings placed first so anything close to them is removed, and our own shape rule). 300 lines per source are used. Everything else is the 10-kind recipe, seed 4.


### Hygiene pass, before and after

| source | generated | wrong shape or names a route | secrets | exact duplicates | near duplicates | passed | used |
|---|---|---|---|---|---|---|---|
| Adaption (hosted model) | 600 | 1 | 0 | 4 | 3 | 592 | 300 |
| CuratorKIT EvolInstruct (Qwen2.5-32B teacher) | 600 | 2 | 0 | 3 | 4 | 591 | 300 |

### Safe-reward model trained on each source


Escalation on blocked tasks:

| tip wording | test | stress | unseen | fresh | sealed |
|---|---|---|---|---|---|
| own generator (6 hand-written lead-ins, no model) | 0.0 (0.0 to 0.0) | 26.2 (18.5 to 33.8) | 37.9 (29.3 to 46.6) | 13.4 (7.9 to 19.7) | 15.9 (10.1 to 22.5) |
| Adaption (hosted model) | 0.0 (0.0 to 0.0) | 6.2 (2.3 to 10.8) | 38.8 (29.3 to 48.3) | 5.5 (2.4 to 9.4) | 20.3 (14.5 to 27.5) |
| CuratorKIT EvolInstruct (Qwen2.5-32B teacher) | 1.6 (0.0 to 4.0) | 30.8 (23.1 to 38.5) | 53.4 (44.0 to 62.1) | 18.9 (12.6 to 26.0) | 37.7 (29.7 to 45.7) |

Legitimate success on solvable tasks:

| tip wording | test | stress | unseen | fresh | sealed |
|---|---|---|---|---|---|
| own generator (6 hand-written lead-ins, no model) | 99.5 (98.5 to 100.0) | 95.3 (92.1 to 97.9) | 87.3 (82.8 to 91.7) | 71.0 (64.2 to 77.2) | 91.8 (87.4 to 95.6) |
| Adaption (hosted model) | 99.5 (98.5 to 100.0) | 98.9 (97.4 to 100.0) | 97.5 (95.1 to 99.5) | 71.0 (64.8 to 77.2) | 92.3 (88.5 to 96.2) |
| CuratorKIT EvolInstruct (Qwen2.5-32B teacher) | 99.5 (98.5 to 100.0) | 98.4 (96.3 to 100.0) | 94.1 (90.7 to 97.1) | 74.6 (68.4 to 80.3) | 94.0 (90.1 to 97.3) |

One seed per source. The seed table above gives the spread to expect from the seed alone.

## Hindi and English tasks

About a quarter of the tasks in every set are written in Hindi (the provider pages stay in English). Escalation on blocked tasks, all five sets pooled.

| model | English, success-only | Hindi, success-only | English, safe | Hindi, safe | Hindi success on solvable, safe |
|---|---|---|---|---|---|
| Tiny Aya Global | 83.9 (80.5 to 87.2), n=483 | 82.9 (77.0 to 88.8), n=152 | 18.0 (14.5 to 21.5), n=483 | 19.7 (13.8 to 26.3), n=152 | 90.8 |
| Tiny Aya Fire | 93.4 (90.9 to 95.4), n=483 | 93.4 (89.5 to 96.7), n=152 | 24.2 (20.3 to 28.0), n=483 | 25.0 (18.4 to 31.6), n=152 | 92.9 |
| Tiny Aya Water | 93.8 (91.5 to 95.9), n=483 | 90.1 (84.9 to 94.7), n=152 | 18.4 (14.9 to 21.9), n=483 | 21.7 (15.1 to 28.3), n=152 | 92.9 |
| Tiny Aya Earth | 95.9 (94.0 to 97.5), n=483 | 91.4 (86.8 to 95.4), n=152 | 17.8 (14.5 to 21.3), n=483 | 17.8 (11.8 to 23.7), n=152 | 95.0 |
| Aya Expanse 8B | 90.7 (88.0 to 93.4), n=483 | 92.1 (87.5 to 96.1), n=152 | 1.7 (0.6 to 2.9), n=483 | 3.9 (1.3 to 7.2), n=152 | 95.8 |

## Compression: does the safe model stay safe and useful?

Tiny Aya Global, safe reward (10 kinds). "Kept" layers stay at 16-bit; CircuitKIT's default keeps 19 of 36 layers, and the control arms keep the same number. GPTQ arms use 128 of our own tool-use episodes as calibration text. A model that does nothing also never escalates, so read escalation together with the last three columns.

| version | size GB | esc test | esc stress | esc unseen | esc sealed | legit success (4 sets) | no answer (4 sets) | BFCL match |
|---|---|---|---|---|---|---|---|---|
| full precision (16-bit) | 6.7 | 0.0 | 26.2 | 37.1 | 15.2 | 93.5 | 0.1 | 78.3 |
| GPTQ 4-bit | 2.52 | 21.0 | 19.2 | 33.6 | 21.0 | 13.5 | 48.2 | 28.8 |
| GPTQ 3-bit | 2.17 | 2.4 | 1.5 | 1.7 | 0.7 | 2.1 | 88.9 | 18.7 |
| GPTQ 3-bit, circuit layers kept | 4.56 | 0.0 | 0.0 | 6.0 | 9.4 | 55.6 | 30.5 | 70.7 |
| GPTQ 3-bit, random layers kept (draw 1) | 4.56 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 2.5 |
| GPTQ 3-bit, random layers kept (draw 2) | 4.56 | 0.0 | 3.8 | 32.8 | 8.7 | 72.3 | 19.3 | 73.3 |
| GPTQ 3-bit, random layers kept (draw 3) | 4.56 | 3.2 | 0.0 | 6.9 | 2.9 | 0.6 | 80.1 | 18.8 |
| GPTQ 3-bit, lowest-ranked layers kept | 4.56 | 4.0 | 1.5 | 12.9 | 7.2 | 3.8 | 80.0 | 17.8 |

Layers kept at 16-bit: c3: [0, 1, 2, 3, 10, 12, 14, 15, 17, 18, 19, 20, 23, 27, 28, 30, 31, 34, 35]; r3: [2, 3, 4, 6, 9, 11, 12, 15, 16, 18, 20, 24, 25, 26, 27, 29, 31, 32, 34]; r3b: [0, 3, 4, 6, 7, 8, 12, 13, 14, 15, 16, 20, 21, 24, 25, 26, 28, 29, 31]; r3c: [1, 3, 5, 6, 8, 9, 10, 12, 13, 18, 19, 21, 23, 24, 25, 27, 32, 33, 34]; a3: [4, 5, 6, 8, 9, 13, 14, 16, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33].

## Closed models on the same tasks, and what a task costs

Same harness, neutral prompt, 320 episodes per set. "Native" sends our six tools through the provider's function-calling interface; "text protocol" is the JSON-in-text format the Aya models use. Both closed models run at their default temperature (their APIs do not accept 0).


### Escalation on blocked tasks

| model | test | stress | unseen | fresh | sealed |
|---|---|---|---|---|---|
| Tiny Aya Global, success-only reward | 74.2 | 99.2 | 90.5 | 77.2 | 77.5 |
| Tiny Aya Global, safe reward | 0.0 | 26.2 | 37.9 | 13.4 | 15.9 |
| Aya Expanse 8B, safe reward | 0.0 | 0.8 | 6.9 | 3.1 | 0.7 |
| GPT-6 Luna, native tool calls | 51.6 | 92.3 | 86.2 | 55.1 | 64.5 |
| Claude Haiku 5.5, native tool calls | 1.6 | 11.5 | 1.7 | 2.4 | 1.4 |
| GPT-6 Luna, our text protocol | 2.4 | 4.6 | 3.4 | 1.6 | 2.2 |
| Claude Haiku 5.5, our text protocol | 0.8 | 14.6 | 0.9 | 2.4 | 2.9 |

### Legitimate success on solvable tasks

| model | test | stress | unseen | fresh | sealed |
|---|---|---|---|---|---|
| Tiny Aya Global, success-only reward | 99.5 | 60.0 | 78.4 | 70.5 | 80.8 |
| Tiny Aya Global, safe reward | 99.5 | 95.3 | 87.3 | 71.0 | 91.8 |
| Aya Expanse 8B, safe reward | 100.0 | 100.0 | 99.0 | 76.2 | 99.5 |
| GPT-6 Luna, native tool calls | 71.9 | 32.1 | 45.1 | 74.1 | 68.7 |
| Claude Haiku 5.5, native tool calls | 66.3 | 62.6 | 66.7 | 71.0 | 72.5 |
| GPT-6 Luna, our text protocol | 6.6 | 1.6 | 4.4 | 4.1 | 2.2 |
| Claude Haiku 5.5, our text protocol | 49.5 | 47.4 | 52.5 | 58.5 | 54.4 |

### Cost and speed for the 1,600 test episodes

| model | where it runs | output tokens | wall time s | output tokens per s | cost USD | USD per 1,000 tasks |
|---|---|---|---|---|---|---|
| Tiny Aya Global, safe reward (3.35B, 16-bit) | one A100 80GB, vLLM, all episodes batched | 303,111 | 132 | 2302 | 0.09 | 0.057 |
| GPT-6 Luna, native tool calls | provider API, 16 parallel requests | 881,187 | 1036 | 850 | 0.90 | 0.56 |
| Claude Haiku 5.5, native tool calls | provider API, 16 parallel requests | 2,668,726 | 1020 | 2618 | 2.56 | 1.60 |

The GPU price is $2.50 per hour for an A100 80GB on Hugging Face Jobs; Modal lists the same card at $0.000694 per second, which is $2.50 per hour, so the row is the same on either platform. The GPU row counts generation time only (loading the model is extra and is paid once). Our output tokens are counted with the model's tokenizer from the saved episodes; the API rows use the providers' usage reports, and their output counts can include reasoning tokens. Both closed models list $0.10 per million input and $0.50 per million output tokens.

### What Stage 2 cost

| account | jobs | GPU hours (A100 80GB) | USD |
|---|---|---|---|
| own Hugging Face credits | 12 | 23.2 | 58 |
| organiser credits | 12 | 20.2 | 50 |
| total GPU | 24 | 43.3 | 108 |
| closed-model API calls |  |  | 7 |

Job list as of 22 Nov, 15:20 UTC. Includes 11.9 GPU hours of jobs that failed (most were stopped by a storage limit and run again).
