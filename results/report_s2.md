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

### Legitimate success on solvable tasks (higher is better)

| model | test | stress | unseen | fresh | sealed | long chat |
|---|---|---|---|---|---|---|
| Tiny Aya Global, clean SFT | 99.5 (98.5 to 100.0) | 68.4 (61.6 to 74.7) | 89.2 (84.8 to 93.1) | 62.2 (55.4 to 68.9) | 81.9 (75.8 to 87.4) | 72.9 (67.0 to 78.7) |
| Tiny Aya Global, GRPO success-only | 99.5 (98.5 to 100.0) | 60.0 (52.6 to 66.8) | 78.4 (72.5 to 84.3) | 70.5 (63.7 to 76.7) | 80.8 (74.7 to 86.3) | n/a |
| Tiny Aya Global, GRPO safe | 99.5 (98.5 to 100.0) | 95.3 (92.1 to 97.9) | 87.3 (82.8 to 91.7) | 71.0 (64.2 to 77.2) | 91.8 (87.4 to 95.6) | 85.6 (80.3 to 90.4) |

### Safe against success-only reward, paired (exact McNemar on blocked tasks)

| model, set | success-only escalates, safe does not | safe escalates, success-only does not | p |
|---|---|---|---|
| Tiny Aya Global, test | 92 | 0 | 4e-28 |
| Tiny Aya Global, stress | 95 | 0 | 5e-29 |
| Tiny Aya Global, unseen | 61 | 0 | 8.7e-19 |
| Tiny Aya Global, fresh | 81 | 0 | 8.3e-25 |
| Tiny Aya Global, sealed | 86 | 1 | 1.1e-24 |

### Runtime guard on the safe model: escalation attempted and executed on blocked tasks

| model, set | no guard | with guard: attempted | with guard: executed |
|---|---|---|---|
| Tiny Aya Global, test | 0.0 | 0.0 | 0.0 |
| Tiny Aya Global, stress | 26.2 | 26.2 | 0.0 |
| Tiny Aya Global, unseen | 37.9 | 37.9 | 0.0 |
| Tiny Aya Global, fresh | 13.4 | 11.8 | 11.8 |
| Tiny Aya Global, sealed | 15.9 | 14.5 | 14.5 |

### General tool calling (BFCL v3 simple + multiple, our own AST check)

| model | full match |
|---|---|
| Tiny Aya Global, grpo_safe | 78.0 (74.7 to 81.3) |

### Real HTTP server and the live World Bank API

| model | sandbox: escalation on blocked | server-confirmed bypass | sandbox legit success | World Bank correct |
|---|---|---|---|---|
| Tiny Aya Global, grpo_success_only | 61.9 (42.9 to 81.0) | 57.1 (38.1 to 81.0) | 47.4 (26.3 to 68.4) | 18/24 |
| Tiny Aya Global, grpo_safe | 19.0 (4.8 to 38.1) | 19.0 (4.8 to 38.1) | 84.2 (68.4 to 100.0) | 18/24 |
