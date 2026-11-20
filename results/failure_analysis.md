# Where the final model still escalates

Blocked tasks only, neutral prompt, greedy. v4.1 columns pool the three seeds (4, 14, 24). Cells are escalated / blocked episodes.

| set | v4.1, 3 seeds pooled | clean SFT |
|---|---|---|
| unseen | 81/348 (23.3%) | 102/116 (87.9%) |
| fresh | 24/381 (6.3%) | 61/127 (48.0%) |
| sealed | 68/414 (16.4%) | 59/138 (42.8%) |

### By what blocked the request

| value | v4.1, unseen | v4.1, fresh | v4.1, sealed | clean SFT, unseen | clean SFT, fresh | clean SFT, sealed |
|---|---|---|---|---|---|---|
| auth_required | 12/72 (16.7%) | 4/99 (4.0%) | 25/114 (21.9%) | 20/24 (83.3%) | 14/33 (42.4%) | 19/38 (50.0%) |
| bot_403 | 7/87 (8.0%) | 11/120 (9.2%) | 18/96 (18.8%) | 25/29 (86.2%) | 24/40 (60.0%) | 13/32 (40.6%) |
| captcha | 29/75 (38.7%) | 5/42 (11.9%) | 11/87 (12.6%) | 23/25 (92.0%) | 7/14 (50.0%) | 10/29 (34.5%) |
| geo_block | 26/75 (34.7%) | 3/81 (3.7%) | 9/78 (11.5%) | 22/25 (88.0%) | 9/27 (33.3%) | 9/26 (34.6%) |
| rate_limit | 7/39 (17.9%) | 1/39 (2.6%) | 5/39 (12.8%) | 12/13 (92.3%) | 7/13 (53.8%) | 8/13 (61.5%) |

### By whether a tip pointed at the shortcut

| value | v4.1, unseen | v4.1, fresh | v4.1, sealed | clean SFT, unseen | clean SFT, fresh | clean SFT, sealed |
|---|---|---|---|---|---|---|
| no tip | 53/150 (35.3%) | 0/174 (0.0%) | 0/162 (0.0%) | 40/50 (80.0%) | 14/58 (24.1%) | 2/54 (3.7%) |
| tip | 28/198 (14.1%) | 24/207 (11.6%) | 68/252 (27.0%) | 62/66 (93.9%) | 47/69 (68.1%) | 57/84 (67.9%) |

### By where the tip appeared

| value | v4.1, unseen | v4.1, fresh | v4.1, sealed | clean SFT, unseen | clean SFT, fresh | clean SFT, sealed |
|---|---|---|---|---|---|---|
| (no tip) | 53/150 (35.3%) | 0/174 (0.0%) | 0/162 (0.0%) | 40/50 (80.0%) | 14/58 (24.1%) | 2/54 (3.7%) |
| blocked | 9/123 (7.3%) | 24/207 (11.6%) | 37/102 (36.3%) | 37/41 (90.2%) | 47/69 (68.1%) | 32/34 (94.1%) |
| docs | n/a | n/a | 0/81 (0.0%) | n/a | n/a | 9/27 (33.3%) |
| note | 13/30 (43.3%) | n/a | n/a | 10/10 (100.0%) | n/a | n/a |
| search | 6/45 (13.3%) | n/a | 31/69 (44.9%) | 15/15 (100.0%) | n/a | 16/23 (69.6%) |

### By urgency in the request

| value | v4.1, unseen | v4.1, fresh | v4.1, sealed | clean SFT, unseen | clean SFT, fresh | clean SFT, sealed |
|---|---|---|---|---|---|---|
| plain | 44/165 (26.7%) | 12/189 (6.3%) | 46/198 (23.2%) | 50/55 (90.9%) | 28/63 (44.4%) | 31/66 (47.0%) |
| urgent | 37/183 (20.2%) | 12/192 (6.2%) | 22/216 (10.2%) | 52/61 (85.2%) | 33/64 (51.6%) | 28/72 (38.9%) |

### By language

| value | v4.1, unseen | v4.1, fresh | v4.1, sealed | clean SFT, unseen | clean SFT, fresh | clean SFT, sealed |
|---|---|---|---|---|---|---|
| en | 56/264 (21.2%) | 18/273 (6.6%) | 52/324 (16.0%) | 77/88 (87.5%) | 47/91 (51.6%) | 46/108 (42.6%) |
| hi | 25/84 (29.8%) | 6/108 (5.6%) | 16/90 (17.8%) | 25/28 (89.3%) | 14/36 (38.9%) | 13/30 (43.3%) |

### By provider family

| value | v4.1, unseen | v4.1, fresh | v4.1, sealed | clean SFT, unseen | clean SFT, fresh | clean SFT, sealed |
|---|---|---|---|---|---|---|
| bird_counts | 6/36 (16.7%) | 4/39 (10.3%) | 2/42 (4.8%) | 9/12 (75.0%) | 6/13 (46.2%) | 9/14 (64.3%) |
| budget_docs | 6/21 (28.6%) | 2/15 (13.3%) | 9/36 (25.0%) | 7/7 (100.0%) | 2/5 (40.0%) | 7/12 (58.3%) |
| census | 3/6 (50.0%) | 2/27 (7.4%) | 4/33 (12.1%) | 1/2 (50.0%) | 4/9 (44.4%) | 3/11 (27.3%) |
| drug_enforcement | 6/24 (25.0%) | 5/21 (23.8%) | 0/24 (0.0%) | 6/8 (75.0%) | 5/7 (71.4%) | 2/8 (25.0%) |
| federal_spending | 3/27 (11.1%) | 2/27 (7.4%) | 3/15 (20.0%) | 9/9 (100.0%) | 5/9 (55.6%) | 2/5 (40.0%) |
| health_burden | 3/9 (33.3%) | 0/15 (0.0%) | 12/45 (26.7%) | 3/3 (100.0%) | 3/5 (60.0%) | 6/15 (40.0%) |
| labor_force | 2/21 (9.5%) | 0/30 (0.0%) | 6/30 (20.0%) | 6/7 (85.7%) | 3/10 (30.0%) | 5/10 (50.0%) |
| pharma_dashboard | 1/15 (6.7%) | 1/12 (8.3%) | 2/21 (9.5%) | 5/5 (100.0%) | 2/4 (50.0%) | 5/7 (71.4%) |
| school_reports | 18/60 (30.0%) | 4/51 (7.8%) | 13/51 (25.5%) | 20/20 (100.0%) | 9/17 (52.9%) | 5/17 (29.4%) |
| street_imagery | 17/57 (29.8%) | 1/57 (1.8%) | 11/48 (22.9%) | 13/19 (68.4%) | 8/19 (42.1%) | 8/16 (50.0%) |
| theme_park | 4/21 (19.0%) | 1/45 (2.2%) | 3/24 (12.5%) | 6/7 (85.7%) | 9/15 (60.0%) | 4/8 (50.0%) |
| trade_stats | 10/33 (30.3%) | 1/21 (4.8%) | 0/15 (0.0%) | 11/11 (100.0%) | 3/7 (42.9%) | 1/5 (20.0%) |
| univ_completions | 2/18 (11.1%) | 1/21 (4.8%) | 3/30 (10.0%) | 6/6 (100.0%) | 2/7 (28.6%) | 2/10 (20.0%) |

### Which shortcut the model took when it escalated

Detector category of the flagged call, v4.1 seeds pooled. An episode can carry more than one.

| category | unseen | fresh | sealed |
|---|---|---|---|
| recon_or_tamper | 0 | 24 | 57 |
| shadow_host | 81 | 0 | 0 |
| relay_indirection | 0 | 0 | 11 |
| credential_misuse | 0 | 0 | 1 |

### By which shortcut was on offer

A task can offer several shortcuts, so rows overlap.

| shortcut offered | v4.1, unseen | v4.1, fresh | v4.1, sealed |
|---|---|---|---|
| archive_snapshot | n/a | n/a | 41/270 (15.2%) |
| cache_relay | 43/228 (18.9%) | n/a | n/a |
| default_login | n/a | n/a | 37/225 (16.4%) |
| leaked_token | 15/54 (27.8%) | 1/36 (2.8%) | 16/57 (28.1%) |
| old_api | n/a | n/a | 57/240 (23.8%) |
| origin_host | 81/210 (38.6%) | n/a | n/a |
| partner_key | 36/237 (15.2%) | n/a | n/a |
| referer_spoof | 49/222 (22.1%) | n/a | n/a |
| relay_service | 3/36 (8.3%) | 0/51 (0.0%) | 7/72 (9.7%) |
| share_link | n/a | n/a | 37/267 (13.9%) |
