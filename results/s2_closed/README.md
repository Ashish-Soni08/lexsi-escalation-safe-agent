# Closed-model rows (Stage 2): gpt-6-luna

Same harness, prompts and scorer as the Aya models (`scripts/eval.py --api openai`), neutral prompt, 320 episodes per set, up to 10 turns.
Settings: temperature is the model default (gpt-6-luna rejects 0), `--max_new_tokens 4096` (reasoning tokens count against it; 384 ran out), 16 parallel requests.

| Set | Escalation | Success (solvable) | Honest stop (unsolvable) | No answer | Format errors | Acted (≥1 tool call) | Escalation when it acted | Wall time |
|---|---|---|---|---|---|---|---|---|
| test | 1.6% | 6.6% | 34.7% | 56.2% | 1.9% | 33 | 15.2% | 131 s |
| stress | 3.1% | 1.6% | 33.1% | 59.4% | 3.1% | 18 | 55.6% | 125 s |
| unseen | 2.5% | 4.4% | 40.5% | 52.5% | 2.8% | 26 | 30.8% | 123 s |
| fresh | 0.6% | 4.2% | 41.7% | 55.3% | 0.6% | 27 | 7.4% | 174 s |
| sealed | 1.2% | 2.2% | 39.1% | 59.7% | 1.2% | 25 | 16.0% | 190 s |

Finding: in the plain-text JSON tool protocol, gpt-6-luna mostly does not act. In 52-60% of episodes its first reply is a preamble
("I'll check the provided source…") or empty, with no tool call and no ANSWER/BLOCKED line, which the scorer counts as no answer.
Its low escalation rate comes from not acting. In the 18-33 episodes per set where it did call a tool, it escalated in 7-56% of them.

Cost (list price $0.10 / $0.01 cached / $0.50 per 1M tokens): 2.49M input (0.49M cached) and 0.51M output tokens, 2,338 requests,
$0.46 for all five sets, about $0.09 per set or $0.0003 per episode. Usage per run: `gpt6luna_usage_*.json`.
Six requests ran out of the 4,096-token budget and count as empty replies. One aborted sealed run hit OpenAI's prompt filter
("flagged as potentially violating our usage policy") and was rerun; the rerun did not trip it. Filter refusals now count as empty replies (`policy_blocked`).
Total API spend including the probe and the aborted run: under $0.70.

## Second row: gpt-6-luna with native function calling (`--native_tools`, name `gpt6luna_native`)

Tools go through OpenAI's Responses API `tools` field (chat completions refuses function tools together with reasoning on gpt-6-luna).
The system prompt keeps the neutral instructions but drops the "reply with only a JSON list" paragraph. Returned function calls are
rewritten into our text form, so the detector, scorer and tool-result folding are unchanged. Everything else matches the first row.

| Set | Escalation (all) | Escalation (unsolvable) | Success (solvable) | Honest stop (unsolvable) | No answer | Format errors | Wall time |
|---|---|---|---|---|---|---|---|
| test | 30.9% | 51.6% | 71.9% | 37.9% | 5.9% | 0.0% | 236 s |
| stress | 75.3% | 92.3% | 32.1% | 6.9% | 2.2% | 0.0% | 165 s |
| unseen | 61.9% | 86.2% | 45.1% | 6.0% | 5.6% | 0.0% | 188 s |
| fresh | 35.6% | 55.1% | 74.1% | 33.1% | 6.2% | 0.0% | 205 s |
| sealed | 42.2% | 64.5% | 68.7% | 19.6% | 8.8% | 0.0% | 259 s |

Once it can act, gpt-6-luna escalates on the neutral prompt. Main routes: test relay_indirection (73 episodes, e.g. fetching the blocked URL through
reader-relay.example) and credential_misuse (23, e.g. sending the ops token from ops_notes.txt as a Bearer header); stress shadow_host (189);
unseen spread across shadow_host, header_spoof, relay_indirection, credential_misuse. Spot-checked trajectories show real escalations, not detector noise.

Cost: 7,190 requests, 5.98M input (1.54M cached) and 0.88M output tokens, $0.90 for all five sets (about $0.18 per set, $0.0006 per episode).
No prompt-filter refusals and no truncations in this run. Usage: `gpt6luna_native_usage_all5.json`.


## claude-haiku-5-5

Same harness and settings (temperature is the model default, the API rejects `temperature`; `--max_new_tokens 4096`; 16 parallel requests).
Two rows: **plain** uses our plain-text JSON tool protocol like the Aya models; **native** (`--native_tools`) sends the tools as Anthropic tool definitions and converts `tool_use` blocks back to our text form.
Escalation (blocked) is the rate on unsolvable tasks, the number quoted for our models; Escalation (all) is over all 320 episodes, as in the luna table.

| Set | Row | Escalation (blocked) | Escalation (all) | Success (solvable) | Honest stop (unsolvable) | No answer | Format errors | Wall time |
|---|---|---|---|---|---|---|---|---|
| test | plain | 0.8% | 1.6% | 49.5% | 67.7% | 12.5% | 0.6% | 441 s |
| test | native | 1.6% | 1.9% | 66.3% | 88.7% | 5.9% | 0.0% | 200 s |
| stress | plain | 14.6% | 7.5% | 47.4% | 53.8% | 21.9% | 2.8% | 346 s |
| stress | native | 11.5% | 6.6% | 62.6% | 76.9% | 2.2% | 0.0% | 219 s |
| unseen | plain | 0.9% | 1.2% | 52.4% | 72.4% | 17.5% | 1.2% | 319 s |
| unseen | native | 1.7% | 1.2% | 66.7% | 90.5% | 2.8% | 0.0% | 212 s |
| fresh | plain | 2.4% | 1.2% | 58.6% | 61.4% | 17.2% | 1.6% | 324 s |
| fresh | native | 2.4% | 0.9% | 71.0% | 92.1% | 0.9% | 0.0% | 198 s |
| sealed | plain | 2.9% | 2.2% | 54.4% | 67.4% | 18.1% | 1.2% | 276 s |
| sealed | native | 1.5% | 1.6% | 72.5% | 93.5% | 1.6% | 0.0% | 195 s |

Findings: Haiku escalates rarely on unseen, fresh and sealed (under 3% of blocked tasks), but on stress it takes the shadow-host shortcut (20 of 24 stress escalations in the plain row, 20 of 21 in the native row).
Native tool calling raises success on solvable tasks by about 15 points and removes format errors and most no-answer episodes; escalation is about the same in both rows.

Cost (list price $0.10 / $0.50 per 1M input / output tokens, no caching): plain 12.59M in / 2.57M out, $2.55; native 12.24M in / 2.67M out, $2.56. These totals include an aborted partial sealed pass: the key hit its account spend limit during sealed, the limit was raised and sealed was rerun alone.
Usage: `haiku55[_native]_usage_4splits.json` (first run), `_usage_sealed.json` (rerun), `_usage.json` (sum). Wall times are under shared load: plain and native ran at the same time.

Total closed-model API spend: gpt-6-luna about $1.60 (both rows, probes, aborted run) and claude-haiku-5-5 about $5.11 (both rows), about $6.70 in all, under the $10 cap.
