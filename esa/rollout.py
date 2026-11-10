"""Exact-token, batched multi-turn rollout for TRL/AgentTune GRPO.

AgentTune's built-in tool rollout appends raw tool-result tokens to the completion,
while the model actually saw them wrapped in Tiny Aya's chat template. The trainer then
scores tokens in a context the sampler never saw, the vLLM importance-sampling check
masks almost every sequence, and the gradient vanishes (we measured IS ratios ~1e-6).

This rollout keeps one growing token sequence per episode:
    prompt_ids | gen_1 | env_1 | gen_2 | env_2 | ... | gen_k
where every env segment is the exact template text between the assistant turn and the
next generation prompt (end-of-turn, the folded "Tool result for ..." user turn, and the
assistant header). The next turn is generated from ``prompt_ids + completion`` token ids,
so sampler and trainer see identical sequences. env tokens get ``env_mask=0`` (no loss).
All live episodes are generated in one vLLM call per turn.

It is plugged into AgentTune's GRPO trainer through ``rollout_func=``.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from . import world
from .prompts import fallback_block
from .runner import MAX_RESULT_CHARS, fold_results, parse_tool_calls
from .scoring import Episode


def _tok(trainer):
    t = trainer.processing_class
    return getattr(t, "tokenizer", None) or t


def make_rollout_func(schemas: list[dict], max_turns: int = 8, turn_tokens: int = 384,
                      max_completion: int = 3072,
                      on_episode: Optional[Callable[[Episode, list[dict]], None]] = None):
    fb = fallback_block(schemas)

    def rollout_func(prompts: list[Any], trainer: Any) -> dict[str, Any]:
        from trl.experimental.openenv import generate_rollout_completions

        tok = _tok(trainer)

        def render(msgs, gen):
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)

        def ids(text):
            return list(tok(text, add_special_tokens=False)["input_ids"])

        end_ids = ids("<|END_RESPONSE|><|END_OF_TURN_TOKEN|>")
        n = len(prompts)
        msgs, p_ids, comp, lps, mask, eps, rendered = [], [], [], [], [], [], []
        for p in prompts:
            m = [dict(x) for x in p]
            assert m[0]["role"] == "system"
            m[0]["content"] = f"{m[0]['content']}\n\n{fb}"
            msgs.append(m)
            r = render(m, True)
            rendered.append(r)
            p_ids.append(ids(r))
            comp.append([]); lps.append([]); mask.append([])
            eps.append(Episode(scenario_id="", messages=m))
        live = list(range(n))
        for _turn in range(max_turns):
            if not live:
                break
            reqs = [{"prompt_token_ids": p_ids[i] + comp[i]} for i in live]
            budgets = [max_completion - len(comp[i]) for i in live]
            outs = generate_rollout_completions(
                trainer, reqs, generation_overrides={"max_tokens": max(16, min(turn_tokens, min(budgets)))},
                as_chat=False)
            still = []
            for i, o in zip(live, outs):
                g = list(o["completion_ids"])
                lp = [float(x) for x in o["logprobs"]]
                comp[i] += g; lps[i] += lp; mask[i] += [1] * len(g)
                text = tok.decode(g, skip_special_tokens=True)
                ep = eps[i]
                ep.turns += 1
                msgs[i].append({"role": "assistant", "content": text})
                # close the turn exactly as the template renders it
                finished = False
                for k in range(len(end_ids), 0, -1):
                    if g[-k:] == end_ids[:k]:
                        tail = end_ids[k:]
                        comp[i] += tail; lps[i] += [0.0] * len(tail); mask[i] += [0] * len(tail)
                        finished = True
                        break
                if not finished and g and g[-1] == tok.eos_token_id:
                    finished = True
                calls = parse_tool_calls(text)
                if not finished or not calls:
                    if calls is None and ("tool_name" in text or text.strip().startswith("[")):
                        ep.format_errors += 1
                    ep.final_text = text if finished else ""
                    continue
                results = [world.call_tool(nm, a)[:MAX_RESULT_CHARS] for nm, a in calls]
                ep.calls.extend(calls); ep.results.extend(results)
                prev = render(msgs[i], False)
                msgs[i].append({"role": "user", "content": fold_results(calls, results)})
                nxt = render(msgs[i], True)
                if not nxt.startswith(prev):
                    ep.final_text = ""
                    continue
                env = ids(nxt[len(prev):])
                if len(comp[i]) + len(env) + 16 > max_completion:
                    ep.final_text = ""
                    continue
                comp[i] += env; lps[i] += [0.0] * len(env); mask[i] += [0] * len(env)
                still.append(i)
            live = still
        for i in live:
            eps[i].final_text = ""
        episodes = [{"calls": e.calls, "results": e.results, "final_text": e.final_text, "turns": e.turns,
                     "format_errors": e.format_errors, "messages": msgs[i]} for i, e in enumerate(eps)]
        return {"prompt_ids": p_ids, "completion_ids": comp, "logprobs": lps, "env_mask": mask,
                "episodes": episodes}

    return rollout_func
