"""Multi-turn tool-use episodes for any text generator.

The loop mirrors AgentTune's Tiny Aya fallback exactly: tool schemas live in
the system prompt, the model replies with a JSON list of calls, and each batch
of tool results comes back as one user turn starting "Tool result for ...".
Episodes are run turn-synchronously in batches so a vLLM engine can generate
every live episode's next turn in one call.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from . import world
from .detector import classify_call
from .prompts import TOOL_RESULT_FORMAT, full_system_prompt
from .scoring import Episode

MAX_TURNS = 10
MAX_RESULT_CHARS = 1200

Generate = Callable[[list[list[dict]]], list[str]]


def _json_candidates(text: str) -> list[Any]:
    text = re.sub(r"```(?:json)?", "", text)
    out = []
    dec = json.JSONDecoder()
    i = 0
    while i < len(text):
        if text[i] in "[{":
            try:
                obj, end = dec.raw_decode(text, i)
                out.append(obj)
                i = end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return out


def parse_tool_calls(text: str) -> Optional[list[tuple[str, dict]]]:
    """Parse ``[{"tool_name": ..., "parameters": {...}}]`` (and common variants).
    Returns None when the text holds no tool call (a final answer)."""
    calls: list[tuple[str, dict]] = []
    for obj in _json_candidates(text or ""):
        items = obj if isinstance(obj, list) else [obj]
        for it in items:
            if not isinstance(it, dict):
                continue
            fn = it.get("function") if isinstance(it.get("function"), dict) else {}
            name = it.get("tool_name") or it.get("name") or fn.get("name")
            args = it.get("parameters", it.get("arguments", fn.get("arguments", {})))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            if isinstance(name, str) and name:
                calls.append((name, args if isinstance(args, dict) else {}))
    return calls or None


def initial_messages(s: world.Scenario, schemas: list[dict], variant: str = "rules") -> list[dict]:
    return [
        {"role": "system", "content": full_system_prompt(schemas, variant)},
        {"role": "user", "content": s.prompt()},
    ]


GUARD_PREFIX = "Request not sent."


def guard_refusal(name: str, args: dict) -> Optional[str]:
    """Runtime guard: the exploit detector placed in front of the tools. It knows nothing about
    the task, only the shape of the call, which is all a real deployment would have."""
    flags = classify_call(name, args)
    if not flags:
        return None
    return (f"{GUARD_PREFIX} The client-side policy guard refused this call because it looks like an attempt "
            f"to get around an access control ({flags[0].category.replace('_', ' ')}).")


def tool_turn(calls: list[tuple[str, dict]], guard: bool = False, call_tool=None) -> list[str]:
    call_tool = call_tool or world.call_tool  # scripts/real_eval.py passes real tools instead
    out = []
    for n, a in calls:
        refused = guard_refusal(n, a) if guard else None
        out.append(refused or call_tool(n, a)[:MAX_RESULT_CHARS])
    return out


def fold_results(calls: list[tuple[str, dict]], results: list[str]) -> str:
    return "\n\n".join(TOOL_RESULT_FORMAT.format(name=n, content=r) for (n, _), r in zip(calls, results))


def run_episodes(scenarios: list[world.Scenario], generate: Generate, schemas: list[dict],
                 max_turns: int = MAX_TURNS, variant: str = "rules", guard: bool = False,
                 call_tool=None) -> list[Episode]:
    world.register(scenarios)
    eps = [Episode(scenario_id=s.id, messages=initial_messages(s, schemas, variant)) for s in scenarios]
    live = list(range(len(eps)))
    for _turn in range(max_turns):
        if not live:
            break
        outs = generate([eps[i].messages for i in live])
        still = []
        for i, text in zip(live, outs):
            ep = eps[i]
            ep.turns += 1
            ep.messages.append({"role": "assistant", "content": text})
            calls = parse_tool_calls(text)
            if not calls:
                if "tool_name" in (text or "") or text.strip().startswith("["):
                    ep.format_errors += 1
                ep.final_text = text
                continue
            results = tool_turn(calls, guard=guard, call_tool=call_tool)
            ep.calls.extend(calls)
            ep.results.extend(results)
            ep.messages.append({"role": "user", "content": fold_results(calls, results)})
            still.append(i)
        live = still
    for i in live:  # ran out of turns
        eps[i].final_text = ""
    return eps


# --------------------------------------------------------------------------- #
# Generators                                                                   #
# --------------------------------------------------------------------------- #


def vllm_generator(model_path: str, max_new_tokens: int = 384, temperature: float = 0.0,
                   max_model_len: int = 8192, gpu_memory_utilization: float = 0.85, seed: int = 0) -> Generate:
    from vllm import LLM, SamplingParams  # type: ignore

    llm = LLM(model=model_path, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
              seed=seed, dtype="bfloat16")
    tok = llm.get_tokenizer()
    top_p = 1.0 if temperature == 0 else 0.95
    state = {"seed": seed, "calls": 0}

    def gen(convs: list[list[dict]]) -> list[str]:
        prompts = []
        for c in convs:
            text = tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            ids = list(tok(text, add_special_tokens=False)["input_ids"])
            prompts.append({"prompt_token_ids": ids[-(max_model_len - max_new_tokens):]})
        if temperature == 0:
            sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, top_p=top_p)
        else:  # per-request seeds make each sampled run reproducible on its own
            state["calls"] += 1
            sp = [SamplingParams(temperature=temperature, max_tokens=max_new_tokens, top_p=top_p,
                                 seed=(state["seed"] * 1_000_003 + state["calls"] * 1009 + i) % (2**31))
                  for i in range(len(prompts))]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def set_seed(s: int) -> None:
        state["seed"], state["calls"] = s, 0

    gen.set_seed = set_seed  # type: ignore[attr-defined]
    return gen


def hf_generator(model_path: str, adapter: Optional[str] = None, max_new_tokens: int = 384,
                 batch_size: int = 16) -> Generate:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    if adapter:
        from peft import PeftModel  # type: ignore

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def gen(convs: list[list[dict]]) -> list[str]:
        outs: list[str] = []
        for b in range(0, len(convs), batch_size):
            chunk = convs[b:b + batch_size]
            texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            for row in g[:, enc["input_ids"].shape[1]:]:
                outs.append(tok.decode(row, skip_special_tokens=True))
        return outs

    return gen
