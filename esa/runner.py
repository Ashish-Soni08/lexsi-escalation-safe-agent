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


def initial_messages(s: world.Scenario, schemas: list[dict], variant: str = "rules", chat: bool = False) -> list[dict]:
    return [
        {"role": "system", "content": full_system_prompt(schemas, variant)},
        {"role": "user", "content": stage_prompt(s, 0) if chat and s.parts else s.prompt()},
    ]


NEXT_REQUEST = {"en": "Thank you. Next request.", "hi": "धन्यवाद। अगला अनुरोध।"}
STAGE_TURNS = 10


def stage_prompt(s: world.Scenario, k: int) -> str:
    """Question k of a long task asked one at a time: the ordinary single-task prompt."""
    p = world.Scenario.from_dict({**s.subtasks()[k].to_dict(), "parts": []})
    return p.prompt() if k == 0 else f"{NEXT_REQUEST[s.lang]}\n\n{p.prompt()}"


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
                 call_tool=None, chat: bool = False) -> list[Episode]:
    """chat=True asks the questions of a long task one at a time in one conversation: when the model
    gives its final reply to one, the next arrives as a new user message (max_turns then counts the
    whole conversation and each question gets at most STAGE_TURNS turns)."""
    world.register(scenarios)
    eps = [Episode(scenario_id=s.id, messages=initial_messages(s, schemas, variant, chat)) for s in scenarios]
    live = list(range(len(eps)))
    staged = {i for i, s in enumerate(scenarios) if chat and s.parts}
    stage_turns = {i: 0 for i in staged}
    for i in staged:
        eps[i].stages.append({"from": 0, "final": ""})
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
            if i in staged:
                stage_turns[i] += 1
                if calls and stage_turns[i] > STAGE_TURNS:  # out of turns for this question
                    calls, text = None, ""
                    ep.messages[-1]["content"] = "BLOCKED: out of turns"
            if not calls:
                if "tool_name" in (text or "") or text.strip().startswith("["):
                    ep.format_errors += 1
                ep.final_text = text
                if i in staged:
                    ep.stages[-1]["final"] = text
                    if len(ep.stages) < len(scenarios[i].parts) + 1:
                        ep.stages.append({"from": len(ep.calls), "final": ""})
                        stage_turns[i] = 0
                        ep.messages.append({"role": "user", "content": stage_prompt(scenarios[i], len(ep.stages) - 1)})
                        still.append(i)
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


def api_generator(provider: str, model: str, max_new_tokens: int = 384, temperature: float = 0.0,
                  base_url: str = "", workers: int = 8, usage_path: str = "") -> Generate:
    """Chat-completion endpoint as the policy: `openai` (also any OpenAI-style server, e.g. llama.cpp or vLLM,
    through base_url) or `anthropic`. The tool protocol is the same plain-text JSON the Aya models use, so the
    conversation is sent as ordinary messages. Token counts and wall time are accumulated in gen.usage."""
    import os as _os
    import time
    from concurrent.futures import ThreadPoolExecutor

    import requests  # type: ignore

    usage = {"provider": provider, "model": model, "requests": 0, "input_tokens": 0, "cached_input_tokens": 0,
             "output_tokens": 0, "seconds": 0.0, "errors": 0}
    state = {"temp": True}
    if provider == "anthropic":
        url = (base_url or "https://api.anthropic.com") + "/v1/messages"
        key = _os.environ.get("ANTHROPIC_EVAL_KEY") or _os.environ["ANTHROPIC_API_KEY"]  # Claude Code reserves the usual name
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    else:
        url = (base_url or "https://api.openai.com/v1") + "/chat/completions"
        headers = {"Authorization": "Bearer " + _os.environ.get("OPENAI_API_KEY", "none")}

    def one(conv: list[dict]) -> str:
        if provider == "anthropic":
            body = {"model": model, "max_tokens": max_new_tokens,
                    "system": "\n\n".join(m["content"] for m in conv if m["role"] == "system"),
                    "messages": [m for m in conv if m["role"] != "system"]}
        else:
            body = {"model": model, "messages": conv,
                    ("max_tokens" if base_url else "max_completion_tokens"): max_new_tokens}
        for attempt in range(6):
            if state["temp"]:
                body["temperature"] = temperature
            else:
                body.pop("temperature", None)
            try:
                r = requests.post(url, headers=headers, json=body, timeout=180)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 400 and "temperature" in r.text and "temperature" in body:
                state["temp"] = False  # the model only accepts its default temperature
                continue
            if r.status_code in (408, 409, 429) or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 400 and "max_tokens" in r.text:  # reasoning used the whole budget: no reply
                usage["requests"] += 1
                usage["truncated"] = usage.get("truncated", 0) + 1
                return ""
            if r.status_code != 200:
                raise RuntimeError(f"{provider} {r.status_code}: {r.text[:300]}")
            d = r.json()
            u = d.get("usage", {})
            usage["requests"] += 1
            if provider == "anthropic":
                usage["input_tokens"] += u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                usage["cached_input_tokens"] += u.get("cache_read_input_tokens", 0)
                usage["output_tokens"] += u.get("output_tokens", 0)
                return "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
            usage["input_tokens"] += u.get("prompt_tokens", 0)
            usage["cached_input_tokens"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            usage["output_tokens"] += u.get("completion_tokens", 0)
            return d["choices"][0]["message"].get("content") or ""
        usage["errors"] += 1
        return ""

    def gen(convs: list[list[dict]]) -> list[str]:
        t = time.time()
        with ThreadPoolExecutor(workers) as ex:
            outs = list(ex.map(one, convs))
        usage["seconds"] += time.time() - t
        if usage_path:
            json.dump(usage, open(usage_path, "w"), indent=1)
        return outs

    gen.usage = usage  # type: ignore[attr-defined]
    gen.set_seed = lambda s: None  # type: ignore[attr-defined]
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
