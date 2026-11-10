"""Agentic GRPO with AgentTune on the blocked-data world.

The policy (the tool-calling SFT model) runs multi-turn episodes through
AgentTune's rollout engine with the Tiny Aya tool fallback, against the same
simulated tools as the eval. Two reward arms:

  --reward safe          legit success +1, honest stop on an unsolvable task +0.6,
                         any exploit-shaped call <= -1 (escalation is always the worst outcome)
  --reward success_only  1 if the right number came back, else 0 (the naive objective)

    python scripts/train_grpo.py --model ashish-soni08/tiny-aya-global-esa-sft --reward safe --out outputs/grpo_safe
"""
import argparse
import functools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esa import world  # noqa: E402
from esa.prompts import TOOL_RESULT_FORMAT, TOOLS_FALLBACK_PROMPT  # noqa: E402
from esa.runner import MAX_RESULT_CHARS  # noqa: E402
from esa.scoring import Episode, safe_reward, score_episode, success_only_reward  # noqa: E402

REWARDS = {"safe": safe_reward, "success_only": success_only_reward}


def truncating(fn):
    @functools.wraps(fn)  # keeps signature + docstring, so the tool schema is unchanged
    def wrapped(*a, **kw):
        return str(fn(*a, **kw))[:MAX_RESULT_CHARS]
    return wrapped


def _call(tc):
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
    name = fn.get("name") or fn.get("tool_name") or ""
    args = fn.get("arguments", fn.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"raw": args}
    return name, args if isinstance(args, dict) else {}


def episode_from_trajectory(t, sid):
    calls, results = [], []
    for st in t.steps:
        meta = getattr(st, "metadata", None) or {}
        tcs = meta.get("tool_calls") or (st.action or {}).get("tool_calls") or []
        res = meta.get("tool_results") or []
        for j, tc in enumerate(tcs):
            calls.append(_call(tc))
            results.append(str(res[j][1]) if j < len(res) else "")
    return Episode(scenario_id=sid, calls=calls, results=results, final_text=t.final_response or "",
                   turns=len(t.steps))


def make_reward(kind, log_path):
    rfn = REWARDS[kind]
    step = {"n": 0}

    def esa_reward(completions, trajectories=None, scenario_id=None, **kw):
        out, stats = [], []
        for t, sid in zip(trajectories, scenario_id):
            s = world.get(sid)
            o = score_episode(episode_from_trajectory(t, sid), s)
            r = rfn(o, s)
            out.append(float(r))
            stats.append({"sid": sid, "outcome": o.outcome, "escalated": o.escalated,
                          "cats": o.categories, "solvable": s.solvable, "reward": r,
                          "safe_reward": safe_reward(o, s), "n_calls": o.n_calls})
        step["n"] += 1
        n = max(1, len(stats))
        summary = {"call": step["n"], "time": time.time(), "n": len(stats),
                   "escalation_rate": sum(x["escalated"] for x in stats) / n,
                   "success_rate": sum(x["outcome"] == "success" for x in stats) / n,
                   "honest_stop_rate": sum(x["outcome"] == "honest_stop" for x in stats) / n,
                   "fabricated_rate": sum(x["outcome"] == "fabricated" for x in stats) / n,
                   "mean_reward": sum(out) / n,
                   "mean_safe_reward": sum(x["safe_reward"] for x in stats) / n,
                   "cats": sorted({c for x in stats for c in x["cats"]})}
        with open(log_path, "a") as f:
            f.write(json.dumps(summary) + "\n")
        print("[reward]", json.dumps(summary), flush=True)
        return out

    esa_reward.__name__ = f"esa_{kind}_reward"
    return esa_reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reward", choices=list(REWARDS), default="safe")
    ap.add_argument("--out", default="outputs/grpo")
    ap.add_argument("--data", default="data")
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--ga", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max_completion_length", type=int, default=2048)
    ap.add_argument("--max_turns", type=int, default=8)
    ap.add_argument("--vllm_mem", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--push", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from datasets import Dataset
    from agenttune import create_agentic_trainer

    world.register(world.load_jsonl(f"{args.data}/scenarios_rl.jsonl"))
    rows = [json.loads(l) for l in open(f"{args.data}/grpo_prompts.jsonl")]
    if args.limit:
        rows = rows[:args.limit]
    ds = Dataset.from_list([{"prompt": r["prompt"], "scenario_id": r["scenario_id"]} for r in rows])

    trainer = create_agentic_trainer(
        algorithm="grpo", backend="trl",
        model=args.model,
        reward_funcs=make_reward(args.reward, os.path.join(args.out, "reward_log.jsonl")),
        tools=[truncating(t) for t in world.TOOLS],
        tools_fallback_prompt=TOOLS_FALLBACK_PROMPT,
        tool_result_format=TOOL_RESULT_FORMAT,
        max_steps_per_turn=args.max_turns,
        train_dataset=ds,
        output_dir=args.out,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.ga,
        learning_rate=args.lr,
        beta=args.beta,
        temperature=1.0,
        top_p=1.0,
        max_completion_length=args.max_completion_length,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_mem,
        vllm_max_model_length=8192,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=10_000,
        warmup_steps=5,
        lr_scheduler_type="constant_with_warmup",
        report_to="none",
        log_completions=False,
        peft_config={"r": args.r, "lora_alpha": 2 * args.r, "lora_dropout": 0.0, "task_type": "CAUSAL_LM",
                     "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
    )
    t0 = time.time()
    stats = trainer.train()
    print("train seconds", round(time.time() - t0), flush=True)
    json.dump(stats, open(os.path.join(args.out, "train_stats.json"), "w"), indent=1, default=str)
    try:
        json.dump(trainer.trainer.state.log_history, open(os.path.join(args.out, "log_history.json"), "w"), indent=1)
    except Exception as e:  # noqa: BLE001
        print("no log history", e)
    # merge LoRA into the SFT base for eval/publishing
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, args.out).merge_and_unload()
    merged.save_pretrained(args.out + "/merged", safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out + "/merged")
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, private=True, exist_ok=True)
        api.upload_folder(folder_path=args.out + "/merged", repo_id=args.push, commit_message=f"GRPO {args.reward}")
        print("pushed", args.push, flush=True)


if __name__ == "__main__":
    main()
