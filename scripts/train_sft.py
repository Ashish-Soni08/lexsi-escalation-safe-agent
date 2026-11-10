"""Tool-calling SFT of Tiny Aya on the scripted expert trajectories (LoRA).

Loss is on assistant turns only (tool calls, final answers, honest stops); system,
task and folded tool-result turns are masked. The format is exactly the AgentTune
Tiny Aya fallback the GRPO stage and the eval loop use.

    python scripts/train_sft.py --out outputs/sft --push ashish-soni08/tiny-aya-global-esa-sft
"""
import argparse
import json
import os
import random
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render(tok, msgs, gen=False):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)


def ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def encode(tok, msgs, max_len):
    """Concatenate per-turn segments of the rendered conversation; label only
    the assistant segments (content + end-of-turn tokens)."""
    input_ids, labels = [], []
    prev = ""
    for k, m in enumerate(msgs):
        if m["role"] == "assistant":
            pre = render(tok, msgs[:k], gen=True)
            assert pre.startswith(prev), "template is not append-only"
            seg = ids(tok, pre[len(prev):])
            input_ids += seg
            labels += [-100] * len(seg)
            full = render(tok, msgs[:k + 1])
            assert full.startswith(pre)
            seg = ids(tok, full[len(pre):])
            input_ids += seg
            labels += seg
            prev = full
    if len(input_ids) > max_len:
        return None
    return {"input_ids": input_ids, "labels": labels}


def collate(pad_id):
    def f(batch):
        n = max(len(b["input_ids"]) for b in batch)
        x = torch.full((len(batch), n), pad_id)
        y = torch.full((len(batch), n), -100)
        a = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            x[i, :L] = torch.tensor(b["input_ids"])
            y[i, :L] = torch.tensor(b["labels"])
            a[i, :L] = 1
        return {"input_ids": x, "labels": y, "attention_mask": a}
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--data", default="data/sft_trajectories.jsonl")
    ap.add_argument("--out", default="outputs/sft")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--ga", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--push", default="")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in open(args.data)]
    random.Random(0).shuffle(rows)
    if args.limit:
        rows = rows[:args.limit]
    enc = [e for e in (encode(tok, r["messages"], args.max_len) for r in rows) if e]
    # sanity: concatenated segments == tokenizing the full render
    r0 = rows[0]["messages"]
    full_ids = ids(tok, render(tok, r0))
    print("segment/full token match:", enc[0]["input_ids"] == full_ids, len(full_ids), flush=True)
    print("label sample:", repr(tok.decode([t for t in enc[0]["labels"] if t != -100])[:400]), flush=True)
    n_tok = sum(len(e["input_ids"]) for e in enc)
    print(f"{len(enc)} trajectories, {n_tok} tokens, "
          f"{sum(sum(1 for t in e['labels'] if t != -100) for e in enc)} trained tokens", flush=True)
    ds = Dataset.from_list(enc)

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out + "/ckpt", num_train_epochs=args.epochs, per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.ga, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_ratio=0.05, logging_steps=5, save_strategy="no", bf16=True, report_to="none",
        remove_unused_columns=False, dataloader_num_workers=2, seed=0)
    tr = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate(tok.pad_token_id))
    res = tr.train()
    print("train result:", res.metrics, flush=True)
    os.makedirs(args.out, exist_ok=True)
    json.dump({"metrics": res.metrics, "log_history": tr.state.log_history, "args": vars(args)},
              open(os.path.join(args.out, "sft_log.json"), "w"), indent=1)
    model.save_pretrained(args.out + "/adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out + "/merged", safe_serialization=True)
    tok.save_pretrained(args.out + "/merged")
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, private=True, exist_ok=True)
        api.upload_folder(folder_path=args.out + "/merged", repo_id=args.push, commit_message="SFT merged")
        print("pushed", args.push, flush=True)


if __name__ == "__main__":
    main()
