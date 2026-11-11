"""Write episodes as AgentTune trajectory records.

The record shape is exactly ``agenttune.agentic.trajectory.dataset.Trajectory.to_dict()``
(task, steps[{step_number, state, action{tool_calls[{id,name,arguments}]}, observation,
thought, reward}], reward, trajectory_id, final_response, metadata{conversation, ...}),
which is what AgentTune's GRPO rollout writes to ``trajectories.jsonl`` and what
AuditKIT's ``episodes_from_agenttune`` reads. Writing it ourselves for the eval runs
means baseline, SFT and GRPO models are all audited from the same file format.
"""

from __future__ import annotations

import json
from typing import Iterable

from .scoring import Episode, Outcome


def episode_record(ep: Episode, o: Outcome, reward: float | None = None, model: str = "") -> dict:
    steps = []
    call_i = 0
    k = 0
    msgs = ep.messages
    # assistant turns after the initial system/user pair
    for m_i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        nxt = msgs[m_i + 1] if m_i + 1 < len(msgs) else None
        is_tool_turn = nxt is not None and nxt["role"] == "user" and nxt["content"].startswith("Tool result")
        if is_tool_turn:
            n = nxt["content"].count("Tool result for ")
            calls = ep.calls[call_i:call_i + n]
            results = ep.results[call_i:call_i + n]
            call_i += n
            steps.append({
                "step_number": k, "state": f"[turn={k}]",
                "action": {"tool_calls": [{"id": f"call_{k}_{j}", "name": c[0], "arguments": c[1]}
                                          for j, c in enumerate(calls)]},
                "observation": str({c[0]: r for c, r in zip(calls, results)}),
                "thought": m["content"], "reward": None,
            })
        else:
            steps.append({"step_number": k, "state": "[terminal]", "action": {},
                          "observation": m["content"], "thought": m["content"], "reward": None})
        k += 1
    return {
        "task": msgs[1]["content"] if len(msgs) > 1 else "",
        "steps": steps,
        "reward": reward,
        "trajectory_id": ep.scenario_id,
        "final_response": ep.final_text,
        "metadata": {
            "conversation": msgs[1:],
            "tool_call_count": len(ep.calls),
            "scenario_id": ep.scenario_id,
            "outcome": o.outcome,
            "escalated": o.escalated,
            "categories": o.categories,
            "model": model,
        },
    }


def write_trajectories(path: str, records: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
