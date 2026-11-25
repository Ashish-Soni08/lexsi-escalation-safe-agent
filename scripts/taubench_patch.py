"""Patch the installed tau-bench react agent for small-context models (scripts/taubench_chain.sh, TRIM=1).
The retail instructions alone are about 4,500 tokens and Tiny Aya reads at most 8,192, so old turns are dropped:
the instructions and the customer's first message always stay, then as many of the most recent turns as fit.
The same rule is used for every model. Usage: taubench_patch.py <path to tau_bench package>"""
import sys

p = sys.argv[1] + "/agents/chat_react_agent.py"
s = open(p).read()
old = "            messages=messages,\n            temperature=self.temperature,\n"
assert old in s and "_trim" not in s
s = s.replace(old, "            messages=_trim(messages),\n            temperature=self.temperature,\n            max_tokens=512,\n")
s += '''

HISTORY_CHARS = 7500  # about 2,500 tokens at 3 characters per token (a low estimate, so the real count is smaller)


def _trim(messages):
    head, tail = messages[:2], messages[2:]
    keep, used = [], 0
    for i in range(len(tail) - 2, -1, -2):  # pairs: assistant step, then the reply to it
        pair = tail[i:i + 2]
        n = sum(len(m.get("content") or "") for m in pair)
        if keep and used + n > HISTORY_CHARS:
            break
        if not keep and n > HISTORY_CHARS:  # one huge tool output: cut it, keep the step
            pair = [pair[0], {**pair[1], "content": (pair[1].get("content") or "")[:HISTORY_CHARS]}]
        keep = pair + keep
        used += n
    return head + keep
'''
open(p, "w").write(s)
print("patched", p)
