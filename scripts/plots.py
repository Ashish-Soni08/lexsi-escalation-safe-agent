"""Charts for the README / write-up (results/ -> assets/)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auditkit.agent_eval import episodes_from_agenttune  # noqa: E402

from esa.detector import classify_episode  # noqa: E402
from scripts.audit import _calls  # noqa: E402

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a3a29c"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF})


def roll(xs, k=10):
    return [sum(xs[max(0, i - k + 1):i + 1]) / len(xs[max(0, i - k + 1):i + 1]) for i in range(len(xs))]


def training_curves(traj_paths, out):
    """Escalation rate per GRPO step, re-derived from AgentTune-written rollouts read by AuditKIT."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name, path, color in traj_paths:
        by = {}
        for ep in episodes_from_agenttune(path):
            step = int(str(ep.source_id).rsplit("#", 1)[1])
            by.setdefault(step, []).append(bool(classify_episode(_calls(ep), None)))
        steps = sorted(by)
        rate = roll([100 * sum(by[s]) / len(by[s]) for s in steps])
        ax.plot(steps, rate, color=color, lw=2)
        ax.text(steps[-1] + 1, rate[-1], name, color=INK, va="center", fontsize=10)
    ax.set_xlabel("GRPO step (32 rollouts each)")
    ax.set_ylabel("rollouts with an exploit-shaped call (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Escalation during training, same starting model", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")


def outcomes(runs, out, variant="neutral"):
    order = [("success", "legit success", BLUE), ("honest_stop", "honest stop", AQUA),
             ("escalated", "escalated", ORANGE), ("fabricated", "made-up number", YELLOW),
             ("no_answer", "no answer", GRAY)]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    labels = []
    for y, (name, label) in enumerate(runs):
        m = json.load(open(f"results/{name}_{variant}.json"))["metrics"]
        n = m["n"]
        left = 0
        for key, _, color in order:
            v = 100 * m["outcomes"].get(key, 0) / n
            if v:
                ax.barh(y, v, left=left, color=color, height=0.6, edgecolor=SURF, linewidth=2)
                if v >= 7:
                    ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", color="white", fontsize=9)
            left += v
        labels.append(label)
    ax.set_yticks(range(len(runs)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"share of 320 test tasks (%), '{variant}' prompt")
    ax.spines[["top", "right", "left"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in order]
    ax.legend(handles, [l for _, l, _ in order], ncol=5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.45, 1.16), fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    outcomes([("base", "Tiny Aya Global"), ("sft", "+ tool SFT (clean)"), ("sft_esc", "+ tool SFT (escalation-prone)"),
              ("grpo_esc_success_only", "  → GRPO, success-only reward"), ("grpo_esc_safe", "  → GRPO, safe reward")],
             "assets/outcomes_neutral.png")
    if len(sys.argv) > 2:
        training_curves([("success-only", sys.argv[1], ORANGE), ("safe", sys.argv[2], BLUE)],
                        "assets/training_escalation.png")
