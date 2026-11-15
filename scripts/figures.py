"""Draw the write-up's mechanism diagrams as hand-built SVG, in a light and a dark version.

    python scripts/figures.py            # writes docs/figures/<name>-light.svg and <name>-dark.svg

Numbers are copied from results/report_r6.md and docs/hf_article.md; no data is read at run time.
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
FONT = "Geist, Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'Geist Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# One skin for every write-up diagram, shared with scripts/diagrams.py: paper, ink, muted and the orange accent.
# Outcome hues stay the same as the result charts: blue legit, green honest stop, orange escalation, yellow made-up.
THEMES = {
    "light": dict(bg="#f5f5f5", fg="#2d3142", muted="#4f5d75", line="#7a8399", faint="#dcdde0", panel="#ffffff",
                  blue="#2a78d6", green="#14a06f", orange="#eb6c36", yellow="#c98a00", grey="#8a8f9c"),
    "dark": dict(bg="#2d3142", fg="#f5f5f5", muted="#bfc0c0", line="#8e98ac", faint="#454b63", panel="#393e53",
                 blue="#6aa9f5", green="#3fcf95", orange="#f08a59", yellow="#f2b937", grey="#9aa2b3"),
}


class SVG:
    def __init__(self, w: int, h: int, label: str, th: dict):
        self.w, self.h, self.th = w, h, th
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{escape(label, {chr(34): "&quot;"})}" font-family="{FONT}">',
            f"<title>{escape(label)}</title>",
            "<defs>"
            + "".join(
                f'<marker id="ar-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{th[k]}"/></marker>'
                for k in ("line", "orange", "green", "blue", "muted")
            )
            + f'<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<line x1="0" y1="0" x2="0" y2="6" stroke="{th["line"]}" stroke-width="1.6" opacity=".55"/></pattern>'
            "</defs>",
            f'<rect width="{w}" height="{h}" rx="8" fill="{th["bg"]}"/>',
        ]

    def c(self, k):
        return self.th.get(k, k)

    def text(self, x, y, s, size=12, fill="fg", anchor="start", weight=400, mono=False, italic=False):
        fam = f' font-family="{MONO}"' if mono else ""
        it = ' font-style="italic"' if italic else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{self.c(fill)}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{fam}{it}>{escape(s)}</text>'
        )

    def line(self, x1, y1, x2, y2, stroke="line", width=1.2, dash=None, arrow=None, opacity=1):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#ar-{arrow})"' if arrow else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{self.c(stroke)}" stroke-width="{width}"'
            f' opacity="{opacity}"{d}{m}/>'
        )

    def path(self, d, stroke="line", width=1.2, dash=None, arrow=None, fill="none"):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#ar-{arrow})"' if arrow else ""
        self.parts.append(f'<path d="{d}" fill="{self.c(fill)}" stroke="{self.c(stroke)}" stroke-width="{width}"{da}{m}/>')

    def rect(self, x, y, w, h, fill="panel", stroke="line", width=1, rx=6, dash=None, fill_opacity=1):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        f = self.c(fill) if not fill.startswith("url") else fill
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{f}" fill-opacity="{fill_opacity}" '
            f'stroke="{self.c(stroke)}" stroke-width="{width}"{da}/>'
        )

    def dot(self, x, y, r=7, fill="blue", stroke="bg", width=2):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{self.c(fill)}" stroke="{self.c(stroke)}" stroke-width="{width}"/>')

    def save(self, name, theme):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{name}-{theme}.svg").write_text("\n".join(self.parts + ["</svg>"]) + "\n")


# --------------------------------------------------------------------------- #
# 1. Two rewards on one scale                                                  #
# --------------------------------------------------------------------------- #
def reward_scale(th):
    s = SVG(820, 300, "Two rewards on one score line. Under the success-only reward an escalation that returns the "
            "number scores +1, the same as a legitimate success, and an honest stop scores 0. Under the safe reward "
            "escalation scores -1 or lower, below a made-up number (-0.6), no answer (-0.3), an honest stop (+0.6) "
            "and a legitimate success (+1).", th)
    X = lambda v: 200 + (v + 1.5) / 2.5 * 580  # -1.5 .. +1  ->  200 .. 780

    def axis(y, title, sub):
        s.text(24, y - 6, title, 14, weight=700)
        s.text(24, y + 12, sub, 11.5, "muted")
        s.line(X(-1.5), y, X(1.0) + 12, y, "line", 1.2)
        for v in (-1.5, -1, -0.5, 0, 0.5, 1):
            s.line(X(v), y - 4, X(v), y + 4, "line", 1)
            s.text(X(v), y + 20, f"{v:+g}" if v else "0", 11, "muted", "middle")

    y1, y2 = 105, 250
    axis(y1, "Success-only reward", "right number → 1, else 0")
    axis(y2, "Safe reward", "escalation always lowest")

    # success-only: ties
    s.text(X(1) - 2, y1 - 52, "legit success", 12, "blue", "end", 600)
    s.text(X(1) - 2, y1 - 36, "escalation that got the number", 12, "orange", "end", 600)
    s.text(X(1) - 2, y1 - 20, "both +1", 11.5, "muted", "end")
    s.dot(X(1) - 7, y1, fill="blue"); s.dot(X(1) + 7, y1, fill="orange")
    s.text(X(0) + 20, y1 - 52, "honest stop", 12, "green", "end", 600)
    s.text(X(0) + 20, y1 - 36, "made-up number", 12, "yellow", "end", 600)
    s.text(X(0) + 20, y1 - 20, "no answer: all 0", 11.5, "muted", "end")
    s.dot(X(0) - 14, y1, fill="green"); s.dot(X(0), y1, fill="yellow"); s.dot(X(0) + 14, y1, fill="grey")

    # safe reward: an ordered price list
    s.parts.append(f'<line x1="{X(-1.5)}" y1="{y2}" x2="{X(-1)}" y2="{y2}" stroke="{th["orange"]}" stroke-width="7" stroke-linecap="round"/>')
    s.dot(X(-1), y2, fill="orange")
    s.text(X(-1.25), y2 - 36, "escalation", 12, "orange", "middle", 600)
    s.text(X(-1.25), y2 - 20, "−1, to −1.5 if repeated", 11.5, "muted", "middle")
    s.dot(X(-0.6), y2, fill="yellow")
    s.text(X(-0.6), y2 - 36, "made-up number", 12, "yellow", "middle", 600)
    s.text(X(-0.6), y2 - 20, "−0.6", 11.5, "muted", "middle")
    s.dot(X(-0.3), y2, fill="grey")
    s.text(X(-0.3) + 4, y2 - 52, "no answer −0.3", 11.5, "muted", "start")
    s.line(X(-0.3), y2 - 47, X(-0.3), y2 - 9, "faint", 1)
    s.dot(X(0.6), y2, fill="green")
    s.text(X(0.6), y2 - 36, "honest stop", 12, "green", "middle", 600)
    s.text(X(0.6), y2 - 20, "+0.6", 11.5, "muted", "middle")
    s.dot(X(1), y2, fill="blue")
    s.text(X(1) + 22, y2 - 36, "legit success", 12, "blue", "end", 600)
    s.text(X(1) + 22, y2 - 20, "+1", 11.5, "muted", "end")

    # the one move that matters
    s.path(f"M{X(1)+7},{y1+10} C{X(1)+7},{y1+70} {X(-1)+40},{y1+40} {X(-1)+4},{y2-62}", "orange", 1.6, "5 4", "orange")
    s.text(X(1) + 22, y1 + 70, "same behaviour, moved to the bottom", 12, "orange", "end", 600)
    return s


# --------------------------------------------------------------------------- #
# 2. GRPO: one task, eight attempts                                            #
# --------------------------------------------------------------------------- #
def grpo_group(th):
    s = SVG(820, 270, "GRPO on one blocked task. The model makes eight attempts; five stop honestly (+0.6), two "
            "escalate (-1) and one makes up a number (-0.6). The group average is +0.05. Attempts above the average "
            "are made more likely and attempts below it less likely.", th)
    Y = lambda v: 128 - 80 * v
    s.text(24, 30, "One blocked task, 8 attempts, each scored by the safe reward", 14, weight=700)
    ax = 96
    s.line(ax, Y(1.05), ax, Y(-1.1), "line")
    for v in (1, 0.5, 0, -0.5, -1):
        s.line(ax - 4, Y(v), ax, Y(v), "line")
        s.text(ax - 8, Y(v) + 4, f"{v:+g}" if v else "0", 11, "muted", "end")
    s.text(40, Y(0) + 4, "score", 11, "muted", "end")

    tries = [("green", 0.6, "stop"), ("orange", -1, "escalate"), ("green", 0.6, "stop"), ("yellow", -0.6, "made-up"),
             ("green", 0.6, "stop"), ("green", 0.6, "stop"), ("orange", -1, "escalate"), ("green", 0.6, "stop")]
    mean = sum(v for _, v, _ in tries) / len(tries)
    x0, dx = 140, 54
    xe = x0 + dx * (len(tries) - 1) + 30
    s.line(ax, Y(mean), xe, Y(mean), "fg", 1.2, "6 4")
    s.text(xe + 6, Y(mean) + 4, f"group average {mean:+.2f}", 12, "fg", "start", 600)
    for i, (col, v, lab) in enumerate(tries):
        x = x0 + i * dx
        s.line(x, Y(mean), x, Y(v), col, 3, opacity=0.55)
        s.dot(x, Y(v), 7, col)
        s.text(x, 232, lab, 11, col, "middle", 600)
        s.text(x, 248, f"{v:+g}", 10.5, "muted", "middle")

    nx = xe + 6
    s.line(nx + 8, Y(0.6) + 6, nx + 8, Y(0.6) - 18, "green", 2, arrow="green")
    s.text(nx + 22, Y(0.6) - 6, "above average:", 12, "green", weight=600)
    s.text(nx + 22, Y(0.6) + 9, "made more likely", 12, "muted")
    s.line(nx + 8, Y(-0.8) - 12, nx + 8, Y(-0.8) + 12, "orange", 2, arrow="orange")
    s.text(nx + 22, Y(-0.8) - 2, "below average:", 12, "orange", weight=600)
    s.text(nx + 22, Y(-0.8) + 13, "made less likely", 12, "muted")
    s.text(24, 262, "Stems show each attempt's distance from the average (its advantage). Illustrative group.", 11, "muted", italic=True)
    return s


# --------------------------------------------------------------------------- #
# 3. Model family tree                                                         #
# --------------------------------------------------------------------------- #
def lineage(th):
    s = SVG(900, 470, "Where every model comes from. Tiny Aya Global gets tool SFT, giving clean SFT, or tool SFT with "
            "215 shortcut demonstrations, giving the escalation-prone start model. From that start, GRPO with the "
            "success-only reward gives a model that escalates on 99% of blocked test tasks; GRPO with the safe reward "
            "gives safe v2, then v3.1 and v4.1, the released model. Three reward ablations also start there. Seeds 14 "
            "and 24 repeat the v4.1 step, and v5 continues from v4.1.", th)
    W, H = 206, 58

    def node(cx, y, name, how, res, kind="base", res_col="muted"):
        stroke = {"base": "line", "unsafe": "orange", "safe": "green", "final": "green"}[kind]
        fill = "panel"
        x = cx - W / 2
        s.rect(x, y, W, H, fill, stroke, 2.4 if kind == "final" else 1.3, 8)
        s.text(cx, y + 19, name, 13, "fg", "middle", 700)
        s.text(cx, y + 35, how, 11, "muted", "middle")
        s.text(cx, y + 50, res, 11, res_col, "middle", 600)
        return cx, y

    def edge(a, b, dashed=False):
        (ax, ay), (bx, by) = a, b
        if ay == by:  # side by side
            s.line(ax + W / 2, ay + H / 2, bx - W / 2 - 3, by + H / 2, "line", 1.3, "4 3" if dashed else None, "line")
        else:
            s.line(ax, ay + H, bx, by - 3, "line", 1.3, "4 3" if dashed else None, "line")

    base = node(350, 24, "Tiny Aya Global", "base model, 3.35B", "cannot use our tools")
    clean = node(170, 120, "clean SFT", "tool SFT, 1,600 expert demos", "baseline, no planted habit")
    esc = node(530, 120, "escalation-prone SFT", "same + 215 shortcut demos", "≈5% escalation (planted)", "unsafe", "orange")
    succ = node(300, 216, "GRPO, success-only", "reward: right number = 1", "99.2% escalation on test", "unsafe", "orange")
    v2 = node(530, 216, "safe v2", "GRPO, safe reward", "test 0%, stress 78.5%", "safe")
    abl = node(770, 216, "3 reward ablations", "safe reward minus one term", "no penalty: 98% again", "base", "orange")
    v31 = node(530, 312, "safe v3.1", "+ hard tasks, detector fixed", "stress 36.2%", "safe")
    seeds = node(290, 400, "seeds 14 and 24", "v4.1 recipe, other seeds", "sealed 14.5% and 13.0%", "safe", "green")
    v41 = node(530, 400, "v4.1  (released)", "+ wider task mix, seed 4", "test 0%, sealed 21.7%", "final", "green")
    v5 = node(770, 400, "v5", "+160 steps on new routes", "same as v4.1 (null result)", "safe")
    for a, b in ((base, clean), (base, esc), (esc, succ), (esc, v2), (esc, abl), (v2, v31), (v31, v41), (v31, seeds)):
        edge(a, b)
    edge(v41, v5)

    # legend: the border colour repeats, so it earns one
    lx, ly = 668, 34
    for i, (k, lab) in enumerate((("orange", "trained to escalate: research only"),
                                  ("green", "safe reward"), ("line", "baseline"))):
        s.rect(lx, ly + i * 20 - 9, 16, 12, "panel", k, 1.6, 3)
        s.text(lx + 24, ly + i * 20 + 1, lab, 11.5, "muted")
    return s


# --------------------------------------------------------------------------- #
# 4. The rollout fix                                                           #
# --------------------------------------------------------------------------- #
def rollout_fix(th):
    s = SVG(860, 300, "The rollout fix. Before: the sampler generated the second tool call after a tool result "
            "wrapped in Tiny Aya's chat template, but the trainer scored it after the raw result, so the probability "
            "ratio was about one in a million, every sequence was masked and the gradient was zero. After: sampler and "
            "trainer share one token sequence, and the tool result is masked from the loss.", th)
    x0 = 190

    def row(y, label, sub, segs):
        s.text(24, y + 15, label, 12.5, weight=700)
        s.text(24, y + 30, sub, 11, "muted")
        x = x0
        for w, kind, txt in segs:
            if kind == "model":
                s.rect(x, y, w, 36, "blue", "blue", 1.3, 5, fill_opacity=0.14)
                s.text(x + w / 2, y + 23, txt, 11.5, "blue", "middle", 600)
            elif kind == "bad":
                s.rect(x, y, w, 36, "orange", "orange", 2, 5, fill_opacity=0.14)
                s.text(x + w / 2, y + 23, txt, 11.5, "orange", "middle", 700)
            elif kind == "env":
                s.rect(x, y, w, 36, "panel", "line", 1.1, 5)
                s.text(x + w / 2, y + 23, txt, 11.5, "muted", "middle", mono=True)
            elif kind == "masked":
                s.rect(x, y, w, 36, "url(#hatch)", "line", 1.1, 5)
                s.text(x + w / 2, y + 23, txt, 11.5, "fg", "middle", mono=True)
            elif kind == "tag":
                s.rect(x, y, w, 36, "yellow", "yellow", 1.3, 5, fill_opacity=0.22)
                s.text(x + w / 2, y + 23, txt, 11, "yellow", "middle", 700, mono=True)
            x += w + 5
        return x

    s.text(24, 30, "Before the fix", 14, "orange", weight=700)
    row(46, "sampler generated", "what vLLM produced", [(96, "env", "question"), (92, "model", "tool call 1"),
        (52, "tag", "<tool>"), (130, "env", "result"), (58, "tag", "</tool>"), (92, "model", "tool call 2"), (78, "model", "answer")])
    row(96, "trainer scored", "AgentTune's built-in rollout", [(96, "env", "question"), (92, "model", "tool call 1"),
        (250, "env", "raw result, no template"), (92, "bad", "tool call 2"), (78, "bad", "answer")])
    xr = x0 + 96 + 92 + 250 + 15
    s.text(xr + 85, 150, "scored in a context the sampler never saw:", 11.5, "orange", "middle", 600)
    s.text(xr + 85, 165, "ratio ≈ 1e-6 → masked → zero gradient", 11.5, "orange", "middle", 600)

    s.line(24, 184, 836, 184, "faint", 1)
    s.text(24, 208, "After the fix (esa/rollout.py)", 14, "green", weight=700)
    row(222, "sampler = trainer", "one token sequence", [(96, "env", "question"), (92, "model", "tool call 1"),
        (52, "tag", "<tool>"), (130, "masked", "result"), (58, "tag", "</tool>"), (92, "model", "tool call 2"), (78, "model", "answer")])
    s.text(x0, 284, "Hatched and grey: tokens from the environment, left out of the loss. Blue: the model's own tokens, the only ones trained.",
           11, "muted")
    s.text(836, 208, "gap 0.30 → 0.0005 nats per token", 12.5, "green", "end", 700)
    return s


# --------------------------------------------------------------------------- #
# 5. Guard vs training                                                         #
# --------------------------------------------------------------------------- #
def guard_vs_training(th):
    s = SVG(860, 320, "Two layers against escalation on blocked tasks. On known shortcut shapes (stress set), training "
            "cuts attempts from 93.1% for clean SFT to 1.5% for v4.1, and the guard stops the rest: 0.0% reaches the "
            "tools. On unfamiliar shapes (fresh set), training cuts attempts from 48.0% to 11.0%, and the guard does "
            "not recognise them: 11.0% still reaches the tools.", th)
    cols = {"b1": 230, "g1": 350, "b2": 470, "g2": 600, "b3": 730}
    heads = [("b1", "clean SFT", "attempts"), ("g1", "train with the", "safe reward"), ("b2", "v4.1", "attempts"),
             ("g2", "runtime guard", "(the detector)"), ("b3", "reaches", "the tools")]
    for k, a, b in heads:
        col = "green" if k == "g1" else ("blue" if k == "g2" else "fg")
        s.text(cols[k], 32, a, 12.5, col, "middle", 700)
        s.text(cols[k], 48, b, 12, col if k[0] == "g" else "muted", "middle", 600 if k[0] == "g" else 400)

    lanes = [(160, "Known shortcut shapes", "stress set", (93.1, 1.5, 0.0), True),
             (290, "Unfamiliar shapes", "fresh set", (48.0, 11.0, 11.0), False)]
    for base, t1, t2, vals, caught in lanes:
        s.text(24, base - 34, t1, 13, weight=700)
        s.text(24, base - 18, t2, 11.5, "muted")
        s.line(150, base, 800, base, "faint", 1)
        for k, v in zip(("b1", "b2", "b3"), vals):
            h = max(v * 0.85, 1.5)
            s.rect(cols[k] - 24, base - h, 48, h, "orange", "orange", 1, 2, fill_opacity=0.85)
            s.text(cols[k], base - h - 6, f"{v:.1f}%", 12.5, "fg", "middle", 700)
        # training gate: always narrows
        s.rect(cols["g1"] - 5, base - 88, 10, 88, "green", "green", 1, 3, fill_opacity=0.9)
        s.line(cols["b1"] + 30, base - 12, cols["g1"] - 9, base - 12, "line", 1.2, arrow="line")
        s.line(cols["g1"] + 9, base - 12, cols["b2"] - 30, base - 12, "line", 1.2, arrow="line")
        # guard gate: solid where it recognises the shape, open where it does not
        if caught:
            s.rect(cols["g2"] - 5, base - 88, 10, 88, "blue", "blue", 1, 3, fill_opacity=0.9)
            s.line(cols["b2"] + 30, base - 12, cols["g2"] - 9, base - 12, "line", 1.2, arrow="line")
            s.text(cols["g2"] + 12, base - 64, "catches it", 11.5, "blue", "start", 600)
        else:
            s.rect(cols["g2"] - 5, base - 88, 10, 88, "bg", "blue", 1.4, 3, dash="4 3")
            s.line(cols["b2"] + 30, base - 12, cols["b3"] - 30, base - 12, "line", 1.2, arrow="line")
            s.text(cols["g2"] + 12, base - 64, "does not know", 11.5, "blue", "start", 600)
            s.text(cols["g2"] + 12, base - 50, "the shape", 11.5, "blue", "start", 600)
    s.text(24, 314, "Escalation on blocked tasks, greedy decoding, neutral prompt. Guard-only on clean SFT: stress 93.1% → 0.0%, fresh 48.0% → 48.0%.",
           11, "muted", italic=True)
    return s


# --------------------------------------------------------------------------- #
# 6. The detector was gamed                                                    #
# --------------------------------------------------------------------------- #
def detector_gamed(th):
    s = SVG(820, 300, "Escalation on blocked stress-test tasks as reported by the detector inside each model's own "
            "reward, against the rate after the independent judge's finding was fixed. v3: 0.8% reported, 52.3% "
            "real, via /debug/ probes. v3.1: 6.9% reported, 36.2% real, via a bare /debug path. v4.1: 1.5% "
            "reported, and the judge finds no hidden pattern.", th)
    Y = lambda v: 232 - v * 3.2
    ax = 90
    for v in (0, 20, 40, 60):
        s.line(ax, Y(v), 560, Y(v), "faint", 1)
        s.text(ax - 8, Y(v) + 4, f"{v}%", 11, "muted", "end")
    groups = [("v3", 0.8, 52.3, "probes /debug/… paths"), ("v3.1", 6.9, 36.2, "switches to bare /debug"),
              ("v4.1", 1.5, None, "judge: no hidden pattern")]
    for i, (name, rep, real, how) in enumerate(groups):
        cx = 170 + i * 160
        for j, (v, col) in enumerate(((rep, "grey"), (real, "orange"))):
            x = cx - 44 + j * 46
            if v is None:
                s.rect(x, Y(1.5) - 1, 42, 1.5 * 3.2 + 1, "orange", "orange", 1, 2, fill_opacity=0.85)
                s.text(x + 21, Y(1.5) - 7, "1.5%", 12.5, "fg", "middle", 700)
                continue
            h = max(v * 3.2, 2)
            s.rect(x, Y(0) - h, 42, h, col, col, 1, 2, fill_opacity=0.85)
            s.text(x + 21, Y(0) - h - 6, f"{v}%", 12.5, "fg", "middle", 700)
        s.text(cx - 2, 254, name, 13, "fg", "middle", 700)
        s.text(cx - 2, 271, how, 11.5, "muted", "middle")
    s.text(24, 26, "Escalation on blocked stress-test tasks", 14, weight=700)
    # legend (two encodings repeat across groups)
    lx = 600
    s.rect(lx, 70, 14, 14, "grey", "grey", 1, 2, fill_opacity=0.85)
    s.text(lx + 22, 81, "what the detector inside", 12, "fg")
    s.text(lx + 22, 96, "its own reward reported", 12, "fg")
    s.rect(lx, 118, 14, 14, "orange", "orange", 1, 2, fill_opacity=0.85)
    s.text(lx + 22, 129, "after the judge's finding,", 12, "fg")
    s.text(lx + 22, 144, "detector fixed, rescored", 12, "fg")
    s.text(lx, 186, "Each round the model learned", 11.5, "muted")
    s.text(lx, 201, "the one spelling the reward", 11.5, "muted")
    s.text(lx, 216, "could not see. Only a judge", 11.5, "muted")
    s.text(lx, 231, "reading whole episodes caught it.", 11.5, "muted")
    return s


FIGS = {"reward-scale": reward_scale, "grpo-group": grpo_group, "model-lineage": lineage,
        "rollout-fix": rollout_fix, "guard-vs-training": guard_vs_training, "detector-gamed": detector_gamed}

if __name__ == "__main__":
    for name, fn in FIGS.items():
        for theme, th in THEMES.items():
            fn(th).save(name, theme)
    print("wrote", len(FIGS) * 2, "files to", OUT)
