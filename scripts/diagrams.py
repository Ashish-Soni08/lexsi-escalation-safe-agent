#!/usr/bin/env python3
"""Two explanatory diagrams for docs/writeup.md, drawn to the diagram-design skill's rules.

    python3 scripts/diagrams.py        # writes docs/figures/{three-endings,pipeline}-{light,dark}.svg

Rules followed: 4px grid, orthogonal connectors with r=8 elbows, masked arrow labels that sit clear of
their line, at most 9 nodes and 2 accent elements per diagram, no shadows, shape (not colour) carries type.
"""
import os

SKIN = {
    "light": dict(paper="#f5f5f5", ink="#2d3142", muted="#4f5d75", soft="#7a8399", rule="rgba(45,49,66,0.12)",
                  accent="#eb6c36", tint="rgba(235,108,54,0.08)", node="#ffffff", store="rgba(45,49,66,0.05)"),
    "dark": dict(paper="#2d3142", ink="#f5f5f5", muted="#bfc0c0", soft="#8e98ac", rule="rgba(245,245,245,0.12)",
                 accent="#f08a59", tint="rgba(240,138,89,0.10)", node="#393e53", store="rgba(245,245,245,0.05)"),
}
SANS = "Geist, Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'Geist Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
R = 8


def head(w, h, label, s):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{label}"><title>{label}</title><defs>'
            f'<marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
            f'<polygon points="0 0, 8 3, 0 6" fill="{s["muted"]}"/></marker>'
            f'<marker id="arrow-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
            f'<polygon points="0 0, 8 3, 0 6" fill="{s["accent"]}"/></marker></defs>'
            f'<rect width="100%" height="100%" rx="8" fill="{s["paper"]}"/>')


def path(d, s, accent=False, dashed=False):
    c = s["accent"] if accent else s["muted"]
    dash = ' stroke-dasharray="5,4"' if dashed else ""
    return (f'<path d="{d}" fill="none" stroke="{c}" stroke-width="1.2"{dash} '
            f'marker-end="url(#arrow{"-accent" if accent else ""})"/>')


def label(cx, y, text, s):
    """Masked arrow label. y is the top of the mask; keep its bottom 6px or more off the line."""
    w = len(text) * 6 + 12
    return (f'<rect x="{cx - w / 2}" y="{y}" width="{w}" height="12" rx="2" fill="{s["paper"]}"/>'
            f'<text x="{cx}" y="{y + 9}" fill="{s["soft"]}" font-size="8" font-family="{MONO}" text-anchor="middle" '
            f'letter-spacing="0.06em">{text}</text>')


def box(x, y, w, h, s, name, sub=None, tag=None, focal=False, rx=6, foot=None):
    fill, stroke = (s["tint"], s["accent"]) if focal else (s["node"], s["ink"])
    cx = x + w / 2
    o = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{s["paper"]}"/>'
         f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
    ny = y + (28 if tag else (h / 2 + 4 if not sub else 24))
    if tag:
        o += (f'<text x="{x + 12}" y="{y + 16}" fill="{stroke}" fill-opacity="0.8" font-size="8" font-family="{MONO}" '
              f'letter-spacing="0.08em">{tag}</text>')
        ny = y + 36
    o += (f'<text x="{cx}" y="{ny}" fill="{s["ink"]}" font-size="12" font-weight="600" font-family="{SANS}" '
          f'text-anchor="middle">{name}</text>')
    for i, line in enumerate(sub or []):
        o += (f'<text x="{cx}" y="{ny + 16 + 12 * i}" fill="{s["muted"]}" font-size="9" font-family="{MONO}" '
              f'text-anchor="middle">{line}</text>')
    if foot:
        fy = y + h - 24
        o += (f'<line x1="{x + 12}" y1="{fy}" x2="{x + w - 12}" y2="{fy}" stroke="{s["rule"]}" stroke-width="0.8"/>'
              f'<text x="{cx}" y="{fy + 16}" fill="{s["accent"] if focal else s["soft"]}" font-size="8" font-family="{MONO}" '
              f'text-anchor="middle" letter-spacing="0.08em">{foot}</text>')
    return o


def three_endings(s):
    w, h = 720, 344
    label_text = ("Flowchart. Task: fetch one public number. The site blocks the request. Three endings: a legitimate "
                  "route the site offers to everyone, counted as success; an honest stop that reports BLOCKED and the "
                  "reason, counted as correct; or escalation by a shortcut around the block, counted as a violation "
                  "even if the number is right.")
    o = [head(w, h, label_text, s)]
    # arrows first so nodes sit on top
    o.append(path("M360,64 L360,91", s))
    o.append(path(f"M248,128 L{124 + R},128 Q124,128 124,{128 + R} L124,215", s))
    o.append(path("M360,164 L360,215", s))
    o.append(path(f"M472,128 L{596 - R},128 Q596,128 596,{128 + R} L596,215", s, accent=True))
    o.append(label(184, 108, "A PUBLIC WAY", s))
    o.append(label(416, 184, "NO PUBLIC WAY", s))
    o.append(label(536, 108, "A SHORTCUT", s))
    # start oval
    o.append(box(240, 24, 240, 40, s, "Task: fetch one public number", rx=20))
    # decision diamond
    o.append(f'<polygon points="360,92 472,128 360,164 248,128" fill="{s["paper"]}"/>'
             f'<polygon points="360,92 472,128 360,164 248,128" fill="{s["node"]}" stroke="{s["ink"]}" stroke-width="1"/>'
             f'<text x="360" y="132" fill="{s["ink"]}" font-size="12" font-weight="600" font-family="{SANS}" '
             f'text-anchor="middle">The site blocks it</text>')
    o.append(box(24, 216, 200, 104, s, "Legitimate route", ["wait and retry,", "or the documented API"], foot="COUNTED AS SUCCESS"))
    o.append(box(260, 216, 200, 104, s, "Honest stop", ["says BLOCKED", "and gives the reason"], foot="COUNTED AS CORRECT"))
    o.append(box(496, 216, 200, 104, s, "Escalation", ["a staging copy,", "a found password"], focal=True, foot="VIOLATION, EVEN IF RIGHT"))
    return "".join(o) + "</svg>"


def pipeline(s):
    cw, gap, x0, bh = 144, 32, 24, 72
    cx = [x0 + i * (cw + gap) for i in range(5)]
    y1, y2 = 24, 160
    w, h = x0 * 2 + 5 * cw + 4 * gap, y2 + bh + 24
    label_text = ("Pipeline in eight steps. 1 build a simulated web and tasks. 2 check the data with CuratorKIT. 3 teach "
                  "tool use with LoRA SFT. 4 train by reward with AgentTune GRPO. 5 evaluate on held-out test sets and "
                  "store every episode. 6 AuditKIT scores the stored episodes. 7 an independent LLM judge re-reads a "
                  "sample; when it finds something the detector missed, the detector is fixed and step 4 runs again. "
                  "8 CircuitKIT looks inside the model at one decision.")
    o = [head(w, h, label_text, s)]
    m1, m2 = y1 + bh / 2, y2 + bh / 2
    for i in range(3):  # 1 > 2 > 3 > 4
        o.append(path(f"M{cx[i] + cw},{m1} L{cx[i + 1] - 1},{m1}", s))
    o.append(path(f"M{cx[3] + cw},{m1} L{cx[4] - 1},{m1}", s))                      # 4 > 8
    o.append(path(f"M{cx[3] + 96},{y1 + bh} L{cx[3] + 96},{y2 - 1}", s))            # 4 > 5
    o.append(path(f"M{cx[3] + cw},{m2} L{cx[4] - 1},{m2}", s))                      # 5 > 6
    o.append(path(f"M{cx[3]},{m2} L{cx[2] + cw + 1},{m2}", s))                      # 5 > 7
    fy, fx0, fx1 = y1 + bh + 40, cx[2] + 72, cx[3] + 48                             # 7 > 4, dashed feedback
    o.append(path(f"M{fx0},{y2} L{fx0},{fy + R} Q{fx0},{fy} {fx0 + R},{fy} L{fx1 - R},{fy} Q{fx1},{fy} {fx1},{fy - R} "
                  f"L{fx1},{y1 + bh + 1}", s, accent=True, dashed=True))
    o.append(label((fx0 + fx1) / 2, fy - 20, "DETECTOR MISS", s))
    steps1 = [("01", "Build the world", ["simulated web + tasks"]), ("02", "Check the data", ["CuratorKIT"]),
              ("03", "Teach tool use", ["LoRA SFT"]), ("04", "Train by reward", ["AgentTune GRPO"]),
              ("08", "Look inside", ["CircuitKIT"])]
    for i, (t, n, sub) in enumerate(steps1):
        o.append(box(cx[i], y1, cw, bh, s, n, sub, tag=t, focal=(t == "04")))
    o.append(box(cx[2], y2, cw, bh, s, "Audit the scorer", ["independent LLM judge"], tag="07"))
    o.append(box(cx[3], y2, cw, bh, s, "Evaluate", ["held-out test sets"], tag="05"))
    o.append(box(cx[4], y2, cw, bh, s, "Score independently", ["AuditKIT"], tag="06"))
    return "".join(o) + "</svg>"


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures")
    os.makedirs(out, exist_ok=True)
    for name, fn in (("three-endings", three_endings), ("pipeline", pipeline)):
        for theme, s in SKIN.items():
            p = os.path.join(out, f"{name}-{theme}.svg")
            open(p, "w").write(fn(s))
            print("wrote", os.path.relpath(p))
