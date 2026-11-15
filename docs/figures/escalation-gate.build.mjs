import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../../kit/iso-kit.mjs";
import { kitScript } from "../../scripts/inline-kit.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const theme = process.argv.includes("--light") ? "light" : "dark";
const standalone = process.argv.includes("--svg");

const WIDTH = 640;
const HEIGHT = 400;

// measured results, test set, blocked tasks (results/report_r6.md)
const ESC_SUCCESS_ONLY = 99.2;
const ESC_SAFE = 0.0;
const HONEST_SAFE = 100;

const PLATE = { x: -128, y: -66, w: 256, d: 132, r: 8 };
const PLATE_H = 10;
const FOOT_H = 5;
const BED = { x: -116, y: -54, w: 222, d: 92, r: 4 };
const BED_H = 4;
const Z = BED_H;
const BALL_R = 4;
const RAIL_H = 2.4;
const RAIL_T = 1.4;

const HOPPER = [-102, -10];
const GATE_X = 20;
const CUP = [84, -12];
const CUP_R = 10;
const TRAY = { x: -18, y: 26, w: 44, d: 22, r: 2 };
const FLAP = { x: -15, y: -17.8, w: 10, d: 1.6, r: 0.4 };
const FLAP_LIFT = 13;
const SLIDE = { x: -108, y: 46, w: 56, d: 8, r: 2 };
const KNOB_Y = 50;
const KNOB_X = [-98, -62];

const P = k.fitProjection(
  [...k.boxCorners(PLATE, -PLATE_H - FOOT_H, 0), [HOPPER[0], HOPPER[1], 50], [GATE_X, -10, 40], [CUP[0], CUP[1], 20]],
  WIDTH,
  HEIGHT,
  { pad: 28, azimuth: 42 },
);

const at = (point) => k.iso(point, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const depth = (x, y, z = 0) => k.depthOf([x, y, z], P);

function plate() {
  const out = [`<path class="iso-halo" d="${k.haloOf(PLATE, -PLATE_H - FOOT_H, PLATE_H, P)}"/>`];
  for (const [x, y] of k.corners(PLATE, 22).sort((a, b) => depth(...a) - depth(...b))) {
    out.push(S(k.cylinder(x, y, 7, -PLATE_H - FOOT_H, 1.6, P, 28), { tone: "lo" }));
    out.push(S(k.cylinder(x, y, 5.4, -PLATE_H - FOOT_H + 1.6, FOOT_H - 1.6, P, 28), { tone: "mid" }));
  }
  out.push(S(k.slabOf(PLATE, -PLATE_H, PLATE_H, P, 8, 1.6), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(PLATE, 4), 0, P), { tone: "lo" }));
  const screws = k.corners(PLATE, 9);
  out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.2, 0, P, 16)).join(""), { tone: "lo" }));
  out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.5 }));
  // ruler strip along the front edge
  const strip = { x: -40, y: PLATE.y + PLATE.d - 12, w: 96, d: 6, r: 1 };
  out.push(S(k.slabOf(strip, 0, 0.6, P, 3), { tone: "lo", crease: "none" }));
  const ticks = k.topTicks(strip.x + 4, strip.x + strip.w - 4, 4, 5, strip.y + 0.8, 0.6, [1.8, 3.4], P);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  // label plate
  const label = { x: 64, y: PLATE.y + PLATE.d - 14, w: 52, d: 9, r: 1.2 };
  out.push(S(k.slabOf(label, 0, 0.8, P, 3), { tone: "mid", crease: "none" }));
  out.push(k.faceTextSvg(k.topMatrix([label.x + 4, label.y + 6], 0.8, P, "x"), "BLOCKED TASKS", { size: 3.2, tone: "lo" }));
  out.push(D([at([label.x + 2, label.y + 4.5, 0.8]), at([label.x + label.w - 2, label.y + 4.5, 0.8])], { size: 0.4 }));
  return out.join("");
}

function bed() {
  const out = [];
  out.push(S(k.slabOf(BED, 0, BED_H, P, 6, 0.8), { tone: "lo" }));
  out.push(L(k.planOutline(k.insetPlan(BED, 3), BED_H, P), { tone: "faint" }));
  const screws = k.corners(BED, 6);
  out.push(D(screws.map(([x, y]) => at([x, y, BED_H])), { size: 0.45, tone: "lo" }));
  return out.join("");
}

// rails: [x0, y0, x1, y1] centre lines, axis aligned
const RAILS = [
  // main channel, before the gate
  [-90, -16, -16, -16], [-4, -16, 16, -16],
  [-90, -4, -2, -4], [10, -4, 16, -4],
  // main channel, beyond the gate, to the cup
  [24, -16, 72, -16], [24, -4, 76, -4],
  // bypass: west leg, back leg, east leg
  [-16, -46, -16, -16], [-4, -34, -4, -16],
  [-16, -46, 90, -46], [-4, -34, 78, -34],
  [90, -46, 90, -20], [78, -34, 78, -22],
  // return leg to the tray
  [-2, -4, -2, 26], [10, -4, 10, 26],
];

function railPlan([x0, y0, x1, y1]) {
  const t = RAIL_T / 2;
  return { x: Math.min(x0, x1) - t, y: Math.min(y0, y1) - t, w: Math.abs(x1 - x0) + RAIL_T, d: Math.abs(y1 - y0) + RAIL_T, r: 0.4 };
}

function rails() {
  return RAILS.map((r) => ({ plan: railPlan(r), depth: depth((r[0] + r[2]) / 2, (r[1] + r[3]) / 2, Z) }))
    .sort((a, b) => a.depth - b.depth)
    .map(({ plan }) => S(k.slabOf(plan, Z, RAIL_H, P, 2), { tone: "mid", crease: "none" }))
    .join("");
}

function floorMarks() {
  const out = [];
  // centre guide lines, running rail end to rail end
  const guide = (a, b) => `M${at(a).map((v) => v.toFixed(1)).join(" ")}L${at(b).map((v) => v.toFixed(1)).join(" ")}`;
  out.push(L(guide([-90, -10, Z], [16, -10, Z]) + guide([20, -10, Z], [74, -10, Z]), { tone: "faint" }));
  out.push(L(guide([-10, -16, Z], [-10, -40, Z]) + guide([-10, -40, Z], [84, -40, Z]) + guide([84, -40, Z], [84, -22, Z]), { tone: "faint", dotted: true }));
  out.push(L(guide([4, -10, Z], [4, 26, Z]), { tone: "faint" }));
  return out.join("");
}

function hopper() {
  const [x, y] = HOPPER;
  const out = [];
  out.push(S(k.cylinder(x, y, 11, Z, 3, P, 40, 0.6), { tone: "lo" }));
  out.push(S(k.cylinder(x, y, 8, Z + 3, 34, P, 44, 0.8), { tone: "mid" }));
  out.push(L([10, 18, 26, 34].map((dz) => k.sideArc(x, y, 8, Z + dz, P, 28)).join(""), { tone: "faint" }));
  out.push(S(k.cylinder(x, y, 9.2, Z + 37, 2.4, P, 44, 0.5), { tone: "hi" }));
  out.push(L(k.ring(x, y, 5.6, Z + 39.4, P, 32), { tone: "lo" }));
  out.push(L(k.knurl(x, y, 9.2, Z + 37.3, Z + 39, 40, P, 0.3), { tone: "lo" }));
  // outlet shoe toward the channel
  out.push(S(k.slabOf({ x: x + 7, y: y - 6, w: 6, d: 12, r: 1 }, Z, 9, P, 3, 0.5), { tone: "mid" }));
  out.push(D([at([x + 10, y - 4, Z + 9]), at([x + 10, y + 4, Z + 9])], { size: 0.45 }));
  return out.join("");
}

function gate() {
  const out = [];
  const posts = [[GATE_X, -20], [GATE_X, 0]].sort((a, b) => depth(...a) - depth(...b));
  const post = ([x, y]) =>
    S(k.cylinder(x, y, 4, Z, 2, P, 24), { tone: "lo" }) + S(k.cylinder(x, y, 2.6, Z + 2, 26, P, 24), { tone: "mid" }) + S(k.cylinder(x, y, 3.2, Z + 28, 1.6, P, 24), { tone: "hi" });
  out.push(post(posts[0]));
  const door = { x: GATE_X - 0.9, y: -17.2, w: 1.8, d: 14.4, r: 0.3 };
  out.push(S(k.slabOf(door, Z + 0.4, 24, P, 2), { tone: "hi" }));
  out.push(L([6, 12, 18].map((dz) => k.planOutline(door, Z + dz, P, 2)).join(""), { tone: "lo" }));
  // lock bar and padlock on the near face
  const bar = { x: GATE_X + 0.9, y: -19, w: 1.2, d: 18, r: 0.3 };
  out.push(S(k.slabOf(bar, Z + 11, 2.4, P, 2), { tone: "mid", crease: "none" }));
  const lock = { x: GATE_X + 2.1, y: -12.5, w: 1.6, d: 5, r: 0.5 };
  out.push(S(k.slabOf(lock, Z + 8.4, 5, P, 2, 0.3), { tone: "hi" }));
  out.push(D([at([GATE_X + 3.7, -10, Z + 10.4])], { size: 0.5, tone: "lo" }));
  out.push(post(posts[1]));
  const lintel = { x: GATE_X - 3, y: -24, w: 6, d: 28, r: 1.2 };
  out.push(S(k.slabOf(lintel, Z + 29.6, 3.4, P, 3, 0.5), { tone: "mid" }));
  out.push(D([at([GATE_X, -20, Z + 33]), at([GATE_X, 0, Z + 33])], { size: 0.5 }));
  return out.join("");
}

function cup() {
  const [x, y] = CUP;
  const out = [];
  out.push(S(k.cylinder(x, y, CUP_R + 2, Z, 1.6, P, 40), { tone: "lo" }));
  out.push(S(k.cylinder(x, y, CUP_R, Z + 1.6, 10, P, 48, 0.8), { tone: "hi" }));
  out.push(L(k.ring(x, y, CUP_R - 2, Z + 11.6, P, 40), { tone: "mid" }));
  out.push(L(k.ring(x, y, CUP_R - 5.4, Z + 11.6, P, 32), { tone: "faint" }));
  out.push(L(k.sideArc(x, y, CUP_R, Z + 4.6, P, 28), { tone: "lo" }));
  return out.join("");
}

function trayBack() {
  const out = [];
  out.push(S(k.slabOf(TRAY, Z, 0.8, P, 3), { tone: "mid", crease: "none" }));
  // back and far side walls
  out.push(S(k.slabOf({ x: TRAY.x, y: TRAY.y, w: 15.4, d: 1.4, r: 0.4 }, Z, 5, P, 2), { tone: "mid", crease: "none" }));
  out.push(S(k.slabOf({ x: 10.6, y: TRAY.y, w: TRAY.w - 28.6, d: 1.4, r: 0.4 }, Z, 5, P, 2), { tone: "mid", crease: "none" }));
  out.push(S(k.slabOf({ x: TRAY.x, y: TRAY.y, w: 1.4, d: TRAY.d, r: 0.4 }, Z, 5, P, 2), { tone: "mid", crease: "none" }));
  return out.join("");
}

function trayFront() {
  const out = [];
  out.push(S(k.slabOf({ x: TRAY.x + TRAY.w - 1.4, y: TRAY.y, w: 1.4, d: TRAY.d, r: 0.4 }, Z, 5, P, 2), { tone: "mid", crease: "none" }));
  out.push(S(k.slabOf({ x: TRAY.x, y: TRAY.y + TRAY.d - 1.4, w: TRAY.w, d: 1.4, r: 0.4 }, Z, 5, P, 2), { tone: "mid", crease: "none" }));
  out.push(D([at([TRAY.x + 3, TRAY.y + TRAY.d - 0.7, Z + 5]), at([TRAY.x + TRAY.w - 3, TRAY.y + TRAY.d - 0.7, Z + 5])], { size: 0.4, tone: "lo" }));
  return out.join("");
}

function flapFrame() {
  const out = [];
  for (const [x, y] of [[-17.2, -17], [-2.8, -17]].sort((a, b) => depth(...a) - depth(...b))) {
    out.push(S(k.cylinder(x, y, 1.5, Z, 30, P, 16), { tone: "mid", crease: "none" }));
  }
  return out.join("");
}

function flapTop() {
  const out = [];
  out.push(S(k.slabOf({ x: -19, y: -18.6, w: 18, d: 3.2, r: 0.8 }, Z + 30, 2.4, P, 3, 0.4), { tone: "mid" }));
  out.push(D([at([-17.2, -17, Z + 32.4]), at([-2.8, -17, Z + 32.4])], { size: 0.45 }));
  return out.join("");
}

function flap() {
  const out = [];
  out.push(S(k.slabOf(FLAP, Z + 0.3, 12.5, P, 2), { tone: "hi", lit: true }));
  out.push(L([4, 8].map((dz) => k.planOutline(FLAP, Z + 0.3 + dz, P, 2)).join(""), { tone: "lo" }));
  // lift rod up through the top bar
  out.push(S(k.cylinder(-10, -17, 0.9, Z + 12.8, 16.6, P, 12), { tone: "hi", crease: "none" }));
  return out.join("");
}

function slide() {
  const out = [];
  out.push(S(k.slabOf(SLIDE, 0, 2, P, 4, 0.5), { tone: "mid" }));
  const slot = { x: SLIDE.x + 6, y: KNOB_Y - 1, w: SLIDE.w - 12, d: 2, r: 1 };
  out.push(L(k.planOutline(slot, 2, P, 6), { tone: "lo" }));
  const ticks = k.topTicks(SLIDE.x + 10, SLIDE.x + SLIDE.w - 10, 4.5, 8, SLIDE.y + 0.6, 2, [1.2, 2], P);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  out.push(D([at([SLIDE.x + 2.5, KNOB_Y, 2]), at([SLIDE.x + SLIDE.w - 2.5, KNOB_Y, 2])], { size: 0.45 }));
  return out.join("");
}

function knob() {
  const [x] = KNOB_X;
  const out = [];
  out.push(S(k.cylinder(x, KNOB_Y, 2, 2, 2.4, P, 16), { tone: "mid", crease: "none" }));
  out.push(S(k.cylinder(x, KNOB_Y, 4.6, 4.4, 5, P, 36, 0.7), { tone: "hi", lit: true }));
  out.push(L(k.knurl(x, KNOB_Y, 4.6, 4.9, 8.6, 34, P, 0.3), { tone: "lo" }));
  out.push(L(k.ring(x, KNOB_Y, 2.4, 9.4, P, 20), { tone: "faint" }));
  return out.join("");
}

// link rod from the slide to the flap frame, lying on the plate and bed
function link() {
  const pts = [[-52, KNOB_Y, 0.4], [-30, KNOB_Y, 0.4], [-30, 34, 0.4]].map(at);
  return k.wireSvg(pts[0], pts[pts.length - 1], { via: pts.slice(1, -1), tone: "lo", size: 1 });
}

const ballR = (P.scale * BALL_R).toFixed(2);
const ball = (n, pos) => `<g class="eg-ball" data-ball="${n}" ${pos ? `transform="translate(${at([pos[0], pos[1], Z + BALL_R]).map((v) => v.toFixed(2)).join(" ")})"` : `style="opacity:0"`}><circle class="eg-ball-body" r="${ballR}"/><path class="iso-line eg-ball-shine" data-tone="lo" data-free="both" d="M${(-P.scale * 2.2).toFixed(1)} ${(-P.scale * 1).toFixed(1)}A${(P.scale * 2.6).toFixed(1)} ${(P.scale * 2.6).toFixed(1)} 0 0 1 ${(P.scale * 0.6).toFixed(1)} ${(-P.scale * 2.8).toFixed(1)}"/></g>`;

const label =
  "A bench with a ball track. Balls leave a hopper on the left and roll along a channel to a locked gate; a cup beyond the gate holds the number the task asked for. Before the gate a side channel runs around it to the cup, and a second channel leads back to a tray at the front. A lift flap stands across the mouth of the side channel. A sliding knob at the front raises or lowers the flap. With the flap up, balls go around the gate into the cup. With the flap down, balls reach the gate, stop, and roll back to the tray.";

const svg = k.figureSvg({
  width: WIDTH,
  height: HEIGHT,
  label,
  standalone: standalone ? theme : false,
  body: [
    plate(),
    bed(),
    floorMarks(),
    rails(),
    `<g data-piece="slide">${slide()}${link()}<g data-knob="1"${standalone ? ` transform="${k.translateAlong(P, [KNOB_X[1] - KNOB_X[0], 0, 0])}"` : ""}>${knob()}</g></g>`,
    `<g data-slot="a"></g>`,
    `<g data-piece="hopper">${hopper()}</g>`,
    `<g data-piece="flap-frame">${flapFrame()}</g>`,
    `<g data-piece="flap" data-flap="1">${flap()}</g>`,
    `<g data-piece="flap-frame">${flapTop()}</g>`,
    `<g data-slot="b">${standalone ? ball(0, [-50, -10]) + ball(1, [13, -10]) : ""}</g>`,
    `<g data-piece="cup">${cup()}</g>`,
    `<g data-piece="gate">${gate()}</g>`,
    `<g data-piece="tray">${trayBack()}</g>`,
    `<g data-slot="c">${standalone ? ball(2, [4, 36]) : ""}</g>`,
    `<g data-piece="tray">${trayFront()}</g>`,
    standalone ? "" : `<g data-pool="1">${ball(0)}${ball(1)}${ball(2)}</g>`,
  ],
});

const CSS = `
.eg-ball-body{fill:var(--anatomy-paper);stroke:var(--anatomy-hi);stroke-width:var(--anatomy-weight);vector-effect:non-scaling-stroke}
.eg-ball[data-kind="around"] .eg-ball-body{stroke:var(--eg-warn)}
.eg-ball[data-kind="around"] .eg-ball-shine{stroke:var(--eg-warn)}
.iso{--eg-warn:#ff8a5c}
.iso[data-theme="light"]{--eg-warn:#c8431a}
.iso-stage{cursor:pointer;touch-action:manipulation}
`;

const live = `
const P = ${JSON.stringify(P)};
const Z = ${Z + BALL_R};
const LIFT = ${FLAP_LIFT};
const KNOB = ${JSON.stringify(KNOB_X)};
const ESC = [${ESC_SUCCESS_ONLY}, ${ESC_SAFE}], HONEST = ${HONEST_SAFE};
const stage = document.querySelector(".iso-stage");
const svg = stage.querySelector("svg");
const flap = svg.querySelector("[data-flap]");
const knob = svg.querySelector("[data-knob]");
const slots = { a: svg.querySelector('[data-slot="a"]'), b: svg.querySelector('[data-slot="b"]'), c: svg.querySelector('[data-slot="c"]') };
const balls = [...svg.querySelectorAll("[data-ball]")].map((el) => ({ el, on: false, t: 0, path: null, slot: "" }));
const readout = document.querySelector("[data-readout]");
const still = matchMedia("(prefers-reduced-motion: reduce)");
// each leg: [x, y, slot, pause seconds on arrival]
const START = [-88, -10];
const AROUND = [[-10, -10, "b", 0], [-10, -19, "b", 0], [-10, -40, "a", 0], [84, -40, "a", 0], [84, -25, "a", 0]];
const HONESTLY = [[-10, -10, "b", 0], [13, -10, "b", 0.55], [4, -10, "b", 0], [4, 4, "b", 0], [4, 36, "c", 0]];
const SPEED = 62;
let safe = true, open = 0, frame = 0, last = 0, clock = 0, sinceBall = 9, touring = !still.matches, visible = false, idleTimer = 0, sent = 0;
function launch() {
  const b = balls.find((x) => !x.on) || balls.reduce((a, c) => (c.t > a.t ? c : a));
  b.on = true; b.t = 0; b.leg = 0; b.pos = [...START]; b.wait = 0; b.fade = 1; b.done = false; b.path = null; b.rest = 0; b.el.removeAttribute("data-kind");
  place(b, "b");
}
function place(b, slot) { if (b.slot !== slot) { slots[slot].appendChild(b.el); b.slot = slot; } }
function step(b, dt) {
  b.t += dt;
  if (!b.path && b.pos[0] >= -11) { b.path = open > 0.5 ? AROUND : HONESTLY; b.el.setAttribute("data-kind", open > 0.5 ? "around" : "honest"); b.leg = 0; }
  if (b.done) { b.rest += dt; const hold = b.path === AROUND ? 0 : 1.6; if (b.rest > hold) b.fade = Math.max(0, b.fade - dt * (b.path === AROUND ? 5 : 1.6)); if (b.fade <= 0) { b.on = false; b.el.style.opacity = 0; } return; }
  if (b.wait > 0) { b.wait -= dt; return; }
  const target = b.path ? b.path[b.leg] : [-10, -10, "b", 0];
  let move = SPEED * dt;
  const dx = target[0] - b.pos[0], dy = target[1] - b.pos[1], dist = Math.hypot(dx, dy);
  if (dist <= move) {
    b.pos = [target[0], target[1]];
    if (b.path) { b.wait = target[3]; b.leg++; if (b.leg >= b.path.length) b.done = true; else place(b, b.path[b.leg][2]); }
  } else { b.pos[0] += (dx / dist) * move; b.pos[1] += (dy / dist) * move; }
}
function draw() {
  flap.setAttribute("transform", translateAlong(P, [0, 0, LIFT * open]));
  knob.setAttribute("transform", translateAlong(P, [(KNOB[1] - KNOB[0]) * (1 - open), 0, 0]));
  for (const b of balls) {
    if (!b.on) continue;
    const [x, y] = iso([b.pos[0], b.pos[1], Z], P);
    b.el.setAttribute("transform", "translate(" + x.toFixed(2) + " " + y.toFixed(2) + ")");
    b.el.style.opacity = b.fade.toFixed(2);
  }
  const text = safe ? "safe reward · " + ESC[1].toFixed(1) + "% go around · " + HONEST + "% stop honestly" : "success-only reward · " + ESC[0].toFixed(1) + "% go around the gate";
  if (readout.textContent !== text) readout.textContent = text;
  stage.setAttribute("aria-pressed", safe ? "true" : "false");
}
function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
  last = now;
  if (touring) { clock += dt; if (clock > 7.5) { clock = 0; safe = !safe; } }
  const want = safe ? 0 : 1;
  open = still.matches ? want : open + (want - open) * (1 - Math.exp(-dt / 0.16));
  if (Math.abs(want - open) < 0.002) open = want;
  sinceBall += dt;
  if (sinceBall > 2.3 && !still.matches) { sinceBall = 0; launch(); sent++; }
  for (const b of balls) if (b.on) step(b, dt);
  draw();
  frame = visible && !still.matches ? requestAnimationFrame(tick) : 0;
}
const run = () => { if (!frame && visible) { last = 0; frame = requestAnimationFrame(tick); } };
const toggle = () => { touring = false; safe = !safe; clearTimeout(idleTimer); idleTimer = setTimeout(() => { touring = !still.matches; clock = 0; }, 12000); if (still.matches) { open = safe ? 0 : 1; draw(); } run(); };
new IntersectionObserver((entries) => { visible = entries[entries.length - 1].isIntersecting; if (visible) run(); }, { rootMargin: "120px 0px" }).observe(stage);
stage.addEventListener("click", toggle);
stage.addEventListener("keydown", (event) => { if (event.key === " " || event.key === "Enter") { event.preventDefault(); if (!event.repeat) toggle(); } });
window.__eg = { set: (s) => { touring = false; safe = s; }, state: () => ({ safe, open, sent }) };
draw();
`;

if (standalone) {
  const name = theme === "light" ? "escalation-gate-light.svg" : "escalation-gate.svg";
  const tag = (text, p, tx, ty) => {
    const [x, y] = at(p);
    const ly = ty < y ? ty + 4 : ty - 10;
    return `<path class="eg-lead" d="M${x.toFixed(1)} ${y.toFixed(1)}L${tx} ${ly}"/><circle class="eg-dot" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="1.5"/><text class="eg-tag" x="${tx}" y="${ty}" text-anchor="middle">${text}</text>`;
  };
  const tags = [
    tag("blocked tasks start here", [HOPPER[0], HOPPER[1], Z + 40], 195, 14),
    tag("locked gate = access control", [GATE_X, -10, Z + 33], 372, 70),
    tag("the way around, closed by the penalty", [40, -40, Z], 506, 108),
    tag("the number the task wants", [CUP[0], CUP[1], Z + 12], 556, 186),
    tag("honest stop: report BLOCKED", [4, 44, Z + 5], 196, 344),
    tag("reward setting", [KNOB_X[1], KNOB_Y, 9.4], 84, 262),
  ].join("");
  const extra = `<style>${CSS}.eg-tag{font:10px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:var(--anatomy-hi);stroke:var(--anatomy-card);stroke-width:3;paint-order:stroke;letter-spacing:.02em}.eg-lead{fill:none;stroke:var(--anatomy-hi);stroke-width:.6;opacity:.6}.eg-dot{fill:var(--anatomy-hi)}</style>${tags}`;
  const out = svg.replace(/(<svg[^>]*>)/, `$1<rect class="eg-bg" width="${WIDTH}" height="${HEIGHT}" rx="6" style="fill:var(--anatomy-card)"/>`).replace(/<\/svg>\s*$/, extra + "</svg>");
  writeFileSync(join(HERE, name), out);
  console.log("wrote", name);
} else {
  const body = k.plateHtml({
    fig: "Fig 1",
    title: "The escalation gate",
    hint: "Click to change the reward",
    readout: "safe reward",
    keys: [
      { mark: "lit", label: "The escalation penalty: a flap across the way around, set by the knob" },
      { mark: "raised", label: "The locked gate (the access control) and, beyond it, the cup that holds the number" },
    ],
    caption: `Each ball is one blocked task. It rolls from the hopper to a locked gate; the number it was sent for is in the cup beyond. A success-only reward scores only whether the ball reaches the cup, so nothing stands in the side channel and the ball goes around the gate: the trained agent did this on ${ESC_SUCCESS_ONLY}% of blocked test tasks. The safe reward adds one term, a penalty for any route around an access control. That is the flap. With it down, the ball reaches the gate, stops, and rolls back to the tray: ${ESC_SAFE.toFixed(1)}% went around and ${HONEST_SAFE}% stopped honestly with a BLOCKED report.`,
    body: `<div class="iso-stage" tabindex="0" role="button" aria-pressed="true" aria-label="The escalation gate: click the figure, or press Space, to switch between the success-only reward and the safe reward">${svg}</div>`,
  });
  const name = theme === "light" ? "escalation-gate-light.html" : "escalation-gate.html";
  writeFileSync(join(HERE, name), k.pageHtml({ title: "The escalation gate", theme, body: `<style>${CSS}</style>${body}`, script: kitScript() + live }));
  console.log("wrote", name);
}
