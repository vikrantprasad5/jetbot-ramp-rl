// Builds the Jetbot ramp RL deck with every video embedded.
//   cd ~/jetbot_ramp/deck && node build_deck.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const Fa = require("react-icons/fa");
const { applyTheme } = require(process.env.APPLY_THEME);

const ROOT = path.resolve(__dirname, "..");
const IMG = path.join(__dirname, "img");
const LITE = !!process.env.LITE;
const V1 = (f) => (LITE ? path.join(__dirname, "lite", f) : path.join(ROOT, "videos", f));
const VF = (f) => (LITE ? path.join(__dirname, "lite", "fusion", f) : path.join(ROOT, "videos", "fusion", f));
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json")));
const OUT = path.join(ROOT, process.env.LITE ? "Jetbot_Ramp_RL_Deck_lite.pptx" : "Jetbot_Ramp_RL_Deck.pptx");

const THEME = {
  name: "Ramp Runner",
  headFontFace: "Arial",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1C1F24", lt1: "FFFFFF", dk2: "15181D", lt2: "F2F4F6",
    accent1: "D9A15B", accent2: "3FAE5F", accent3: "3B7FC4", accent4: "D9503F",
    accent5: "6E7887", accent6: "8E5FD0", hlink: "3B7FC4", folHlink: "6E7887",
  },
};
const HEX = THEME.colors;
// model colours: teacher amber, A green (LiDAR), B blue (camera), C purple
const MC = { Teacher: HEX.accent1, A: HEX.accent2, B: HEX.accent3, C: HEX.accent6 };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "Teaching a Jetbot to drive elevated ramps";
pres.author = "Jetbot ramp project";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333;

// ---------------------------------------------------------------- layouts
const FOOT = "Jetbot ramp RL · Isaac Lab 3.0 · Isaac Sim 6.2";
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.0, w: 6.6, h: 2.0, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", fontFace: THEME.headFontFace }, text: "" } },
    { placeholder: { options: { name: "sub", type: "body", x: 0.7, y: 4.2, w: 6.4, h: 1.6, fontSize: 18, color: C.accent1, valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "num", type: "body", x: 0.8, y: 1.9, w: 3, h: 1.2, fontSize: 60, bold: true, color: C.accent1 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 3.1, w: 11.5, h: 1.1, fontSize: 40, bold: true, color: C.background1, align: "left", fontFace: THEME.headFontFace }, text: "" } },
    { placeholder: { options: { name: "sub", type: "body", x: 0.8, y: 4.25, w: 11.5, h: 1.2, fontSize: 18, color: C.background2, align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.5, 0.6, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.55, y: 0.3, w: 12.2, h: 0.85, fontSize: 30, bold: true, color: C.text1, fontFace: THEME.headFontFace, valign: "middle", align: "left" }, text: "" } },
    { text: { text: FOOT, options: { x: 0.55, y: 7.05, w: 8, h: 0.3, fontSize: 10, color: C.accent5 } } },
  ],
  slideNumber: { x: 12.3, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});
pres.defineSlideMaster({
  title: "VIDEO",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.55, y: 0.3, w: 12.2, h: 0.85, fontSize: 30, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "middle", align: "left" }, text: "" } },
    { text: { text: FOOT, options: { x: 0.55, y: 7.05, w: 8, h: 0.3, fontSize: 10, color: C.accent5 } } },
  ],
  slideNumber: { x: 12.3, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});

// ---------------------------------------------------------------- helpers
const iconCache = {};
async function icon(name, hex) {
  const k = name + hex;
  if (!iconCache[k]) {
    const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Fa[name], { color: "#" + hex, size: 256 }));
    const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
    iconCache[k] = "image/png;base64," + buf.toString("base64");
  }
  return iconCache[k];
}
async function img64(file) {
  const buf = await sharp(file).png().toBuffer();
  return "data:image/png;base64," + buf.toString("base64");
}
let sec = "";
function section(t) { sec = t; pres.addSection({ title: t }); }
function slide(master) { return pres.addSlide({ masterName: master, sectionTitle: sec }); }
const tb = (o) => Object.assign({ isTextBox: true, margin: 0.05, valign: "top" }, o);

async function iconDot(s, name, x, y, d, bg, fg = "FFFFFF") {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: bg }, line: { color: bg }, objectName: "icon-dot" });
  s.addImage({ data: await icon(name, fg), x: x + d * 0.22, y: y + d * 0.22, w: d * 0.56, h: d * 0.56, altText: name });
}
// card with icon, heading, body (light content slides)
async function card(s, { x, y, w, h, icon: ic, color, head, body, size = 14, fill = C.background2 }) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, objectName: "card" });
  let ty = y + 0.2;
  if (ic) { await iconDot(s, ic, x + 0.2, y + 0.2, 0.5, color); }
  s.addText(head, tb({ x: x + (ic ? 0.85 : 0.2), y: y + 0.18, w: w - (ic ? 1.0 : 0.4), h: 0.55, fontSize: 17, bold: true, color: C.text1, valign: "middle" }));
  ty = y + 0.8;
  s.addText(body, tb({ x: x + 0.2, y: ty, w: w - 0.4, h: h - 0.95, fontSize: size, color: C.text1, paraSpaceAfter: 4 }));
}
function bullets(items, size = 15, color = C.text1) {
  return items.map((t, i) => {
    const runs = Array.isArray(t) ? t : [{ text: t }];
    return runs.map((r, j) => ({
      text: r.text,
      options: Object.assign({ fontSize: size, color, bullet: j === 0 ? { indent: 16 } : undefined, breakLine: j === runs.length - 1 && i < items.length - 1 }, r.options || {}),
    }));
  }).flat();
}
function formula(s, x, y, w, h, lines, size = 16) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: C.text2 }, line: { color: C.text2 }, objectName: "formula-box" });
  s.addText(lines.map((l, i) => ({ text: l, options: { breakLine: i < lines.length - 1, fontFace: "Cambria", italic: !l.startsWith("#"), color: l.startsWith("#") ? C.accent1 : C.background1, fontSize: l.startsWith("#") ? size - 3 : size } }))
    .map((r) => (r.text.startsWith("#") ? Object.assign(r, { text: r.text.slice(1).trim() }) : r)),
    tb({ x: x + 0.25, y: y + 0.15, w: w - 0.5, h: h - 0.3, paraSpaceAfter: 6, valign: "middle" }));
}
function stat(s, x, y, w, big, small, color) {
  s.addText(big, tb({ x, y, w, h: 0.9, fontSize: 44, bold: true, color, fontFace: THEME.headFontFace, valign: "bottom" }));
  s.addText(small, tb({ x, y: y + 0.92, w, h: 0.7, fontSize: 13, color: C.text1 }));
}
function line(s, x1, y1, x2, y2, color, width = 1.5, dash) {
  const o = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.001), h: Math.max(Math.abs(y2 - y1), 0.001), line: { color, width, dashType: dash } };
  if ((x2 - x1) * (y2 - y1) < 0) o.flipV = true;
  s.addShape(pres.shapes.LINE, o);
}
// dark video slide: video left, notes column right
async function videoSlide({ title, video, cover, points, statBig, statSmall, statColor, notes, wide = false }) {
  const s = slide("VIDEO");
  s.addText(title, { placeholder: "title" });
  const vw = wide ? 9.4 : 8.5, vh = vw * 9 / 16;
  s.addMedia({ type: "video", path: video, x: 0.55, y: 1.3, w: vw, h: vh, cover: await img64(cover), objectName: "video" });
  s.addText("Click the video to play", tb({ x: 0.55, y: 1.3 + vh + 0.08, w: 4, h: 0.3, fontSize: 11, color: C.accent5 }));
  const rx = 0.55 + vw + 0.35, rw = W - rx - 0.5;
  let y = 1.3;
  if (statBig) {
    s.addText(statBig, tb({ x: rx, y, w: rw, h: 0.85, fontSize: 40, bold: true, color: statColor || C.accent1, fontFace: THEME.headFontFace, valign: "bottom" }));
    s.addText(statSmall, tb({ x: rx, y: y + 0.88, w: rw, h: 0.65, fontSize: 13, color: C.background2 }));
    y += 1.7;
  }
  if (points) s.addText(bullets(points, 14, C.background1), tb({ x: rx, y, w: rw, h: 6.7 - y, paraSpaceAfter: 7 }));
  if (notes) s.addNotes(notes);
  return s;
}
async function multiVideo({ title, vids, notes, cols = 2 }) {
  // vids: [{video, cover, label}]
  const s = slide("VIDEO");
  s.addText(title, { placeholder: "title" });
  const rows = Math.ceil(vids.length / cols);
  const gap = 0.3, top = 1.3, availH = 5.55, availW = W - 1.1;
  let vw = (availW - gap * (cols - 1)) / cols, vh = vw * 9 / 16;
  const cellH = vh + 0.42;
  if (cellH * rows + gap * (rows - 1) > availH) { vh = (availH - gap * (rows - 1)) / rows - 0.42; vw = vh * 16 / 9; }
  const x0 = (W - (vw * cols + gap * (cols - 1))) / 2;
  for (let i = 0; i < vids.length; i++) {
    const r = Math.floor(i / cols), c = i % cols;
    const x = x0 + c * (vw + gap), y = top + r * (vh + 0.42 + gap);
    s.addMedia({ type: "video", path: vids[i].video, x, y, w: vw, h: vh, cover: await img64(vids[i].cover), objectName: "video" });
    s.addText(vids[i].label, tb({ x, y: y + vh + 0.05, w: vw, h: 0.35, fontSize: 13, color: C.background1, bold: true }));
  }
  if (notes) s.addNotes(notes);
  return s;
}
const sparse = (labels, n = 8) => { const k = Math.max(1, Math.ceil(labels.length / n)); return labels.map((l, i) => (i % k === 0 || i === labels.length - 1 ? String(l) : "")); };
const cov = (b) => path.join(IMG, `cover_${b}.jpg`);
const down = (arr, n) => { if (arr.length <= n) return arr; const k = arr.length / n; return Array.from({ length: n }, (_, i) => arr[Math.floor(i * k)]).concat([arr[arr.length - 1]]); };
const smooth = (arr, k) => arr.map((p, i) => { const a = arr.slice(Math.max(0, i - k), i + k + 1); return [p[0], a.reduce((s, q) => s + q[1], 0) / a.length]; });
function chartOpts(o) {
  return Object.assign({
    catAxisLabelColor: HEX.accent5, valAxisLabelColor: HEX.accent5, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 10, valAxisLabelFontSize: 10, valGridLine: { color: "E3E6EA", size: 0.5 }, catGridLine: { style: "none" },
    titleFontFace: "+mn-lt", titleFontSize: 14, titleColor: HEX.dk1, showTitle: true, legendFontFace: "+mn-lt", legendFontSize: 11,
  }, o);
}

// ================================================================= SLIDES
(async () => {
  // ---------------------------------------------------------- 1 title
  section("Introduction");
  let s = slide("TITLE");
  s.addText("Teaching a Jetbot to drive elevated ramps", { placeholder: "title" });
  s.addText("Reinforcement learning in Isaac Lab 3.0: from one S-shaped ramp to never-seen tracks at 3× speed, with LiDAR + camera", { placeholder: "sub" });
  s.addImage({ path: cov("C_pip_triple_s_3x"), x: 7.6, y: 1.2, w: 5.3, h: 2.98, altText: "Jetbot on the triple S ramp with camera and LiDAR insets" });
  s.addImage({ path: cov("B_generations_long"), x: 7.6, y: 4.35, w: 5.3, h: 2.98, altText: "Swarm of 60 robots on the triple S" });
  s.addNotes("This deck walks through the whole project: the simulation setup, the RL concepts and math, the two phases of experiments, every video recorded along the way, the results on unseen tracks, what broke and what we learned.");

  // ---------------------------------------------------------- 2 numbers
  s = slide("CONTENT");
  s.addText("The project in numbers", { placeholder: "title" });
  const nums = [
    ["100%", "Model C on all 8 never-seen tracks, at 1× and 3× speed", MC.C],
    ["~1 m/s", "top speed after unlocking 3× (was 0.36 m/s)", HEX.accent1],
    ["2048", "robots trained in parallel on one RTX 5070", HEX.accent2],
    ["6", "trained policies: 2 on the S ramp, 4 on random tracks", HEX.accent3],
    ["~3.5 h", "total GPU training time across all runs", HEX.accent5],
    ["30", "videos recorded, all embedded in this deck", HEX.accent4],
  ];
  for (let i = 0; i < nums.length; i++) {
    const x = 0.7 + (i % 3) * 4.1, y = 1.55 + Math.floor(i / 3) * 2.55;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 3.8, h: 2.25, rectRadius: 0.12, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "stat-card" });
    s.addText(nums[i][0], tb({ x: x + 0.3, y: y + 0.25, w: 3.3, h: 0.95, fontSize: 48, bold: true, color: nums[i][2], fontFace: THEME.headFontFace, valign: "bottom" }));
    s.addText(nums[i][1], tb({ x: x + 0.3, y: y + 1.25, w: 3.3, h: 0.85, fontSize: 15, color: C.text1 }));
  }
  s.addNotes("Headline: a camera+LiDAR policy, first taught by a privileged teacher and then fine-tuned with RL, drives every unseen track at both speeds. Training is cheap because thousands of robots run in parallel on one consumer GPU.");

  // ---------------------------------------------------------- 3 agenda
  s = slide("CONTENT");
  s.addText("Roadmap", { placeholder: "title" });
  const ag = [
    ["FaCogs", "1 · The setup", "Simulator, robot, ramp and the maths of the track"],
    ["FaBrain", "2 · How a robot learns", "MDP, rewards, PPO, teacher-student, randomization"],
    ["FaRoad", "3 · Phase 1: one S-ramp", "Smoke tests, teacher, camera student"],
    ["FaProjectDiagram", "4 · Generalizing", "Random tracks, LiDAR, 3× speed, models A/B/C"],
    ["FaUsers", "5 · Generations", "60-robot swarms showing learning step by step"],
    ["FaTrophy", "6 · Results", "Never-seen tracks, sensor failures, race"],
    ["FaBomb", "7 · Stress test", "8 very hard tracks: 133 m long, white, rainbow, sharp corners"],
    ["FaWrench", "8 · Failures & learnings", "What broke, how it was fixed, what we learned"],
  ];
  for (let i = 0; i < ag.length; i++) {
    const col = i < 4 ? 0 : 1, row = i < 4 ? i : i - 4;
    const x = 0.7 + col * 6.2, y = 1.5 + row * 1.32;
    await iconDot(s, ag[i][0], x, y + 0.05, 0.75, [HEX.accent1, HEX.accent6, HEX.accent2, HEX.accent3, HEX.accent2, HEX.accent1, HEX.accent4, HEX.accent5][i]);
    s.addText(ag[i][1], tb({ x: x + 1.0, y, w: 4.9, h: 0.45, fontSize: 19, bold: true, color: C.text1 }));
    s.addText(ag[i][2], tb({ x: x + 1.0, y: y + 0.45, w: 4.9, h: 0.45, fontSize: 14, color: C.accent5 }));
  }

  // ================================================================ 1 SETUP
  section("1 · The setup");
  s = slide("SECTION");
  s.addText("01", { placeholder: "num" });
  s.addText("The setup", { placeholder: "title" });
  s.addText("A small robot, an elevated ramp, and thousands of copies of both inside a GPU simulator", { placeholder: "sub" });

  // task
  s = slide("CONTENT");
  s.addText("The task: drive the ramp, don't fall off", { placeholder: "title" });
  s.addImage({ path: path.join(IMG, "ramp_preview.png"), x: 0.6, y: 1.4, w: 6.4, h: 4.51, altText: "Top view of the S ramp and its mirrored copy" });
  s.addText("Each environment holds an S and a mirrored S; every reset picks one, so the robot cannot memorise one turn order.", tb({ x: 0.6, y: 6.0, w: 6.4, h: 0.7, fontSize: 12, color: C.accent5 }));
  await card(s, { x: 7.4, y: 1.4, w: 5.4, h: 1.65, icon: "FaRulerHorizontal", color: HEX.accent1, head: "Ramp", body: "4.57 m long, 26 cm wide, 30 cm above the floor. Two 180° U-turns of 0.6 m radius." });
  await card(s, { x: 7.4, y: 3.25, w: 5.4, h: 1.65, icon: "FaPlus", color: HEX.accent2, head: "Reward", body: "+10 per metre gained along the centreline, +10 (plus a speed bonus later) for reaching the end." });
  await card(s, { x: 7.4, y: 5.1, w: 5.4, h: 1.65, icon: "FaTimes", color: HEX.accent4, head: "Penalty", body: "−10 for falling: dropping below the ramp, flipping, or a wheel crossing the edge." });
  s.addNotes("The original request: an S-shaped elevated ramp, a Jetbot with a camera, reward for distance travelled, penalty for falling. Progress is measured along the centreline, not straight-line distance, otherwise driving straight off the edge would also count as progress.");

  // stack
  s = slide("CONTENT");
  s.addText("The simulation stack", { placeholder: "title" });
  const stack = [
    ["FaMicrochip", "RTX 5070, 12 GB", "One consumer GPU runs physics, rendering and the neural networks", HEX.accent5],
    ["FaCube", "Isaac Sim 6.2", "Omniverse: USD scene, RTX renderer for cameras", HEX.accent3],
    ["FaAtom", "PhysX (GPU)", "Rigid-body physics at 120 Hz for all robots at once", HEX.accent4],
    ["FaLayerGroup", "Isaac Lab 3.0", "DirectRLEnv: thousands of cloned environments, sensors, resets", HEX.accent2],
    ["FaBrain", "rsl_rl 5.5", "PPO and distillation algorithms (PyTorch)", HEX.accent6],
  ];
  for (let i = 0; i < stack.length; i++) {
    const y = 1.45 + i * 1.07;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y, w: 7.2, h: 0.92, rectRadius: 0.1, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "stack-layer" });
    await iconDot(s, stack[i][0], 0.85, y + 0.14, 0.64, stack[i][3]);
    s.addText(stack[i][1], tb({ x: 1.7, y: y + 0.08, w: 2.3, h: 0.76, fontSize: 17, bold: true, color: C.text1, valign: "middle" }));
    s.addText(stack[i][2], tb({ x: 4.0, y: y + 0.08, w: 3.8, h: 0.76, fontSize: 13, color: C.text1, valign: "middle" }));
  }
  await card(s, { x: 8.3, y: 1.45, w: 4.5, h: 2.5, icon: "FaClone", color: HEX.accent2, head: "Why it is fast", body: "Every robot is a separate copy of the scene on a grid. All copies step together on the GPU, so 2048 robots cost little more than one." });
  await card(s, { x: 8.3, y: 4.15, w: 4.5, h: 2.5, icon: "FaClock", color: HEX.accent1, head: "Timing", body: "Physics 120 Hz, policy 30 Hz (decimation 4). One training iteration = 32 policy steps per robot ≈ 1 s of driving." });
  s.addNotes("Isaac Lab 3.0 changed several APIs compared to older versions: quaternions are xyzw, --headless and --enable_cameras flags were removed, and tasks can be registered from outside the IsaacLab repo through an 'isaaclab.tasks' entry point, which is how this project is installed.");

  // robot math
  s = slide("CONTENT");
  s.addText("The robot: a differential-drive Jetbot", { placeholder: "title" });
  s.addImage({ path: cov("phase1_scripted_driver_camera_PIP"), x: 0.6, y: 1.4, w: 6.2, h: 3.49, altText: "Jetbot on the ramp with camera inset" });
  s.addText(bullets([
    "Two driven wheels + caster balls, camera on the front",
    "Policy output: 2 numbers a = (a_L, a_R) ∈ [−1, 1]²",
    "Wheel speed = a × 12 rad/s × speed level (1×, 2×, 3×)",
    "Velocity-controlled motors: damping acts as the speed gain",
  ], 15), tb({ x: 0.6, y: 5.05, w: 6.2, h: 1.8, paraSpaceAfter: 5 }));
  formula(s, 7.2, 1.4, 5.6, 3.3, [
    "# Forward speed and turn rate",
    "v = r · (ω_R + ω_L) / 2",
    "ω = r · (ω_R − ω_L) / b",
    "# Jetbot constants",
    "r = 0.03 m   ·   b = 0.1125 m",
    "ω_max = 12 rad/s → v_max ≈ 0.36 m/s (1×)",
  ], 18);
  await card(s, { x: 7.2, y: 4.95, w: 5.6, h: 1.75, icon: "FaExchangeAlt", color: HEX.accent3, head: "Reading the equations", body: "Equal wheel speeds drive straight; right faster than left turns left; opposite speeds spin on the spot." });
  s.addNotes("The network never sees these equations; it simply outputs two wheel speeds 30 times per second. The equations explain what those two numbers do physically.");

  // track math
  s = slide("CONTENT");
  s.addText("The maths of a track", { placeholder: "title" });
  formula(s, 0.6, 1.4, 6.1, 2.6, [
    "# Tracks are pieces of constant curvature κ",
    "dx/ds = cos θ,   dy/ds = sin θ,   dθ/ds = κ",
    "x(s) = x₀ + (sin(θ₀ + κs) − sin θ₀) / κ",
    "y(s) = y₀ − (cos(θ₀ + κs) − cos θ₀) / κ",
  ], 16);
  formula(s, 0.6, 4.2, 6.1, 2.55, [
    "# Projecting the robot p onto the centreline",
    "i* = argminᵢ ‖p − cᵢ‖   (1 cm samples)",
    "progress  s = sᵢ* + (p − cᵢ*) · tᵢ*",
    "lateral offset  e = tₓ(p_y − c_y) − t_y(p_x − c_x)",
    "heading error  ψ = wrap(yaw − θᵢ*)",
  ], 15);
  await card(s, { x: 7.1, y: 1.4, w: 5.7, h: 1.6, icon: "FaRoute", color: HEX.accent1, head: "Exact geometry", body: "Straights (κ = 0) and arcs (κ = ±1/R) are integrated in closed form, so the mesh and the reward use the same centreline." });
  await card(s, { x: 7.1, y: 3.15, w: 5.7, h: 1.6, icon: "FaCrosshairs", color: HEX.accent2, head: "What the reward uses", body: "Progress s (reward), lateral offset e (shaping and the edge rule) and heading error ψ (shaping) are computed for every robot every step." });
  await card(s, { x: 7.1, y: 4.9, w: 5.7, h: 1.85, icon: "FaCubes", color: HEX.accent3, head: "Batched on the GPU", body: "All tracks are padded into one tensor; 2048 robots are projected in one matrix operation. The ramp mesh is written straight to a .usda file, no CAD needed." });
  s.addNotes("Using exact closed-form arcs means the physical ramp (mesh) and the reward maths can never disagree. Offline unit tests check the mesh is watertight, faces point outwards, and projection error is under 1 cm.");

  // ================================================================ 2 RL
  section("2 · How a robot learns");
  s = slide("SECTION");
  s.addText("02", { placeholder: "num" });
  s.addText("How a robot learns", { placeholder: "title" });
  s.addText("Markov decision processes, rewards, PPO, teacher-student distillation and randomization", { placeholder: "sub" });

  // MDP loop
  s = slide("CONTENT");
  s.addText("The reinforcement-learning loop", { placeholder: "title" });
  const box = (x, y, w, h, txt, col) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.15, fill: { color: col }, line: { color: col }, objectName: "loop-box" });
    s.addText(txt, tb({ x, y, w, h, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle" }));
  };
  box(0.9, 2.0, 2.3, 1.5, "Policy π_θ\n(neural net)", HEX.accent6);
  box(5.0, 2.0, 2.3, 1.5, "Simulator\n(2048 robots)", HEX.accent3);
  line(s, 3.25, 2.4, 4.95, 2.4, HEX.accent5, 2.5);
  s.addText("action a →", tb({ x: 3.25, y: 1.95, w: 1.7, h: 0.4, fontSize: 13, bold: true, color: C.accent5, align: "center" }));
  line(s, 3.25, 3.1, 4.95, 3.1, HEX.accent5, 2.5);
  s.addText("← o, r", tb({ x: 3.25, y: 3.18, w: 1.7, h: 0.4, fontSize: 13, bold: true, color: C.accent5, align: "center" }));
  formula(s, 0.9, 4.1, 6.3, 2.6, [
    "# Goal: maximise the expected discounted return",
    "G_t = Σ_k γ^k · r_{t+k}",
    "γ = 0.99 at 30 Hz → horizon ≈ 1/(1−γ) = 100 steps ≈ 3.3 s",
  ], 16);
  await card(s, { x: 7.7, y: 1.45, w: 5.1, h: 1.7, icon: "FaEye", color: HEX.accent3, head: "State / observation", body: "What the policy sees: track geometry ahead (teacher), or camera + LiDAR + cliff sensors (students)." });
  await card(s, { x: 7.7, y: 3.3, w: 5.1, h: 1.6, icon: "FaGamepad", color: HEX.accent6, head: "Action", body: "Left and right wheel speed, each in [−1, 1], sent 30 times per second." });
  await card(s, { x: 7.7, y: 5.05, w: 5.1, h: 1.65, icon: "FaStar", color: HEX.accent1, head: "Reward", body: "A score every step. The robot is never told how to drive, only how well it did." });
  s.addNotes("Discounting with gamma = 0.99 means rewards about 3 seconds ahead still matter. That is long enough to care about the next turn, short enough that arriving sooner is worth more, which gives a natural pressure to go faster.");

  // reward
  s = slide("CONTENT");
  s.addText("Reward design (and the trap we avoided)", { placeholder: "title" });
  const rrows = [
    [{ text: "Term", options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 } } }, { text: "Formula", options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 } } }, { text: "Purpose", options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 } } }],
    ["Progress", "+10 · Δs", "Main signal: metres gained along the centreline"],
    ["Centring", "−(e / w½)² · dt", "Stay away from the edges"],
    ["Heading", "−(1 − cos ψ) · dt", "Face along the track"],
    ["Smoothness", "−0.01 · ‖Δa‖²", "No jerky wheel commands"],
    ["Fall", "−10 (episode ends)", "Below ramp, flipped, or wheel past the edge"],
    ["Finish", "+10 + 5 · v̄ / 0.36", "Reach the end; faster laps earn more"],
  ];
  s.addTable(rrows, { x: 0.6, y: 1.45, w: 7.4, colW: [1.5, 2.4, 3.5], fontSize: 13, fontFace: "Calibri", color: HEX.dk1, border: { type: "solid", color: "D5D9DE", pt: 0.75 }, rowH: 0.55, valign: "middle" });
  await card(s, { x: 8.4, y: 1.45, w: 4.4, h: 2.75, icon: "FaExclamationTriangle", color: HEX.accent4, head: "The trap", body: "A per-second time penalty sounds natural for 'go faster'. But if waiting costs more than falling (−10), the policy learns to jump off on purpose to end the penalty." });
  await card(s, { x: 8.4, y: 4.4, w: 4.4, h: 2.35, icon: "FaCheck", color: HEX.accent2, head: "The fix", body: "Speed is rewarded only positively: discounting plus a finish bonus that grows with average speed. Falling is always the worst option." });
  s.addNotes("Δs is clamped to ±0.1 m per step so teleporting at reset cannot create reward spikes. Shaping terms are multiplied by dt so they scale with time, not with the number of steps.");

  // PPO
  s = slide("CONTENT");
  s.addText("PPO: Proximal Policy Optimization", { placeholder: "title" });
  formula(s, 0.6, 1.4, 7.1, 3.25, [
    "# Probability ratio between new and old policy",
    "r_t(θ) = π_θ(a_t | s_t) / π_θold(a_t | s_t)",
    "# Clipped objective (ε = 0.2)",
    "L^CLIP = E_t [ min( r_t · Â_t ,  clip(r_t, 1−ε, 1+ε) · Â_t ) ]",
    "# Total loss",
    "L = −L^CLIP + c_v (V_φ(s_t) − G_t)² − c_e · H[π_θ]",
  ], 15);
  formula(s, 0.6, 4.85, 7.1, 1.9, [
    "# Advantage: how much better than expected (GAE, λ = 0.95)",
    "δ_t = r_t + γ V(s_{t+1}) − V(s_t)",
    "Â_t = Σ_k (γλ)^k δ_{t+k}",
  ], 15);
  await card(s, { x: 8.1, y: 1.4, w: 4.7, h: 1.7, icon: "FaUserAstronaut", color: HEX.accent6, head: "Actor", body: "Outputs a Gaussian over the 2 wheel speeds: mean from the network, std learned. Sampling = exploration." });
  await card(s, { x: 8.1, y: 3.25, w: 4.7, h: 1.7, icon: "FaBalanceScale", color: HEX.accent3, head: "Critic", body: "Predicts the value V(s): expected future reward. Used to judge whether an action beat expectations (Â > 0)." });
  await card(s, { x: 8.1, y: 5.1, w: 4.7, h: 1.65, icon: "FaShieldAlt", color: HEX.accent2, head: "Why 'proximal'", body: "Clipping stops one update from changing the policy too much: stable learning, no collapse." });
  s.addNotes("Intuition: actions that led to more reward than the critic expected become more likely; worse ones less likely. The clip keeps each step small. Learning rate adapts to keep the KL divergence near 0.01.");

  // one iteration
  s = slide("CONTENT");
  s.addText("One training iteration = one \"generation\"", { placeholder: "title" });
  const steps = [
    ["FaPlay", "Drive", "2048 robots × 32 steps ≈ 1 s each, with exploration noise", HEX.accent3],
    ["FaDatabase", "Collect", "65,536 transitions: observations, actions, rewards", HEX.accent5],
    ["FaCalculator", "Score", "Critic values → advantages with GAE", HEX.accent1],
    ["FaSyncAlt", "Update", "5 epochs × 4 mini-batches of gradient steps on the clipped loss", HEX.accent6],
    ["FaRedo", "Repeat", "New weights drive the next round", HEX.accent2],
  ];
  for (let i = 0; i < steps.length; i++) {
    const x = 0.6 + i * 2.5;
    await iconDot(s, steps[i][0], x + 0.75, 1.6, 0.9, steps[i][3]);
    s.addText(steps[i][1], tb({ x, y: 2.65, w: 2.3, h: 0.45, fontSize: 18, bold: true, color: C.text1, align: "center" }));
    s.addText(steps[i][2], tb({ x: x + 0.05, y: 3.1, w: 2.2, h: 1.2, fontSize: 13, color: C.text1, align: "center" }));
    if (i < steps.length - 1) line(s, x + 1.85, 2.05, x + 3.05, 2.05, HEX.accent5, 2);
  }
  await card(s, { x: 0.6, y: 4.55, w: 6.0, h: 2.15, icon: "FaUsers", color: HEX.accent2, head: "Not one robot per generation", body: "Every generation is thousands of robots, each on a randomly chosen track and speed. One shared model learns from all of them at once, so no track is learned 'first' and then forgotten." });
  await card(s, { x: 6.8, y: 4.55, w: 6.0, h: 2.15, icon: "FaSave", color: HEX.accent1, head: "Checkpoints", body: "Weights are saved every 1–20 iterations. Replaying old checkpoints is how the generation videos show the robots getting better." });
  s.addNotes("This is the answer to the question 'is it track 1 for 50 generations, then track 2?' No: all tracks are mixed in every iteration. Sequential training would cause catastrophic forgetting.");

  // teacher-student
  s = slide("CONTENT");
  s.addText("Teacher → student: learning from a privileged expert", { placeholder: "title" });
  box(0.8, 1.6, 3.2, 1.4, "Teacher\nsees exact track", HEX.accent1);
  box(0.8, 4.2, 3.2, 1.4, "Student\nsees camera + LiDAR", HEX.accent6);
  s.addText(bullets(["26 numbers: offset, heading, 7 centreline points ahead, speeds", "Trains in minutes, no rendering"], 13), tb({ x: 4.2, y: 1.6, w: 3.6, h: 1.4, paraSpaceAfter: 4 }));
  s.addText(bullets(["6×60×80 image + 61 LiDAR ranges + 3 cliff + 5 proprio", "What a real robot could have"], 13), tb({ x: 4.2, y: 4.2, w: 3.6, h: 1.4, paraSpaceAfter: 4 }));
  line(s, 2.4, 3.0, 2.4, 4.2, HEX.accent5, 2.5);
  s.addText("labels every step", tb({ x: 2.55, y: 3.4, w: 2.0, h: 0.35, fontSize: 12, color: C.accent5 }));
  formula(s, 8.1, 1.4, 4.7, 2.4, [
    "# Distillation loss (DAgger style)",
    "L = ‖μ_s(o_t) − μ_t(s_t)‖²",
    "μ_s: student, from sensors o_t",
    "μ_t: teacher, from exact state s_t",
    "on states visited by the STUDENT",
  ], 15);
  await card(s, { x: 8.1, y: 3.95, w: 4.7, h: 2.75, icon: "FaGraduationCap", color: HEX.accent6, head: "Why let the student drive", body: "If it only copied recordings of the teacher, it would never see its own mistakes. Driving itself and getting the teacher's correction for every situation teaches recovery (DAgger)." });
  s.addNotes("Supervised imitation is far more sample-efficient than RL from pixels: the student gets a target action every step instead of a weak reward signal. Model C adds RL fine-tuning on top.");

  // randomization
  s = slide("CONTENT");
  s.addText("Robustness: asymmetric critic, randomization, dropout", { placeholder: "title" });
  await card(s, { size: 17, x: 0.6, y: 1.45, w: 3.95, h: 5.25, icon: "FaBalanceScaleLeft", color: HEX.accent3, head: "Asymmetric actor-critic", body: "The critic is only needed during training, so it may cheat: it always gets the privileged track state. The actor (what ships) uses only sensors. Better advantage estimates, same deployable policy." });
  await card(s, { size: 17, x: 4.7, y: 1.45, w: 3.95, h: 5.25, icon: "FaDice", color: HEX.accent1, head: "Domain randomization", body: "Wheel friction 0.6–1.2 · chassis mass ±20% · motor strength ±30% · 0–2 steps of actuation delay · action noise · LiDAR noise 1 cm and 2% ray dropout · image brightness, contrast, colour shift, pixel noise." });
  await card(s, { size: 17, x: 8.8, y: 1.45, w: 3.95, h: 5.25, icon: "FaPowerOff", color: HEX.accent4, head: "Sensor dropout", body: "For fusion models, 10% of episodes blank the camera and 10% blank the LiDAR for the whole episode. The network must learn to drive with either sensor, not lean on one." });
  s.addNotes("Randomization makes the policy robust to things that differ between simulation and reality, and between episodes. The user said sim-to-real is not a goal, but it was kept because it is cheap and also improves generalization in simulation.");

  // ================================================================ 3 PHASE 1
  section("3 · Phase 1: one S-ramp");
  s = slide("SECTION");
  s.addText("03", { placeholder: "num" });
  s.addText("Phase 1: one S-ramp", { placeholder: "title" });
  s.addText("Smoke tests, a state teacher, and a camera-only student", { placeholder: "sub" });

  await videoSlide({
    title: "Smoke test: scripted driver, camera inset", video: V1("phase1_scripted_driver_camera_PIP.mp4"), cover: cov("phase1_scripted_driver_camera_PIP"),
    statBig: "Physics ✓", statSmall: "Pure-pursuit controller (no learning) finishes the S", statColor: HEX.accent2,
    points: ["Checks wheel direction, progress tracking and finish detection before any RL", "Robot camera in the top-right: what the student will learn from", "Camera rendered at 160×120 and averaged to 80×60: removed RTX speckle"],
    notes: "Before training anything, a scripted pure-pursuit driver verifies the environment. Found and fixed here: noisy, washed-out camera images (now 2× render + average pooling, plywood ramp colour, dimmer light).",
  });
  await multiVideo({
    title: "Smoke tests: falling and 8 robots",
    vids: [
      { video: V1("phase1_fall_test.mp4"), cover: cov("phase1_fall_test"), label: "Driving straight: every drive-off detected" },
      { video: V1("phase1_scripted_driver_8robots.mp4"), cover: cov("phase1_scripted_driver_8robots"), label: "Scripted driver, 8 robots: 8/8 finish" },
    ],
    notes: "Bug found: a robot can hang on the ramp edge with a wheel over the side and never drop, so the height-based fall check never fired. Fix: a wheel past the edge (|lateral| > half width − 6 cm) counts as a fall.",
  });

  // teacher phase1 chart + video
  await videoSlide({
    title: "Phase 1 teacher: from sitting still to laps", video: V1("phase2_teacher_learning_progress.mp4"), cover: cov("phase2_teacher_learning_progress"),
    statBig: "40 iterations", statSmall: "≈15 s of training to 100% success (1024 robots)", statColor: HEX.accent1,
    points: ["Iteration 0: untrained, outputs ≈ 0 and sits still", "Iteration 15: creeps forward", "Iterations 25–35: takes the first turn slowly", "Final (1500): full lap in ≈10.8 s, 0 falls"],
    notes: "The teacher sees the exact track (offset, heading, centreline points ahead), so PPO solves it almost immediately. Full training: 1500 iterations in about 9 minutes.",
  });
  await multiVideo({
    title: "Phase 1 teacher: iteration 100 and final",
    vids: [
      { video: V1("phase2_teacher_iter100.mp4"), cover: cov("phase2_teacher_iter100"), label: "Iteration 100: already completes the S" },
      { video: V1("phase2_teacher_final.mp4"), cover: cov("phase2_teacher_final"), label: "Final teacher: clean, fast lap" },
    ],
  });
  await videoSlide({
    title: "Phase 1 camera student: learning by imitation", video: V1("phase3_student_learning_progress.mp4"), cover: cov("phase3_student_learning_progress"),
    statBig: "100%", statSmall: "camera-only success after 600 distillation iterations (≈15 min)", statColor: HEX.accent6,
    points: ["Iterations 30–80: drives but falls off", "Imitation loss 1.5 → 0.001", "Lap 10.3 s: as fast as the teacher, using only pixels"],
    notes: "128 robots with cameras (rendering limits the count). Exported to ONNX: inputs camera [1,6,60,80] and last action [1,2], output two wheel speeds.",
  });
  await multiVideo({
    title: "Phase 1 camera student: iteration 100 and the mirrored S",
    vids: [
      { video: V1("phase3_student_iter100.mp4"), cover: cov("phase3_student_iter100"), label: "Iteration 100: camera only, completes the S" },
      { video: V1("phase3_student_final_mirrored.mp4"), cover: cov("phase3_student_final_mirrored"), label: "Final: 2 laps of the mirrored S, 0 falls" },
    ],
  });

  // phase1 reward curves
  s = slide("CONTENT");
  s.addText("Phase 1 training curves", { placeholder: "title" });
  const rt = down(smooth(D.rew_s_teacher, 5), 120), rs = down(smooth(D.rew_s_student, 5), 120);
  s.addChart(pres.charts.LINE, [{ name: "Teacher", labels: sparse(rt.map((p) => p[0])), values: rt.map((p) => +p[1].toFixed(2)) }],
    chartOpts({ x: 0.6, y: 1.4, w: 6.0, h: 4.6, title: "Teacher: mean episode reward vs iteration", chartColors: [HEX.accent1], lineSize: 2, lineDataSymbol: "none", showLegend: false }));
  s.addChart(pres.charts.LINE, [{ name: "Student", labels: sparse(rs.map((p) => p[0])), values: rs.map((p) => +p[1].toFixed(2)) }],
    chartOpts({ x: 6.8, y: 1.4, w: 6.0, h: 4.6, title: "Camera student (distillation): mean reward", chartColors: [HEX.accent6], lineSize: 2, lineDataSymbol: "none", showLegend: false }));
  s.addText("Reward rises as robots drive further before falling, then plateaus once every robot finishes. The student plateaus lower only because its episodes include randomized delays and image noise.", tb({ x: 0.6, y: 6.15, w: 12.2, h: 0.7, fontSize: 13, color: C.accent5 }));

  // ================================================================ 4 GENERALIZATION
  section("4 · Generalizing");
  s = slide("SECTION");
  s.addText("04", { placeholder: "num" });
  s.addText("Generalizing: new tracks, LiDAR, 3× speed", { placeholder: "title" });
  s.addText("Branch feat/lidar-camera-fusion: can one model drive tracks it has never seen?", { placeholder: "sub" });

  s = slide("CONTENT");
  s.addText("Procedural tracks: 12 to learn from, 8 to test on", { placeholder: "title" });
  s.addImage({ path: path.join(IMG, "library_preview.png"), x: 0.6, y: 1.35, w: 8.2, h: 5.51, altText: "Train and test track libraries" });
  s.addText(bullets([
    "Random sequences of straights (0.3–2 m) and arcs (R 0.4–1.2 m, 40–180°, either direction)",
    "Width 22–32 cm, 8 plywood / paint colours",
    "Rejected if any two parts pass closer than 35 cm",
    "Test set never trained on: original S, triple S, long straight + S, tight S, narrow 20 cm S, 3 random",
    "Every reset: random track from the library, mixed in every iteration",
  ], 14), tb({ x: 9.1, y: 1.4, w: 3.7, h: 5.4, paraSpaceAfter: 8 }));
  s.addNotes("The generator runs once at startup; tracks can't change shape mid-training because they are baked into the physics scene. What changes constantly is which track each robot drives. The narrow S (20 cm) is narrower than anything in training on purpose.");

  // LiDAR geometry
  s = slide("CONTENT");
  s.addText("What the LiDAR sees", { placeholder: "title" });
  // front-view diagram: ramp cross-section, robot, fan of rays
  const gx = 0.7, gy = 1.5;
  s.addText("Front view (looking along the track)", tb({ x: gx, y: gy, w: 6.4, h: 0.35, fontSize: 13, bold: true, color: C.accent5 }));
  const floorY = gy + 4.6, rampY = gy + 3.0, cx = gx + 3.2, halfW = 1.1;
  s.addShape(pres.shapes.RECTANGLE, { x: gx, y: floorY, w: 6.4, h: 0.08, fill: { color: "B8BEC6" }, line: { color: "B8BEC6" }, objectName: "floor" });
  s.addShape(pres.shapes.RECTANGLE, { x: cx - halfW, y: rampY, w: 2 * halfW, h: 0.22, fill: { color: HEX.accent1 }, line: { color: HEX.accent1 }, objectName: "ramp" });
  s.addShape(pres.shapes.RECTANGLE, { x: cx - 0.06, y: rampY + 0.22, w: 0.12, h: floorY - rampY - 0.22, fill: { color: "9AA1AB" }, line: { color: "9AA1AB" }, objectName: "pillar" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx - 0.5, y: rampY - 0.75, w: 1.0, h: 0.55, rectRadius: 0.08, fill: { color: HEX.accent2 }, line: { color: HEX.accent2 }, objectName: "robot" });
  const lx = cx, ly = rampY - 0.85;
  s.addShape(pres.shapes.OVAL, { x: lx - 0.09, y: ly - 0.09, w: 0.18, h: 0.18, fill: { color: HEX.dk1 }, line: { color: HEX.dk1 }, objectName: "lidar" });
  for (const dx of [-0.9, -0.55, -0.2, 0.2, 0.55, 0.9]) line(s, lx, ly, lx + dx * 1.15, rampY, HEX.accent2, 1.5);
  for (const dx of [-2.7, -2.0, 2.0, 2.7]) line(s, lx, ly, lx + dx, floorY, "E8902E", 1.5);
  s.addText("hits on the ramp (short)", tb({ x: gx + 0.1, y: gy + 0.5, w: 3, h: 0.3, fontSize: 12, color: HEX.accent2, bold: true }));
  s.addText("beams past the edge hit the floor (long)", tb({ x: gx + 0.1, y: gy + 0.85, w: 4, h: 0.3, fontSize: 12, color: "C76E14", bold: true }));
  s.addImage({ path: path.join(IMG, "lidar_inset.png"), x: 7.4, y: 1.45, w: 2.6, h: 2.6, altText: "LiDAR top view inset" });
  s.addText("Top view from the videos: the green line is where the scan crosses the ramp ≈ 21 cm ahead; its ends are the ramp edges. Blue dots: 3 cliff sensors.", tb({ x: 10.15, y: 1.45, w: 2.7, h: 2.6, fontSize: 13, color: C.text1 }));
  formula(s, 7.4, 4.3, 5.4, 2.45, [
    "# Tilted 2D LiDAR (LD19 class), 10 Hz",
    "61 rays, azimuth α ∈ [−60°, 60°], tilt β = 35° down",
    "d = (cos α cos β,  sin α,  −cos α sin β)",
    "centre hit ≈ h / sin β = 0.12 / 0.57 ≈ 0.21 m",
  ], 15);
  s.addNotes("A flat 2D LiDAR would scan above the ramp and see walls, not the drop. Tilting it down makes every scan a cross-section of the ramp: short ranges where the surface is, long ranges past the edges. The measured centre range in simulation was 0.217 m, matching the 0.21 m design.");

  // analytic lidar math
  s = slide("CONTENT");
  s.addText("An analytic LiDAR (after the built-in one ran out of memory)", { placeholder: "title" });
  formula(s, 0.6, 1.4, 6.4, 3.0, [
    "# For every ray: origin o, direction d (world frame)",
    "t_top = (o_z − z_top) / (−d_z)        ramp-top plane",
    "hit ramp  if  grid[ o + t_top · d ] = 1   (1 cm map)",
    "else  t_floor = o_z / (−d_z)          floor plane",
    "range = min(t, 3 m)",
  ], 15);
  await card(s, { x: 0.6, y: 4.6, w: 6.4, h: 2.15, icon: "FaMap", color: HEX.accent2, head: "Ramp occupancy map", body: "All ramp tops rasterized once into a 1 cm grid: 100% coverage on the ramp, 0% false hits 2 cm outside the edges. Exact for ramp tops and floor; ignores the 3 cm side faces and thin pillars." });
  await card(s, { x: 7.4, y: 1.4, w: 5.4, h: 2.55, icon: "FaMemory", color: HEX.accent4, head: "Why not Isaac Lab's ray caster", body: "Its multi-mesh ray caster made a separate GPU copy of the ramp meshes for each environment. At ~1400 of 2048 robots the GPU ran out of memory and training crashed." });
  await card(s, { x: 7.4, y: 4.15, w: 5.4, h: 2.6, icon: "FaBolt", color: HEX.accent1, head: "Result", body: "64 rays × 2048 robots is now a few tensor operations per step: almost free compared to camera rendering. That is what makes LiDAR-only training ~30× cheaper than camera training." });

  // speed curriculum chart
  s = slide("CONTENT");
  s.addText("Speed curriculum: 1× → 2× → 3×", { placeholder: "title" });
  const sa = down(D.speed_A.slice(0, 400), 100);
  s.addChart(pres.charts.LINE, [{ name: "Model A speed cap", labels: sparse(sa.map((p) => p[0])), values: sa.map((p) => p[1]) }],
    chartOpts({ x: 0.6, y: 1.4, w: 7.0, h: 4.4, title: "Model A: unlocked speed cap vs iteration (first 400)", chartColors: [HEX.accent2], lineSize: 3, lineDataSymbol: "none", showLegend: false, valAxisMinVal: 0, valAxisMaxVal: 3.5 }));
  await card(s, { x: 8.0, y: 1.4, w: 4.8, h: 2.2, icon: "FaTachometerAlt", color: HEX.accent1, head: "Rule", body: "Unlock the next level when ≥ 85% of episodes at the current top speed finish (after 3000 such episodes; 800 for camera runs)." });
  await card(s, { x: 8.0, y: 3.8, w: 4.8, h: 2.95, icon: "FaRandom", color: HEX.accent3, head: "Mixing speeds", body: "60% of robots drive at the current top speed, the rest at a random speed in [1×, top]. The speed cap is an input, so one network handles all speeds." });
  s.addText("Teacher reached 3× by iteration ~60; A by ~93; camera models more slowly (fewer robots → fewer episodes).", tb({ x: 0.6, y: 5.95, w: 7.0, h: 0.8, fontSize: 13, color: C.accent5 }));
  s.addNotes("Jumping straight to 3× produces robots that mostly fall. The curriculum keeps the task just hard enough. Side effect seen in the results: most training happens at top speed, so several models are slightly better at 3× than at 1×.");

  // models A B C + architecture
  s = slide("CONTENT");
  s.addText("Four models, three ways to learn from sensors", { placeholder: "title" });
  const mods = [
    ["Teacher", "FaChalkboardTeacher", "PPO on exact track info (26 numbers). Fast, but not deployable."],
    ["A", "FaSatelliteDish", "LiDAR + cliff + proprio (69 numbers). PPO from scratch, 2048 robots."],
    ["B", "FaCamera", "Camera + LiDAR + cliff + proprio. PPO from scratch, 192 robots."],
    ["C", "FaGraduationCap", "Same network as B. Distilled from the teacher, then PPO fine-tuned."],
  ];
  for (let i = 0; i < 4; i++) {
    const y = 1.4 + i * 1.35;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 5.6, h: 1.2, rectRadius: 0.1, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "model-card" });
    await iconDot(s, mods[i][1], 0.75, y + 0.25, 0.7, MC[mods[i][0]]);
    s.addText(mods[i][0] === "Teacher" ? "Teacher" : "Model " + mods[i][0], tb({ x: 1.6, y: y + 0.08, w: 4.4, h: 0.4, fontSize: 17, bold: true, color: C.text1 }));
    s.addText(mods[i][2], tb({ x: 1.6, y: y + 0.48, w: 4.5, h: 0.65, fontSize: 13, color: C.text1 }));
  }
  // fusion network diagram
  const nb = (x, y, w, h, t, col, fs = 13) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: col }, line: { color: col }, objectName: "net-block" });
    s.addText(t, tb({ x, y, w, h, fontSize: fs, bold: true, color: C.background1, align: "center", valign: "middle" }));
  };
  s.addText("Fusion network (B and C)", tb({ x: 6.7, y: 1.4, w: 6, h: 0.35, fontSize: 14, bold: true, color: C.accent5 }));
  nb(6.7, 1.9, 2.0, 0.75, "Camera\n6×60×80", HEX.accent3);
  nb(6.7, 2.85, 2.0, 0.75, "LiDAR 61", HEX.accent2);
  nb(6.7, 3.8, 2.0, 0.75, "Cliff 3 · proprio 5", HEX.accent5, 12);
  nb(9.0, 1.9, 1.7, 0.75, "CNN 16-32-32", HEX.accent3, 12);
  line(s, 8.7, 2.27, 9.0, 2.27, HEX.accent5, 2);
  nb(11.0, 2.4, 1.8, 1.7, "Merge →\nMLP 256-128", HEX.accent6, 13);
  line(s, 10.7, 2.27, 11.0, 2.8, HEX.accent5, 2);
  line(s, 8.7, 3.22, 11.0, 3.22, HEX.accent5, 2);
  line(s, 8.7, 4.17, 11.0, 3.7, HEX.accent5, 2);
  nb(11.0, 4.6, 1.8, 0.8, "2 wheel\nspeeds", HEX.accent1);
  line(s, 11.9, 4.1, 11.9, 4.6, HEX.accent5, 2);
  s.addText("CNN: conv 5×5/2 → 3×3/2 → 3×3/2, ReLU, flattened and concatenated with the 1D sensors. Critic: MLP 256-128 on the 26 privileged numbers. ~450k weights.", tb({ x: 6.7, y: 5.6, w: 6.1, h: 1.2, fontSize: 13, color: C.text1 }));

  await videoSlide({
    title: "Branch smoke test: camera + LiDAR insets", video: VF("f1_smoke_triple_s.mp4"), cover: cov("f1_smoke_triple_s"),
    statBig: "0.89 m/s", statSmall: "measured top speed with the 3× unlock (target 1.08 m/s × 0.8)", statColor: HEX.accent1,
    points: ["Scripted driver on the held-out triple S at 1.5×", "Bottom-right: LiDAR top view, green = ramp hits, blue = cliff sensors", "At 3× the scripted driver falls ~50%: speed makes the task genuinely hard"],
    notes: "Insets were resized at the user's request: camera at half area, flush top-right; LiDAR view under it.",
  });

  // ================================================================ 5 GENERATIONS
  section("5 · Generations");
  s = slide("SECTION");
  s.addText("05", { placeholder: "num" });
  s.addText("Watching generations learn", { placeholder: "title" });
  s.addText("60 robots per checkpoint, released together on the same never-seen track", { placeholder: "sub" });

  s = slide("CONTENT");
  s.addText("How the generation videos are made", { placeholder: "title" });
  const gsteps = [
    ["FaSave", "Save often", "Extra short runs save every 1–2 iterations around the learning jump", HEX.accent1],
    ["FaSearch", "Scan", "Score every checkpoint (64 robots, no rendering)", HEX.accent5],
    ["FaListOl", "Pick", "A few early ones, then one per ~6% gain once robots pass 30%", HEX.accent6],
    ["FaCarSide", "Drive", "60 robots, each on its own copy of the track, no viewer", HEX.accent3],
    ["FaFilm", "Replay", "Trajectories replayed onto one track and recorded", HEX.accent2],
  ];
  for (let i = 0; i < gsteps.length; i++) {
    const x = 0.6 + i * 2.5;
    await iconDot(s, gsteps[i][0], x + 0.75, 1.5, 0.85, gsteps[i][3]);
    s.addText(gsteps[i][1], tb({ x, y: 2.45, w: 2.3, h: 0.45, fontSize: 16, bold: true, color: C.text1, align: "center" }));
    s.addText(gsteps[i][2], tb({ x: x + 0.05, y: 2.9, w: 2.2, h: 1.3, fontSize: 13, color: C.text1, align: "center" }));
    if (i < gsteps.length - 1) line(s, x + 1.8, 1.93, x + 3.05, 1.93, HEX.accent5, 2);
  }
  s.addMedia({ type: "video", path: VF("teacher_generations.mp4"), x: 0.6, y: 4.35, w: 4.3, h: 2.42, cover: await img64(cov("teacher_generations")), objectName: "video" });
  s.addText(bullets([
    "First version (left): checkpoints only every 20 iterations, so it jumps from 'all fall' to 'all finish'",
    "Fix: save every iteration around the jump; pick generations automatically from the scan",
    "Robots don't collide: Isaac Lab filters collisions between environments",
  ], 14), tb({ x: 5.2, y: 4.4, w: 7.6, h: 2.4, paraSpaceAfter: 6 }));
  s.addNotes("The two-process drive-then-replay method exists because of a bug: an active viewer changed what the robots' own cameras rendered, and stacked robots could see each other. Driving without a viewer on separate track copies matches evaluation conditions exactly.");

  await videoSlide({
    title: "Teacher generations: the climb in ~8 iterations", video: VF("teacher_generations_long.mp4"), cover: cov("teacher_generations_long"), wide: true,
    points: ["Median distance: 1% → 8% → 14% → 35% → 89%", "Finishers: 14 → 16 → 45 → 55 → 58 → 60 of 60", "Best lap 19.1 s → 17.1 s (final)", "Held-out triple S, 2× speed"],
    notes: "Privileged information makes the teacher's learning almost a step function: iterations 40 to 48. Everything before that is robots falling off near the start.",
  });
  await videoSlide({
    title: "Model A (LiDAR only) generations", video: VF("A_generations_long.mp4"), cover: cov("A_generations_long"), wide: true,
    points: ["Iterations 0–61: everyone falls", "62 → 63 → 64: 18 → 39 → 52 finish", "Iteration 70: 60/60", "Dip at 120–132 when 3× unlocked: hesitant, none fall", "Final best lap 17.3 s"],
    notes: "The dip is the speed curriculum: training switched mostly to 3×, and for a few iterations the policy is less confident at 2×. It recovers by iteration 138.",
  });
  await videoSlide({
    title: "Model B (camera + LiDAR, RL from scratch) generations", video: VF("B_generations_long.mp4"), cover: cov("B_generations_long"), wide: true,
    points: ["The most gradual climb", "Median: 2% → 9% → 27% → 60% → 77% → 89% → 93%", "Finishers: 3 → 7 → 16 → 20 → 53 → 60", "60/60 by iteration 140"],
    notes: "Learning from reward and raw sensors, with no teacher, is slower and steadier: every 10 iterations the swarm gets a bit further. This is the 'inching' the generation videos were designed to show.",
  });
  await videoSlide({
    title: "Model C (teacher-distilled, then RL) generations", video: VF("C_generations_long.mp4"), cover: cov("C_generations_long"), wide: true,
    points: ["Imitation is bumpy early: 120 → 5 finish, 160 → 27, 180 → 2", "Stable 60/60 from distillation iteration 200", "RL fine-tune keeps it there (59/60) and adds 3× speed"],
    notes: "Copying the teacher is non-monotonic at first because the student visits states the teacher never did; DAgger fixes that over time.",
  });

  // learning curves small multiples
  s = slide("CONTENT");
  s.addText("Learning curves on the held-out triple S (2×)", { placeholder: "title" });
  const lc = [["Teacher", D.scan_teacher, 0.6, 1.35], ["A", D.scan_A.filter((r) => r[0] <= 160), 6.8, 1.35], ["B", D.scan_B.filter((r) => r[0] <= 400), 0.6, 4.05], ["C (distillation)", D.scan_Cd, 6.8, 4.05]];
  for (const [name, rows, x, y] of lc) {
    const col = MC[name[0] === "C" ? "C" : name] || MC.C;
    s.addChart(pres.charts.LINE, [
      { name: "median progress %", labels: sparse(rows.map((r) => r[0])), values: rows.map((r) => r[1]) },
      { name: "finished %", labels: sparse(rows.map((r) => r[0])), values: rows.map((r) => r[2]) },
    ], chartOpts({ x, y, w: 5.9, h: 2.6, title: name + ": progress and finish rate vs iteration", chartColors: [col, HEX.accent5], lineSize: 2, lineDataSymbol: "none", showLegend: true, legendPos: "b", valAxisMinVal: 0, valAxisMaxVal: 100 }));
  }
  s.addNotes("Teacher and A jump almost vertically (within a handful of iterations). B rises gradually over ~140 iterations. C's distillation is noisy until ~200. All end near 99% median progress, where the cap is the finish line.");

  // reward curves
  s = slide("CONTENT");
  s.addText("Training reward: all four models", { placeholder: "title" });
  const rc = [["Teacher", D.rew_teacher, 0.6, 1.35], ["A", D.rew_A, 6.8, 1.35], ["B", D.rew_B, 0.6, 4.05], ["C", D.rew_Cft, 6.8, 4.05]];
  for (const [name, rows, x, y] of rc) {
    const r2 = down(smooth(rows, 5), 100);
    s.addChart(pres.charts.LINE, [{ name, labels: sparse(r2.map((p) => p[0])), values: r2.map((p) => +p[1].toFixed(1)) }],
      chartOpts({ x, y, w: 5.9, h: 2.6, title: (name === "C" ? "C (RL fine-tune)" : name === "Teacher" ? "Teacher" : "Model " + name) + ": mean episode reward", chartColors: [MC[name]], lineSize: 2, lineDataSymbol: "none", showLegend: false }));
  }
  s.addNotes("C starts high because the RL fine-tune begins from the distilled student. Rewards are not directly comparable across runs: speed levels and sensor dropout differ during training.");

  // ================================================================ 6 RESULTS
  section("6 · Results");
  s = slide("SECTION");
  s.addText("06", { placeholder: "num" });
  s.addText("Results on never-seen tracks", { placeholder: "title" });
  s.addText("8 test tracks × 8 robots, deterministic driving, no viewer", { placeholder: "sub" });

  // results table
  s = slide("CONTENT");
  s.addText("Held-out evaluation: success rate", { placeholder: "title" });
  const E = D.eval;
  const pct = (k) => (E[k] ? Math.round(E[k].success_rate * 100) + "%" : "–");
  const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, align: "center" } });
  const trs = [
    [hdr("Model"), hdr("1×"), hdr("3×"), hdr("3× LiDAR off"), hdr("3× camera off"), hdr("Fails on")],
    ["Teacher (exact track)", pct("teacher_1x"), pct("teacher_3x"), "n/a", "n/a", "narrow S (+ tight S at 1×)"],
    ["A · LiDAR only", pct("A_1x"), pct("A_3x"), pct("A_3x_lidar_off") + " (blind)", "n/a", "narrow S at 1×"],
    ["B · camera + LiDAR, RL", pct("B_1x"), pct("B_3x"), pct("B_3x_lidar_off"), pct("B_3x_camera_off"), "narrow S"],
    ["C · teacher → fusion → RL", pct("C_1x"), pct("C_3x"), pct("C_3x_lidar_off"), pct("C_3x_camera_off"), "nothing (both sensors on)"],
  ];
  s.addTable(trs, { x: 0.6, y: 1.45, w: 12.2, colW: [3.0, 1.2, 1.2, 1.9, 1.9, 3.0], fontSize: 15, fontFace: "Calibri", color: HEX.dk1, border: { type: "solid", color: "D5D9DE", pt: 0.75 }, rowH: 0.62, valign: "middle", align: "center" });
  await card(s, { x: 0.6, y: 4.85, w: 3.9, h: 1.9, icon: "FaTrophy", color: MC.C, head: "Best driver: C", body: "100% at both speeds, including tracks the teacher fails." });
  await card(s, { x: 4.7, y: 4.85, w: 3.9, h: 1.9, icon: "FaShieldAlt", color: MC.B, head: "Most robust: B", body: "Still drives with either sensor off: RL + dropout used both." });
  await card(s, { x: 8.8, y: 4.85, w: 4.0, h: 1.9, icon: "FaSatelliteDish", color: MC.A, head: "Cheapest: A", body: "100% at 3× with only 69 numbers and no rendering." });
  s.addNotes("Single training run per model and 8 near-identical starts per track, so small differences (98% vs 100%) are not meaningful. 'Falls' include a wheel crossing the edge margin, which on the 20 cm ramp is only 4 cm off-centre.");

  // results charts
  s = slide("CONTENT");
  s.addText("Speed and sensor-failure comparison", { placeholder: "title" });
  const sr = (k) => (E[k] ? Math.round(E[k].success_rate * 100) : 0);
  s.addChart(pres.charts.BAR, [
    { name: "1×", labels: ["Teacher", "A", "B", "C"], values: [sr("teacher_1x"), sr("A_1x"), sr("B_1x"), sr("C_1x")] },
    { name: "3×", labels: ["Teacher", "A", "B", "C"], values: [sr("teacher_3x"), sr("A_3x"), sr("B_3x"), sr("C_3x")] },
  ], chartOpts({ x: 0.6, y: 1.4, w: 6.0, h: 4.7, barDir: "col", barGrouping: "clustered", title: "Success on unseen tracks (%)", chartColors: [HEX.accent5, HEX.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: HEX.dk1, valAxisMinVal: 0, valAxisMaxVal: 110, showLegend: true, legendPos: "b" }));
  s.addChart(pres.charts.BAR, [
    { name: "both sensors", labels: ["A", "B", "C"], values: [sr("A_3x"), sr("B_3x"), sr("C_3x")] },
    { name: "LiDAR off", labels: ["A", "B", "C"], values: [sr("A_3x_lidar_off"), sr("B_3x_lidar_off"), sr("C_3x_lidar_off")] },
    { name: "camera off", labels: ["A", "B", "C"], values: [0, sr("B_3x_camera_off"), sr("C_3x_camera_off")] },
  ], chartOpts({ x: 6.8, y: 1.4, w: 6.0, h: 4.7, barDir: "col", barGrouping: "clustered", title: "At 3×: what if a sensor fails? (%)", chartColors: [HEX.accent2, HEX.accent4, HEX.accent3], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: HEX.dk1, valAxisMinVal: 0, valAxisMaxVal: 110, showLegend: true, legendPos: "b" }));
  s.addText("A has no camera, so 'camera off' doesn't apply (shown as 0). Most models are better at 3× than 1×: the curriculum spends most training at top speed.", tb({ x: 0.6, y: 6.2, w: 12.2, h: 0.65, fontSize: 13, color: C.accent5 }));

  await videoSlide({
    title: "Teacher on all 8 test tracks at once (3×)", video: VF("teacher_test_mosaic.mp4"), cover: cov("teacher_test_mosaic"), wide: true,
    statBig: "7 / 8", statSmall: "test tracks finished by the teacher at 3×", statColor: MC.Teacher,
    points: ["One robot per track, follow cameras tiled", "Fails the narrow S at 96%", "Tight S in 3.4 s, triple S in 11.9 s"],
    notes: "Tile backgrounds are black because Isaac Lab's tiled follow cameras don't render the floor. The teacher relies on exact track coordinates, and that transfers worst to unfamiliar widths.",
  });
  await videoSlide({
    title: "Model C: first drive on all 8 unseen tracks", video: VF("unseen_tracks_first_drive_C.mp4"), cover: cov("unseen_tracks_first_drive_C"), wide: true,
    statBig: "8 / 8", statSmall: "finished on the first attempt, 0 falls (1× speed)", statColor: MC.C,
    points: ["3 min: title card with map, then the full run", "Triple S 35.9 s, narrow S 13.8 s", "Camera and LiDAR insets throughout"],
    notes: "Each track appears with its top-down map first. The model has never seen any of these shapes.",
  });
  await multiVideo({
    title: "Single-robot highlights at 3× (camera + LiDAR insets)",
    vids: [
      { video: VF("C_pip_triple_s_3x.mp4"), cover: cov("C_pip_triple_s_3x"), label: "C · triple S · 12.0 s" },
      { video: VF("C_pip_narrow_s_3x.mp4"), cover: cov("C_pip_narrow_s_3x"), label: "C · narrow 20 cm S · 4.7 s" },
      { video: VF("B_pip_camera_off_3x.mp4"), cover: cov("B_pip_camera_off_3x"), label: "B · CAMERA OFF · 7.4 s" },
      { video: VF("A_pip_tight_s_3x.mp4"), cover: cov("A_pip_tight_s_3x"), label: "A · tight S · 3.4 s" },
    ],
    notes: "B with its camera switched off (inset darkened) still finishes on LiDAR alone, which is the payoff of sensor-dropout training.",
  });
  await videoSlide({
    title: "The race: hardest track, all four models", video: VF("race_narrow_s_3x.mp4"), cover: cov("race_narrow_s_3x"), wide: true,
    points: ["Narrow 20 cm S at 3×, 10 robots each", "Teacher 0/10 · A 10/10 · B 7/10 · C 6/10", "C got 8/8 in the evaluation: the race uses wider random starts"],
    notes: "On the narrowest ramp the LiDAR's direct edge signal is the most reliable input. Starting variation matters a lot when the safety margin is only 4 cm.",
  });

  // ================================================================ 7 LEARNINGS
  // ================================================================ 7 STRESS TEST
  section("7 · Stress test");
  s = slide("SECTION");
  s.addText("07", { placeholder: "num" });
  s.addText("Stress test: 8 very difficult tracks", { placeholder: "title" });
  s.addText("Built to break model C: 133 m long, floor-coloured, rainbow stripes, hairpins, sharp corners, 18 cm wide", { placeholder: "sub" });

  s = slide("CONTENT");
  s.addText("The stress-test tracks", { placeholder: "title" });
  s.addImage({ path: path.join(IMG, "stress_preview.png"), x: 0.6, y: 1.35, w: 12.1, h: 4.76, altText: "The 8 stress-test tracks" });
  s.addText("None of these appear in training: lengths up to 133 m (training 4–9 m), a ramp nearly the floor's colour, 8-colour stripes, 0.3 m hairpins (training minimum 0.4 m), 90° near-sharp corners (training: smooth arcs only), 18 cm width (training 22–32 cm).", tb({ x: 0.6, y: 6.2, w: 12.1, h: 0.75, fontSize: 13, color: C.text1 }));
  s.addNotes("Each track isolates one difficulty, so a failure points to a specific cause. The rainbow ramp is built from one mesh per 40 cm stripe; physics sees one continuous surface.");

  const ST = D.stress;
  const sres = (k, t) => { const r = ST[k] && ST[k].tracks[t]; return r ? `${r.finished}/${r.robots}` : "–"; };
  const tnames = ["long_wiggle_60m", "serpentine_120m", "white_s", "rainbow_s", "hairpins_r03", "sharp_zigzag", "narrow_18cm_s", "narrow_chicane"];
  const tlabel = { long_wiggle_60m: "Long wiggle 58 m", serpentine_120m: "Serpentine 133 m", white_s: "White S", rainbow_s: "Rainbow S", hairpins_r03: "Hairpins 0.3 m", sharp_zigzag: "Sharp zigzag", narrow_18cm_s: "Narrow 18 cm S", narrow_chicane: "Narrow chicane" };
  s = slide("CONTENT");
  s.addText("Stress-test results (robots finished out of 8)", { placeholder: "title" });
  const sh = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, align: "center" } });
  const srows = [[sh("Track"), sh("C · 1×"), sh("C · 3×"), sh("B · 3×"), sh("A · 3× (LiDAR)")]];
  for (const t of tnames) srows.push([tlabel[t], sres("C_1x", t), sres("C_3x", t), sres("B_3x", t), sres("A_3x", t)]);
  const tot = (k) => (ST[k] ? Math.round(ST[k].success_rate * 100) + "%" : "–");
  srows.push([{ text: "Overall", options: { bold: true } }, { text: tot("C_1x"), options: { bold: true } }, { text: tot("C_3x"), options: { bold: true } }, { text: tot("B_3x"), options: { bold: true } }, { text: tot("A_3x"), options: { bold: true } }]);
  s.addTable(srows, { x: 0.6, y: 1.4, w: 7.4, colW: [2.6, 1.15, 1.15, 1.15, 1.35], fontSize: 14, fontFace: "Calibri", color: HEX.dk1, border: { type: "solid", color: "D5D9DE", pt: 0.75 }, rowH: 0.5, valign: "middle", align: "center" });
  await card(s, { x: 8.3, y: 1.4, w: 4.5, h: 1.65, icon: "FaRulerHorizontal", color: HEX.accent2, head: "Length: no problem", body: "C drove 133 m and 58 m with zero falls at both speeds: ≈3 km without a fall." });
  await card(s, { x: 8.3, y: 3.2, w: 4.5, h: 1.75, icon: "FaPalette", color: HEX.accent4, head: "Colour breaks the camera", body: "White and rainbow ramps: C 0–1/8. A cannot see colour and gets 8/8." });
  await card(s, { x: 8.3, y: 5.1, w: 4.5, h: 1.65, icon: "FaDrawPolygon", color: HEX.accent1, head: "Corners hurt too", body: "Sharp zigzag: C 2–3/8, A 8/8. Geometry beats appearance here." });
  s.addNotes("B finished 5/8 on rainbow with 0 falls: slow and hesitant (3 still driving at the time limit) but safe, consistent with B using its LiDAR more than C does.");

  s = slide("CONTENT");
  s.addText("Who handles what: success per stress track at 3×", { placeholder: "title" });
  const pc = (k, t) => { const r = ST[k] && ST[k].tracks[t]; return r ? Math.round((100 * r.finished) / r.robots) : 0; };
  s.addChart(pres.charts.BAR, ["C_3x", "B_3x", "A_3x"].map((k) => ({ name: { C_3x: "C (camera-heavy)", B_3x: "B (fusion RL)", A_3x: "A (LiDAR only)" }[k], labels: tnames.map((t) => tlabel[t]), values: tnames.map((t) => pc(k, t)) })),
    chartOpts({ x: 0.6, y: 1.35, w: 12.2, h: 5.0, barDir: "col", barGrouping: "clustered", title: "Robots finished (%), 8 per track, 3× speed", chartColors: [MC.C, MC.B, MC.A], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: HEX.dk1, valAxisMinVal: 0, valAxisMaxVal: 115, showLegend: true, legendPos: "b", catAxisLabelFontSize: 11 }));
  s.addText("The cheapest model (LiDAR only) is the most robust once appearance changes. C's earlier 100% depended on ramps looking like training ramps.", tb({ x: 0.6, y: 6.4, w: 12.2, h: 0.5, fontSize: 13, color: C.accent5 }));

  await videoSlide({
    title: "Model C on all 8 stress tracks (3×)", video: VF("stress_tour_C_3x.mp4"), cover: cov("stress_tour_C_3x"), wide: true,
    statBig: tot("C_3x"), statSmall: "of 64 runs finished at 3× (8 robots per track)", statColor: MC.C,
    points: ["Title card per track: map, why it is hard, evaluation result", "Then one full run with camera + LiDAR insets", "Watch the white and rainbow ramps: the camera loses the edges"],
    notes: "Each clip is one deterministic run, so on tracks with mixed results (e.g. 3/8) the clip shows one possible outcome; the card shows the 8-robot evaluation.",
  });
  await multiVideo({
    title: "Colour-blind wins: C vs A on the white and rainbow ramps",
    vids: [
      { video: VF("stress_white_C.mp4"), cover: cov("stress_white_C"), label: "C (camera + LiDAR) · white S · falls" },
      { video: VF("stress_white_A.mp4"), cover: cov("stress_white_A"), label: "A (LiDAR only) · white S · 4.6 s" },
      { video: VF("stress_rainbow_C.mp4"), cover: cov("stress_rainbow_C"), label: "C · rainbow S · falls" },
      { video: VF("stress_rainbow_A.mp4"), cover: cov("stress_rainbow_A"), label: "A · rainbow S · 9.5 s" },
    ],
    notes: "Same tracks, same speed (3×). C relies on the camera's ramp-vs-floor contrast learned from its teacher-guided training; A only measures distances, so colour is irrelevant to it.",
  });
  await multiVideo({
    title: "Sharp corners: C vs A on the zigzag",
    vids: [
      { video: VF("stress_zigzag_C.mp4"), cover: cov("stress_zigzag_C"), label: "C · sharp zigzag · 3× · falls" },
      { video: VF("stress_zigzag_A.mp4"), cover: cov("stress_zigzag_A"), label: "A · sharp zigzag · 3× · 9.2 s" },
    ],
  });
  await videoSlide({
    title: "Side by side: C vs A on the three hardest-looking tracks", video: VF("stress_C_vs_A.mp4"), cover: cov("stress_C_vs_A"), wide: true,
    points: ["White S: C falls, A 4.6 s", "Rainbow S: C falls, A 9.5 s", "Sharp zigzag: C falls, A 9.2 s", "Same tracks, same speed, one run each"],
    notes: "A useful demo for 'why fusion is not automatically better': the extra sensor helps on familiar-looking tracks and hurts when its input changes in a way training never covered.",
  });

  s = slide("CONTENT");
  s.addText("What the stress test teaches", { placeholder: "title" });
  const stl = [
    ["FaInfinity", "Reactive policies scale with length", "No memory of 'where am I' means metre 130 looks like metre 2. 133 m was as easy as 5 m.", HEX.accent2],
    ["FaEye", "Cameras overfit to appearance", "C learned 'dark strip on a light floor'. Change the look and it fails, even though the shape is easy.", HEX.accent4],
    ["FaSatelliteDish", "Geometry sensors generalize", "LiDAR ranges don't change with paint. A (98%) beat the 'best' model C (59%) on these tracks.", HEX.accent3],
    ["FaWrench", "Fixes to try next", "Train with ramp/floor colour randomization incl. floor-coloured ramps and stripes; add sharp corners to the generator; raise camera dropout so fusion models lean on LiDAR.", HEX.accent1],
  ];
  for (let i = 0; i < stl.length; i++) {
    const x = 0.6 + (i % 2) * 6.2, y = 1.45 + Math.floor(i / 2) * 2.7;
    await card(s, { x, y, w: 5.95, h: 2.5, icon: stl[i][0], color: stl[i][3], head: stl[i][1], body: stl[i][2], size: 15 });
  }

  section("8 · Failures & learnings");
  s = slide("SECTION");
  s.addText("08", { placeholder: "num" });
  s.addText("What went wrong, and what we learned", { placeholder: "title" });
  s.addText("Bugs are part of the result: each one changed how the system is built", { placeholder: "sub" });

  s = slide("CONTENT");
  s.addText("Failures and fixes", { placeholder: "title" });
  const fails = [
    ["FaAnchor", "Robot hung on the edge", "Never dropped, so no fall registered → wheel-over-edge rule"],
    ["FaMemory", "Ray caster out of memory", "Per-env mesh copies → analytic LiDAR on a 1 cm map"],
    ["FaEyeSlash", "Viewer changed the cameras", "Camera models failed only in videos → drive with no viewer, then replay"],
    ["FaUserFriends", "Robots saw each other", "Stacked swarm in every camera → each robot on its own track copy"],
    ["FaSnowflake", "Frozen robots", "Two GPU jobs → silent PhysX out-of-memory → one job at a time"],
    ["FaFileVideo", "Wrong videos sent", "Recorder numbers clips instead of overwriting → clear old clips first"],
    ["FaExclamationCircle", "LiDAR mount crash", "Isaac Lab 3.0 stopped spawning sensor frames → reuse the Jetbot camera mount"],
    ["FaQuestionCircle", "False alarm: curriculum", "Unlocked 2× by iter 33: not a bug, the teacher really was 80% successful"],
  ];
  for (let i = 0; i < fails.length; i++) {
    const x = 0.6 + (i % 4) * 3.1, y = 1.4 + Math.floor(i / 4) * 2.75;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 2.9, h: 2.55, rectRadius: 0.12, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "fail-card" });
    await iconDot(s, fails[i][0], x + 0.2, y + 0.2, 0.6, i === 7 ? HEX.accent5 : HEX.accent4);
    s.addText(fails[i][1], tb({ x: x + 0.2, y: y + 0.9, w: 2.5, h: 0.6, fontSize: 15, bold: true, color: C.text1 }));
    s.addText(fails[i][2], tb({ x: x + 0.2, y: y + 1.5, w: 2.5, h: 0.95, fontSize: 13, color: C.text1 }));
  }
  s.addNotes("The viewer and stacked-robot issues only affected camera models, which is why the teacher and A videos looked fine while C's first video showed robots failing. A controlled test (same checkpoint with and without a viewer: 6/60 vs 96.7%) isolated the cause.");

  s = slide("CONTENT");
  s.addText("What we learned about RL", { placeholder: "title" });
  const learn = [
    ["FaBolt", "Privileged learning is sudden", "Teacher and A go from 'all fall' to 'all finish' in under 10 iterations. B, learning from pixels, takes ~140."],
    ["FaChalkboardTeacher", "Teachers make the best students", "Distillation + RL (C) beat RL from scratch (B) and even its own teacher on unseen tracks."],
    ["FaMapMarkedAlt", "Exact maps generalize worse", "The teacher memorises geometry; sensor-based students react to edges and handle new widths."],
    ["FaRandom", "Dropout buys robustness", "B (RL with dropout) survives either sensor failing; C copied a camera-heavy teacher and leans on it."],
    ["FaTachometerAlt", "Curricula shape skills", "Training mostly at 3× made several models slightly worse at slow speed."],
    ["FaVial", "Test like you train", "Evaluation conditions (viewer, neighbours, start noise) can swing results from 6% to 97%."],
  ];
  for (let i = 0; i < learn.length; i++) {
    const x = 0.6 + (i % 2) * 6.2, y = 1.4 + Math.floor(i / 2) * 1.8;
    await iconDot(s, learn[i][0], x, y + 0.1, 0.7, [HEX.accent1, HEX.accent6, HEX.accent2, HEX.accent3, HEX.accent1, HEX.accent4][i]);
    s.addText(learn[i][1], tb({ x: x + 0.9, y, w: 5.1, h: 0.45, fontSize: 17, bold: true, color: C.text1 }));
    s.addText(learn[i][2], tb({ x: x + 0.9, y: y + 0.48, w: 5.1, h: 1.15, fontSize: 14, color: C.text1 }));
  }

  s = slide("CONTENT");
  s.addText("Limitations, and what to do next", { placeholder: "title" });
  await card(s, { x: 0.6, y: 1.4, w: 5.9, h: 5.35, icon: "FaBalanceScaleRight", color: HEX.accent5, head: "Limitations (honest caveats)", body: bullets([
    "One training run per model: no error bars across seeds",
    "8 robots per track from nearly the same start: small differences aren't meaningful",
    "'Fall' includes a wheel crossing a safety margin, strict on narrow ramps",
    "Camera-heavy C fails on floor-coloured or striped ramps (stress test)",
    "Robots never learned to stop at the end of the ramp",
    "Simulation only; real-world transfer was explicitly out of scope",
  ], 17) });
  await card(s, { x: 6.8, y: 1.4, w: 6.0, h: 5.35, icon: "FaRocket", color: HEX.accent2, head: "Next steps", body: bullets([
    "Rotate track libraries during training (fresh shapes every N iterations)",
    "Train narrower widths, sharp corners and colour variety into the library",
    "Recurrent (LSTM) students to remember what the camera saw",
    "Several seeds per model for confidence intervals",
    "Teach a 'stop at the end' behaviour; optional sim-to-real later",
  ], 17) });

  // appendix
  section("Appendix");
  s = slide("CONTENT");
  s.addText("Appendix: training runs and settings", { placeholder: "title" });
  const ah = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 } } });
  s.addTable([
    [ah("Run"), ah("Algorithm"), ah("Robots"), ah("Iterations"), ah("Time"), ah("Network")],
    ["S-ramp teacher", "PPO", "1024", "1500", "~9 min", "MLP 128-128"],
    ["S-ramp camera student", "Distillation", "128", "600", "~15 min", "CNN 16-32-32 + MLP 256-128"],
    ["Tracks teacher", "PPO + speed curriculum", "2048", "780", "~9 min", "MLP 256-128"],
    ["A · LiDAR", "PPO", "2048", "3000", "~45 min", "MLP 256-256-128"],
    ["B · fusion", "PPO", "192", "1500", "~58 min", "CNN + MLP 256-128"],
    ["C · fusion", "Distill 400 + PPO 500", "192", "900", "~37 min", "CNN + MLP 256-128"],
  ], { x: 0.6, y: 1.4, w: 12.2, colW: [2.6, 2.6, 1.2, 1.3, 1.2, 3.3], fontSize: 13, fontFace: "Calibri", color: HEX.dk1, border: { type: "solid", color: "D5D9DE", pt: 0.75 }, rowH: 0.5, valign: "middle" });
  s.addText("PPO: γ = 0.99, λ = 0.95, clip ε = 0.2, 32 steps per robot per iteration, 5 epochs × 4 mini-batches (8 for camera runs), adaptive learning rate (KL target 0.01), entropy 0.005. Physics 120 Hz, policy 30 Hz. Distillation: Adam lr 5e-4, 2 epochs, gradient length 16.", tb({ x: 0.6, y: 5.15, w: 12.2, h: 1.0, fontSize: 13, color: C.text1 }));
  s.addText("Code: ~/jetbot_ramp (git; branch feat/lidar-camera-fusion) · tools/: swarm.py, pip_replay.py, progress_scan.py, evaluate.py, mosaic.py", tb({ x: 0.6, y: 6.2, w: 12.2, h: 0.6, fontSize: 12, color: C.accent5 }));

  // closing
  s = slide("TITLE");
  s.addText("From one S to any track", { placeholder: "title" });
  s.addText("A camera + LiDAR policy, taught by a privileged teacher and polished with RL, drove 8 never-seen tracks at ~1 m/s without falling, and 133 m in one go. Its limit: ramps that look different (colour) or bend sharply, where the LiDAR-only model wins.", { placeholder: "sub" });
  s.addImage({ path: cov("unseen_tracks_first_drive_C"), x: 7.6, y: 2.25, w: 5.3, h: 2.98, altText: "Model C on an unseen track" });

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
