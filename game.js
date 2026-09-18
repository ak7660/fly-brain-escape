// Mini game (game-hud track): top-down arena + visual-field strip. Pure logic lives in game-logic.js.
// Drawing is split into a static layer (rendered once per size/token change) and a light per-frame layer.
import { loomAngleDeg, rfGain } from "./stimulus.js";
import { createGameCore, threatGeometry, azimuthFromPoint, scoringThreat } from "./game-logic.js";

const TAU = Math.PI * 2;
const RAD = Math.PI / 180;
const LOGICAL = 300; // arena drawing units; the canvas CSS size may differ (e.g. 240px on short windows)
const ARENA = { outerR: 128, rimR: 141 };
const ACTION = { out: 0.2, hold: 0.22, back: 0.42 };
const MARK_S = 1.1;
const CONTACT_S = 0.5;
const DASH = [2, 4];
const NO_DASH = [];
const SIDES = [-1, 1];
const LEGS = [
  [2.2, -3, 6.5, -6.5, 8.2, -10.5],
  [2.8, -0.5, 8, 0.6, 11, 2.4],
  [2.4, 2, 6.8, 6.2, 8.4, 10.4],
];
const VF_TICKS = [-180, -135, -90, -45, 0, 45, 90, 135, 180];

const easeOut = (x) => 1 - Math.pow(1 - clamp01(x), 3);
const easeInOut = (x) => { x = clamp01(x); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

function parseColor(c) {
  c = c.trim();
  if (c.startsWith("#")) {
    let h = c.slice(1);
    if (h.length === 3) h = [...h].map((x) => x + x).join("");
    const n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255, 1];
  }
  const m = c.match(/[\d.]+/g) ?? ["0", "0", "0"];
  return [+m[0], +m[1], +m[2], m[3] === undefined ? 1 : +m[3]];
}
const rgbaOf = ([r, g, b, a0], a) => `rgba(${r},${g},${b},${+(a0 * a).toFixed(3)})`;

/** Every color string the renderer needs, computed once from the tokens (no per-frame string building). */
function readTokens() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  const P = (n) => parseColor(v(n));
  const ink1 = P("--ink-1"), ink2 = P("--ink-2"), bg = P("--bg"), shadow = P("--shadow");
  const cls = ["--class-none", "--class-dodge-right", "--class-dodge-left", "--class-takeoff"].map(P);
  return {
    bg: rgbaOf(bg, 1),
    grid: v("--grid"),
    ink1: rgbaOf(ink1, 1),
    ink1Floor: rgbaOf(ink1, 0.035),
    ink1Clear: rgbaOf(ink1, 0),
    ink1Wing: rgbaOf(ink1, 0.1),
    ink1WingEdge: rgbaOf(ink1, 0.42),
    ink1Vein: rgbaOf(ink1, 0.18),
    ink1Abdomen: rgbaOf(ink1, 0.82),
    ink1Head: rgbaOf(ink1, 0.9),
    ink1Contact: rgbaOf(ink1, 0.6),
    ink1Extent: rgbaOf(ink1, 0.85),
    ink2: rgbaOf(ink2, 1),
    ink2Legs: rgbaOf(ink2, 0.75),
    ink2Rf: rgbaOf(ink2, 0.4),
    ink2Mark: rgbaOf(ink2, 0.8),
    ink3: v("--ink-3"),
    bgBand: rgbaOf(bg, 0.55),
    bgMark: rgbaOf(bg, 0.85),
    shadow,
    mono: v("--font-mono"),
    cls: cls.map((c) => rgbaOf(c, 1)),
    clsRim: cls.map((c) => rgbaOf(c, 0.6)),
    clsBase: cls.map((c) => rgbaOf(c, 0.5)),
    clsFill: cls.map((c) => rgbaOf(c, 0.12)),
    clsEdgeHot: cls.map((c) => rgbaOf(c, 0.9)),
    clsEdge: cls.map((c) => rgbaOf(c, 0.5)),
    clsTrackHot: cls.map((c) => rgbaOf(c, 0.4)),
    clsTrack: cls.map((c) => rgbaOf(c, 0.18)),
  };
}

/** A soft round shadow, pre-blurred once: drawn scaled for discs and the fly (no shadowBlur / ctx.filter per frame). */
function makeShadowSprite(T) {
  const s = document.createElement("canvas");
  s.width = s.height = 64;
  const c = s.getContext("2d");
  const g = c.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, rgbaOf(T.shadow, 1.6));
  g.addColorStop(0.55, rgbaOf(T.shadow, 1.1));
  g.addColorStop(1, rgbaOf(T.shadow, 0));
  c.fillStyle = g;
  c.fillRect(0, 0, 64, 64);
  return s;
}

// ---------------------------------------------------------------- fly glyph
/** Top-down Drosophila, head toward −y, ~30px long at scale 1. Caller sets globalAlpha. */
function drawFly(ctx, T, scale, wingSpread) {
  ctx.save();
  ctx.scale(scale, scale);
  ctx.lineCap = "round";

  ctx.strokeStyle = T.ink2Legs;
  ctx.lineWidth = 0.9;
  for (const s of SIDES) {
    for (const l of LEGS) {
      ctx.beginPath();
      ctx.moveTo(s * l[0], l[1]);
      ctx.lineTo(s * l[2], l[3]);
      ctx.lineTo(s * l[4], l[5]);
      ctx.stroke();
    }
  }

  for (const s of SIDES) {
    ctx.save();
    ctx.translate(s * 1.4, 0.6);
    ctx.rotate(s * (0.2 + 0.75 * wingSpread));
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.bezierCurveTo(s * 5.8, 2.5, s * 6.2, 12.5, s * 2.4, 16.2);
    ctx.bezierCurveTo(s * 0.4, 17.4, s * -0.8, 12, 0, 0);
    ctx.fillStyle = T.ink1Wing;
    ctx.fill();
    ctx.strokeStyle = T.ink1WingEdge;
    ctx.lineWidth = 0.7;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(s * 0.6, 1.5);
    ctx.quadraticCurveTo(s * 3.6, 8, s * 2.6, 14.5);
    ctx.strokeStyle = T.ink1Vein;
    ctx.stroke();
    ctx.restore();
  }

  ctx.save();
  ctx.beginPath();
  ctx.ellipse(0, 7.2, 3.5, 6.4, 0, 0, TAU);
  ctx.fillStyle = T.ink1Abdomen;
  ctx.fill();
  ctx.clip();
  ctx.strokeStyle = T.bgBand;
  ctx.lineWidth = 0.9;
  for (let y = 4.2; y < 13; y += 2.1) {
    ctx.beginPath();
    ctx.moveTo(-4, y);
    ctx.quadraticCurveTo(0, y + 1.1, 4, y);
    ctx.stroke();
  }
  ctx.restore();

  ctx.beginPath();
  ctx.ellipse(0, -1.2, 3.3, 3.9, 0, 0, TAU);
  ctx.fillStyle = T.ink1;
  ctx.fill();

  ctx.beginPath();
  ctx.ellipse(0, -6.3, 2.9, 2.1, 0, 0, TAU);
  ctx.fillStyle = T.ink1Head;
  ctx.fill();
  for (const s of SIDES) {
    ctx.beginPath();
    ctx.ellipse(s * 2.2, -6.6, 1.35, 1.9, s * 0.35, 0, TAU);
    ctx.fillStyle = T.ink2;
    ctx.fill();
  }
  ctx.restore();
}

// ---------------------------------------------------------------- game
export function createGame({ arena, vfield, stim, classes, fps = 60 }) {
  const core = createGameCore({ stim, fps, lingerS: CONTACT_S });
  const outcomeCbs = [];
  const ac = new AbortController();
  const { signal } = ac;
  const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)");
  const geo = { theta: 0, x: 0, y: 0, r: 0, progress: 0 };
  const geoOpts = { outerR: ARENA.outerR };
  const pose = { dx: 0, scale: 1, lift: 0 };
  const ghost = { dx: 0, scale: 1, lift: 0 };
  let action = null; // {classIndex, t0, hold}
  const marks = []; // {kind, t0, cls, dir}
  const added = [];

  let T = readTokens();
  let shadowSprite = makeShadowSprite(T);
  let A = null, V = null; // {ctx, w, h, k, dpr, layer}

  decoratePanel();
  const live = arena.parentElement?.querySelector(".gp-live");
  resize();

  window.addEventListener("resize", () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    if (!A || dpr !== A.dpr || arena.clientWidth !== A.w || vfield.clientWidth !== V.w) resize();
  }, { signal });

  arena.addEventListener("click", (e) => {
    const rect = arena.getBoundingClientRect();
    const k = LOGICAL / rect.width;
    const dx = (e.clientX - rect.left - rect.width / 2) * k;
    const dy = (e.clientY - rect.top - rect.height / 2) * k;
    if (Math.hypot(dx, dy) < 12) return;
    spawn(azimuthFromPoint(dx, dy), e.shiftKey ? 0.02 : 0.04);
  }, { signal });
  arena.style.cursor = "crosshair";

  core.onSpawn((th) => {
    const side = th.cls === 3 ? "front" : th.azimuthDeg < 0 ? "left" : "right";
    announce(`Threat from the ${side}, ${Math.round(th.azimuthDeg)} degrees`);
  });

  function decoratePanel() {
    const panel = arena.parentElement;
    if (!panel || panel.querySelector(".gp-head")) return;
    const head = document.createElement("div");
    head.className = "gp-head";
    head.innerHTML = `<span class="gp-title">Arena</span><span class="gp-hint"><span class="gp-hint-text">Click or </span><kbd>←</kbd><kbd>↑</kbd><kbd>→</kbd><span class="gp-hint-text"> to launch a threat</span></span>`;
    panel.insertBefore(head, arena);
    const cap = document.createElement("div");
    cap.className = "gp-caption";
    cap.innerHTML = `<span>Visual field</span><span class="gp-eyes"><i class="eye-line"></i>eye receptive fields</span>`;
    vfield.before(cap);
    const axis = document.createElement("div");
    axis.className = "gp-axis";
    axis.setAttribute("aria-hidden", "true");
    axis.innerHTML = `<span>−180°</span><span>−90°</span><span>0°</span><span class="gp-axis-fold">Visual field</span><span>+90°</span><span>180°</span>`;
    vfield.after(axis);
    const liveEl = document.createElement("p");
    liveEl.className = "sr-only gp-live";
    liveEl.setAttribute("aria-live", "polite");
    panel.append(liveEl);
    added.push(head, cap, axis, liveEl);
  }

  function announce(text) {
    if (live) live.textContent = text;
  }

  // ---- sizing + static layers
  function setup(canvas, logicalW, logicalH) {
    const w = canvas.clientWidth || logicalW;
    const h = canvas.clientHeight || logicalH;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const layer = document.createElement("canvas");
    layer.width = canvas.width;
    layer.height = canvas.height;
    return { ctx: canvas.getContext("2d"), w, h, dpr, layer, lctx: layer.getContext("2d") };
  }

  function resize() {
    A = setup(arena, LOGICAL, LOGICAL);
    A.k = (A.w / LOGICAL) * A.dpr; // logical units → device pixels
    // labels stay 10 CSS px even when the arena is drawn smaller than its logical size
    const px = +(10 * (LOGICAL / A.w)).toFixed(2);
    A.font = `${px}px ${T.mono}`;
    A.fontB = `500 ${px}px ${T.mono}`;
    V = setup(vfield, 300, 28);
    V.k = V.dpr;
    renderStatic();
    draw(core.now());
  }

  function renderStatic() {
    renderArenaStatic(A.lctx);
    renderVfieldStatic(V.lctx);
  }

  function renderArenaStatic(ctx) {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.setTransform(A.k, 0, 0, A.k, (LOGICAL / 2) * A.k, (LOGICAL / 2) * A.k);

    const floor = ctx.createRadialGradient(0, 0, 0, 0, 0, ARENA.rimR);
    floor.addColorStop(0, T.ink1Floor);
    floor.addColorStop(1, T.ink1Clear);
    ctx.fillStyle = floor;
    ctx.beginPath();
    ctx.arc(0, 0, ARENA.rimR, 0, TAU);
    ctx.fill();

    ctx.lineWidth = 1;
    ctx.strokeStyle = T.grid;
    for (const r of [ARENA.outerR / 3, (2 * ARENA.outerR) / 3, ARENA.outerR]) {
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, TAU);
      ctx.stroke();
    }
    for (let a = -180; a < 180; a += 15) {
      const major = a % 45 === 0;
      const r0 = ARENA.outerR + 2, r1 = ARENA.outerR + (major ? 7 : 4);
      const s = Math.sin(a * RAD), c = -Math.cos(a * RAD);
      ctx.strokeStyle = major ? T.ink3 : T.grid;
      ctx.beginPath();
      ctx.moveTo(s * r0, c * r0);
      ctx.lineTo(s * r1, c * r1);
      ctx.stroke();
    }

    // class sectors on the rim, labeled with the correct response for a threat from that direction
    const fw = stim.front_half_width_deg;
    const sectors = [[-180, -fw, 1, "DODGE →"], [-fw, fw, 3, "TAKEOFF ↑"], [fw, 180, 2, "← DODGE"]];
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.font = A.fontB;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const [a0, a1, c, label] of sectors) {
      ctx.strokeStyle = T.clsRim[c];
      ctx.beginPath();
      ctx.arc(0, 0, ARENA.rimR, (a0 + 3 - 90) * RAD, (a1 - 3 - 90) * RAD);
      ctx.stroke();
      const mid = ((a0 + a1) / 2) * RAD;
      const lr = c === 3 ? ARENA.outerR - 12 : ARENA.outerR - 26;
      ctx.fillStyle = T.cls[c];
      ctx.fillText(label, Math.sin(mid) * lr, -Math.cos(mid) * lr);
    }

    ctx.font = A.font;
    ctx.fillStyle = T.ink2;
    ctx.fillText("+90°", ARENA.outerR - 18, 0);
    ctx.fillText("−90°", -ARENA.outerR + 18, 0);
    ctx.fillText("180°", 0, ARENA.outerR - 11);
  }

  function vx(az) { return 6 + ((az + 180) / 360) * (V.w - 12); }

  function renderVfieldStatic(ctx) {
    const { w, h } = V;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.setTransform(V.k, 0, 0, V.k, 0, 0);
    const base = h - 4;
    const fw = stim.front_half_width_deg;
    ctx.lineWidth = 2;
    for (const [a0, a1, c] of [[-180, -fw, 1], [-fw, fw, 3], [fw, 180, 2]]) {
      ctx.strokeStyle = T.clsBase[c];
      ctx.beginPath();
      ctx.moveTo(vx(a0) + 1, base + 1);
      ctx.lineTo(vx(a1) - 1, base + 1);
      ctx.stroke();
    }
    ctx.lineWidth = 1;
    for (const a of VF_TICKS) {
      ctx.strokeStyle = a % 90 === 0 ? T.ink3 : T.grid;
      const x = Math.round(vx(a)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, base - (a === 0 ? 6 : 3));
      ctx.lineTo(x, base);
      ctx.stroke();
    }
    const amp = base - 5;
    ctx.strokeStyle = T.ink2Rf;
    for (const side of [1, 2]) {
      ctx.beginPath();
      for (let az = -180; az <= 180; az += 3) {
        const y = base - 1 - amp * rfGain(az, side, stim);
        az === -180 ? ctx.moveTo(vx(az), y) : ctx.lineTo(vx(az), y);
      }
      ctx.stroke();
    }
  }

  // ---- game API
  function spawn(azimuthDeg, lv = 0.04, approachS = 0.6) {
    return core.spawn(azimuthDeg, lv, approachS);
  }

  function emit(ev) {
    const name = classes[ev.classIndex] ?? "none";
    const out = { ...ev, name, expectedName: classes[ev.expected] ?? "none", correct: ev.kind === "correct" };
    // fore or aft of the fly, on the half away from the threat: clear of both the threat and a sideways dodge
    const dir = ev.threat && Math.abs(ev.threat.azimuthDeg) <= 90 ? Math.PI : 0;
    marks.push({ kind: ev.kind, t0: core.now(), cls: ev.classIndex, dir });
    const pretty = name.replace("_", " ");
    announce(
      ev.kind === "miss" ? "Missed: the threat made contact"
        : ev.kind === "false_alarm" ? `False alarm: ${pretty} with no threat`
          : `${pretty}, ${out.correct ? "correct" : "wrong"}, ${ev.latencyMs} milliseconds before contact`,
    );
    for (const cb of outcomeCbs) cb(out);
  }

  function tick(tSim) {
    const evs = core.tick(tSim);
    for (let i = 0; i < evs.length; i++) emit(evs[i]);
    while (marks.length && tSim - marks[0].t0 > MARK_S) marks.shift();
    if (action && tSim - action.t0 > ACTION.out + action.hold + ACTION.back) action = null;
  }

  function resolve(decision) {
    const ev = core.resolve(decision);
    if (!ev) return null;
    // hold the evasive pose until the threat has passed, then ease home
    const t0 = core.now();
    const hold = ev.threat ? Math.max(ACTION.hold, ev.threat.tCollision + 0.2 - (t0 + ACTION.out)) : ACTION.hold;
    action = { classIndex: decision.classIndex, t0, hold };
    emit(ev);
    return ev;
  }

  // ---- motion
  function actionPose(t, out) {
    out.dx = 0; out.scale = 1; out.lift = 0;
    if (!action) return out;
    const dt = t - action.t0;
    const end = ACTION.out + action.hold;
    let e;
    if (reduced?.matches) e = dt >= 0 && dt < end ? 1 : 0; // snap, no easing
    else if (dt < 0) e = 0;
    else if (dt < ACTION.out) e = easeOut(dt / ACTION.out);
    else if (dt < end) e = 1;
    else e = 1 - easeInOut((dt - end) / ACTION.back);
    const c = action.classIndex;
    if (c === 1) out.dx = 40 * e;
    else if (c === 2) out.dx = -40 * e;
    else if (c === 3) { out.scale = 1 + 0.5 * e; out.lift = e; }
    return out;
  }

  // ---- per-frame drawing
  function drawArena(t) {
    const { ctx } = A;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, arena.width, arena.height);
    ctx.drawImage(A.layer, 0, 0);
    ctx.setTransform(A.k, 0, 0, A.k, (LOGICAL / 2) * A.k, (LOGICAL / 2) * A.k);

    const list = core.all();
    const hot = scoringThreat(list, t, 0);
    for (let i = list.length - 1; i >= 0; i--) if (list[i].tSpawn <= t) drawThreat(ctx, list[i], t, list[i] === hot);
    drawFlyWithMotion(ctx, t);
    for (let i = 0; i < marks.length; i++) drawMark(ctx, marks[i], t);
  }

  function drawThreat(ctx, th, t, isHot) {
    const past = t - th.tCollision;
    const c = th.cls;
    if (past >= 0) {
      const k = clamp01(past / CONTACT_S);
      if (th.outcome?.kind === "correct") {
        // the fly got out of the way: the object sails through and fades
        threatGeometry(th, th.tCollision - 1e-3, geoOpts, geo);
        const a = th.azimuthDeg * RAD;
        const d = -ARENA.outerR * 0.6 * (reduced?.matches ? 1 : easeOut(k));
        ctx.globalAlpha = 1 - k;
        disc(ctx, Math.sin(a) * d, -Math.cos(a) * d, geo.r * (1 - 0.3 * k), c, false);
        ctx.globalAlpha = 1;
      } else if (!reduced?.matches) {
        ctx.strokeStyle = T.ink1Contact;
        ctx.globalAlpha = 1 - k;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(0, 0, 18 + 60 * easeOut(k), 0, TAU);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      return;
    }
    threatGeometry(th, t, geoOpts, geo);
    const a = th.azimuthDeg * RAD;
    const sa = Math.sin(a), ca = Math.cos(a);
    ctx.strokeStyle = isHot ? T.clsTrackHot[c] : T.clsTrack[c];
    ctx.lineWidth = 1;
    ctx.setLineDash(DASH);
    ctx.beginPath();
    ctx.moveTo(sa * ARENA.outerR, -ca * ARENA.outerR);
    ctx.lineTo(geo.x, geo.y);
    ctx.stroke();
    ctx.setLineDash(NO_DASH);

    ctx.globalAlpha = clamp01((t - th.tSpawn) / 0.08);
    disc(ctx, geo.x, geo.y, geo.r, c, isHot);
    ctx.globalAlpha = 1;

    if (isHot) {
      // θ readout on the side away from the fly, nudged off the approach track
      const out = geo.r + 11;
      const lx = geo.x + sa * out + ca * 12, ly = geo.y - ca * out + sa * 12;
      if (lx * lx + ly * ly < (ARENA.rimR - 4) ** 2) {
        ctx.font = A.font;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = T.ink2;
        ctx.fillText(`${Math.round(geo.theta)}°`, lx, ly);
      }
    }
  }

  function disc(ctx, x, y, r, c, isHot) {
    const s = r * 1.9;
    ctx.drawImage(shadowSprite, x - s, y - s + 2, 2 * s, 2 * s);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, TAU);
    ctx.fillStyle = T.bg;
    ctx.fill();
    ctx.fillStyle = T.clsFill[c];
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, Math.max(r - 0.75, 0.5), 0, TAU);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = isHot ? T.clsEdgeHot[c] : T.clsEdge[c];
    ctx.stroke();
  }

  function drawFlyWithMotion(ctx, t) {
    actionPose(t, pose);
    if (action && (action.classIndex === 1 || action.classIndex === 2) && !reduced?.matches) {
      for (let k = 3; k >= 1; k--) {
        actionPose(t - k * 0.04, ghost);
        if (Math.abs(ghost.dx - pose.dx) < 3) continue;
        ctx.save();
        ctx.translate(ghost.dx, 0);
        ctx.globalAlpha = (0.16 * (4 - k)) / 3;
        drawFly(ctx, T, 1, 0);
        ctx.restore();
      }
    }
    // ground shadow: as the fly takes off it falls behind, spreads and fades
    const lift = pose.lift;
    ctx.globalAlpha = 0.8 - 0.45 * lift;
    const sw = 10 + 8 * lift, sh = 17 + 8 * lift;
    ctx.drawImage(shadowSprite, pose.dx + 6 * lift - sw, 7 + 12 * lift - sh, 2 * sw, 2 * sh);

    ctx.save();
    ctx.globalAlpha = 1 - 0.4 * lift;
    ctx.translate(pose.dx, -10 * lift);
    drawFly(ctx, T, pose.scale, lift);
    ctx.restore();
    ctx.globalAlpha = 1;
  }

  function drawMark(ctx, m, t) {
    const k = (t - m.t0) / MARK_S;
    if (k < 0 || k > 1) return;
    const ok = m.kind === "correct";
    ctx.save();
    ctx.globalAlpha = k < 0.15 ? k / 0.15 : 1 - clamp01((k - 0.6) / 0.4);
    const d = 34 + (reduced?.matches ? 0 : 8 * easeOut(k));
    ctx.translate(Math.sin(m.dir) * d, -Math.cos(m.dir) * d);
    ctx.beginPath();
    ctx.arc(0, 0, 9, 0, TAU);
    ctx.fillStyle = T.bgMark;
    ctx.fill();
    ctx.strokeStyle = ok ? T.clsEdgeHot[m.cls] : T.ink2Mark;
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.strokeStyle = T.ink1;
    ctx.lineWidth = 1.6;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    if (ok) {
      ctx.moveTo(-3.6, 0.2);
      ctx.lineTo(-1, 2.8);
      ctx.lineTo(3.8, -2.6);
    } else {
      ctx.moveTo(-3, -3); ctx.lineTo(3, 3);
      ctx.moveTo(3, -3); ctx.lineTo(-3, 3);
    }
    ctx.stroke();
    ctx.restore();
  }

  function drawVfield(t) {
    const { ctx, h } = V;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, vfield.width, vfield.height);
    ctx.drawImage(V.layer, 0, 0);
    ctx.setTransform(V.k, 0, 0, V.k, 0, 0);
    const base = h - 4;
    const list = core.all();
    for (let i = 0; i < list.length; i++) {
      const th = list[i];
      if (!(th.tSpawn <= t && t < th.tCollision)) continue;
      const theta = Math.min(loomAngleDeg(t, th.lv, th.tCollision), 359);
      const lo = th.azimuthDeg - theta / 2, hi = th.azimuthDeg + theta / 2;
      ctx.fillStyle = T.ink1Extent;
      // split across the ±180° seam
      extent(ctx, Math.max(lo, -180), Math.min(hi, 180), base);
      if (lo < -180) extent(ctx, lo + 360, 180, base);
      if (hi > 180) extent(ctx, -180, hi - 360, base);
      ctx.fillStyle = T.cls[th.cls];
      ctx.fillRect(Math.round(vx(th.azimuthDeg)) - 0.5, base - 16, 1.5, 13);
    }
  }

  function extent(ctx, a0, a1, base) {
    const x0 = vx(a0), w = Math.max(vx(a1) - x0, 1.5), y = base - 13, hh = 7, r = Math.min(2, w / 2);
    ctx.beginPath();
    ctx.moveTo(x0 + r, y);
    ctx.arcTo(x0 + w, y, x0 + w, y + hh, r);
    ctx.arcTo(x0 + w, y + hh, x0, y + hh, r);
    ctx.arcTo(x0, y + hh, x0, y, r);
    ctx.arcTo(x0, y, x0 + w, y, r);
    ctx.closePath();
    ctx.fill();
  }

  function draw(tSim = core.now()) {
    drawArena(tSim);
    drawVfield(tSim);
  }

  return {
    spawn,
    threats: () => core.threats(),
    tick,
    resolve,
    onOutcome(cb) { outcomeCbs.push(cb); },
    onSpawn(cb) { core.onSpawn(cb); },
    score: () => core.score(),
    draw,
    reset() { core.reset(); action = null; marks.length = 0; announce("Arena reset"); draw(); },
    refreshTokens() { T = readTokens(); shadowSprite = makeShadowSprite(T); resize(); },
    dispose() {
      ac.abort();
      for (const el of added) el.remove();
      arena.style.cursor = "";
      outcomeCbs.length = 0;
    },
  };
}
