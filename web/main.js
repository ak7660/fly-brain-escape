// Integration entry point: the real connectome model, the 3D scene, the game and the HUD (see CONTRACT.md).
//
// Frame order (fixed 60 Hz sim step, decoupled from rendering):
//   game.tick(t) → drive = inputDrive(t, game.threats()) → model.step(drive) → decider.update(probs)
//   → on decision: game.resolve + scene.highlightDecision → hud.setProbs
// Rendering each animation frame: scene.setActivity(model.r) → scene.render(wall-clock dt) → game.draw(t) → hud.setStats.
import { loadBundle } from "./data.js";
import { createModel } from "./model.js";
import { inputDrive } from "./stimulus.js";
import { createDecider } from "./decision.js";
import { createScene } from "./scene.js";
import { createGame } from "./game.js";
import { createHud, LAYER_NAMES } from "./hud.js";
import { neuronTour } from "./tour.js";

const params = new URLSearchParams(location.search);
if (params.get("selftest") === "1") location.replace("dev/selftest.html");

const $ = (id) => document.getElementById(id);
const fmtInt = new Intl.NumberFormat("en-US");
const ACTIVE_DELTA = 0.01; // a neuron counts as "active" when it is this far above its resting rate
const FPS_FLOOR = 45;          // below this for 2 s → degrade to low quality
const FPS_RESTORE = 57;       // above this for 6 s at low quality → try high again
const QUALITY_GRACE_MS = 6000; // ignore the load spike before judging frame rate
const MAX_RESTORES = 2;       // don't flap between tiers forever

// ---- boot progress: the markup is in index.html so it paints before this module is even fetched
const boot = {
  bar: document.querySelector("[data-boot-bar]"),
  fill: document.querySelector("[data-boot-fill]"),
  pct: document.querySelector("[data-boot-pct]"),
  set(fraction) {
    const p = Math.max(0, Math.min(1, fraction));
    if (this.bar?.classList.contains("is-indeterminate")) { // real progress from here on
      this.bar.classList.remove("is-indeterminate");
      if (this.fill) { // land on empty without animating back from wherever the sweep was
        this.fill.style.transition = "none";
        this.fill.style.transform = "scaleX(0)";
        void this.fill.offsetWidth;
        this.fill.style.transition = "";
      }
    }
    if (this.fill) this.fill.style.transform = `scaleX(${p})`;
    if (this.pct) this.pct.textContent = `${Math.round(p * 100)}%`;
    this.bar?.setAttribute("aria-valuenow", String(Math.round(p * 100)));
  },
  done() {
    document.body.classList.remove("is-booting");
  },
};

async function main() {
  const requested = params.get("bundle") ?? "assets/circuit_v1/";
  if (/^[a-z]+:|^\/\//i.test(requested)) throw new Error(`?bundle must be a relative path, got ${requested}`);
  // the webfonts load in parallel with the bundle; the scene and the canvas game read CSS tokens at construction
  const [bundle] = await Promise.all([
    loadBundle(requested, fetch, { onProgress: (f) => boot.set(f) }),
    document.fonts?.ready,
  ]);
  boot.done();
  const { manifest } = bundle;
  const { classes, stimulus: stim, decision: rule, counts } = manifest;
  const FPS = manifest.sim.fps;
  const rMax = manifest.sim.r_max;

  const model = createModel(bundle);
  const decider = createDecider({ classes, p: rule.p, frames: rule.frames });
  const scene = createScene($("scene"), bundle);
  const game = createGame({ arena: $("arena"), vfield: $("vfield"), stim, classes, fps: FPS });
  const hud = createHud(
    {
      title: $("title"), controls: $("controls"), decisionPanel: $("decision-panel"), tooltip: $("tooltip"),
      layerLabels: $("layer-labels"),
    },
    { classes, counts, decision: rule },
  );

  const noiseSd = stim.noise_sd ?? 0.03; // training-time input noise, from the manifest
  const drive = new Float64Array(bundle.NI);
  const dnWeights = new Float32Array(bundle.ND);
  let noiseOn = false;

  const sideOf = (azimuthDeg) => {
    const [left, right] = stim.rf_centers_deg; // −45, +45
    if (azimuthDeg < left) return 1;
    if (azimuthDeg > right) return 2;
    return "both";
  };

  game.onSpawn((threat) => {
    decider.reset();
    hud.showIncoming(threat);
    scene.flashInputs(sideOf(threat.azimuthDeg));
  });
  game.onOutcome((outcome) => hud.showDecision({ ...outcome, score: game.score() }));

  function onDecision(decision) {
    if (!game.resolve(decision)) return; // ignored (e.g. within the linger window after a miss)
    // rank DNs by what the decision class actually read from them: W_out[c, j] · (r − rest), positives only
    const c = decision.classIndex;
    for (let j = 0; j < bundle.ND; j++) {
      const dn = bundle.dnIdx[j];
      dnWeights[j] = Math.max(0, bundle.Wout[c * bundle.ND + j] * (model.r[dn] - bundle.r0[dn]));
    }
    scene.highlightDecision(c, dnWeights);
  }

  // ---- fixed-step simulation
  const dt = 1 / FPS;
  let frame = 0, tSim = 0, paused = false, speed = 1, acc = 0;
  let gaussSpare = null;
  const gauss = () => {
    if (gaussSpare !== null) { const g = gaussSpare; gaussSpare = null; return g; }
    const u = Math.random() || 1e-12, v = Math.random();
    const m = Math.sqrt(-2 * Math.log(u));
    gaussSpare = m * Math.sin(2 * Math.PI * v);
    return m * Math.cos(2 * Math.PI * v);
  };

  function step() {
    tSim = ++frame * dt;
    demoTick();
    game.tick(tSim);
    inputDrive(tSim, game.threats(), bundle.inputKind, bundle.inputSide, stim, drive);
    if (noiseOn) for (let i = 0; i < drive.length; i++) drive[i] += noiseSd * gauss();
    model.step(drive);
    const decision = decider.update(model.probs);
    if (decision) onDecision(decision);
    // once per SIMULATED frame: the scene's activity baseline adapts per call, so calling it per rendered
    // frame would decay the glow while paused and adapt 4x too fast in slow motion
    scene.setActivity(model.r, rMax);
    hud.setProbs(model.probs);
  }

  function reset() {
    model.reset();
    decider.reset();
    game.reset();
    hud.resetDecision();
    frame = 0; tSim = 0; acc = 0; nextDemo = 1.5;
  }

  hud.onControl((name, arg) => {
    if (name === "spawn") game.spawn(arg.azimuthDeg, arg.lv);
    else if (name === "layout") { layoutName = arg; scene.setLayout(arg); describe(true); }
    else if (name === "pause") paused = !!arg;
    else if (name === "slowmo") speed = arg ? 0.25 : 1;
    else if (name === "noise") noiseOn = !!arg;
    else if (name === "bloom") scene.setBloom(!!arg);
    else if (name === "reset") reset();
  });

  // ---- hover: pick a neuron in the 3D view (bound to the WebGL canvas, not the window)
  const canvas = scene.canvas;
  const neuronInfo = (i) => ({
    index: i, type: bundle.info.types[bundle.typeId[i]], instance: bundle.info.instance[i],
    superclass: bundle.info.superclass[i], role: bundle.role[i], layer: bundle.layer[i], sign: bundle.sign[i],
    activity: model.r[i], rMax,
  });
  canvas.addEventListener("pointermove", (e) => {
    if (tourAt >= 0) clearTour(); // the pointer takes over from the keyboard selection
    const i = scene.pick(e.clientX, e.clientY);
    if (i < 0) { hud.showTooltip(0, 0, null); return; }
    hud.showTooltip(e.clientX, e.clientY, neuronInfo(i));
  });
  canvas.addEventListener("pointerleave", () => { if (tourAt < 0) hud.showTooltip(0, 0, null); });
  // (the scene resizes itself via a ResizeObserver)

  // ---- keyboard: the 3D view orbits, zooms and steps through neurons without a pointer.
  // Arrows only orbit when the canvas has *keyboard* focus, so clicking the scene and pressing ← still
  // launches a threat; preventDefault keeps those keys from reaching the HUD's global handler.
  const tour = neuronTour(bundle);
  const ORBIT_STEP = 0.1, ZOOM_STEP = 1.15;
  const ORBIT = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
  let tourAt = -1;

  function showTour() {
    const i = tour[tourAt];
    scene.setSelected(i);
    const p = scene.screenPos(i);
    const r = canvas.getBoundingClientRect();
    if (p?.visible) hud.showTooltip(r.left + p.x, r.top + p.y, neuronInfo(i));
    else hud.showTooltip(0, 0, null);
  }
  function stepTour(step) {
    if (!tour.length) return;
    tourAt = tourAt < 0 ? (step > 0 ? 0 : tour.length - 1) : (tourAt + step + tour.length) % tour.length;
    showTour();
    describe(true);
  }
  function clearTour() {
    tourAt = -1;
    scene.setSelected(-1);
    hud.showTooltip(0, 0, null);
    describe(true);
  }

  // Was the canvas reached by Tab or by a click? Firefox matches neither :focus-visible nor a focus event on a
  // canvas, so track it from the two inputs that can put focus there. Only a Tab arrival takes over the arrow
  // keys (and shows the focus ring); after a click on the view they keep launching threats, as they always did.
  let pointerFocus = false;
  const kbFocused = () => document.activeElement === canvas && !pointerFocus;
  const refreshRing = () => canvas.classList.toggle("is-kb-focus", kbFocused());
  canvas.addEventListener("pointerdown", () => { pointerFocus = true; refreshRing(); });
  addEventListener("keyup", (e) => { if (e.key === "Tab") { pointerFocus = false; refreshRing(); } }, true);
  addEventListener("focusout", () => requestAnimationFrame(refreshRing), true);

  canvas.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || e.defaultPrevented) return;
    const key = e.key;
    const dir = ORBIT[key];
    if (dir && kbFocused()) {
      e.preventDefault();
      scene.orbit(dir[0] * ORBIT_STEP, dir[1] * ORBIT_STEP);
      if (tourAt >= 0) showTour();
      return;
    }
    if (key === "+" || key === "=" || key === "-" || key === "_") {
      e.preventDefault();
      scene.zoomBy(key === "+" || key === "=" ? 1 / ZOOM_STEP : ZOOM_STEP);
      if (tourAt >= 0) showTour();
      return;
    }
    if (key === "]" || key === "[") { e.preventDefault(); stepTour(key === "]" ? 1 : -1); return; }
    if (key === "Escape" && tourAt >= 0) { e.preventDefault(); clearTour(); }
  });

  // ---- what the 3D view currently shows, for screen readers
  let layoutName = "anatomy", activeNow = 0, lastDescribed = 0;
  function describe(now = false) {
    if (!now && performance.now() - lastDescribed < 2000) return;
    lastDescribed = performance.now();
    const sel = tourAt >= 0 ? tour[tourAt] : -1;
    const parts = [
      `Fly brain circuit, ${layoutName} view`,
      `${fmtInt.format(bundle.N)} neurons, ${fmtInt.format(activeNow)} active`,
    ];
    if (sel >= 0) {
      const layerName = (LAYER_NAMES[bundle.layer[sel]] ?? "").toLowerCase();
      parts.push(`selected ${bundle.info.types[bundle.typeId[sel]]}, ${layerName}, ${tourAt + 1} of ${tour.length}`);
    }
    scene.setLabel(`${parts.join(". ")}.`, now);
  }

  // ---- ?demo=1: hands-free threats for screen recordings and slides
  const demo = params.get("demo") === "1";
  const demoScript = [[-90, 0.04], [0, 0.04], [90, 0.04], [-60, 0.03], [15, 0.02], [120, 0.05]];
  let demoIdx = 0, nextDemo = 1.5;
  function demoTick() {
    if (!demo || tSim < nextDemo || game.threats().length) return;
    const [az, lv] = demoScript[demoIdx++ % demoScript.length];
    game.spawn(az, lv);
    nextDemo = tSim + 2.2;
  }

  // ---- render loop with an automatic quality ladder
  const startedAt = performance.now();
  let last = startedAt, fpsEma = FPS, slowSince = null, fastSince = null, lowQuality = false, restores = 0;
  let settleUntil = 0;
  window.__tier = "high";
  const TIERS = ["high", "medium", "low"]; // medium keeps the bloom and trims decorative geometry
  let tier = 0;
  function setTier(next) {
    tier = Math.max(0, Math.min(TIERS.length - 1, next));
    lowQuality = tier > 0;
    slowSince = fastSince = null;
    // the smoothed frame rate lags a tier change by ~20 frames; without a settle window the ladder would
    // step straight to the bottom on the old readings
    settleUntil = performance.now() + 2500;
    fpsEma = FPS;
    scene.setQuality(TIERS[tier]);
    document.body.classList.toggle("hud-lowq", tier === 2);
    window.__tier = TIERS[tier];
  }
  requestAnimationFrame(function loop(now) {
    const real = Math.min((now - last) / 1000, 0.1);
    last = now;
    fpsEma += (1 / Math.max(real, 1e-3) - fpsEma) * 0.05;
    if (!paused) {
      acc += real * speed;
      let n = 0;
      while (acc >= dt && n < 4) { step(); acc -= dt; n++; }
      if (acc >= dt) acc = 0; // hit the cap: drop time rather than spiral further behind
    }
    scene.render(real);
    hud.setLayerAnchors(scene.layerAnchors());
    game.draw(tSim);
    let active = 0;
    for (let i = 0; i < bundle.N; i++) if (model.r[i] - bundle.r0[i] > ACTIVE_DELTA) active++;
    hud.setStats({ fps: fpsEma, activeNeurons: active });
    activeNow = active;
    describe();
    if (tourAt >= 0) showTour(); // the tooltip follows its neuron as the view moves

    // Quality ladder with hysteresis. The first seconds after load are always slow (bundle parse, shader
    // compile, texture upload), so ignore them: an early dip used to pin the whole session to low quality.
    if (now - startedAt > QUALITY_GRACE_MS && now > settleUntil) {
      if (fpsEma < FPS_FLOOR && tier < TIERS.length - 1) {
        slowSince ??= now;
        fastSince = null;
        if (now - slowSince > 2000) setTier(tier + 1);
      } else if (fpsEma > FPS_RESTORE && tier > 0 && restores < MAX_RESTORES) {
        fastSince ??= now;
        slowSince = null;
        if (now - fastSince > 8000) { restores++; setTier(tier - 1); }
      } else { slowSince = null; fastSince = null; }
    }
    requestAnimationFrame(loop);
  });
}

function showError(err) {
  console.error(err);
  if (document.querySelector(".load-error")) return;
  boot.bar?.classList.add("is-stalled");
  const box = document.createElement("div");
  box.className = "load-error";
  box.setAttribute("role", "alert");
  const detail = String(err?.message ?? err ?? "unknown error");
  box.innerHTML = `
    <p class="le-head">The demo could not start.</p>
    <p class="le-body">Either the 4,296-neuron bundle or the three.js library failed to load. three.js comes from a
      CDN, so check the connection and try again — the browser console has the full story.</p>
    <p class="le-detail"></p>
    <button type="button" class="le-retry">Try again</button>`;
  box.querySelector(".le-detail").textContent = detail;
  box.querySelector(".le-retry").addEventListener("click", () => location.reload());
  document.body.append(box);
  box.querySelector(".le-retry").focus();
}
// a failed module import (blocked CDN, offline) rejects before main() runs, so listen globally too
addEventListener("error", (e) => showError(e.error ?? e.message));
addEventListener("unhandledrejection", (e) => showError(e.reason));
main().catch(showError);
