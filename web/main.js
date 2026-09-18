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
import { createHud } from "./hud.js";

const params = new URLSearchParams(location.search);
if (params.get("selftest") === "1") location.replace("dev/selftest.html");

const $ = (id) => document.getElementById(id);
const DRIVE_NOISE_SD = 0.03; // same input noise as training (flybrain/stimulus.sample_episodes)
const ACTIVE_DELTA = 0.01; // a neuron counts as "active" when it is this far above its resting rate
const FPS_FLOOR = 45; // below this for 2 s → degrade to low quality

async function main() {
  await document.fonts?.ready; // scene reads CSS tokens at construction
  const bundle = await loadBundle(params.get("bundle") ?? "assets/circuit_v1/");
  const { manifest } = bundle;
  const { classes, stimulus: stim, decision: rule, counts } = manifest;
  const FPS = manifest.sim.fps;
  const rMax = manifest.sim.r_max;

  const model = createModel(bundle);
  const decider = createDecider({ classes, p: rule.p, frames: rule.frames });
  const scene = createScene($("scene"), bundle);
  const game = createGame({ arena: $("arena"), vfield: $("vfield"), stim, classes, fps: FPS });
  const hud = createHud(
    { title: $("title"), controls: $("controls"), decisionPanel: $("decision-panel"), tooltip: $("tooltip") },
    { classes, counts, decision: rule },
  );

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
    game.resolve(decision);
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
    if (noiseOn) for (let i = 0; i < drive.length; i++) drive[i] += DRIVE_NOISE_SD * gauss();
    model.step(drive);
    const decision = decider.update(model.probs);
    if (decision) onDecision(decision);
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
    else if (name === "layout") scene.setLayout(arg);
    else if (name === "pause") paused = !!arg;
    else if (name === "slowmo") speed = arg ? 0.25 : 1;
    else if (name === "noise") noiseOn = !!arg;
    else if (name === "bloom") scene.setBloom(!!arg);
    else if (name === "reset") reset();
  });

  // ---- hover: pick a neuron in the 3D view (bound to the WebGL canvas, not the window)
  const canvas = scene.canvas;
  canvas.addEventListener("pointermove", (e) => {
    const i = scene.pick(e.clientX, e.clientY);
    if (i < 0) { hud.showTooltip(0, 0, null); return; }
    hud.showTooltip(e.clientX, e.clientY, {
      index: i, type: bundle.info.types[bundle.typeId[i]], instance: bundle.info.instance[i],
      superclass: bundle.info.superclass[i], role: bundle.role[i], layer: bundle.layer[i], sign: bundle.sign[i],
      activity: model.r[i], rMax,
    });
  });
  canvas.addEventListener("pointerleave", () => hud.showTooltip(0, 0, null));
  // (the scene resizes itself via a ResizeObserver)

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
  let last = performance.now(), fpsEma = FPS, slowSince = null, lowQuality = false;
  requestAnimationFrame(function loop(now) {
    const real = Math.min((now - last) / 1000, 0.1);
    last = now;
    fpsEma += (1 / Math.max(real, 1e-3) - fpsEma) * 0.05;
    if (!paused) {
      acc += real * speed;
      let n = 0;
      while (acc >= dt && n++ < 4) { step(); acc -= dt; }
      if (n >= 4) acc = 0; // never spiral: drop time rather than fall further behind
    }
    scene.setActivity(model.r, rMax);
    scene.render(real);
    game.draw(tSim);
    let active = 0;
    for (let i = 0; i < bundle.N; i++) if (model.r[i] - bundle.r0[i] > ACTIVE_DELTA) active++;
    hud.setStats({ fps: fpsEma, activeNeurons: active });

    if (!lowQuality && fpsEma < FPS_FLOOR) {
      slowSince ??= now;
      if (now - slowSince > 2000) {
        lowQuality = true;
        scene.setQuality("low");
        document.body.classList.add("hud-lowq");
      }
    } else slowSince = null;
    requestAnimationFrame(loop);
  });
}

main().catch((err) => {
  console.error(err);
  const box = document.createElement("pre");
  box.className = "load-error";
  box.textContent = `Could not start the fly brain demo:\n${err.message}`;
  document.body.append(box);
});
