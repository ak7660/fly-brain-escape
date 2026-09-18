// Parity with flybrain/sim_numpy.py through tests/fixtures/parity_v1.json (made by flybrain/export.py:make_parity_fixture).
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadBundle } from "../data.js";
import { createModel } from "../model.js";
import { inputDrive } from "../stimulus.js";
import { BUNDLE_DIR, fsFetch, loadFixture } from "./helpers.mjs";

const bundle = await loadBundle(BUNDLE_DIR, fsFetch());
const fx = await loadFixture();
const { fps } = bundle.manifest.sim;
const stim = bundle.manifest.stimulus;
const threats = fx.threats.map((t) => ({ azimuthDeg: t.azimuth_deg, lv: t.lv, tSpawn: t.t_spawn, tCollision: t.t_collision }));

const argmax = (a) => a.reduce((best, v, i) => (v > a[best] ? i : best), 0);

function replay(model) {
  const drive = new Float64Array(bundle.NI);
  const err = { logits: 0, rates: 0, ratesFull: 0, argmaxMismatch: [] };
  for (let f = 0; f < fx.frames; f++) {
    inputDrive(f / fps, threats, bundle.inputKind, bundle.inputSide, stim, drive);
    model.step(drive);
    const want = fx.logits[f];
    for (let c = 0; c < bundle.C; c++) err.logits = Math.max(err.logits, Math.abs(model.logits[c] - want[c]));
    if (argmax([...model.logits]) !== argmax(want)) err.argmaxMismatch.push(f);
    fx.sample_neurons.forEach((n, j) => (err.rates = Math.max(err.rates, Math.abs(model.r[n] - fx.rates_sampled[f][j]))));
    const full = fx.rates_full[String(f)];
    if (full) for (let i = 0; i < bundle.N; i++) err.ratesFull = Math.max(err.ratesFull, Math.abs(model.r[i] - full[i]));
  }
  return err;
}

// The real bundle has inputGain [1,1], uniform alpha and rates far below r_max, so the fixture cannot see per-kind
// input gain, per-neuron alpha or the upper clip. This tiny bundle exercises all three against a dense reference.
function tinyBundle() {
  const N = 4, NI = 2, ND = 2, C = 3;
  const dense = [
    [0.0, 0.5, -0.8, 0.0],
    [0.3, 0.0, 0.0, -0.4],
    [1.2, -0.6, 0.0, 0.25],
    [0.0, 0.9, 0.7, 0.1], // self-connection too
  ];
  const rowPtr = [0], col = [], val = [];
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) if (dense[i][j] !== 0) (col.push(j), val.push(dense[i][j]));
    rowPtr.push(col.length);
  }
  return {
    dense,
    bundle: {
      manifest: { sim: { substeps: 3, r_max: 1.0 } },
      N, E: col.length, NI, ND, C,
      rowPtr: Uint32Array.from(rowPtr), col: Uint32Array.from(col), val: Float32Array.from(val),
      bias: Float32Array.from([0.05, -0.1, 0.2, 3.0]), // neuron 3 saturates at r_max
      alpha: Float32Array.from([0.1, 0.3, 0.5, 0.9]),
      r0: Float32Array.from([0.2, 0.0, 0.4, 0.1]),
      inputIdx: Uint32Array.from([2, 0]),
      inputKind: Uint32Array.from([1, 0]),
      inputGain: Float32Array.from([2.0, 0.5]),
      dnIdx: Uint32Array.from([3, 1]),
      Wout: Float32Array.from([1, -1, 0.5, 2, -0.25, 0.75]),
      bOut: Float32Array.from([0.1, -0.2, 0.3]),
    },
  };
}

test("tiny hand-built bundle matches a naive dense reference (per-kind gain, per-neuron alpha, r_max clip)", () => {
  const { dense, bundle: tb } = tinyBundle();
  const f32 = (x) => Math.fround(x);
  const m = createModel(tb);
  let r = Array.from(tb.r0, Number);
  let saturated = false, clippedLow = false;
  for (let frame = 0; frame < 25; frame++) {
    const drive = [0.7 * Math.sin(frame / 3) + 0.4, 0.9 * ((frame % 5) / 4)];
    m.step(Float64Array.from(drive));
    // reference: independent dense loops, straight from the CLAUDE.md frame contract
    const u = [0, 0, 0, 0];
    for (let j = 0; j < tb.NI; j++) u[tb.inputIdx[j]] = f32([2.0, 0.5][tb.inputKind[j]]) * drive[j];
    for (let s = 0; s < 3; s++) {
      const next = [];
      for (let i = 0; i < 4; i++) {
        let cur = f32(tb.bias[i]) + u[i];
        for (let j = 0; j < 4; j++) cur += f32(dense[i][j]) * r[j];
        if (cur > 1.0) saturated = true;
        if (cur < 0) clippedLow = true;
        const a = tb.alpha[i];
        next.push((1 - a) * r[i] + a * Math.min(Math.max(cur, 0), 1.0));
      }
      r = next;
    }
    const logits = [0, 1, 2].map((c) => tb.bOut[c] + tb.Wout[c * 2] * r[3] + tb.Wout[c * 2 + 1] * r[1]);
    for (let i = 0; i < 4; i++) assert.ok(Math.abs(m.r[i] - r[i]) < 1e-12, `frame ${frame} r[${i}] ${m.r[i]} vs ${r[i]}`);
    for (let c = 0; c < 3; c++) assert.ok(Math.abs(m.logits[c] - logits[c]) < 1e-12, `frame ${frame} logit ${c}`);
  }
  assert.ok(saturated && clippedLow, "the tiny circuit must exercise both clip bounds");
  assert.ok(m.r[3] > 0.99, "neuron 3 sits at r_max");
});

test("step() throws if drive is shorter than NI", () => {
  const m = createModel(bundle);
  assert.throws(() => m.step(new Float64Array(bundle.NI - 1)), /drive/);
});

test("model starts at r0 with the contract's shape", () => {
  const m = createModel(bundle);
  assert.ok(m.r instanceof Float64Array && m.r.length === bundle.N);
  assert.ok(m.logits instanceof Float64Array && m.logits.length === bundle.C);
  assert.ok(m.probs instanceof Float64Array && m.probs.length === bundle.C);
  for (let i = 0; i < bundle.N; i++) if (m.r[i] !== bundle.r0[i]) assert.fail(`r[${i}] != r0`);
});

test("parity: logits, argmax, sampled and full rates match the fixture on every frame", () => {
  const err = replay(createModel(bundle));
  console.log(`  parity max |err|: logits ${err.logits.toExponential(2)}, rates_sampled ${err.rates.toExponential(2)}, rates_full ${err.ratesFull.toExponential(2)}`);
  assert.deepEqual(err.argmaxMismatch, [], "argmax differs on these frames");
  // fx.tolerance (1e-3) is about the size of the logits themselves (~5e-3); the fixture keeps 7 significant digits,
  // so hold the JS port much tighter than the cross-implementation tolerance.
  assert.ok(err.logits <= Math.min(fx.tolerance, 1e-6), `logits max err ${err.logits}`);
  assert.ok(err.rates <= Math.min(fx.tolerance, 1e-5), `rates_sampled max err ${err.rates}`);
  assert.ok(err.ratesFull <= Math.min(fx.tolerance, 1e-5), `rates_full max err ${err.ratesFull}`);
});

test("probs are the softmax of logits", () => {
  const m = createModel(bundle);
  m.step(new Float64Array(bundle.NI));
  const mx = Math.max(...m.logits);
  const z = [...m.logits].reduce((s, v) => s + Math.exp(v - mx), 0);
  let sum = 0;
  for (let c = 0; c < bundle.C; c++) {
    assert.ok(Math.abs(m.probs[c] - Math.exp(m.logits[c] - mx) / z) < 1e-12);
    sum += m.probs[c];
  }
  assert.ok(Math.abs(sum - 1) < 1e-12);
});

test("reset() restores r0 (and the same replay reproduces exactly)", () => {
  const m = createModel(bundle);
  const rRef = m.r;
  const first = replay(m);
  m.reset();
  assert.equal(m.r, rRef, "reset keeps the same buffer (scene holds a reference)");
  for (let i = 0; i < bundle.N; i++) if (m.r[i] !== bundle.r0[i]) assert.fail(`r[${i}] != r0 after reset`);
  assert.deepEqual(replay(m), first);
});

test("setLesion(true) gives exactly the no-input dynamics; setLesion(false) restores input", () => {
  const lesioned = createModel(bundle);
  const blind = createModel(bundle);
  lesioned.setLesion(true);
  const drive = new Float64Array(bundle.NI);
  const zero = new Float64Array(bundle.NI);
  for (let f = 0; f < 60; f++) {
    inputDrive(f / fps, threats, bundle.inputKind, bundle.inputSide, stim, drive);
    lesioned.step(drive);
    blind.step(zero);
  }
  assert.deepEqual([...lesioned.r], [...blind.r]);
  assert.deepEqual([...lesioned.logits], [...blind.logits]);

  lesioned.setLesion(false);
  const before = lesioned.r.slice();
  const noInput = createModel(bundle);
  noInput.r.set(before);
  inputDrive(0.5, threats, bundle.inputKind, bundle.inputSide, stim, drive);
  lesioned.step(drive);
  noInput.step(zero);
  assert.notDeepEqual([...lesioned.r], [...noInput.r]);
});

test("decaying rates never become subnormal floats (they stall the CPU in long sessions)", () => {
  const m = createModel(bundle);
  const drive = new Float64Array(bundle.NI).fill(0.3); // constant input silences some neurons for good
  for (let f = 0; f < 1500; f++) m.step(drive);
  let subnormal = 0;
  for (let i = 0; i < bundle.N; i++) if (m.r[i] !== 0 && Math.abs(m.r[i]) < 2.2250738585072014e-308) subnormal++;
  assert.equal(subnormal, 0);
});

test("benchmark: 600 frames average under 4 ms/frame", () => {
  const m = createModel(bundle);
  const drives = [];
  for (let f = 0; f < 180; f++) drives.push(inputDrive(f / fps, threats, bundle.inputKind, bundle.inputSide, stim));
  for (let f = 0; f < 60; f++) m.step(drives[f % 180]); // JIT warm-up
  m.reset();
  // Each run averages 600 frames of wall time; the best of 3 runs is the estimate, since other load on the machine
  // (parallel test files, other processes) only ever adds time.
  const runs = [];
  for (let run = 0; run < 3; run++) {
    m.reset();
    const t0 = performance.now();
    for (let f = 0; f < 600; f++) m.step(drives[f % 180]);
    runs.push((performance.now() - t0) / 600);
  }
  const ms = Math.min(...runs);
  console.log(`  model.step: ${ms.toFixed(3)} ms/frame (runs ${runs.map((v) => v.toFixed(2)).join(", ")}; N=${bundle.N}, E=${bundle.E}, substeps=${bundle.manifest.sim.substeps})`);
  assert.ok(ms < 4, `${ms} ms/frame`);
});
