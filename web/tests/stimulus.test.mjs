// Mirrors tests/test_stimulus.py, plus the fixture's drive_first8 rows.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { inputDrive, loomAngleDeg, loomRateDps, rfGain, threatClass } from "../stimulus.js";
import { BUNDLE_DIR, loadFixture } from "./helpers.mjs";

const manifest = JSON.parse(await readFile(BUNDLE_DIR + "manifest.json", "utf8"));
const stim = manifest.stimulus;
const fps = manifest.sim.fps;

function approx(actual, expected, { rel = 1e-9, abs = 1e-12 } = {}, msg = "") {
  const tol = Math.max(abs, rel * Math.abs(expected));
  assert.ok(Math.abs(actual - expected) <= tol, `${msg} ${actual} != ${expected} (tol ${tol})`);
}

test("loom angle is 90° at one lv before collision and increasing", () => {
  const lv = 0.04, tc = 1.0;
  approx(loomAngleDeg(tc - lv, lv, tc), 90.0);
  let prev = -Infinity;
  for (let i = 0; i < 200; i++) {
    const t = (i * (tc - 1e-3)) / 199;
    const a = loomAngleDeg(t, lv, tc);
    assert.ok(a > prev);
    prev = a;
  }
});

test("loom rate matches the numerical derivative", () => {
  const lv = 0.04, tc = 1.0, t = 0.8, h = 1e-6;
  const numeric = (loomAngleDeg(t + h, lv, tc) - loomAngleDeg(t - h, lv, tc)) / (2 * h);
  approx(loomRateDps(t, lv, tc), numeric, { rel: 1e-5 });
});

test("receptive fields are side tuned and azimuth wraps", () => {
  approx(rfGain(-90, 1, stim), Math.exp(-0.5 * (45 / 50) ** 2));
  approx(rfGain(-90, 2, stim), Math.exp(-0.5 * (135 / 50) ** 2));
  approx(rfGain(0, 1, stim), rfGain(0, 2, stim));
  approx(rfGain(180, 1, stim), rfGain(-180, 1, stim));
  approx(rfGain(-90 + 720, 1, stim), rfGain(-90, 1, stim));
  approx(rfGain(270, 1, stim), rfGain(-90, 1, stim)); // 270° is the left side (-90°)
});

test("input drive: kinds, sides and activity window", () => {
  const kind = [0, 0, 1]; // LPLC2 L, LPLC2 R, LC4 L
  const side = [1, 2, 1];
  const th = [{ azimuthDeg: -90, lv: 0.04, tSpawn: 0.1, tCollision: 0.7 }];
  assert.deepEqual([...inputDrive(0.05, th, kind, side, stim)], [0, 0, 0]); // before spawn
  assert.deepEqual([...inputDrive(0.7, th, kind, side, stim)], [0, 0, 0]); // at collision: gone
  const d = inputDrive(0.65, th, kind, side, stim);
  assert.ok(d instanceof Float64Array);
  const theta = loomAngleDeg(0.65, 0.04, 0.7);
  approx(d[0], Math.min(theta / 90, 1) * rfGain(-90, 1, stim));
  assert.ok(d[1] < 0.05 * d[0]);
  approx(d[2], Math.min(loomRateDps(0.65, 0.04, 0.7) / 1000, 1) * rfGain(-90, 1, stim));
});

test("input drive writes into `out`, clearing old values, and sums overlapping threats", () => {
  const kind = [0, 1], side = [1, 2];
  const out = new Float64Array([9, 9]);
  const a = { azimuthDeg: -90, lv: 0.04, tSpawn: 0, tCollision: 1 };
  const b = { azimuthDeg: 90, lv: 0.02, tSpawn: 0, tCollision: 1 };
  assert.equal(inputDrive(2, [a], kind, side, stim, out), out);
  assert.deepEqual([...out], [0, 0]);
  const da = [...inputDrive(0.9, [a], kind, side, stim)];
  const db = [...inputDrive(0.9, [b], kind, side, stim)];
  inputDrive(0.9, [a, b], kind, side, stim, out);
  approx(out[0], da[0] + db[0]);
  approx(out[1], da[1] + db[1]);
  assert.deepEqual([...inputDrive(0.5, [], kind, side, stim)], [0, 0]);
});

for (const [az, cls] of [[-90, 1], [-60, 1], [90, 2], [60, 2], [0, 3], [30, 3], [-44, 3]]) {
  test(`threat class ${az}° → ${cls}`, () => assert.equal(threatClass(az, stim), cls));
}

test("threat class front boundary is inclusive at ±front_half_width_deg", () => {
  assert.equal(threatClass(45, stim), 3);
  assert.equal(threatClass(-45, stim), 3);
  assert.equal(threatClass(45.001, stim), 2);
  assert.equal(threatClass(-45.001, stim), 1);
});

test("rf gain wraps for the right eye too", () => {
  assert.equal(rfGain(-170, 2, stim), rfGain(190, 2, stim));
  approx(rfGain(-170, 2, stim), Math.exp(-0.5 * (145 / 50) ** 2));
});

test("threat class wraps azimuth", () => {
  assert.equal(threatClass(270, stim), 1);
  assert.equal(threatClass(-270, stim), 2);
  assert.equal(threatClass(360, stim), 3);
});

test("drive_first8 matches the parity fixture on every frame", async () => {
  const fx = await loadFixture();
  const io = manifest.files["io.bin"].layout;
  const buf = await readFile(BUNDLE_DIR + "io.bin");
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const ni = io[0][2];
  const kind = new Uint32Array(ab, 4 * ni, ni);
  const side = new Uint32Array(ab, 8 * ni, ni);
  const threats = fx.threats.map((t) => ({ azimuthDeg: t.azimuth_deg, lv: t.lv, tSpawn: t.t_spawn, tCollision: t.t_collision }));
  const out = new Float64Array(ni);
  let maxErr = 0, nonzero = 0;
  for (let f = 0; f < fx.frames; f++) {
    inputDrive(f / fps, threats, kind, side, stim, out);
    fx.drive_first8[f].forEach((v, j) => {
      maxErr = Math.max(maxErr, Math.abs(out[j] - v));
      if (v) nonzero++;
    });
  }
  assert.ok(nonzero > 100, "fixture drive should be mostly active");
  assert.ok(maxErr <= 1e-6, `max drive error ${maxErr}`);
});
