// scene track: layeredPositions is pure math, so it is tested in Node against the real bundle.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { layeredPositions, anatomyExtent } from "../layout.js";

const dir = new URL("../assets/circuit_v1/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("manifest.json", dir)));

function load() {
  const N = manifest.counts.neurons;
  const nb = readFileSync(new URL("neurons.bin", dir));
  const mb = readFileSync(new URL("neuron_meta.bin", dir));
  const copy = (buf) => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const neurons = copy(nb), meta = copy(mb);
  return {
    N,
    pos: new Float32Array(neurons, 0, 3 * N),
    layer: new Uint8Array(meta, 0, N),
    side: new Uint8Array(meta, 2 * N, N),
    typeId: new Uint16Array(meta, 4 * N, N),
  };
}

const bundle = load();
const P = layeredPositions(bundle);

test("returns 3N finite values", () => {
  assert.equal(P.length, 3 * bundle.N);
  for (let i = 0; i < P.length; i++) assert.ok(Number.isFinite(P[i]), `P[${i}] = ${P[i]}`);
});

test("extent matches the anatomy bounding scale", () => {
  const S = anatomyExtent(bundle.pos);
  const L = anatomyExtent(P);
  assert.ok(L <= 1.001 * S, `layered extent ${L} exceeds anatomy ${S}`);
  assert.ok(L >= 0.5 * S, `layered extent ${L} is too small vs anatomy ${S}`);
});

test("layers are ordered along x (input at +X, motor at −X) without overlap", () => {
  const lo = new Array(5).fill(Infinity), hi = new Array(5).fill(-Infinity);
  for (let i = 0; i < bundle.N; i++) {
    const L = bundle.layer[i], x = P[3 * i];
    lo[L] = Math.min(lo[L], x);
    hi[L] = Math.max(hi[L], x);
  }
  for (let L = 1; L < 5; L++) assert.ok(lo[L - 1] > hi[L], `layer ${L - 1} min ${lo[L - 1]} ≤ layer ${L} max ${hi[L]}`);
});

test("left side sits in the upper half, right side in the lower half", () => {
  for (let i = 0; i < bundle.N; i++) {
    const y = P[3 * i + 1];
    if (bundle.side[i] === 1) assert.ok(y >= 0, `left neuron ${i} at y=${y}`);
    if (bundle.side[i] === 2) assert.ok(y <= 0, `right neuron ${i} at y=${y}`);
  }
});

test("works on a tiny synthetic bundle", () => {
  const b = {
    N: 3,
    pos: new Float32Array([1, 2, 3, -4, 5, 6, 7, -8, 9]),
    layer: new Uint8Array([0, 2, 4]),
    side: new Uint8Array([0, 1, 2]),
    typeId: new Uint16Array([0, 0, 1]),
  };
  const p = layeredPositions(b);
  assert.ok(p.every(Number.isFinite));
  assert.ok(p[0] > p[3] && p[3] > p[6]);
});
