import { test } from "node:test";
import assert from "node:assert/strict";
import { loadBundle } from "../data.js";
import { neuronTour, TOUR_TYPES_PER_LAYER } from "../tour.js";
import { BUNDLE_DIR, fsFetch } from "./helpers.mjs";

const bundle = await loadBundle(BUNDLE_DIR, fsFetch());

test("the tour is a short, in-range, duplicate-free list", () => {
  const tour = neuronTour(bundle);
  assert.ok(tour.length > 10 && tour.length <= 5 * TOUR_TYPES_PER_LAYER + 5, `length ${tour.length}`);
  assert.equal(new Set(tour).size, tour.length, "no neuron appears twice");
  for (const i of tour) assert.ok(Number.isInteger(i) && i >= 0 && i < bundle.N, `index ${i}`);
});

test("it runs input → motor and never repeats a type", () => {
  const tour = neuronTour(bundle);
  const layers = tour.map((i) => bundle.layer[i]);
  for (let k = 1; k < layers.length; k++) assert.ok(layers[k] >= layers[k - 1], "layers are non-decreasing");
  assert.deepEqual([...new Set(layers)], [0, 1, 2, 3, 4]);
  const seen = new Set();
  for (const i of tour) {
    assert.ok(!seen.has(bundle.typeId[i]), `type ${bundle.info.types[bundle.typeId[i]]} appears twice`);
    seen.add(bundle.typeId[i]);
  }
});

test("each layer contributes at most perLayer types, most common first", () => {
  const tour = neuronTour(bundle, 3);
  const perLayer = new Map();
  for (const i of tour) perLayer.set(bundle.layer[i], (perLayer.get(bundle.layer[i]) ?? 0) + 1);
  // at most perLayer, plus the Giant Fiber when it is not already in the top of its layer
  for (const [L, n] of perLayer) assert.ok(n <= 4, `layer ${L} has ${n}`);
  // the first neuron of layer 1 is of the most common type in layer 1
  const firstHidden = tour.find((i) => bundle.layer[i] === 1);
  let best = -1, bestN = 0;
  const l1 = new Map();
  for (let i = 0; i < bundle.N; i++) if (bundle.layer[i] === 1) l1.set(bundle.typeId[i], (l1.get(bundle.typeId[i]) ?? 0) + 1);
  for (const [t, n] of l1) if (n > bestN) { best = t; bestN = n; }
  assert.equal(bundle.typeId[firstHidden], best);
});

test("the Giant Fiber leads the descending layer when the bundle has one", () => {
  const gf = bundle.info.types.indexOf("DNp01");
  if (gf < 0) return;
  const tour = neuronTour(bundle);
  const dn = tour.filter((i) => bundle.layer[i] === 3);
  assert.equal(bundle.info.types[bundle.typeId[dn[0]]], "DNp01");
});

test("a bundle with no type names still works", () => {
  const tiny = { N: 4, layer: [0, 0, 1, 1], typeId: [7, 7, 2, 3] };
  assert.deepEqual(neuronTour(tiny, 5), [0, 2, 3]);
});
