import { test } from "node:test";
import assert from "node:assert/strict";
import { createDecider } from "../decision.js";

const classes = ["none", "dodge_right", "dodge_left", "takeoff"];
const make = () => createDecider({ classes, p: 0.7, frames: 3 });
const P = {
  none: [0.9, 0.05, 0.03, 0.02],
  right: [0.1, 0.8, 0.05, 0.05],
  left: [0.1, 0.05, 0.8, 0.05],
  takeoff: [0.05, 0.05, 0.1, 0.8],
  unsure: [0.3, 0.3, 0.2, 0.2],
  atP: [0.1, 0.7, 0.1, 0.1],
};

test("fires only after `frames` consecutive frames above p for the same class", () => {
  const d = make();
  assert.equal(d.update(P.right), null);
  assert.equal(d.update(P.right), null);
  assert.deepEqual(d.update(P.right), { classIndex: 1, name: "dodge_right" });
});

test("a dip below p restarts the count; prob exactly p does not count", () => {
  const d = make();
  d.update(P.takeoff);
  d.update(P.takeoff);
  assert.equal(d.update(P.unsure), null);
  assert.equal(d.update(P.atP), null);
  assert.equal(d.update(P.takeoff), null);
  assert.equal(d.update(P.takeoff), null);
  assert.deepEqual(d.update(P.takeoff), { classIndex: 3, name: "takeoff" });
});

test("probability exactly p never fires (strictly greater than p)", () => {
  const d = make();
  for (let i = 0; i < 10; i++) assert.equal(d.update(P.atP), null);
});

test("alternating classes never fire", () => {
  const d = make();
  for (let i = 0; i < 30; i++) assert.equal(d.update(i % 2 ? P.left : P.right), null);
});

test("switching class restarts the count for the new class", () => {
  const d = make();
  d.update(P.right);
  d.update(P.right);
  assert.equal(d.update(P.left), null);
  assert.equal(d.update(P.left), null);
  assert.deepEqual(d.update(P.left), { classIndex: 2, name: "dodge_left" });
});

test("fires once until reset()", () => {
  const d = make();
  for (let i = 0; i < 3; i++) d.update(P.left);
  for (let i = 0; i < 20; i++) assert.equal(d.update(i % 5 === 0 ? P.right : P.left), null);
  for (let i = 0; i < 5; i++) assert.equal(d.update(P.takeoff), null);
  d.reset();
  assert.equal(d.update(P.takeoff), null);
  assert.equal(d.update(P.takeoff), null);
  assert.deepEqual(d.update(P.takeoff), { classIndex: 3, name: "takeoff" });
});

test("never fires for none, and none interrupts a streak", () => {
  const d = make();
  for (let i = 0; i < 50; i++) assert.equal(d.update(P.none), null);
  d.update(P.right);
  d.update(P.right);
  assert.equal(d.update(P.none), null);
  assert.equal(d.update(P.right), null);
});

test("accepts typed arrays and honours frames/p from the manifest", () => {
  const d = createDecider({ classes, p: 0.5, frames: 1 });
  assert.deepEqual(d.update(new Float64Array([0.2, 0.2, 0.6, 0.0])), { classIndex: 2, name: "dodge_left" });
});
