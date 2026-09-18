import test from "node:test";
import assert from "node:assert/strict";
import { wrapDeg, threatClass, loomAngleDeg } from "../stimulus.js";
import {
  azimuthFromPoint, threatGeometry, scoringThreat, createGameCore, accuracy,
} from "../game-logic.js";

const stim = { rf_centers_deg: [-45, 45], rf_sigma_deg: 50, front_half_width_deg: 45, label_onset_deg: 15 };
const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);

// t at which a threat with (lv, tCollision) reaches θ = deg
const tAtTheta = (deg, lv, tc) => tc - lv / Math.tan((deg * Math.PI) / 360);

test("wrapDeg (from stimulus.js) maps into [-180, 180)", () => {
  near(wrapDeg(190), -170);
  near(wrapDeg(-190), 170);
  near(wrapDeg(180), -180);
  near(wrapDeg(-180), -180);
  near(wrapDeg(315), -45);
  near(wrapDeg(-315), 45);
});

test("azimuthFromPoint: up front, right +90, left -90", () => {
  near(azimuthFromPoint(0, -10), 0);
  near(azimuthFromPoint(10, 0), 90);
  near(azimuthFromPoint(-10, 0), -90);
  near(Math.abs(azimuthFromPoint(0, 10)), 180);
});

test("threatGeometry moves inward and grows with θ", () => {
  const th = { azimuthDeg: 90, lv: 0.04, tSpawn: 0, tCollision: 0.6 };
  const a = threatGeometry(th, 0, { outerR: 100 });
  const b = threatGeometry(th, 0.5, { outerR: 100 });
  near(a.x, 100);
  near(a.y, 0);
  assert.ok(b.x < a.x && b.r > a.r && b.theta > a.theta);
  near(threatGeometry(th, 0.6).progress, 1);
  near(threatGeometry({ ...th, azimuthDeg: 0 }, 0, { outerR: 100 }).y, -100);
  const out = {};
  assert.equal(threatGeometry(th, 0.3, {}, out), out, "writes into a reusable object");
});

test("scoringThreat: largest θ among active, unresolved threats old enough to be decided on", () => {
  const fast = { id: 1, azimuthDeg: -90, lv: 0.02, tSpawn: 0, tCollision: 1.0 };
  const slow = { id: 2, azimuthDeg: 90, lv: 0.06, tSpawn: 0.5, tCollision: 1.2 };
  // at t = 0.9: fast θ = 2·atan(0.02/0.1) ≈ 22.6°, slow θ = 2·atan(0.06/0.3) ≈ 22.6° + tiny → compare explicitly
  const t = 0.95; // fast θ ≈ 43.6°, slow θ ≈ 27°: largest θ wins over shortest time to contact
  assert.equal(scoringThreat([fast, slow], t).id, 1);
  assert.equal(scoringThreat([fast, slow], 0.55, 0.1).id, 1, "slow is only 0.05 s old: not yet eligible");
  assert.equal(scoringThreat([slow], 0.55, 0.1), null);
  fast.outcome = "correct";
  assert.equal(scoringThreat([fast, slow], t).id, 2, "resolved threats are skipped");
  assert.equal(scoringThreat([fast, slow], 2), null, "nothing active after collision");
});

test("core: correct and wrong decisions", () => {
  const g = createGameCore({ stim, fps: 60 });
  g.tick(0);
  g.spawn(-90); // dodge_right expected
  g.tick(0.4); // θ ≈ 22.6°
  assert.equal(g.threats().length, 1);
  assert.deepEqual(Object.keys(g.threats()[0]).sort(), ["azimuthDeg", "lv", "tCollision", "tSpawn"]);
  const ev = g.resolve({ classIndex: 1 });
  assert.equal(ev.kind, "correct");
  assert.equal(ev.latencyMs, 200);
  g.spawn(90, 0.04, 0.6);
  g.tick(0.9);
  assert.equal(g.resolve({ classIndex: 3 }).kind, "wrong");
  const s = g.score();
  assert.deepEqual([s.correct, s.wrong, s.total], [1, 1, 2]);
});

test("core: a correct decision below the label-onset angle still scores (the network may act early)", () => {
  const g = createGameCore({ stim, fps: 60 });
  g.tick(0);
  g.spawn(0, 0.04, 0.6);          // front threat
  g.tick(0.27);                    // θ = 2·atan(0.04/0.33) ≈ 13.8° < label_onset_deg 15
  const ev = g.resolve({ classIndex: 3 });
  assert.equal(ev.kind, "correct");
  assert.equal(g.score().falseAlarms, 0);
});

test("core: overlapping threats — a stale decision right after a spawn is not scored against the fresh threat", () => {
  const g = createGameCore({ stim, fps: 60 });
  g.tick(0);
  g.spawn(-90, 0.04, 0.6);        // A
  g.tick(0.45);
  assert.equal(g.resolve({ classIndex: 1 }).threat.azimuthDeg, -90);
  g.tick(0.5);
  g.spawn(90, 0.04, 0.6);         // B spawns while the network may still be reacting to A
  g.tick(0.55);                    // B is 0.05 s old
  const stale = g.resolve({ classIndex: 1 });
  assert.equal(stale.kind, "false_alarm", "decision within the minimum decision age is not attributed to B");
  g.tick(0.8);                     // B 0.3 s old
  const ev = g.resolve({ classIndex: 2 });
  assert.equal(ev.kind, "correct");
  assert.equal(ev.threat.azimuthDeg, 90);
});

test("core: a decision during the linger window after a miss is ignored (late), not a false alarm", () => {
  const g = createGameCore({ stim, fps: 60, lingerS: 0.5 });
  g.tick(0);
  g.spawn(0);
  const miss = g.tick(0.6);
  assert.equal(miss.length, 1);
  assert.equal(miss[0].kind, "miss");
  g.tick(0.8);
  assert.equal(g.resolve({ classIndex: 3 }), null);
  let s = g.score();
  assert.deepEqual([s.missed, s.falseAlarms, s.total], [1, 0, 1]);
  g.tick(1.3); // past linger
  assert.equal(g.resolve({ classIndex: 3 }).kind, "false_alarm");
  s = g.score();
  assert.equal(s.falseAlarms, 1);
});

test("core: unresolved threat at contact is a miss; lingering threats drop", () => {
  const g = createGameCore({ stim, fps: 60, lingerS: 0.3 });
  g.tick(0);
  g.spawn(0);
  assert.deepEqual(g.tick(0.59), []);
  const ev = g.tick(0.6);
  assert.equal(ev[0].expected, 3);
  assert.equal(g.threats().length, 0);
  assert.equal(g.all().length, 1, "still lingering for the outcome animation");
  g.tick(0.95);
  assert.equal(g.all().length, 0);
});

test("core: decisions with no threat are false alarms; none is ignored", () => {
  const g = createGameCore({ stim });
  g.tick(1);
  assert.equal(g.resolve({ classIndex: 0 }), null);
  assert.equal(g.resolve({ classIndex: 3 }).kind, "false_alarm");
  assert.equal(g.score().falseAlarms, 1);
  assert.equal(accuracy(g.score()), 0);
});

test("core: spawn wraps azimuth (±315) and classifies with stimulus.threatClass", () => {
  const g = createGameCore({ stim, fps: 60 });
  g.tick(0);
  const a = g.spawn(315);
  const b = g.spawn(-315);
  near(a.azimuthDeg, -45);
  near(b.azimuthDeg, 45);
  assert.equal(a.cls, 3);
  assert.equal(b.cls, 3);
  assert.equal(g.spawn(-226).cls, 2); // wraps to +134°
  assert.equal(threatClass(-316, stim), 3); // wraps to +44°
});

test("core: onSpawn fires for every spawn", () => {
  const g = createGameCore({ stim, fps: 60 });
  const seen = [];
  g.onSpawn((th) => seen.push(th.azimuthDeg));
  g.spawn(-90);
  g.spawn(10);
  assert.deepEqual(seen, [-90, 10]);
});

test("core: threats() reuses its view objects (no per-frame allocation)", () => {
  const g = createGameCore({ stim, fps: 60 });
  g.tick(0);
  g.spawn(-90);
  g.tick(0.1);
  const a = g.threats();
  g.tick(0.2);
  const b = g.threats();
  assert.equal(a, b);
  assert.equal(a[0], b[0]);
  assert.equal(b.length, 1);
});

test("core: spawn snaps to whole frames", () => {
  const g = createGameCore({ stim, fps: 60 });
  g.tick(0.1234);
  const th = g.spawn(10, 0.02, 0.6);
  near(th.tSpawn * 60, Math.round(th.tSpawn * 60));
  near(th.tCollision * 60, Math.round(th.tCollision * 60), 1e-6);
  near(tAtTheta(90, 0.04, 0.6), 0.56);
});
