// Pure game logic (no DOM): arena geometry, threat bookkeeping, decision attribution and scoring.
// Looming math and classes come from stimulus.js (the port of flybrain/stimulus.py).
import { loomAngleDeg, threatClass, wrapDeg } from "./stimulus.js";

export const CLASS_NONE = 0;

/** Azimuth of a point relative to the arena center; screen up is front (0°), right is +90°. */
export function azimuthFromPoint(dx, dy) {
  return wrapDeg((Math.atan2(dx, -dy) * 180) / Math.PI);
}

/**
 * Top-down placement of a threat: it travels from `outerR` to the center over its approach,
 * and its drawn radius is proportional to the looming angle θ(t). Pass `out` to avoid allocating.
 */
export function threatGeometry(th, t, { outerR = 130, pxPerDeg = 0.34, minR = 2.5, maxR = 56 } = {}, out = {}) {
  const dur = Math.max(th.tCollision - th.tSpawn, 1e-6);
  const f = Math.min(Math.max((th.tCollision - t) / dur, 0), 1); // 1 at spawn → 0 at contact
  const theta = loomAngleDeg(t, th.lv, th.tCollision);
  const a = (th.azimuthDeg * Math.PI) / 180;
  const dist = outerR * f;
  out.theta = theta;
  out.x = Math.sin(a) * dist;
  out.y = -Math.cos(a) * dist;
  out.r = Math.min(maxR, Math.max(minR, theta * pxPerDeg));
  out.progress = 1 - f;
  return out;
}

/**
 * The threat a decision is attributed to: the active, unresolved threat with the largest looming angle θ among those
 * at least `minAgeS` old. This matches the offline evaluation (flybrain/train.decide accepts any decision before
 * contact, even below label_onset_deg — the trained network often acts early). The minimum age only guards against a
 * decision that is really the network still reacting to the previous threat.
 */
export function scoringThreat(threats, t, minAgeS = 0) {
  let best = null;
  let bestTheta = -1;
  for (const th of threats) {
    if (th.outcome || !(th.tSpawn <= t && t < th.tCollision) || t - th.tSpawn < minAgeS) continue;
    const theta = loomAngleDeg(t, th.lv, th.tCollision);
    if (theta > bestTheta) {
      best = th;
      bestTheta = theta;
    }
  }
  return best;
}

export function emptyScore() {
  return { correct: 0, wrong: 0, missed: 0, falseAlarms: 0, total: 0 };
}

/** Hit rate over threats; false alarms count against it. */
export function accuracy(score) {
  const n = score.total + score.falseAlarms;
  return n ? score.correct / n : null;
}

/**
 * Game state machine. Time is sim seconds, supplied by the caller.
 * Events: {kind: "correct"|"wrong"|"false_alarm"|"miss", threat|null, classIndex, expected, latencyMs, t}
 */
export function createGameCore({ stim, fps = 60, lingerS = 0.45, minDecisionAgeS = 0.1 }) {
  const approachS = stim?.approach_s ?? 0.6; // seconds from spawn to contact (manifest: flybrain/config.APPROACH_S)
  let list = [];
  let now = 0;
  let nextId = 1;
  let lastMissT = -Infinity;
  let sc = emptyScore();
  const spawnCbs = [];
  const views = []; // reusable {azimuthDeg, lv, tSpawn, tCollision} objects for threats()

  const snap = (t) => Math.round(t * fps) / fps;

  function spawn(azimuthDeg, lv = 0.04, approach = approachS) {
    const tSpawn = snap(now);
    const th = {
      id: nextId++,
      azimuthDeg: wrapDeg(azimuthDeg),
      lv,
      tSpawn,
      tCollision: snap(tSpawn + approach),
      cls: threatClass(azimuthDeg, stim),
      outcome: null,
    };
    list.push(th);
    for (const cb of spawnCbs) cb(th);
    return th;
  }

  /** Active threats as {azimuthDeg, lv, tSpawn, tCollision}. The array and its objects are reused between calls. */
  function threats() {
    let n = 0;
    for (const th of list) {
      if (!(th.tSpawn <= now && now < th.tCollision)) continue;
      const v = views[n] ?? (views[n] = { azimuthDeg: 0, lv: 0, tSpawn: 0, tCollision: 0 });
      v.azimuthDeg = th.azimuthDeg;
      v.lv = th.lv;
      v.tSpawn = th.tSpawn;
      v.tCollision = th.tCollision;
      n++;
    }
    viewsOut.length = n;
    for (let i = 0; i < n; i++) viewsOut[i] = views[i];
    return viewsOut;
  }
  const viewsOut = [];

  const NO_EVENTS = Object.freeze([]);
  /** Advance to tSim; returns miss events for threats that reached contact unresolved. */
  function tick(tSim) {
    now = tSim;
    let events = NO_EVENTS;
    let drop = false;
    for (const th of list) {
      if (!th.outcome && now >= th.tCollision) {
        const ev = { kind: "miss", threat: th, classIndex: CLASS_NONE, expected: th.cls, latencyMs: 0, t: th.tCollision };
        th.outcome = ev;
        lastMissT = Math.max(lastMissT, th.tCollision);
        sc.missed++;
        sc.total++;
        if (events === NO_EVENTS) events = [];
        events.push(ev);
      }
      if (now >= th.tCollision + lingerS) drop = true;
    }
    if (drop) list = list.filter((th) => now < th.tCollision + lingerS);
    return events;
  }

  /**
   * Score a decision. Returns the event, or null when ignored (class none, or a late decision
   * arriving within the linger window after a miss).
   */
  function resolve({ classIndex }) {
    if (!classIndex) return null;
    const th = scoringThreat(list, now, minDecisionAgeS);
    if (!th) {
      if (now - lastMissT < lingerS) return null;
      sc.falseAlarms++;
      return { kind: "false_alarm", threat: null, classIndex, expected: CLASS_NONE, latencyMs: null, t: now };
    }
    const ok = classIndex === th.cls;
    const ev = {
      kind: ok ? "correct" : "wrong",
      threat: th,
      classIndex,
      expected: th.cls,
      latencyMs: Math.round((th.tCollision - now) * 1000),
      t: now,
    };
    th.outcome = ev;
    ok ? sc.correct++ : sc.wrong++;
    sc.total++;
    return ev;
  }

  function reset() {
    list = [];
    lastMissT = -Infinity;
    sc = emptyScore();
  }

  return {
    spawn,
    threats,
    tick,
    resolve,
    reset,
    onSpawn(cb) { spawnCbs.push(cb); },
    all: () => list,
    now: () => now,
    minDecisionAgeS,
    score: () => ({ ...sc, accuracy: accuracy(sc) }),
  };
}
