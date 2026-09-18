// Looming threats → drive for LPLC2 (angular size) and LC4 (angular velocity) cells.
// Mirrors flybrain/stimulus.py function by function; stimulus constants come from manifest.stimulus (`stim`).
// threat = {azimuthDeg, lv /* half-size / approach speed, s */, tSpawn /* s */, tCollision /* s */}

const RAD2DEG = 180.0 / Math.PI;

/** Python-style float modulo (result has the sign of the divisor). */
function pymod(x, m) {
  const r = x % m;
  return r !== 0 && r < 0 !== m < 0 ? r + m : r;
}

/** Wrap an azimuth into [-180, 180). */
export function wrapDeg(az) {
  return pymod(az + 180.0, 360.0) - 180.0;
}

export function loomAngleDeg(t, lv, tCollision) {
  return 2.0 * Math.atan(lv / Math.max(tCollision - t, 1e-9)) * RAD2DEG;
}

export function loomRateDps(t, lv, tCollision) {
  const dt = Math.max(tCollision - t, 1e-9);
  return ((2.0 * lv) / (dt * dt + lv * lv)) * RAD2DEG;
}

export function rfGain(azimuthDeg, side, stim) {
  const center = stim.rf_centers_deg[side - 1];
  const d = pymod(azimuthDeg - center + 180.0, 360.0) - 180.0;
  const z = d / stim.rf_sigma_deg;
  return Math.exp(-0.5 * z * z);
}

export function threatClass(azimuthDeg, stim) {
  const az = wrapDeg(azimuthDeg);
  if (Math.abs(az) <= stim.front_half_width_deg) return 3;
  return az < 0 ? 1 : 2;
}

// per-kind drive of one threat at time t: [size (LPLC2), speed (LC4)], or null outside [tSpawn, tCollision)
const kindScratch = new Float64Array(2);
function kindDrive(t, th, stim) {
  if (!(th.tSpawn <= t && t < th.tCollision)) return null;
  kindScratch[0] = Math.min(loomAngleDeg(t, th.lv, th.tCollision) / stim.theta_sat_deg, 1.0);
  kindScratch[1] = Math.min(loomRateDps(t, th.lv, th.tCollision) / stim.thetadot_sat_dps, 1.0);
  return kindScratch;
}

const gainBySide = new Float64Array(2); // rf gain for side 1 (left) and side 2 (right) of the current threat

/** Drive for every input neuron (Float64Array(NI)); pass `out` to reuse a buffer (it is zeroed first). */
export function inputDrive(t, threats, inputKind, inputSide, stim, out) {
  const ni = inputKind.length;
  const drive = out ?? new Float64Array(ni);
  drive.fill(0);
  for (const th of threats) {
    const perKind = kindDrive(t, th, stim);
    if (perKind === null || (perKind[0] === 0 && perKind[1] === 0)) continue;
    gainBySide[0] = rfGain(th.azimuthDeg, 1, stim);
    gainBySide[1] = rfGain(th.azimuthDeg, 2, stim);
    for (let j = 0; j < ni; j++) drive[j] += perKind[inputKind[j]] * gainBySide[inputSide[j] - 1];
  }
  return drive;
}
