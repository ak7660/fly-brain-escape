// Layered ("wiring diagram") positions for the circuit. Pure math, no DOM: runs in Node for tests.
//
// x is set by layer, evenly spaced over the anatomy's half-extent S: input (layer 0) at +X, motor (layer 4) at −X.
// From the default front camera (fly's left, −X, on screen right) that reads input → motor from left to right,
// so the anatomy ↔ layers morph needs no camera swing.
// Within a layer the neurons form a disc in the y–z plane: the fly's left side fills the upper half
// (y ≥ 0), the right side the lower half, and unknown-side neurons are dealt alternately into both.
// Radius grows with type rank (the most common type sits at the centre), so types read as rings.
// Disc area is proportional to the layer's neuron count, so point density is the same in every layer.

const N_LAYERS = 5;
const GOLDEN = Math.PI * (3 - Math.sqrt(5)); // golden angle (rad)

/** Largest |coordinate| of the anatomy positions: the common bounding scale of both layouts. */
export function anatomyExtent(pos) {
  let m = 0;
  for (let i = 0; i < pos.length; i++) {
    const a = Math.abs(pos[i]);
    if (a > m) m = a;
  }
  return m || 1;
}

/** @returns {Float32Array} 3N positions in µm. */
export const LAYER_COUNT = N_LAYERS;

export function layeredPositions(bundle) {
  const { N, pos, layer, side, typeId } = bundle;
  const out = new Float32Array(3 * N);
  const S = anatomyExtent(pos);
  const xSpan = 0.95 * S;                    // layer 0 at +xSpan, layer 4 at −xSpan
  const rMax = 0.5 * S;                      // radius of the most populous layer

  // group neurons by layer
  const byLayer = Array.from({ length: N_LAYERS }, () => []);
  for (let i = 0; i < N; i++) byLayer[Math.min(layer[i], N_LAYERS - 1)].push(i);
  let nMaxLayer = 1;
  for (const g of byLayer) nMaxLayer = Math.max(nMaxLayer, g.length);

  for (let L = 0; L < N_LAYERS; L++) {
    const idx = byLayer[L];
    if (!idx.length) continue;
    // type rank inside this layer: most common type first
    const count = new Map();
    for (const i of idx) count.set(typeId[i], (count.get(typeId[i]) || 0) + 1);
    const rank = new Map(
      [...count.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]).map(([t], r) => [t, r]),
    );
    idx.sort((a, b) => rank.get(typeId[a]) - rank.get(typeId[b]) || a - b);

    // split into halves: left (side 1) upper, right (side 2) lower, unknown alternates
    const halves = [[], []];
    let alt = 0;
    for (const i of idx) {
      if (side[i] === 1) halves[0].push(i);
      else if (side[i] === 2) halves[1].push(i);
      else halves[alt++ & 1].push(i);
    }

    const R = rMax * Math.max(0.3, Math.sqrt(idx.length / nMaxLayer));
    const x0 = xSpan - (2 * xSpan * L) / (N_LAYERS - 1);
    for (let h = 0; h < 2; h++) {
      const g = halves[h];
      const n = g.length;
      for (let j = 0; j < n; j++) {
        const i = g[j];
        const r = R * Math.sqrt((j + 0.5) / n);
        // golden-angle spiral folded into a half disc; keep a hairline gap at the midline
        const u = ((j * GOLDEN) / Math.PI) % 1;
        const theta = Math.PI * (0.03 + 0.94 * u) + h * Math.PI;
        // a little depth along x so each disc reads as a slab, deterministic per neuron
        const jitter = (((i * 2654435761) >>> 0) / 4294967296 - 0.5) * 0.035 * S;
        out[3 * i] = x0 + jitter;
        out[3 * i + 1] = r * Math.sin(theta);
        out[3 * i + 2] = r * Math.cos(theta);
      }
    }
  }
  return out;
}
