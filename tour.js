// The keyboard tour of the circuit: a short, ordered list of neurons the 3D view can step through with [ and ]
// so neuron identity is reachable without a pointer. Pure: no DOM, runs in node for tests.
//
// Stepping through all 4,296 neurons one by one would be useless, so the tour takes one representative neuron
// per type — the most common types first, layer by layer from input to motor. The Giant Fiber (DNp01) is the
// one named cell in this circuit and always leads its layer.

export const TOUR_TYPES_PER_LAYER = 10;

/**
 * @param {{N: number, layer: ArrayLike<number>, typeId: ArrayLike<number>, info?: {types?: string[]}}} bundle
 * @param {number} perLayer  how many types to take from each layer
 * @returns {number[]} neuron indices, ordered input → motor
 */
export function neuronTour(bundle, perLayer = TOUR_TYPES_PER_LAYER) {
  const { N, layer, typeId } = bundle;
  const types = bundle.info?.types ?? [];
  const byLayer = new Map(); // layer → Map(typeId → {count, first})
  for (let i = 0; i < N; i++) {
    const L = layer[i];
    let m = byLayer.get(L);
    if (!m) byLayer.set(L, (m = new Map()));
    const t = typeId[i], e = m.get(t);
    if (e) e.count++;
    else m.set(t, { count: 1, first: i });
  }
  const out = [];
  for (const L of [...byLayer.keys()].sort((a, b) => a - b)) {
    const entries = [...byLayer.get(L)].sort((a, b) => b[1].count - a[1].count || a[1].first - b[1].first);
    const picked = entries.slice(0, Math.max(1, perLayer));
    const gf = entries.find(([t]) => types[t] === "DNp01");
    if (gf) {
      const at = picked.indexOf(gf);
      if (at >= 0) picked.splice(at, 1);
      picked.unshift(gf);
    }
    for (const [, e] of picked) out.push(e.first);
  }
  return out;
}
