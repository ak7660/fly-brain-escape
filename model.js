// Rate network over the exported (folded) bundle — the browser twin of flybrain/sim_numpy.py:NumpySim.frame.
// Frame: u = inputGain[kind]·drive → `substeps` × synchronous r ← (1−α)r + α·clip(W r + b + u, 0, r_max)
// → logits = W_out · r[dn_idx] + b_out, probs = softmax(logits).
// Float64 state for parity with the float64 reference; CSR SpMV with plain loops and no per-step allocation.

const R_FLOOR = 1e-30;

export function createModel(bundle) {
  const { N, NI, ND, C, val, bias, alpha, r0, inputKind, inputGain, Wout, bOut } = bundle;
  // Int32 index copies: V8 keeps Uint32Array reads on a slower (possibly >2^31) path; ~2x faster SpMV
  const rowPtr = Int32Array.from(bundle.rowPtr);
  const col = Int32Array.from(bundle.col);
  const inputIdx = Int32Array.from(bundle.inputIdx);
  const dnIdx = Int32Array.from(bundle.dnIdx);
  const { substeps, r_max: rMax } = bundle.manifest.sim;

  // float64 copies of the per-neuron parameters, mirroring NumpySim's astype(float64)
  const b = Float64Array.from(bias);
  const a = Float64Array.from(alpha);
  const oneMinusA = new Float64Array(N);
  for (let i = 0; i < N; i++) oneMinusA[i] = 1 - a[i];
  const inScale = new Float64Array(NI);
  for (let j = 0; j < NI; j++) inScale[j] = inputGain[inputKind[j]];
  const u = new Float64Array(N); // dense input vector, rebuilt each step (zeros off the input neurons)
  const cur = new Float64Array(N); // W r + b + u from the old r
  const wOut = Float64Array.from(Wout);
  const bO = Float64Array.from(bOut);

  const r = new Float64Array(N);
  const logits = new Float64Array(C);
  const probs = new Float64Array(C);
  let lesion = false;

  function readout() {
    for (let c = 0; c < C; c++) {
      let s = 0;
      const row = c * ND;
      for (let d = 0; d < ND; d++) s += wOut[row + d] * r[dnIdx[d]];
      logits[c] = s + bO[c];
    }
    let mx = -Infinity;
    for (let c = 0; c < C; c++) if (logits[c] > mx) mx = logits[c];
    let z = 0;
    for (let c = 0; c < C; c++) z += Math.exp(logits[c] - mx);
    for (let c = 0; c < C; c++) probs[c] = Math.exp(logits[c] - mx) / z;
  }

  function step(drive) {
    if (!(drive.length >= NI)) throw new Error(`model.step: drive has length ${drive.length}, expected ${NI} (NI)`);
    u.fill(0);
    if (!lesion) for (let j = 0; j < NI; j++) u[inputIdx[j]] = inScale[j] * drive[j];
    for (let s = 0; s < substeps; s++) {
      let k = 0;
      for (let i = 0; i < N; i++) {
        let acc = 0;
        const end = rowPtr[i + 1];
        for (; k < end; k++) acc += val[k] * r[col[k]];
        cur[i] = acc + b[i] + u[i];
      }
      for (let i = 0; i < N; i++) {
        const x = cur[i];
        const clipped = x < 0 ? 0 : x > rMax ? rMax : x;
        const next = oneMinusA[i] * r[i] + a[i] * clipped;
        // snap vanishing rates to 0: silenced neurons otherwise decay into subnormal floats (stuck at the smallest
        // subnormal forever), which slows every later SpMV several-fold. |Δ| < R_FLOOR, far below parity tolerance.
        r[i] = next < R_FLOOR ? 0 : next;
      }
    }
    readout();
  }

  function reset() {
    for (let i = 0; i < N; i++) r[i] = r0[i];
    logits.fill(0);
    probs.fill(1 / C);
  }

  reset();
  return {
    r,
    logits,
    probs,
    step,
    reset,
    setLesion(on) {
      lesion = !!on;
    },
    get lesioned() {
      return lesion;
    },
  };
}
