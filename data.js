// Loads flybrain-bundle v1 (see CONTRACT.md "Data" and flybrain/bundle.py). Works in the browser (fetch) and in
// node (tests pass a disk-backed fetchImpl returning {ok, json(), arrayBuffer()}).

const TYPED = {
  u8: Uint8Array,
  i8: Int8Array,
  u16: Uint16Array,
  i16: Int16Array,
  u32: Uint32Array,
  f32: Float32Array,
};

// every .bin the loader reads; any other files listed in the manifest are still parsed and length-checked
const REQUIRED_FILES = ["neurons.bin", "neuron_meta.bin", "csr.bin", "io.bin", "vis_edges.bin", "dust.bin", "shell_brain.bin", "shell_vnc.bin"];

function fail(msg) {
  throw new Error(`loadBundle: ${msg}`);
}

/** fetch + body read, with the URL in every error (network, HTTP status, JSON parse, body read). */
async function fetchBody(fetchImpl, url, kind /* "json" | "arrayBuffer" */) {
  let res;
  try {
    res = await fetchImpl(url);
  } catch (e) {
    fail(`failed to fetch ${url}: ${e?.message ?? e}`);
  }
  if (!res || !res.ok) fail(`failed to fetch ${url} (${res ? `${res.status} ${res.statusText ?? ""}`.trim() : "no response"})`);
  try {
    return await res[kind]();
  } catch (e) {
    fail(`failed to read ${url} as ${kind}: ${e?.message ?? e}`);
  }
}

/** Split one .bin into typed-array views per its manifest layout; throws if the byte length does not match. */
export function parseBin(fname, buffer, entry) {
  const expected = entry.layout.reduce((n, [, code, count]) => {
    const T = TYPED[code];
    if (!T) fail(`${fname}: unknown dtype code ${JSON.stringify(code)}`);
    if (n % T.BYTES_PER_ELEMENT) fail(`${fname}: field misaligned at byte ${n}`);
    return n + T.BYTES_PER_ELEMENT * count;
  }, 0);
  if (buffer.byteLength !== expected || (entry.bytes != null && entry.bytes !== expected)) {
    fail(
      `${fname}: file has ${buffer.byteLength} bytes but its layout needs ${expected} bytes` +
        (entry.bytes != null ? ` (manifest says ${entry.bytes} bytes)` : ""),
    );
  }
  const out = {};
  let offset = 0;
  for (const [name, code, count] of entry.layout) {
    const T = TYPED[code];
    out[name] = new T(buffer, offset, count); // views share the file's ArrayBuffer (little-endian hosts)
    offset += T.BYTES_PER_ELEMENT * count;
  }
  return out;
}

function field(f, fname, name) {
  const v = f[fname][name];
  if (!v) fail(`${fname}: layout has no field ${JSON.stringify(name)}`);
  return v;
}

function shell(f, fname) {
  const nVerts = field(f, fname, "n_verts")[0];
  const nFaces = field(f, fname, "n_faces")[0];
  const verts = field(f, fname, "verts");
  const faces = field(f, fname, "faces");
  if (nVerts * 3 !== verts.length || nFaces * 3 !== faces.length) {
    fail(`${fname}: shell counts disagree (n_verts=${nVerts}, verts=${verts.length}, n_faces=${nFaces}, faces=${faces.length})`);
  }
  for (let k = 0; k < faces.length; k++) if (faces[k] >= nVerts) fail(`${fname}: faces[${k}]=${faces[k]} >= n_verts ${nVerts}`);
  return { nVerts, nFaces, verts, faces };
}

function checkLength(name, arr, expected, what) {
  if (arr.length !== expected) fail(`${name} has length ${arr.length}, expected ${expected} (${what})`);
}

function checkRange(name, arr, limit, what) {
  for (let k = 0; k < arr.length; k++) if (arr[k] >= limit) fail(`${name}[${k}]=${arr[k]} out of range (must be < ${what} = ${limit})`);
}

export async function loadBundle(baseUrl = "assets/circuit_v1/", fetchImpl = fetch) {
  if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) fail("big-endian hosts are not supported");
  const base = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
  const manifest = await fetchBody(fetchImpl, base + "manifest.json", "json");
  if (manifest.format !== "flybrain-bundle" || manifest.version !== 1) {
    fail(`unsupported bundle ${manifest.format} v${manifest.version} at ${base}manifest.json`);
  }
  const files = manifest.files ?? {};
  for (const n of REQUIRED_FILES) if (!files[n]) fail(`manifest.json does not list required file ${n}`);
  const dustScale = manifest.units?.dust_scale_um;
  if (!(typeof dustScale === "number" && dustScale > 0)) fail(`manifest.json units.dust_scale_um missing or invalid (${dustScale})`);
  const counts = manifest.counts;
  if (!counts) fail("manifest.json has no counts");
  if (!Array.isArray(manifest.classes) || !manifest.classes.length) fail("manifest.json has no classes");

  const names = Object.keys(files);
  const [info, ...buffers] = await Promise.all([
    fetchBody(fetchImpl, base + "neuron_info.json", "json"),
    ...names.map((n) => fetchBody(fetchImpl, base + n, "arrayBuffer")),
  ]);
  const f = {};
  names.forEach((n, i) => (f[n] = parseBin(n, buffers[i], files[n])));

  const N = counts.neurons, E = counts.edges, NI = counts.inputs, ND = counts.dn, C = manifest.classes.length;
  const b = {
    manifest,
    info,
    N,
    E,
    C,
    NI,
    ND,
    pos: field(f, "neurons.bin", "pos"),
    bias: field(f, "neurons.bin", "bias"),
    alpha: field(f, "neurons.bin", "alpha"),
    r0: field(f, "neurons.bin", "r0"),
    layer: field(f, "neuron_meta.bin", "layer"),
    role: field(f, "neuron_meta.bin", "role"),
    sign: field(f, "neuron_meta.bin", "sign"),
    side: field(f, "neuron_meta.bin", "side"),
    typeId: field(f, "neuron_meta.bin", "type_id"),
    rowPtr: field(f, "csr.bin", "row_ptr"),
    col: field(f, "csr.bin", "col"),
    val: field(f, "csr.bin", "val"),
    edgePost: null,
    inputIdx: field(f, "io.bin", "input_idx"),
    inputKind: field(f, "io.bin", "input_kind"),
    inputSide: field(f, "io.bin", "input_side"),
    inputGain: field(f, "io.bin", "input_gain"),
    dnIdx: field(f, "io.bin", "dn_idx"),
    Wout: field(f, "io.bin", "W_out"),
    bOut: field(f, "io.bin", "b_out"),
    visEdges: field(f, "vis_edges.bin", "edge_idx"),
    dust: null,
    shells: { brain: shell(f, "shell_brain.bin"), vnc: shell(f, "shell_vnc.bin") },
  };

  // lengths vs manifest.counts
  const nWhat = `counts.neurons`;
  checkLength("pos", b.pos, 3 * N, `3 × ${nWhat}`);
  for (const k of ["bias", "alpha", "r0", "layer", "role", "sign", "side", "typeId"]) checkLength(k, b[k], N, nWhat);
  checkLength("rowPtr", b.rowPtr, N + 1, `${nWhat} + 1`);
  for (const k of ["col", "val"]) checkLength(k, b[k], E, "counts.edges");
  for (const k of ["inputIdx", "inputKind", "inputSide"]) checkLength(k, b[k], NI, "counts.inputs");
  checkLength("inputGain", b.inputGain, manifest.stimulus?.input_kinds?.length ?? 2, "stimulus.input_kinds");
  checkLength("dnIdx", b.dnIdx, ND, "counts.dn");
  checkLength("Wout", b.Wout, C * ND, "classes × counts.dn");
  checkLength("bOut", b.bOut, C, "classes");
  checkLength("visEdges", b.visEdges, counts.vis_edges, "counts.vis_edges");
  const rawDust = field(f, "dust.bin", "pos");
  checkLength("dust", rawDust, 3 * counts.dust, "3 × counts.dust");

  // CSR structure and index ranges
  const rp = b.rowPtr;
  if (rp[0] !== 0 || rp[N] !== E) fail(`csr.bin row_ptr must start at 0 and end at E=${E} (got ${rp[0]} … ${rp[N]})`);
  for (let i = 0; i < N; i++) if (rp[i] > rp[i + 1]) fail(`csr.bin row_ptr decreases at ${i}`);
  checkRange("col", b.col, N, "N");
  checkRange("inputIdx", b.inputIdx, N, "N");
  checkRange("dnIdx", b.dnIdx, N, "N");
  checkRange("visEdges", b.visEdges, E, "E");
  checkRange("inputKind", b.inputKind, b.inputGain.length, "inputGain.length");
  for (let k = 0; k < NI; k++) {
    const s = b.inputSide[k];
    if (s !== 1 && s !== 2) fail(`inputSide[${k}]=${s} must be 1 (left) or 2 (right)`);
  }

  b.edgePost = new Uint32Array(E);
  for (let i = 0; i < N; i++) b.edgePost.fill(i, rp[i], rp[i + 1]);

  b.dust = new Float32Array(rawDust.length);
  for (let k = 0; k < rawDust.length; k++) b.dust[k] = rawDust[k] * dustScale;

  return b;
}
