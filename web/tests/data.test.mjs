import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { loadBundle } from "../data.js";
import { BUNDLE_DIR, fsFetch } from "./helpers.mjs";

const bundle = await loadBundle(BUNDLE_DIR, fsFetch());
const m = bundle.manifest;

test("counts match the manifest", () => {
  const c = m.counts;
  assert.equal(bundle.N, c.neurons);
  assert.equal(bundle.E, c.edges);
  assert.equal(bundle.NI, c.inputs);
  assert.equal(bundle.ND, c.dn);
  assert.equal(bundle.C, m.classes.length);
  assert.equal(bundle.pos.length, 3 * c.neurons);
  for (const k of ["bias", "alpha", "r0", "layer", "role", "sign", "side", "typeId"]) assert.equal(bundle[k].length, c.neurons, k);
  assert.equal(bundle.rowPtr.length, c.neurons + 1);
  assert.equal(bundle.col.length, c.edges);
  assert.equal(bundle.val.length, c.edges);
  assert.equal(bundle.edgePost.length, c.edges);
  for (const k of ["inputIdx", "inputKind", "inputSide"]) assert.equal(bundle[k].length, c.inputs, k);
  assert.equal(bundle.inputGain.length, 2);
  assert.equal(bundle.dnIdx.length, c.dn);
  assert.equal(bundle.Wout.length, bundle.C * c.dn);
  assert.equal(bundle.bOut.length, bundle.C);
  assert.equal(bundle.visEdges.length, c.vis_edges);
  assert.equal(bundle.dust.length, 3 * c.dust);
  assert.equal(bundle.info.types.length, c.types);
});

test("typed array kinds follow the layout codes", () => {
  assert.ok(bundle.pos instanceof Float32Array);
  assert.ok(bundle.sign instanceof Int8Array);
  assert.ok(bundle.typeId instanceof Uint16Array);
  assert.ok(bundle.rowPtr instanceof Uint32Array);
  assert.ok(bundle.val instanceof Float32Array);
  assert.ok(bundle.dust instanceof Float32Array);
  assert.ok(bundle.shells.brain.faces instanceof Uint32Array);
});

test("CSR is consistent: rowPtr[N] === E, monotone, edgePost matches rows, cols in range", () => {
  const { N, E, rowPtr, col, edgePost, visEdges } = bundle;
  assert.equal(rowPtr[0], 0);
  assert.equal(rowPtr[N], E);
  for (let i = 0; i < N; i++) {
    assert.ok(rowPtr[i] <= rowPtr[i + 1]);
    for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) if (edgePost[k] !== i) assert.fail(`edgePost[${k}]=${edgePost[k]} != ${i}`);
  }
  for (let k = 0; k < E; k++) if (col[k] >= N) assert.fail(`col[${k}] out of range`);
  for (let k = 0; k < visEdges.length; k++) if (visEdges[k] >= E) assert.fail(`visEdges[${k}] out of range`);
});

test("dust is scaled from i16 by units.dust_scale_um", async () => {
  const raw = await readFile(BUNDLE_DIR + "dust.bin");
  const i16 = new Int16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2);
  const s = m.units.dust_scale_um;
  for (const k of [0, 1, 2, 3000, i16.length - 1]) assert.equal(bundle.dust[k], Math.fround(i16[k] * s));
  const big = i16.findIndex((v) => Math.abs(v) > 100);
  assert.ok(Math.abs(bundle.dust[big]) < Math.abs(i16[big]));
});

test("shells parse: n_verts*3 === verts.length, faces index verts", () => {
  for (const name of ["brain", "vnc"]) {
    const sh = bundle.shells[name];
    assert.ok(sh.verts.length > 0);
    assert.equal(sh.nVerts * 3, sh.verts.length, name);
    assert.equal(sh.nFaces * 3, sh.faces.length, name);
    let max = 0;
    for (let k = 0; k < sh.faces.length; k++) if (sh.faces[k] > max) max = sh.faces[k];
    assert.ok(max < sh.nVerts);
  }
});

test("a byte-length mismatch throws a clear error", async () => {
  const good = await readFile(BUNDLE_DIR + "io.bin");
  const truncated = good.subarray(0, good.length - 4);
  await assert.rejects(loadBundle(BUNDLE_DIR, fsFetch({ [BUNDLE_DIR + "io.bin"]: truncated })), /io\.bin.*bytes/);
  const manifest = JSON.parse(await readFile(BUNDLE_DIR + "manifest.json", "utf8"));
  manifest.files["csr.bin"].layout[1][2] += 1; // layout now claims more bytes than the file has
  await assert.rejects(
    loadBundle(BUNDLE_DIR, fsFetch({ [BUNDLE_DIR + "manifest.json"]: Buffer.from(JSON.stringify(manifest)) })),
    /csr\.bin/,
  );
});

// --- validation: build a corrupted in-memory copy of one file / the manifest and expect a clear error ---
const fileBuf = async (name) => Buffer.from(await readFile(BUNDLE_DIR + name)); // private copy
const manifestCopy = async () => JSON.parse(await readFile(BUNDLE_DIR + "manifest.json", "utf8"));
const withFile = (name, buf) => loadBundle(BUNDLE_DIR, fsFetch({ [BUNDLE_DIR + name]: buf }));
const withManifest = (man) => withFile("manifest.json", Buffer.from(JSON.stringify(man)));

test("rejects an input side outside {1, 2}", async () => {
  const io = await fileBuf("io.bin");
  io.writeUInt32LE(0, 4 * 2 * bundle.NI + 4 * 7); // input_side[7] = 0
  await assert.rejects(withFile("io.bin", io), /inputSide\[7\].*0/);
});

test("rejects counts that disagree with manifest.counts", async () => {
  for (const [key, re] of [["neurons", /neurons/], ["edges", /edges/], ["inputs", /inputs/], ["dn", /dn/], ["vis_edges", /vis_edges/], ["dust", /dust/]]) {
    const man = await manifestCopy();
    man.counts[key] += 1;
    await assert.rejects(withManifest(man), re, key);
  }
});

test("rejects arrays whose lengths disagree with N, NI or C·ND", async () => {
  const cases = [
    ["neuron_meta.bin", 0, /layer/], // meta arrays vs N
    ["io.bin", 1, /inputKind/], // inputKind vs NI
    ["io.bin", 5, /Wout/], // W_out vs C·ND
  ];
  const BYTES = { u8: 1, i8: 1, u16: 2, u32: 4, f32: 4 };
  for (const [fname, field, re] of cases) {
    const man = await manifestCopy();
    const entry = man.files[fname];
    // drop 4 bytes' worth of items from the field (keeps later fields aligned), in both the layout and the file
    const off = entry.layout.slice(0, field).reduce((n, [, code, count]) => n + BYTES[code] * count, 0);
    entry.layout[field][2] -= 4 / BYTES[entry.layout[field][1]];
    entry.bytes -= 4;
    const buf = await fileBuf(fname);
    const shrunk = Buffer.concat([buf.subarray(0, off), buf.subarray(off + 4)]);
    const over = { [BUNDLE_DIR + "manifest.json"]: Buffer.from(JSON.stringify(man)), [BUNDLE_DIR + fname]: shrunk };
    await assert.rejects(loadBundle(BUNDLE_DIR, fsFetch(over)), re, fname);
  }
});

test("rejects out-of-range indices (col, inputIdx, dnIdx, visEdges)", async () => {
  const csr = await fileBuf("csr.bin");
  csr.writeUInt32LE(bundle.N, 4 * (bundle.N + 1) + 4 * 10); // col[10] = N
  await assert.rejects(withFile("csr.bin", csr), /col\[10\]/);

  const io1 = await fileBuf("io.bin");
  io1.writeUInt32LE(bundle.N, 4 * 3); // input_idx[3] = N
  await assert.rejects(withFile("io.bin", io1), /inputIdx\[3\]/);

  const io2 = await fileBuf("io.bin");
  io2.writeUInt32LE(bundle.N + 5, 4 * 3 * bundle.NI + 4 * 2 + 4 * 2); // after input_gain (2 f32): dn_idx[2]
  await assert.rejects(withFile("io.bin", io2), /dnIdx\[2\]/);

  const vis = await fileBuf("vis_edges.bin");
  vis.writeUInt32LE(bundle.E, 4 * 42);
  await assert.rejects(withFile("vis_edges.bin", vis), /visEdges\[42\]/);
});

test("clear errors for missing dust_scale_um and missing shell files", async () => {
  const man = await manifestCopy();
  delete man.units.dust_scale_um;
  await assert.rejects(withManifest(man), /dust_scale_um/);

  const man2 = await manifestCopy();
  delete man2.files["shell_vnc.bin"];
  await assert.rejects(withManifest(man2), /shell_vnc\.bin/);
});

test("fetch / json errors name the URL", async () => {
  await assert.rejects(withFile("neuron_info.json", Buffer.from("{not json")), (e) => e.message.includes(BUNDLE_DIR + "neuron_info.json"));
  const broken = async (url) => (url.endsWith("dust.bin") ? { ok: true, arrayBuffer: async () => { throw new Error("boom"); } } : fsFetch()(url));
  await assert.rejects(loadBundle(BUNDLE_DIR, broken), (e) => e.message.includes(BUNDLE_DIR + "dust.bin") && e.message.includes("boom"));
  const throwing = async (url) => (url.endsWith("csr.bin") ? Promise.reject(new Error("net down")) : fsFetch()(url));
  await assert.rejects(loadBundle(BUNDLE_DIR, throwing), (e) => e.message.includes(BUNDLE_DIR + "csr.bin") && e.message.includes("net down"));
});

test("a missing file throws", async () => {
  await assert.rejects(loadBundle(BUNDLE_DIR + "nope/", fsFetch()), /manifest\.json/);
});
