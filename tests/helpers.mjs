// Shared test helpers (not a test file): a disk-backed fetch so web/data.js runs unchanged under node.
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

export const WEB_DIR = fileURLToPath(new URL("../", import.meta.url));
export const BUNDLE_DIR = WEB_DIR + "assets/circuit_v1/";

export function fsFetch(overrides = {}) {
  return async (url) => {
    if (url in overrides) return response(overrides[url]);
    try {
      return response(await readFile(url));
    } catch {
      return { ok: false, status: 404, statusText: `not found: ${url}` };
    }
  };
}

function response(buf) {
  return {
    ok: true,
    status: 200,
    json: async () => JSON.parse(Buffer.from(buf).toString("utf8")),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  };
}

export async function loadFixture() {
  return JSON.parse(await readFile(new URL("./fixtures/parity_v1.json", import.meta.url), "utf8"));
}
