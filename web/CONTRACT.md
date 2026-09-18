# Web app module contract (P5)

The browser app: plain ES modules, no build step. three.js 0.186.0 is loaded through the importmap in `index.html`.

- **Serve:** `python3 -m http.server -d web 8000`
- **Tests:** `cd web && node --test` (Node 20+; passing a directory to `node --test` fails on Node 24)
- **Integration:** `main.js` is written by the lead after the three tracks merge. Each track builds against this contract and uses stubs for the other tracks.

## Ownership (one owner per file)
| Track | Owns |
|---|---|
| **js-model** | `data.js`, `stimulus.js`, `model.js`, `decision.js`, `tests/*.test.mjs`, `dev/selftest.html` |
| **scene** | `scene.js`, `layout.js`, `shaders/*.js`, `styles/app.css` (only the `#scene` section), `dev/scene-demo.html` |
| **game-hud** | `game.js`, `hud.js`, `styles/app.css` (all other regions), `dev/game-demo.html` |
| **lead** | `index.html`, `main.js`, `styles/tokens.css`, `CONTRACT.md` |

`styles/app.css` is shared. Each track adds its rules under its own banner comment (`/* === scene === */` or `/* === game-hud === */`). Never edit the other track's block.

## Data: bundle v1 (`assets/circuit_v1/`)
- `manifest.json` lists `files[fname].layout = [[field, code, count], …]`.
- Each `.bin` is its fields concatenated in order, little-endian, and every field is aligned to its item size.
- Codes: `u8`→Uint8Array, `i8`→Int8Array, `u16`→Uint16Array, `i16`→Int16Array, `u32`→Uint32Array, `f32`→Float32Array.
- CSR rows are the post neuron and `col` is the pre neuron.
- World units are µm. The fly's **left is −X**, the brain is up (+Y).
- `layer`: 0 input, 1 hidden (1 hop), 2 hidden (2+ hops), 3 descending, 4 motor.
- `role`: 0 input, 1 hidden, 2 descending, 3 motor. `sign` is +1, −1 or 0.
- `side`: 0 unknown, 1 left, 2 right.
- `neuron_info.json` has `{types[], instance[], superclass[], body[]}`, indexed by neuron (`types[type_id[i]]`).
- `manifest.viz.role_ref` = [input, hidden, DN, motor] display scales: the p90 peak rise above rest during probe looms. The scene maps activity as `1 − exp(−(r − rest)/role_ref[role])`.

## `data.js` (js-model)
```js
export async function loadBundle(baseUrl = "assets/circuit_v1/", fetchImpl = fetch) // → Bundle
// Bundle = {
//   manifest, info,                          // parsed JSON
//   N, E, C, NI, ND,                         // counts (C = classes.length)
//   pos: Float32Array(3N), bias, alpha, r0: Float32Array(N),
//   layer: Uint8Array(N), role: Uint8Array(N), sign: Int8Array(N), side: Uint8Array(N), typeId: Uint16Array(N),
//   rowPtr: Uint32Array(N+1), col: Uint32Array(E), val: Float32Array(E), edgePost: Uint32Array(E),  // derived row index
//   inputIdx, inputKind, inputSide: Uint32Array(NI), inputGain: Float32Array(2),
//   dnIdx: Uint32Array(ND), Wout: Float32Array(C*ND) /* row-major [class][dn] */, bOut: Float32Array(C),
//   visEdges: Uint32Array(V),                // indices into CSR entries
//   dust: Float32Array(3M) /* µm, already scaled */,
//   shells: { brain: {verts: Float32Array, faces: Uint32Array}, vnc: {...} }
// }
```
`loadBundle` must verify each file's byte length against its layout and throw a clear error if they don't match. sha256 checking is optional and can be skipped in dev.

## `stimulus.js` (js-model): mirrors `flybrain/stimulus.py` function by function
```js
export function loomAngleDeg(t, lv, tCollision)
export function loomRateDps(t, lv, tCollision)
export function rfGain(azimuthDeg, side, stim)            // stim = manifest.stimulus
export function threatClass(azimuthDeg, stim)             // 1 dodge_right (threat left) · 2 dodge_left · 3 takeoff
export function inputDrive(t, threats, inputKind, inputSide, stim, out?) // → Float64Array(NI); threat = {azimuthDeg, lv, tSpawn, tCollision}
```

## `model.js` (js-model): the frame contract (docs/model-contract.md), synchronous substeps
```js
export function createModel(bundle) // → Model
// Model = {
//   r: Float64Array(N)            // current rates (read by scene each frame)
//   logits: Float64Array(C), probs: Float64Array(C)   // updated by step()
//   step(drive /* ArrayLike(NI) */)   // one frame: u = inputGain[kind]*drive; `substeps` × r ← (1−α)r + α·clip(W r + bias + u, 0, r_max); readout
//   reset()                       // r ← r0
//   setLesion(on: boolean)        // when on, inputs are silenced (u = 0)
// }
```
Parity: `tests/model.test.mjs` replays `tests/fixtures/parity_v1.json` (a threat script → drive → logits and rates). It must match within `fixture.tolerance`, and the argmax must match on every frame.

## `decision.js` (js-model)
```js
export function createDecider({classes, p, frames}) // → { update(probs) → null | {classIndex, name}, reset() }
// Fires when the same non-"none" class has prob > p for `frames` consecutive frames; fires once per threat until reset().
```

## `layout.js` / `scene.js` (scene)
```js
export function layeredPositions(bundle) // → Float32Array(3N): x by layer, within a layer a disc (left side upper half, right lower), radius by type rank; same bounding scale as anatomy
export function createScene(container, bundle, opts = {}) // → Scene
// Scene = {
//   setActivity(r /* ArrayLike(N) */, rMax),   // per-frame activity → glow + synapse pulses
//   setLayout(name /* "anatomy" | "layers" */),  // eased 1.2 s morph
//   flashInputs(side /* 1 | 2 | "both" */),     // spawn feedback ring at the lobula
//   highlightDecision(classIndex, dnWeights /* Float32Array(ND) contribution */),
//   pick(clientX, clientY) → neuronIndex | -1,
//   setQuality("high" | "low"), setBloom(on), resize(), render(dtSeconds),
//   dispose()
// }
```

## `game.js` / `hud.js` (game-hud)
```js
export function createGame({arena, vfield, stim, classes, fps}) // → Game
// Game = {
//   spawn(azimuthDeg, lv = 0.04, approachS = 0.6),  // at the current sim time
//   threats(): Threat[],                           // active threats (for stimulus.inputDrive)
//   tick(tSim),                                    // advance, drop finished threats, resolve outcomes
//   resolve(decision /* {classIndex} */),          // animate dodge/takeoff; score ✓/✗ against the nearest active threat
//   onOutcome(cb), score(), draw(tSim)
// }
export function createHud({title, controls, decisionPanel, tooltip}, {classes, counts}) // → Hud
// Hud = {
//   setProbs(probs), showDecision({name, correct, latencyMs}),
//   setStats({fps, activeNeurons}), showTooltip(x, y, info | null),
//   onControl(cb /* ("layout"|"pause"|"slowmo"|"noise"|"bloom"|"reset"|"spawn", arg) */)
// }
```
Keys: ←/A left threat · ↑/W front · →/D right (Shift = fast loom) · L layout · Space pause · S slow-mo · N noise · B bloom · R reset · ? help.

### Additions agreed after review (game-hud)
- `game.onSpawn(cb /* (threat) */)` fires for **every** spawn (keys, clicks, autoplay). main.js uses it to call `scene.flashInputs(side)` and `decider.reset()`.
- Game gains `reset()`, `refreshTokens()` and `dispose()`; Hud gains `setScore`, `resetDecision`, `setControlState(name, value)`, `toggleHelp`, `dispose()`; `createHud` also takes the manifest's `decision`.
- Scoring: a decision is attributed to the active, unresolved threat with the **largest θ** among threats at least `minDecisionAgeS` (0.1 s) old. There is no θ threshold: offline evaluation (`train.decide`) accepts any decision before contact, and the trained network often commits below `label_onset_deg`; an onset threshold scored those correct, early decisions as false alarms plus misses in live play. The minimum age rejects decisions still driven by the previous threat; such a decision counts as a false alarm (see tests/game-logic.test.mjs). A decision with no qualifying threat is ignored within the linger window after a miss ("late"), otherwise it is a false alarm.
- Tooltip `info` = `{index, type, instance, superclass, role, layer, sign, activity, rMax}`. The Hud maps `layer` (0–4) and `sign` to labels.
- **Frame order in main.js:** `game.tick(t)` → `drive = inputDrive(t, game.threats(), …)` → `model.step(drive)` → `probs` → `decider.update(probs)` → if it fires, `game.resolve(decision)` + `scene.highlightDecision(…)` → `scene.setActivity(model.r, rMax)` → `hud.setProbs(probs)`.
- **OrbitControls must never call `listenToKeyEvents`**: it would `preventDefault` the arrow keys that spawn threats.
- Game/HUD key handlers ignore `e.repeat` for spawn and toggle keys.

## Visual language
- Tokens live in `styles/tokens.css`; hard-coded colors are not allowed.
- Background `--bg`; light comes only from additive glow.
- **Quiet by default:** idle neurons at 10–15% brightness, idle edges nearly invisible.
- Pulse color follows the sign of the presynaptic neuron (`--glow-excitatory` / `--glow-inhibitory`).
- Inputs are cyan and descending neurons are lime. The Giant Fiber (`types[type_id] == "DNp01"`) gets a ring.
- Motion is slow and eased; `prefers-reduced-motion` turns off auto-orbit.
- Typography: `--font-mono` for numbers and labels, `--font-sans` for prose.
