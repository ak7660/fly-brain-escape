// 3D circuit view (scene track). Depends only on the Bundle *shape* from CONTRACT.md, never on how it was loaded.
//
// Usage notes
// - createScene must run after styles/tokens.css has loaded: colors are read from CSS custom properties once.
// - render(dt) takes WALL-CLOCK seconds (pulses, morph, flashes and the idle orbit are real-time animations),
//   independent of simulation speed / slow-motion. Call setActivity(model.r) once per simulated frame.
// - Activity is shown as change from rest: a slowly adapting per-neuron baseline (starting at bundle.r0) is subtracted
//   and the rise is normalized per role by manifest.viz.role_ref (p90 peak rise for probe looms).
// - pick(clientX, clientY) is meant for a pointermove listener on the canvas (scene.canvas), not on window.
// - highlightDecision(classIndex, dnWeights): dnWeights[j] = Wout[c, j] · (r[dnIdx[j]] − r0[dnIdx[j]]); only positive
//   entries count. The top 12 DNs flare (size, brightness, white core) and so do the motor neurons they feed.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

import { layeredPositions, anatomyExtent, LAYER_COUNT } from "./layout.js";
import { neuronVertex, neuronFragment } from "./shaders/neurons.js";
import { synapseVertex, synapseFragment } from "./shaders/synapses.js";
import { shellVertex, shellFragment } from "./shaders/shell.js";
import { dustVertex, dustFragment } from "./shaders/dust.js";
import { gfRingVertex, gfRingFragment, flashVertex, flashFragment } from "./shaders/rings.js";

const MORPH_S = 1.2;
const IDLE_RESUME_S = 4;
const FLASH_S = 1.3;
const HIGHLIGHT_TOP = 12;
const FLASH_POOL = 8;
const ACT_WIDTH = 128;
const FOV = 30;
const LOW_EDGES = 25000;

const BLOOM = { strength: 0.7, radius: 0.25, threshold: 0.25 };
const ROLE_REF_FALLBACK = [0.8, 0.1, 0.15, 0.02]; // input, hidden, DN, motor
const BASE_DOWN = 0.05, BASE_UP = 0.0015;         // baseline adaptation per setActivity call
const TAU_UP = 0.05, TAU_DOWN = 0.35, TAU_HI = 1.1; // display smoothing (s)

const ROLE_INPUT = 0, ROLE_HIDDEN = 1, ROLE_DN = 2, ROLE_MN = 3;

const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

function readToken(name) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  if (!v) console.warn(`[scene] design token ${name} is not defined (did styles/tokens.css load first?)`);
  return new THREE.Color(v || "white"); // sRGB token → linear working color
}

function additive(params) {
  return new THREE.ShaderMaterial({ blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, transparent: true, ...params });
}

/**
 * @param {HTMLElement} container
 * @param {object} bundle  Bundle per CONTRACT.md
 * @param {{quality?: "high"|"low", bloom?: boolean, autoOrbit?: boolean, reducedMotion?: boolean}} opts
 */
export function createScene(container, bundle, opts = {}) {
  const { N, pos, role, sign, typeId, info, layer } = bundle;
  const reducedMotion = opts.reducedMotion ?? matchMedia("(prefers-reduced-motion: reduce)").matches;
  const autoOrbit = (opts.autoOrbit ?? true) && !reducedMotion;
  let quality = opts.quality === "low" ? "low" : "high";
  let bloomOn = opts.bloom ?? true;
  let disposed = false;

  const C = {
    bg: readToken("--bg"),
    input: readToken("--glow-input"),
    exc: readToken("--glow-excitatory"),
    inh: readToken("--glow-inhibitory"),
    mod: readToken("--glow-modulatory"),
    dn: readToken("--glow-dn"),
    mn: readToken("--glow-mn"),
    dust: readToken("--glow-dust"),
    shell: readToken("--glow-shell"),
  };
  const roleRef = Float32Array.from(
    Array.isArray(bundle.manifest?.viz?.role_ref) && bundle.manifest.viz.role_ref.length === 4 ? bundle.manifest.viz.role_ref : ROLE_REF_FALLBACK,
  );

  // ---------- renderer ----------
  const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false, powerPreference: "high-performance" });
  const pixelRatio = () => (quality === "low" ? 1 : Math.min(window.devicePixelRatio || 1, 1.25));
  renderer.setPixelRatio(pixelRatio());
  // No tone mapping: tone curves crush and hue-shift the near-black background; brightness is budgeted instead.
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.domElement.classList.add("scene-canvas");
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  // scene.background (not setClearColor): the clear color is not linearized for float render targets
  scene.background = C.bg;
  const camera = new THREE.PerspectiveCamera(FOV, 1, 5, 20000);
  const S = anatomyExtent(pos);

  // ---------- activity texture (float RGBA, one texel per neuron) ----------
  const actHeight = Math.ceil(N / ACT_WIDTH);
  const actData = new Float32Array(ACT_WIDTH * actHeight * 4);
  const actTex = new THREE.DataTexture(actData, ACT_WIDTH, actHeight, THREE.RGBAFormat, THREE.FloatType);
  actTex.minFilter = actTex.magFilter = THREE.NearestFilter;
  actTex.generateMipmaps = false;
  actTex.needsUpdate = true;

  const baseline = Float32Array.from(bundle.r0); // adapting rest level
  const target = new Float32Array(N);            // normalized rise above rest, from setActivity
  const display = new Float32Array(N);           // smoothed activity on screen
  const highlight = new Float32Array(N);         // per-neuron decision flare (DNs and their motor targets)

  const shared = {
    uAct: { value: actTex },
    uActWidth: { value: ACT_WIDTH },
    uMix: { value: 0 },
    uTime: { value: 0 },
    uSizeScale: { value: 1 },
    uMaxPoint: { value: 96 },
  };

  // ---------- layouts ----------
  const layered = layeredPositions(bundle);

  // slab geometry per layer (for framing and HUD anchors)
  const slab = Array.from({ length: LAYER_COUNT }, () => ({ x: 0, n: 0, rad: 0 }));
  const lb = { minX: Infinity, maxX: -Infinity, maxR: 0 };
  for (let i = 0; i < N; i++) {
    const s = slab[layer[i]], x = layered[3 * i], rr = Math.hypot(layered[3 * i + 1], layered[3 * i + 2]);
    s.x += x; s.n++; s.rad = Math.max(s.rad, rr);
    lb.minX = Math.min(lb.minX, x); lb.maxX = Math.max(lb.maxX, x); lb.maxR = Math.max(lb.maxR, rr);
  }
  for (const s of slab) s.x /= s.n || 1;

  // ---------- neurons ----------
  const nColor = new Float32Array(3 * N), nSize = new Float32Array(N), nIndex = new Float32Array(N), nIdle = new Float32Array(N);
  const gfIdx = [];
  for (let i = 0; i < N; i++) {
    let c, s, idle = 1;
    if (role[i] === ROLE_INPUT) { c = C.input; s = 5.5; }
    else if (role[i] === ROLE_HIDDEN) {
      c = sign[i] > 0 ? C.exc : sign[i] < 0 ? C.inh : C.mod; s = 4.2;
      if (sign[i] < 0) idle = 0.6; // the inhibitory magenta otherwise speckles the hidden cloud
    } else if (role[i] === ROLE_DN) { c = C.dn; s = 7.5; idle = 1.3; } // bigger + brighter: not hue alone (CVD)
    else { c = C.mn; s = 5.0; }
    nColor[3 * i] = c.r; nColor[3 * i + 1] = c.g; nColor[3 * i + 2] = c.b;
    nSize[i] = s; nIndex[i] = i; nIdle[i] = idle;
    if (info?.types?.[typeId[i]] === "DNp01") { gfIdx.push(i); nSize[i] = 10; }
  }
  const neuronGeo = new THREE.BufferGeometry();
  neuronGeo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  neuronGeo.setAttribute("aLayered", new THREE.BufferAttribute(layered, 3));
  neuronGeo.setAttribute("aColor", new THREE.BufferAttribute(nColor, 3));
  neuronGeo.setAttribute("aSize", new THREE.BufferAttribute(nSize, 1));
  neuronGeo.setAttribute("aIndex", new THREE.BufferAttribute(nIndex, 1));
  neuronGeo.setAttribute("aIdle", new THREE.BufferAttribute(nIdle, 1));
  const neuronMat = additive({
    vertexShader: neuronVertex,
    fragmentShader: neuronFragment,
    uniforms: { ...shared, uIdle: { value: 0.1 }, uGain: { value: 1.0 }, uCamDist: { value: 2000 }, uDepthRange: { value: S } },
  });
  const neurons = new THREE.Points(neuronGeo, neuronMat);
  neurons.frustumCulled = false;
  neurons.renderOrder = 3;

  // ---------- synapses (built strongest first, so a draw-range prefix keeps the strongest edges) ----------
  const { col, edgePost, val, visEdges } = bundle;
  const V = visEdges.length;
  const order = Uint32Array.from({ length: V }, (_, k) => k).sort(
    (a, b) => Math.abs(val[visEdges[b]]) - Math.abs(val[visEdges[a]]),
  );
  const wRef = Math.abs(val[visEdges[order[Math.floor(0.05 * (V - 1))]]]) || 1; // 95th percentile |w|
  const ePos = new Float32Array(6 * V), eLay = new Float32Array(6 * V), eAttr = new Float32Array(8 * V);
  for (let k = 0; k < V; k++) {
    const e = visEdges[order[k]], pre = col[e], post = edgePost[e];
    const w = Math.pow(Math.min(1, Math.abs(val[e]) / wRef), 0.6);
    for (let end = 0; end < 2; end++) {
      const n = end === 0 ? pre : post, o = 6 * k + 3 * end;
      ePos[o] = pos[3 * n]; ePos[o + 1] = pos[3 * n + 1]; ePos[o + 2] = pos[3 * n + 2];
      eLay[o] = layered[3 * n]; eLay[o + 1] = layered[3 * n + 1]; eLay[o + 2] = layered[3 * n + 2];
      const a = 8 * k + 4 * end;
      eAttr[a] = end; eAttr[a + 1] = pre; eAttr[a + 2] = sign[pre]; eAttr[a + 3] = w;
    }
  }
  const edgeGeo = new THREE.BufferGeometry();
  edgeGeo.setAttribute("position", new THREE.BufferAttribute(ePos, 3));
  edgeGeo.setAttribute("aLayered", new THREE.BufferAttribute(eLay, 3));
  edgeGeo.setAttribute("aEdge", new THREE.BufferAttribute(eAttr, 4));
  const EDGE_IDLE_ALPHA = 0.0003;
  const edgeMat = additive({
    vertexShader: synapseVertex,
    fragmentShader: synapseFragment,
    uniforms: {
      ...shared,
      uExc: { value: C.exc }, uInh: { value: C.inh }, uMod: { value: C.mod },
      uSpeed: { value: 1.8 },
      uIdleAlpha: { value: EDGE_IDLE_ALPHA },
      uGain: { value: 0.2 },
      uTail: { value: 16 },
      uBase: { value: 0.02 },
      uFade: { value: 1 },
    },
  });
  const edges = new THREE.LineSegments(edgeGeo, edgeMat);
  edges.frustumCulled = false;
  edges.renderOrder = 2;

  // ---------- shells ----------
  const shellMat = additive({
    vertexShader: shellVertex,
    fragmentShader: shellFragment,
    side: THREE.DoubleSide,
    uniforms: { uColor: { value: C.shell }, uAlpha: { value: 0.012 }, uFade: { value: 1 } },
  });
  const shellMeshes = [];
  for (const key of ["brain", "vnc"]) {
    const sh = bundle.shells?.[key];
    if (!sh || !sh.verts?.length) continue;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(sh.verts, 3));
    g.setIndex(new THREE.BufferAttribute(sh.faces, 1));
    g.computeVertexNormals();
    const m = new THREE.Mesh(g, shellMat);
    m.renderOrder = 0;
    m.frustumCulled = false;
    shellMeshes.push(m);
  }

  // ---------- dust: evens then odds, so drawing the first half is an unbiased half sample ----------
  const M = Math.floor((bundle.dust?.length || 0) / 3);
  const dustPos = new Float32Array(3 * M);
  {
    let o = 0;
    for (let pass = 0; pass < 2; pass++) {
      for (let j = pass; j < M; j += 2) {
        dustPos[o++] = bundle.dust[3 * j]; dustPos[o++] = bundle.dust[3 * j + 1]; dustPos[o++] = bundle.dust[3 * j + 2];
      }
    }
  }
  const dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute("position", new THREE.BufferAttribute(dustPos, 3));
  const dustMat = additive({
    vertexShader: dustVertex,
    fragmentShader: dustFragment,
    uniforms: { uColor: { value: C.dust }, uAlpha: { value: 0.015 }, uFade: { value: 1 }, uPointSize: { value: 1 } },
  });
  const dust = new THREE.Points(dustGeo, dustMat);
  dust.frustumCulled = false;
  dust.renderOrder = 1;

  // ---------- Giant Fiber reticle ----------
  const gfGeo = new THREE.BufferGeometry();
  {
    const p = new Float32Array(3 * gfIdx.length), l = new Float32Array(3 * gfIdx.length), ix = new Float32Array(gfIdx.length);
    gfIdx.forEach((n, j) => {
      for (let d = 0; d < 3; d++) { p[3 * j + d] = pos[3 * n + d]; l[3 * j + d] = layered[3 * n + d]; }
      ix[j] = n;
    });
    gfGeo.setAttribute("position", new THREE.BufferAttribute(p, 3));
    gfGeo.setAttribute("aLayered", new THREE.BufferAttribute(l, 3));
    gfGeo.setAttribute("aIndex", new THREE.BufferAttribute(ix, 1));
  }
  const gfMat = additive({
    vertexShader: gfRingVertex,
    fragmentShader: gfRingFragment,
    uniforms: { ...shared, uColor: { value: C.dn }, uWorldSize: { value: 34 }, uLayersScale: { value: 2.2 } },
  });
  const gfRings = new THREE.Points(gfGeo, gfMat);
  gfRings.frustumCulled = false;
  gfRings.renderOrder = 4;

  // ---------- input flash rings ----------
  const sideMean = { 1: [new Float32Array(3), new Float32Array(3)], 2: [new Float32Array(3), new Float32Array(3)] };
  {
    const cnt = { 1: 0, 2: 0 };
    const { inputIdx, inputSide } = bundle;
    for (let j = 0; j < inputIdx.length; j++) {
      const sd = inputSide[j], n = inputIdx[j];
      if (!sideMean[sd]) continue;
      cnt[sd]++;
      for (let d = 0; d < 3; d++) { sideMean[sd][0][d] += pos[3 * n + d]; sideMean[sd][1][d] += layered[3 * n + d]; }
    }
    for (const sd of [1, 2]) for (const arr of sideMean[sd]) for (let d = 0; d < 3; d++) arr[d] /= cnt[sd] || 1;
  }
  const flashPos = new Float32Array(3 * FLASH_POOL), flashLay = new Float32Array(3 * FLASH_POOL), flashStart = new Float32Array(FLASH_POOL).fill(-1e6);
  const flashGeo = new THREE.BufferGeometry();
  flashGeo.setAttribute("position", new THREE.BufferAttribute(flashPos, 3));
  flashGeo.setAttribute("aLayered", new THREE.BufferAttribute(flashLay, 3));
  flashGeo.setAttribute("aStart", new THREE.BufferAttribute(flashStart, 1));
  const FLASH_WORLD = 0.42 * S;
  const flashMat = additive({
    vertexShader: flashVertex,
    fragmentShader: flashFragment,
    uniforms: { ...shared, uColor: { value: C.input }, uDuration: { value: FLASH_S }, uWorldSize: { value: FLASH_WORLD } },
  });
  const flashes = new THREE.Points(flashGeo, flashMat);
  flashes.frustumCulled = false;
  flashes.renderOrder = 5;
  let flashNext = 0;

  scene.add(...shellMeshes, dust, edges, neurons, gfRings, flashes);

  // ---------- DN → motor targets (for the decision flare) ----------
  const { dnIdx, rowPtr } = bundle;
  const dnSlot = new Int32Array(N).fill(-1);
  for (let j = 0; j < dnIdx.length; j++) dnSlot[dnIdx[j]] = j;
  const dnMotorLists = Array.from({ length: dnIdx.length }, () => []);
  for (let post = 0; post < N; post++) {
    if (role[post] !== ROLE_MN) continue;
    for (let k = rowPtr[post]; k < rowPtr[post + 1]; k++) {
      const j = dnSlot[col[k]];
      if (j >= 0 && val[k] > 0) dnMotorLists[j].push(post);
    }
  }
  const dnMotor = dnMotorLists.map((l) => Uint32Array.from(l));

  // ---------- post ----------
  const composer = new EffectComposer(renderer);
  const renderPass = new RenderPass(scene, camera);
  composer.addPass(renderPass);
  // UnrealBloomPass starts its mip chain at resolution / 2 → half-resolution bloom.
  const bloomPass = new UnrealBloomPass(new THREE.Vector2(256, 256), BLOOM.strength, BLOOM.radius, BLOOM.threshold);
  composer.addPass(bloomPass);
  const outputPass = new OutputPass();
  composer.addPass(outputPass);

  // ---------- camera + controls (never listenToKeyEvents: arrow keys belong to the game) ----------
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.rotateSpeed = 0.6;
  controls.zoomSpeed = 0.8;
  controls.minDistance = 0.3 * S;
  controls.maxDistance = 12 * S;

  // Views are spherical about a target: az = 0 looks from +Z, az = π from −Z (front: fly's left on screen right).
  // Both views look from the front, so the morph is a dolly with a slight turn, not a swing around the fly.
  const layersCenterX = (lb.minX + lb.maxX) / 2;
  const VIEWS = {
    anatomy: { az: Math.PI, el: 0.1, target: new THREE.Vector3(0, 0.02 * S, 0), halfH: 1.2 * S, halfW: 0.9 * S },
    layers: {
      az: Math.PI + 0.42, el: 0.14,
      target: new THREE.Vector3(layersCenterX, 0, 0),
      // room for the flash ring at the input slab and the enlarged GF reticle
      halfH: lb.maxR + 0.3 * S,
      halfW: (lb.maxX - lb.minX) / 2 + 0.35 * S,
    },
  };
  let layoutName = "anatomy";
  const tanHalf = Math.tan(THREE.MathUtils.degToRad(FOV / 2));
  const viewDistance = (v) => Math.max(v.halfH / tanHalf, v.halfW / (tanHalf * Math.max(camera.aspect, 0.3)));
  function placeCamera(az, el, dist, tgt) {
    camera.position.set(
      tgt.x + dist * Math.cos(el) * Math.sin(az),
      tgt.y + dist * Math.sin(el),
      tgt.z + dist * Math.cos(el) * Math.cos(az),
    );
    controls.target.copy(tgt);
    camera.lookAt(tgt);
  }

  const tween = { active: false, camera: false, t: 0, from: 0, to: 0, az0: 0, el0: 0, d0: 0, az1: 0, el1: 0, d1: 0, tg0: new THREE.Vector3(), tg1: new THREE.Vector3() };
  const tmpV = new THREE.Vector3();

  let clock = 0;
  let lastInteraction = -1e9, interacting = false, idleT = 0;
  const onStart = () => { interacting = true; lastInteraction = clock; tween.camera = false; }; // the user wins over the tween
  const onEnd = () => { interacting = false; lastInteraction = clock; };
  controls.addEventListener("start", onStart);
  controls.addEventListener("end", onEnd);

  // ---------- quality / bloom ----------
  function applyQuality() {
    const low = quality === "low";
    edgeGeo.setDrawRange(0, 2 * (low ? Math.min(LOW_EDGES, V) : V));
    dustGeo.setDrawRange(0, low ? Math.ceil(M / 2) : M);
    edgeMat.uniforms.uIdleAlpha.value = low ? 0 : EDGE_IDLE_ALPHA; // low: silent edges are collapsed (no overdraw)
    shellMat.side = low ? THREE.FrontSide : THREE.DoubleSide;
    shellMat.needsUpdate = true;
    bloomPass.enabled = bloomOn && !low;
    neuronMat.uniforms.uGain.value = bloomPass.enabled ? 1.0 : 1.25; // without bloom, lift the gain so activity reads
    if (renderer.getPixelRatio() !== pixelRatio()) { renderer.setPixelRatio(pixelRatio()); resize(); }
  }

  // ---------- resize ----------
  let viewW = 1, viewH = 1;
  function resize() {
    if (disposed) return;
    viewW = Math.max(1, container.clientWidth);
    viewH = Math.max(1, container.clientHeight);
    renderer.setSize(viewW, viewH);
    composer.setPixelRatio(renderer.getPixelRatio());
    composer.setSize(viewW, viewH);
    camera.aspect = viewW / viewH;
    camera.updateProjectionMatrix();
    const dpr = renderer.getPixelRatio();
    shared.uSizeScale.value = (viewH * dpr) / (2 * tanHalf);
    shared.uMaxPoint.value = Math.max(24, (96 * viewH * dpr) / 1000);
  }
  resize();
  applyQuality();
  {
    const v = VIEWS.anatomy;
    placeCamera(v.az, v.el, viewDistance(v), v.target);
    controls.update();
  }
  const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => resize()) : null;
  ro?.observe(container);

  // ---------- picking ----------
  const raycaster = new THREE.Raycaster();
  const pickPos = new Float32Array(3 * N);
  const pickGeo = new THREE.BufferGeometry();
  pickGeo.setAttribute("position", new THREE.BufferAttribute(pickPos, 3));
  const pickPoints = new THREE.Points(pickGeo);
  pickPoints.updateMatrixWorld(true);
  let pickMix = -1;
  const hits = [];
  const ndc = new THREE.Vector2();
  const chosen = new Int32Array(HIGHLIGHT_TOP);

  // ---------- HUD anchors ----------
  const anchors = Array.from({ length: LAYER_COUNT }, (_, L) => ({ layer: L, x: 0, y: 0, opacity: 0 }));

  // ---------- API ----------
  const api = {
    canvas: renderer.domElement,

    /** Per simulated frame. r: rates (ArrayLike(N)); rMax is accepted for the contract but not needed. */
    setActivity(r /* , rMax */) {
      const n = Math.min(N, r.length);
      for (let i = 0; i < n; i++) {
        const d = r[i] - baseline[i];
        baseline[i] += d * (d < 0 ? BASE_DOWN : BASE_UP);
        target[i] = d > 0 ? 1 - Math.exp(-d / roleRef[role[i]]) : 0;
      }
    },

    setLayout(name) {
      if (name !== "anatomy" && name !== "layers") return;
      if (name === layoutName && !tween.active) return;
      layoutName = name;
      const v = VIEWS[name];
      if (reducedMotion) { // instant cut
        tween.active = false;
        shared.uMix.value = name === "layers" ? 1 : 0;
        placeCamera(v.az, v.el, viewDistance(v), v.target);
        return;
      }
      tween.active = true;
      tween.camera = true;
      tween.t = 0;
      tween.from = shared.uMix.value;
      tween.to = name === "layers" ? 1 : 0;
      tmpV.copy(camera.position).sub(controls.target);
      tween.d0 = tmpV.length();
      tween.el0 = Math.asin(THREE.MathUtils.clamp(tmpV.y / tween.d0, -1, 1));
      tween.az0 = Math.atan2(tmpV.x, tmpV.z);
      tween.tg0.copy(controls.target);
      tween.el1 = v.el; tween.d1 = viewDistance(v); tween.tg1.copy(v.target);
      const dAz = Math.atan2(Math.sin(v.az - tween.az0), Math.cos(v.az - tween.az0)); // shortest way round
      tween.az1 = tween.az0 + dAz;
    },

    flashInputs(side) {
      const sides = side === "both" ? [1, 2] : [side];
      for (const sd of sides) {
        const m = sideMean[sd];
        if (!m) continue;
        const j = flashNext++ % FLASH_POOL;
        flashPos.set(m[0], 3 * j);
        flashLay.set(m[1], 3 * j);
        flashStart[j] = clock;
      }
      flashGeo.attributes.position.needsUpdate = true;
      flashGeo.attributes.aLayered.needsUpdate = true;
      flashGeo.attributes.aStart.needsUpdate = true;
    },

    /** dnWeights[j] = Wout[classIndex, j] · (r[dnIdx[j]] − r0[dnIdx[j]]); positives only. classIndex is informational. */
    highlightDecision(classIndex, dnWeights) {
      const ND = Math.min(dnIdx.length, dnWeights?.length || 0);
      chosen.fill(-1);
      for (let k = 0; k < HIGHLIGHT_TOP; k++) {
        let best = -1, bestW = 0;
        for (let j = 0; j < ND; j++) {
          const w = dnWeights[j];
          if (w > bestW && !chosen.includes(j)) { best = j; bestW = w; }
        }
        if (best < 0) break;
        chosen[k] = best;
        highlight[dnIdx[best]] = 1;
        const mns = dnMotor[best];
        for (let m = 0; m < mns.length; m++) highlight[mns[m]] = Math.max(highlight[mns[m]], 0.75);
      }
    },

    pick(clientX, clientY) {
      if (disposed) return -1;
      const rect = renderer.domElement.getBoundingClientRect();
      ndc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
      const mix = shared.uMix.value;
      if (mix !== pickMix) {
        for (let i = 0; i < 3 * N; i++) pickPos[i] = pos[i] + (layered[i] - pos[i]) * mix;
        pickGeo.attributes.position.needsUpdate = true;
        pickGeo.computeBoundingSphere();
        pickMix = mix;
      }
      raycaster.setFromCamera(ndc, camera);
      // threshold: ~6 CSS px at the target distance (DPR-independent because it is measured in CSS px)
      const dist = camera.position.distanceTo(controls.target);
      raycaster.params.Points.threshold = (6 * 2 * dist * tanHalf) / rect.height;
      hits.length = 0;
      raycaster.intersectObject(pickPoints, false, hits);
      let best = -1, bestD = Infinity;
      for (const h of hits) {
        const score = h.distanceToRay + 0.002 * h.distance; // closest to the ray, lightly biased toward the camera
        if (score < bestD) { bestD = score; best = h.index; }
      }
      return best;
    },

    /** Screen anchors (CSS px, relative to the canvas) just below each layer slab; opacity follows the layers morph. */
    layerAnchors() {
      for (let L = 0; L < LAYER_COUNT; L++) {
        tmpV.set(slab[L].x, -slab[L].rad - 0.06 * S, 0).project(camera);
        const a = anchors[L];
        a.x = ((tmpV.x + 1) / 2) * viewW;
        a.y = ((1 - tmpV.y) / 2) * viewH;
        a.opacity = tmpV.z < 1 ? shared.uMix.value : 0;
      }
      return anchors;
    },

    setQuality(q) { quality = q === "low" ? "low" : "high"; applyQuality(); },
    setBloom(on) { bloomOn = !!on; applyQuality(); },
    getLayout: () => layoutName,
    getQuality: () => quality,
    getBloom: () => bloomOn,
    resize,

    /** dt: wall-clock seconds since the previous render. */
    render(dt) {
      if (disposed) return;
      dt = Math.min(Math.max(dt || 0, 0), 0.1);
      clock += dt;
      shared.uTime.value = clock;

      if (tween.active) {
        tween.t = Math.min(1, tween.t + dt / MORPH_S);
        const e = easeInOutCubic(tween.t);
        shared.uMix.value = tween.from + (tween.to - tween.from) * e;
        if (tween.camera) {
          tmpV.lerpVectors(tween.tg0, tween.tg1, e);
          placeCamera(tween.az0 + (tween.az1 - tween.az0) * e, tween.el0 + (tween.el1 - tween.el0) * e, tween.d0 + (tween.d1 - tween.d0) * e, tmpV);
        }
        if (tween.t >= 1) {
          tween.active = false;
          lastInteraction = Math.max(lastInteraction, clock - IDLE_RESUME_S + 1.5); // settle before drifting again
        }
      }
      const fade = 1 - shared.uMix.value;
      dustMat.uniforms.uFade.value = fade * fade;
      shellMat.uniforms.uFade.value = fade * fade;
      dust.visible = fade > 0.001;
      for (const m of shellMeshes) m.visible = fade > 0.001;

      // idle orbit: a slow turntable sweep (±40°) that eases in 4 s after the last interaction
      if (autoOrbit && !interacting && !tween.active && clock - lastInteraction > IDLE_RESUME_S) {
        idleT += dt;
        const ramp = Math.min(1, idleT / 3);
        const omega = (2 * Math.PI) / 48;
        const dAz = 0.7 * omega * Math.cos(omega * idleT) * dt * ramp * ramp * (3 - 2 * ramp);
        tmpV.copy(camera.position).sub(controls.target);
        const c = Math.cos(dAz), s = Math.sin(dAz);
        camera.position.set(controls.target.x + tmpV.x * c + tmpV.z * s, camera.position.y, controls.target.z - tmpV.x * s + tmpV.z * c);
      } else if (interacting || tween.active) {
        idleT = 0;
      }
      if (!(tween.active && tween.camera)) controls.update();
      neuronMat.uniforms.uCamDist.value = camera.position.distanceTo(controls.target);

      // activity smoothing → texture
      const kUp = 1 - Math.exp(-dt / TAU_UP), kDown = 1 - Math.exp(-dt / TAU_DOWN), kHi = Math.exp(-dt / TAU_HI);
      for (let i = 0; i < N; i++) {
        const d = display[i], t = target[i];
        const nd = d + (t - d) * (t > d ? kUp : kDown);
        display[i] = nd;
        const h = highlight[i];
        if (h > 0) highlight[i] = h > 0.002 ? h * kHi : 0;
        actData[4 * i] = nd;
        actData[4 * i + 1] = highlight[i];
      }
      actTex.needsUpdate = true;

      composer.render(dt);
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      ro?.disconnect();
      controls.removeEventListener("start", onStart);
      controls.removeEventListener("end", onEnd);
      controls.dispose();
      for (const obj of [...shellMeshes, dust, edges, neurons, gfRings, flashes]) obj.geometry.dispose();
      pickGeo.dispose();
      for (const m of [neuronMat, edgeMat, shellMat, dustMat, gfMat, flashMat]) m.dispose();
      actTex.dispose();
      renderPass.dispose?.();
      bloomPass.dispose();
      outputPass.dispose();
      composer.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    },

    // dev tooling only
    three: { renderer, scene, camera, controls, composer, bloomPass, materials: { neuronMat, edgeMat, shellMat, dustMat, gfMat, flashMat } },
  };
  return api;
}
