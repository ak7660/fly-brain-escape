// HUD (game-hud track): title, controls, decision panel, tooltip, key help, keyboard shortcuts.

const fmtInt = new Intl.NumberFormat("en-US");
const CLASS_VAR = { none: "--class-none", dodge_right: "--class-dodge-right", dodge_left: "--class-dodge-left", takeoff: "--class-takeoff" };
const SIGN = { 1: ["Excitatory", "--glow-excitatory"], "-1": ["Inhibitory", "--glow-inhibitory"], 0: ["Modulatory or unknown", "--glow-modulatory"] };
const LAYER = ["Input", "Hidden · 1 hop", "Hidden · 2+ hops", "Descending", "Motor"];

export const prettyClass = (name) => {
  const s = String(name ?? "none").replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
};

const ICONS = {
  pause: `<svg viewBox="0 0 16 16" aria-hidden="true"><path class="i-pause" d="M5.5 3.5v9M10.5 3.5v9"/><path class="i-play" d="M5 3.2v9.6L12.6 8z"/></svg>`,
  slowmo: `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5h8M4 13.5h8M5 2.5c0 3 6 3.5 6 5.5s-6 2.5-6 5.5M11 2.5c0 3-6 3.5-6 5.5"/></svg>`,
  noise: `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M1.5 8h2l1.5-4 2 8 2-9 2 8 1.5-3h2"/></svg>`,
  bloom: `<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="2.6"/><path d="M8 1.5v1.8M8 12.7v1.8M1.5 8h1.8M12.7 8h1.8M3.4 3.4l1.3 1.3M11.3 11.3l1.3 1.3M3.4 12.6l1.3-1.3M11.3 4.7l1.3-1.3"/></svg>`,
  reset: `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8a5 5 0 1 0 1.5-3.6"/><path d="M3 2.2v3h3"/></svg>`,
  help: `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6 6.2a2 2 0 1 1 2.8 1.8c-.5.3-.8.7-.8 1.3v.5"/><circle class="i-dot" cx="8" cy="12" r=".6"/></svg>`,
};

const KEYS = [
  ["←  A", "Threat from the left"],
  ["↑  W", "Threat from the front"],
  ["→  D", "Threat from the right"],
  ["Shift", "Fast loom (with a direction)"],
  ["Click", "Threat from that direction"],
  ["L", "Anatomy or layers"],
  ["Space", "Pause"],
  ["S", "Slow motion"],
  ["N", "Input noise"],
  ["B", "Bloom"],
  ["R", "Reset"],
  ["?", "Show or hide this help"],
];

const fmtAz = (az) => `${az > 0.5 ? "+" : az < -0.5 ? "−" : ""}${Math.abs(Math.round(az))}°`;
const threatSide = (th) => (th.cls === 3 || Math.abs(th.azimuthDeg) <= 45 ? "front" : th.azimuthDeg < 0 ? "left" : "right");

export function createHud({ title, controls, decisionPanel, tooltip }, { classes, counts, decision = { p: 0.7 } }) {
  const cbs = [];
  const emit = (name, arg) => { for (const cb of cbs) cb(name, arg); };
  const state = { layout: "anatomy", pause: false, slowmo: false, noise: false, bloom: true };
  const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)");
  const ac = new AbortController();
  const { signal } = ac;

  // ------------------------------------------------------------ title
  title.classList.add("hud-titleblock");
  title.innerHTML = `
    <h1 class="hud-title">Drosophila · Male CNS · Looming escape</h1>
    <p class="hud-counts">
      <span><b>${fmtInt.format(counts.neurons)}</b> neurons</span>
      <span><b>${fmtInt.format(counts.edges)}</b> synapses</span>
      <span><b data-fps>–</b> fps</span>
    </p>
    <p class="hud-active"><i class="hud-live" aria-hidden="true"></i><b data-active>0</b> neurons active</p>`;
  const fpsEl = title.querySelector("[data-fps]");
  const activeEl = title.querySelector("[data-active]");

  // ------------------------------------------------------------ controls
  controls.classList.add("hud-controls");
  controls.setAttribute("aria-label", "View and simulation controls");
  const btn = (name, label, key, pressed) => `
    <button type="button" class="icon-btn" data-control="${name}" aria-label="${label}" aria-keyshortcuts="${key}"
      ${pressed === undefined ? "" : `aria-pressed="${pressed}"`} data-tip="${label}" data-key="${key}">${ICONS[name]}</button>`;
  controls.innerHTML = `
    <div class="seg" role="radiogroup" aria-label="Layout (L)">
      <span class="seg-thumb" aria-hidden="true"></span>
      <button type="button" role="radio" data-layout="anatomy" aria-checked="true">Anatomy</button>
      <button type="button" role="radio" data-layout="layers" aria-checked="false" tabindex="-1">Layers</button>
    </div>
    <div class="icon-row">
      ${btn("pause", "Pause", "Space", false)}
      ${btn("slowmo", "Slow motion", "S", false)}
      ${btn("noise", "Input noise", "N", false)}
      ${btn("bloom", "Bloom", "B", true)}
      <span class="icon-sep" aria-hidden="true"></span>
      ${btn("reset", "Reset", "R")}
      ${btn("help", "Keyboard shortcuts", "?", false)}
    </div>`;
  const seg = controls.querySelector(".seg");
  const segBtns = [...seg.querySelectorAll("[data-layout]")];

  function setLayout(name, fromUser = true) {
    state.layout = name;
    seg.dataset.value = name;
    for (const b of segBtns) {
      const on = b.dataset.layout === name;
      b.setAttribute("aria-checked", String(on));
      b.tabIndex = on ? 0 : -1;
    }
    if (fromUser) emit("layout", name);
  }
  segBtns.forEach((b) => b.addEventListener("click", () => setLayout(b.dataset.layout), { signal }));
  seg.addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) return;
    // arrows inside the radio group move the selection; they must not reach the global spawn handler
    e.preventDefault();
    e.stopPropagation();
    if (e.repeat) return;
    const next = state.layout === "anatomy" ? "layers" : "anatomy";
    setLayout(next);
    seg.querySelector(`[data-layout="${next}"]`).focus();
  }, { signal });
  setLayout("anatomy", false);

  const ctl = (name) => controls.querySelector(`[data-control="${name}"]`);
  function setToggle(name, on) {
    state[name] = on;
    const b = ctl(name);
    b?.setAttribute("aria-pressed", String(on));
    if (name === "pause" && b) {
      const label = on ? "Resume" : "Pause";
      b.setAttribute("aria-label", label);
      b.dataset.tip = label;
    }
  }
  function flash(el) {
    if (!el || reduced?.matches) return;
    el.classList.remove("is-flash");
    void el.offsetWidth;
    el.classList.add("is-flash");
  }
  function trigger(name, viaKey = false) {
    if (viaKey) flash(name === "layout" ? seg : ctl(name));
    if (name === "layout") return setLayout(state.layout === "anatomy" ? "layers" : "anatomy");
    if (name === "help") return toggleHelp();
    if (name === "reset") return emit("reset");
    setToggle(name, !state[name]);
    emit(name, state[name]);
  }
  controls.querySelectorAll("[data-control]").forEach((b) => b.addEventListener("click", () => trigger(b.dataset.control), { signal }));

  // ------------------------------------------------------------ decision panel
  decisionPanel.classList.add("hud-panel", "hud-decision");
  decisionPanel.setAttribute("aria-label", "Escape decision");
  decisionPanel.innerHTML = `
    <div class="dp-head">
      <h2>Decision</h2>
      <span class="dp-legend"><i class="thr-mark" aria-hidden="true"></i>fires at p &gt; ${decision.p.toFixed(2)}</span>
    </div>
    <ul class="dp-bars" role="list">
      ${classes.map((c, i) => `
        <li class="dp-row" data-i="${i}" style="--c: var(${CLASS_VAR[c] ?? "--ink-3"})">
          <span class="dp-label"><i class="dp-sw" aria-hidden="true"></i>${prettyClass(c)}</span>
          <span class="dp-track" role="meter" aria-label="${prettyClass(c)} probability" aria-valuemin="0" aria-valuemax="1" aria-valuenow="0">
            <span class="dp-fill"></span><span class="dp-thr" style="left:${decision.p * 100}%"></span>
          </span>
          <span class="dp-val">0.00</span>
        </li>`).join("")}
    </ul>
    <div class="dp-last">
      <div class="dp-status"><i class="dp-dot" aria-hidden="true"></i><span class="dp-status-text">No threat yet</span></div>
      <div class="dp-verdict is-empty"><span class="dp-mark" aria-hidden="true"></span><span class="dp-name">Waiting for a decision</span></div>
      <div class="dp-latency">Launch a threat with ← ↑ →</div>
    </div>
    <dl class="dp-score">
      <div><dt>Score</dt><dd><b data-s="correct">0</b><span class="of">/</span><span data-s="total">0</span></dd></div>
      <div><dt>Hit rate</dt><dd data-s="acc">–</dd></div>
      <div><dt>Wrong</dt><dd data-s="wrong">0</dd></div>
      <div><dt>Missed</dt><dd data-s="missed">0</dd></div>
      <div><dt>False alarms</dt><dd data-s="fa">0</dd></div>
    </dl>`;
  const rows = [...decisionPanel.querySelectorAll(".dp-row")];
  const fills = rows.map((r) => r.querySelector(".dp-fill"));
  const vals = rows.map((r) => r.querySelector(".dp-val"));
  const tracks = rows.map((r) => r.querySelector(".dp-track"));
  const lastBox = decisionPanel.querySelector(".dp-last");
  const statusText = decisionPanel.querySelector(".dp-status-text");
  const verdict = decisionPanel.querySelector(".dp-verdict");
  const verdictName = decisionPanel.querySelector(".dp-name");
  const verdictMark = decisionPanel.querySelector(".dp-mark");
  const latency = decisionPanel.querySelector(".dp-latency");
  const scoreEls = Object.fromEntries([...decisionPanel.querySelectorAll("[data-s]")].map((e) => [e.dataset.s, e]));
  let lastWinner = -1;
  const lastText = [];
  const lastScale = [];

  function setProbs(probs) {
    let w = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[w]) w = i;
    for (let i = 0; i < rows.length; i++) {
      const p = Math.max(0, Math.min(1, probs[i] ?? 0));
      const q = Math.round(p * 500) / 500;
      if (lastScale[i] !== q) { fills[i].style.transform = `scaleX(${q})`; lastScale[i] = q; }
      const txt = p.toFixed(2);
      if (lastText[i] !== txt) {
        vals[i].textContent = txt;
        tracks[i].setAttribute("aria-valuenow", txt);
        lastText[i] = txt;
      }
    }
    if (w !== lastWinner) {
      rows.forEach((r, i) => r.classList.toggle("is-winner", i === w));
      lastWinner = w;
    }
  }

  function setScore(s) {
    if (!s) return;
    scoreEls.correct.textContent = s.correct;
    scoreEls.total.textContent = s.total + (s.falseAlarms ?? 0);
    scoreEls.wrong.textContent = s.wrong ?? 0;
    scoreEls.missed.textContent = s.missed ?? 0;
    scoreEls.fa.textContent = s.falseAlarms ?? 0;
    const acc = s.accuracy ?? null;
    scoreEls.acc.textContent = acc == null ? "–" : `${Math.round(acc * 100)}%`;
  }

  const describeThreat = (th) => `Threat · ${threatSide(th)} · ${fmtAz(th.azimuthDeg)}`;

  /** A new threat is on its way: the verdict box shows it and the previous verdict recedes. */
  function showIncoming(threat) {
    statusText.textContent = describeThreat(threat);
    lastBox.classList.add("is-incoming");
    lastBox.style.setProperty("--c", "var(--ink-2)");
  }

  function showDecision({ name, correct, latencyMs, kind, score, expectedName, threat }) {
    const miss = kind === "miss" || !name || name === "none";
    const fa = kind === "false_alarm";
    lastBox.classList.remove("is-incoming");
    verdict.classList.remove("is-empty");
    verdict.classList.toggle("is-correct", !!correct);
    verdict.classList.toggle("is-wrong", !correct);
    lastBox.style.setProperty("--c", `var(${CLASS_VAR[miss ? "none" : name] ?? "--ink-3"})`);
    verdictMark.textContent = correct ? "✓" : "✗";
    statusText.textContent = threat ? describeThreat(threat) : "No threat in view";
    if (miss) {
      verdictName.textContent = "No response";
      latency.textContent = "Threat made contact";
    } else {
      verdictName.textContent = prettyClass(name);
      if (fa) latency.textContent = "False alarm";
      else {
        const tail = correct || !expectedName ? "" : ` · needed ${prettyClass(expectedName).toLowerCase()}`;
        latency.innerHTML = `<b>${Math.round(latencyMs ?? 0)} ms</b> before contact${tail}`;
      }
      flash(rows[classes.indexOf(name)]);
    }
    flash(lastBox);
    setScore(score);
  }

  function resetDecision() {
    lastBox.classList.remove("is-incoming");
    lastBox.style.removeProperty("--c");
    verdict.className = "dp-verdict is-empty";
    verdictMark.textContent = "";
    statusText.textContent = "No threat yet";
    verdictName.textContent = "Waiting for a decision";
    latency.textContent = "Launch a threat with ← ↑ →";
    setScore({ correct: 0, wrong: 0, total: 0, missed: 0, falseAlarms: 0, accuracy: null });
    setProbs(classes.map((_, i) => (i === 0 ? 1 : 0)));
  }

  let lastFps = "", lastActive = "";
  function setStats({ fps, activeNeurons }) {
    if (fps != null) {
      const f = String(Math.round(fps));
      if (f !== lastFps) { fpsEl.textContent = f; lastFps = f; }
    }
    if (activeNeurons != null) {
      const a = fmtInt.format(activeNeurons);
      if (a !== lastActive) { activeEl.textContent = a; lastActive = a; }
    }
  }

  // ------------------------------------------------------------ tooltip
  tooltip.classList.add("hud-tooltip");
  let tipIndex = null;
  let tipMeter = null, tipValue = null;
  function showTooltip(x, y, info) {
    if (!info) {
      tooltip.hidden = true;
      tipIndex = null;
      return;
    }
    if (info.index !== tipIndex || tipMeter === null) {
      const sign = SIGN[String(Math.sign(info.sign ?? 0))];
      const layer = LAYER[info.layer] ?? null;
      const sub = [info.superclass ? prettyClass(info.superclass) : null, layer].filter(Boolean).map(esc).join(" · ");
      tooltip.innerHTML = `
        <div class="tt-type">${esc(info.type ?? "Unknown type")}</div>
        ${info.instance && info.instance !== info.type ? `<div class="tt-inst">${esc(info.instance)}</div>` : ""}
        ${sub ? `<div class="tt-sub">${sub}</div>` : ""}
        <div class="tt-grid">
          <span>Sign</span><span class="tt-sign" style="--c: var(${sign[1]})"><i></i>${sign[0]}</span>
          <span>Activity</span><span class="tt-act"><span class="tt-meter"><i></i></span><b></b></span>
        </div>`;
      tipMeter = tooltip.querySelector(".tt-meter i");
      tipValue = tooltip.querySelector(".tt-act b");
      tipIndex = info.index;
    }
    const act = Number(info.activity ?? 0);
    tipMeter.style.transform = `scaleX(${Math.min(1, Math.max(0, act / Number(info.rMax || 4)))})`;
    tipValue.textContent = act.toFixed(2);
    tooltip.hidden = false;
    const pad = 14, w = tooltip.offsetWidth, h = tooltip.offsetHeight;
    let left = x + pad, top = y + pad;
    if (left + w > window.innerWidth - 8) left = x - pad - w;
    if (top + h > window.innerHeight - 8) top = y - pad - h;
    tooltip.style.left = `${Math.max(8, left)}px`;
    tooltip.style.top = `${Math.max(8, top)}px`;
  }

  // ------------------------------------------------------------ help overlay
  const help = document.createElement("div");
  help.className = "hud-help";
  help.hidden = true;
  help.innerHTML = `
    <div class="help-card hud-panel" role="dialog" aria-modal="true" aria-labelledby="help-title">
      <div class="help-head">
        <h2 id="help-title">Keyboard shortcuts</h2>
        <button type="button" class="icon-btn help-close" aria-label="Close shortcuts"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8"/></svg></button>
      </div>
      <p class="help-lede">Launch a looming threat and watch the connectome decide: dodge away from the side it comes from, or take off when it comes from the front.</p>
      <dl class="help-keys">${KEYS.map(([k, d]) => `<div><dt>${k.split("  ").map((x) => `<kbd>${x}</kbd>`).join("")}</dt><dd>${d}</dd></div>`).join("")}</dl>
    </div>`;
  document.body.append(help);
  let helpReturn = null;
  let inerted = [];
  function toggleHelp(force) {
    const open = force ?? help.hidden;
    if (open === !help.hidden) return;
    help.hidden = !open;
    ctl("help")?.setAttribute("aria-pressed", String(open));
    if (open) {
      helpReturn = document.activeElement;
      inerted = [...document.body.children].filter((el) => el !== help && !el.inert && el.tagName !== "SCRIPT");
      for (const el of inerted) el.inert = true;
      help.querySelector(".help-close").focus();
    } else {
      for (const el of inerted) el.inert = false;
      inerted = [];
      if (helpReturn?.focus) helpReturn.focus();
    }
  }
  help.querySelector(".help-close").addEventListener("click", () => toggleHelp(false), { signal });
  help.addEventListener("click", (e) => { if (e.target === help) toggleHelp(false); }, { signal });
  help.addEventListener("keydown", (e) => {
    if (e.key === "Tab") { e.preventDefault(); help.querySelector(".help-close").focus(); }
  }, { signal });

  // ------------------------------------------------------------ keyboard
  const SPAWN_AZ = { ArrowLeft: -90, a: -90, A: -90, ArrowUp: 0, w: 0, W: 0, ArrowRight: 90, d: 90, D: 90 };
  const TOGGLES = { l: "layout", s: "slowmo", n: "noise", b: "bloom", r: "reset" };
  window.addEventListener("keydown", (e) => {
    if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
    const t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    const key = e.key;
    if (!help.hidden) {
      if ((key === "Escape" || key === "?") && !e.repeat) { e.preventDefault(); toggleHelp(false); }
      return;
    }
    const az = SPAWN_AZ[key];
    if (az !== undefined) {
      e.preventDefault();
      if (!e.repeat) emit("spawn", { azimuthDeg: az, lv: e.shiftKey ? 0.02 : 0.04 });
      return;
    }
    if (key === "?") { e.preventDefault(); if (!e.repeat) trigger("help", true); return; }
    if (key === " " || key === "Spacebar") {
      if (t && t.tagName === "BUTTON") return; // native button activation
      e.preventDefault();
      if (!e.repeat) trigger("pause", true);
      return;
    }
    const name = TOGGLES[key.toLowerCase()];
    if (name && !e.shiftKey) { e.preventDefault(); if (!e.repeat) trigger(name, true); }
  }, { signal });

  return {
    setProbs,
    showDecision,
    showIncoming,
    setScore,
    resetDecision,
    setStats,
    showTooltip,
    onControl(cb) { cbs.push(cb); },
    setControlState(name, value) {
      if (name === "layout") setLayout(value, false);
      else if (name in state) setToggle(name, !!value);
    },
    toggleHelp,
    dispose() {
      if (!help.hidden) toggleHelp(false);
      ac.abort();
      help.remove();
      cbs.length = 0;
      for (const el of [title, controls, decisionPanel, tooltip]) el.innerHTML = "";
      tooltip.hidden = true;
    },
  };
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
