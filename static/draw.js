(() => {
  // Tells the inline watchdog in index.html that the app scripts arrived.
  // Set first thing, before anything that could throw.
  window.__vertsixLoaded = true;

  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const off = document.createElement("canvas");
  const offCtx = off.getContext("2d");
  const stage = document.getElementById("stage");

  const nSlider = document.getElementById("n");
  const nOut = document.getElementById("n-out");
  const symmetricCheck = document.getElementById("symmetric-check");
  const a1Slider = document.getElementById("a1-weight");
  const a1Out = document.getElementById("a1-weight-out");
  const a2Slider = document.getElementById("a2-weight");
  const a2Out = document.getElementById("a2-weight-out");
  const b1Slider = document.getElementById("b1-weight");
  const b1Out = document.getElementById("b1-weight-out");
  const b2Slider = document.getElementById("b2-weight");
  const b2Out = document.getElementById("b2-weight-out");
  const c1Slider = document.getElementById("c1-weight");
  const c1Out = document.getElementById("c1-weight-out");
  const c2Slider = document.getElementById("c2-weight");
  const c2Out = document.getElementById("c2-weight-out");
  const deltaDisplay = document.getElementById("delta-display");
  const exactGateNote = document.getElementById("exact-gate-note");
  const orderedRegimeNote = document.getElementById("ordered-regime-note");
  const speedSlider = document.getElementById("speed");
  const speedOut = document.getElementById("speed-out");
  const btnInit = document.getElementById("btn-init");
  const btnPlay = document.getElementById("btn-play");
  const btnStep = document.getElementById("btn-step");
  const btnSave = document.getElementById("btn-save");
  const btnSaveSvg = document.getElementById("btn-save-svg");
  const btnExact = document.getElementById("btn-exact");
  const viewMode = document.getElementById("viewmode");
  const sweepCount = document.getElementById("sweep-count");
  const exactInfo = document.getElementById("exact-info");
  const hudStatus = document.getElementById("hud-status");
  const hudDevice = document.getElementById("hud-device");
  const hudZoom = document.getElementById("hud-zoom");

  let errorBanner = null;
  function dlog(msg) {
    console.error(msg);
    if (!errorBanner) {
      errorBanner = document.createElement("div");
      errorBanner.style.cssText = "position:fixed;bottom:0;left:0;right:0;background:#3a1010;color:#ffb3b3;font-family:monospace;font-size:0.75rem;padding:0.6em 1em;z-index:2000;max-height:30vh;overflow-y:auto;border-top:2px solid #ff5555;";
      document.body.appendChild(errorBanner);
    }
    const line = document.createElement("div");
    line.textContent = msg;
    errorBanner.appendChild(line);
  }
  function dinfo(msg) {
    console.log(msg);
  }
  window.addEventListener("error", (e) => {
    dlog(`UNCAUGHT ERROR: ${e.message} (${e.filename}:${e.lineno})`);
  });

  // Bumped whenever n or the weights change. An exact-sampling request
  // captures the current value and its result is discarded if the value has
  // moved on -- otherwise a job started at one n lands after the user has
  // already changed n, silently overwriting the current chain. Observed:
  // start exact at n=13, drag to n=60, and the finished job reinstated a
  // 13x13 chain while the panel read 60 and the exported SVG came out 13x13.
  let paramGen = 0;
  function bumpParams() { paramGen++; }

  // Ownership of the Exact Sample button, kept separate from the general
  // `busy` flag. localInit() sets and then CLEARS `busy` as part of a normal
  // rebuild, so when the debounced n-slider rebuild fired during an exact
  // job it cleared the job's own busy flag and re-enabled the button --
  // letting a second job launch on top of the first (observed: 2 requests to
  // /api/exact/start where there should be 1, both running on a single
  // worker).
  let exactInFlight = false;

  // Shadow chain for a live equilibration check.
  //
  // A static "mixes slowly here" note is easy to read as boilerplate. This
  // runs a SECOND chain from the opposite extremal start with the same
  // weights and compares an observable. If two chains started from opposite
  // corners of the state space disagree, the run has provably not mixed and
  // what is on screen is not a sample from the measure.
  //
  // Measured at Delta=-3, 20000 sweeps: at n=40 the two starts agree to
  // 0.0004 (seed spread 0.009); at n=80 they differ by 0.131, eight times the
  // spread. The breakdown sits between those sizes, which is why a fixed
  // size threshold would be the wrong warning.
  let shadow = null;

  // Height-function fluctuations: two independent exact samples, differenced
  // and divided by sqrt(2). That normalisation preserves the variance of a
  // single height function, so the field shown has the same scale as the
  // fluctuation of one sample about its mean. Rendered on a diverging scale
  // (red positive, blue negative, white near zero).
  let fluctFrame = null;

  function setFluctAvailable(on) {
    const opt = document.getElementById("opt-fluct");
    if (!opt) return;
    opt.disabled = !on;
    opt.textContent = on ? "height fluctuations (two samples)"
                         : "height fluctuations (needs two samples)";
  }

  // Shared by both poll loops. A long job is ~1000 status requests; an
  // isolated bad response must not abandon a computation the server is still
  // running. A proxy timeout returns HTML, so .json() throws rather than
  // returning {ok:false}.
  const MAX_CONSECUTIVE_FAILURES = 15;

  let sampler = null;
  let lastFrame = null;
  let playing = false;
  let totalSweeps = 0;
  let busy = false;

  const DEFAULT_VERTEX6_COLORS = {
    a1: [232, 92, 92],
    a2: [232, 160, 92],
    b1: [232, 220, 92],
    b2: [180, 92, 232],
    c1: [127, 216, 232],
    c2: [92, 140, 232],
  };
  let VERTEX6_COLORS = { ...DEFAULT_VERTEX6_COLORS };

  function hexToRgb(hex) {
    const v = parseInt(hex.slice(1), 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  }
  function rgbToHex([r, g, b]) {
    return "#" + [r, g, b].map(x => x.toString(16).padStart(2, "0")).join("");
  }

  function currentSeed() {
    const el = document.getElementById("seed");
    if (!el) return null;
    const v = el.value.trim();
    if (v === "") return null;
    const k = parseInt(v, 10);
    return Number.isFinite(k) ? k : null;
  }

  function currentWeights() {
    return {
      a1: parseFloat(a1Slider.value), a2: parseFloat(a2Slider.value),
      b1: parseFloat(b1Slider.value), b2: parseFloat(b2Slider.value),
      c1: parseFloat(c1Slider.value), c2: parseFloat(c2Slider.value),
    };
  }

  // Default only; the server sends the authoritative value in /api/init
  // (field `max_exact_n`). Keeping a hardcoded copy in sync by hand is how
  // client and server gating drift apart.
  let MAX_EXACT_N = 14;

  function isExactSafe(w, n) {
    // Small n: the sequential transfer-matrix sampler handles any weights.
    // Large n: CFTP, which is monotone exactly on b1b2 >= a1a2 and
    // b1b2 >= c1c2 -- a region, not just the uniform point. Outside it no
    // monotone coupling of the update exists at all, so there is nothing to
    // fall back on.
    if (n <= MAX_EXACT_N) return true;
    const A = w.a1 * w.a2, B = w.b1 * w.b2, C = w.c1 * w.c2;
    return C >= A - 1e-12 && C >= B - 1e-12;
  }

  function pairedSlider(sliderA, outA, sliderB, outB) {
    sliderA.addEventListener("input", () => {
      bumpParams();
      outA.textContent = parseFloat(sliderA.value).toFixed(2);
      if (symmetricCheck.checked) {
        sliderB.value = sliderA.value;
        bumpParams();
      outB.textContent = parseFloat(sliderB.value).toFixed(2);
      }
      updateDeltaDisplay();
    });
    sliderB.addEventListener("input", () => {
      outB.textContent = parseFloat(sliderB.value).toFixed(2);
      if (symmetricCheck.checked) {
        sliderA.value = sliderB.value;
        outA.textContent = parseFloat(sliderA.value).toFixed(2);
      }
      updateDeltaDisplay();
    });
  }

  function updateDeltaDisplay() {
    const w = currentWeights();

    // Apply weight changes to a LIVE chain.
    //
    // `SixVertexJS` captured its weights at construction and nothing ever
    // updated them, so moving a slider mid-run changed only the Delta
    // readout: the UI could show "-3.03 antiferroelectric" while the chain
    // kept sampling at Delta=0.5 until the user happened to press Reset.
    // A picture that does not match its own stated parameters is exactly the
    // failure this tool has been bitten by before.
    //
    // Retargeting a Markov chain mid-run is legitimate -- it simply needs
    // time to re-equilibrate, which is what the user watching the slider
    // expects to see.
    if (sampler) sampler.w = w;

    const a1a2 = w.a1 * w.a2, b1b2 = w.b1 * w.b2, c1c2 = w.c1 * w.c2;
    const delta = (a1a2 + b1b2 - c1c2) / (2 * Math.sqrt(a1a2 * b1b2));
    let regime;
    if (delta > 1) regime = "ferroelectric";
    else if (delta < -1) regime = "antiferroelectric";
    else regime = "disordered";
    if (w.a1 === 1 && w.a2 === 1 && w.b1 === 1 && w.b2 === 1 && w.c1 === 1 && w.c2 === 1) regime += " (uniform weights)";
    deltaDisplay.textContent = `${delta.toFixed(2)} \u00b7 ${regime}`;

    orderedRegimeNote.style.display = Math.abs(delta) > 1 ? "block" : "none";

    const n = parseInt(nSlider.value, 10);
    const safe = isExactSafe(w, n);
    btnExact.disabled = !safe || exactInFlight;

    // Warn when CFTP is valid but likely to be impractically slow.
    //
    // Being inside the monotone region guarantees correctness, not speed.
    // Coalescence time grows exponentially in n deep in the ferroelectric
    // phase: measured at Delta=5, the two extremal chains took 78, 1956 and
    // 32506 sweeps to meet at n = 6, 8, 10 -- roughly 20x per step of two,
    // where n^2 growth would be about 1.6x. The onset is near Delta = 2
    // (at n=12: ~100 sweeps below Delta 1.5, 293 at Delta 2, 4564 at
    // Delta 3), so warn from there rather than at the phase boundary.
    const A = w.a1 * w.a2, B = w.b1 * w.b2, C = w.c1 * w.c2;
    const dlt = (A + B - C) / (2 * Math.sqrt(A * B));
    const slowNote = document.getElementById("slow-exact-note");
    if (slowNote) {
      const risky = safe && n > MAX_EXACT_N && dlt > 2;
      slowNote.style.display = risky ? "block" : "none";
      if (risky) {
        slowNote.textContent =
          `\u0394 = ${dlt.toFixed(2)} is deep in the ordered phase. Exact ` +
          `sampling is still valid here, but the time for the two chains to ` +
          `meet grows exponentially with n, so this may not finish. Reduce n ` +
          `or \u0394 if it stalls.`;
      }
    }
    btnExact.style.opacity = safe ? "1" : "0.4";
    btnExact.style.cursor = safe ? "pointer" : "not-allowed";
    exactGateNote.style.display = safe ? "none" : "block";
    if (!safe) {
      exactGateNote.innerHTML =
        `Exact sampling for non-uniform weights uses an exact sequential ` +
        `(transfer-matrix) method whose cost grows exponentially with n, so ` +
        `it is limited to n &le; ${MAX_EXACT_N}. Beyond that, the only exact ` +
        `method available (CFTP) requires c1c2 &ge; a1a2 and c1c2 &ge; b1b2 ` +
        `(all weights = 1). Reduce n to ${MAX_EXACT_N} or below, or set all ` +
        `weights to 1.00.`;
    }
  }

  // Changing n must actually resize the lattice.
  //
  // The chain's size is fixed when the sampler is built, and unlike the
  // weights it cannot be retargeted in place -- a different n is a different
  // state space. Previously the slider only relabelled the UI: with the
  // slider dragged to 200 while a chain built at n=20 kept running, the
  // exported SVG came out 20x20 while the panel read "200". An exported
  // artifact that contradicts its own stated parameters is the worst version
  // of this bug, so rebuild instead.
  //
  // Debounced: dragging the slider fires a stream of input events and each
  // rebuild allocates a fresh lattice.
  let nRebuildTimer = null;
  nSlider.addEventListener("input", () => {
    nOut.textContent = nSlider.value;
    bumpParams();
    updateDeltaDisplay();
    clearTimeout(nRebuildTimer);
    nRebuildTimer = setTimeout(() => {
      // localInit() leaves `playing` alone, so a running chain keeps running
      // at the new size rather than silently stopping.
      localInit();
    }, 200);
  });
  pairedSlider(a1Slider, a1Out, a2Slider, a2Out);
  pairedSlider(b1Slider, b1Out, b2Slider, b2Out);
  pairedSlider(c1Slider, c1Out, c2Slider, c2Out);
  symmetricCheck.addEventListener("change", () => {
    if (symmetricCheck.checked) {
      a2Slider.value = a1Slider.value; a2Out.textContent = parseFloat(a2Slider.value).toFixed(2);
      b2Slider.value = b1Slider.value; b2Out.textContent = parseFloat(b2Slider.value).toFixed(2);
      c2Slider.value = c1Slider.value; c2Out.textContent = parseFloat(c2Slider.value).toFixed(2);
      updateDeltaDisplay();
    }
  });
  speedSlider.addEventListener("input", () => (speedOut.textContent = speedSlider.value));
  updateDeltaDisplay();

  let dpr = 1;
  function sizeStage() {
    const mainEl = stage.parentElement;

    // Clear any inline size we set on a previous call BEFORE measuring.
    // Otherwise we measure the width we ourselves pinned last time, so the
    // stage can never grow when the window is enlarged -- it stays frozen at
    // whatever size it had on first load. (Regression: resizing 1200 -> 2560
    // left the canvas stuck at 905px.)
    stage.style.width = "";
    stage.style.height = "";

    const rect = stage.getBoundingClientRect();
    const availWidth = rect.width > 10 ? rect.width : mainEl.clientWidth;
    const availHeight = mainEl.clientHeight;
    let w = Math.round(Math.max(280, Math.min(availWidth, 2000)));
    let h = Math.round(Math.max(280, Math.min(availHeight, 1400)));

    // Single-column (narrow) layout: cap the stage at square.
    //
    // The lattice is square and is fitted to min(width, height), so on a
    // phone a 360x1142 stage drew a 360x360 picture and left 68% of the
    // canvas empty -- pushing the controls 375px below the fold behind a
    // band of dead black. Nothing is gained by the extra height.
    if (window.matchMedia("(max-width: 55rem) and (min-height: 500px)").matches) {
      // Square, AND bounded by the actual viewport height.
      //
      // Capping to square alone was not enough: in landscape the column is
      // wide but short, so min(h, w) produced a 780x780 stage on a 390px-tall
      // screen -- the canvas was twice the height of the display and the
      // controls landed at y=820. Leave room for them.
      const side = Math.min(w, Math.round(window.innerHeight * 0.7));
      w = Math.max(280, side);
      h = w;
    }

    stage.style.width = w + "px";
    stage.style.height = h + "px";

    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";

    fitCamera();
    draw();
  }

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(sizeStage, 150);
  });

  function colorFor(t) {
    const stops = [
      [5, 8, 20],
      [30, 70, 110],
      [127, 216, 232],
      [232, 200, 150],
      [232, 160, 92],
    ];
    const seg = t * (stops.length - 1);
    const i0 = Math.min(stops.length - 2, Math.floor(seg));
    const f = seg - i0;
    const a = stops[i0], b = stops[i0 + 1];
    return [
      a[0] + (b[0] - a[0]) * f,
      a[1] + (b[1] - a[1]) * f,
      a[2] + (b[2] - a[2]) * f,
    ];
  }

  function classifyFaceLocal(tl, tr, bl, br) {
    const top = tr - tl, bottom = br - bl, left = bl - tl, right = br - tr;
    const t = top === 1, b = bottom === 1, l = left === 1, r = right === 1;
    if (!l && !t && !b && !r) return "a1";
    if (l && t && b && r) return "a2";
    // Standard convention: (l,t) is a c-type, (t,b) is a b-type. This
    // classifier was missed when the labels were corrected elsewhere, so the
    // colour pickers for b1/b2 were in fact colouring c1/c2. Diagnostic: in a
    // large simulation the frozen corners must be a- and b-types.
    if (l && t && !b && !r) return "c1";
    if (!l && !t && b && r) return "c2";
    if (!l && t && b && !r) return "b1";
    if (l && !t && !b && r) return "b2";
    return "a1";
  }

  function frameFromSampler(s) {
    const size = s.size;
    const { min, max } = s.minMax();
    const active = s.activeMask();
    return {
      n: s.n,
      get: (i, j) => s.H[i * size + j],
      getActive: (i, j) => active[i * size + j] === 1,
      min, max,
    };
  }

  function frameFromServerData(data) {
    const heightBytes = Uint8Array.from(atob(data.height_b64), c => c.charCodeAt(0));
    const activeBytes = Uint8Array.from(atob(data.active_b64), c => c.charCodeAt(0));
    const height = new Int16Array(heightBytes.buffer);
    const size = data.n + 1;
    const expected = size * size;

    // Validate the frame against the n the server reported.
    //
    // Without this a short frame decodes happily and every out-of-range read
    // returns undefined, so the height field renders as NaN -- a visibly
    // corrupt picture with no error anywhere. Measured on a deliberately
    // truncated frame: 31 undefined reads, 41 NaN pixels, zero exceptions.
    //
    // This project has already shipped one silent frame-format mismatch (the
    // server moved to binary frames while this function still parsed the
    // plain-JSON ones, and "Exact Sample" quietly did nothing for weeks). Fail
    // loudly instead: the caller's try/catch surfaces it to the user.
    if (height.length !== expected || activeBytes.length !== expected) {
      throw new Error(
        `malformed frame from server: n=${data.n} implies ${expected} values, ` +
        `got ${height.length} heights and ${activeBytes.length} active flags`);
    }
    return {
      n: data.n,
      get: (i, j) => height[i * size + j],
      getActive: (i, j) => activeBytes[i * size + j] === 1,
      min: data.min, max: data.max,
    };
  }

  function buildOffscreen(frame, mode) {
    const n = frame.n;
    off.width = n;
    off.height = n;
    const img = offCtx.createImageData(n, n);

    if (mode === "active") {
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const isActive = frame.getActive(i, j) || frame.getActive(i + 1, j) ||
                            frame.getActive(i, j + 1) || frame.getActive(i + 1, j + 1);
          const idx = (i * n + j) * 4;
          if (isActive) {
            img.data[idx] = 232; img.data[idx + 1] = 160; img.data[idx + 2] = 92;
          } else {
            img.data[idx] = 20; img.data[idx + 1] = 30; img.data[idx + 2] = 42;
          }
          img.data[idx + 3] = 255;
        }
      }
    } else if (mode === "fluct") {
      // Selecting this view before any fluctuation field exists used to
      // dereference null and throw out of draw(). The option is disabled
      // until one exists, but guard anyway: a stale selection can survive a
      // reset, and a view mode should never be able to break rendering.
      if (!fluctFrame) {
        viewMode.value = "height";
        return buildOffscreen(frame, "height");
      }
      const F = fluctFrame;
      let m = 1e-9;
      for (let i = 0; i <= n; i++)
        for (let j = 0; j <= n; j++) m = Math.max(m, Math.abs(F.get(i, j)));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const v = (F.get(i, j) + F.get(i + 1, j) +
                     F.get(i, j + 1) + F.get(i + 1, j + 1)) / 4 / m;
          // white at zero, red for positive, blue for negative
          const t = Math.max(-1, Math.min(1, v));
          let r, g, b;
          if (t >= 0) { r = 255; g = Math.round(255 * (1 - t)); b = g; }
          else        { b = 255; r = Math.round(255 * (1 + t)); g = r; }
          const idx = (i * n + j) * 4;
          img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b;
          img.data[idx + 3] = 255;
        }
      }
    } else if (mode === "paths") {
      // Filled by drawPaths() after the blit; nothing per-pixel here.
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const idx = (i * n + j) * 4;
          img.data[idx] = 12; img.data[idx + 1] = 14; img.data[idx + 2] = 18;
          img.data[idx + 3] = 255;
        }
      }
    } else if (mode === "vertex6") {
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const type = classifyFaceLocal(frame.get(i, j), frame.get(i, j + 1), frame.get(i + 1, j), frame.get(i + 1, j + 1));
          const [r, g, b] = VERTEX6_COLORS[type];
          const idx = (i * n + j) * 4;
          img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b;
          img.data[idx + 3] = 255;
        }
      }
    } else {
      const min = frame.min, max = frame.max;
      const span = Math.max(1, max - min);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const avg = (frame.get(i, j) + frame.get(i + 1, j) + frame.get(i, j + 1) + frame.get(i + 1, j + 1)) / 4;
          const t = (avg - min) / span;
          const [r, g, b] = colorFor(t);
          const idx = (i * n + j) * 4;
          img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b;
          img.data[idx + 3] = 255;
        }
      }
    }
    offCtx.putImageData(img, 0, 0);
    return n;
  }

  let camera = { x: 0, y: 0, scale: 1 };

  function fitCamera() {
    const n = lastFrame ? lastFrame.n : 100;
    const cssWidth = canvas.width / dpr;
    const cssHeight = canvas.height / dpr;
    const scale = Math.min(cssWidth, cssHeight) / n;
    const x = -((cssWidth / scale) - n) / 2;
    const y = -((cssHeight / scale) - n) / 2;
    camera = { x, y, scale };
    applyZoomLabel();
  }

  function applyZoomLabel() {
    if (hudZoom) hudZoom.textContent = `${Math.round((camera.scale / baseScale()) * 100)}%`;
  }

  function baseScale() {
    const n = lastFrame ? lastFrame.n : 100;
    return Math.min(canvas.width, canvas.height) / dpr / n;
  }

  function drawGrid() {
    const s = dpr * camera.scale;
    const cssW = canvas.width / dpr, cssH = canvas.height / dpr;
    const pad = 2;
    const worldLeft = camera.x - pad;
    const worldTop = camera.y - pad;
    const worldRight = camera.x + cssW / camera.scale + pad;
    const worldBottom = camera.y + cssH / camera.scale + pad;

    const targetPx = 60;
    const rawSpacing = targetPx / camera.scale;
    const mag = Math.pow(10, Math.floor(Math.log10(rawSpacing)));
    const candidates = [1, 2, 5, 10];
    let spacing = mag;
    for (const c of candidates) {
      if (mag * c >= rawSpacing) { spacing = mag * c; break; }
      spacing = mag * 10;
    }

    ctx.save();
    ctx.setTransform(s, 0, 0, s, -camera.x * s, -camera.y * s);
    ctx.lineWidth = 1 / camera.scale;
    ctx.strokeStyle = "rgba(127, 216, 232, 0.08)";
    ctx.beginPath();
    const startX = Math.floor(worldLeft / spacing) * spacing;
    for (let x = startX; x <= worldRight; x += spacing) {
      ctx.moveTo(x, worldTop);
      ctx.lineTo(x, worldBottom);
    }
    const startY = Math.floor(worldTop / spacing) * spacing;
    for (let y = startY; y <= worldBottom; y += spacing) {
      ctx.moveTo(worldLeft, y);
      ctx.lineTo(worldRight, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  function drawPaths(frame, scale) {
    // Standard six-vertex path convention (Gorin-Kenyon, arXiv:2408.14446,
    // Figure 1): paths run along lattice edges through each vertex --
    //   (1) a1 empty   (2) a2 crossing   (3) b1 vertical
    //   (4) b2 horizontal   (5) c1 turn   (6) c2 turn
    //
    // The four bits already computed for the face type ARE the occupied
    // edges: l,t,b,r set means the left/top/bottom/right edge carries path.
    // So drawing a stub from the vertex centre to each set edge reproduces
    // the figure exactly -- turns for the c types, straight lines for b,
    // a crossing for a2, nothing for a1.
    //
    // (An earlier version drew level lines of the height function. That is a
    // valid picture of the same configuration but it is the dual one, and it
    // is not the convention used in the literature or described in this
    // interface.)
    const n = frame.n;
    ctx.strokeStyle = "#e8e2d8";
    ctx.lineWidth = Math.max(0.05, Math.min(0.16, 10 / n));
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const tl = frame.get(i, j),     tr = frame.get(i, j + 1);
        const bl = frame.get(i + 1, j), br = frame.get(i + 1, j + 1);
        const t = (tr - tl) === 1, b = (br - bl) === 1;
        const l = (bl - tl) === 1,  r = (br - tr) === 1;
        const cx = j + 0.5, cy = i + 0.5;
        if (l) { ctx.moveTo(cx, cy); ctx.lineTo(j, cy); }
        if (r) { ctx.moveTo(cx, cy); ctx.lineTo(j + 1, cy); }
        if (t) { ctx.moveTo(cx, cy); ctx.lineTo(cx, i); }
        if (b) { ctx.moveTo(cx, cy); ctx.lineTo(cx, i + 1); }
      }
    }
    ctx.stroke();
  }

  function downloadText(name, text) {
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a); URL.revokeObjectURL(url);
  }

  function heightsAsText(frame) {
    // (N+1)x(N+1) integer table, tab separated -- loads with numpy.loadtxt
    const n = frame.n, rows = [];
    for (let i = 0; i <= n; i++) {
      const row = [];
      for (let j = 0; j <= n; j++) row.push(frame.get(i, j));
      rows.push(row.join("\t"));
    }
    return rows.join("\n") + "\n";
  }

  const TYPE_CODE = { a1: 1, a2: 2, b1: 3, b2: 4, c1: 5, c2: 6 };

  function typesAsText(frame) {
    // NxN table of 1..6 = a1,a2,b1,b2,c1,c2
    const n = frame.n, rows = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < n; j++) {
        row.push(TYPE_CODE[classifyFaceLocal(
          frame.get(i, j), frame.get(i, j + 1),
          frame.get(i + 1, j), frame.get(i + 1, j + 1))]);
      }
      rows.push(row.join("\t"));
    }
    return rows.join("\n") + "\n";
  }

  function draw() {
    if (!lastFrame) return;
    try {
      const n = buildOffscreen(lastFrame, viewMode.value);
      ctx.imageSmoothingEnabled = viewMode.value === "height";
      ctx.imageSmoothingQuality = "high";
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = "#050607";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      drawGrid();
      const s = dpr * camera.scale;
      ctx.setTransform(s, 0, 0, s, -camera.x * s, -camera.y * s);
      ctx.drawImage(off, 0, 0, n, n);
      if (viewMode.value === "paths") drawPaths(lastFrame, s);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
    } catch (err) {
      dlog(`DRAW ERROR: ${err.message}`);
    }
  }

  function renderFrame(frame) {
    lastFrame = frame;
    fitCamera();
    draw();
  }

  const vertex6Legend = document.getElementById("vertex6-legend");
  viewMode.addEventListener("change", () => {
    vertex6Legend.style.display = viewMode.value === "vertex6" ? "block" : "none";
    draw();
  });

  for (const type of Object.keys(DEFAULT_VERTEX6_COLORS)) {
    const input = document.getElementById(`color-${type}`);
    if (input) {
      input.addEventListener("input", () => {
        VERTEX6_COLORS[type] = hexToRgb(input.value);
        if (viewMode.value === "vertex6") draw();
      });
    }
  }
  const btnResetColors = document.getElementById("btn-reset-colors");
  if (btnResetColors) {
    btnResetColors.addEventListener("click", () => {
      VERTEX6_COLORS = { ...DEFAULT_VERTEX6_COLORS };
      for (const type of Object.keys(DEFAULT_VERTEX6_COLORS)) {
        const input = document.getElementById(`color-${type}`);
        if (input) input.value = rgbToHex(DEFAULT_VERTEX6_COLORS[type]);
      }
      if (viewMode.value === "vertex6") draw();
    });
  }

  function cssPos(e) {
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  canvas.addEventListener("wheel", (e) => {
    try {
      e.preventDefault();
      const p = cssPos(e);
      const worldX = camera.x + p.x / camera.scale;
      const worldY = camera.y + p.y / camera.scale;
      const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
      const minScale = baseScale() * 0.2;
      const maxScale = baseScale() * 40;
      const newScale = Math.max(minScale, Math.min(maxScale, camera.scale * factor));
      camera.x = worldX - p.x / newScale;
      camera.y = worldY - p.y / newScale;
      camera.scale = newScale;
      applyZoomLabel();
      draw();
    } catch (err) {
      dlog(`WHEEL ERROR: ${err.message}`);
    }
  }, { passive: false });

  let dragPointerId = null;
  let dragAnchorWorld = { x: 0, y: 0 };

  canvas.addEventListener("pointerdown", (e) => {
    try {
      dragPointerId = e.pointerId;
      canvas.setPointerCapture(dragPointerId);
      canvas.classList.add("grabbing");
      const p = cssPos(e);
      dragAnchorWorld = { x: camera.x + p.x / camera.scale, y: camera.y + p.y / camera.scale };
    } catch (err) {
      dlog(`POINTERDOWN ERROR: ${err.message}`);
    }
  });

  canvas.addEventListener("pointermove", (e) => {
    if (dragPointerId === null || e.pointerId !== dragPointerId) return;
    try {
      const p = cssPos(e);
      camera.x = dragAnchorWorld.x - p.x / camera.scale;
      camera.y = dragAnchorWorld.y - p.y / camera.scale;
      draw();
    } catch (err) {
      dlog(`POINTERMOVE ERROR: ${err.message}`);
    }
  });

  function endDrag(e) {
    if (dragPointerId === null || (e && e.pointerId !== dragPointerId)) return;
    try { canvas.releasePointerCapture(dragPointerId); } catch (err) {}
    dragPointerId = null;
    canvas.classList.remove("grabbing");
  }
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("dblclick", () => {
    fitCamera();
    draw();
  });

  const loadStartTime = Date.now();
  // Floor on how long the loading screen shows, so a fast load does not make
  // it flash. It was 3000 ms, which was pure dead time: measured, the app is
  // ready in ~57 ms, so the screen was holding ~3.3 s -- 58x the actual load
  // -- on every single visit. That directly undercuts the work done on
  // sampling speed, and this tool's first review complaint was that it felt
  // slow. A short floor keeps the animation from flickering without making
  // the tool feel sluggish.
  const MIN_LOADING_MS = 600;

  function hideLoadingScreen() {
    const el = document.getElementById("loading-screen");
    if (!el) return;
    const elapsed = Date.now() - loadStartTime;
    const remaining = Math.max(0, MIN_LOADING_MS - elapsed);
    setTimeout(() => {
      el.classList.add("hidden");
      setTimeout(() => el.remove(), 500);
    }, remaining);
  }

  const isTouchOnly = "ontouchstart" in window && !window.matchMedia("(pointer: fine)").matches;
  if (isTouchOnly) {
    nSlider.value = "40"; nOut.textContent = "40";
    speedSlider.value = "3"; speedOut.textContent = "3";
    dinfo("mobile/touch device detected: reduced default n=40, sweeps=3 for performance");
  }

  function localInit() {
    // Any rebuild of the chain invalidates an exact-sampling request that is
    // still in flight. Bump here rather than in each caller: Reset went
    // unguarded when only the sliders bumped, so pressing Reset during a job
    // left the user with a fresh chain that the finished job then silently
    // replaced -- they asked to start over and got an exact sample instead.
    bumpParams();
    // A reset builds a new chain, so any fluctuation field from earlier
    // samples no longer describes what is on screen. Drop it and leave the
    // view, otherwise the canvas stays frozen on stale data.
    fluctFrame = null;
    setFluctAvailable(false);
    if (viewMode.value === "fluct") viewMode.value = "height";
    busy = true;
    hudStatus.textContent = "initializing...";
    const n = parseInt(nSlider.value, 10);
    const w = currentWeights();
    try {
      sampler = new SixVertexJS(n, w, Date.now() & 0xffffffff);
      shadow = makeShadow(n, w);
      totalSweeps = 0;
      sweepCount.textContent = totalSweeps;
      hudDevice.textContent = "in-browser (JS)";
      hudStatus.textContent = "ready";
      renderFrame(frameFromSampler(sampler));
      updateDeltaDisplay();
    } catch (err) {
      dlog(`LOCALINIT ERROR: ${err.message}`);
      hudStatus.textContent = "error (see log)";
    }
    busy = false;
    hideLoadingScreen();
  }

  function makeShadow(n, w) {
    // same weights, opposite extremal start
    const S = new SixVertexJS(n, w, (Date.now() ^ 0x5bf03635) & 0xffffffff);
    const size = n + 1;
    const corners = [[0, 0, 0], [n, 0, n], [0, n, n], [n, n, 0]];
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        let v = 1e9;                       // the MAX height function
        for (const [a, b, hv] of corners) {
          v = Math.min(v, hv + Math.abs(i - a) + Math.abs(j - b));
        }
        S.H[i * size + j] = v;
      }
    }
    return S;
  }

  function cFraction(S) {
    const n = S.n, size = n + 1;
    let cc = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const tl = S.H[i*size+j], tr = S.H[i*size+j+1],
              bl = S.H[(i+1)*size+j], br = S.H[(i+1)*size+j+1];
        const t = (tr-tl) === 1 ? 1 : 0, bo = (br-bl) === 1 ? 1 : 0,
              l  = (bl-tl) === 1 ? 1 : 0, r  = (br-tr) === 1 ? 1 : 0;
        if ((l===0 && t===1 && bo===1 && r===0) ||
            (l===1 && t===0 && bo===0 && r===1)) cc++;
      }
    }
    return cc / (n * n);
  }

  function updateMixingBadge() {
    const el = document.getElementById("mixing-badge");
    if (!el || !sampler || !shadow) return;
    // only meaningful once both chains have had some time
    if (totalSweeps < 200) { el.style.display = "none"; return; }
    const gap = Math.abs(cFraction(sampler) - cFraction(shadow));
    el.style.display = "block";
    if (gap > 0.02) {
      el.textContent =
        `NOT EQUILIBRATED - a second chain started from the opposite corner ` +
        `disagrees by ${gap.toFixed(3)} in c-vertex density. What is on ` +
        `screen is not yet a sample from the measure; it is where this ` +
        `particular run happens to be stuck.`;
      el.style.color = "#ffb3b3";
      el.style.borderLeft = "3px solid #b03a3a";
    } else {
      el.textContent =
        `two chains from opposite starts agree to ${gap.toFixed(3)} - ` +
        `consistent with equilibrium (necessary, not sufficient)`;
      el.style.color = "#8a7c6c";
      el.style.borderLeft = "3px solid #3a5a3a";
    }
  }

  function localStep() {
    if (busy || !sampler) return;
    const sweeps = parseInt(speedSlider.value, 10);
    hudStatus.textContent = playing ? "running" : "ready";
    sampler.step(sweeps);
    totalSweeps += sweeps;
    sweepCount.textContent = totalSweeps;
    if (shadow) shadow.step(sweeps);
    lastFrame = frameFromSampler(sampler);
    draw();
    updateMixingBadge();
  }

  function loop() {
    if (!playing) return;
    localStep();
    requestAnimationFrame(loop);
  }

  const btnSaveHeights = document.getElementById("btn-save-heights");
  if (btnSaveHeights) btnSaveHeights.addEventListener("click", () => {
    if (!lastFrame) return;
    downloadText(`vertsix_heights_n${lastFrame.n}.txt`, heightsAsText(lastFrame));
  });
  const btnSaveTypes = document.getElementById("btn-save-types");
  if (btnSaveTypes) btnSaveTypes.addEventListener("click", () => {
    if (!lastFrame) return;
    downloadText(`vertsix_types_n${lastFrame.n}.txt`, typesAsText(lastFrame));
  });

  btnInit.addEventListener("click", () => {
    playing = false;
    btnPlay.classList.remove("active");
    btnPlay.textContent = "run";
    localInit();
  });

  btnPlay.addEventListener("click", () => {
    playing = !playing;
    btnPlay.classList.toggle("active", playing);
    btnPlay.textContent = playing ? "pause" : "run";
    if (playing) loop();
  });

  btnStep.addEventListener("click", () => {
    if (!playing) localStep();
  });

  async function fetchExactSample(n, w) {
    // One exact sample from the server, with the same poll-failure tolerance
    // as the main handler: a long job is many requests and an isolated bad
    // response must not abandon a computation the server is still running.
    const startRes = await fetch("/api/exact/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        n, a1: w.a1, a2: w.a2, b1: w.b1, b2: w.b2,
        c_up: w.c1, c_down: w.c2, seed: currentSeed() }),
    });
    const startData = await startRes.json();
    if (!startData.ok) throw new Error(startData.error || "failed to start");
    const jobId = startData.job_id;

    let waited = 0, fails = 0;
    while (waited < 45 * 60) {
      const interval = waited < 30 ? 1000 : waited < 120 ? 2000 : 5000;
      await new Promise(r => setTimeout(r, interval));
      waited += interval / 1000;
      let st;
      try {
        st = await (await fetch(`/api/exact/status/${jobId}`)).json();
        fails = 0;
      } catch (e) {
        // a proxy hiccup must not abandon a job the server is still running
        if (++fails >= MAX_CONSECUTIVE_FAILURES)
          throw new Error("lost contact with the server");
        continue;
      }
      if (!st.ok) throw new Error(st.error || "status check failed");
      if (st.status === "error") throw new Error(st.error || "sampling failed");
      if (st.status === "done") return frameFromServerData(st.frame);
    }
    throw new Error("gave up waiting for the server");
  }

  const btnFluct = document.getElementById("btn-fluct");
  if (btnFluct) btnFluct.addEventListener("click", async () => {
    if (exactInFlight) return;
    const w = currentWeights();
    const n = parseInt(nSlider.value, 10);
    if (!isExactSafe(w, n)) return;
    exactInFlight = true;
    playing = false;
    btnPlay.classList.remove("active");
    btnPlay.textContent = "run";
    busy = true;
    btnExact.disabled = true; btnFluct.disabled = true;
    const requestedGen = paramGen;
    hudStatus.textContent = "sampling two copies (server)...";
    exactInfo.textContent = "drawing the first of two independent samples...";
    try {
      const A = await fetchExactSample(n, w);
      exactInfo.textContent = "drawing the second of two independent samples...";
      const B = await fetchExactSample(n, w);
      if (paramGen !== requestedGen) {
        exactInfo.textContent =
          "discarded: the chain was reset or its parameters changed";
        return;
      }
      // (H_A - H_B)/sqrt(2) -- preserves the variance of a single height
      const inv = 1 / Math.SQRT2;
      const size = n + 1;
      const diff = new Float64Array(size * size);
      for (let i = 0; i <= n; i++)
        for (let j = 0; j <= n; j++)
          diff[i * size + j] = (A.get(i, j) - B.get(i, j)) * inv;
      fluctFrame = { n, get: (i, j) => diff[i * size + j] };
      setFluctAvailable(true);
      lastFrame = A;
      sampler = null;
      viewMode.value = "fluct";
      const legend = document.getElementById("vertex6-legend");
      if (legend) legend.style.display = "none";
      fitCamera();
      draw();
      let mx = 0;
      for (let k = 0; k < diff.length; k++) mx = Math.max(mx, Math.abs(diff[k]));
      exactInfo.textContent =
        `height fluctuations: (H1 - H2)/sqrt(2) from two independent exact ` +
        `samples; range +/-${mx.toFixed(2)}`;
      hudStatus.textContent = "fluctuations";
    } catch (err) {
      exactInfo.textContent = `fluctuation sampling failed: ${err.message}`;
      hudStatus.textContent = "ready";
    } finally {
      exactInFlight = false;
      busy = false;
      btnFluct.disabled = false;
      updateDeltaDisplay();
    }
  });

  const modelSel = document.getElementById("model");
  const stochParams = document.getElementById("stoch-params");
  const stochNote = document.getElementById("stochastic-note");
  const sb1 = document.getElementById("sb1"), sb2 = document.getElementById("sb2");

  function updateModelUI() {
    if (!modelSel) return;
    const stoch = modelSel.value === "stochastic";
    if (stochParams) stochParams.style.display = stoch ? "block" : "none";
    if (stochNote) stochNote.style.display = stoch ? "block" : "none";
    // The DWBC controls drive a different model with different boundary
    // conditions. Disable rather than hide them: leaving them clickable would
    // invite a silent mismatch, but hiding them makes the tool look like it
    // lost features. Disabled with a reason is the honest middle.
    for (const id of ["btn-play", "btn-step", "btn-exact", "btn-fluct"]) {
      const el = document.getElementById(id);
      if (!el) continue;
      el.style.display = "";
      if (stoch) {
        el.disabled = true;
        el.title = "This control drives the DWBC model. The stochastic model " +
                   "has free-exit boundary conditions and is sampled in one " +
                   "sweep, so there is no chain to run and no separate exact " +
                   "step. Switch the model selector back to use it.";
      } else {
        el.title = "";
        if (id !== "btn-exact" && id !== "btn-fluct") el.disabled = false;
      }
    }
    if (!stoch) {
      // A stochastic sample clears the live chain (different model, different
      // boundary conditions), so switching back must rebuild it -- otherwise
      // Run is silently dead, exactly as it was after an exact sample once.
      if (!sampler) localInit();
      updateDeltaDisplay();
    }
    if (stoch && sb1 && sb2) {
      const b1 = parseFloat(sb1.value), b2 = parseFloat(sb2.value);
      const d = (b1 + b2) / (2 * Math.sqrt(b1 * b2));
      document.getElementById("sb1-out").textContent = b1.toFixed(2);
      document.getElementById("sb2-out").textContent = b2.toFixed(2);
      document.getElementById("stoch-delta").textContent =
        `\u0394 = ${d.toFixed(3)}  (ferroelectric)`;
    }
  }
  if (modelSel) modelSel.addEventListener("change", updateModelUI);
  if (sb1) sb1.addEventListener("input", updateModelUI);
  if (sb2) sb2.addEventListener("input", updateModelUI);
  updateModelUI();

  const btnStoch = document.getElementById("btn-stoch");
  if (btnStoch) btnStoch.addEventListener("click", async () => {
    if (busy) return;
    busy = true; btnStoch.disabled = true;
    hudStatus.textContent = "sampling (stochastic)...";
    try {
      const res = await fetch("/api/stochastic", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          n: parseInt(nSlider.value, 10),
          b1: parseFloat(sb1.value), b2: parseFloat(sb2.value),
          seed: currentSeed() }),
      });
      const data = await res.json();
      if (!data.ok) { exactInfo.textContent = data.error; return; }
      lastFrame = frameFromServerData(data.frame);
      sampler = null;
      fitCamera(); draw();
      const inf = data.info;
      exactInfo.textContent =
        `exact stochastic sample (one sweep, no Markov chain) - ` +
        `b1=${inf.b1}, b2=${inf.b2}, \u0394=${inf.delta.toFixed(3)}` +
        (inf.seed !== null && inf.seed !== undefined ? `, seed ${inf.seed}` : "");
      hudStatus.textContent = "stochastic sample";
      hudDevice.textContent = "server (sequential, exact)";
    } catch (err) {
      exactInfo.textContent = `stochastic sampling failed: ${err.message}`;
      hudStatus.textContent = "ready";
    } finally {
      busy = false; btnStoch.disabled = false;
    }
  });

  btnExact.addEventListener("click", async () => {
    if (exactInFlight) return;
    const w = currentWeights();
    if (!isExactSafe(w, parseInt(nSlider.value, 10))) return;
    exactInFlight = true;
    playing = false;
    btnPlay.classList.remove("active");
    btnPlay.textContent = "run";
    busy = true;
    btnExact.disabled = true;
    hudStatus.textContent = "exact sampling (server)...";
    const n = parseInt(nSlider.value, 10);

    const requestedGen = paramGen;
    const startTime = Date.now();

    function updateProgressUI(lastT, attempts) {
      const elapsed = Math.round((Date.now() - startTime) / 1000);
      btnExact.textContent = `coalescing... (${elapsed}s)`;
      if (elapsed > 5) {
        let msg = `still running -- larger n or extreme weights can take a while (${elapsed}s elapsed)`;
        if (lastT) msg += `, currently attempting ${lastT} half-sweeps (doubling ${attempts})`;
        exactInfo.textContent = msg;
      }
    }

    try {
      const startRes = await fetch("/api/exact/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ n, c_up: w.c1, c_down: w.c2, a1: w.a1, a2: w.a2, b1: w.b1, b2: w.b2 }),
      });
      const startData = await startRes.json();
      if (!startData.ok) {
        exactInfo.textContent = `exact sampling failed to start: ${startData.error}`;
        hudStatus.textContent = "ready";
        return;                       // cleanup runs in `finally`
      }

      const jobId = startData.job_id;
      // Bound the poll loop. If the worker thread ever dies without writing
      // a terminal status (OOM kill, hard restart), the job sits at
      // "running" forever and this loop would poll once a second until the
      // tab is closed -- with the button stuck disabled the whole time.
      const POLL_LIMIT_SECONDS = 45 * 60;
      let finished = false;
      let elapsedPolling = 0;
      let consecutiveFailures = 0;
      // A long job means many polls -- a 16 minute run at one poll a second
      // is about a thousand requests. Previously ANY single failure among
      // them threw out of the loop and abandoned the computation, even
      // though the server was still working on it happily. One hiccup in a
      // thousand requests over a free-tier host behind a proxy is close to
      // certain, and a proxy timeout returns an HTML error page, which makes
      // .json() throw rather than returning {ok:false}.
      //
      // So: tolerate isolated failures, and back the interval off as the job
      // ages so a long run does not hammer the server it is waiting on.
      while (!finished) {
        const interval = elapsedPolling < 30 ? 1000
                       : elapsedPolling < 120 ? 2000 : 5000;
        await new Promise(resolve => setTimeout(resolve, interval));
        elapsedPolling += interval / 1000;
        if (elapsedPolling > POLL_LIMIT_SECONDS) {
          exactInfo.textContent =
            "gave up waiting for the server after 45 minutes; the job may " +
            "still be running. Try a smaller n.";
          hudStatus.textContent = "ready";
          break;
        }

        let statusData = null;
        try {
          const statusRes = await fetch(`/api/exact/status/${jobId}`);
          statusData = await statusRes.json();
          consecutiveFailures = 0;
        } catch (pollErr) {
          consecutiveFailures++;
          if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
            exactInfo.textContent =
              `lost contact with the server after ${consecutiveFailures} ` +
              `consecutive failed status checks; the job may still be ` +
              `running on the server.`;
            hudStatus.textContent = "ready";
            break;
          }
          // transient: keep waiting, the computation is still going
          exactInfo.textContent =
            `still running (status check retrying, ` +
            `${consecutiveFailures}/${MAX_CONSECUTIVE_FAILURES})`;
          continue;
        }

        if (!statusData.ok) {
          exactInfo.textContent = `status check failed: ${statusData.error}`;
          hudStatus.textContent = "ready";
          finished = true;
          break;
        }
        if (statusData.status === "running") {
          updateProgressUI(statusData.last_T, statusData.attempts);
        } else if (statusData.status === "done") {
          totalSweeps = 0;
          sweepCount.textContent = totalSweeps;
          const inf = statusData.info || {};
          if (inf.method === "exact-sequential") {
            exactInfo.textContent =
              `exact sample (sequential transfer-matrix method \u2014 valid for all weights)`;
          } else {
            // the sweep count is reported as `sweeps` by cftp_exact.py and as
            // `half_sweeps` by the legacy module; accept either.
            const sw = inf.sweeps !== undefined ? inf.sweeps : inf.half_sweeps;
            exactInfo.textContent =
              `exact sample (CFTP, coalesced after ${sw} sweeps, ${inf.attempts} doublings)`;
          }
          // Seed the client-side chain FROM the exact sample rather than
          // discarding it.
          //
          // This used to set `sampler = null`, which left Run silently dead:
          // clicking it flipped the button to "pause" (so it looked live) but
          // localStep() bails on `!sampler`, so the sweep counter sat at 0
          // until the user happened to press Reset. Nothing reported an error.
          //
          // Continuing the chain from an exact draw is also the right thing
          // scientifically: the chain starts in equilibrium, so there is no
          // burn-in to discard.
          if (paramGen !== requestedGen) {
            exactInfo.textContent =
              "exact sample discarded: the chain was reset or its parameters changed while it was running";
            hudStatus.textContent = "ready";
            finished = true;
            break;
          }
          const exactFrame = frameFromServerData(statusData.frame);
          const en = exactFrame.n;
          const esize = en + 1;
          sampler = new SixVertexJS(en, w, Date.now() & 0xffffffff);
          for (let i = 0; i < esize; i++) {
            for (let j = 0; j < esize; j++) {
              sampler.H[i * esize + j] = exactFrame.get(i, j);
            }
          }
          renderFrame(frameFromSampler(sampler));
          hudDevice.textContent = "in-browser (JS), seeded from exact sample";
          hudStatus.textContent = "exact sample";
          finished = true;
        } else if (statusData.status === "error") {
          exactInfo.textContent = `exact sampling failed: ${statusData.error}`;
          hudStatus.textContent = "ready";
          finished = true;
        }
      }
    } catch (e) {
      exactInfo.textContent = "request to server failed (is it reachable?)";
      hudStatus.textContent = "ready";
    } finally {
      // MUST be `finally`. The early `return` on a refused start skipped this
      // cleanup entirely, so exactInFlight stayed true and the button was
      // disabled permanently -- reachable as soon as the server started
      // returning 429 for too many concurrent jobs. Every exit path has to
      // release the flag, not just the happy one.
      btnExact.textContent = "exact sample";
      exactInFlight = false;
      busy = false;
      updateDeltaDisplay();
    }
  });

  btnSave.addEventListener("click", () => {
    const link = document.createElement("a");
    link.download = "six-vertex-sample.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
  });

  btnSaveSvg.addEventListener("click", () => {
    if (!lastFrame) return;
    const n = lastFrame.n;
    const cell = 4;
    const size = n * cell;
    let rects = "";
    if (viewMode.value === "active") {
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const isActive = lastFrame.getActive(i, j) || lastFrame.getActive(i + 1, j) ||
                            lastFrame.getActive(i, j + 1) || lastFrame.getActive(i + 1, j + 1);
          const fill = isActive ? "#e8a05c" : "#141e2a";
          rects += `<rect x="${j * cell}" y="${i * cell}" width="${cell}" height="${cell}" fill="${fill}"/>`;
        }
      }
    } else if (viewMode.value === "vertex6") {
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const type = classifyFaceLocal(lastFrame.get(i, j), lastFrame.get(i, j + 1), lastFrame.get(i + 1, j), lastFrame.get(i + 1, j + 1));
          const [r, g, b] = VERTEX6_COLORS[type];
          rects += `<rect x="${j * cell}" y="${i * cell}" width="${cell}" height="${cell}" fill="rgb(${r},${g},${b})"/>`;
        }
      }
    } else {
      const min = lastFrame.min, max = lastFrame.max;
      const span = Math.max(1, max - min);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const avg = (lastFrame.get(i, j) + lastFrame.get(i + 1, j) + lastFrame.get(i, j + 1) + lastFrame.get(i + 1, j + 1)) / 4;
          const t = (avg - min) / span;
          const [r, g, b] = colorFor(t);
          const fill = `rgb(${r | 0},${g | 0},${b | 0})`;
          rects += `<rect x="${j * cell}" y="${i * cell}" width="${cell}" height="${cell}" fill="${fill}"/>`;
        }
      }
    }
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">${rects}</svg>`;
    const blob = new Blob([svg], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.download = "six-vertex-sample.svg";
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);
  });

  // Ask the server for the authoritative exact-sampling limit. Non-blocking:
  // the hardcoded default above is used until (or unless) this returns.
  fetch("/api/config")
    .then(r => r.json())
    .then(cfg => {
      if (cfg && typeof cfg.max_exact_n === "number") {
        MAX_EXACT_N = cfg.max_exact_n;
        updateDeltaDisplay();
      }
    })
    .catch(() => {});

  sizeStage();
  localInit();
  setTimeout(hideLoadingScreen, 8000);
})();
