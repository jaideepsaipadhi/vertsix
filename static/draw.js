(() => {
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

  function currentWeights() {
    return {
      a1: parseFloat(a1Slider.value), a2: parseFloat(a2Slider.value),
      b1: parseFloat(b1Slider.value), b2: parseFloat(b2Slider.value),
      c1: parseFloat(c1Slider.value), c2: parseFloat(c2Slider.value),
    };
  }

  function isExactSafe(w) {
    return w.a1 === 1 && w.a2 === 1 && w.b1 === 1 && w.b2 === 1;
  }

  function pairedSlider(sliderA, outA, sliderB, outB) {
    sliderA.addEventListener("input", () => {
      outA.textContent = parseFloat(sliderA.value).toFixed(2);
      if (symmetricCheck.checked) {
        sliderB.value = sliderA.value;
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
    const a1a2 = w.a1 * w.a2, b1b2 = w.b1 * w.b2, c1c2 = w.c1 * w.c2;
    const delta = (a1a2 + b1b2 - c1c2) / (2 * Math.sqrt(a1a2 * b1b2));
    let regime;
    if (delta > 1) regime = "ferroelectric";
    else if (delta < -1) regime = "antiferroelectric";
    else regime = "disordered";
    if (w.a1 === 1 && w.a2 === 1 && w.b1 === 1 && w.b2 === 1 && w.c1 === 1 && w.c2 === 1) regime += " (uniform weights)";
    deltaDisplay.textContent = `${delta.toFixed(2)} \u00b7 ${regime}`;

    orderedRegimeNote.style.display = Math.abs(delta) > 1 ? "block" : "none";

    const safe = isExactSafe(w);
    btnExact.disabled = !safe;
    btnExact.style.opacity = safe ? "1" : "0.4";
    btnExact.style.cursor = safe ? "pointer" : "not-allowed";
    exactGateNote.style.display = safe ? "none" : "block";
  }

  nSlider.addEventListener("input", () => (nOut.textContent = nSlider.value));
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
    const rect = stage.getBoundingClientRect();
    const availWidth = rect.width > 10 ? rect.width : mainEl.clientWidth;
    const availHeight = mainEl.clientHeight;
    const w = Math.round(Math.max(280, Math.min(availWidth, 2000)));
    const h = Math.round(Math.max(280, Math.min(availHeight, 1400)));

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
    if (l && t && !b && !r) return "b1";
    if (!l && !t && b && r) return "b2";
    if (!l && t && b && !r) return "c1";
    if (l && !t && !b && r) return "c2";
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
  const MIN_LOADING_MS = 3000;

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
    busy = true;
    hudStatus.textContent = "initializing...";
    const n = parseInt(nSlider.value, 10);
    const w = currentWeights();
    try {
      sampler = new SixVertexJS(n, w, Date.now() & 0xffffffff);
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

  function localStep() {
    if (busy || !sampler) return;
    const sweeps = parseInt(speedSlider.value, 10);
    hudStatus.textContent = playing ? "running" : "ready";
    sampler.step(sweeps);
    totalSweeps += sweeps;
    sweepCount.textContent = totalSweeps;
    lastFrame = frameFromSampler(sampler);
    draw();
  }

  function loop() {
    if (!playing) return;
    localStep();
    requestAnimationFrame(loop);
  }

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

  btnExact.addEventListener("click", async () => {
    if (busy) return;
    const w = currentWeights();
    if (!isExactSafe(w)) return;
    playing = false;
    btnPlay.classList.remove("active");
    btnPlay.textContent = "run";
    busy = true;
    btnExact.disabled = true;
    hudStatus.textContent = "CFTP running (server)...";
    const n = parseInt(nSlider.value, 10);

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
        body: JSON.stringify({ n, c_up: w.c1, c_down: w.c2 }),
      });
      const startData = await startRes.json();
      if (!startData.ok) {
        exactInfo.textContent = `CFTP failed to start: ${startData.error}`;
        hudStatus.textContent = "ready";
        btnExact.textContent = "exact sample (CFTP)";
        busy = false;
        updateDeltaDisplay();
        return;
      }

      const jobId = startData.job_id;
      let finished = false;
      while (!finished) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        const statusRes = await fetch(`/api/exact/status/${jobId}`);
        const statusData = await statusRes.json();
        if (!statusData.ok) {
          exactInfo.textContent = `CFTP status check failed: ${statusData.error}`;
          hudStatus.textContent = "ready";
          finished = true;
          break;
        }
        if (statusData.status === "running") {
          updateProgressUI(statusData.last_T, statusData.attempts);
        } else if (statusData.status === "done") {
          totalSweeps = 0;
          sweepCount.textContent = totalSweeps;
          exactInfo.textContent = `exact: coalesced after ${statusData.info.half_sweeps} half-sweeps (${statusData.info.attempts} doublings)`;
          sampler = null;
          renderFrame(frameFromServerData(statusData.frame));
          hudDevice.textContent = "server (numpy/torch, CFTP only)";
          hudStatus.textContent = "exact sample";
          finished = true;
        } else if (statusData.status === "error") {
          exactInfo.textContent = `CFTP failed: ${statusData.error}`;
          hudStatus.textContent = "ready";
          finished = true;
        }
      }
    } catch (e) {
      exactInfo.textContent = "CFTP request failed";
      hudStatus.textContent = "ready";
    }
    btnExact.textContent = "exact sample (CFTP)";
    busy = false;
    updateDeltaDisplay();
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

  sizeStage();
  localInit();
  setTimeout(hideLoadingScreen, 8000);
})();
