/* Same rigid registration error, measured at two sets of points:
 *   green:  pose.inverse()(f), the CT fiducials in the camera frame (on the anatomy)
 *   purple: pose(f), what xvr's mTRE uses; CT coordinates read as camera coordinates (around the source)
 * All numbers are computed from the poses in data/scene.js. */
'use strict';
(() => {
  const $ = (s) => document.querySelector(s);
  const S = window.SCENE;

  // 4x4 rigid transforms: flat row-major, column vectors
  const flat = (m) => m.flat();
  const mul = (a, b) => { const o = new Array(16).fill(0); for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) for (let k = 0; k < 4; k++) o[4 * i + j] += a[4 * i + k] * b[4 * k + j]; return o; };
  const inv = (m) => {
    const o = [m[0], m[4], m[8], 0, m[1], m[5], m[9], 0, m[2], m[6], m[10], 0, 0, 0, 0, 1];
    for (let i = 0; i < 3; i++) o[4 * i + 3] = -(o[4 * i] * m[3] + o[4 * i + 1] * m[7] + o[4 * i + 2] * m[11]);
    return o;
  };
  const ap = (m, p) => [0, 1, 2].map((i) => m[4 * i] * p[0] + m[4 * i + 1] * p[1] + m[4 * i + 2] * p[2] + m[4 * i + 3]);
  const sub = (a, b) => a.map((x, i) => x - b[i]);
  const add = (a, b) => a.map((x, i) => x + b[i]);
  const scl = (a, s) => a.map((x) => x * s);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => Math.hypot(...a);
  const unit = (a) => scl(a, 1 / (norm(a) || 1));
  const mean = (ps) => scl(ps.reduce(add, [0, 0, 0]), 1 / ps.length);
  const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const rotExp = (w) => {
    const th = norm(w); if (th < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const [x, y, z] = scl(w, 1 / th), c = Math.cos(th), s = Math.sin(th), C = 1 - c;
    return [c + x * x * C, x * y * C - z * s, x * z * C + y * s, y * x * C + z * s, c + y * y * C, y * z * C - x * s, z * x * C - y * s, z * y * C + x * s, c + z * z * C];
  };
  const rotLog = (m) => {
    const th = Math.acos(Math.max(-1, Math.min(1, (m[0] + m[5] + m[10] - 1) / 2))), v = [m[9] - m[6], m[2] - m[8], m[4] - m[1]];
    return th < 1e-9 ? scl(v, 0.5) : scl(v, th / (2 * Math.sin(th)));
  };
  const r3 = (R, p) => [R[0] * p[0] + R[1] * p[1] + R[2] * p[2], R[3] * p[0] + R[4] * p[1] + R[5] * p[2], R[6] * p[0] + R[7] * p[1] + R[8] * p[2]];
  const fromRt = (R, t) => [R[0], R[1], R[2], t[0], R[3], R[4], R[5], t[1], R[6], R[7], R[8], t[2], 0, 0, 0, 1];
  const about = (c, R, t) => fromRt(R, add(sub(c, r3(R, c)), t)); // rotate about c, then translate by t
  const scaleMotion = (B, c, k) => { const Rk = rotExp(scl(rotLog(B), k)); return fromRt(Rk, sub(add(c, scl(sub(ap(B, c), c), k)), r3(Rk, c))); };

  // Scene data. T = inv(pose @ reorient): CT -> camera frame (source at origin, z towards the detector).
  // With DiffDRR's AP reorientation, pose(f) = (pose @ reorient)(reorient^T f), so xvr's points are
  // reorient^T f = (x, z, -y) in the camera frame: the CT coordinate numbers placed at the source.
  const GHOST = [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1];
  const decode = (b64, T) => { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new T(u.buffer); };
  const V = decode(S.mesh.vertices, Float32Array), F = decode(S.mesh.faces, Uint32Array);
  const FN = new Float32Array(F.length);
  for (let j = 0; j < F.length; j += 3) {
    const p = (i) => [V[3 * F[i]], V[3 * F[i] + 1], V[3 * F[i] + 2]];
    FN.set(unit(cross(sub(p(j + 1), p(j)), sub(p(j + 2), p(j)))), j);
  }
  const Tgt = flat(S.T_gt), P = S.fiducials.map((f) => ap(Tgt, f)), G = S.fiducials.map((f) => ap(GHOST, f)), c = mean(P);
  const D = S.detector, fx = D.sdd / D.delx, fy = D.sdd / D.dely, cx = -D.x0 / D.delx + D.width / 2, cy = -D.y0 / D.dely + D.height / 2;
  const pixel = (u, vi) => [(u - cx) / fx * D.sdd, (D.height - vi - cy) / fy * D.sdd, D.sdd];
  const corners = [pixel(0, 0), pixel(D.width, 0), pixel(D.width, D.height), pixel(0, D.height)];
  const xray = new Image(); xray.onload = () => request(); xray.src = S.xray;

  const deg = Math.PI / 180;
  const PRESETS = {
    tilt: () => about(c, rotExp([deg, 0, 0]), [0, 0, 0]),
    inplane: () => about(c, rotExp([0, 0, deg]), [0, 0, 0]),
    shift: () => about(c, rotExp([0, 0, 0]), [1, 0, 0]),
    result: () => mul(flat(S.result.T), inv(Tgt)),
  };
  const VIEWS = { side: [-Math.PI / 2, 0.15], oblique: [-2.25, 0.4], beam: [0, 0] };
  const state = { preset: 'tilt', B: PRESETS.tilt(), frame: 'carm', ex: 10, animate: false, phase: 1, az: VIEWS.side[0], el: VIEWS.side[1], zoom: 1 };

  function metrics(B) {
    const tre = P.map((p) => norm(sub(ap(B, p), p))), xvr = G.map((g) => norm(sub(ap(B, g), g)));
    return { tre: avg(tre), treMax: Math.max(...tre), xvr: avg(xvr), xvrMax: Math.max(...xvr), rot: norm(rotLog(B)) / deg,
             trans: norm(sub(ap(B, c), c)), radius: avg(P.map((p) => norm(sub(p, c)))), lever: avg(G.map((g) => norm(sub(g, c)))) };
  }
  function show() {
    const m = metrics(state.B), f2 = (x) => x.toFixed(2);
    $('#tre').textContent = f2(m.tre); $('#tre-max').textContent = f2(m.treMax);
    $('#xvr').textContent = f2(m.xvr); $('#xvr-max').textContent = f2(m.xvrMax);
    $('#facts').textContent = `Rotation error ${f2(m.rot)}°, translation at the anatomy centre ${f2(m.trans)} mm. xvr mTRE is ${(m.xvr / m.tre).toFixed(1)}× the error at the anatomy.`;
    $('#radius').textContent = m.radius.toFixed(0); $('#lever').textContent = m.lever.toFixed(0);
    state.metrics = m;
  }

  // Orthographic canvas renderer (painter's algorithm), no WebGL needed
  const cv = $('#scene'), ctx = cv.getContext('2d');
  let queued = false;
  function request() { if (!queued) { queued = true; requestAnimationFrame(draw); } }
  function draw() {
    queued = false;
    if (state.animate) { state.phase = 0.5 - 0.5 * Math.cos(performance.now() / 2600 * 2 * Math.PI); request(); }
    const dpr = Math.min(devicePixelRatio || 1, 2), W = Math.round(cv.clientWidth * dpr), H = Math.round(cv.clientHeight * dpr);
    if (!W || !H) return;
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    const w = [Math.cos(state.el) * Math.sin(state.az), Math.sin(state.el), Math.cos(state.el) * Math.cos(state.az)];
    const u = unit(cross([0, 1, 0], w)), v = cross(w, u);
    // frame the true scene (C-arm, anatomy, measurement points) so the view does not jump
    const anatomyPts = [], ghostPts = [];
    for (let i = 0; i < V.length; i += 30) { anatomyPts.push(ap(Tgt, [V[i], V[i + 1], V[i + 2]])); ghostPts.push(ap(GHOST, [V[i], V[i + 1], V[i + 2]])); }
    const frame = [[0, 0, 0], ...corners, ...P, ...G, ...anatomyPts, ...ghostPts];
    let a0 = 1e9, a1 = -1e9, b0 = 1e9, b1 = -1e9;
    for (const p of frame) { const a = dot(p, u), b = dot(p, v); a0 = Math.min(a0, a); a1 = Math.max(a1, a); b0 = Math.min(b0, b); b1 = Math.max(b1, b); }
    const s = 0.86 * Math.min(W / (a1 - a0), H / (b1 - b0)) * state.zoom, ma = (a0 + a1) / 2, mb = (b0 + b1) / 2;
    const pr = (p) => [W / 2 + s * (dot(p, u) - ma), H / 2 - s * (dot(p, v) - mb), dot(p, w)];
    ctx.fillStyle = '#f7f9fb'; ctx.fillRect(0, 0, W, H);
    const font = (px, weight = 500) => { ctx.font = `${weight} ${px * dpr}px Inter, "Segoe UI", sans-serif`; };
    const line = (p, q, color, width, dash) => { const a = pr(p), b = pr(q); ctx.setLineDash(dash ? dash.map((x) => x * dpr) : []); ctx.strokeStyle = color; ctx.lineWidth = width * dpr; ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); ctx.setLineDash([]); };
    const point = (p, color, r) => { const a = pr(p); ctx.fillStyle = color; ctx.beginPath(); ctx.arc(a[0], a[1], r * dpr, 0, 2 * Math.PI); ctx.fill(); };
    // text with a light halo, kept inside the canvas
    const label = (str, x, y, color, px = 12) => {
      font(px, 600); const tw = ctx.measureText(str).width, m = 6 * dpr;
      x = Math.max(m, Math.min(W - tw - m, x - tw / 2)); y = Math.max(px * dpr + m, y);
      ctx.lineJoin = 'round'; ctx.lineWidth = 4 * dpr; ctx.strokeStyle = 'rgba(247,249,251,0.92)'; ctx.strokeText(str, x, y);
      ctx.fillStyle = color; ctx.fillText(str, x, y);
    };
    const box = (pts) => { let x0 = 1e9, x1 = -1e9, y0 = 1e9; for (const p of pts) { const a = pr(p); x0 = Math.min(x0, a[0]); x1 = Math.max(x1, a[0]); y0 = Math.min(y0, a[1]); } return [(x0 + x1) / 2, y0]; };

    const Bd = scaleMotion(state.B, c, state.ex * state.phase), Bi = inv(Bd);
    const anatomyFixed = state.frame === 'anatomy';
    // C-arm: detector with the X-ray, rays, source. With the anatomy fixed, the estimated C-arm moves by inv(B).
    const carm = (M, color, textured) => {
      const q = corners.map((p) => pr(ap(M, p))), src = pr(ap(M, [0, 0, 0]));
      if (textured && xray.complete && xray.naturalWidth) {
        ctx.save(); ctx.globalAlpha = 0.9;
        ctx.setTransform((q[1][0] - q[0][0]) / xray.naturalWidth, (q[1][1] - q[0][1]) / xray.naturalWidth, (q[3][0] - q[0][0]) / xray.naturalHeight, (q[3][1] - q[0][1]) / xray.naturalHeight, q[0][0], q[0][1]);
        ctx.drawImage(xray, 0, 0); ctx.restore();
      }
      ctx.lineWidth = dpr; ctx.strokeStyle = color + '55';
      for (const p of q) { ctx.beginPath(); ctx.moveTo(src[0], src[1]); ctx.lineTo(p[0], p[1]); ctx.stroke(); }
      ctx.strokeStyle = color; ctx.lineWidth = 2 * dpr; ctx.beginPath(); q.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.closePath(); ctx.stroke();
      return src;
    };
    const I4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const sources = [[carm(I4, '#2f8fe0', true), '#2f8fe0', anatomyFixed ? 'X-ray source (ground truth)' : 'X-ray source = camera origin']];
    if (anatomyFixed) sources.push([carm(Bi, '#ff4f7b', false), '#ff4f7b', 'X-ray source (as registered)']);

    // meshes, all faces depth-sorted together
    const items = [{ M: Tgt, rgb: [201, 208, 214], a: 1 }];
    if (!anatomyFixed) items.push({ M: mul(Bd, Tgt), rgb: [255, 138, 31], a: 0.5 });
    items.push({ M: GHOST, rgb: [155, 77, 255], a: 0.12 }, { M: mul(anatomyFixed ? Bi : Bd, GHOST), rgb: [155, 77, 255], a: 0.4 });
    const L = unit(add(add(scl(u, -0.35), scl(v, 0.55)), scl(w, 0.75))), faces = [];
    for (const it of items) {
      const M = it.M, R = [M[0], M[1], M[2], M[4], M[5], M[6], M[8], M[9], M[10]], q = new Float32Array(V.length);
      for (let i = 0; i < V.length; i += 3) { const p = pr(ap(M, [V[i], V[i + 1], V[i + 2]])); q[i] = p[0]; q[i + 1] = p[1]; q[i + 2] = p[2]; }
      it.q = q; it.styles = Array.from({ length: 21 }, (_, k) => { const f = 0.55 + 0.45 * k / 20; return `rgba(${it.rgb.map((x) => Math.round(x * f)).join(',')},${it.a})`; });
      for (let j = 0; j < F.length; j += 3) {
        const shade = Math.abs(dot(r3(R, [FN[j], FN[j + 1], FN[j + 2]]), L));
        faces.push([q[3 * F[j] + 2] + q[3 * F[j + 1] + 2] + q[3 * F[j + 2] + 2], it, j, Math.round(shade * 20)]);
      }
    }
    faces.sort((x, y) => x[0] - y[0]);
    for (const [, it, j, k] of faces) {
      const q = it.q, a = 3 * F[j], b = 3 * F[j + 1], d = 3 * F[j + 2];
      ctx.fillStyle = it.styles[k]; ctx.beginPath(); ctx.moveTo(q[a], q[a + 1]); ctx.lineTo(q[b], q[b + 1]); ctx.lineTo(q[d], q[d + 1]); ctx.closePath(); ctx.fill();
    }

    // displacements: green at the anatomy, purple around the source
    if (!anatomyFixed) for (const p of P) { const q = ap(Bd, p); line(p, q, '#12a663', 1.8); point(p, '#0c8a52', 2.2); point(q, '#e06a00', 2.2); }
    else for (const p of P) point(p, '#0c8a52', 2.2);
    const Mg = anatomyFixed ? Bi : Bd;
    for (const g of G) { const q = ap(Mg, g); line(g, q, '#9b4dff', 1.5); point(g, '#b98cff', 1.8); point(q, '#7a2fe0', 2); }
    const gc = mean(G), m = state.metrics;
    line(c, gc, '#5d6f7b', 1, [3, 4]);
    const mid = pr(scl(add(c, gc), 0.5)); label(`≈ ${m.lever.toFixed(0)} mm lever arm`, mid[0], mid[1] - 8 * dpr, '#3d4f5c');
    const ghostTop = box(ghostPts), anatomyTop = box(anatomyPts);
    label('pose(f): points used by xvr mTRE', ghostTop[0], ghostTop[1] - 12 * dpr, '#7a35d6', 13);
    label('pose.inverse()(f): fiducials on the anatomy', anatomyTop[0], anatomyTop[1] - 12 * dpr, '#0c8a52', 13);
    for (const [src, color, text] of sources) {
      ctx.fillStyle = color; ctx.beginPath(); ctx.arc(src[0], src[1], 5 * dpr, 0, 2 * Math.PI); ctx.fill();
      label(text, src[0], src[1] + 26 * dpr, color);
    }
    $('#badge').textContent = state.ex > 1 ? `Motion drawn ×${state.ex}; the numbers use the true error` : '';
  }

  // controls
  document.querySelectorAll('[data-preset]').forEach((b) => {
    b.onclick = () => {
      state.preset = b.dataset.preset; state.B = PRESETS[state.preset]();
      document.querySelectorAll('[data-preset]').forEach((x) => x.classList.toggle('on', x === b));
      show(); request();
    };
  });
  if (!S.result) document.querySelector('[data-preset=result]').remove();
  document.querySelectorAll('[name=frame]').forEach((i) => {
    i.onchange = () => { state.frame = i.value; $('#legend-moved').textContent = i.value === 'carm' ? 'Anatomy as registered' : 'C-arm as registered'; $('#legend-moved').previousElementSibling.style.background = i.value === 'carm' ? '#ff8a1f' : '#ff4f7b'; request(); };
  });
  $('#view').onchange = () => { [state.az, state.el] = VIEWS[$('#view').value]; state.zoom = 1; request(); };
  $('#exaggerate').onchange = () => { state.ex = Number($('#exaggerate').value); request(); };
  $('#animate').onchange = () => { state.animate = $('#animate').checked; state.phase = 1; request(); };
  let drag = null;
  cv.onpointerdown = (e) => { drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); };
  cv.onpointermove = (e) => { if (!drag) return; state.az -= (e.clientX - drag[0]) * 0.008; state.el = Math.max(-1.45, Math.min(1.45, state.el + (e.clientY - drag[1]) * 0.008)); drag = [e.clientX, e.clientY]; request(); };
  cv.onpointerup = cv.onpointercancel = () => { drag = null; };
  cv.addEventListener('wheel', (e) => { e.preventDefault(); state.zoom = Math.max(0.3, Math.min(40, state.zoom * Math.exp(-e.deltaY * 0.0012))); request(); }, { passive: false });
  cv.ondblclick = () => { [state.az, state.el] = VIEWS[$('#view').value]; state.zoom = 1; request(); };
  new ResizeObserver(request).observe(cv);
  show(); request();

  window.demoState = () => ({ preset: state.preset, frame: state.frame, ex: state.ex, az: state.az, zoom: state.zoom, metrics: { ...state.metrics } });
})();
