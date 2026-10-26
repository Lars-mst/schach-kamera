// Schach Kamera: Brett finden (OpenCV), Züge über Feld-Veränderungen erkennen (chess.js),
// besten Zug berechnen (Stockfish WASM) und perspektivisch korrekt ins Kamerabild zeichnen.
import { Chess } from './lib/chess.js';

const $ = (s) => document.querySelector(s);
const video = $('#video');
const overlay = $('#overlay');
const octx = overlay.getContext('2d');
const DEBUG = new URLSearchParams(location.search).has('debug');

// Alle Bildverarbeitung läuft auf einer verkleinerten Kopie des Kamerabilds ("work"-Koordinaten).
const WORK_W = 640;
const work = document.createElement('canvas');
const wctx = work.getContext('2d', { willReadFrequently: true });

const PIECE_DE = { p: 'Bauer', n: 'Springer', b: 'Läufer', r: 'Turm', q: 'Dame', k: 'König' };
const GLYPH = { p: '♟', n: '♞', b: '♝', r: '♜', q: '♛', k: '♚' };
const FILES = 'abcdefgh';

const S = {
  mode: 'idle',          // idle | search | track | manual | adjust
  stream: null,
  quad: null,            // Brett-Ecken in work-Koordinaten, Reihenfolge TL, TR, BR, BL im Bild
  a1: 3,                 // Index der Ecke in quad, an der a1 liegt
  H: null,               // Brett (0..8, 0..8) -> work
  Hinv: null,
  game: new Chess(),
  myColor: 'w',
  thinkMs: 1000,
  ref: null,             // Feld-Merkmale der zuletzt bestätigten Stellung
  prev: null,
  stable: 0,
  evaluated: false,
  unclear: false,
  cand: null, candHits: 0,
  best: null,            // { from, to, san, piece, color }
  evalText: '',
  lastMove: null,
  tapFrom: null,
  manualPts: [],
  drag: -1,
  cv: null,
  lastTick: 0,
};
window.__S = S; // für Tests / Debugging

/* ---------- Geometrie ---------- */

function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c];
    if (Math.abs(d) < 1e-12) return null;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / d;
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

function homography(src, dst) {
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i], [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  const h = solve(A, b);
  return h ? [...h, 1] : null;
}

function project(h, x, y) {
  const w = h[6] * x + h[7] * y + h[8];
  return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w];
}

function orderQuad(pts) {
  // TL = kleinste x+y, BR = größte x+y, TR = kleinste y-x, BL = größte y-x
  const by = (f) => pts.reduce((a, p) => (f(p) < f(a) ? p : a));
  const byMax = (f) => pts.reduce((a, p) => (f(p) > f(a) ? p : a));
  return [by((p) => p[0] + p[1]), by((p) => p[1] - p[0]), byMax((p) => p[0] + p[1]), byMax((p) => p[1] - p[0])];
}

function setBoard(quad, a1) {
  S.quad = quad;
  S.a1 = a1;
  // Im Uhrzeigersinn ab a1 folgen im Bild a8, h8, h1.
  const q = (i) => quad[(a1 + i) % 4];
  const board = [[0, 0], [0, 8], [8, 8], [8, 0]];
  S.H = homography(board, [q(0), q(1), q(2), q(3)]);
  S.Hinv = homography([q(0), q(1), q(2), q(3)], board);
}

/* ---------- Kamerabild lesen ---------- */

function grab() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw) return null;
  const h = Math.round((WORK_W * vh) / vw);
  if (work.width !== WORK_W || work.height !== h) { work.width = WORK_W; work.height = h; }
  wctx.drawImage(video, 0, 0, WORK_W, h);
  return wctx.getImageData(0, 0, WORK_W, h);
}

function lumAt(img, x, y) {
  const xi = Math.min(img.width - 1, Math.max(0, x | 0)), yi = Math.min(img.height - 1, Math.max(0, y | 0));
  const i = (yi * img.width + xi) * 4;
  return 0.299 * img.data[i] + 0.587 * img.data[i + 1] + 0.114 * img.data[i + 2];
}

// Mittelwert und Streuung der Helligkeit im inneren Bereich eines Feldes (u, v in Einheiten von Hq).
function cellStats(img, Hq, u0, v0, size, inner = 0.6, n = 6) {
  let s = 0, s2 = 0, k = 0;
  const pad = (1 - inner) / 2;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const u = u0 + size * (pad + (inner * (i + 0.5)) / n);
    const v = v0 + size * (pad + (inner * (j + 0.5)) / n);
    const [x, y] = project(Hq, u, v);
    const L = lumAt(img, x, y);
    s += L; s2 += L * L; k++;
  }
  const m = s / k;
  return { m, sd: Math.sqrt(Math.max(0, s2 / k - m * m)) };
}

/* ---------- Brett finden ---------- */

function findQuads(img) {
  const cv = S.cv;
  const out = [];
  const src = cv.matFromImageData(img);
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0);
  const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
  const area0 = img.width * img.height;
  for (const mode of ['canny', 'thresh']) {
    const edges = new cv.Mat();
    if (mode === 'canny') {
      cv.Canny(gray, edges, 40, 120);
      cv.dilate(edges, edges, kernel, new cv.Point(-1, -1), 1);
    } else {
      cv.adaptiveThreshold(gray, edges, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 7);
    }
    const contours = new cv.MatVector(), hier = new cv.Mat();
    cv.findContours(edges, contours, hier, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i);
      const area = cv.contourArea(c);
      if (area > 0.06 * area0 && area < 0.97 * area0) {
        const approx = new cv.Mat();
        cv.approxPolyDP(c, approx, 0.02 * cv.arcLength(c, true), true);
        if (approx.rows === 4 && cv.isContourConvex(approx)) {
          const d = approx.data32S;
          out.push({ area, pts: orderQuad([[d[0], d[1]], [d[2], d[3]], [d[4], d[5]], [d[6], d[7]]]) });
        }
        approx.delete();
      }
      c.delete();
    }
    contours.delete(); hier.delete(); edges.delete();
  }
  src.delete(); gray.delete(); kernel.delete();
  out.sort((a, b) => b.area - a.area);
  return out.slice(0, 8);
}

// Der gefundene Umriss ist oft der Brettrand mit Koordinaten. Wir suchen den Einzug t,
// bei dem die mittleren 4x4 Felder (in der Grundstellung leer) das klarste Schachmuster zeigen.
function refineGrid(img, quad) {
  const Hq = homography([[0, 0], [1, 0], [1, 1], [0, 1]], quad);
  if (!Hq) return null;
  const results = [];
  for (let t = 0; t <= 0.16; t += 0.005) {
    const sz = (1 - 2 * t) / 8;
    const sums = [0, 0], sds = [];
    for (let i = 2; i < 6; i++) for (let j = 2; j < 6; j++) {
      const st = cellStats(img, Hq, t + i * sz, t + j * sz, sz, 0.9, 8);
      sums[(i + j) % 2] += st.m; sds.push(st.sd);
    }
    const contrast = Math.abs(sums[0] - sums[1]) / 8;
    const noise = sds.reduce((a, b) => a + b, 0) / sds.length;
    const score = contrast - 1.2 * noise;
    results.push({ t, score, contrast, darkParity: sums[0] < sums[1] ? 0 : 1 });
  }
  // Bei sauberen Bildern gibt es ein Plateau gleich guter Werte: dessen Mitte nehmen.
  const top = Math.max(...results.map((r) => r.score));
  const plateau = results.filter((r) => r.score >= top - 1);
  const best = plateau[Math.floor(plateau.length / 2)];
  const t = best.t;
  best.grid = [[t, t], [1 - t, t], [1 - t, 1 - t], [t, 1 - t]].map(([u, v]) => project(Hq, u, v));
  return best;
}

// a1 ist ein dunkles Eckfeld; Weiß steht auf der Seite mit den helleren Figuren.
function guessOrientation(img, grid, darkParity) {
  const Hg = homography([[0, 0], [8, 0], [8, 8], [0, 8]], grid);
  const band = (fn) => {
    let s = 0, k = 0;
    for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) if (fn(i, j)) { s += cellStats(img, Hg, i, j, 1, 0.5, 4).m; k++; }
    return s / k;
  };
  if (darkParity === 1) {
    // dunkle Ecken unten links / oben rechts: Weiß unten (a1 = BL) oder oben (a1 = TR)
    return band((i, j) => j >= 6) > band((i, j) => j <= 1) ? 3 : 1;
  }
  // dunkle Ecken oben links / unten rechts: Weiß links (a1 = TL) oder rechts (a1 = BR)
  return band((i) => i <= 1) > band((i) => i >= 6) ? 0 : 2;
}

function searchStep(img) {
  if (!S.cv) { setStatus('Lade Bilderkennung …', 'search'); return; }
  let found = null;
  for (const q of findQuads(img)) {
    const r = refineGrid(img, q.pts);
    if (r && r.contrast > 25 && r.score > 10 && (!found || r.score > found.score)) found = r;
  }
  if (!found) { S.cand = null; S.candHits = 0; setStatus('Suche Schachbrett … ganzes Brett ins Bild, Kamera ruhig halten', 'search'); return; }
  const close = S.cand && found.grid.every((p, i) => Math.hypot(p[0] - S.cand.grid[i][0], p[1] - S.cand.grid[i][1]) < img.width * 0.025);
  S.candHits = close ? S.candHits + 1 : 1;
  S.cand = found;
  setStatus('Schachbrett gefunden … halte still', 'search');
  if (S.candHits >= 4) {
    setBoard(found.grid, guessOrientation(img, found.grid, found.darkParity));
    enterTrack();
  }
}

/* ---------- Züge erkennen ---------- */

const N = 5; // Stichproben pro Feld und Richtung
function features(img) {
  const f = new Float32Array(64 * N * N * 3);
  let k = 0;
  for (let sq = 0; sq < 64; sq++) {
    const file = sq % 8, rank = sq >> 3;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const [x, y] = project(S.H, file + 0.2 + (0.6 * (i + 0.5)) / N, rank + 0.2 + (0.6 * (j + 0.5)) / N);
      const xi = Math.min(img.width - 1, Math.max(0, x | 0)), yi = Math.min(img.height - 1, Math.max(0, y | 0));
      const p = (yi * img.width + xi) * 4;
      f[k++] = img.data[p]; f[k++] = img.data[p + 1]; f[k++] = img.data[p + 2];
    }
  }
  return f;
}

function squareDiffs(a, b) {
  const per = N * N * 3, d = new Float32Array(64);
  for (let sq = 0; sq < 64; sq++) {
    let s = 0;
    for (let k = sq * per, e = k + per; k < e; k++) s += Math.abs(a[k] - b[k]);
    d[sq] = s / per;
  }
  return d;
}

const sqIndex = (name) => FILES.indexOf(name[0]) + 8 * (Number(name[1]) - 1);
const sqName = (i) => FILES[i % 8] + ((i >> 3) + 1);

function involvedSquares(m) {
  const s = [m.from, m.to];
  const r = m.from[1];
  if (m.flags.includes('k')) s.push('h' + r, 'f' + r);
  if (m.flags.includes('q')) s.push('a' + r, 'd' + r);
  if (m.flags.includes('e')) s.push(m.to[0] + m.from[1]);
  return s.map(sqIndex);
}

function matchMove(d) {
  const sorted = [...d].sort((a, b) => a - b);
  const thr = Math.max(14, sorted[32] * 3 + 6);
  const changed = [];
  d.forEach((v, i) => { if (v > thr) changed.push(i); });
  if (!changed.length) return { kind: 'none' };
  if (changed.length > 22) return { kind: 'camera' };
  let best = null, second = null;
  for (const m of S.game.moves({ verbose: true })) {
    if (m.promotion && m.promotion !== 'q') continue;
    const inv = involvedSquares(m);
    if (Math.min(d[inv[0]], d[inv[1]]) < thr * 0.6) continue;
    let score = inv.reduce((a, i) => a + d[i], 0);
    for (const c of changed) if (!inv.includes(c)) score -= 1.5 * d[c];
    const cand = { m, score };
    if (!best || score > best.score) { second = best; best = cand; } else if (!second || score > second.score) second = cand;
  }
  if (!best || best.score <= 0) return { kind: 'unclear', changed };
  if (second && best.score - second.score < thr * 0.4) return { kind: 'unclear', changed };
  return { kind: 'move', move: best.m };
}

function trackStep(img) {
  const cur = features(img);
  if (DEBUG) drawDebug(img);
  if (!S.prev) { S.prev = cur; return; }
  const motion = squareDiffs(cur, S.prev);
  S.prev = cur;
  const moving = Math.max(...motion) > 12;
  if (moving) {
    S.stable = 0; S.evaluated = false;
    setStatus('Bewegung erkannt …', 'busy');
    return;
  }
  S.stable++;
  if (S.stable < 4 || S.evaluated) return;
  S.evaluated = true;
  if (!S.ref) { S.ref = cur; idleStatus(); return; }

  const res = matchMove(squareDiffs(cur, S.ref));
  if (res.kind === 'none') { S.ref = cur; if (!S.unclear) idleStatus(); return; }
  if (res.kind === 'camera') {
    setStatus('Kamera hat sich bewegt – suche Brett neu', 'warn');
    startSearch();
    return;
  }
  if (res.kind === 'unclear') {
    S.unclear = true;
    setStatus('Zug nicht eindeutig – tippe Start- und Zielfeld an', 'warn');
    return;
  }
  applyMove(res.move, cur);
}

function applyMove(m, cur) {
  const played = S.game.move({ from: m.from, to: m.to, promotion: 'q' });
  if (!played) return;
  S.lastMove = played;
  S.ref = cur || null;
  S.unclear = false;
  S.tapFrom = null;
  S.best = null;
  if (!cur) { S.stable = 0; S.evaluated = false; } // neue Referenz beim nächsten ruhigen Bild
  vibrate();
  idleStatus();
  renderMini();
  requestSuggestion();
}

/* ---------- Stockfish ---------- */

const engine = { worker: null, ready: false, busy: false, fen: null, next: null, info: null };

function startEngine() {
  try {
    engine.worker = new Worker('engine/stockfish.js');
  } catch (e) {
    console.error(e);
    return;
  }
  engine.worker.onmessage = (e) => {
    const line = String(e.data);
    if (line === 'readyok') { engine.ready = true; if (engine.next) go(engine.next); }
    else if (line.startsWith('info') && line.includes(' pv ')) engine.info = line;
    else if (line.startsWith('bestmove')) onBestMove(line.split(' ')[1]);
  };
  engine.worker.onerror = (e) => console.error('Stockfish', e);
  engine.worker.postMessage('uci');
  engine.worker.postMessage('isready');
}

function go(fen) {
  if (!engine.ready) { engine.next = fen; return; }
  if (engine.busy) { engine.next = fen; engine.worker.postMessage('stop'); return; }
  engine.next = null; engine.busy = true; engine.fen = fen; engine.info = null;
  engine.worker.postMessage('position fen ' + fen);
  engine.worker.postMessage('go movetime ' + S.thinkMs);
}

function onBestMove(uci) {
  engine.busy = false;
  if (engine.next) { go(engine.next); return; }
  if (engine.fen !== S.game.fen() || !uci || uci === '(none)') return;
  const probe = new Chess(engine.fen);
  const m = probe.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] || 'q' });
  if (!m) return;
  S.best = { from: m.from, to: m.to, san: m.san, piece: m.piece, color: m.color, promotion: m.promotion };
  S.evalText = describeEval(engine.info, probe.turn() === 'w' ? 'b' : 'w');
  renderSuggestion();
  renderMini();
  if (!S.unclear) idleStatus();
}

function describeEval(info, side) {
  if (!info) return '';
  const mate = info.match(/score mate (-?\d+)/);
  const cp = info.match(/score cp (-?\d+)/);
  const sign = side === 'w' ? 1 : -1;
  if (mate) {
    const n = Number(mate[1]) * sign;
    return n > 0 ? `Weiß setzt matt in ${Math.abs(n)}` : `Schwarz setzt matt in ${Math.abs(n)}`;
  }
  if (cp) {
    const v = (Number(cp[1]) * sign) / 100;
    return `Bewertung ${v > 0 ? '+' : ''}${v.toFixed(1).replace('.', ',')} (aus Sicht von Weiß)`;
  }
  return '';
}

function requestSuggestion() {
  S.best = null;
  const g = S.game;
  if (g.isGameOver()) {
    S.best = null;
    renderSuggestion(g.isCheckmate() ? `Schachmatt – ${g.turn() === 'w' ? 'Schwarz' : 'Weiß'} gewinnt` : 'Remis');
    return;
  }
  if (S.myColor !== 'both' && g.turn() !== S.myColor) {
    renderSuggestion('Gegner ist am Zug');
    return;
  }
  renderSuggestion('Stockfish rechnet …');
  go(g.fen());
}

/* ---------- Anzeige ---------- */

function setStatus(text, tone = 'idle') {
  $('#statusText').textContent = text;
  $('#status').dataset.tone = tone;
}

function idleStatus() {
  const turn = S.game.turn() === 'w' ? 'Weiß' : 'Schwarz';
  setStatus(`Brett erkannt · ${turn} am Zug`, 'ok');
}

function renderSuggestion(placeholder) {
  const label = $('#sugLabel'), move = $('#sugMove'), meta = $('#sugMeta');
  const turn = S.game.turn() === 'w' ? 'Weiß' : 'Schwarz';
  if (S.best) {
    const b = S.best;
    label.textContent = `Empfehlung für ${turn}`;
    let text = `${PIECE_DE[b.piece]} ${b.from} → ${b.to}`;
    if (b.san.startsWith('O-O-O')) text = 'Lange Rochade';
    else if (b.san.startsWith('O-O')) text = 'Kurze Rochade';
    if (b.promotion) text += ' (Umwandlung in Dame)';
    move.textContent = text;
    meta.textContent = [b.san, S.evalText].filter(Boolean).join(' · ');
  } else {
    label.textContent = `${turn} am Zug`;
    move.textContent = placeholder || '–';
    meta.textContent = S.lastMove ? `Letzter Zug: ${S.lastMove.san}` : '';
  }
}

function renderMini() {
  const el = $('#mini');
  const flip = S.myColor === 'b';
  const board = S.game.board(); // board[0] = Reihe 8
  const hl = new Set(S.best ? [S.best.from, S.best.to] : []);
  let html = '';
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
    const row = flip ? 7 - r : r, col = flip ? 7 - c : c;
    const p = board[row][col];
    const name = FILES[col] + (8 - row);
    const dark = (row + col) % 2 === 1;
    const cls = [dark ? 'd' : 'l', hl.has(name) ? 'hl' : '', p && p.color === 'w' ? 'w' : ''].join(' ');
    html += `<div class="${cls}">${p ? GLYPH[p.type] : ''}</div>`;
  }
  el.innerHTML = html;
}

function vibrate() { try { navigator.vibrate?.(30); } catch { /* iOS kennt das nicht */ } }

// Abbildung work-Koordinaten -> Bildschirm (Video ist mit object-fit: cover skaliert).
function viewMap() {
  const cw = overlay.clientWidth, ch = overlay.clientHeight;
  const ww = work.width || 1, wh = work.height || 1;
  const s = Math.max(cw / ww, ch / wh);
  return { s, ox: (cw - ww * s) / 2, oy: (ch - wh * s) / 2 };
}
const toScreen = (p, vm) => [vm.ox + p[0] * vm.s, vm.oy + p[1] * vm.s];
const toWork = (x, y, vm) => [(x - vm.ox) / vm.s, (y - vm.oy) / vm.s];
const boardToScreen = (u, v, vm) => toScreen(project(S.H, u, v), vm);

function polyPath(pts) {
  octx.beginPath();
  pts.forEach(([x, y], i) => (i ? octx.lineTo(x, y) : octx.moveTo(x, y)));
  octx.closePath();
}

function squarePoly(name, vm, inset = 0) {
  const f = FILES.indexOf(name[0]), r = Number(name[1]) - 1;
  return [[f + inset, r + inset], [f + 1 - inset, r + inset], [f + 1 - inset, r + 1 - inset], [f + inset, r + 1 - inset]].map(([u, v]) => boardToScreen(u, v, vm));
}

function draw(t) {
  requestAnimationFrame(draw);
  const dpr = window.devicePixelRatio || 1;
  const cw = overlay.clientWidth, ch = overlay.clientHeight;
  if (overlay.width !== Math.round(cw * dpr)) { overlay.width = Math.round(cw * dpr); overlay.height = Math.round(ch * dpr); }
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  octx.clearRect(0, 0, cw, ch);
  const vm = viewMap();
  const css = getComputedStyle(document.documentElement);
  const cBoard = css.getPropertyValue('--board').trim(), cMove = css.getPropertyValue('--move').trim();

  if (S.mode === 'search' && S.cand) {
    octx.setLineDash([8, 6]);
    octx.lineWidth = 2; octx.strokeStyle = cBoard;
    polyPath(S.cand.grid.map((p) => toScreen(p, vm)));
    octx.stroke();
    octx.setLineDash([]);
  }

  if (S.mode === 'manual') {
    octx.fillStyle = cBoard;
    S.manualPts.forEach((p) => { const [x, y] = toScreen(p, vm); octx.beginPath(); octx.arc(x, y, 8, 0, 7); octx.fill(); });
  }

  if ((S.mode === 'track' || S.mode === 'adjust') && S.H) {
    // Brett umranden: leuchtende Kontur mit umlaufendem Lichtstreifen
    const outline = [[0, 0], [8, 0], [8, 8], [0, 8]].map(([u, v]) => boardToScreen(u, v, vm));
    octx.save();
    octx.shadowColor = cBoard; octx.shadowBlur = 16;
    octx.lineWidth = 3; octx.strokeStyle = cBoard; octx.globalAlpha = 0.75;
    polyPath(outline); octx.stroke();
    octx.globalAlpha = 1; octx.lineWidth = 4; octx.setLineDash([40, 400]); octx.lineDashOffset = -(t / 8) % 440;
    octx.strokeStyle = '#ffffff'; polyPath(outline); octx.stroke();
    octx.restore();

    // feines Raster
    octx.lineWidth = 1; octx.strokeStyle = 'rgba(127,211,255,0.22)';
    octx.beginPath();
    for (let i = 1; i < 8; i++) {
      let [x, y] = boardToScreen(i, 0, vm); octx.moveTo(x, y); [x, y] = boardToScreen(i, 8, vm); octx.lineTo(x, y);
      [x, y] = boardToScreen(0, i, vm); octx.moveTo(x, y); [x, y] = boardToScreen(8, i, vm); octx.lineTo(x, y);
    }
    octx.stroke();

    // Ecke a1 markieren, damit man die Ausrichtung prüfen kann
    const [ax, ay] = boardToScreen(0.5, 0.5, vm);
    octx.fillStyle = 'rgba(13,16,20,.7)'; octx.beginPath(); octx.arc(ax, ay, 11, 0, 7); octx.fill();
    octx.fillStyle = cBoard; octx.font = '600 11px -apple-system, sans-serif'; octx.textAlign = 'center'; octx.textBaseline = 'middle';
    octx.fillText('a1', ax, ay);

    if (S.lastMove && !S.best) {
      octx.fillStyle = 'rgba(127,211,255,0.18)';
      [S.lastMove.from, S.lastMove.to].forEach((n) => { polyPath(squarePoly(n, vm, 0.04)); octx.fill(); });
    }
    if (S.tapFrom) {
      octx.fillStyle = 'rgba(127,211,255,0.45)';
      polyPath(squarePoly(S.tapFrom, vm, 0.04)); octx.fill();
    }
    if (S.best) drawBest(t, vm, cMove);

    if (S.mode === 'adjust') {
      S.quad.forEach((p) => {
        const [x, y] = toScreen(p, vm);
        octx.fillStyle = 'rgba(13,16,20,.6)'; octx.beginPath(); octx.arc(x, y, 18, 0, 7); octx.fill();
        octx.strokeStyle = cBoard; octx.lineWidth = 3; octx.stroke();
      });
    }
  }
}

function drawBest(t, vm, color) {
  const b = S.best;
  const pulse = 0.5 + 0.5 * Math.sin(t / 260);
  octx.save();
  octx.fillStyle = color;
  octx.globalAlpha = 0.28; polyPath(squarePoly(b.from, vm, 0.04)); octx.fill();
  octx.globalAlpha = 0.35 + 0.3 * pulse; polyPath(squarePoly(b.to, vm, 0.04)); octx.fill();
  octx.globalAlpha = 1; octx.lineWidth = 2; octx.strokeStyle = color;
  polyPath(squarePoly(b.to, vm, 0.04)); octx.stroke();

  // Pfeil liegt in der Brettebene; Springer laufen über die L-Form.
  const f = [FILES.indexOf(b.from[0]) + 0.5, Number(b.from[1]) - 0.5];
  const to = [FILES.indexOf(b.to[0]) + 0.5, Number(b.to[1]) - 0.5];
  let path = [f, to];
  if (b.piece === 'n') {
    const dx = to[0] - f[0], dy = to[1] - f[1];
    path = Math.abs(dx) > Math.abs(dy) ? [f, [to[0], f[1]], to] : [f, [f[0], to[1]], to];
  }
  const pts = path.map(([u, v]) => boardToScreen(u, v, vm));
  const sqPx = Math.hypot(...[0, 1].map((k) => boardToScreen(to[0] + 0.5, to[1], vm)[k] - boardToScreen(to[0] - 0.5, to[1], vm)[k]));
  const w = Math.max(5, sqPx * 0.16);
  // Ende etwas vor der Feldmitte, damit die Spitze sitzt
  const [px, py] = pts[pts.length - 2], [ex, ey] = pts[pts.length - 1];
  const ang = Math.atan2(ey - py, ex - px);
  const head = w * 2.4;
  const tip = [ex, ey], base = [ex - Math.cos(ang) * head, ey - Math.sin(ang) * head];
  octx.shadowColor = 'rgba(0,0,0,.55)'; octx.shadowBlur = 8;
  octx.strokeStyle = color; octx.lineWidth = w; octx.lineCap = 'round'; octx.lineJoin = 'round';
  octx.beginPath();
  pts.slice(0, -1).forEach(([x, y], i) => (i ? octx.lineTo(x, y) : octx.moveTo(x, y)));
  octx.lineTo(base[0], base[1]);
  octx.stroke();
  octx.fillStyle = color;
  octx.beginPath();
  octx.moveTo(tip[0], tip[1]);
  octx.lineTo(base[0] + Math.cos(ang + Math.PI / 2) * w * 1.3, base[1] + Math.sin(ang + Math.PI / 2) * w * 1.3);
  octx.lineTo(base[0] + Math.cos(ang - Math.PI / 2) * w * 1.3, base[1] + Math.sin(ang - Math.PI / 2) * w * 1.3);
  octx.closePath(); octx.fill();

  // Schwebende Markierung über dem Zielfeld (Lichtsäule + Raute mit Figur)
  const bob = Math.sin(t / 400) * sqPx * 0.08;
  const topY = ey - sqPx * 1.5 - bob;
  const grad = octx.createLinearGradient(ex, ey, ex, topY);
  grad.addColorStop(0, 'rgba(245,196,81,0.0)'); grad.addColorStop(1, 'rgba(245,196,81,0.55)');
  octx.shadowBlur = 0;
  octx.fillStyle = grad;
  octx.beginPath(); octx.moveTo(ex - sqPx * 0.12, ey); octx.lineTo(ex + sqPx * 0.12, ey); octx.lineTo(ex + sqPx * 0.04, topY); octx.lineTo(ex - sqPx * 0.04, topY); octx.closePath(); octx.fill();
  const r = Math.max(16, sqPx * 0.38);
  octx.shadowColor = 'rgba(0,0,0,.6)'; octx.shadowBlur = 10;
  octx.fillStyle = color;
  octx.beginPath(); octx.moveTo(ex, topY - r); octx.lineTo(ex + r * 0.85, topY); octx.lineTo(ex, topY + r); octx.lineTo(ex - r * 0.85, topY); octx.closePath(); octx.fill();
  octx.shadowBlur = 0;
  octx.fillStyle = '#1b1405'; octx.font = `${Math.round(r * 0.95)}px -apple-system, "Segoe UI Symbol", sans-serif`;
  octx.textAlign = 'center'; octx.textBaseline = 'middle';
  octx.fillText(GLYPH[b.piece], ex, topY + r * 0.05);
  octx.restore();
}

function drawDebug(img) {
  const c = $('#debug'); c.hidden = false;
  const x = c.getContext('2d'); const id = x.createImageData(160, 160);
  for (let py = 0; py < 160; py++) for (let px = 0; px < 160; px++) {
    const [wx, wy] = project(S.H, px / 20, 8 - py / 20);
    const xi = Math.min(img.width - 1, Math.max(0, wx | 0)), yi = Math.min(img.height - 1, Math.max(0, wy | 0));
    const s = (yi * img.width + xi) * 4, d = (py * 160 + px) * 4;
    id.data[d] = img.data[s]; id.data[d + 1] = img.data[s + 1]; id.data[d + 2] = img.data[s + 2]; id.data[d + 3] = 255;
  }
  x.putImageData(id, 0, 0);
}

/* ---------- Ablauf ---------- */

function loop(t) {
  requestAnimationFrame(loop);
  const interval = S.mode === 'search' ? 250 : 180;
  if (t - S.lastTick < interval) return;
  S.lastTick = t;
  if (S.mode !== 'search' && S.mode !== 'track') return;
  const img = grab();
  if (!img) return;
  try {
    if (S.mode === 'search') searchStep(img);
    else trackStep(img);
  } catch (e) { console.error(e); }
}

function startSearch() {
  S.mode = 'search'; S.cand = null; S.candHits = 0; S.ref = null; S.prev = null;
  setStatus('Suche Schachbrett …', 'search');
}

function enterTrack() {
  S.mode = 'track'; S.ref = null; S.prev = null; S.stable = 0; S.evaluated = false; S.unclear = false;
  $('#panel').hidden = false; $('#mini').hidden = false;
  idleStatus(); renderMini();
  requestSuggestion();
}

async function startCamera(deviceId) {
  S.stream?.getTracks().forEach((t) => t.stop());
  const video0 = deviceId
    ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
    : { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } };
  S.stream = await navigator.mediaDevices.getUserMedia({ video: video0, audio: false });
  video.srcObject = S.stream;
  await video.play().catch(() => {});
  await listCameras();
  try { await navigator.wakeLock?.request('screen'); } catch { /* optional */ }
}

async function listCameras() {
  const sel = $('#camSelect');
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
  const current = S.stream?.getVideoTracks()[0]?.getSettings().deviceId;
  sel.innerHTML = devs.map((d, i) => `<option value="${d.deviceId}" ${d.deviceId === current ? 'selected' : ''}>${d.label || 'Kamera ' + (i + 1)}</option>`).join('');
}

function waitForCv() {
  return new Promise((resolve) => {
    const check = () => {
      const cv = window.cv;
      if (cv && cv.Mat) return resolve({ cv });
      if (cv && typeof cv.then === 'function') { cv.then((m) => { window.cv = m; resolve({ cv: m }); }); return; }
      if (cv && !cv.Mat && !cv.__hooked) { cv.__hooked = true; cv.onRuntimeInitialized = () => resolve({ cv }); }
      setTimeout(check, 150);
    };
    check();
  });
}

/* ---------- Bedienung ---------- */

overlay.addEventListener('pointerdown', (e) => {
  const vm = viewMap();
  const p = toWork(e.offsetX, e.offsetY, vm);
  if (S.mode === 'manual') {
    S.manualPts.push(p);
    setStatus(`Ecke ${S.manualPts.length} von 4 gesetzt`, 'search');
    if (S.manualPts.length === 4) {
      const quad = orderQuad(S.manualPts);
      const img = grab();
      const r = img ? refineGrid(img, quad) : null;
      // Bei manuellen Ecken keinen Einzug, nur Ausrichtung schätzen
      setBoard(quad, img && r ? guessOrientation(img, quad, r.darkParity) : 3);
      S.manualPts = [];
      enterTrack();
    }
    return;
  }
  if (S.mode === 'adjust') {
    S.drag = S.quad.findIndex((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) * vm.s < 40);
    if (S.drag >= 0) overlay.setPointerCapture(e.pointerId);
    return;
  }
  if (S.mode === 'track' && S.Hinv) {
    const [u, v] = project(S.Hinv, p[0], p[1]);
    if (u < 0 || v < 0 || u >= 8 || v >= 8) { S.tapFrom = null; return; }
    const name = FILES[Math.floor(u)] + (Math.floor(v) + 1);
    const piece = S.game.get(name);
    if (!S.tapFrom || (piece && piece.color === S.game.turn())) {
      S.tapFrom = piece && piece.color === S.game.turn() ? name : null;
      return;
    }
    const legal = S.game.moves({ square: S.tapFrom, verbose: true }).find((m) => m.to === name);
    if (legal) applyMove(legal, null);
    else S.tapFrom = null;
  }
});
overlay.addEventListener('pointermove', (e) => {
  if (S.mode !== 'adjust' || S.drag < 0) return;
  S.quad[S.drag] = toWork(e.offsetX, e.offsetY, viewMap());
  setBoard(S.quad, S.a1);
});
overlay.addEventListener('pointerup', () => { S.drag = -1; });

$('#btnStart').addEventListener('click', async () => {
  S.myColor = $('#introColor').value;
  $('#colorSelect').value = S.myColor;
  $('#introMsg').textContent = 'Kamera wird gestartet …';
  try {
    await startCamera();
  } catch (e) {
    $('#introMsg').textContent = window.isSecureContext
      ? 'Kein Kamerazugriff. Erlaube die Kamera in den Browser-Einstellungen und lade die Seite neu.'
      : 'Die Kamera funktioniert nur über https:// – öffne die App über lars-mst.github.io/schach-kamera.';
    return;
  }
  $('#intro').hidden = true;
  startSearch();
});

$('#btnSettings').addEventListener('click', () => { $('#fenInput').value = S.game.fen(); $('#settings').hidden = false; });
$('#btnClose').addEventListener('click', () => { $('#settings').hidden = true; });
$('#settings').addEventListener('click', (e) => { if (e.target.id === 'settings') $('#settings').hidden = true; });
$('#camSelect').addEventListener('change', async (e) => { await startCamera(e.target.value); startSearch(); });
$('#colorSelect').addEventListener('change', (e) => { S.myColor = e.target.value; renderMini(); requestSuggestion(); });
$('#thinkSelect').addEventListener('change', (e) => { S.thinkMs = Number(e.target.value); });
$('#btnFen').addEventListener('click', () => {
  try {
    S.game = new Chess($('#fenInput').value.trim());
  } catch {
    $('#fenInput').setCustomValidity('Ungültige FEN'); $('#fenInput').reportValidity(); return;
  }
  $('#fenInput').setCustomValidity('');
  S.lastMove = null; S.ref = null; S.stable = 0; S.evaluated = false; S.unclear = false;
  renderMini(); requestSuggestion(); $('#settings').hidden = true;
});
$('#btnNew').addEventListener('click', () => {
  S.game = new Chess(); S.lastMove = null; S.ref = null; S.stable = 0; S.evaluated = false; S.unclear = false;
  renderMini(); requestSuggestion(); $('#settings').hidden = true;
});
$('#btnManual').addEventListener('click', () => {
  $('#settings').hidden = true;
  if (!S.stream) return;
  S.mode = 'manual'; S.manualPts = [];
  setStatus('Tippe die 4 Ecken der Spielfläche an', 'search');
});
$('#btnUndo').addEventListener('click', () => {
  if (!S.game.undo()) return;
  const h = S.game.history({ verbose: true });
  S.lastMove = h[h.length - 1] || null;
  S.ref = null; S.stable = 0; S.evaluated = false; S.unclear = false;
  renderMini(); requestSuggestion();
});
$('#btnRotate').addEventListener('click', () => {
  if (!S.quad) return;
  setBoard(S.quad, (S.a1 + 1) % 4);
  S.ref = null; S.prev = null; S.stable = 0; S.evaluated = false;
});
$('#btnAdjust').addEventListener('click', (e) => {
  if (!S.quad) return;
  if (S.mode === 'adjust') {
    S.mode = 'track'; e.target.classList.remove('on'); e.target.textContent = 'Ecken anpassen';
    S.ref = null; S.prev = null; S.stable = 0; S.evaluated = false; idleStatus();
  } else {
    S.mode = 'adjust'; e.target.classList.add('on'); e.target.textContent = 'Fertig';
    setStatus('Ziehe die Ecken auf die Ecken der Spielfläche', 'search');
  }
});
$('#btnRedetect').addEventListener('click', () => { S.best = null; startSearch(); });

window.addEventListener('pagehide', () => S.stream?.getTracks().forEach((t) => t.stop()));

renderSuggestion();
startEngine();
waitForCv().then(({ cv }) => { S.cv = cv; });
requestAnimationFrame(loop);
requestAnimationFrame(draw);
if (DEBUG) window.__dbg = { findQuads, refineGrid, grab, guessOrientation };
