// Schach Kamera: Brett finden (OpenCV), Züge über Feld-Veränderungen erkennen (chess.js),
// besten Zug berechnen (Stockfish WASM) und perspektivisch korrekt ins Kamerabild zeichnen.
import { Chess } from './lib/chess.js';
import { CLASSES, ColorModel, TemplateStore, classMap, colorFeatures, cropPatch, loadModel, saveModel, forgetModel, assignTypes, toFen, plausibility } from './vision.js';

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
  thinkMs: 800,
  ref: null,             // Feld-Merkmale der zuletzt bestätigten Stellung
  prev: null,
  stable: 0,
  evaluated: false,
  unclear: false,
  cand: null, candHits: 0,
  best: null,            // { uci, from, to, san, piece, color, final }
  evalText: '',
  lastMove: null,
  tapFrom: null,
  manualPts: [],
  drag: -1,
  cv: null,
  lastTick: 0,
  tick: 0,
  fresh: true,           // noch keine Partie begonnen: beim Finden des Bretts Stellung bestimmen
  outerT: null,          // Einzug des Rasters im gefundenen Brettumriss (für das Nachführen)
  camMoving: false,
  model: null,           // Farbmodell leer/weiß/schwarz
  templates: null,       // Figurenvorlagen
  needLearn: false,      // nächste ruhige, passende Aufnahme zum Lernen nutzen
  mismatch: [],          // Felder, die nicht zur Stellung passen
  badStreak: 0,
  inconsistent: false,
  notice: null,          // { text, until }
  scan: null,            // offene Scan-Bestätigung
  searchKey: null,       // zuletzt erfolglos durchsuchte Abweichung
};
{
  const m = loadModel();
  S.model = m?.color || new ColorModel();
  S.templates = m?.templates || new TemplateStore();
}
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

const BOARD_CORNERS = [[0, 0], [0, 8], [8, 8], [8, 0]];
// Im Uhrzeigersinn ab a1 folgen im Bild a8, h8, h1.
const cornersFrom = (quad, a1) => [0, 1, 2, 3].map((i) => quad[(a1 + i) % 4]);

function setBoard(quad, a1) {
  S.quad = quad;
  S.a1 = a1;
  S.H = homography(BOARD_CORNERS, cornersFrom(quad, a1));
  S.Hinv = homography(cornersFrom(quad, a1), BOARD_CORNERS);
}

// Für jede der 4 möglichen Ausrichtungen: welches Feld der aktuellen Beschriftung entspricht Feld s?
function orientationPerms() {
  const perms = [];
  for (let o = 0; o < 4; o++) {
    const Ho = homography(BOARD_CORNERS, cornersFrom(S.quad, o));
    const perm = new Int8Array(64);
    for (let s = 0; s < 64; s++) {
      const [x, y] = project(Ho, (s % 8) + 0.5, (s >> 3) + 0.5);
      const [u, v] = project(S.Hinv, x, y);
      perm[s] = Math.min(7, Math.max(0, Math.floor(u))) + 8 * Math.min(7, Math.max(0, Math.floor(v)));
    }
    perms.push({ o, perm });
  }
  return perms;
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
function refineGrid(img, quad, fixedT = null) {
  const Hq = homography([[0, 0], [1, 0], [1, 1], [0, 1]], quad);
  if (!Hq) return null;
  const results = [];
  const ts = [];
  if (fixedT === null) for (let t = 0; t <= 0.16; t += 0.005) ts.push(t);
  else ts.push(Math.max(0, fixedT - 0.005), fixedT, fixedT + 0.005);
  for (const t of ts) {
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
// Liegt die Schätzung daneben, korrigiert die Zugerkennung sie beim ersten Zug.
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
  if (!found) { S.cand = null; S.candHits = 0; setStatus('Suche Schachbrett … ganzes Brett ins Bild', 'search'); return; }
  const close = S.cand && found.grid.every((p, i) => Math.hypot(p[0] - S.cand.grid[i][0], p[1] - S.cand.grid[i][1]) < img.width * 0.025);
  S.candHits = close ? S.candHits + 1 : 1;
  S.cand = found;
  setStatus('Schachbrett gefunden … halte still', 'search');
  if (S.candHits >= 3) onBoardLocked(img, found);
}

// Brett ist gefunden: Ausrichtung und Stellung festlegen
function onBoardLocked(img, found) {
  if (!S.fresh && S.quad) {
    // Brett nach Kamerabewegung wiedergefunden: Ausrichtung beibehalten, Partie weiterführen
    setBoard(alignQuad(found.grid, S.quad), S.a1);
    S.outerT = found.t;
    enterTrack();
    return;
  }
  S.fresh = false;
  S.outerT = found.t;
  setBoard(found.grid, guessOrientation(img, found.grid, found.darkParity));
  if (S.model.ready && S.templates.ready) {
    const scan = scanPosition(img);
    if (scan) {
      setBoard(S.quad, scan.o);
      enterTrack();
      if (toFen(scan.pieces, 'w').split(' ')[0] === new Chess().fen().split(' ')[0]) { S.game = new Chess(); S.needLearn = true; requestSuggestion(); }
      else openScanDialog(scan);
      return;
    }
  }
  enterTrack();
  if (!startLooksRight(img)) setStatus('Keine Grundstellung erkannt – Grundstellung aufbauen oder im Zahnrad die Stellung eingeben', 'warn');
}

function startLooksRight(img) {
  const { occ, reliable } = occupancy(squareStats(features(img)), new Chess());
  if (!reliable) return false;
  let ok = 0;
  for (let sq = 0; sq < 64; sq++) if ((sq < 16 || sq >= 48) === occ[sq] > 0.5) ok++;
  return ok >= 54;
}

// Ecken eines neu gefundenen Umrisses in dieselbe Reihenfolge wie die bisherigen bringen
function alignQuad(grid, old) {
  let bestK = 0, bestD = Infinity;
  for (let k = 0; k < 4; k++) {
    let d = 0;
    for (let i = 0; i < 4; i++) d += Math.hypot(grid[(i + k) % 4][0] - old[i][0], grid[(i + k) % 4][1] - old[i][1]);
    if (d < bestD) { bestD = d; bestK = k; }
  }
  return [0, 1, 2, 3].map((i) => grid[(i + bestK) % 4]);
}

// Brett während der Partie nachführen (Kamera bewegt sich). Gibt true zurück, wenn die Lage aktualisiert wurde.
function followBoard(img) {
  let found = null;
  for (const q of findQuads(img)) {
    const r = refineGrid(img, q.pts, S.outerT ?? null);
    if (r && r.contrast > 15 && r.score > 0 && (!found || r.score > found.score)) found = r;
  }
  if (!found) return false;
  const aligned = alignQuad(found.grid, S.quad);
  const shift = Math.max(...aligned.map((p, i) => Math.hypot(p[0] - S.quad[i][0], p[1] - S.quad[i][1])));
  if (shift < img.width * 0.006) return false;
  setBoard(aligned, S.a1);
  return true;
}

/* ---------- Züge erkennen ---------- */

const N = 5; // Stichproben pro Feld und Richtung
const PER = N * N * 3;

function features(img) {
  const f = new Float32Array(64 * PER);
  let k = 0;
  for (let sq = 0; sq < 64; sq++) {
    const file = sq % 8, rank = sq >> 3;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const [x, y] = project(S.H, file + 0.18 + (0.64 * (i + 0.5)) / N, rank + 0.18 + (0.64 * (j + 0.5)) / N);
      const xi = Math.min(img.width - 1, Math.max(0, x | 0)), yi = Math.min(img.height - 1, Math.max(0, y | 0));
      const p = (yi * img.width + xi) * 4;
      f[k++] = img.data[p]; f[k++] = img.data[p + 1]; f[k++] = img.data[p + 2];
    }
  }
  return f;
}

// Mittlere Farbe und Struktur (Streuung) pro Feld
function squareStats(f) {
  const mean = new Float32Array(64 * 3), tex = new Float32Array(64);
  const n = PER / 3;
  for (let sq = 0; sq < 64; sq++) {
    const o = sq * PER;
    let r = 0, g = 0, b = 0;
    for (let k = 0; k < PER; k += 3) { r += f[o + k]; g += f[o + k + 1]; b += f[o + k + 2]; }
    r /= n; g /= n; b /= n;
    let v = 0;
    for (let k = 0; k < PER; k += 3) v += Math.abs(f[o + k] - r) + Math.abs(f[o + k + 1] - g) + Math.abs(f[o + k + 2] - b);
    mean[sq * 3] = r; mean[sq * 3 + 1] = g; mean[sq * 3 + 2] = b;
    tex[sq] = v / n;
  }
  return { mean, tex };
}

const median = (arr) => { const s = Float32Array.from(arr).sort(); return s.length ? s[s.length >> 1] : 0; };

// Unterschied pro Feld, nachdem Helligkeitsschwankungen der Kamera (Belichtung) herausgerechnet sind.
function squareDiffs(a, b) {
  const ma = squareStats(a).mean, mb = squareStats(b).mean;
  const gain = [0, 1, 2].map((c) => {
    const ratios = [];
    for (let sq = 0; sq < 64; sq++) ratios.push((ma[sq * 3 + c] + 1) / (mb[sq * 3 + c] + 1));
    return median(ratios);
  });
  const d = new Float32Array(64);
  for (let sq = 0; sq < 64; sq++) {
    let s = 0;
    for (let k = sq * PER, e = k + PER, c = 0; k < e; k++, c = (c + 1) % 3) s += Math.abs(a[k] - b[k] * gain[c]);
    d[sq] = s / PER;
  }
  return d;
}

const sqIndex = (name) => FILES.indexOf(name[0]) + 8 * (Number(name[1]) - 1);
const isDark = (sq) => ((sq % 8) + (sq >> 3)) % 2 === 0; // a1 ist dunkel

function involvedSquares(m) {
  const s = [[m.from, 0], [m.to, 1]]; // [Feld, nach dem Zug belegt?]
  const r = m.from[1];
  if (m.flags.includes('k')) s.push(['h' + r, 0], ['f' + r, 1]);
  if (m.flags.includes('q')) s.push(['a' + r, 0], ['d' + r, 1]);
  if (m.flags.includes('e')) s.push([m.to[0] + m.from[1], 0]);
  return s.map(([n, occ]) => [sqIndex(n), occ]);
}

// Wie "belegt" sieht jedes Feld aus? Vergleich mit leeren Feldern gleicher Farbe,
// die wir aus der bekannten Stellung kennen. 0 = sieht leer aus, 1 = sieht belegt aus.
function occupancy(stats, game) {
  const board = game.board();
  const occupied = (sq) => !!board[7 - (sq >> 3)][sq % 8];
  const emptyRef = [false, true].map((dark) => {
    const cols = [[], [], []];
    for (let sq = 0; sq < 64; sq++) if (!occupied(sq) && isDark(sq) === dark) for (let c = 0; c < 3; c++) cols[c].push(stats.mean[sq * 3 + c]);
    return cols.map(median);
  });
  const dist = new Float32Array(64);
  for (let sq = 0; sq < 64; sq++) {
    const e = emptyRef[isDark(sq) ? 1 : 0];
    let s = 0;
    for (let c = 0; c < 3; c++) s += Math.abs(stats.mean[sq * 3 + c] - e[c]);
    dist[sq] = s / 3 + stats.tex[sq];
  }
  const emptyD = [], occD = [];
  for (let sq = 0; sq < 64; sq++) (occupied(sq) ? occD : emptyD).push(dist[sq]);
  const De = median(emptyD), Do = median(occD);
  const reliable = Do - De > 8;
  const occ = new Float32Array(64);
  for (let sq = 0; sq < 64; sq++) occ[sq] = reliable ? Math.min(1, Math.max(0, (dist[sq] - De) / (Do - De))) : 0.5;
  return { occ, reliable };
}

// Nachbarfelder (auch zwei Reihen weit), in die eine Figur schräg von der Kamera aus hineinragen kann
const NEIGH = Array.from({ length: 64 }, (_, sq) => {
  const f = sq % 8, r = sq >> 3, out = [];
  for (let df = -1; df <= 1; df++) for (let dr = -2; dr <= 2; dr++) {
    const ff = f + df, rr = r + dr;
    if ((df || dr) && ff >= 0 && ff < 8 && rr >= 0 && rr < 8) out.push(ff + 8 * rr);
  }
  return out;
});

function matchMove(d, stats) {
  const thr = Math.max(12, median(d) * 3 + 5);
  const changed = [];
  d.forEach((v, i) => { if (v > thr) changed.push(i); });
  if (DEBUG && !changed.length) console.log('detect-dbg none thr', thr.toFixed(1), 'top', [...d].map((v, q) => [v, q]).sort((a, b) => b[0] - a[0]).slice(0, 4).map(([v, q]) => FILES[q % 8] + ((q >> 3) + 1) + ':' + (v / thr).toFixed(2)).join(' '));
  const maxD = Math.max(...d);
  // Gar nichts verändert: Referenz darf nachgeführt werden (langsame Lichtänderungen)
  if (maxD < thr * 0.45) return { kind: 'none' };
  if (changed.length > 24) return { kind: 'camera' };
  const { occ, reliable } = occupancy(stats, S.game);
  let best = null, second = null;
  for (const m of S.game.moves({ verbose: true })) {
    if (m.promotion && m.promotion !== 'q') continue;
    const inv = involvedSquares(m);
    if (d[inv[0][0]] < thr * 0.4 || d[inv[1][0]] < thr * 0.3) continue;
    let score = 0;
    const near = new Set();
    for (const [sq, after] of inv) {
      score += Math.min(4, d[sq] / thr);
      if (reliable) score -= 1.2 * Math.abs(occ[sq] - after);
      NEIGH[sq].forEach((n) => near.add(n));
    }
    // Veränderungen außerhalb des Zuges sprechen dagegen; direkt daneben sind es oft nur
    // Figuren, die schräg ins Nachbarfeld ragen.
    for (const c of changed) {
      if (inv.some(([sq]) => sq === c)) continue;
      score -= (near.has(c) ? 0.15 : 0.6) * Math.min(4, d[c] / thr);
    }
    const cand = { m, score };
    if (DEBUG) (matchMove.log ||= []).push([m.san, score.toFixed(2), inv.map(([q]) => FILES[q % 8] + ((q >> 3) + 1) + ':' + (d[q] / thr).toFixed(1) + '/' + occ[q].toFixed(2)).join(' ')]);
    if (!best || score > best.score) { second = best; best = cand; } else if (!second || score > second.score) second = cand;
  }
  if (DEBUG) {
    console.log('detect-dbg thr', thr.toFixed(1), 'reliable', reliable, 'changed', changed.map((q) => FILES[q % 8] + ((q >> 3) + 1) + ':' + (d[q] / thr).toFixed(1)).join(' '));
    (matchMove.log || []).sort((a, b) => b[1] - a[1]).slice(0, 4).forEach((l) => console.log('detect-dbg   ', l.join(' | ')));
    matchMove.log = [];
  }
  const margin = best && second ? best.score - second.score : 9;
  if (best && best.score >= 0.6 && margin >= 0.35) return { kind: 'move', move: best.m, score: best.score, margin, reliable };
  // Nur schwache Veränderung ohne passenden Zug: abwarten, aber Referenz behalten,
  // damit ein langsam sichtbar werdender Zug nicht verloren geht.
  if (!changed.length) return { kind: 'weak' };
  return { kind: 'unclear', changed, score: best?.score, margin };
}

const permute = (arr, perm, width) => {
  const out = new Float32Array(arr.length);
  for (let s = 0; s < 64; s++) for (let c = 0; c < width; c++) out[s * width + c] = arr[perm[s] * width + c];
  return out;
};

function detectMove(cur, ref) {
  const d = squareDiffs(cur, ref);
  const stats = squareStats(cur);
  const res = matchMove(d, stats);
  // Nach den ersten Zügen steht die Ausrichtung fest; dann nie mehr umdrehen
  if (res.kind !== 'unclear' || S.game.history().length >= 6) return res;
  // Passt kein Zug, war vielleicht die Ausrichtung falsch geschätzt: die anderen 3 Drehungen prüfen.
  let best = null;
  for (const { o, perm } of orientationPerms()) {
    if (o === S.a1) continue;
    const r = matchMove(permute(d, perm, 1), { mean: permute(stats.mean, perm, 3), tex: permute(stats.tex, perm, 1) });
    if (r.kind === 'move' && (!best || r.score > best.score)) best = { ...r, orientation: o };
  }
  // Nur bei sehr eindeutigem Treffer umdrehen, sonst lieber nachfragen
  return best && best.reliable && best.score > 2 && best.margin > 0.8 ? best : res;
}

function trackStep(img) {
  S.tick++;
  const cur0 = features(img);
  if (DEBUG) drawDebug(img);
  if (!S.prev) { S.prev = cur0; return; }
  const motion = squareDiffs(cur0, S.prev);
  // Bewegt sich das ganze Bild, hat sich die Kamera bewegt (nicht nur eine Hand über dem Brett)
  const globalMotion = median(motion) > 9;
  if (globalMotion) {
    S.camMoving = true;
    if (S.cv && S.tick % 2 === 0) followBoard(img);
    S.prev = features(img); S.stable = 0; S.evaluated = false; S.ref = null;
    return;
  }
  if (S.camMoving) {
    // Kamera steht wieder: Brett genau nachführen; was in der Zwischenzeit gezogen wurde,
    // findet der Abgleich mit der Stellung.
    S.camMoving = false;
    if (S.cv) followBoard(img);
    S.prev = features(img); S.stable = 0; S.evaluated = false; S.ref = null;
    return;
  }
  const cur = cur0;
  S.prev = cur;
  const moving = Math.max(...motion) > Math.max(10, median(motion) * 3 + 6);
  if (moving) { S.stable = 0; S.evaluated = false; return; }
  S.stable++;
  if (S.stable < 3) return;
  if (S.evaluated) {
    // Passt die Stellung nicht, in Ruhe noch ein paar Mal nachsehen
    if (S.badStreak > 0 && S.stable % 7 === 0) verifyPosition(img, cur);
    return;
  }
  S.evaluated = true;
  if (!S.ref) { S.ref = cur; if (!verifyPosition(img, cur) && !S.inconsistent) idleStatus(); return; }

  const res = detectMove(cur, S.ref);
  if (DEBUG) console.log('detect', res.kind, res.move?.san || '', res.score?.toFixed(2) || '', res.margin?.toFixed(2) || '', res.orientation ?? '');
  if (res.kind === 'none') { S.ref = cur; if (!verifyPosition(img, cur) && !S.unclear && !S.inconsistent) idleStatus(); return; }
  if (res.kind === 'weak') { verifyPosition(img, cur); return; }
  if (res.kind === 'camera') {
    if (!(S.cv && followBoard(img))) { setStatus('Brett verloren – suche neu', 'warn'); startSearch(); }
    S.ref = null; S.prev = null;
    return;
  }
  if (res.kind === 'unclear') {
    if (verifyPosition(img, cur)) return;
    S.unclear = true;
    if (!S.inconsistent) setStatus('Zug nicht erkannt – tippe Start- und Zielfeld an', 'warn');
    return;
  }
  if (res.orientation !== undefined) {
    // Ausrichtung korrigieren; die Referenz gehört zur alten Beschriftung, also neu aufnehmen
    setBoard(S.quad, res.orientation);
    applyMove(res.move, features(img));
    return;
  }
  applyMove(res.move, cur);
}

function applyMove(m, cur) {
  const played = S.game.move({ from: m.from, to: m.to, promotion: 'q' });
  if (!played) return;
  S.lastMove = played;
  afterPositionChange(cur);
}

function afterPositionChange(cur) {
  S.ref = cur || null;
  S.prev = cur || null;
  S.unclear = false;
  S.inconsistent = false;
  S.badStreak = 0;
  S.mismatch = [];
  S.tapFrom = null;
  S.best = null;
  S.needLearn = true;
  S.searchKey = null;
  // Beim nächsten ruhigen Bild die neue Stellung mit dem Bild abgleichen (und daraus lernen)
  S.evaluated = false;
  if (!cur) S.stable = 0;
  vibrate();
  idleStatus();
  requestSuggestion();
}

/* ---------- Abgleich Bild <-> Stellung ---------- */

const CI = { e: 0, w: 1, b: 2 };

function cloneGame(game) {
  const g = new Chess();
  try { g.loadPgn(game.pgn()); return g; } catch { return new Chess(game.fen()); }
}

// Prüft, ob das Bild zur verfolgten Stellung passt. Passt es nicht, sucht es die Zugfolge
// (bis zu 3 Halbzüge, auch mit Korrektur des letzten Zuges), die das Bild am besten erklärt.
// Gibt true zurück, wenn die Stellung dabei geändert wurde.
function verifyPosition(img, cur) {
  if (!S.model.ready) {
    if (S.needLearn && S.game.history().length) learn(img, cur);
    return false;
  }
  const feats = colorFeatures(cur, PER);
  const ll = S.model.logLik(feats);
  const cls = classMap(S.game);
  const mism = [];
  for (let sq = 0; sq < 64; sq++) {
    const row = [ll[sq * 3], ll[sq * 3 + 1], ll[sq * 3 + 2]];
    if (Math.max(...row) - row[CI[cls[sq]]] > 5) mism.push(sq);
  }
  S.mismatch = mism;
  if (!mism.length) {
    S.badStreak = 0;
    if (S.inconsistent) { S.inconsistent = false; idleStatus(); renderSuggestion(); requestSuggestion(); }
    if (S.needLearn) learn(img, cur, feats);
    return false;
  }
  S.badStreak++;
  if (S.badStreak < 2) return false; // einmal abwarten: Hand, Schatten
  // Dieselbe Abweichung nicht immer wieder durchsuchen
  const key = S.game.fen() + '|' + mism.join(',');
  let fix = null;
  // Weichen mehr als 6 Felder ab, ist Neu-Einlesen sinnvoller als eine lange Suche
  if (key !== S.searchKey && mism.length <= 6) {
    const t0 = performance.now();
    fix = searchSequence(ll, 3);
    S.searchKey = fix ? null : key;
    if (DEBUG) console.log('verify mismatch', mism.length, fix ? 'fix ' + (fix.undo ? 'undo+' : '') + fix.seq.map((m) => m.san).join(' ') : 'kein fix', Math.round(performance.now() - t0) + 'ms');
  }
  if (fix) {
    if (fix.undo) S.game.undo();
    for (const m of fix.seq) S.game.move({ from: m.from, to: m.to, promotion: 'q' });
    const h = S.game.history({ verbose: true });
    S.lastMove = h[h.length - 1] || null;
    S.notice = { text: `Erkannt: ${fix.seq.map((m) => m.san).join(', ') || 'letzter Zug zurückgenommen'}`, until: performance.now() + 5000 };
    afterPositionChange(cur);
    return true;
  }
  if (mism.length >= 2 || S.badStreak >= 3) {
    if (!S.inconsistent) {
      S.inconsistent = true;
      renderSuggestion();
    }
    setStatus('Stellung passt nicht zum Bild – hier tippen zum Neu-Einlesen', 'warn');
  }
  return false;
}

function searchSequence(ll, maxPly) {
  const score = (g) => {
    const c = classMap(g);
    let s = 0;
    for (let sq = 0; sq < 64; sq++) s += ll[sq * 3 + CI[c[sq]]];
    return s;
  };
  const base = score(S.game);
  const results = [];
  const roots = [{ g: cloneGame(S.game), undo: 0, pen: 0 }];
  if (S.game.history().length) { const g = cloneGame(S.game); g.undo(); roots.push({ g, undo: 1, pen: 1.5 }); }
  const PLY = 2.5;
  let nodes = 0;
  for (const root of roots) {
    // Nur Züge, die ein Feld berühren, das nicht zur Ausgangsstellung passt: hält die Suche
    // auch bei drei Halbzügen klein genug fürs Handy.
    const c0 = classMap(root.g);
    const odd = new Set();
    for (let sq = 0; sq < 64; sq++) {
      const row = [ll[sq * 3], ll[sq * 3 + 1], ll[sq * 3 + 2]];
      if (Math.max(...row) - row[CI[c0[sq]]] > 2) odd.add(sq);
    }
    const seq = [];
    const dfs = (g, depth) => {
      if (seq.length || root.undo) results.push({ s: score(g) - root.pen - PLY * seq.length, seq: [...seq], undo: root.undo, key: g.fen().split(' ')[0] });
      if (!depth || ++nodes > 800) return;
      for (const m of g.moves({ verbose: true })) {
        if (m.promotion && m.promotion !== 'q') continue;
        if (!involvedSquares(m).some(([sq]) => odd.has(sq))) continue;
        g.move({ from: m.from, to: m.to, promotion: 'q' });
        seq.push(m);
        dfs(g, depth - 1);
        seq.pop();
        g.undo();
      }
    };
    dfs(root.g, maxPly);
  }
  results.sort((a, b) => b.s - a.s);
  const best = results[0];
  if (!best || best.s - base < 4) return null;
  const rival = results.find((r) => r.key !== best.key);
  if (rival && best.s - rival.s < 2) return null;
  return best;
}

// Aus einer bestätigten Stellung lernen: Farbmodell und Figurenvorlagen
function learn(img, cur, feats) {
  S.model.fit(feats || colorFeatures(cur, PER), classMap(S.game));
  const proj = (u, v) => project(S.H, u, v);
  const board = S.game.board();
  for (let sq = 0; sq < 64; sq++) {
    const p = board[7 - (sq >> 3)][sq % 8];
    if (p) S.templates.add(p.color, p.type, cropPatch(img, proj, sq), isDark(sq));
  }
  S.needLearn = false;
  saveModel(S.model, S.templates);
  updateLearnInfo();
}

function updateLearnInfo() {
  $('#learnInfo').textContent = S.model.ready && S.templates.ready
    ? 'Dein Figurenset ist gelernt: Stellungen mitten in der Partie können eingelesen werden.'
    : 'Noch kein Figurenset gelernt: einmal mit der Grundstellung starten und einen Zug spielen.';
}

/* ---------- Stellung einlesen ---------- */

function scanPosition(img) {
  if (!S.model.ready || !S.templates.ready || !S.H) return null;
  const cur = features(img);
  const ll = S.model.logLik(colorFeatures(cur, PER));
  const cls = [];
  for (let sq = 0; sq < 64; sq++) {
    const row = [ll[sq * 3], ll[sq * 3 + 1], ll[sq * 3 + 2]];
    cls.push(CLASSES[row.indexOf(Math.max(...row))]);
  }
  const proj = (u, v) => project(S.H, u, v);
  const sims = new Map();
  for (let sq = 0; sq < 64; sq++) if (cls[sq] !== 'e') sims.set(sq, S.templates.similarities(cls[sq], cropPatch(img, proj, sq), isDark(sq)));
  // Die zuletzt verfolgte Stellung hilft beim Zuordnen: was dort stand, steht oft noch da
  const prior = new Map();
  if (S.game.history().length || S.game.fen() !== new Chess().fen()) {
    const board = S.game.board();
    for (let sq = 0; sq < 64; sq++) { const p = board[7 - (sq >> 3)][sq % 8]; if (p) prior.set(sq, p); }
  }
  let best = null;
  // Nur die beiden Ausrichtungen, bei denen a1 dunkel ist
  for (const { o, perm } of orientationPerms()) {
    if (o !== S.a1 && o !== (S.a1 + 2) % 4) continue;
    const cands = [];
    for (let s = 0; s < 64; s++) if (cls[perm[s]] !== 'e') cands.push({ sq: s, color: cls[perm[s]], sims: sims.get(perm[s]) });
    if (!cands.some((c) => c.color === 'w') || !cands.some((c) => c.color === 'b')) continue;
    const pieces = assignTypes(cands, prior);
    if (DEBUG && o === S.a1) console.log('verify-scan', cands.map((c) => FILES[c.sq % 8] + ((c.sq >> 3) + 1) + ':' + Object.entries(c.sims).sort((x, y) => y[1] - x[1]).slice(0, 2).map(([t, v]) => t + v.toFixed(2)).join('/')).join(' '));
    let fit = 0;
    for (const c of cands) fit += c.sims[pieces.get(c.sq).type];
    const score = plausibility(pieces) + fit * 0.3;
    if (!best || score > best.score) best = { o, pieces, score };
  }
  return best;
}

const PIECE_CYCLE = [null, ...['w', 'b'].flatMap((c) => ['p', 'n', 'b', 'r', 'q', 'k'].map((t) => ({ color: c, type: t })))];

function openScanDialog(scan) {
  S.scan = { o: scan.o, pieces: new Map(scan.pieces) };
  $('#scanTurn').value = S.myColor === 'b' ? 'b' : 'w';
  $('#scanMsg').textContent = 'Tippe auf ein Feld, um eine falsch erkannte Figur zu ändern.';
  renderScanBoard();
  $('#scan').hidden = false;
}

function renderScanBoard() {
  const el = $('#scanBoard');
  const flip = S.myColor === 'b';
  let html = '';
  for (let r = 7; r >= 0; r--) for (let f = 0; f < 8; f++) {
    const rank = flip ? 7 - r : r, file = flip ? 7 - f : f;
    const sq = file + 8 * rank;
    const p = S.scan.pieces.get(sq);
    html += `<div data-sq="${sq}" class="${(file + rank) % 2 === 0 ? 'd' : 'l'} ${p ? 'p' + p.color : ''}">${p ? GLYPH[p.type] : ''}</div>`;
  }
  el.innerHTML = html;
}

$('#scanBoard').addEventListener('click', (e) => {
  const sq = Number(e.target.closest('[data-sq]')?.dataset.sq);
  if (Number.isNaN(sq) || !S.scan) return;
  const p = S.scan.pieces.get(sq);
  const i = PIECE_CYCLE.findIndex((c) => (c === null ? !p : p && c.color === p.color && c.type === p.type));
  const next = PIECE_CYCLE[(i + 1) % PIECE_CYCLE.length];
  if (next) S.scan.pieces.set(sq, next); else S.scan.pieces.delete(sq);
  renderScanBoard();
});

$('#btnScanOk').addEventListener('click', () => {
  if (!S.scan) return;
  const fen = toFen(S.scan.pieces, $('#scanTurn').value);
  let g;
  try { g = new Chess(fen); } catch (err) {
    $('#scanMsg').textContent = 'Diese Stellung ist so nicht möglich (z. B. König fehlt oder der Gegner steht im Schach). Bitte korrigieren.';
    return;
  }
  setBoard(S.quad, S.scan.o);
  S.game = g; S.lastMove = null; S.scan = null;
  $('#scan').hidden = true;
  resetTracking();
  S.inconsistent = false; S.mismatch = []; S.needLearn = true;
  idleStatus();
  requestSuggestion();
});
$('#btnScanRetry').addEventListener('click', () => {
  const scan = scanPosition(grab());
  if (scan) openScanDialog(scan); else $('#scanMsg').textContent = 'Brett gerade nicht gut zu sehen. Hand weg und nochmal versuchen.';
});
$('#btnScanCancel').addEventListener('click', () => { S.scan = null; $('#scan').hidden = true; });

function startScan() {
  if (!S.quad) { setStatus('Erst muss das Brett erkannt sein', 'warn'); return; }
  if (!S.model.ready || !S.templates.ready) {
    setStatus('Noch kein Figurenset gelernt – einmal mit Grundstellung starten', 'warn');
    return;
  }
  const scan = scanPosition(grab());
  if (scan) openScanDialog(scan);
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
    else if (line.startsWith('info') && line.includes(' pv ')) onInfo(line);
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

// Schon während Stockfish rechnet, den bisher besten Zug zeigen: so steht nach wenigen
// Zehntelsekunden ein Vorschlag da, der danach höchstens noch präzisiert wird.
function onInfo(line) {
  engine.info = line;
  if (engine.next || engine.fen !== S.game.fen()) return;
  const depth = Number(line.match(/ depth (\d+)/)?.[1] || 0);
  if (depth < 8) return;
  const uci = line.match(/ pv (\S+)/)?.[1];
  if (uci && (!S.best || S.best.uci !== uci)) setBest(uci, false);
}

function onBestMove(uci) {
  engine.busy = false;
  if (engine.next) { go(engine.next); return; }
  if (engine.fen !== S.game.fen() || !uci || uci === '(none)') return;
  setBest(uci, true);
}

function setBest(uci, final) {
  const probe = new Chess(engine.fen);
  let m;
  try { m = probe.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] || 'q' }); } catch { return; }
  if (!m) return;
  S.best = { uci, from: m.from, to: m.to, san: m.san, piece: m.piece, color: m.color, promotion: m.promotion, final };
  S.evalText = describeEval(engine.info, m.color);
  renderSuggestion();
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
    if (Math.abs(v) < 0.3) return 'Stellung ausgeglichen';
    return `${v > 0 ? 'Weiß' : 'Schwarz'} steht besser (${v > 0 ? '+' : ''}${v.toFixed(1).replace('.', ',')})`;
  }
  return '';
}

function requestSuggestion() {
  S.best = null;
  const g = S.game;
  $('#btnUndo').hidden = g.history().length === 0;
  if (g.isGameOver()) {
    renderSuggestion(g.isCheckmate() ? `Schachmatt – ${g.turn() === 'w' ? 'Schwarz' : 'Weiß'} gewinnt` : 'Remis');
    return;
  }
  if (S.myColor !== 'both' && g.turn() !== S.myColor) {
    renderSuggestion(`Warte auf den Zug von ${g.turn() === 'w' ? 'Weiß' : 'Schwarz'}`);
    return;
  }
  renderSuggestion('Stockfish rechnet …');
  go(g.fen());
}

/* ---------- Anzeige ---------- */

// Die Statuszeile erscheint nur, wenn etwas zu tun ist; im Normalbetrieb bleibt das Bild frei.
function setStatus(text, tone = 'idle') {
  $('#statusText').textContent = text;
  $('#status').dataset.tone = tone;
  $('#status').hidden = tone === 'ok';
}

function idleStatus() { setStatus('', 'ok'); }

function moveText(b) {
  if (b.san.startsWith('O-O-O')) return 'Lange Rochade';
  if (b.san.startsWith('O-O')) return 'Kurze Rochade';
  let t = `${PIECE_DE[b.piece]} ${b.from} → ${b.to}`;
  if (b.san.includes('x')) t += ' (schlägt)';
  if (b.promotion) t += ', wird Dame';
  return t;
}

function renderSuggestion(placeholder) {
  const panel = $('#panel'), move = $('#sugMove'), meta = $('#sugMeta');
  if (S.inconsistent) {
    // Lieber nichts empfehlen als einen Zug aus einer falschen Stellung
    panel.dataset.tone = 'wait';
    move.textContent = 'Stellung passt nicht zum Bild';
    meta.textContent = 'Orange Felder prüfen oder oben tippen zum Neu-Einlesen';
    return;
  }
  const notice = S.notice && S.notice.until > performance.now() ? S.notice.text : '';
  if (S.best) {
    panel.dataset.tone = 'move';
    move.textContent = moveText(S.best);
    meta.textContent = [notice, S.best.san.includes('+') ? 'Schach!' : '', S.evalText].filter(Boolean).join(' · ');
  } else {
    panel.dataset.tone = 'wait';
    if (placeholder) S.placeholder = placeholder;
    move.textContent = S.placeholder || '–';
    meta.textContent = notice || (S.lastMove ? `Letzter Zug: ${S.lastMove.san}` : '');
  }
}

function vibrate() { try { navigator.vibrate?.(30); } catch { /* iOS kennt das nicht */ } }

// Abbildung work-Koordinaten -> Bildschirm (Video ist mit object-fit: contain skaliert).
function viewMap() {
  const cw = overlay.clientWidth, ch = overlay.clientHeight;
  const ww = work.width || 1, wh = work.height || 1;
  const s = Math.min(cw / ww, ch / wh);
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
  if (S.notice && S.notice.until <= performance.now()) { S.notice = null; renderSuggestion(); }
  const dpr = window.devicePixelRatio || 1;
  const cw = overlay.clientWidth, ch = overlay.clientHeight;
  if (overlay.width !== Math.round(cw * dpr) || overlay.height !== Math.round(ch * dpr)) { overlay.width = Math.round(cw * dpr); overlay.height = Math.round(ch * dpr); }
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
    // Brett umranden: dezente leuchtende Kontur
    const outline = BOARD_CORNERS.map(([u, v]) => boardToScreen(u, v, vm));
    octx.save();
    octx.shadowColor = cBoard; octx.shadowBlur = 14;
    octx.lineWidth = 2.5; octx.strokeStyle = cBoard; octx.globalAlpha = 0.7;
    polyPath(outline); octx.stroke();
    octx.restore();

    drawPieces(vm);
    if (S.mismatch.length && S.badStreak >= 2) {
      octx.save();
      octx.fillStyle = css.getPropertyValue('--warn').trim();
      octx.strokeStyle = octx.fillStyle; octx.lineWidth = 2;
      for (const sq of S.mismatch) {
        const poly = squarePoly(FILES[sq % 8] + ((sq >> 3) + 1), vm, 0.06);
        octx.globalAlpha = 0.25; polyPath(poly); octx.fill();
        octx.globalAlpha = 0.9; polyPath(poly); octx.stroke();
      }
      octx.restore();
    }

    if (S.tapFrom) {
      octx.fillStyle = 'rgba(127,211,255,0.45)';
      polyPath(squarePoly(S.tapFrom, vm, 0.04)); octx.fill();
    }
    if (S.best && !S.inconsistent) drawBest(t, vm, cMove);

    if (S.mode === 'adjust') {
      // Raster nur beim Anpassen der Ecken, damit man die Felder ausrichten kann
      octx.lineWidth = 1; octx.strokeStyle = 'rgba(127,211,255,0.45)';
      octx.beginPath();
      for (let i = 1; i < 8; i++) {
        let [x, y] = boardToScreen(i, 0, vm); octx.moveTo(x, y); [x, y] = boardToScreen(i, 8, vm); octx.lineTo(x, y);
        [x, y] = boardToScreen(0, i, vm); octx.moveTo(x, y); [x, y] = boardToScreen(8, i, vm); octx.lineTo(x, y);
      }
      octx.stroke();
      S.quad.forEach((p) => {
        const [x, y] = toScreen(p, vm);
        octx.fillStyle = 'rgba(13,16,20,.6)'; octx.beginPath(); octx.arc(x, y, 18, 0, 7); octx.fill();
        octx.strokeStyle = cBoard; octx.lineWidth = 3; octx.stroke();
      });
    }
  }
}

// Jede bekannte Figur bekommt einen Ring, der flach auf ihrem Feld liegt (perspektivisch
// mit der Brettebene verzerrt), damit man sieht, was die App wo vermutet.
const RING = Array.from({ length: 28 }, (_, i) => [Math.cos((i / 28) * Math.PI * 2), Math.sin((i / 28) * Math.PI * 2)]);
function drawPieces(vm) {
  const board = S.game.board();
  octx.save();
  octx.lineWidth = 2;
  for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) {
    const p = board[7 - r][f];
    if (!p) continue;
    const pts = RING.map(([c, s]) => boardToScreen(f + 0.5 + c * 0.36, r + 0.5 + s * 0.36, vm));
    polyPath(pts);
    if (p.color === 'w') {
      octx.strokeStyle = 'rgba(255,255,255,0.8)'; octx.stroke();
    } else {
      octx.strokeStyle = 'rgba(255,255,255,0.35)'; octx.lineWidth = 4; octx.stroke();
      octx.strokeStyle = 'rgba(20,24,30,0.9)'; octx.lineWidth = 2; octx.stroke();
    }
  }
  octx.restore();
}

function drawBest(t, vm, color) {
  const b = S.best;
  const pulse = 0.5 + 0.5 * Math.sin(t / 260);
  octx.save();
  octx.fillStyle = color;
  octx.globalAlpha = 0.3; polyPath(squarePoly(b.from, vm, 0.04)); octx.fill();
  octx.globalAlpha = 0.35 + 0.3 * pulse; polyPath(squarePoly(b.to, vm, 0.04)); octx.fill();
  octx.globalAlpha = 1; octx.lineWidth = 2; octx.strokeStyle = color;
  polyPath(squarePoly(b.from, vm, 0.04)); octx.stroke();
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
  // Feldnamen direkt über der Markierung, damit man nicht nach unten schauen muss
  const label = `${b.from}→${b.to}`;
  octx.font = `700 ${Math.round(Math.max(13, r * 0.55))}px -apple-system, sans-serif`;
  const lw = octx.measureText(label).width + 12, lh = Math.max(20, r * 0.8), ly = topY - r - lh - 4;
  octx.fillStyle = 'rgba(13,16,20,.8)';
  octx.beginPath();
  if (octx.roundRect) octx.roundRect(ex - lw / 2, ly, lw, lh, 6); else octx.rect(ex - lw / 2, ly, lw, lh);
  octx.fill();
  octx.fillStyle = color;
  octx.fillText(label, ex, ly + lh / 2);
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
  const interval = S.mode === 'search' ? 220 : 140;
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

function resetTracking() { S.ref = null; S.prev = null; S.stable = 0; S.evaluated = false; S.unclear = false; }

function startSearch() {
  S.mode = 'search'; S.cand = null; S.candHits = 0; resetTracking();
  setStatus('Suche Schachbrett …', 'search');
}

function enterTrack() {
  S.mode = 'track'; resetTracking();
  $('#panel').hidden = false;
  idleStatus();
  requestSuggestion();
}

async function startCamera(deviceId) {
  S.stream?.getTracks().forEach((t) => t.stop());
  const size = { width: { ideal: 1280 }, height: { ideal: 720 } };
  const video0 = deviceId ? { deviceId: { exact: deviceId }, ...size } : { facingMode: { ideal: 'environment' }, ...size };
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
    else { S.mode = 'track'; resetTracking(); idleStatus(); }
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

const closeSettings = () => { $('#settings').hidden = true; };

$('#btnStart').addEventListener('click', async () => {
  S.myColor = $('#introColor').value;
  $('#colorSelect').value = S.myColor;
  $('#introMsg').textContent = 'Kamera wird gestartet …';
  try {
    await startCamera();
  } catch (e) {
    $('#introMsg').textContent = window.isSecureContext
      ? 'Kein Kamerazugriff. Erlaube die Kamera in den Browser-Einstellungen und lade die Seite neu.'
      : 'Die Kamera funktioniert nur über https:// – öffne die App über lars-mst.github.io/schach-kamera';
    return;
  }
  $('#intro').hidden = true;
  startSearch();
});

$('#btnSettings').addEventListener('click', () => { $('#fenInput').value = S.game.fen(); updateLearnInfo(); $('#settings').hidden = false; });
$('#btnScan').addEventListener('click', () => { closeSettings(); startScan(); });
$('#btnForget').addEventListener('click', () => {
  forgetModel();
  S.model = new ColorModel(); S.templates = new TemplateStore();
  S.needLearn = true; S.mismatch = []; S.badStreak = 0;
  if (S.inconsistent) { S.inconsistent = false; idleStatus(); requestSuggestion(); }
  updateLearnInfo();
});
const statusTap = () => { if (S.inconsistent) startScan(); };
$('#status').addEventListener('click', statusTap);
$('#status').addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') statusTap(); });
$('#btnClose').addEventListener('click', closeSettings);
$('#settings').addEventListener('click', (e) => { if (e.target.id === 'settings') closeSettings(); });
$('#camSelect').addEventListener('change', async (e) => { await startCamera(e.target.value); startSearch(); });
$('#colorSelect').addEventListener('change', (e) => { S.myColor = e.target.value; requestSuggestion(); });
$('#thinkSelect').addEventListener('change', (e) => { S.thinkMs = Number(e.target.value); });
$('#btnFen').addEventListener('click', () => {
  try {
    S.game = new Chess($('#fenInput').value.trim());
  } catch {
    $('#fenInput').setCustomValidity('Ungültige FEN'); $('#fenInput').reportValidity(); return;
  }
  $('#fenInput').setCustomValidity('');
  S.lastMove = null; resetTracking();
  requestSuggestion(); closeSettings();
});
$('#btnNew').addEventListener('click', () => {
  S.game = new Chess(); S.lastMove = null; resetTracking();
  requestSuggestion(); closeSettings();
});
$('#btnManual').addEventListener('click', () => {
  closeSettings();
  if (!S.stream) return;
  S.mode = 'manual'; S.manualPts = [];
  setStatus('Tippe die 4 Ecken der Spielfläche an', 'search');
});
$('#btnUndo').addEventListener('click', () => {
  if (!S.game.undo()) return;
  const h = S.game.history({ verbose: true });
  S.lastMove = h[h.length - 1] || null;
  resetTracking();
  requestSuggestion();
});
$('#btnRotate').addEventListener('click', () => {
  closeSettings();
  if (!S.quad) return;
  setBoard(S.quad, (S.a1 + 1) % 4);
  resetTracking();
  requestSuggestion();
});
$('#btnAdjust').addEventListener('click', () => {
  closeSettings();
  if (!S.quad) return;
  S.mode = 'adjust';
  setStatus('Ecken ziehen · daneben tippen zum Beenden', 'search');
});
$('#btnRedetect').addEventListener('click', () => { closeSettings(); S.best = null; startSearch(); });

window.addEventListener('pagehide', () => S.stream?.getTracks().forEach((t) => t.stop()));

renderSuggestion();
updateLearnInfo();
startEngine();
waitForCv().then(({ cv }) => { S.cv = cv; });
requestAnimationFrame(loop);
requestAnimationFrame(draw);
if (DEBUG) window.__dbg = { findQuads, refineGrid, grab, guessOrientation };
