// Feld- und Figurenerkennung.
// 1. Farbmodell: ordnet jedes Feld als leer (e), weiße Figur (w) oder schwarze Figur (b) ein.
//    Es lernt aus Bildern, deren Stellung bekannt ist (Grundstellung, danach jede bestätigte Stellung).
// 2. Figurenvorlagen: kleine Bildausschnitte jeder Figur, ebenfalls aus bekannten Stellungen gelernt.
//    Damit lässt sich eine unbekannte Stellung (mitten in der Partie) einlesen.
// Beides wird im Browser gespeichert, damit das eigene Figurenset nur einmal gelernt werden muss.

export const CLASSES = ['e', 'w', 'b'];
const TYPES = ['p', 'n', 'b', 'r', 'q', 'k'];
const STORE_KEY = 'schachkamera.modell.v3';
const F = 5;                       // Merkmale pro Feld
const VAR_FLOOR = [36, 0.004, 0.004, 9, 9];
export const PATCH_W = 12, PATCH_H = 24;
const MAX_TEMPLATES = 40;

export const isDark = (sq) => ((sq % 8) + (sq >> 3)) % 2 === 0; // a1 ist dunkel

// Klasse eines Feldes in einer bekannten Stellung (chess.js)
export function classMap(game) {
  const board = game.board();
  const out = new Array(64);
  for (let sq = 0; sq < 64; sq++) {
    const p = board[7 - (sq >> 3)][sq % 8];
    out[sq] = p ? p.color : 'e';
  }
  return out;
}

/* ---------- Farbmerkmale ---------- */

// f: Stichproben aus features() (pro Feld n RGB-Werte), per: Werte pro Feld
export function colorFeatures(f, per) {
  const n = per / 3;
  const lum = new Float32Array(64 * n), mean = new Float32Array(64), sd = new Float32Array(64), chroma = new Float32Array(64);
  for (let sq = 0; sq < 64; sq++) {
    let s = 0, s2 = 0, c = 0;
    for (let i = 0; i < n; i++) {
      const k = sq * per + i * 3;
      const L = 0.299 * f[k] + 0.587 * f[k + 1] + 0.114 * f[k + 2];
      lum[sq * n + i] = L; s += L; s2 += L * L; c += f[k] - f[k + 2];
    }
    mean[sq] = s / n; sd[sq] = Math.sqrt(Math.max(0, s2 / n - (s / n) ** 2)); chroma[sq] = c / n;
  }
  // Leere Felder je Feldfarbe schätzen: die ruhigsten 40 % der Felder dieser Farbe.
  const ref = [false, true].map((dark) => {
    const idx = [];
    for (let sq = 0; sq < 64; sq++) if (isDark(sq) === dark) idx.push(sq);
    idx.sort((a, b) => sd[a] - sd[b]);
    const calm = idx.slice(0, Math.max(4, Math.round(idx.length * 0.4)));
    const med = (arr) => { const s = [...arr].sort((a, b) => a - b); return s[s.length >> 1]; };
    return { L: med(calm.map((q) => mean[q])), C: med(calm.map((q) => chroma[q])) };
  });
  const out = new Float32Array(64 * F);
  for (let sq = 0; sq < 64; sq++) {
    const r = ref[isDark(sq) ? 1 : 0];
    let bright = 0, darkc = 0;
    for (let i = 0; i < n; i++) {
      const L = lum[sq * n + i];
      if (L > r.L + 30) bright++;
      else if (L < r.L - 30) darkc++;
    }
    const o = sq * F;
    out[o] = mean[sq] - r.L;
    out[o + 1] = bright / n;
    out[o + 2] = darkc / n;
    out[o + 3] = sd[sq];
    out[o + 4] = chroma[sq] - r.C;
  }
  return out;
}

export class ColorModel {
  constructor(data) {
    // stats[dark ? 1 : 0][klasse] = { mu, v, n }
    this.stats = data?.stats || [0, 1].map(() => Object.fromEntries(CLASSES.map((c) => [c, null])));
  }

  get ready() { return this.stats.every((g) => CLASSES.every((c) => g[c] && g[c].n > 0)); }

  fit(feats, classes, rate = 0.35) {
    for (const dark of [0, 1]) for (const c of CLASSES) {
      const rows = [];
      for (let sq = 0; sq < 64; sq++) if ((isDark(sq) ? 1 : 0) === dark && classes[sq] === c) rows.push(feats.subarray(sq * F, sq * F + F));
      if (rows.length < 2) continue;
      const mu = new Array(F).fill(0), v = new Array(F).fill(0);
      rows.forEach((r) => r.forEach((x, i) => { mu[i] += x / rows.length; }));
      rows.forEach((r) => r.forEach((x, i) => { v[i] += (x - mu[i]) ** 2 / rows.length; }));
      const old = this.stats[dark][c];
      const a = old ? rate : 1;
      this.stats[dark][c] = {
        mu: mu.map((m, i) => (old ? old.mu[i] * (1 - a) : 0) + m * a),
        v: v.map((x, i) => Math.max(VAR_FLOOR[i], (old ? old.v[i] * (1 - a) : 0) + x * a)),
        n: (old?.n || 0) + rows.length,
      };
    }
  }

  // log-Wahrscheinlichkeit je Feld und Klasse, begrenzt, damit ein einzelnes Ausreißerfeld nicht alles bestimmt
  logLik(feats) {
    const ll = new Float32Array(64 * 3);
    for (let sq = 0; sq < 64; sq++) {
      const g = this.stats[isDark(sq) ? 1 : 0];
      const row = CLASSES.map((c) => {
        const st = g[c];
        if (!st) return -50;
        let s = 0;
        for (let i = 0; i < F; i++) s += (feats[sq * F + i] - st.mu[i]) ** 2 / st.v[i] + Math.log(st.v[i]);
        return -0.5 * s;
      });
      const top = Math.max(...row);
      row.forEach((x, i) => { ll[sq * 3 + i] = Math.max(top - 12, x); });
    }
    return ll;
  }
}

/* ---------- Figurenvorlagen ---------- */

function bilinear(img, x, y) {
  const w = img.width, h = img.height;
  x = Math.min(w - 1.001, Math.max(0, x)); y = Math.min(h - 1.001, Math.max(0, y));
  const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0;
  const at = (xx, yy) => { const i = (yy * w + xx) * 4; return 0.299 * img.data[i] + 0.587 * img.data[i + 1] + 0.114 * img.data[i + 2]; };
  return at(x0, y0) * (1 - fx) * (1 - fy) + at(x0 + 1, y0) * fx * (1 - fy) + at(x0, y0 + 1) * (1 - fx) * fy + at(x0 + 1, y0 + 1) * fx * fy;
}

// Bildausschnitt eines Feldes, nach oben verlängert, weil Figuren von der Kamera aus nach oben ragen.
// project: (u, v) Brettkoordinaten -> Bildkoordinaten
export function cropPatch(img, project, sq) {
  const f = sq % 8, r = sq >> 3;
  const pts = [[f, r], [f + 1, r], [f + 1, r + 1], [f, r + 1]].map(([u, v]) => project(u, v));
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y1 = Math.max(...ys), yt = Math.min(...ys);
  const w = x1 - x0, top = yt - 0.6 * w, h = y1 - top;
  // Grauwerte in doppelter Auflösung abtasten, daraus Helligkeitsmuster und Kantenrichtungen
  const W2 = PATCH_W * 2, H2 = PATCH_H * 2;
  const g = new Float32Array(W2 * H2);
  for (let j = 0; j < H2; j++) for (let i = 0; i < W2; i++) g[j * W2 + i] = bilinear(img, x0 + ((i + 0.5) / W2) * w, top + ((j + 0.5) / H2) * h);
  const raw = new Float32Array(PATCH_W * PATCH_H);
  for (let j = 0; j < PATCH_H; j++) for (let i = 0; i < PATCH_W; i++) {
    const k = 2 * j * W2 + 2 * i;
    raw[j * PATCH_W + i] = (g[k] + g[k + 1] + g[k + W2] + g[k + W2 + 1]) / 4;
  }
  normalize(raw, true);
  // Kantenrichtungen (HOG): Zellen 4x4 Pixel, 6 Richtungen. Unabhängiger von Licht und Feldfarbe.
  const CW = W2 / 4, CH = H2 / 4, BINS = 6;
  const hog = new Float32Array(CW * CH * BINS);
  for (let j = 1; j < H2 - 1; j++) for (let i = 1; i < W2 - 1; i++) {
    const gx = g[j * W2 + i + 1] - g[j * W2 + i - 1], gy = g[(j + 1) * W2 + i] - g[(j - 1) * W2 + i];
    const mag = Math.hypot(gx, gy);
    if (mag < 1) continue;
    let a = Math.atan2(gy, gx); if (a < 0) a += Math.PI;
    const b = Math.min(BINS - 1, Math.floor((a / Math.PI) * BINS));
    hog[((j >> 2) * CW + (i >> 2)) * BINS + b] += Math.sqrt(mag);
  }
  normalize(hog, false);
  // Zusammen ergibt das Skalarprodukt 0,3 · Helligkeit + 0,7 · Kanten
  const out = new Float32Array(PATCH_LEN);
  const a = Math.sqrt(0.3), b = Math.sqrt(0.7);
  raw.forEach((v, k) => { out[k] = v * a; });
  hog.forEach((v, k) => { out[raw.length + k] = v * b; });
  return out;
}

const PATCH_LEN = PATCH_W * PATCH_H + (PATCH_W / 2) * (PATCH_H / 2) * 6;

function normalize(v, center) {
  if (center) { let m = 0; v.forEach((x) => { m += x; }); m /= v.length; for (let k = 0; k < v.length; k++) v[k] -= m; }
  let n2 = 0; v.forEach((x) => { n2 += x * x; });
  const n = Math.sqrt(n2) || 1;
  for (let k = 0; k < v.length; k++) v[k] /= n;
}

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

export class TemplateStore {
  constructor(data) {
    this.t = { w: {}, b: {} };
    // Pro Vorlage merken, ob sie auf einem dunklen Feld stand: der Feldhintergrund prägt den Ausschnitt mit
    for (const c of ['w', 'b']) for (const ty of TYPES) this.t[c][ty] = (data?.t?.[c]?.[ty] || []).map((s) => ({ d: s[0] === 'd', p: decode(s.slice(1)) }));
  }

  get ready() { return ['w', 'b'].every((c) => ['p', 'n', 'b', 'r', 'q', 'k'].every((ty) => this.t[c][ty].length > 0)); }

  add(color, type, patch, dark) {
    const list = this.t[color][type];
    list.push({ d: dark, p: patch });
    // je Feldfarbe höchstens die Hälfte, die ältesten fliegen raus
    const same = list.filter((x) => x.d === dark);
    if (same.length > MAX_TEMPLATES / 2) list.splice(list.indexOf(same[0]), 1);
  }

  // Ähnlichkeit zu jedem Figurentyp (beste Übereinstimmung, gemittelt mit der zweitbesten).
  // Vorlagen von der anderen Feldfarbe zählen nur, wenn es keine passenden gibt, und schwächer.
  similarities(color, patch, dark) {
    const out = {};
    for (const ty of TYPES) {
      const list = this.t[color][ty];
      const same = list.filter((x) => x.d === dark);
      const use = same.length ? same : list;
      const pen = same.length ? 0 : 0.15;
      const sims = use.map((x) => dot(x.p, patch)).sort((a, b) => b - a);
      out[ty] = sims.length ? (sims.length > 1 ? 0.6 * sims[0] + 0.4 * sims[1] : sims[0]) - pen : -1;
    }
    return out;
  }

  toJSON() {
    const t = { w: {}, b: {} };
    for (const c of ['w', 'b']) for (const ty of TYPES) t[c][ty] = this.t[c][ty].map((x) => (x.d ? 'd' : 'l') + encode(x.p));
    return { t };
  }
}

function encode(arr) {
  const bytes = new Uint8Array(arr.length);
  for (let i = 0; i < arr.length; i++) bytes[i] = Math.max(0, Math.min(255, Math.round(arr[i] * 400 + 128)));
  let s = '';
  bytes.forEach((b) => { s += String.fromCharCode(b); });
  return btoa(s);
}
function decode(str) {
  const s = atob(str), out = new Float32Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = (s.charCodeAt(i) - 128) / 400;
  return out;
}

export function loadModel() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (!raw) return null;
    const d = JSON.parse(raw);
    return { color: new ColorModel(d.color), templates: new TemplateStore(d.templates) };
  } catch { return null; }
}

export function saveModel(color, templates) {
  try { localStorage.setItem(STORE_KEY, JSON.stringify({ color: { stats: color.stats }, templates: templates.toJSON() })); } catch { /* Speicher voll oder gesperrt */ }
}

export function forgetModel() { try { localStorage.removeItem(STORE_KEY); } catch { /* egal */ } }

/* ---------- Stellung aus dem Bild ableiten ---------- */

// Figuren so verteilen, dass die Stellung möglich ist: je ein König, keine Bauern auf Reihe 1/8,
// höchstens so viele Figuren eines Typs wie zu Beginn (außer bei sehr klarer Erkennung).
const MAX_COUNT = { p: 8, n: 2, b: 2, r: 2, q: 1, k: 1 };

const BACK = 'rnbqkbnr';
// Welche Figur stand in der Grundstellung auf diesem Feld (in dieser Farbe)?
function startType(sq, color) {
  const r = sq >> 3, f = sq % 8;
  if (color === 'w') return r === 0 ? BACK[f] : r === 1 ? 'p' : null;
  return r === 7 ? BACK[f] : r === 6 ? 'p' : null;
}

// Kleinste Summe der Linienwechsel, mit der Bauern auf diesen Linien aus je einer eigenen
// Ausgangslinie stammen können (jeder Linienwechsel braucht einen Schlagzug).
function pawnShift(files) {
  const f = [...files].sort((a, b) => a - b), k = f.length;
  if (k > 8) return Infinity;
  // dp[j]: beste Kosten, wenn die ersten i Bauern auf Ausgangslinien < j verteilt sind
  let dp = new Array(9).fill(0);
  for (let i = 0; i < k; i++) {
    const nd = new Array(9).fill(Infinity);
    for (let j = 1; j <= 8; j++) nd[j] = Math.min(nd[j - 1], dp[j - 1] + Math.abs(f[i] - (j - 1)));
    dp = nd;
  }
  return dp[8];
}

export function assignTypes(cands, prior = null) {
  // cands: [{ sq, color, sims: {p..k} }]
  // prior: Map sq -> { color, type } aus einer zuvor verfolgten Stellung (optional)
  const result = new Map();
  const counts = { w: {}, b: {} };
  TYPES.forEach((t) => { counts.w[t] = 0; counts.b[t] = 0; });
  // Jede Seite hat (ohne Umwandlung) einen Läufer pro Feldfarbe
  const bishopOn = { w: new Set(), b: new Set() };
  const pawnFiles = { w: [], b: [] };
  const captures = { w: 16 - cands.filter((c) => c.color === 'b').length, b: 16 - cands.filter((c) => c.color === 'w').length };
  const options = [];
  for (const c of cands) for (const ty of TYPES) {
    const rank = c.sq >> 3;
    if (ty === 'p' && (rank === 0 || rank === 7)) continue;
    const pr = prior?.get(c.sq);
    const bonus = (startType(c.sq, c.color) === ty ? 0.12 : 0) + (pr && pr.color === c.color && pr.type === ty ? 0.25 : 0);
    options.push({ sq: c.sq, color: c.color, ty, s: c.sims[ty] + bonus });
  }
  // Könige zuerst: pro Farbe das Feld, das am meisten nach König aussieht
  for (const color of ['w', 'b']) {
    const k = options.filter((o) => o.color === color && o.ty === 'k').sort((a, b) => b.s - a.s)[0];
    if (k) { result.set(k.sq, { color, type: 'k' }); counts[color].k = 1; }
  }
  options.sort((a, b) => b.s - a.s);
  for (const o of options) {
    if (result.has(o.sq) || o.ty === 'k') continue;
    if (counts[o.color][o.ty] >= MAX_COUNT[o.ty] && o.ty !== 'q') continue;
    if (o.ty === 'q' && counts[o.color].q >= 1 && o.s < 0.9) continue;
    if (o.ty === 'b' && bishopOn[o.color].has(isDark(o.sq))) continue;
    if (o.ty === 'p' && pawnShift([...pawnFiles[o.color], o.sq % 8]) > captures[o.color]) continue;
    if (o.ty === 'b') bishopOn[o.color].add(isDark(o.sq));
    if (o.ty === 'p') pawnFiles[o.color].push(o.sq % 8);
    result.set(o.sq, { color: o.color, type: o.ty });
    counts[o.color][o.ty]++;
  }
  for (const c of cands) if (!result.has(c.sq)) result.set(c.sq, { color: c.color, type: (c.sq >> 3) % 7 === 0 ? 'q' : 'p' });
  return result;
}

export function toFen(pieces, turn) {
  let fen = '';
  for (let r = 7; r >= 0; r--) {
    let empty = 0;
    for (let f = 0; f < 8; f++) {
      const p = pieces.get(f + 8 * r);
      if (!p) { empty++; continue; }
      if (empty) { fen += empty; empty = 0; }
      fen += p.color === 'w' ? p.type.toUpperCase() : p.type;
    }
    if (empty) fen += empty;
    if (r) fen += '/';
  }
  // Rochaderechte nur, wenn König und Turm noch auf ihren Ausgangsfeldern stehen
  const at = (sq, color, type) => { const p = pieces.get(sq); return p && p.color === color && p.type === type; };
  let castle = '';
  if (at(4, 'w', 'k')) { if (at(7, 'w', 'r')) castle += 'K'; if (at(0, 'w', 'r')) castle += 'Q'; }
  if (at(60, 'b', 'k')) { if (at(63, 'b', 'r')) castle += 'k'; if (at(56, 'b', 'r')) castle += 'q'; }
  return `${fen} ${turn} ${castle || '-'} - 0 1`;
}

// Plausibilität einer eingelesenen Stellung (für die Wahl der Brettausrichtung)
export function plausibility(pieces) {
  let s = 0, wp = [], bp = [];
  for (const [sq, p] of pieces) {
    const rank = sq >> 3;
    if (p.type === 'p') (p.color === 'w' ? wp : bp).push(rank);
    if (p.type === 'k') s += p.color === 'w' ? (rank <= 3 ? 1 : -1) : (rank >= 4 ? 1 : -1);
  }
  const avg = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 3.5);
  s += (avg(bp) - avg(wp)) * 0.8;
  return s;
}
