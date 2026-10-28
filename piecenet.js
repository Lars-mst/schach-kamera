// Trainiertes Figurennetz: ein kleines Faltungsnetz (ca. 75 000 Gewichte), das für jedes Feld sagt,
// ob es leer ist oder welche der 12 Figuren darauf steht. Läuft in reinem JavaScript (kein WebGL,
// keine Zusatzbibliothek), damit es auf jedem iPhone gleich funktioniert.
// Training und Lizenzen der Trainingsdaten: siehe README, Abschnitt „Figurennetz“.

export const NET_CLASSES = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k', '.'];
export const NET_EMPTY = 12;
const TYPES = ['p', 'n', 'b', 'r', 'q', 'k'];

export class PieceNet {
  // meta: Inhalt von figuren.json, buf: ArrayBuffer von figuren.bin (float32, BN schon eingerechnet)
  constructor(meta, buf) {
    this.meta = meta;
    this.inW = meta.input.w; this.inH = meta.input.h; this.up = meta.input.up;
    const all = new Float32Array(buf);
    this.layers = meta.layers.map((l) => {
      const w = all.subarray(l.w, l.w + l.wn), b = all.subarray(l.b, l.b + l.out);
      return { ...l, w, bias: b };
    });
    // Zwischenspeicher, damit beim Rechnen kaum Speicher angelegt wird
    this.buf = [new Float32Array(meta.maxAct), new Float32Array(meta.maxAct)];
  }

  static async load(base = './model/') {
    const meta = await (await fetch(base + 'figuren.json')).json();
    const buf = await (await fetch(base + 'figuren.bin')).arrayBuffer();
    return new PieceNet(meta, buf);
  }

  // Bildausschnitt eines Feldes wie im Training: Feld plus eine Feldbreite nach oben (Figuren ragen
  // im Kamerabild nach oben). img: ImageData, project: (u, v) Brettkoordinaten -> Bildkoordinaten.
  crop(img, project, sq, out = new Float32Array(3 * this.inW * this.inH)) {
    const f = sq % 8, r = sq >> 3;
    const pts = [[f, r], [f + 1, r], [f + 1, r + 1], [f, r + 1]].map(([u, v]) => project(u, v));
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y1 = Math.max(...ys);
    const w = x1 - x0, top = Math.min(...ys) - this.up * w, h = y1 - top;
    const W = this.inW, H = this.inH, plane = W * H, iw = img.width, ih = img.height, d = img.data;
    for (let j = 0; j < H; j++) {
      const y = top + ((j + 0.5) / H) * h - 0.5;
      for (let i = 0; i < W; i++) {
        const x = x0 + ((i + 0.5) / W) * w - 0.5;
        const k = j * W + i;
        if (x < 0 || y < 0 || x > iw - 1 || y > ih - 1) { out[k] = out[k + plane] = out[k + 2 * plane] = 0; continue; }
        const xi = Math.min(iw - 2, x | 0), yi = Math.min(ih - 2, y | 0), fx = x - xi, fy = y - yi;
        const a = (yi * iw + xi) * 4, b = a + 4, c = a + iw * 4, e = c + 4;
        for (let ch = 0; ch < 3; ch++) {
          const v = (d[a + ch] * (1 - fx) + d[b + ch] * fx) * (1 - fy) + (d[c + ch] * (1 - fx) + d[e + ch] * fx) * fy;
          out[k + ch * plane] = v / 255 - 0.5;
        }
      }
    }
    return out;
  }

  // Ein Ausschnitt -> { probs (13), emb (Merkmalsvektor vor der letzten Schicht) }
  run(input) {
    let cur = input, H = this.inH, W = this.inW, C = 3, flip = 0;
    let emb = null;
    for (const l of this.layers) {
      if (l.type === 'conv') {
        const out = this.buf[flip]; flip ^= 1;
        conv3x3relu(cur, C, H, W, l.w, l.bias, l.out, out);
        cur = out; C = l.out;
        if (l.pool) {
          const o2 = this.buf[flip]; flip ^= 1;
          maxpool2(cur, C, H, W, o2);
          cur = o2; H >>= 1; W >>= 1;
        }
      } else if (l.type === 'gap') {
        emb = new Float32Array(C);
        const n = H * W;
        for (let c = 0; c < C; c++) { let s = 0; for (let k = 0; k < n; k++) s += cur[c * n + k]; emb[c] = s / n; }
      } else if (l.type === 'fc') {
        const logits = new Float32Array(l.out);
        for (let o = 0; o < l.out; o++) { let s = l.bias[o]; for (let i = 0; i < C; i++) s += l.w[o * C + i] * emb[i]; logits[o] = s; }
        const m = Math.max(...logits);
        let z = 0;
        for (let o = 0; o < l.out; o++) { logits[o] = Math.exp(logits[o] - m); z += logits[o]; }
        for (let o = 0; o < l.out; o++) logits[o] /= z;
        return { probs: logits, emb };
      }
    }
    return null;
  }

  // Alle angefragten Felder eines Bildes. Ergebnis: Map sq -> { probs, emb }
  classify(img, project, squares) {
    const out = new Map();
    const input = new Float32Array(3 * this.inW * this.inH);
    for (const sq of squares) out.set(sq, this.run(this.crop(img, project, sq, input)));
    return out;
  }

  // Wie classify, aber im Hintergrund: die Felder werden auf mehrere Web Worker verteilt
  // (je freiem Prozessorkern einer, höchstens drei). Ohne Worker rechnet es hier.
  classifyAsync(img, project, squares) {
    if (!this.workers) {
      this.workers = [];
      this.pending = new Map();
      this.nextId = 1;
      const n = Math.max(1, Math.min(3, (navigator.hardwareConcurrency || 2) - 1));
      try {
        for (let i = 0; i < n; i++) {
          const w = new Worker(new URL('./networker.js', import.meta.url), { type: 'module' });
          w.onmessage = (e) => { const p = this.pending.get(e.data.id); this.pending.delete(e.data.id); p?.(e.data); };
          w.onerror = () => { this.workers = []; for (const p of this.pending.values()) p({ error: 'worker' }); this.pending.clear(); };
          this.workers.push(w);
        }
      } catch { this.workers = []; }
    }
    if (!this.workers.length || !squares.length) return Promise.resolve(this.classify(img, project, squares));
    const size = 3 * this.inW * this.inH;
    const per = Math.ceil(squares.length / this.workers.length);
    const parts = [];
    for (let w = 0; w * per < squares.length; w++) {
      const part = squares.slice(w * per, (w + 1) * per);
      const inputs = new Float32Array(part.length * size);
      part.forEach((sq, i) => this.crop(img, project, sq, inputs.subarray(i * size, (i + 1) * size)));
      const id = this.nextId++;
      parts.push(new Promise((resolve) => {
        this.pending.set(id, (d) => {
          if (d.error) { resolve(this.classify(img, project, part)); return; }
          const out = new Map(), k = d.probs.length / part.length, e = d.embs.length / part.length;
          part.forEach((sq, i) => out.set(sq, { probs: d.probs.subarray(i * k, (i + 1) * k), emb: d.embs.subarray(i * e, (i + 1) * e) }));
          resolve(out);
        });
        this.workers[w].postMessage({ id, inputs, count: part.length }, [inputs.buffer]);
      }));
    }
    return Promise.all(parts).then((maps) => new Map(maps.flatMap((m) => [...m])));
  }
}

// Wahrscheinlichkeiten je Figurenart, wenn die Farbe bekannt ist
export function typeProbs(probs, color) {
  const o = color === 'w' ? 0 : 6;
  let s = 0;
  for (let t = 0; t < 6; t++) s += probs[o + t];
  const out = {};
  TYPES.forEach((t, i) => { out[t] = s > 0 ? probs[o + i] / s : 1 / 6; });
  return out;
}

// Leer / weiß / schwarz
export function colorProbs(probs) {
  let w = 0, b = 0;
  for (let t = 0; t < 6; t++) { w += probs[t]; b += probs[6 + t]; }
  return { e: probs[NET_EMPTY], w, b };
}

// 3x3-Faltung mit Rand 1, danach ReLU. Layout: Kanal, Zeile, Spalte.
function conv3x3relu(inp, C, H, W, w, bias, O, out) {
  const n = H * W;
  for (let o = 0; o < O; o++) {
    const ob = o * n, bo = bias[o];
    for (let k = 0; k < n; k++) out[ob + k] = bo;
    for (let c = 0; c < C; c++) {
      const ib = c * n, wb = (o * C + c) * 9;
      for (let dy = -1; dy <= 1; dy++) {
        const y0 = dy < 0 ? 1 : 0, y1 = dy > 0 ? H - 1 : H;
        for (let dx = -1; dx <= 1; dx++) {
          const wv = w[wb + (dy + 1) * 3 + dx + 1];
          if (wv === 0) continue;
          const x0 = dx < 0 ? 1 : 0, x1 = dx > 0 ? W - 1 : W;
          for (let y = y0; y < y1; y++) {
            const orow = ob + y * W, irow = ib + (y + dy) * W + dx;
            for (let x = x0; x < x1; x++) out[orow + x] += wv * inp[irow + x];
          }
        }
      }
    }
    for (let k = 0; k < n; k++) if (out[ob + k] < 0) out[ob + k] = 0;
  }
}

function maxpool2(inp, C, H, W, out) {
  const h2 = H >> 1, w2 = W >> 1;
  for (let c = 0; c < C; c++) for (let y = 0; y < h2; y++) for (let x = 0; x < w2; x++) {
    const i = c * H * W + 2 * y * W + 2 * x;
    out[c * h2 * w2 + y * w2 + x] = Math.max(inp[i], inp[i + 1], inp[i + W], inp[i + W + 1]);
  }
}

// Anpassung an das eigene Figurenset (Idee aus chesscog: Feinabstimmung mit wenigen Bildern des
// eigenen Sets). Aus jeder sicher bekannten Stellung kommen 64 Beispiele (Merkmalsvektor des Netzes
// plus richtige Klasse). Darauf wird die letzte Schicht des Netzes nachtrainiert, und zwar so, dass
// sie nah an der ursprünglichen bleibt. Das geht im Browser in wenigen Millisekunden.
const ADAPT_KEY = 'schachkamera.anpassung.v2';
const PER_CLASS = 48;

export class Adapter {
  constructor(net, data) {
    const fc = net.layers.find((l) => l.type === 'fc');
    this.D = fc.wn / fc.out; this.K = fc.out;
    this.W0 = Float32Array.from(fc.w); this.b0 = Float32Array.from(fc.bias);
    this.W = Float32Array.from(fc.w); this.b = Float32Array.from(fc.bias);
    this.samples = [];   // { e: Float32Array, y }
    this.trained = false;
    for (const [y, list] of Object.entries(data?.s || {})) for (const str of list) {
      const e = decode(str);
      if (e.length === this.D) this.samples.push({ e, y: Number(y) });
    }
    if (this.samples.length) this.train();
  }

  get ready() { return this.trained; }
  get count() { return this.samples.length; }

  // Beispiele einer bekannten Stellung: labels[sq] = Klasse 0..12, embs = Map sq -> Merkmalsvektor
  add(embs, labels) {
    for (const [sq, e] of embs) this.samples.push({ e: Float32Array.from(e), y: labels[sq] });
    // je Klasse nur die neuesten Beispiele behalten
    const byClass = new Map();
    for (let i = this.samples.length - 1; i >= 0; i--) {
      const s = this.samples[i];
      const n = byClass.get(s.y) || 0;
      if (n >= PER_CLASS) this.samples.splice(i, 1); else byClass.set(s.y, n + 1);
    }
    this.train();
    this.save();
  }

  // Softmax-Regression mit Abstandsstrafe zu den Original-Gewichten
  train(steps = 150, lr = 0.5, lambda = 0.005) {
    const { D, K, W, b, W0, b0 } = this;
    const n = this.samples.length;
    if (!n) return;
    W.set(W0); b.set(b0);
    // seltene Klassen (z. B. Könige) zählen mehr
    const cnt = new Float32Array(K);
    for (const s of this.samples) cnt[s.y]++;
    const wt = this.samples.map((s) => 1 / Math.sqrt(cnt[s.y]));
    const wsum = wt.reduce((a, c) => a + c, 0);
    const gW = new Float32Array(K * D), gb = new Float32Array(K), z = new Float32Array(K);
    // Schrittweite passend zur Größe der Merkmale (stabil für Softmax-Regression)
    let sq = 0; for (const s of this.samples) for (const v of s.e) sq += v * v;
    const step = n / Math.max(1e-6, sq);
    for (let it = 0; it < steps; it++) {
      gW.fill(0); gb.fill(0);
      this.samples.forEach((s, i) => {
        logits(W, b, s.e, D, K, z);
        softmaxInPlace(z);
        z[s.y] -= 1;
        const f = wt[i] / wsum;
        for (let k = 0; k < K; k++) {
          const g = z[k] * f;
          gb[k] += g;
          const o = k * D;
          for (let d = 0; d < D; d++) gW[o + d] += g * s.e[d];
        }
      });
      for (let i = 0; i < W.length; i++) W[i] -= lr * step * (gW[i] + lambda * (W[i] - W0[i]));
      for (let k = 0; k < K; k++) b[k] -= lr * (gb[k] + lambda * (b[k] - b0[k]));
    }
    this.trained = new Set(this.samples.map((s) => s.y)).size >= 8;
  }

  // Wahrscheinlichkeiten (13) für einen Merkmalsvektor
  probs(emb) {
    const z = new Float32Array(this.K);
    logits(this.W, this.b, emb, this.D, this.K, z);
    return softmaxInPlace(z);
  }

  save() {
    const s = {};
    for (const x of this.samples) (s[x.y] ||= []).push(encode(x.e));
    try { localStorage.setItem(ADAPT_KEY, JSON.stringify({ s })); } catch { /* Speicher voll oder gesperrt */ }
  }
  static load(net) { try { return new Adapter(net, JSON.parse(localStorage.getItem(ADAPT_KEY) || 'null')); } catch { return new Adapter(net); } }
  static forget() { try { localStorage.removeItem(ADAPT_KEY); localStorage.removeItem('schachkamera.netz.v1'); } catch { /* egal */ } }
}

function logits(W, b, e, D, K, z) {
  for (let k = 0; k < K; k++) { let s = b[k]; const o = k * D; for (let d = 0; d < D; d++) s += W[o + d] * e[d]; z[k] = s; }
}
function softmaxInPlace(z) {
  let m = -Infinity; for (const v of z) if (v > m) m = v;
  let t = 0; for (let k = 0; k < z.length; k++) { z[k] = Math.exp(z[k] - m); t += z[k]; }
  for (let k = 0; k < z.length; k++) z[k] /= t;
  return z;
}

// Merkmalsvektoren als Float32 in Base64 (für localStorage)
function encode(arr) {
  const b = new Uint8Array(Float32Array.from(arr).buffer);
  let s = '';
  for (let i = 0; i < b.length; i++) s += String.fromCharCode(b[i]);
  return btoa(s);
}
function decode(str) {
  const s = atob(str), b = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i);
  return new Float32Array(b.buffer);
}
