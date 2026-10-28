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

  // Ein Ausschnitt -> { probs (13), emb (Merkmalsvektor, Länge 1) }
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
        let n2 = 0; for (const v of emb) n2 += v * v;
        const nn = Math.sqrt(n2) || 1;
        for (let i = 0; i < emb.length; i++) emb[i] /= nn;
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

  // Wie classify, aber im Hintergrund (Web Worker). Ohne Worker rechnet es hier.
  classifyAsync(img, project, squares) {
    if (!this.worker) {
      try {
        this.worker = new Worker(new URL('./networker.js', import.meta.url), { type: 'module' });
        this.pending = new Map();
        this.nextId = 1;
        this.worker.onmessage = (e) => { const p = this.pending.get(e.data.id); this.pending.delete(e.data.id); p?.(e.data); };
        this.worker.onerror = () => { this.worker = false; for (const p of this.pending.values()) p({ error: 'worker' }); this.pending.clear(); };
      } catch { this.worker = false; }
    }
    if (this.worker === false || !squares.length) return Promise.resolve(this.classify(img, project, squares));
    const size = 3 * this.inW * this.inH;
    const inputs = new Float32Array(squares.length * size);
    squares.forEach((sq, i) => this.crop(img, project, sq, inputs.subarray(i * size, (i + 1) * size)));
    const id = this.nextId++;
    return new Promise((resolve) => {
      this.pending.set(id, (d) => {
        if (d.error) { resolve(this.classify(img, project, squares)); return; }
        const out = new Map(), k = d.probs.length / squares.length, e = d.embs.length / squares.length;
        squares.forEach((sq, i) => out.set(sq, { probs: d.probs.subarray(i * k, (i + 1) * k), emb: d.embs.subarray(i * e, (i + 1) * e) }));
        resolve(out);
      });
      this.worker.postMessage({ id, inputs, count: squares.length }, [inputs.buffer]);
    });
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

// Merkmalsvektoren des eigenen Figurensets: aus bestätigten Stellungen gelernt. So passt sich das
// allgemeine Netz an deine Figuren, dein Brett und dein Licht an (nächster Nachbar je Figurenart).
const PROTO_KEY = 'schachkamera.netz.v1';
const MAX_PROTOS = 24;

export class ProtoStore {
  constructor(data) {
    this.t = { w: {}, b: {} };
    for (const c of ['w', 'b']) for (const ty of TYPES) this.t[c][ty] = (data?.t?.[c]?.[ty] || []).map(decode);
  }

  ready(color) { return TYPES.every((ty) => this.t[color][ty].length > 0); }

  add(color, type, emb) {
    const list = this.t[color][type];
    list.push(Float32Array.from(emb));
    if (list.length > MAX_PROTOS) list.shift();
  }

  // Wahrscheinlichkeit je Figurenart aus der Ähnlichkeit zu den gelernten Beispielen
  probs(color, emb) {
    const sims = TYPES.map((ty) => {
      let best = -1;
      for (const p of this.t[color][ty]) { let s = 0; for (let i = 0; i < p.length; i++) s += p[i] * emb[i]; if (s > best) best = s; }
      return best;
    });
    const m = Math.max(...sims);
    const e = sims.map((s) => Math.exp((s - m) * 25));
    const z = e.reduce((a, b) => a + b, 0);
    const out = {};
    TYPES.forEach((ty, i) => { out[ty] = e[i] / z; });
    return out;
  }

  toJSON() {
    const t = { w: {}, b: {} };
    for (const c of ['w', 'b']) for (const ty of TYPES) t[c][ty] = this.t[c][ty].map(encode);
    return { t };
  }

  save() { try { localStorage.setItem(PROTO_KEY, JSON.stringify(this.toJSON())); } catch { /* Speicher voll oder gesperrt */ } }
  static load() { try { return new ProtoStore(JSON.parse(localStorage.getItem(PROTO_KEY) || 'null')); } catch { return new ProtoStore(); } }
  static forget() { try { localStorage.removeItem(PROTO_KEY); } catch { /* egal */ } }
}

// Figurenart bei bekannter Farbe: Netz und eigenes Set zusammen (gewichtetes geometrisches Mittel;
// das Netz zählt mehr, das eigene Set gibt den Ausschlag, wenn das Netz schwankt)
export function combinedTypeProbs(probs, emb, color, protos) {
  const net = typeProbs(probs, color);
  if (!protos || !protos.ready(color)) return net;
  const own = protos.probs(color, emb);
  const out = {};
  let z = 0;
  for (const ty of TYPES) { out[ty] = Math.max(1e-4, net[ty]) ** 0.7 * Math.max(1e-4, own[ty]) ** 0.3; z += out[ty]; }
  for (const ty of TYPES) out[ty] /= z;
  return out;
}

function encode(arr) {
  let s = '';
  for (let i = 0; i < arr.length; i++) s += String.fromCharCode(Math.max(0, Math.min(255, Math.round(arr[i] * 127 + 128))));
  return btoa(s);
}
function decode(str) {
  const s = atob(str), out = new Float32Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = (s.charCodeAt(i) - 128) / 127;
  return out;
}
