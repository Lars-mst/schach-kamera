// Rechnet das Figurennetz im Hintergrund, damit Kamerabild und Bedienung flüssig bleiben.
import { PieceNet } from './piecenet.js';

const ready = PieceNet.load('./model/');

self.onmessage = async (e) => {
  const { id, inputs, count } = e.data;
  try {
    const net = await ready;
    const size = 3 * net.inW * net.inH;
    let probs = null, embs = null;
    for (let i = 0; i < count; i++) {
      const r = net.run(inputs.subarray(i * size, (i + 1) * size));
      if (!probs) { probs = new Float32Array(count * r.probs.length); embs = new Float32Array(count * r.emb.length); }
      probs.set(r.probs, i * r.probs.length); embs.set(r.emb, i * r.emb.length);
    }
    self.postMessage({ id, probs, embs }, probs ? [probs.buffer, embs.buffer] : []);
  } catch (err) {
    self.postMessage({ id, error: String(err) });
  }
};
