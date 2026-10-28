# Erzeugt Trainingsausschnitte (wie in der App: Feld + eine Feldbreite nach oben) aus Detektionsdatensätzen.
# Aufruf: RAW=rohdaten DATA=ziel python ausschnitte.py dom|rf|aca train|valid|test
# Rohdaten (siehe README im Repo):
#   RAW/merged_<split>.parquet  huggingface.co/datasets/dopaul/chess-pieces-merged (Teil „dominique“)
#   RAW/rf/<split>/...           huggingface.co/datasets/Francesco/chess-pieces-mjzgj (dataset.tar.gz)
#   RAW/aca/dataset/...          huggingface.co/datasets/acapitani/chesspiece-detection-yolo
import sys, io, json, glob, os, random, math
import numpy as np, pandas as pd
from PIL import Image

RAW = os.environ.get('RAW', 'rohdaten')
OUT_W, OUT_H, UP = 32, 64, 1.0
TYPES = 'pnbrqk'
def cls_of(color, t): return (0 if color == 'w' else 6) + TYPES.index(t)
NAME = {'pawn': 'p', 'knight': 'n', 'bishop': 'b', 'rook': 'r', 'queen': 'q', 'king': 'k'}
def from_name(n):
    c, t = n.split('-'); return cls_of('w' if c == 'white' else 'b', NAME[t])

def crop(im, cx, by, sw, hsq):
    x0, x1 = cx - sw / 2, cx + sw / 2
    y0, y1 = by - hsq / 2 - UP * sw, by + hsq / 2
    return np.asarray(im.transform((OUT_W, OUT_H), Image.EXTENT, (x0, y0, x1, y1), Image.BILINEAR, fillcolor=(127, 127, 127)))

def samples(im, boxes, rng, variants):
    """boxes: list of (cls, x0, y0, x1, y1) in Pixeln"""
    if not boxes: return []
    ws = sorted(b[3] - b[1] for b in boxes)
    med = ws[len(ws) // 2]
    out = []
    bases = []
    for c, x0, y0, x1, y1 in boxes:
        bw = x1 - x0
        bases.append(((x0 + x1) / 2, y1 - 0.12 * bw))
    for (c, x0, y0, x1, y1), (bx, byy) in zip(boxes, bases):
        for _ in range(variants):
            sw = med * rng.uniform(1.15, 1.6)
            sw = max(sw, (x1 - x0) * 1.02)
            hsq = sw * rng.uniform(0.35, 1.0)
            out.append((crop(im, bx + rng.uniform(-0.08, 0.08) * sw, byy + rng.uniform(-0.12, 0.12) * hsq, sw, hsq), c))
    # schwierige leere Felder: direkt vor einer Figur (im Bild darunter). Der Ausschnitt ragt dort
    # nach oben in die Figur hinein, das Feld selbst ist aber leer.
    for (c, x0, y0, x1, y1), (bx, byy) in zip(boxes, bases):
        if rng.random() > 0.5: continue
        sw = med * rng.uniform(1.15, 1.6)
        hsq = sw * rng.uniform(0.35, 1.0)
        px, py = bx + rng.uniform(-0.1, 0.1) * sw, byy + hsq * rng.uniform(0.85, 1.15)
        if py > im.height or any(math.hypot(px - qx, (py - qy) * sw / hsq) < 0.8 * sw for qx, qy in bases): continue
        out.append((crop(im, px, py, sw, hsq), 12))
    # leere Felder: Punkte im Bereich der Figuren, an denen keine Figur steht
    xs = [b[0] for b in bases]; ys = [b[1] for b in bases]
    n_empty = max(2, int(len(boxes) * variants * 0.35))
    tries = 0
    while n_empty > 0 and tries < n_empty * 30:
        tries += 1
        sw = med * rng.uniform(1.15, 1.6)
        px = rng.uniform(min(xs) - sw, max(xs) + sw); py = rng.uniform(min(ys) - sw, max(ys) + sw)
        if px < 0 or py < 0 or px > im.width or py > im.height: continue
        if any(math.hypot(px - bx, py - byy) < 0.8 * sw for bx, byy in bases): continue
        hsq = sw * rng.uniform(0.35, 1.0)
        out.append((crop(im, px, py, sw, hsq), 12)); n_empty -= 1
    return out

def src_dominique(split):
    d = pd.read_parquet(f'{RAW}/merged_{split}.parquet')
    names = ['black-bishop','black-king','black-knight','black-pawn','black-queen','black-rook','white-bishop','white-king','white-knight','white-pawn','white-queen','white-rook']
    for _, r in d[d.source_dataset == 'chess_pieces_dominique'].iterrows():
        im = Image.open(io.BytesIO(r.image['bytes'])).convert('RGB'); W, H = im.size
        a = r.annotations
        yield im, [(from_name(names[c]), (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H) for c, x, y, w, h in zip(a['class_id'], a['x_center'], a['y_center'], a['width'], a['height'])]

def src_rf(split):
    split = {'valid': 'valid', 'train': 'train', 'test': 'test'}[split]
    d = json.load(open(f'{RAW}/rf/{split}/_annotations.coco.json'))
    cats = {c['id']: c['name'] for c in d['categories']}
    by = {}
    for a in d['annotations']: by.setdefault(a['image_id'], []).append(a)
    for img in d['images']:
        im = Image.open(f"{RAW}/rf/{split}/{img['file_name']}").convert('RGB')
        bx = []
        for a in by.get(img['id'], []):
            n = cats[a['category_id']]
            if '-' not in n: continue
            x, y, w, h = a['bbox']; bx.append((from_name(n), x, y, x + w, y + h))
        yield im, bx

ACA = ['wp','wr','wn','wb','wq','wk','bp','br','bn','bb','bq','bk']
def src_aca(split):
    sp = 'train' if split == 'train' else 'val'
    files = sorted(glob.glob(f'{RAW}/aca/**/images/{sp}/*', recursive=True))
    if split == 'test': return
    for f in files:
        lab = f.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        if not os.path.exists(lab): continue
        im = Image.open(f).convert('RGB'); W, H = im.size
        if max(W, H) > 1000:
            s = 1000 / max(W, H); im = im.resize((int(W * s), int(H * s)), Image.BILINEAR); W, H = im.size
        bx = []
        for line in open(lab):
            p = line.split()
            if len(p) < 5: continue
            c = int(p[0])
            if c >= 12: continue
            x, y, w, h = map(float, p[1:5])
            bx.append((cls_of(ACA[c][0], ACA[c][1]), (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
        yield im, bx

if __name__ == '__main__':
    name, split = sys.argv[1], sys.argv[2]
    variants = 4 if split == 'train' else 1
    rng = random.Random(hash((name, split)) & 0xffff)
    src = {'dom': src_dominique, 'rf': src_rf, 'aca': src_aca}[name](split)
    X, Y = [], []
    for im, bx in src or []:
        for a, c in samples(im, bx, rng, variants): X.append(a); Y.append(c)
    X = np.stack(X) if X else np.zeros((0, OUT_H, OUT_W, 3), np.uint8)
    np.savez_compressed(f"{os.environ.get('DATA', '.')}/ausschnitte_{name}_{split}.npz", X=X, Y=np.array(Y, np.int64))
    print(name, split, X.shape, np.bincount(Y, minlength=13))
