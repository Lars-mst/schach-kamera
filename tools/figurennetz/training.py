# Training des Figurennetzes. Aufruf: python training.py dom,rf,aca 14 netz.pt
# Erwartet die Ausschnitte von ausschnitte.py als ausschnitte_<quelle>_<split>.npz im Ordner DATA.
import os, sys, time, numpy as np, torch, torch.nn.functional as F
from netz import Net

DATA = os.environ.get('DATA', '.')
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(os.cpu_count() or 4)
srcs_train = sys.argv[1].split(',')
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 14
out = sys.argv[3] if len(sys.argv) > 3 else 'netz.pt'


def load(names, split):
    X, Y = [], []
    for n in names:
        p = f'{DATA}/ausschnitte_{n}_{split}.npz'
        if not os.path.exists(p): continue
        d = np.load(p); X.append(d['X']); Y.append(d['Y'])
    return np.concatenate(X), np.concatenate(Y)


Xtr, Ytr = load(srcs_train, 'train')
Xva, Yva = load(['dom', 'rf', 'aca'], 'valid')
print('train', Xtr.shape, np.bincount(Ytr), 'valid', Xva.shape)

def to_t(X): return torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255.0

def augment(x):
    n = x.shape[0]
    # Farbe/Helligkeit
    b = 1 + (torch.rand(n, 1, 1, 1) - 0.5) * 0.7
    c = 1 + (torch.rand(n, 1, 1, 1) - 0.5) * 0.7
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * c + m * b
    g = x.mean(1, keepdim=True)
    sat = 1 + (torch.rand(n, 1, 1, 1) - 0.5) * 0.8
    x = g + (x - g) * sat
    x = x + (torch.rand(n, 3, 1, 1) - 0.5) * 0.12
    # Spiegeln
    flip = torch.rand(n) < 0.5
    x[flip] = x[flip].flip(3)
    # kleine Verschiebung, Skalierung, Drehung
    ang = (torch.rand(n) - 0.5) * 0.25
    sc = 1 + (torch.rand(n) - 0.5) * 0.2
    tx = (torch.rand(n) - 0.5) * 0.16; ty = (torch.rand(n) - 0.5) * 0.12
    th = torch.zeros(n, 2, 3)
    th[:, 0, 0] = torch.cos(ang) * sc; th[:, 0, 1] = -torch.sin(ang) * sc * 2; th[:, 0, 2] = tx
    th[:, 1, 0] = torch.sin(ang) * sc / 2; th[:, 1, 1] = torch.cos(ang) * sc; th[:, 1, 2] = ty
    grid = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode='border', align_corners=False)
    # Unschärfe durch Verkleinern
    lo = torch.rand(n) < 0.4
    if lo.any():
        k = x[lo]; s = float(np.random.uniform(0.4, 0.8))
        k = F.interpolate(F.interpolate(k, scale_factor=s, mode='bilinear', align_corners=False), size=(64, 32), mode='bilinear', align_corners=False)
        x[lo] = k
    x = x + torch.randn_like(x) * 0.02
    return x.clamp(0, 1)

MEAN = 0.5
net = Net()
if os.environ.get('INIT'): net.load_state_dict(torch.load(os.environ['INIT']))  # weitertrainieren
LR = float(os.environ.get('LR', '3e-3'))
opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=1e-3)
steps = epochs * (len(Xtr) // 128)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps)
w = torch.tensor(1.0 / np.sqrt(np.bincount(Ytr, minlength=13) + 1), dtype=torch.float32); w = w / w.mean()
Xt, Yt = to_t(Xtr), torch.from_numpy(Ytr)
Xv, Yv = to_t(Xva), torch.from_numpy(Yva)

def evaluate():
    net.eval()
    with torch.no_grad():
        p = torch.cat([net(Xv[i:i + 1024] - MEAN).argmax(1) for i in range(0, len(Xv), 1024)])
    net.train()
    acc = (p == Yv).float().mean().item()
    # Figurenart bei bekannter Farbe (so nutzt es die App)
    occ = Yv < 12
    ty = ((p[occ] % 6) == (Yv[occ] % 6)) & (p[occ] < 12)
    return acc, ty.float().mean().item()

t0 = time.time()
for ep in range(epochs):
    perm = torch.randperm(len(Xt))
    tl = 0
    for i in range(0, len(perm) - 127, 128):
        idx = perm[i:i + 128]
        x = augment(Xt[idx].clone()) - MEAN
        loss = F.cross_entropy(net(x), Yt[idx], weight=w, label_smoothing=0.05)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tl += loss.item()
    print(ep, round(tl / (len(perm) // 128), 4), evaluate(), int(time.time() - t0), 's', flush=True)
torch.save(net.state_dict(), out)
