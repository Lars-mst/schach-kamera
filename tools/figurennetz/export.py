# Gewichte für die App exportieren (BatchNorm eingerechnet, Float32 little endian).
# Aufruf: python export.py netz.pt ../../src/model
import sys, json, numpy as np, torch
from netz import Net, CH, POOL_AFTER

pt, outdir = sys.argv[1], sys.argv[2]
net = Net(); net.load_state_dict(torch.load(pt, map_location='cpu')); net.eval()
arrs, layers, off = [], [], 0
for name in ['b1', 'b2', 'b3', 'b4', 'b5', 'b6']:
    conv, bn = getattr(net, name)[0], getattr(net, name)[1]
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = (conv.weight * s[:, None, None, None]).detach().numpy().astype(np.float32).ravel()
    b = (bn.bias - bn.running_mean * s).detach().numpy().astype(np.float32)
    layers.append({'type': 'conv', 'out': int(conv.out_channels), 'pool': POOL_AFTER[name], 'w': off, 'wn': int(w.size), 'b': off + int(w.size)})
    arrs += [w, b]; off += w.size + b.size
layers.append({'type': 'gap'})
w = net.fc.weight.detach().numpy().astype(np.float32).ravel(); b = net.fc.bias.detach().numpy().astype(np.float32)
layers.append({'type': 'fc', 'out': 13, 'w': off, 'wn': int(w.size), 'b': off + int(w.size)})
arrs += [w, b]; off += w.size + b.size
np.concatenate(arrs).astype('<f4').tofile(f'{outdir}/figuren.bin')
meta = {'version': 1, 'classes': 'PNBRQKpnbrqk.', 'input': {'w': 32, 'h': 64, 'up': 1.0}, 'maxAct': 64 * 32 * max(CH), 'layers': layers, 'params': int(off)}
json.dump(meta, open(f'{outdir}/figuren.json', 'w'))
print('Gewichte', off, 'Bytes', off * 4)
