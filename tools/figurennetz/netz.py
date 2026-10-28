# Netzarchitektur (gemeinsam für Training und Export)
import os
import torch.nn as nn
import torch.nn.functional as F

# Kanäle je Schicht (Standard: das Netz in src/model)
CH = [int(c) for c in os.environ.get('NETZ_CH', '16,32,32,64,64,128').split(',')]


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        def cb(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU())
        s.b1 = cb(3, CH[0]); s.b2 = cb(CH[0], CH[1]); s.b3 = cb(CH[1], CH[2])
        s.b4 = cb(CH[2], CH[3]); s.b5 = cb(CH[3], CH[4]); s.b6 = cb(CH[4], CH[5])
        s.drop = nn.Dropout(0.2); s.fc = nn.Linear(CH[5], 13)

    def embed(s, x):
        x = F.max_pool2d(s.b1(x), 2)
        x = F.max_pool2d(s.b3(s.b2(x)), 2)
        x = F.max_pool2d(s.b5(s.b4(x)), 2)
        return s.b6(x).mean((2, 3))

    def forward(s, x): return s.fc(s.drop(s.embed(x)))


POOL_AFTER = {'b1': True, 'b2': False, 'b3': True, 'b4': False, 'b5': True, 'b6': False}
