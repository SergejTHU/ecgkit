import numpy as np
import torch
from torchdiffeq import odeint       

device = torch.device(0)

class FHN_DDE:
    def __init__(self, params_array, fs=300):
        self.a1 = params_array[:, 0]
        self.a2 = params_array[:, 1]
        self.a3 = params_array[:, 2]
        self.b1 = params_array[:, 3]
        self.b2 = params_array[:, 4]
        self.b3 = params_array[:, 5]
        self.b4 = params_array[:, 6]
        self.c1 = params_array[:, 7]
        self.c2 = params_array[:, 8]
        self.c3 = params_array[:, 9]
        self.c4 = params_array[:, 10]
        self.d1 = params_array[:, 11]
        self.d2 = params_array[:, 12]
        self.d3 = params_array[:, 13]
        self.e1 = params_array[:, 14]
        self.e2 = params_array[:, 15]
        self.e3 = params_array[:, 16]
        self.f1 = params_array[:, 17]
        self.f2 = params_array[:, 18]
        self.f3 = params_array[:, 19]
        self.g1 = params_array[:, 20]
        self.g2 = params_array[:, 21]
        self.g3 = params_array[:, 22]
        self.g4 = params_array[:, 23]
        self.h1 = params_array[:, 24]
        self.h2 = params_array[:, 25]
        self.h3 = params_array[:, 26]
        self.h4 = params_array[:, 27]
        self.k1 = params_array[:, 28]
        self.k2 = params_array[:, 29]
        self.k3 = params_array[:, 30]
        self.k4 = params_array[:, 31]
        self.r1 = params_array[:, 32]
        self.r2 = params_array[:, 33]
        self.r3 = params_array[:, 34]
        self.r4 = params_array[:, 35]
        self.u11 = params_array[:, 36]
        self.u12 = params_array[:, 37]
        self.u21 = params_array[:, 38]
        self.u22 = params_array[:, 39]
        self.u31 = params_array[:, 40]
        self.u32 = params_array[:, 41]
        self.w11 = params_array[:, 42]
        self.w12 = params_array[:, 43]
        self.w21 = params_array[:, 44]
        self.w22 = params_array[:, 45]
        self.w31 = params_array[:, 46]
        self.w32 = params_array[:, 47]
        self.w41 = params_array[:, 48]
        self.w42 = params_array[:, 49]
        self.katde = 4 * 1e-5
        self.katre = 4 * 1e-5
        self.kvnde = 9 * 1e-5
        self.kvnre = 6 * 1e-5
        self.ksaav = self.f1.data
        self.kavhp = self.f1.data
        self.tsaav = 1.145 / self.f1.data + 0.04
        self.tavhp = 1.145 / self.f1.data + 0.04
        self.fs = fs
        self.size = params_array.shape[0]
        self.time_current = 0
        self._setup_delay()
        
    def _setup_delay(self):
        self.delay_saav = (self.tsaav * self.fs).data.detach().cpu().numpy().astype(np.int)
        self.delay_avhp = (self.tavhp * self.fs).data.detach().cpu().numpy().astype(np.int)
        self.delay_pool_saav = []
        self.delay_pool_avhp = []
        for i in range(self.size):
            self.delay_pool_saav.append([0 for _ in range(int(self.delay_saav[i]))])
            self.delay_pool_avhp.append([0 for _ in range(int(self.delay_avhp[i]))])
    
    def __call__(self, ts, xs):
        # [x1, y1, x2, y2, x3, y3, z1, v1, z2, v2, z3, v3, z4, v4]
        x1 = xs[:, 0]
        y1 = xs[:, 1]
        x2 = xs[:, 2]
        y2 = xs[:, 3]
        x3 = xs[:, 4]
        y3 = xs[:, 5]
        z1 = xs[:, 6]
        v1 = xs[:, 7]
        z2 = xs[:, 8]
        v2 = xs[:, 9]
        z3 = xs[:, 10]
        v3 = xs[:, 11]
        z4 = xs[:, 12]
        v4 = xs[:, 13]
        y1_latest = y1.data.detach().cpu().numpy()
        y2_latest = y2.data.detach().cpu().numpy()
        for i in range(self.size):
            self.delay_pool_saav[i].append(y1_latest[i])
            self.delay_pool_avhp[i].append(y2_latest[i])
        y1_delay = torch.FloatTensor([self.delay_pool_saav[i][self.time_current]]).to(device)
        y2_delay = torch.FloatTensor([self.delay_pool_avhp[i][self.time_current]]).to(device)
        self.time_current += 1
        c1 = y1
        c2 = -torch.mul(torch.mul(self.a1, y1), torch.mul(x1 - self.u11, x1 - self.u12)) - torch.mul(torch.mul(self.f1, x1), torch.mul(x1 + self.d1, x1 + self.e1))
        c3 = y2
        c4 = -torch.mul(torch.mul(self.a2, y2), torch.mul(x2 - self.u21, x2 - self.u22)) - torch.mul(torch.mul(self.f2, x2), torch.mul(x2 + self.d2, x2 + self.e2)) + torch.mul(self.ksaav, (y1_delay - y2))
        c5 = y3
        c6 = -torch.mul(torch.mul(self.a3, y3), torch.mul(x3 - self.u31, x3 - self.u32)) - torch.mul(torch.mul(self.f3, x3), torch.mul(x3 + self.d3, x3 + self.e3)) + torch.mul(self.kavhp, (y2_delay - y3))
        c7 = torch.mul(self.k1, (-torch.mul(torch.mul(self.c1, z1), torch.mul(z1-self.w11, z1-self.w12)) - torch.mul(self.b1, v1) - torch.mul(self.r1, torch.mul(v1, z1)) + self.katde * torch.relu(y1)))
        c8 = torch.mul(self.k1, (torch.mul(self.h1, z1 - torch.mul(self.g1, v1))))
        c9 = torch.mul(self.k2, (-torch.mul(torch.mul(self.c2, z2), torch.mul(z2-self.w21, z2-self.w22)) - torch.mul(self.b2, v2) - torch.mul(self.r2, torch.mul(v2, z2)) + self.katre * torch.relu(-y1)))
        c10 = torch.mul(self.k2, (torch.mul(self.h2, z2 - torch.mul(self.g2, v2))))
        c11 = torch.mul(self.k3, (-torch.mul(torch.mul(self.c3, z3), torch.mul(z3-self.w31, z3-self.w32)) - torch.mul(self.b3, v3) - torch.mul(self.r3, torch.mul(v3, z3)) + self.kvnde * torch.relu(y3)))
        c12 = torch.mul(self.k3, (torch.mul(self.h3, z3 - torch.mul(self.g3, v3))))
        c13 = torch.mul(self.k4, (-torch.mul(torch.mul(self.c4, z4), torch.mul(z4-self.w41, z4-self.w42)) - torch.mul(self.b4, v4) - torch.mul(self.r4, torch.mul(v4, z4)) + self.kvnre * torch.relu(-y3)))
        c14 = torch.mul(self.k4, (torch.mul(self.h4, z4 - torch.mul(self.g4, v4))))
        xs_dr = torch.vstack([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14]).T
        return xs_dr
    
def generate_ecg_fhn(fs, length_seconds, ecg_type='normal'):
    if ecg_type != 'normal':
        raise NotImplementedError
    x0 = torch.FloatTensor([[0 for _ in range(14)]]).to(device)
    x0[0] = 0.1
    ts = torch.FloatTensor(np.linspace(0, 2 * length_seconds, num=length_seconds*fs)).to(device)
    param_standard = [40, 50, 50, 0, 0, 0.015, 0, 0.26, 0.26, 0.12, 0.1, 3, 3, 3, 3.5, 5, 12, 22, 8.4, 1.5, 1, 1, 1, 1, 0.004, 0.004, 0.008, 0.008, 2000, 400, 10000, 2000, 0.4, 0.4, 0.09, 0.1, 0.83, -0.83, 0.83, -0.83, 0.83, -0.83, 0.13, 1, 0.19, 1, 0.12, 1.1, 0.22, 0.8]
    param_standard = torch.FloatTensor([param_standard]).to(device)
    fhn_dde = FHN_DDE(param_standard)
    xs = odeint(fhn_dde, x0, ts, method='rk4')
    xs_np = xs.data.detach().cpu().numpy()
    xs_np = xs_np.squeeze()
    ys = xs_np[:, 6] - xs_np[:, 8] + xs_np[:, 10] + xs_np[:, 12] + 0.2
    ys_ans = ys[400:]
    return ys_ans