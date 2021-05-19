from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from torchdiffeq import odeint


class ECGDynamicSystem(metaclass=ABCMeta):
    def __init__(self, params_array, param_h, device, fs=300):
        super(ECGDynamicSystem, self).__init__()
        self._initialize(params_array, param_h, device, fs=300)
        
    @abstractmethod
    def _initialize(self, **params):
        pass
    
    @abstractmethod
    def forward(self, ts, xs):
        pass
    
    @abstractmethod
    def backward(self, ts, ls):
        pass
    
    @abstractmethod
    def sensitivity(self):
        pass
    
    @abstractmethod
    def df_dx(self, xs):
        pass
    
    @abstractmethod
    def df_dp(self, xs):
        pass
    
    @abstractmethod
    def step(self, length_seconds, x0, ys_true):
        pass


class FHN_DDE(ECGDynamicSystem):
    def _initialize(self, params_array, param_h, device, fs=300):
        self.a1 = params_array[0]
        self.a2 = params_array[1]
        self.a3 = params_array[2]
        self.b1 = params_array[3]
        self.b2 = params_array[4]
        self.b3 = params_array[5]
        self.b4 = params_array[6]
        self.c1 = params_array[7]
        self.c2 = params_array[8]
        self.c3 = params_array[9]
        self.c4 = params_array[10]
        self.d1 = params_array[11]
        self.d2 = params_array[12]
        self.d3 = params_array[13]
        self.e1 = params_array[14]
        self.e2 = params_array[15]
        self.e3 = params_array[16]
        self.f1 = params_array[17]
        self.f2 = params_array[18]
        self.f3 = params_array[19]
        self.g1 = params_array[20]
        self.g2 = params_array[21]
        self.g3 = params_array[22]
        self.g4 = params_array[23]
        self.h1 = params_array[24]
        self.h2 = params_array[25]
        self.h3 = params_array[26]
        self.h4 = params_array[27]
        self.k1 = params_array[28]
        self.k2 = params_array[29]
        self.k3 = params_array[30]
        self.k4 = params_array[31]
        self.r1 = params_array[32]
        self.r2 = params_array[33]
        self.r3 = params_array[34]
        self.r4 = params_array[35]
        self.u11 = params_array[36]
        self.u12 = params_array[37]
        self.u21 = params_array[38]
        self.u22 = params_array[39]
        self.u31 = params_array[40]
        self.u32 = params_array[41]
        self.w11 = params_array[42]
        self.w12 = params_array[43]
        self.w21 = params_array[44]
        self.w22 = params_array[45]
        self.w31 = params_array[46]
        self.w32 = params_array[47]
        self.w41 = params_array[48]
        self.w42 = params_array[49]
        self.katde = 4 * 1e-5
        self.katre = 4 * 1e-5
        self.kvnde = 9 * 1e-5
        self.kvnre = 6 * 1e-5
        self.ksaav = self.f1.data
        self.kavhp = self.f1.data
        self.tsaav = ((2.29 / self.f1.data) + 0.08) / 2
        self.tavhp = ((2.29 / self.f1.data) + 0.08) / 2
        self.fs = fs
        self.fs_forward = 4 * fs
        self.param_h = param_h
        self.time_current = 0
        self.ts = []
        self.fx = []
        self.fp = []
        self.xs = []
        self.ys = []
        self.ys_true = None
        self.lmda = None
        self.device = device
        self._setup_delay()
        
    def _setup_delay(self):
        self.delay_saav = (self.tsaav * self.fs_forward).data.detach().cpu().numpy().astype(np.int)
        self.delay_avhp = (self.tavhp * self.fs_forward).data.detach().cpu().numpy().astype(np.int)
        self.delay_pool_saav = [0 for _ in range(int(self.delay_saav))]
        self.delay_pool_avhp = [0 for _ in range(int(self.delay_avhp))]
    
    def forward(self, ts, xs):
        # [x1, y1, x2, y2, x3, y3, z1, v1, z2, v2, z3, v3, z4, v4]
        x1 = xs[0]
        y1 = xs[1]
        x2 = xs[2]
        y2 = xs[3]
        x3 = xs[4]
        y3 = xs[5]
        z1 = xs[6]
        v1 = xs[7]
        z2 = xs[8]
        v2 = xs[9]
        z3 = xs[10]
        v3 = xs[11]
        z4 = xs[12]
        v4 = xs[13]
        self.delay_pool_saav.append(y1.data.detach().cpu().numpy())
        self.delay_pool_avhp.append(y2.data.detach().cpu().numpy())
        y1_delay = torch.FloatTensor(np.array([self.delay_pool_saav[self.time_current]])).to(self.device)[0]
        y2_delay = torch.FloatTensor(np.array([self.delay_pool_avhp[self.time_current]])).to(self.device)[0]
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
        self.fx.append(self.df_dx(xs))
        self.fp.append(self.df_dp(xs))
        self.ts.append(ts.data.detach().cpu().numpy())
        self.xs.append(xs)
        self.ys.append(torch.dot(xs, self.param_h))
        xs_dr = torch.FloatTensor([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14]).to(self.device)
        return xs_dr
    
    def backward(self, ts, ls):
        ts_index = torch.argmin(torch.square(ts - torch.FloatTensor(np.array(self.ts)).to(self.device)))
        p1 = -torch.matmul(ls.reshape((1, 14)), self.fx[ts_index])
        p2 = self.param_h * (self.ys[ts_index] - self.ys_true[int(ts_index / 4)])
        ans = p1 + p2.reshape((1, 14))
        return ans
    
    def sensitivity(self):
        z = torch.zeros(64).to(self.device)
        z[50:] = -self.lmda[0, :]
        inters = []
        length = self.lmda.shape[0]
        for i in range(length):
            tmp = -torch.matmul(self.lmda[i, :].reshape((1, 14)), self.fp[int(4*i)])
            inters.append(tmp)
        inters = torch.vstack(inters)
        summ = inters.sum(0)
        return z + summ / self.fs
    
    def df_dx(self, xs):
        x1 = xs[0]
        y1 = xs[1]
        x2 = xs[2]
        y2 = xs[3]
        x3 = xs[4]
        y3 = xs[5]
        z1 = xs[6]
        v1 = xs[7]
        z2 = xs[8]
        v2 = xs[9]
        z3 = xs[10]
        v3 = xs[11]
        z4 = xs[12]
        v4 = xs[13]
        ans = torch.zeros((14, 14)).to(self.device)       
        ans[0, 1] = 1
        ans[2, 3] = 1
        ans[4, 5] = 1
        ans[1, 0] = -torch.mul(torch.mul(self.a1, y1), (2 * x1 - self.u11 - self.u12)) - torch.mul(self.f1, torch.mul(x1 + self.d1, x1 + self.e1)) - torch.mul(torch.mul(self.f1, x1), 2 * x1 + self.d1 + self.e1)
        ans[1, 1] = -torch.mul(self.a1, torch.mul(x1 - self.u11, x1 - self.u12))
        ans[3, 2] = -torch.mul(torch.mul(self.a2, y2), (2 * x2 - self.u21 - self.u22)) - torch.mul(self.f2, torch.mul(x2 + self.d2, x2 + self.e2)) - torch.mul(torch.mul(self.f2, x2), 2 * x2 + self.d2 + self.e2)
        ans[3, 3] = -torch.mul(self.a2, torch.mul(x2 - self.u21, x2 - self.u22)) - self.ksaav
        ans[5, 4] = -torch.mul(torch.mul(self.a3, y3), (2 * x3 - self.u31 - self.u32)) - torch.mul(self.f3, torch.mul(x3 + self.d3, x3 + self.e3)) - torch.mul(torch.mul(self.f3, x3), 2 * x3 + self.d3 + self.e3)
        ans[5, 5] = -torch.mul(self.a3, torch.mul(x3 - self.u31, x3 - self.u32)) - self.kavhp
        ans[6, 6] = torch.mul(self.k1, (-torch.mul(self.c1, torch.mul(z1 - self.w11, z1 - self.w12)) - torch.mul(torch.mul(self.c1, z1), 2 * z1 - self.w11 - self.w12) - torch.mul(self.r1, v1)))
        ans[6, 7] = torch.mul(self.k1, (-self.b1 - torch.mul(self.r1, z1)))
        ans[6, 1] = self.katde * (torch.sign(y1) + 1) / 2
        ans[8, 8] = torch.mul(self.k2, (-torch.mul(self.c2, torch.mul(z2 - self.w21, z1 - self.w22)) - torch.mul(torch.mul(self.c2, z2), 2 * z2 - self.w21 - self.w22) - torch.mul(self.r2, v2)))
        ans[8, 9] = torch.mul(self.k2, (-self.b2 - torch.mul(self.r2, z2)))
        ans[8, 1] = self.katre * (torch.sign(y1) - 1) / 2
        ans[10, 10] = torch.mul(self.k3, (-torch.mul(self.c3, torch.mul(z3 - self.w31, z3 - self.w32)) - torch.mul(torch.mul(self.c3, z3), 2 * z3 - self.w31 - self.w32) - torch.mul(self.r3, v3)))
        ans[10, 11] = torch.mul(self.k3, (-self.b3 - torch.mul(self.r3, z3)))
        ans[10, 5] = self.kvnde * (torch.sign(y3) + 1) / 2
        ans[12, 12] = torch.mul(self.k4, (-torch.mul(self.c4, torch.mul(z4 - self.w41, z4 - self.w42)) - torch.mul(torch.mul(self.c4, z4), 2 * z4 - self.w41 - self.w42) - torch.mul(self.r4, v4)))
        ans[12, 13] = torch.mul(self.k4, (-self.b4 - torch.mul(self.r4, z4)))
        ans[12, 5] = self.kvnre * (torch.sign(y3) - 1) / 2
        ans[7, 6] = torch.mul(self.k1, self.h1)
        ans[7, 7] = -torch.mul(self.k1, torch.mul(self.h1, self.g1))
        ans[9, 8] = torch.mul(self.k2, self.h2)
        ans[9, 9] = -torch.mul(self.k2, torch.mul(self.h2, self.g2))
        ans[11, 10] = torch.mul(self.k3, self.h3)
        ans[11, 11] = -torch.mul(self.k3, torch.mul(self.h3, self.g3))
        ans[13, 12] = torch.mul(self.k4, self.h4)
        ans[13, 13] = -torch.mul(self.k4, torch.mul(self.h4, self.g4))   
        return ans
    
    def df_dp(self, xs):
        x1 = xs[0]
        y1 = xs[1]
        x2 = xs[2]
        y2 = xs[3]
        x3 = xs[4]
        y3 = xs[5]
        z1 = xs[6]
        v1 = xs[7]
        z2 = xs[8]
        v2 = xs[9]
        z3 = xs[10]
        v3 = xs[11]
        z4 = xs[12]
        v4 = xs[13]
        ans = torch.zeros((14, 64)).to(self.device)
        ans[1, 0] = -torch.mul(y1, torch.mul(x1 - self.u11, x1 - self.u12))
        ans[1, 36] = torch.mul(torch.mul(self.a1, y1), x1 - self.u12)
        ans[1, 37] = torch.mul(torch.mul(self.a1, y1), x1 - self.u11)
        ans[1, 17] = -torch.mul(x1, torch.mul(x1 + self.d1, x1 + self.e1))
        ans[1, 11] = -torch.mul(torch.mul(self.f1, x1), x1 + self.e1)
        ans[1, 14] = -torch.mul(torch.mul(self.f1, x1), x1 + self.d1)
        ans[3, 1] = -torch.mul(y2, torch.mul(x2 - self.u21, x2 - self.u22))
        ans[3, 38] = torch.mul(torch.mul(self.a2, y2), x2 - self.u22)
        ans[3, 39] = torch.mul(torch.mul(self.a2, y2), x2 - self.u21)
        ans[3, 18] = -torch.mul(x2, torch.mul(x2 + self.d2, x2 + self.e2))
        ans[3, 12] = -torch.mul(torch.mul(self.f2, x2), x2 + self.e2)
        ans[3, 15] = -torch.mul(torch.mul(self.f2, x2), x2 + self.d2)
        ans[5, 2] = -torch.mul(y3, torch.mul(x3 - self.u31, x3 - self.u32))
        ans[5, 40] = torch.mul(torch.mul(self.a3, y3), x3 - self.u32)
        ans[5, 41] = torch.mul(torch.mul(self.a3, y3), x3 - self.u31)
        ans[5, 19] = -torch.mul(x3, torch.mul(x3 + self.d3, x3 + self.e3))
        ans[5, 13] = -torch.mul(torch.mul(self.f3, x3), x3 + self.e3)
        ans[5, 16] = -torch.mul(torch.mul(self.f3, x3), x3 + self.d3)
        ans[6, 28] = -torch.mul(torch.mul(self.c1, z1), torch.mul(z1-self.w11, z1-self.w12)) - torch.mul(self.b1, v1) - torch.mul(self.r1, torch.mul(v1, z1)) + self.katde * torch.relu(y1)
        ans[6, 7] = -torch.mul(self.k1, torch.mul(z1, torch.mul(z1 - self.w11, z1 - self.w12)))
        ans[6, 42] = torch.mul(self.k1, torch.mul(self.c1, torch.mul(z1, z1 - self.w12)))
        ans[6, 43] = torch.mul(self.k1, torch.mul(self.c1, torch.mul(z1, z1 - self.w11)))
        ans[6, 3] = -torch.mul(self.k1, v1)
        ans[6, 32] = -torch.mul(self.k1, torch.mul(v1, z1))
        ans[8, 29] = -torch.mul(torch.mul(self.c2, z2), torch.mul(z2-self.w21, z2-self.w22)) - torch.mul(self.b2, v2) - torch.mul(self.r2, torch.mul(v2, z2)) + self.katre * torch.relu(-y1)
        ans[8, 8] = -torch.mul(self.k2, torch.mul(z2, torch.mul(z2 - self.w21, z2 - self.w22)))
        ans[8, 44] = torch.mul(self.k2, torch.mul(self.c2, torch.mul(z2, z2 - self.w22)))
        ans[8, 45] = torch.mul(self.k2, torch.mul(self.c2, torch.mul(z2, z2 - self.w21)))
        ans[8, 4] = -torch.mul(self.k2, v2)
        ans[8, 33] = -torch.mul(self.k2, torch.mul(v2, z2))
        ans[10, 30] = -torch.mul(torch.mul(self.c3, z3), torch.mul(z3-self.w31, z3-self.w32)) - torch.mul(self.b3, v3) - torch.mul(self.r3, torch.mul(v3, z3)) + self.kvnde * torch.relu(y3)
        ans[10, 9] = -torch.mul(self.k3, torch.mul(z3, torch.mul(z3 - self.w31, z3 - self.w32)))
        ans[10, 46] = torch.mul(self.k3, torch.mul(self.c3, torch.mul(z3, z3 - self.w32)))
        ans[10, 47] = torch.mul(self.k3, torch.mul(self.c3, torch.mul(z3, z3 - self.w31)))
        ans[10, 5] = -torch.mul(self.k3, v3)
        ans[10, 34] = -torch.mul(self.k3, torch.mul(v3, z3))
        ans[12, 31] = -torch.mul(torch.mul(self.c4, z4), torch.mul(z4-self.w41, z4-self.w42)) - torch.mul(self.b4, v4) - torch.mul(self.r4, torch.mul(v4, z4)) + self.kvnre * torch.relu(-y3)
        ans[12, 10] = -torch.mul(self.k4, torch.mul(z4, torch.mul(z4 - self.w41, z4 - self.w42)))
        ans[12, 48] = torch.mul(self.k4, torch.mul(self.c4, torch.mul(z4, z4 - self.w42)))
        ans[12, 49] = torch.mul(self.k4, torch.mul(self.c4, torch.mul(z4, z4 - self.w41)))
        ans[12, 6] = -torch.mul(self.k4, v4)
        ans[12, 35] = -torch.mul(self.k4, torch.mul(v4, z4))
        ans[7, 28] = torch.mul(self.h1, z1 - torch.mul(self.g1, v1))
        ans[7, 20] = -torch.mul(self.k1, torch.mul(self.h1, v1))
        ans[7, 24] = torch.mul(self.k1, z1 - torch.mul(self.g1, v1))
        ans[9, 29] = torch.mul(self.h2, z2 - torch.mul(self.g2, v2))
        ans[9, 21] = -torch.mul(self.k2, torch.mul(self.h2, v2))
        ans[9, 25] = torch.mul(self.k2, z2 - torch.mul(self.g2, v2))
        ans[11, 30] = torch.mul(self.h3, z3 - torch.mul(self.g3, v3))
        ans[11, 22] = -torch.mul(self.k3, torch.mul(self.h3, v3))
        ans[11, 26] = torch.mul(self.k3, z3 - torch.mul(self.g3, v3))
        ans[13, 31] = torch.mul(self.h4, z4 - torch.mul(self.g4, v4))
        ans[13, 23] = -torch.mul(self.k4, torch.mul(self.h4, v4))
        ans[13, 27] = torch.mul(self.k4, z4 - torch.mul(self.g4, v4))
        return ans
    
    def step(self, length_seconds, x0, ys_true):
        self.ys_true = ys_true
        ts = torch.FloatTensor(np.linspace(0, length_seconds, num=length_seconds*self.fs+1)).to(self.device)
        xs = odeint(self.forward, x0, ts, method='rk4')
        ys_pred = torch.matmul(xs, self.param_h)[:-1]
        loss = torch.sum(torch.square(ys_pred - ys_true)) / self.fs
        l0 = torch.FloatTensor([0 for _ in range(14)]).to(self.device)
        ts_l = torch.FloatTensor(length_seconds - np.linspace(0, length_seconds, num=length_seconds*self.fs+1)).to(self.device)
        ls_reverse = odeint(self.backward, l0, ts_l, method='rk4')
        ls_reverse_np = ls_reverse.data.detach().cpu().numpy()[::-1][:-1, :]
        self.lmda = torch.FloatTensor(ls_reverse_np.copy()).to(self.device)
        sens = self.sensitivity()
        ratio = torch.sqrt(torch.sum(torch.square(ys_true)) / torch.sum(torch.square(ys_pred)))
        return sens[:50], sens[50:], ratio, loss, ys_pred
        