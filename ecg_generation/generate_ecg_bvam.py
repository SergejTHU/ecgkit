import numpy as np
import torch
from torchdiffeq import odeint  


class ODE_Params:
    def __init__(self, a, c, h, g, b):
        self.a = a
        self.c = c
        self.h = h
        self.g = g
        self.b = b
        
    @property    
    def p(self):
        return np.concatenate([self.a, np.array([self.c, self.h, self.g, self.b])])


class ODE_Params_List:
    def __init__(self, params_list=None, params_array=None):
        if params_list is not None and params_array is None:
            self.params_list = params_list
            self._generate_array()
        else:
            self.params_array = params_array
            self._generate_list()
        self.params_array = torch.FloatTensor(np.array(self.params_array))
        
    def _generate_array(self):
        self.params_array = []
        for p in self.params_list:
            self.params_array.append(p.p)
        
    def _generate_list(self):
        self.params_list = []
        p = self.params_array
        for i in range(p.shape[0]):
            pi = ODE_Params(p[i, :4], p[i, 4], p[i, 5], p[i, 6], p[i, 7])
            self.params_list.append(pi)
    
    @property
    def a(self):
        return self.params_array[:, :4]
    
    @property
    def c(self):
        return self.params_array[:, 4]
        
    @property
    def h(self):
        return self.params_array[:, 5]
        
    @property
    def g(self):
        return self.params_array[:, 6]
        
    @property
    def b(self):
        return self.params_array[:, 7]
    
    @property
    def size(self):
        return len(self.params_list)


def func_rs(xs, params, debug=False):
    """
    Should be batched
    x: torch.Tensor, (batch_size, 4)
    """
    assert isinstance(params, ODE_Params_List)
    c1 = xs[:, 0] - xs[:, 1] - torch.mul(params.c, torch.mul(xs[:, 0], xs[:, 1])) - torch.mul(xs[:, 0], torch.square(xs[:, 1]))
    c1 = torch.mul(c1, params.g)
    c2 = torch.mul(params.h, xs[:, 0]) - torch.mul(3, xs[:, 1]) + torch.mul(params.c, torch.mul(xs[:, 0], xs[:, 1])) + torch.mul(xs[:, 0], torch.square(xs[:, 1])) + torch.mul(params.b, (xs[:, 3] - xs[:, 1]))
    c2 = torch.mul(c2, params.g)
    c3 = xs[:, 2] - xs[:, 3] - torch.mul(params.c, torch.mul(xs[:, 2], xs[:, 3])) - torch.mul(xs[:, 2], torch.square(xs[:, 3]))
    c3 = torch.mul(c3, params.g)
    c4 = torch.mul(params.h, xs[:, 2]) - torch.mul(3, xs[:, 3]) + torch.mul(params.c, torch.mul(xs[:, 2], xs[:, 3])) + torch.mul(xs[:, 2], torch.square(xs[:, 3])) + torch.mul(2, torch.mul(params.b, (xs[:, 1] - xs[:, 3])))
    c4 = torch.mul(c4, params.g)
    xs_dr = torch.vstack([c1, c2, c3, c4]).T
    return xs_dr


def calc_ecg(xs, params):
    """
    xs: torch.Tensor, (length, batch_size, 4)
    """
    s = torch.mul(xs, params.a)
    v = s.sum(dim=2).T
    return v


def func_gen(params):
    def func(t, y):
        return func_rs(y, params)
    return func


def forward(params, length_seconds=10, fs=300, debug=False):
    func_rk = func_gen(params)
    size = params.size
    x0 = torch.FloatTensor([[0, 0, 0.1, 0] for _ in range(size)])
    ts = torch.FloatTensor(np.linspace(0, length_seconds, num=length_seconds*fs))
    ys = odeint(func_rk, x0, ts, method='rk4')
    signal = calc_ecg(ys, params)
    if debug:
        return signal, ys
    else:
        return signal


def generate_ecg_bvam(fs, length_seconds, ecg_type='normal'):
    if ecg_type == 'normal':
        c, h, g, b = 1.35, 3, 7, 4
        a = np.array([-0.024, 0.0216, -0.0012, 0.12])
    elif ecg_type == 'af':
        c, h, g, b = 1.35, 2.848, 13, 4
        a = np.array([-0.068, 0.028, -0.024, 0.12])
    else:
        raise NotImplementedError
    p0_center = ODE_Params(a, c, h, g, b)
    p0_list = ODE_Params_List(params_list=[p0_center])
    xs = forward(p0_list, length_seconds=length_seconds, fs=fs, debug=False)
    xs_np = xs.data.detach().cpu().numpy()
    return xs_np[0]

    