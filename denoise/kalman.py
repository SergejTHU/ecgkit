from __future__ import division
import numpy as np
import sympy
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy import eye, array, asarray
from sympy import symbols, Matrix, diff
from filterpy.kalman import ExtendedKalmanFilter
from scipy.signal import butter, filtfilt, convolve
import peakutils


def qrs_peak_detect(seg, qonset, soffset):
    base = peakutils.baseline(seg, 2)
    qrs_seg = seg[qonset:soffset + 1]
    pos_indexs = peakutils.peak.indexes(qrs_seg, thres=0.5, min_dist=20) + qonset
    neg_indexs = peakutils.peak.indexes(-qrs_seg, thres=0.3, min_dist=5) + qonset
    pos_indexs = [x for x in pos_indexs if seg[x] > base[x]]
    neg_indexs = [x for x in neg_indexs if seg[x] < base[x]]
    return pos_indexs, neg_indexs, base


def detect_qrs(x, fs):
    """
    OKB method
    :param x:
    :param fs:
    :return:
    """
    assert isinstance(x, np.ndarray)
    if len(x.shape) > 1:
        assert len(x.shape) == 2 and min(x.shape) == 1
        x = np.squeeze(x)
    f1 = 8.0
    f2 = 20.0
    wn = np.array([f1, f2]) * 2 / fs
    N = 3
    [b, a] = butter(N, wn, 'bandpass')
    ecg_h = filtfilt(b, a, x)
    ecg_s = np.multiply(ecg_h, ecg_h)

    tmp = round(0.097*fs)
    if tmp % 2 == 0:
        qrsl_conv = int(tmp+1)
    else:
        qrsl_conv = int(tmp)
    tmp = round(0.611*fs)
    if tmp % 2 == 0:
        beatl_conv = int(tmp+1)
    else:
        beatl_conv = int(tmp)

    maqrs_conv = convolve(ecg_s, np.ones(qrsl_conv) / qrsl_conv)
    mabeat_conv = convolve(ecg_s, np.ones(beatl_conv) / beatl_conv)
    qrsl = int(round((qrsl_conv - 1) / 2))
    beatl = int(round((beatl_conv - 1) / 2))
    maqrs = maqrs_conv[qrsl:(maqrs_conv.size-qrsl)]
    mabeat = mabeat_conv[beatl:(mabeat_conv.size-beatl)]

    len_uncontam = int(ecg_s.size * 0.9)
    ecg_s_sort = np.sort(ecg_s)
    z = np.mean(ecg_s_sort[:len_uncontam])
    A = 0.08 * z + mabeat
    thr1 = A
    thr2 = qrsl_conv
    block_of_interest = np.array(maqrs > thr1).astype(np.int)
    D_block_of_interest = block_of_interest[1:] - block_of_interest[:-1]

    tmp_index = np.array(range(D_block_of_interest.size))
    on_qrs = tmp_index[D_block_of_interest == 1]
    off_qrs = tmp_index[D_block_of_interest == -1]
    if len(on_qrs) <= 2 or len(off_qrs) <= 2:
        qrs_locations = 0
    else:
        if on_qrs[0] > off_qrs[0]:
            off_qrs = off_qrs[1:]
        if on_qrs[-1] > off_qrs[-1]:
            on_qrs = on_qrs[:-1]
        dur_qrs = off_qrs - on_qrs
        qrs_locations = []
        for j in range(len(dur_qrs)):
            if dur_qrs[j] > thr2:
                max_pos = np.argmax(x[(on_qrs[j]+1):(off_qrs[j]+2)])
                qrs_locations.append(max_pos + on_qrs[j] + 1)
    return qrs_locations


def HJacobian_at(x):
    return np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])


def hx(x):
    return x[20]


def f_ag(theta_var, a, theta, b1, b2, p):
    v1 = a * sympy.exp(-(theta_var - theta) ** 2 / (2 * b1 ** 2)) * (1 / (1 + sympy.exp(-p * (theta_var - theta))))
    v2 = a * sympy.exp(-(theta_var - theta) ** 2 / (2 * b2 ** 2)) * (1 - 1 / (1 + sympy.exp(-p * (theta_var - theta))))
    return v1 + v2


def eval_var(var, val, pv=5):
    # val: [a1, ..., theta1, ..., b11, ..., b12, ..., theta_var]
    # val is not x
    subs = dict()
    subs[a[0]] = val[0]
    subs[a[1]] = val[1]
    subs[a[2]] = val[2]
    subs[a[3]] = val[3]
    subs[a[4]] = val[4]
    subs[theta[0]] = val[5]
    subs[theta[1]] = val[6]
    subs[theta[2]] = val[7]
    subs[theta[3]] = val[8]
    subs[theta[4]] = val[9]
    subs[b1[0]] = val[10]
    subs[b1[1]] = val[11]
    subs[b1[2]] = val[12]
    subs[b1[3]] = val[13]
    subs[b1[4]] = val[14]
    subs[b2[0]] = val[15]
    subs[b2[1]] = val[16]
    subs[b2[2]] = val[17]
    subs[b2[3]] = val[18]
    subs[b2[4]] = val[19]
    subs[theta_var] = val[20]
    subs[p] = pv
    ans = var.evalf(subs=subs)
    return ans


a = []
theta = []
b1 = []
b2 = []
for i in range(5):
    a.append(symbols('a_{}'.format(i+1)))
    theta.append(symbols('theta_{}'.format(i+1)))
    b1.append(symbols('b_{}1'.format(i+1)))
    b2.append(symbols('b_{}2'.format(i+1)))
theta_var = symbols('theta')
p = symbols('p')
ag_all = 0
for i in range(5):
    agi = f_ag(theta_var, a[i], theta[i], b1[i], b2[i], p)
    ag_all += agi
ag_d_theta_var = Matrix([diff(ag_all, theta_var)])
x20 = []
x20.extend(a)
x20.extend(theta)
x20.extend(b1)
x20.extend(b2)
ag_d_theta_var_d_params = ag_d_theta_var.jacobian(x20)


def kalman_filter(y, fs):
    r_locs = detect_qrs(y, fs)
    r_vals = y[r_locs]
    r_val_mean = r_vals.mean()
    y_truncated = y[r_locs[0]:(r_locs[-1]+1)]
    theta_value = 0
    theta_init = np.array([-np.pi / 3, -np.pi / 12, 0, np.pi / 12, np.pi / 2])
    a_init = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
    b1_init = np.array([0.25, 0.1, 0.1, 0.1, 0.4])
    b2_init = np.array([0.25, 0.1, 0.1, 0.1, 0.4])
    a_init = a_init * r_val_mean / 30
    x_init = np.concatenate([a_init, theta_init, b1_init, b2_init, np.array([y_truncated[0]])])
    val_init = np.concatenate([a_init, theta_init, b1_init, b2_init, np.array([theta_value])])
    d_theta = 0.005 * 2 * np.pi
    rk = ExtendedKalmanFilter(dim_x=21, dim_z=1)
    rk.x = np.array([x_init]).T
    rk.F = eye(21)
    rk.R = np.diag([0.1])
    rk.Q = np.diag([0.1 for _ in range(21)])
    val_v = val_init
    xs = []
    for i in tqdm(range(len(y_truncated))):
        z = y_truncated[i]
        for j in range(20):
            rk.F[20, j] = eval_var(ag_d_theta_var_d_params[j], val_v)
        rk.update(array([[z]]), HJacobian_at, hx)
        xs.append(rk.x)
        rk.predict()
        val_v[20] += d_theta
        if val_v[20] > np.pi:
            val_v[20] -= 2 * np.pi
        val_v[:20] = rk.x[0, :20]
    return np.array(xs)[:, 20, 0]

