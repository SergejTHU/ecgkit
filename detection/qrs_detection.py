from __future__ import division
import numpy as np
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