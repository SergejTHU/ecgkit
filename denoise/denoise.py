import numpy as np
import pywt


def wt_denoise(signal):
    wt_base = pywt.Wavelet('coif2')
    coeffs = pywt.wavedec(signal, wt_base, level=8)
    zero_level = [0, 1]
    for x in zero_level:
        coeffs[x] = np.array([0] * len(coeffs[x]))
    denoised_signal = pywt.waverec(coeffs, wt_base)
    return denoised_signal


def median_denoise(signal, fs=360):
    p_win = int(fs * 100.0 / 1000.0)
    t_win = 3 * p_win
    b0 = [np.median(signal[max(0, x - p_win): min(x + p_win, len(signal) - 1)]) for x in range(len(signal))]
    b1 = [np.median(b0[max(0, x - t_win): min(x + t_win, len(b0) - 1)]) for x in range(len(b0))]
    denoised_signal = signal - b1
    return denoised_signal


def normalize(signal):
    """
    try mean + 3 * sigma
    :param signal:
    :return:
    """
    if len(signal.shape) == 2:
        max_val = np.max(signal, axis=1, keepdims=True)
        min_val = np.min(signal, axis=1, keepdims=True)

        norm = signal / (max_val - min_val)
    else:
        max_val = np.max(signal)
        min_val = np.min(signal)

        norm = signal / (max_val - min_val)

    return norm