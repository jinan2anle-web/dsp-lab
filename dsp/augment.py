"""测试实验使用的确定性音频扰动工具。"""

from __future__ import annotations

import numpy as np

from .endpoint import detect


def add_white_noise(x, snr_db, *, fs=16000, seed=0):
    """按语音段功率加入白噪声，并把噪声加到整条录音。

    先由端点检测得到半开区间 ``[s0, s1)``，令语音段均方功率为
    ``P_s``，噪声方差取 ``sigma^2 = P_s / 10**(snr_db/10)``。
    检测失败时退化为整条信号的均方功率；正无穷 SNR 原样返回副本。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if np.isposinf(snr_db):
        return x.copy()
    snr_db = float(snr_db)
    got = detect(x, fs)
    speech = x if got is None else x[got[0]:got[1]]
    signal_power = float(np.mean(speech ** 2)) if len(speech) else 0.0
    if signal_power <= 0.0:
        return x.copy()
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(noise_power), len(x))
    return x + noise


def apply_gain(x, factor):
    """整条信号乘常数 ``factor``，包括静音段，长度保持不变。"""
    return np.asarray(x, dtype=np.float64) * float(factor)
