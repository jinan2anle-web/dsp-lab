"""MFCC：谱能量 → Mel 三角滤波 → 取对数 → DCT（全部按讲义 p9–p11 的式子写）。

流程（讲义原文顺序，六步）：

1. 预加重      H(z) = 1 − 0.97 z⁻¹
2. 分帧加窗    帧长 400 点（25 ms）、帧移 160 点（10 ms）、汉明窗
3. 512 点 FFT  400 点补零到 512（N 取 2^L，见 p8）
4. 谱线能量    E(i,k) = |X_i(k)|²
5. Mel 滤波    三角滤波器 H_m(k)，S(i,m) = Σ_k E(i,k)·H_m(k)
               Mel(f) = 2595·lg(1 + f/700)
               f(m) = (N/f_s)·F⁻¹_Mel( F_Mel(f_l) + m·(F_Mel(f_h)−F_Mel(f_l))/(M+1) )
               F⁻¹_Mel(b) = 700·(e^{b/1125} − 1)      ← 用 1125，不是 1127
               f(m) 算出来是**频点序号**，不是 Hz
6. 取对数 + DCT
               mfcc(i,n) = √(2/M)·Σ_{m=1..M} ln S(i,m)·cos[πn(2m−1)/(2M)]，n = 0..12

注意两处讲义口径，别被常识带跑：
* f(m) 用 1125 的反变换而正向用 2595·lg，两者严格来说差 0.18%（2595/ln10 = 1127.01），
  这是讲义原样，不改。
* 逆变换给出的 f(m) 是频点序号（乘了 N/f_s），直接当 k 用。
"""

from __future__ import annotations

import numpy as np

from . import fft_radix2, framing

EPS = 1e-10  # ln 保护，避免 log(0)


# ---------- Mel 刻度 ----------
def hz_to_mel(f):
    """Mel(f) = 2595·lg(1 + f/700)。"""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    """F⁻¹_Mel(b) = 700·(e^{b/1125} − 1)。"""
    return 700.0 * (np.exp(np.asarray(m, dtype=np.float64) / 1125.0) - 1.0)


def mel_centers(n_mels=24, fs=16000.0, f_low=0.0, f_high=None):
    """M 个三角滤波器的中心频率（Hz），按讲义式子逐点算，不用 linspace 近似。"""
    f_high = fs / 2.0 if f_high is None else float(f_high)
    lo, hi = hz_to_mel(f_low), hz_to_mel(f_high)
    step = (hi - lo) / (n_mels + 1)
    m = np.arange(1, n_mels + 1)
    return mel_to_hz(lo + m * step)


def mel_filterbank(n_mels=24, nfft=512, fs=16000.0, f_low=0.0, f_high=None):
    """三角滤波器组 H，返回 ``[n_mels, nfft//2+1]``。

    第 m 个三角以 f(m) 为峰、f(m−1)/f(m+1) 为零点，用的是**频点序号**
    ``f(m)·nfft/fs``。相邻三角首尾相接，故 Σ_m H_m(k) = 1（k 在滤波器组
    覆盖范围内），这正是验收清单里那条自测。
    """
    f_high = fs / 2.0 if f_high is None else float(f_high)
    lo, hi = hz_to_mel(f_low), hz_to_mel(f_high)
    step = (hi - lo) / (n_mels + 1)
    # 含两端点共 n_mels+2 个频率，其中 1..M 是峰，0 与 M+1 是外侧零点
    edges_hz = mel_to_hz(lo + np.arange(n_mels + 2) * step)
    edges_bin = edges_hz * nfft / float(fs)
    # 讲义正变换用 2595·lg(1+f/700)（分母等价 1127），逆变换用 700(e^{b/1125}−1)，
    # 两者不互逆：f_high=8000 会还原成 8038.95 Hz（频点 257.25），末端三角在
    # k=256 上还剩 0.046 的权重。把首末外沿钉在覆盖范围的边界频点上，使
    # 「覆盖范围内 Σ_m H_m(k) = 1、两端为 0」这条口径精确成立。
    edges_bin = np.clip(edges_bin, 0.0, nfft / 2.0)

    n_bins = nfft // 2 + 1
    k = np.arange(n_bins, dtype=np.float64)
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        left, center, right = edges_bin[m], edges_bin[m + 1], edges_bin[m + 2]
        rise = (k - left) / (center - left)
        fall = (right - k) / (right - center)
        fb[m] = np.clip(np.minimum(rise, fall), 0.0, None)
    return fb


def dct_matrix(n_mfcc=13, n_mels=24):
    """mfcc 的 DCT 变换矩阵，``[n_mfcc, n_mels]``。

    mfcc(n) = √(2/M)·Σ_m lnS(m)·cos[πn(2m−1)/(2M)]，n = 0..n_mfcc−1。
    讲义对 n=0 也乘同一个 √(2/M)，不额外乘 1/√2（与 scipy 的 ortho DCT 不同）。
    """
    n = np.arange(n_mfcc, dtype=np.float64)[:, None]
    m = np.arange(1, n_mels + 1, dtype=np.float64)[None, :]
    return np.sqrt(2.0 / n_mels) * np.cos(np.pi * n * (2.0 * m - 1.0) / (2.0 * n_mels))


# ---------- 主流程 ----------
def spectrum_energy(fr, nfft=512):
    """第 4 步：谱线能量 E(i,k) = |X_i(k)|²，返回 ``[T, nfft//2+1]``。"""
    spec = fft_radix2.rfft(fr, nfft)
    return (spec.real ** 2 + spec.imag ** 2)


def mel_spectrum(fr, n_mels=24, nfft=512, fs=16000.0, f_low=0.0, f_high=None):
    """第 5 步：Mel 滤波后的能量 S(i,m)，返回 ``[T, n_mels]``。"""
    energy = spectrum_energy(fr, nfft)
    fb = mel_filterbank(n_mels, nfft, fs, f_low, f_high)
    return energy @ fb.T


def from_frames(fr, *, n_mfcc=13, n_mels=24, nfft=512, fs=16000.0,
                f_low=0.0, f_high=None, normalize="none"):
    """已经分好帧的信号（``[T, win_len]``）→ MFCC ``[T, n_mfcc]``。"""
    s = mel_spectrum(fr, n_mels, nfft, fs, f_low, f_high)
    log_s = np.log(s + EPS)
    mfcc = log_s @ dct_matrix(n_mfcc, n_mels).T
    return _normalize(mfcc, normalize)


def extract(x, fs=16000.0, *, n_mfcc=13, n_mels=24, nfft=512,
            win_len=400, hop_len=160, window="hamming", preemph=0.97,
            f_low=0.0, f_high=None, normalize="none"):
    """原始采样 ``x`` → MFCC ``[T, n_mfcc]``（六步一条龙）。"""
    fr = framing.frames(x, win_len, hop_len, window, preemph)
    return from_frames(fr, n_mfcc=n_mfcc, n_mels=n_mels, nfft=nfft, fs=fs,
                       f_low=f_low, f_high=f_high, normalize=normalize)


def _normalize(mfcc, mode):
    """normalize="none"（主结果）或 "cmn"（段内逐维去均值，消融用）。"""
    mode = (mode or "none").lower()
    if mode in ("none", "", "raw"):
        return mfcc
    if mode == "cmn":
        return mfcc - mfcc.mean(axis=0, keepdims=True)
    if mode == "cmvn":
        sd = mfcc.std(axis=0, keepdims=True)
        return (mfcc - mfcc.mean(axis=0, keepdims=True)) / np.where(sd < 1e-12, 1.0, sd)
    raise ValueError(f"未知 normalize 口径：{mode}")
