"""分帧 + 预加重 + 加窗。

接口（已冻结，见 docs/实验二三技术路线.md）：

    window(n, kind="hamming")                                -> [n]
    preemph(x, alpha=0.97)                                   -> [len(x)]
    frames(x, win_len=400, hop_len=160, window="hamming",
           preemph=0.97)                                     -> [T, win_len]

1.5 s / 16 kHz 的录音分帧后 T = 148（= (24000-400)/160 + 1），端点一律"左对齐"，
不往尾部补零（多出来的不足一帧的样本自然丢掉）。
"""

from __future__ import annotations

import numpy as np


def window(n, kind="hamming"):
    """生成长度 n 的窗。kind: rect | hamming | hanning（hann 也认）。"""
    n = int(n)
    if n <= 0:
        raise ValueError("窗长必须为正")
    kind = kind.lower()
    if kind in ("rect", "rectangle", "boxcar"):
        return np.ones(n, dtype=np.float64)
    if kind in ("hamming", "ham"):
        # 0.54 - 0.46 cos(2πn/(N-1))，n = 0..N-1
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))
    if kind in ("hanning", "hann"):
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))
    raise ValueError(f"未知窗类型：{kind}")


def preemph(x, alpha=0.97):
    """预加重 y[n] = x[n] - α·x[n-1]，y[0] = x[0]。α=0 时原样返回。"""
    x = np.asarray(x, dtype=np.float64)
    if not alpha:
        return x.copy()
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - alpha * x[:-1]
    return y


def num_frames(n_samples, win_len=400, hop_len=160):
    """左对齐不补零时的帧数。"""
    if n_samples < win_len:
        return 1
    return 1 + (n_samples - win_len) // hop_len


def frames(x, win_len=400, hop_len=160, window="hamming", preemph=0.97):
    """把一维信号切成 ``[T, win_len]`` 的帧矩阵（每帧已乘窗）。

    preemph 在**分帧之前**对整条信号做（标准做法：先预加重再分帧并不改变结果，
    因为 y[n] 只依赖相邻两点，跨帧边界也是连续的）。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if preemph:
        x = _preemph(x, preemph)
    if len(x) < win_len:
        x = np.pad(x, (0, win_len - len(x)))  # 短信号补零成一帧
    n = num_frames(len(x), win_len, hop_len)
    idx = np.arange(win_len)[None, :] + hop_len * np.arange(n)[:, None]
    return x[idx] * _win(win_len, window)


# 形参名 window / preemph 会遮住同名函数，内部走这两个别名
_preemph = preemph
_win = window
