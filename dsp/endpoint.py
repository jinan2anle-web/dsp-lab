"""端点检测：去直流 + 双门限（短时能量 + 过零率）。

口径只有一种，三种写法都会导致结果对不上，所以写死在这里：

    返回**半开区间** ``[s0, s1)``，单位是**样本下标**，不是帧号。

讲义 p3 的双门限：高门限 T2 用于找到语音段起止，低门限 T1 越过 T2 的帧向后/向前
延伸，过零率 T3 作为清音段（能量低但过零率高）的补充判据。N1–N6 的编号沿用讲义：

    N1: 从前往后第一个 E ≥ T2 的帧
    N2: 从 N1 往回，最后一个 E ≥ T1 的帧      → 起点
    N5: 从后往前第一个 E ≥ T2 的帧
    N6: 从 N5 往后，最后一个 E ≥ T1 的帧      → 终点
"""

from __future__ import annotations

import numpy as np

from .framing import frames


def _frame_energy(x, win_len, hop_len):
    fr = frames(x, win_len, hop_len, window="rect", preemph=0.0)
    return np.mean(fr ** 2, axis=1), fr


def _frame_zcr(fr):
    """逐帧过零率。

    录音的「静音」不是精确的 0，而是 ``±1 LSB`` 的量化残差（幅值 3.0518e-5）：
    ``0_01.wav`` 前 0.35 s 的 5600 个样本里 1273 个非零（22.73%），100 条录音
    这个比例完全相同。由此静音帧的过零率实测约 0.20，反而高于全句 ZCR 的
    60 百分位（``0_01`` 为 0.161）——只靠 ZCR 判「有没有语音」会把整条录音
    判成语音，所以 ZCR 判据必须与低能量门限绑定（见 ``detect`` 里的
    ``voiced`` 规则：能量够响，或能量够低且过零率偏高）。
    另核过：100 条录音里 ``numpy.signbit`` 命中「数值为 0」的样本数是 0
    （``-0.0`` 并不存在），真正的成因是上面这条 ±1 LSB 量化残差。
    """
    neg = fr < 0
    return np.mean(neg[:, 1:] != neg[:, :-1], axis=1)


LOW_HIGH_RATIO = 0.35  # 低门限 = 高门限 × 0.35（0.002 → 0.0007，即 −27 dB / −31.5 dB）


def detect(x, fs, *, frame_ms=25.0, hop_ms=10.0, min_len_ms=100.0,
           high_ratio=0.002, low_ratio=None, zcr_ratio=0.60, dc_remove=True):
    """返回语音段的样本下标半开区间 ``(s0, s1)``；找不到返回 ``None``。

    阈值全部相对"本句峰值"取，不看绝对电平，100 条录音同一说话人、同一增益，
    这样比固定阈值稳。``high_ratio``/``low_ratio`` 对应 T2/T1 占峰值的比例，
    默认 −27 dB / −31.5 dB：在 100 条全集上扫了 6 档（见 report/tables/
    exp3_endpoint_sweep.csv），这一档的段长中位 0.67 s，最接近人工核对的中位
    语音时长 0.60 s；取 −10 dB 那档会把每个词都砍到只剩响的核（中位 0.27 s）。
    ``low_ratio`` 默认 ``high_ratio * LOW_HIGH_RATIO``（= 0.0007），门限扫描脚本
    也走这条换算，保证表里那一档与主结果是同一条件。
    ``zcr_ratio`` 是清音补充判据（过零率超过全句该分位的低能量帧也算语音）。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if low_ratio is None:
        low_ratio = high_ratio * LOW_HIGH_RATIO
    if dc_remove and len(x):
        x = x - x.mean()
    win_len = int(round(fs * frame_ms / 1000.0))
    hop_len = int(round(fs * hop_ms / 1000.0))
    win_len = max(win_len, 2)
    hop_len = max(hop_len, 1)
    if len(x) < win_len:
        return None

    energy, fr = _frame_energy(x, win_len, hop_len)
    zcr = _frame_zcr(fr)
    peak = float(energy.max())
    if peak <= 1e-12:
        return None

    t_high = high_ratio * peak
    t_low = low_ratio * peak
    zcr_hi = float(np.percentile(zcr, 100.0 * zcr_ratio))

    voiced = (energy >= t_high) | ((energy >= t_low) & (zcr >= zcr_hi))
    if not voiced.any():
        return None

    idx = np.flatnonzero(voiced)
    n1, n5 = int(idx[0]), int(idx[-1])
    # 用低门限向两侧延伸
    n2 = n1
    while n2 > 0 and energy[n2 - 1] >= t_low:
        n2 -= 1
    n6 = n5
    while n6 + 1 < len(energy) and energy[n6 + 1] >= t_low:
        n6 += 1

    s0 = n2 * hop_len
    s1 = min(len(x), n6 * hop_len + win_len)
    if s1 <= s0:
        return None

    # 太短就向两侧补长到 min_len_ms，但不越界、也不跨出原信号
    min_len = int(round(fs * min_len_ms / 1000.0))
    if s1 - s0 < min_len:
        mid = (s0 + s1) // 2
        s0 = max(0, mid - min_len // 2)
        s1 = min(len(x), s0 + min_len)
        s0 = max(0, s1 - min_len)
    return int(s0), int(s1)


def segment(x, fs, **kwargs):
    """返回切好的语音段；检测失败时退回整段（保证下游永远拿到数据）。"""
    got = detect(x, fs, **kwargs)
    if got is None:
        return np.asarray(x, dtype=np.float64), (0, len(x))
    s0, s1 = got
    return np.asarray(x, dtype=np.float64)[s0:s1], got
