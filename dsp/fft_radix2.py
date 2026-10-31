"""自行实现的基 2 FFT（时间抽取 DIT，含位倒序与补零到 2^L）。

讲义的得分点 p10：「按短时傅里叶变换原理**编写其函数，自行实现 FFT 过程**」。
所以这里不调用 ``np.fft`` / ``scipy.fft``，只借 numpy 做数组运算。

接口：

    fft(x, n=None)          -> complex [..., n]      复数谱（n 是 2 的幂）
    rfft(x, n=None)         -> complex [..., n//2+1] 只留非负频率
    mag_spectrum(x, n=None) -> float   [..., n//2+1] 幅度 |X(k)|

x 可以是 ``[N]`` 也可以是 ``[T, N]``（一次算一整帧矩阵）。
"""

from __future__ import annotations

import numpy as np


def next_pow2(n):
    """不小于 n 的最小 2 的幂。"""
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def bit_reverse_indices(n):
    """位倒序置换下标：把 0..n-1 的二进制位左右翻转。"""
    bits = int(n).bit_length() - 1
    idx = np.arange(n, dtype=np.int64)
    rev = np.zeros(n, dtype=np.int64)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


def fft(x, n=None):
    """基 2 时间抽取 FFT。

    x : 实数或复数，形状 ``[N]`` 或 ``[T, N]``
    n : 变换长度；不是 2 的幂时向上取到 2 的幂，不足补零，超过输入长度才报错

    返回 ``[..., n]`` 的复数数组（未做 FFTShift，k=0..n-1）。
    """
    a = np.asarray(x)
    squeeze = a.ndim == 1
    if squeeze:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"只支持 1 维或 2 维输入，收到 {a.shape}")

    m = a.shape[1]
    n = m if n is None else int(n)
    if n < m:
        raise ValueError(f"n={n} 小于输入长度 {m}，不做截断")
    if n & (n - 1):
        n = next_pow2(n)

    buf = np.zeros((a.shape[0], n), dtype=np.complex128)
    buf[:, :m] = a                     # 补零
    buf = buf[:, bit_reverse_indices(n)]   # 位倒序

    size = 2
    while size <= n:
        half = size >> 1
        # 旋转因子 W_size^k = exp(-2πi k / size)，k = 0..half-1
        tw = np.exp(-2j * np.pi * np.arange(half) / size)
        blk = buf.reshape(a.shape[0], -1, size)
        even = blk[:, :, :half].copy()      # 必须先拷贝：下面会就地改写 blk
        odd = blk[:, :, half:].copy() * tw
        blk[:, :, :half] = even + odd       # 蝶形上支
        blk[:, :, half:] = even - odd       # 蝶形下支
        size <<= 1

    return buf[0] if squeeze else buf


def rfft(x, n=None):
    """只留非负频率的 FFT，返回 ``[..., n//2+1]``。"""
    full = fft(x, n)
    n = full.shape[-1]
    return full[..., : n // 2 + 1]


def mag_spectrum(x, n=None):
    """幅度谱 ``|X(k)|``，返回实数 ``[..., n//2+1]``。"""
    return np.abs(rfft(x, n))
