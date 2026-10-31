"""DTW：欧氏帧距 + 三条前驱递推。

讲义 p13 的式子原样实现：

* 帧距      d(x, y) = (1/K)·√Σ_i (x_i − y_i)² ，**K 是特征维数 13，不是帧数**
* 递推      D(i,j) = d(i,j) + min{ D(i−1,j), D(i−1,j−1), D(i,j−1) }
* 起点      D(1,1) = d(1,1)，从 (2,2) 开始递推
* 越界      一律 +inf（不许用 0，否则多条零成本斜边会把距离压平）
* 输出      D(N,M)，N、M 分别是两句的帧数

前驱只有三条，**没有** (i−2,j−1) 这类扩展步，这是 p13 的原式。
"""

from __future__ import annotations

import numpy as np

INF = float("inf")


def frame_distance(a, b):
    """两帧的欧氏距离 d = (1/K)·√Σ(x_i−y_i)²。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    k = a.size
    return float(np.sqrt(np.sum((a - b) ** 2)) / k)


def distance_matrix(a, b):
    """整段帧距矩阵 ``[T1, T2]``，元素 d(i,j) 已含 1/K。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a、b 都必须是 [T, K]")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"维数不一致：{a.shape[1]} vs {b.shape[1]}")
    k = a.shape[1]
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)) / k


def accumulate(a, b):
    """返回 (D, d)：``D`` 是补了 +inf 边界的累计代价矩阵，形状 ``[T1+1, T2+1]``（1 基）。"""
    d = distance_matrix(a, b)
    t1, t2 = d.shape
    dl = d.tolist()
    D = [[INF] * (t2 + 1) for _ in range(t1 + 1)]
    D[1][1] = dl[0][0]
    for i in range(1, t1 + 1):
        row, prev, dr = D[i], D[i - 1], dl[i - 1]
        for j in range(1, t2 + 1):
            if i == 1 and j == 1:
                continue
            m = prev[j]
            if prev[j - 1] < m:
                m = prev[j - 1]
            if row[j - 1] < m:
                m = row[j - 1]
            row[j] = dr[j - 1] + m
    return D, d


def distance(a, b):
    """两段 MFCC 序列的 DTW 距离 D(N,M)。"""
    if len(a) == 0 or len(b) == 0:
        return INF
    D, _ = accumulate(a, b)
    return float(D[len(D) - 1][len(D[0]) - 1])


def alignment_path(a, b):
    """返回 (距离, 路径)。路径是 0 基的 ``[(i, j), ...]``，从 (0,0) 到 (T1−1,T2−1)。"""
    D, d = accumulate(a, b)
    t1, t2 = d.shape
    dist = float(D[t1][t2])
    i, j = t1, t2
    path = [(i - 1, j - 1)]
    while (i, j) != (1, 1):
        cand = [(D[i - 1][j], i - 1, j),
                (D[i - 1][j - 1], i - 1, j - 1),
                (D[i][j - 1], i, j - 1)]
        cand.sort(key=lambda t: t[0])
        _, i, j = cand[0]
        path.append((i - 1, j - 1))
    path.reverse()
    return dist, path
