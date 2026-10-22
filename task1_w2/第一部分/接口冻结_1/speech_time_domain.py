# -*- coding: utf-8 -*-
"""
实验1 公共算法模块（接口冻结版，2026-10-05）

本文件是"接口冻结"的**参考实现**：函数名、参数名、返回结构一律以
docs/接口冻结说明.md（即 接口冻结/docs/接口冻结说明.md）为准，三人（A 数据 / B 特征 / C 识别）共同调用同一份，
不允许各自复制修改。任何签名变更必须走冻结说明第 6 节的变更流程。

模块分层：
    数据层  read_wav / write_wav
    预处理  remove_dc / pre_emphasis / enframe
    特征层  short_time_energy / short_time_amplitude / short_time_zcr / short_time_zcr_rate
    检测层  endpoint_detect
    识别层  time_domain_feature / dtw_distance / DTWClassifier / extract_file

自检：python tools/contract_test.py  ->  results/contract_test.txt
"""

import struct
import numpy as np


# ======================================================================
# 数据层
# ======================================================================
def read_wav(filepath):
    """
    手工解析 RIFF/WAVE 文件，**不依赖 scipy/标准库 wave**，逐字段读取。

    返回四元组：
        wave_data         numpy 一维 float64 数组，16bit PCM 原始整数量化值，
                          取值范围 [-32768, 32767]（不做归一化，保持"原始采样值"语义）
        sample_rate       int，本实验恒为 16000
        num_channels      int，本实验恒为 1；立体声输入按左声道输出为单声道
        bits_per_sample   int，本实验恒为 16
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) < 44 or data[0:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise ValueError('不是合法的 WAVE 文件: %s' % filepath)

    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        csize = struct.unpack('<I', data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + csize]
        if cid == b'fmt ':
            (audio_format, channels, sample_rate,
             byte_rate, block_align, bits) = struct.unpack('<HHIIHH', body[:16])
            fmt = dict(audio_format=audio_format, num_channels=channels,
                       sample_rate=sample_rate, byte_rate=byte_rate,
                       block_align=block_align, bits_per_sample=bits)
        elif cid == b'data':
            raw = body
        # RIFF 规定：子块长度为奇数时补 1 个填充字节（LIST/fact 等块常见）
        pos += 8 + csize + (csize & 1)

    if fmt is None or raw is None:
        raise ValueError('缺少 fmt 或 data 子块: %s' % filepath)

    bits = fmt['bits_per_sample']
    if bits == 16:
        wave_data = np.frombuffer(raw, dtype='<i2').astype(np.float64)
    elif bits == 8:
        wave_data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0
    else:
        raise ValueError('仅支持 8/16 bit PCM，当前 %d bit' % bits)

    if fmt['num_channels'] == 2:                      # 立体声 -> 左声道
        wave_data = wave_data[0::2]

    return wave_data, fmt['sample_rate'], fmt['num_channels'], bits


def write_wav(path, x, fs=16000, bits=16):
    """把 float 信号（范围 [-1,1]）写成标准 16bit 单声道 WAV"""
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    pcm = (x * 32767).astype('<i2').tobytes()
    hdr = b'RIFF' + struct.pack('<I', 36 + len(pcm)) + b'WAVE'
    hdr += b'fmt ' + struct.pack('<IHHIIHH', 16, 1, 1, fs, fs * 2, 2, bits)
    hdr += b'data' + struct.pack('<I', len(pcm))
    with open(path, 'wb') as f:
        f.write(hdr + pcm)


# ======================================================================
# 预处理层
# ======================================================================
def remove_dc(x):
    """去直流：x - mean(x)"""
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean()


def pre_emphasis(x, a=0.97):
    """预加重 H(z) = 1 - a·z⁻¹（实验2 频域法使用，本实验不强制）"""
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - a * x[:-1]
    return y


def enframe(x, frame_len, hop, win='hamming'):
    """
    分帧加窗。返回二元组 (frames, w)：
        frames  [n_frames, frame_len] float64，已乘窗
        w       [frame_len]，窗函数序列
    帧数 n_frames = 1 + floor((len(x) - frame_len) / hop)；len(x) < frame_len 时补零。
    win ∈ {'rect', 'hamming', 'hanning'}。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    frame_len, hop = int(frame_len), int(hop)
    if frame_len < 1 or hop < 1:
        raise ValueError('frame_len 与 hop 必须为正整数')

    if win == 'rect':
        w = np.ones(frame_len)
    elif win == 'hamming':
        w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_len) / (frame_len - 1))
    elif win == 'hanning':
        w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame_len) / (frame_len - 1)))
    else:
        raise ValueError('win 必须是 rect / hamming / hanning，当前 %r' % (win,))

    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n = 1 + (len(x) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return x[idx] * w[None, :], w


# ======================================================================
# 特征层
# ======================================================================
def short_time_energy(frames):
    """短时能量 E_n = Σ_{m=0}^{N-1} x_n²(m)   ->  [n_frames]"""
    return np.sum(np.asarray(frames, dtype=np.float64) ** 2, axis=1)


def short_time_amplitude(frames):
    """短时平均幅度 M_n = Σ_{m=0}^{N-1} |x_n(m)|   ->  [n_frames]"""
    return np.sum(np.abs(np.asarray(frames, dtype=np.float64)), axis=1)


def short_time_zcr(frames):
    """短时过零次数 Z_n = ½·Σ|sgn[x_n(m)] − sgn[x_n(m−1)]|   ->  [n_frames]"""
    s = np.sign(np.asarray(frames, dtype=np.float64))
    return np.sum(np.abs(np.diff(s, axis=1)), axis=1) / 2.0


def short_time_zcr_rate(frames):
    """归一化短时过零率 = Z_n / N，取值 [0, 1]   ->  [n_frames]"""
    frames = np.asarray(frames, dtype=np.float64)
    return short_time_zcr(frames) / frames.shape[1]


# ======================================================================
# 检测层
# ======================================================================
def endpoint_detect(x, fs, frame_ms=25.0, hop_ms=10.0, win='hamming',
                    is_ratio=0.4, noise_ratio=0.3, zcr_ratio=5.0,
                    min_len_ms=100.0):
    """
    双门限法端点检测（T2 初判 N3/N4 → T1 搜索 N2/N5 → T3 定位 N1/N6）。

    返回 dict（纯静音/无语音返回 None，不抛异常）：
        frames, E, M, Z          逐帧特征
        T1, T2, T3               三条门限
        N1..N6                   六个端点帧号
        start, end               语音段起止时间（秒）
        seg                      语音段波形（一维）
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    frame_len = int(round(frame_ms * fs / 1000.0))
    hop = int(round(hop_ms * fs / 1000.0))

    x = remove_dc(x)
    frames, _ = enframe(x, frame_len, hop, win)
    E = short_time_energy(frames)
    M = short_time_amplitude(frames)
    Z = short_time_zcr_rate(frames)

    n_noise = max(5, int(0.1 * len(E)))                # 前 10% 帧估计背景噪声
    E_noise = float(np.mean(E[:n_noise])) + 1e-12
    Z_noise = float(np.mean(Z[:n_noise])) + 1e-12

    T2 = max(E_noise * 10.0, float(np.max(E)) * is_ratio)
    T1 = max(T2 * noise_ratio, E_noise * 8.0)
    T3 = max(Z_noise * zcr_ratio, 1e-6)

    above = np.where(E > T2)[0]
    if len(above) == 0:
        return None                                    # 无语音，交由调用方跳过
    N3, N4 = int(above[0]), int(above[-1])

    N2 = N3
    while N2 > 0 and E[N2 - 1] > T1:
        N2 -= 1
    N5 = N4
    while N5 < len(E) - 1 and E[N5 + 1] > T1:
        N5 += 1

    N1 = N2
    while N1 > 0 and (Z[N1 - 1] > T3 or E[N1 - 1] > 2 * T1 / 3):
        N1 -= 1
    N6 = N5
    while N6 < len(Z) - 1 and (Z[N6 + 1] > T3 or E[N6 + 1] > 2 * T1 / 3):
        N6 += 1

    if (N6 - N1) * hop_ms < min_len_ms:                # 最短语音长度约束
        N6 = min(len(E) - 1, N1 + int(min_len_ms / hop_ms))

    def t(i):
        return i * hop / float(fs)

    return dict(frames=frames, E=E, M=M, Z=Z, T1=T1, T2=T2, T3=T3,
                N1=N1, N2=N2, N3=N3, N4=N4, N5=N5, N6=N6,
                start=t(N1), end=t(N6 + 1),
                seg=x[N1 * hop: min(len(x), (N6 + 1) * hop + frame_len)],
                E_noise=E_noise, Z_noise=Z_noise)


# ======================================================================
# 识别层
# ======================================================================
def time_domain_feature(x, fs, frame_ms=25.0, hop_ms=10.0, win='hamming',
                        normalize=True):
    """
    逐帧提取时域特征序列，返回 [T, 3]：
        d0 = ln E_n      短时能量（对数压缩动态范围）
        d1 = ln M_n      短时平均幅度
        d2 = Z_n         归一化短时过零率
    normalize=True 时**在本段序列内**逐维零均值单位方差归一化（逐序列归一化），
    消除音量/增益/噪声电平差异，只保留特征随时间的形状。T 随语音时长变化，
    因此必须用 DTW 做非等长时间规整。
    """
    frames, _ = enframe(remove_dc(x), int(frame_ms * fs / 1000),
                        int(hop_ms * fs / 1000), win)
    E = np.log(short_time_energy(frames) + 1e-10)
    M = np.log(short_time_amplitude(frames) + 1e-10)
    Z = short_time_zcr_rate(frames)
    seq = np.stack([E, M, Z], axis=1).astype(np.float64)
    if normalize:
        seq = (seq - seq.mean(axis=0)) / (seq.std(axis=0) + 1e-9)
    return seq


def dtw_distance(a, b, band=None):
    """
    动态时间规整距离（欧氏逐帧距离 + 累积最小值），返回**按路径长度归一化**的标量：
        D(i,j) = d(i,j) + min{D(i-1,j), D(i,j-1), D(i-1,j-1)}
        return  D(n,m) / (n + m)
    band: Sakoe-Chiba 规整窗口半宽；None 时不限制。
    满足 d(a,a)=0、d(a,b)=d(b,a)≥0。
    """
    A = np.asarray(a, dtype=np.float64)
    B = np.asarray(b, dtype=np.float64)
    n, m = len(A), len(B)
    if band is None or band < 1:
        band = max(n, m)
    band = max(int(band), abs(n - m) + 1)

    Al = [[float(v) for v in row] for row in A]        # 纯 Python 列表比 numpy
    Bl = [[float(v) for v in row] for row in B]        # 逐元素访问快一个数量级
    INF = float('inf')
    prev = [INF] * (m + 1)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = [INF] * (m + 1)
        ai = Al[i - 1]
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            bj = Bl[j - 1]
            s = 0.0
            for k in range(len(ai)):
                dv = ai[k] - bj[k]
                s += dv * dv
            cur[j] = s ** 0.5 + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[m] / (n + m)


class DTWClassifier:
    """最近邻 + DTW 模板匹配分类器（模板库 = 训练集特征序列）"""

    def __init__(self, band_ratio=0.35):
        self.band_ratio = band_ratio
        self.templates = []                            # list[(label, feature_seq)]

    def fit(self, X, y):
        self.templates = [(lab, np.asarray(f, dtype=np.float64))
                          for lab, f in zip(y, X)]
        return self

    def predict_one(self, f):
        f = np.asarray(f, dtype=np.float64)
        best, best_label = np.inf, None
        for lab, t in self.templates:
            band = int(max(len(f), len(t)) * self.band_ratio)
            d = dtw_distance(f, t, band=band)
            if d < best:
                best, best_label = d, lab
        return best_label, best

    def predict(self, X):
        return np.array([self.predict_one(np.asarray(f, dtype=np.float64))[0]
                         for f in X])


def extract_file(path, **kw):
    """
    单文件识别流程入口：读文件 → 去直流端点检测 → 时域特征序列。
    返回 (feature, det, fs, info)；无语音时 feature 与 det 均为 None。
    """
    x, fs, nch, bits = read_wav(path)
    info = dict(sample_rate=fs, num_channels=nch, bits_per_sample=bits)
    det = endpoint_detect(x, fs, **kw)
    if det is None:
        return None, None, fs, info
    return time_domain_feature(det['seg'], fs, **kw), det, fs, info
