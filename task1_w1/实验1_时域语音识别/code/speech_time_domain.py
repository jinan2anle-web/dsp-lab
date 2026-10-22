# -*- coding: utf-8 -*-
"""
实验1：基于时域分析技术的语音识别
核心算法模块（可直接用于真实录音数据）

包含：
  1. WAV 文件读写与格式字段解析（手写 RIFF 解析，不使用 scipy.io.wavfile）
  2. 预处理：去直流 / 预加重 / 分帧 / 加窗（矩形窗、汉明窗、海宁窗）
  3. 短时域特征：短时能量、短时平均幅度、短时过零率
  4. 双门限法端点检测
  5. 时域特征序列 + DTW 分类器（用于"0"~"9"孤立字识别）
"""

import numpy as np
import struct
import os


# ----------------------------------------------------------------------
# 1. WAV 文件解析（对应报告 2.2 节：WAV 格式字段读取）
# ----------------------------------------------------------------------
def read_wav(path):
    """手工解析 RIFF/WAVE 文件，返回 (信号 float64 一维数组, 采样率, 字段字典)"""
    with open(path, "rb") as f:
        data = f.read()

    if data[0:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise ValueError('不是合法的 WAVE 文件')

    info = {'RIFF': data[0:4].decode(), 'ChunkSize': struct.unpack('<I', data[4:8])[0]}
    pos = 12
    fmt = {}
    raw = None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        csize = struct.unpack('<I', data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + csize]
        if cid == b'fmt ':
            (audio_format, channels, sample_rate, byte_rate,
             block_align, bits) = struct.unpack('<HHIIHH', body[:16])
            fmt = dict(audio_format=audio_format, channels=channels,
                       sample_rate=sample_rate, byte_rate=byte_rate,
                       block_align=block_align, bits_per_sample=bits)
        elif cid == b'data':
            raw = body
        pos += 8 + csize + (csize & 1)

    info.update(fmt)
    if info['bits_per_sample'] == 16:
        x = np.frombuffer(raw, dtype='<i2').astype(np.float64) / 32768.0
    elif info['bits_per_sample'] == 8:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128) / 128.0
    else:
        raise ValueError('仅支持 8/16 bit PCM')

    if info['channels'] == 2:                       # 立体声取左声道
        x = x[0::2]
    return x, info['sample_rate'], info


def write_wav(path, x, fs, bits=16):
    """把信号写为标准 16bit 单声道 WAV（便于生成测试音频）"""
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767).astype('<i2').tobytes()
    hdr = b'RIFF' + struct.pack('<I', 36 + len(pcm)) + b'WAVE'
    hdr += b'fmt ' + struct.pack('<IHHIIHH', 16, 1, 1, fs, fs * 2, 2, bits)
    hdr += b'data' + struct.pack('<I', len(pcm))
    with open(path, 'wb') as f:
        f.write(hdr + pcm)


# ----------------------------------------------------------------------
# 2. 预处理
# ----------------------------------------------------------------------
def remove_dc(x):
    return x - np.mean(x)


def pre_emphasis(x, a=0.97):
    """预加重 H(z)=1-a*z^-1，提升高频分量"""
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - a * x[:-1]
    return y


def enframe(x, frame_len, hop, win='hamming'):
    """分帧加窗。返回 (帧矩阵 [n_frames, frame_len], 窗函数)"""
    if win == 'rect':
        w = np.ones(frame_len)
    elif win == 'hamming':
        w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_len) / (frame_len - 1))
    elif win == 'hanning':
        w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame_len) / (frame_len - 1)))
    else:
        raise ValueError('窗型必须是 rect / hamming / hanning')

    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n = 1 + (len(x) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return x[idx] * w[None, :], w


# ----------------------------------------------------------------------
# 3. 短时域特征  (报告式 (4-1)~(4-3))
# ----------------------------------------------------------------------
def short_time_energy(frames):
    """E_n = sum_{m=0}^{N-1} x_n^2(m)"""
    return np.sum(frames ** 2, axis=1)


def short_time_amplitude(frames):
    """M_n = sum_{m=0}^{N-1} |x_n(m)|"""
    return np.sum(np.abs(frames), axis=1)


def short_time_zcr(frames):
    """Z_n = 1/2 * sum |sgn[x_n(m)] - sgn[x_n(m-1)]| ，再按帧长归一化"""
    s = np.sign(frames)
    z = np.sum(np.abs(np.diff(s, axis=1)), axis=1) / 2.0
    return z


def short_time_zcr_rate(frames):
    """归一化过零率（报告 2.5 节步骤 c）"""
    return short_time_zcr(frames) / frames.shape[1]


# ----------------------------------------------------------------------
# 4. 双门限法端点检测（报告 2.6 节 / 图 1.5）
# ----------------------------------------------------------------------
def endpoint_detect(x, fs, frame_ms=25.0, hop_ms=10.0, win='hamming',
                    is_ratio=0.4, noise_ratio=0.3, zcr_ratio=5.0,
                    min_len_ms=100.0):
    """
    双门限端点检测。
    is_ratio     : 高门限 T2 = 最大短时能量 * is_ratio（一次初判）
    noise_ratio  : 低门限 T1 = max(T2 * noise_ratio, 噪声能量 * 8)（二次搜索，兼顾信噪比与动态范围）
    zcr_ratio    : 过零率门限 T3 = 前若干帧平均过零率 * zcr_ratio
    返回 dict：起点/终点(秒)、逐帧特征与三条门限
    """
    frame_len = int(round(frame_ms * fs / 1000.0))
    hop = int(round(hop_ms * fs / 1000.0))

    x = remove_dc(x)
    frames, _ = enframe(x, frame_len, hop, win)
    E = short_time_energy(frames)
    M = short_time_amplitude(frames)
    Z = short_time_zcr_rate(frames)

    n_noise = max(5, int(0.1 * len(E)))           # 前 10% 帧估计背景噪声
    E_noise = float(np.mean(E[:n_noise])) + 1e-12
    Z_noise = float(np.mean(Z[:n_noise])) + 1e-12

    # 高门限 T2：语音能量包络中"绝大部分帧都在其上"的电平（一次初判）
    T2 = max(E_noise * 10.0, np.max(E) * is_ratio)
    # 低门限 T1：比背景噪声高一个安全余量，同时与 T2 保持固定比例（相对量），
    #           使门限随说话人音量自适应，又不至于把噪声段当作语音
    T1 = max(T2 * noise_ratio, E_noise * 8.0)
    # 过零率门限 T3：背景噪声过零率的一个倍数
    T3 = Z_noise * zcr_ratio

    # ---- 步骤 1/2：高门限初判，得到 N3、N4 ----
    above = np.where(E > T2)[0]
    if len(above) == 0:
        return None
    n3, n4 = int(above[0]), int(above[-1])

    # ---- 步骤 3：由 N3 向左、N4 向右搜索第一次与 T1 的交点 N2、N5 ----
    n2 = n3
    while n2 > 0 and E[n2 - 1] > T1:
        n2 -= 1
    n5 = n4
    while n5 < len(E) - 1 and E[n5 + 1] > T1:
        n5 += 1

    # ---- 步骤 4：以过零率 T3 精确定位起点 N1、终点 N6 ----
    #      条件：该帧能量仍高于噪声水平(2*T1/3) 或 过零率高于 T3
    n1 = n2
    while n1 > 0 and (Z[n1 - 1] > T3 or E[n1 - 1] > 2 * T1 / 3):
        n1 -= 1
    n6 = n5
    while n6 < len(Z) - 1 and (Z[n6 + 1] > T3 or E[n6 + 1] > 2 * T1 / 3):
        n6 += 1

    # 最短语音长度约束，抑制短促噪声
    if (n6 - n1) * hop_ms < min_len_ms:
        n6 = min(len(E) - 1, n1 + int(min_len_ms / hop_ms))

    def t(i):
        return i * hop / fs

    return dict(frames=frames, E=E, M=M, Z=Z, T1=T1, T2=T2, T3=T3,
                N1=n1, N2=n2, N3=n3, N4=n4, N5=n5, N6=n6,
                start=t(n1), end=t(n6 + 1),
                seg=x[n1 * hop: min(len(x), (n6 + 1) * hop + frame_len)],
                E_noise=E_noise, Z_noise=Z_noise)


def split_syllables(x, fs, min_gap_ms=60.0, **kw):
    """在整段语音上先做端点检测，再按低能量谷值切分为多个音节（数字串切分）"""
    frame_len = int(round(kw.get('frame_ms', 25.0) * fs / 1000.0))
    hop = int(round(kw.get('hop_ms', 10.0) * fs / 1000.0))
    det = endpoint_detect(x, fs, **kw)
    if det is None:
        return det, []
    seg = det['seg']
    f2, _ = enframe(remove_dc(seg), frame_len, hop, kw.get('win', 'hamming'))
    E2 = short_time_energy(f2)
    quiet = E2 < 0.03 * np.max(E2)
    # 找连续静音段的中点作为切分点
    cuts, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - i) * hop / fs * 1000 >= min_gap_ms:
                cuts.append(((i + j) // 2) * hop)
            i = j
        else:
            i += 1
    bounds = [0] + cuts + [len(seg)]
    return det, [seg[bounds[k]:bounds[k + 1]] for k in range(len(bounds) - 1)]


# ----------------------------------------------------------------------
# 5. 时域特征序列 + DTW 分类器
# ----------------------------------------------------------------------
def time_domain_feature(x, fs, frame_ms=25.0, hop_ms=10.0, win='hamming',
                        normalize=True):
    """
    逐帧提取"时域特征序列"（模板匹配的输入），返回 [T, D] 的帧级特征矩阵：
        d0 = ln(E_n)      短时能量（对数压缩，压缩强弱动态范围）
        d1 = ln(M_n)      短时平均幅度
        d2 = Z_n          归一化短时过零率
    normalize=True 时对每一维在本段内做零均值单位方差归一化（逐序列归一化），
    目的是消除说话人音量、录音增益和环境噪声电平的影响，只保留"特征随时间变化的
    形状"（即音节的起-落结构）—— 这正是时域模板匹配所依赖的信息。
    序列长度 T 随语音时长变化，因此必须用 DTW 做时间规整（非等长比较）。
    """
    frames, _ = enframe(remove_dc(x), int(frame_ms * fs / 1000),
                        int(hop_ms * fs / 1000), win)
    E = np.log(short_time_energy(frames) + 1e-10)
    M = np.log(short_time_amplitude(frames) + 1e-10)
    Z = short_time_zcr_rate(frames)
    seq = np.stack([E, M, Z], axis=1).astype(float)
    if normalize:
        seq = (seq - seq.mean(axis=0)) / (seq.std(axis=0) + 1e-9)
    return seq                                              # [T, D]


def fit_normalizer(feature_list):
    """在训练集上统计各维特征的均值/标准差（全局归一化，保留类别间的绝对差异）"""
    allf = np.concatenate([np.asarray(f, float) for f in feature_list], axis=0)
    return allf.mean(axis=0), allf.std(axis=0) + 1e-9


def apply_normalizer(seq, mu, sigma):
    return (np.asarray(seq, float) - mu) / sigma


def dtw_distance(a, b, band=None):
    """
    动态时间规整（欧氏距离，报告式(3-2)）。
        D(i,j) = d(i,j) + min{D(i-1,j), D(i,j-1), D(i-1,j-1)}
    其中 d(i,j) 为第 i 帧与第 j 帧特征矢量之欧氏距离。
    band: 可选的规整窗口宽度（Sakoe-Chiba 带），限制路径偏离对角线，防止过度规整。
    返回值按路径长度归一化，使不同时长的样本可比。
    """
    A = [[float(v) for v in row] for row in a]
    B = [[float(v) for v in row] for row in b]
    n, m = len(A), len(B)
    if band is None or band < 1:
        band = max(n, m)
    band = max(band, abs(n - m) + 1)
    INF = float('inf')
    prev = [INF] * (m + 1)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = [INF] * (m + 1)
        ai = A[i - 1]
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            bj = B[j - 1]
            s = 0.0
            for k in range(len(ai)):
                dv = ai[k] - bj[k]
                s += dv * dv
            d = s ** 0.5
            cur[j] = d + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[m] / (n + m)          # 按路径长度归一化


class DTWClassifier:
    """最近邻 + DTW 的孤立字分类器（模板库保存训练集特征序列）"""

    def __init__(self, band_ratio=0.35):
        self.templates = []           # list of (label, feature_seq)
        self.band_ratio = band_ratio

    def fit(self, X, y):
        self.templates = [(lab, np.asarray(f, float)) for lab, f in zip(y, X)]

    def predict_one(self, f):
        best, bl = np.inf, None
        for lab, t in self.templates:
            band = int(max(len(f), len(t)) * self.band_ratio)
            d = dtw_distance(f, t, band=band)
            if d < best:
                best, bl = d, lab
        return bl, best

    def predict(self, X):
        return np.array([self.predict_one(np.asarray(f, float))[0] for f in X])


def extract_file(path, fs_expected=None, **kw):
    """读文件 -> 端点检测 -> 时域特征序列（识别流程入口）"""
    x, fs, info = read_wav(path)
    det = endpoint_detect(x, fs, **kw)
    if det is None:
        return None, None, fs, info
    return time_domain_feature(det['seg'], fs, **kw), det, fs, info
