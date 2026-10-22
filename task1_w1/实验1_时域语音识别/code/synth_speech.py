# -*- coding: utf-8 -*-
"""
实验数据合成模块（缺少真实录音时用于驱动与验证算法）

合成模型参考 PDF 图 4.2 的语音产生模型：
    准周期脉冲序列发生器(浊音) / 随机噪声发生器(清音) -> 时变线性系统(声道) -> 语音采样值
每个数字用"基频轨迹 + 前三条共振峰 + 时长"刻画，不同说话人通过改变基频与共振峰实现。
真实实验时把本模块替换为读 WAV 的门控函数即可，其余算法代码不用改动。
"""
import numpy as np

FS = 16000
DIGIT_MODEL = {
    '0': dict(f0=[130, 125, 118], F=[700, 1100, 2400], dur=0.36, voiced=1.0),
    '1': dict(f0=[135, 150, 140], F=[320, 2100, 2900], dur=0.26, voiced=0.85),
    '2': dict(f0=[140, 128, 122], F=[500, 1600, 2500], dur=0.30, voiced=0.95),
    '3': dict(f0=[132, 138, 130], F=[420, 2300, 2800], dur=0.28, voiced=0.90),
    '4': dict(f0=[142, 130, 126], F=[380, 1400, 2600], dur=0.30, voiced=0.92),
    '5': dict(f0=[136, 128, 124], F=[600, 1800, 2700], dur=0.30, voiced=0.95),
    '6': dict(f0=[128, 124, 120], F=[350, 1200, 2300], dur=0.32, voiced=0.90),
    '7': dict(f0=[140, 136, 132], F=[450, 1900, 2600], dur=0.28, voiced=0.88),
    '8': dict(f0=[134, 130, 126], F=[520, 1500, 2450], dur=0.32, voiced=0.96),
    '9': dict(f0=[130, 136, 132], F=[400, 2000, 2750], dur=0.30, voiced=0.92),
}


def synth_digit(digit, speaker=0, dur_jit=0.0, fs=FS, seed=0):
    """合成一个孤立数字（不含环境噪声，由 synth_utterance 统一加入）"""
    rnd = np.random.default_rng(seed)
    m = DIGIT_MODEL[digit]
    sp_f = 1.0 + 0.05 * speaker                       # 说话人声道差异
    dur = m['dur'] * (1.0 + dur_jit)
    n = int(dur * fs)
    t = np.arange(n) / fs
    f0 = np.interp(t / t[-1], [0, 0.5, 1], m['f0']) * (1.0 + 0.12 * speaker)
    f0 = f0 * (1 + 0.02 * np.sin(2 * np.pi * 4 * t))  # 基频微扰
    phase = 2 * np.pi * np.cumsum(f0) / fs

    sig = np.zeros(n)
    for h in range(1, 26):                            # 谐波叠加 + 共振峰加权
        fh = h * np.mean(f0)
        if fh > fs / 2 - 300:
            break
        g = 0.0
        for Fi, F in enumerate(m['F']):
            g += np.exp(-((fh - F * (sp_f if Fi < 2 else 1.0)) / 250.0) ** 2)
        amp = (0.35 ** (h - 1)) * (0.25 + g) / h ** 0.4
        sig += amp * np.sin(h * phase + rnd.uniform(0, 2 * np.pi))

    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.7     # 音节包络（升-平-降）
    env *= (1 + 0.12 * np.sin(2 * np.pi * 3 * t + rnd.uniform(0, 6)))
    sig *= env

    if m['voiced'] < 1.0:                             # 清音：摩擦噪声成分
        nz = rnd.normal(0, 1, n)
        b = np.exp(-2 * np.pi * 1800 / fs)
        nz = np.convolve(nz, [1, -b], 'same')
        sig += 0.22 * nz * (1 - m['voiced'])
    return sig / (np.max(np.abs(sig)) + 1e-9) * 0.8


def synth_utterance(digits=('0', '1', '2'), speaker=1, snr_db=20.0, gain=0.7,
                    fs=FS, seed=1, lead=0.30, tail=0.25):
    """
    合成一段完整"数字串"录音：前置静音 + 各数字（间歇 120~200 ms）+ 后置静音，
    整段叠加连续背景噪声（模拟真实环境本底噪声），最后定标到目标幅度。
    返回 (信号, [(数字, 起始秒, 结束秒), ...])
    """
    rnd = np.random.default_rng(seed)
    core, marks, pos = [], [], 0
    for d in digits:
        s = synth_digit(d, speaker=speaker, dur_jit=rnd.uniform(-0.08, 0.12),
                        fs=fs, seed=int(rnd.integers(1 << 30)))
        gap = np.zeros(int(rnd.uniform(0.12, 0.20) * fs))
        marks.append((d, pos / fs, (pos + len(s)) / fs))
        core.append(s)
        core.append(gap)
        pos += len(s) + len(gap)
    core = np.concatenate(core) if core else np.zeros(0)
    x = np.concatenate([np.zeros(int(lead * fs)), core, np.zeros(int(tail * fs))])
    marks = [(d, t0 + lead, t1 + lead) for d, t0, t1 in marks]

    k = np.sqrt(np.mean(core ** 2) / 10 ** (snr_db / 10))    # 按语音段平均功率定标 SNR
    x = x + rnd.normal(0, 1, len(x)) * k
    return x / (np.max(np.abs(x)) + 1e-9) * gain, marks


def decimate(x, n=2200, fs=FS):
    """波形抽稀：每段取绝对值最大的样点，保证绘图不丢失峰值"""
    L = len(x)
    if L <= n:
        return np.arange(L) / fs, x
    step = L / n
    xs, ys = [], []
    for i in range(n):
        a = int(i * step)
        b = max(a + 1, int((i + 1) * step))
        seg = x[a:b]
        xs.append(((a + b) / 2) / fs)
        ys.append(seg[np.argmax(np.abs(seg))])
    return np.array(xs), np.array(ys)
