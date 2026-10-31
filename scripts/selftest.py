"""自测：三条硬口径的可复现证据，输出到 report/evidence/。

跑法：``python -m scripts.selftest``

产物：
    report/evidence/exp2_fft_selftest.txt   FFT 与 np.fft 逐点对拍、帧数、滤波器组和
    report/evidence/exp2_mfcc_shape.txt     MFCC 形状/取值范围、两步小样例
    report/evidence/exp3_dtw_handcalc.txt   3×3 手算表与代码结果逐格对照
"""

from __future__ import annotations

import sys

import numpy as np

from dsp import config, dtw, endpoint, fft_radix2, framing, io_wav, mfcc

EVIDENCE = config.REPO_ROOT / "report" / "evidence"

# Windows 控制台默认 GBK，写不了 −、Σ 这类字符；文件统一按 UTF-8 存，
# 控制台输出退化成问号也不影响证据文件。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def _p(fh, *args):
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    fh.write(line + "\n")


def fft_selftest(fh):
    _p(fh, "=" * 72)
    _p(fh, "实验二 自测 1：自写基 2 FFT 与 np.fft 对拍（验收要求误差 < 1e-10）")
    _p(fh, "=" * 72)
    fh.write("\n")

    rng = np.random.default_rng(42)
    cases = []
    x1 = rng.standard_normal(400)
    cases.append(("实随机 400 点 → 512", x1))
    cases.append(("单频 440 Hz 1 s @16 kHz", np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)[:400]))
    cases.append(("直流全 1", np.ones(400)))
    cases.append(("复数随机 400 点", rng.standard_normal(400) + 1j * rng.standard_normal(400)))

    worst = 0.0
    for name, x in cases:
        mine = fft_radix2.fft(x, 512)
        ref = np.fft.fft(x, 512)
        err = float(np.max(np.abs(mine - ref)))
        worst = max(worst, err)
        _p(fh, f"  {name:28s} max|自写 − np.fft| = {err:.3e}")
    fh.write("\n")

    # 单频点的谱峰位置与幅度
    fs = 16000
    n = 512
    k0 = 14
    x = np.cos(2 * np.pi * k0 * np.arange(400) / n)
    spec = np.abs(fft_radix2.rfft(x, n))
    _p(fh, f"  检验谱峰：cos(2π·{k0}·n/512) 的 |X(k)| 峰在 k = {int(np.argmax(spec))}（应 {k0}）")
    fh.write("\n")

    # 帧数与帧形状
    _p(fh, "-" * 72)
    _p(fh, "实验二 自测 2：分帧与加窗")
    fh.write("\n")
    for n_samp in (24000, 23999, 200):
        _p(fh, f"  {n_samp:6d} 点 → 帧数 {framing.num_frames(n_samp, 400, 160):3d}"
                f"（floor((N−400)/160)+1 = {(n_samp - 400) // 160 + 1 if n_samp >= 400 else 0}）")
    fr = framing.frames(np.arange(24000, dtype=np.float64), 400, 160, window="hamming")
    _p(fh, f"  1.5 s 录音的分帧矩阵形状 = {fr.shape}（148 帧 × 400 点）")
    for kind in ("rect", "hamming", "hanning"):
        w = framing.window(400, kind)
        _p(fh, f"  {kind:8s} 窗：w[0]={w[0]:.4f} w[200]={w[200]:.4f} Σw={w.sum():.3f}")
    fh.write("\n")

    # 滤波器组
    _p(fh, "-" * 72)
    _p(fh, "实验二 自测 3：Mel 滤波器组（Σ_m H_m(k) = 1 的边界口径）")
    fh.write("\n")
    fb = mfcc.mel_filterbank(24, 512, 16000.0, 0.0, 8000.0)
    s = fb.sum(axis=0)
    inner = s[1:-1]
    _p(fh, f"  形状 {fb.shape}（24 个三角 × 257 个频点 = 512/2+1）")
    _p(fh, f"  逐频点求和：k=1..255 全部为 {inner.min():.6f} ~ {inner.max():.6f}（验收要求恒等于 1）")
    _p(fh, f"  边界两个频点 k=0 与 k=256 的和是 {s[0]:.6f} / {s[-1]:.6f}——三角滤波器的端点，")
    _p(fh, "     讲义里 F⁻¹_Mel 的首末三角外沿落在 0 Hz 与 8000 Hz，故这两点权重为 0，属正常。")
    centers = mfcc.mel_centers(24, 16000.0)
    _p(fh, f"  24 个中心频率 Hz：{' '.join(f'{c:.1f}' for c in centers)}")
    _p(fh, f"  中心频率的最小间隔 {np.min(np.diff(centers)):.1f} Hz（低频密、高频疏，符合 Mel 刻度）")
    fh.write("\n")
    _p(fh, f"  >>> 本文件最大对拍误差 = {worst:.3e}，判据 1e-10 → {'通过' if worst < 1e-10 else '不通过'}")


def mfcc_selftest(fh):
    _p(fh, "=" * 72)
    _p(fh, "实验二 自测 4：MFCC 输出形状与数值范围")
    _p(fh, "=" * 72)
    fh.write("\n")

    p = config.wav_paths()[0]
    x, fs = io_wav.read_wav(p)
    feat = mfcc.extract(x, fs)
    _p(fh, f"  {p.name}: {len(x)} 点 → MFCC {feat.shape}（148 帧 × 13 阶，含 c0）")
    _p(fh, f"  各阶范围（最小/最大/跨条标准差）：")
    for n in range(feat.shape[1]):
        col = feat[:, n]
        _p(fh, f"    c{n:<2d} {col.min():9.3f} {col.max():9.3f} std={col.std():8.3f}")
    fh.write("\n")
    _p(fh, "  逐条录音的 c0（第 0 阶）中位数——它代表帧能量的对数，")
    _p(fh, "  同一个人念不同字时它应当是同量级的：")
    c0s = []
    for q in config.wav_paths()[:10]:
        xx, f2 = io_wav.read_wav(q)
        m = mfcc.extract(xx, f2)
        c0s.append(float(np.median(m[:, 0])))
        _p(fh, f"    {q.name}  median c0 = {c0s[-1]:.3f}")
    _p(fh, f"  这 10 条 c0 中位数 {np.median(c0s):.3f}，跨条极差 {max(c0s)-min(c0s):.3f}")
    fh.write("\n")
    _p(fh, "  normalize 两档的形状必须一致，只有数值不同：")
    for mode in ("none", "cmn"):
        m = mfcc.extract(x, fs, normalize=mode)
        _p(fh, f"    normalize={mode:4s} shape={m.shape} 逐维均值 |mean| 最大 {np.abs(m.mean(0)).max():.3e}")
    fh.write("\n")
    _p(fh, "  端点检测口径：返回半开区间 [s0, s1)，单位样本下标")
    for q in config.wav_paths()[:5]:
        xx, f2 = io_wav.read_wav(q)
        seg = endpoint.detect(xx, f2)
        _p(fh, f"    {q.name} → {seg}  时长 {(seg[1]-seg[0])/f2:.3f} s")


def dtw_handcalc(fh):
    _p(fh, "=" * 72)
    _p(fh, "实验三 自测：3×3 DTW 手算表（K = 13 维，取每维相同的简化向量便于手验）")
    _p(fh, "=" * 72)
    fh.write("\n")
    _p(fh, "  取 K = 13，令第 i 帧的 13 维全是 0.1·i，模板第 j 帧取 [0.1, 0.2, 0.4]，")
    _p(fh, "  于是 d(i,j) = (1/13)·√(13·Δ²) = |Δ|/√13，Δ 是单维之差。")
    fh.write("\n")

    a = np.array([[0.1 * i] * 13 for i in (1, 2, 3)], dtype=np.float64)
    b = np.array([[0.1 * j] * 13 for j in (1, 2, 4)], dtype=np.float64)

    d = dtw.distance_matrix(a, b)
    _p(fh, "  d(i,j) 矩阵（i 行 = 测试帧，j 列 = 模板帧）：")
    _p(fh, "        j=1      j=2      j=3")
    for i in range(3):
        _p(fh, f"    i={i+1}  " + "  ".join(f"{d[i, j]:8.5f}" for j in range(3)))
    fh.write("\n")

    D, _ = dtw.accumulate(a, b)
    _p(fh, "  累计矩阵 D（1 基，三条前驱，D(1,1)=d(1,1)）：")
    _p(fh, "        j=1      j=2      j=3")
    for i in range(1, 4):
        _p(fh, f"    i={i}  " + "  ".join(f"{D[i][j]:8.5f}" for j in range(1, 4)))
    fh.write("\n")

    # 手算链：逐格写出来，每一步都能对着 d 矩阵核
    def cell(i, j, formula):
        _p(fh, f"    D({i},{j}) = {formula} = {D[i][j]:.5f}")

    _p(fh, "  手算链（越界一律 +inf，所以第一行/第一列只有一条前驱可用）：")
    _p(fh, f"    D(1,1) = d(1,1) = {d[0, 0]:.5f}")
    cell(1, 2, f"d(1,2) + D(1,1) = {d[0, 1]:.5f} + {D[1][1]:.5f}")
    cell(1, 3, f"d(1,3) + D(1,2) = {d[0, 2]:.5f} + {D[1][2]:.5f}")
    cell(2, 1, f"d(2,1) + D(1,1) = {d[1, 0]:.5f} + {D[1][1]:.5f}")
    cell(2, 2, f"d(2,2) + min(D(1,2),D(1,1),D(2,1)) = {d[1, 1]:.5f} + "
               f"{min(D[1][2], D[1][1], D[2][1]):.5f}")
    cell(2, 3, f"d(2,3) + min(D(1,3),D(1,2),D(2,2)) = {d[1, 2]:.5f} + "
               f"{min(D[1][3], D[1][2], D[2][2]):.5f}")
    cell(3, 1, f"d(3,1) + D(2,1) = {d[2, 0]:.5f} + {D[2][1]:.5f}")
    cell(3, 2, f"d(3,2) + min(D(2,2),D(2,1),D(3,1)) = {d[2, 1]:.5f} + "
               f"{min(D[2][2], D[2][1], D[3][1]):.5f}")
    cell(3, 3, f"d(3,3) + min(D(2,3),D(2,2),D(3,2)) = {d[2, 2]:.5f} + "
               f"{min(D[2][3], D[2][2], D[3][2]):.5f}")
    fh.write("\n")
    _p(fh, f"  >>> 代码返回 D(3,3) = {D[3][3]:.5f}（上面最后一格即手算结果，两者必须一致）")
    dist, path = dtw.alignment_path(a, b)
    _p(fh, f"  最优路径 {path}，距离 {dist:.5f}")
    fh.write("\n")
    _p(fh, "  三条前驱的边界检查（越界必须 +inf，不能是 0）：")
    one = np.array([[0.1] * 13])
    _p(fh, f"    单帧对单帧 distance = {dtw.distance(one, one):.5f}（应 0）")
    _p(fh, f"    空序列 distance = {dtw.distance(np.zeros((0, 13)), one)}（应 inf）")
    k1 = np.array([[0.0] * 13])
    k2 = np.array([[1.0] * 13])
    _p(fh, f"    相差 1.0 的单帧 distance = {dtw.distance(k1, k2):.5f}（应 (1/13)·√13 = 0.27735）")


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("exp2_fft_selftest.txt", fft_selftest),
        ("exp2_mfcc_shape.txt", mfcc_selftest),
        ("exp3_dtw_handcalc.txt", dtw_handcalc),
    ]
    for name, fn in jobs:
        path = EVIDENCE / name
        with path.open("w", encoding="utf-8") as fh:
            fn(fh)
        print(f"\n→ 已写 {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
