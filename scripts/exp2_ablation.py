"""实验二：分帧窗口 / 预加重的消融，以及 5 张自绘图。

跑法：``python -m scripts.exp2_ablation``

消融的判据用**下游识别率**：同一套 MFCC + DTW 流程，只换实验二里被消融的那一项，
识别率变了多少就是这一项值多少。单看 MFCC 数值没有可比性。

产物：
    report/tables/exp2_ablation.csv
    report/fig/fig2_1_melbank.png / fig2_2_triangles.png / fig2_3_mfcc_flow.png
"""

from __future__ import annotations

import csv
import sys

import numpy as np

from dsp import config, fft_radix2, framing, io_wav, mfcc
from scripts import exp3_recognize as rec

TABLES_DIR = config.REPO_ROOT / "report" / "tables"
FIG_DIR = config.REPO_ROOT / "report" / "fig"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


# ---------- 消融 ----------
def accuracy_of(**extract_kw):
    """换一个 MFCC 参数，重算特征并跑识别，返回 (acc, 用时)。"""
    train = rec.featurize("train", **extract_kw)
    test = rec.featurize("test", **extract_kw)
    rows = rec.recognize(train, test, n_templates=5)
    return rec.accuracy(rows), rows


def run_ablation():
    grid = []
    for window in ("rect", "hamming", "hanning"):
        for preemph in (0.97, 0.0):
            grid.append((window, preemph))

    out = []
    for window, preemph in grid:
        acc, _ = accuracy_of(window=window, preemph=preemph)
        out.append((window, preemph, acc))
        print(f"  window={window:8s} preemph={preemph:.2f}  acc={acc:.3f}", flush=True)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / "exp2_ablation.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["窗口", "预加重系数", "识别率", "说明"])
        for window, preemph, acc in out:
            note = "主口径" if (window == "hamming" and preemph == 0.97) else ""
            w.writerow([window, f"{preemph:.2f}", f"{acc:.4f}", note])
    print(f"→ 已写 {path}")
    return out


# ---------- 图 ----------
def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
                 "SimSun", "DengXian"):
        if cand in names:
            plt.rcParams["font.sans-serif"] = [cand]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"  中文字体：{cand}")
            return plt
    print("  !! 没找到中文字体，图里中文会变方块")
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def fig_melbank(plt):
    fb = mfcc.mel_filterbank(config.N_MELS, config.NFFT, config.FS, config.F_LOW, config.F_HIGH)
    freqs = np.arange(fb.shape[1]) * config.FS / config.NFFT

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for m in range(config.N_MELS):
        ax.plot(freqs, fb[m], lw=1.0, color="C0", alpha=0.8)
    ax.set_xlim(0, config.FS / 2)
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel("权值 $H_m(k)$")
    ax.set_title(f"图 2.1  Mel 三角滤波器组（M={config.N_MELS}，N={config.NFFT}，"
                 f"$f_s$={config.FS} Hz，f∈[{config.F_LOW:.0f}, {config.F_HIGH:.0f}] Hz）")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(freqs, fb.sum(axis=0), color="C3", lw=1.6, ls="--")
    ax2.set_ylabel("$\\sum_m H_m(k)$（右轴）", color="C3")
    ax2.set_ylim(0, 1.6)
    ax2.tick_params(axis="y", colors="C3")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_1_melbank.png", dpi=300)
    plt.close(fig)


def fig_triangles(plt):
    centers = mfcc.mel_centers(config.N_MELS, config.FS, config.F_LOW, config.F_HIGH)
    edges_hz = np.concatenate(([config.F_LOW], centers, [config.F_HIGH]))
    # 取第 8、9、10 个三角，画出相邻三者的重叠关系
    m0 = 8
    fb = mfcc.mel_filterbank(config.N_MELS, config.NFFT, config.FS, config.F_LOW, config.F_HIGH)
    freqs = np.arange(fb.shape[1]) * config.FS / config.NFFT

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for m in (m0, m0 + 1, m0 + 2):
        ax.plot(freqs, fb[m], lw=1.8, label=f"$H_{{{m + 1}}}(k)$，峰 {centers[m]:.0f} Hz")
    ax.axvline(edges_hz[m0 + 1], color="0.5", lw=0.8, ls=":")
    ax.axvline(edges_hz[m0 + 2], color="0.5", lw=0.8, ls=":")
    ax.axvline(edges_hz[m0 + 3], color="0.5", lw=0.8, ls=":")
    for m in (m0, m0 + 1, m0 + 2):
        ax.annotate(f"$F_{{{m + 1}}}$={centers[m]:.0f} Hz\n"
                    f"$b_{{{m + 1}}}$={centers[m] * config.NFFT / config.FS:.1f}",
                    xy=(centers[m], 1.0), xytext=(centers[m], 1.12),
                    ha="center", fontsize=8,
                    arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlim(edges_hz[m0] * 0.9, edges_hz[m0 + 4] * 1.02)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel("权值")
    ax.set_title("图 2.2  $F_m$（Hz）换算成频点坐标 $b_m=N\\cdot F_m/f_s$，重叠段权值和为 1",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_2_triangles.png", dpi=300)
    plt.close(fig)


def fig_flow(plt):
    """用一条真实录音把六步画成一条流水线；每一步都贴真实数据，不画示意图。"""
    p = config.wav_paths()[0]
    x, fs = io_wav.read_wav(p)
    fr = framing.frames(x, config.WIN_LEN, config.HOP_LEN)
    energy = mfcc.spectrum_energy(fr, config.NFFT)
    fb = mfcc.mel_filterbank(config.N_MELS, config.NFFT, fs, config.F_LOW, config.F_HIGH)
    mel = energy @ fb.T
    logmel = np.log(mel + mfcc.EPS)
    feat = mfcc.extract(x, fs)

    # 找能量最大的一帧当作示例帧
    k = int(np.argmax(energy.sum(axis=1)))
    spec = energy[k]
    freqs = np.arange(len(spec)) * fs / config.NFFT

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))

    ax = axes[0, 0]
    t = np.arange(len(x)) / fs
    ax.plot(t, x, lw=0.5)
    ax.axvspan(k * config.HOP_LEN / fs, (k * config.HOP_LEN + config.WIN_LEN) / fs,
               color="C3", alpha=0.25)
    ax.set_title("① 输入 1.5 s 波形（红带 = 示例帧）")
    ax.set_xlabel("时间 / s")

    ax = axes[0, 1]
    raw = x[k * config.HOP_LEN: k * config.HOP_LEN + config.WIN_LEN]
    pre = framing.preemph(x)[k * config.HOP_LEN: k * config.HOP_LEN + config.WIN_LEN]
    n = np.arange(config.WIN_LEN)
    ax.plot(n, raw, lw=0.7, label="原始帧")
    ax.plot(n, pre, lw=0.7, label="预加重后 $H(z)=1-0.97z^{-1}$", alpha=0.85)
    ax.set_title("② 预加重：高频被抬起来")
    ax.set_xlabel("帧内点号")
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    w = framing.window(config.WIN_LEN, "hamming")
    for off in (0, 1, 2):
        i = max(k - off, 0)
        seg = x[i * config.HOP_LEN: i * config.HOP_LEN + config.WIN_LEN]
        ax.plot(n + off * config.HOP_LEN, seg, lw=0.6, alpha=0.5)
    ax.plot(n, w * np.abs(x[k * config.HOP_LEN: k * config.HOP_LEN + config.WIN_LEN]).max(),
            color="k", lw=1.2, ls="--", label="汉明窗包络")
    ax.set_title("③ 分帧（相隔 160 点）+ 加窗")
    ax.set_xlabel("样本序号")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(freqs, spec, lw=0.8)
    ax.set_title("④ 512 点 FFT 后的谱线能量 $|X_k|^2$")
    ax.set_xlabel("频率 / Hz")
    ax.set_yscale("log")

    ax = axes[1, 1]
    ax.bar(np.arange(config.N_MELS), mel[k], color="C0")
    ax.set_title("⑤ Mel 滤波：$S_m=\\sum_k |X_k|^2 H_m(k)$（24 个三角）")
    ax.set_xlabel("滤波器 m")

    ax = axes[1, 2]
    im = ax.imshow(feat.T, aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_title("⑥ DCT 后的 MFCC 矩阵 $[T,13]$（含 $c_0$）")
    ax.set_xlabel("帧号")
    ax.set_ylabel("倒谱阶 n")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(f"图 2.3  MFCC 六步流程（{p.name}，含 148 帧；②③④⑤ 画的是第 {k} 帧）",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_3_mfcc_flow.png", dpi=300)
    plt.close(fig)


def make_figs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt = _setup_matplotlib()
    fig_melbank(plt)
    fig_triangles(plt)
    fig_flow(plt)
    print(f"→ 已写 {FIG_DIR}\\fig2_1..2_3")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--no-ablation" in argv:
        print("实验二：--no-ablation → 跳过消融表，只出图 2.1–2.3")
    else:
        print("实验二 消融（判据 = 下游识别率）：")
        run_ablation()
    make_figs()
    return 0


if __name__ == "__main__":
    sys.exit(main())
