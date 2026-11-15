"""生成图 2.4 语谱图与图 2.5 三时刻功率谱对比。"""

from __future__ import annotations

import sys

import numpy as np

from dsp import config, endpoint, framing, io_wav, mfcc
from scripts.exp2_ablation import _setup_matplotlib

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

FIG_DIR = config.REPORT_DIR / "fig"
EVIDENCE_DIR = config.REPORT_DIR / "evidence"
WAV_PATH = config.DATA_DIR / "0_01.wav"
FRAME_IDS = (5, 40, 95)


def _spectra(x, preemph):
    fr = framing.frames(x, config.WIN_LEN, config.HOP_LEN,
                        window="hamming", preemph=preemph)
    return mfcc.spectrum_energy(fr, config.NFFT)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    x, fs = io_wav.read_wav(WAV_PATH)
    raw_energy = _spectra(x, 0.0)
    pre_energy = _spectra(x, config.PREEMPH)
    times = (np.arange(raw_energy.shape[0]) * config.HOP_LEN + config.WIN_LEN / 2) / fs
    freqs = np.arange(raw_energy.shape[1]) * fs / config.NFFT
    got = endpoint.detect(x, fs)
    if got is None:
        s0, s1 = 0, len(x)
    else:
        s0, s1 = got
    start_s, end_s = s0 / fs, s1 / fs

    plt = _setup_matplotlib()
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10})

    db = 10.0 * np.log10(pre_energy + mfcc.EPS)
    vmin, vmax = -90.0, 25.0
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(db.T, origin="lower", aspect="auto", cmap="magma",
                   extent=(times[0], times[-1], freqs[0], freqs[-1]),
                   vmin=vmin, vmax=vmax)
    ax.axvspan(0.0, start_s, color="#5B8FF9", alpha=0.18, label="静音前导区")
    ax.axvline(start_s, color="#00A087", lw=1.5, ls="--", label=f"语音起点 {start_s:.3f} s")
    ax.axvline(end_s, color="#D55E00", lw=1.5, ls="--", label=f"语音终点 {end_s:.3f} s")
    ax.set_xlim(0.0, len(x) / fs)
    ax.set_ylim(0.0, fs / 2)
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("频率 / Hz")
    ax.set_title("图 2.4  0_01.wav 的短时语谱图")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("谱能量 / dB")
    fig.tight_layout()
    path_spectrogram = FIG_DIR / "fig2_4_spectrogram.png"
    fig.savefig(path_spectrogram, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    high = (freqs >= 4000.0) & (freqs <= 8000.0)
    lift_db = []
    for ax, frame_id in zip(axes, FRAME_IDS):
        raw = raw_energy[frame_id]
        pre = pre_energy[frame_id]
        time_s = (frame_id * config.HOP_LEN + config.WIN_LEN / 2) / fs
        ax.semilogy(freqs, raw + mfcc.EPS, lw=1.0, color="#4C78A8", label="预加重前")
        ax.semilogy(freqs, pre + mfcc.EPS, lw=1.0, color="#E45756", label="预加重后")
        ax.axvspan(4000, 8000, color="#F2CF5B", alpha=0.16)
        ax.set_xlim(0, 8000)
        ax.set_xlabel("频率 / Hz")
        ax.set_title(f"第 {frame_id} 帧（{time_s:.3f} s）")
        ax.grid(alpha=0.25)
        raw_hf = float(raw[high].sum())
        pre_hf = float(pre[high].sum())
        lift = 10.0 * np.log10((pre_hf + mfcc.EPS) / (raw_hf + mfcc.EPS))
        lift_db.append(lift)
        ax.text(0.98, 0.04, f"4–8 kHz: {lift:+.2f} dB", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8)
    axes[0].set_ylabel("功率谱 $|X[k]|^2$（对数轴）")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("图 2.5  静音、语音起始与语音中段的功率谱：预加重前后对比", fontsize=11)
    fig.tight_layout()
    path_power = FIG_DIR / "fig2_5_power_spectrum.png"
    fig.savefig(path_power, dpi=300)
    plt.close(fig)

    size_24 = (2700, 1440)
    size_25 = (3900, 1260)
    lines = [
        "实验二频域图证据",
        f"source = {WAV_PATH.name}",
        f"spectrogram_matrix = {pre_energy.shape}",
        f"fig2_4_pixels = {size_24[0]} x {size_24[1]}",
        f"fig2_5_pixels = {size_25[0]} x {size_25[1]}",
        f"endpoint_samples = [{s0}, {s1})",
        f"endpoint_seconds = [{start_s:.6f}, {end_s:.6f})",
        f"color_limits_db = [{vmin:.1f}, {vmax:.1f}]",
        f"frame_ids = {FRAME_IDS}",
    ]
    for frame_id, lift in zip(FRAME_IDS, lift_db):
        lines.append(f"frame_{frame_id}_max_energy_raw = {raw_energy[frame_id].max():.12e}")
        lines.append(f"frame_{frame_id}_max_energy_preemph = {pre_energy[frame_id].max():.12e}")
        lines.append(f"frame_{frame_id}_4_8k_lift_db = {lift:.6f}")
    evidence = EVIDENCE_DIR / "exp2_freq_figs.txt"
    evidence.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"→ 已写 {path_spectrogram}")
    print(f"→ 已写 {path_power}")
    print(f"→ 已写 {evidence}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
