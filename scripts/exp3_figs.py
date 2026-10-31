"""实验三的两张图：系统框图、DTW 代价矩阵与最优路径。

跑法：``python -m scripts.exp3_figs``
产物：report/fig/fig3_1_system.png、report/fig/fig3_2_dtw_grid.png
"""

from __future__ import annotations

import sys

import numpy as np

from dsp import config, dtw, endpoint, io_wav, mfcc, templates
from scripts.exp2_ablation import FIG_DIR, _setup_matplotlib

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def fig_system(plt):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12.4, 5.0))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#EAF1FB", ec="#2E5C9A", fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.1",
                                    fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                linespacing=1.45)

    def arrow(x1, y1, x2, y2, ls="-", color="#333333"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.25, ls=ls, color=color))

    # ---- 识别路径：一行 6 个盒子，箭头全部水平 ----
    y1, h1, w1, gap = 3.35, 1.0, 1.62, 0.34
    xs = [0.35 + i * (w1 + gap) for i in range(6)]
    labels1 = [
        "读入 wav\n1.5 s / 16 kHz",
        "预加重 + 分帧\n400 / 160 点",
        "端点检测\n双门限 E + ZCR",
        "MFCC [T, 13]\n（含 $c_0$）",
        "DTW\n三条前驱递推",
        "判决输出\n最近模板",
    ]
    for x, t in zip(xs, labels1):
        box(x, y1, w1, h1, t, fs=8.6)
    for i in range(5):
        arrow(xs[i] + w1, y1 + h1 / 2, xs[i + 1], y1 + h1 / 2)
    ax.text(xs[0], y1 + h1 + 0.2, "① 识别路径：测试集 06–10，每类 5 条，共 50 条",
            fontsize=9.5, color="#2E5C9A")
    ax.text(xs[2] + w1 / 2, y1 - 0.42,
            "（端点检测是消融项，可关；\n主结果用整段 1.5 s）",
            ha="center", fontsize=7.5, color="#777777")

    # ---- 建模板路径：一行 3 个盒子 ----
    y2, h2 = 1.05, 1.0
    box(0.35, y2, 1.62, h2, "训练录音 50 条\n每类 01–05", fc="#FBF3E6", ec="#A9741F", fs=8.6)
    arrow(0.35 + 1.62, y2 + h2 / 2, 2.55, y2 + h2 / 2)
    box(2.55, y2, 1.85, h2, "MFCC\n同一套参数", fc="#FBF3E6", ec="#A9741F", fs=8.6)
    arrow(4.4, y2 + h2 / 2, 5.9, y2 + h2 / 2)
    box(5.9, y2, 2.3, h2, "模板库\n每类 5 条 [T,13]", fc="#FBF3E6", ec="#A9741F", fs=8.6)
    ax.text(0.35, y2 - 0.45, "② 建模板路径：训练集 01–05", fontsize=9.5, color="#A9741F")

    # 模板库 → DTW（斜向上，接到第 5 个盒子）
    x_dtw = xs[4] + w1 / 2
    arrow(7.2, y2 + h2, x_dtw - 0.12, y1, ls="--", color="#A9741F")
    ax.text(8.55, 2.45, "取最小距离", fontsize=7.5, color="#A9741F")

    # MFCC（建模板）与 MFCC（识别）之间：同一套参数
    ax.text(3.5, 2.62, "mfcc.extract() 是唯一入口：\n模板与查询必须同参数",
            ha="center", fontsize=7.5, color="#777777")

    ax.set_title("图 3.1  实验三系统框图：两条路径共用同一套 MFCC，DTW 只负责帧数不等时的时间对齐",
                 fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_1_system.png", dpi=300)
    plt.close(fig)


def fig_grid(plt):
    """一张真实的 148×148 代价矩阵 + 最优路径。"""
    test_path = templates.files("test")[0]
    train_path = templates.files("train")[0]
    q, fs = io_wav.read_wav(test_path)
    r, _ = io_wav.read_wav(train_path)
    q = mfcc.extract(q, fs)
    r = mfcc.extract(r, fs)

    dist, path = dtw.alignment_path(q, r)
    D, d = dtw.accumulate(q, r)
    Dm = np.array([[D[i][j] for j in range(1, D[0].__len__())] for i in range(1, len(D))])
    Dm = np.where(np.isinf(Dm), np.nan, Dm)

    fig = plt.figure(figsize=(12.5, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                          wspace=0.28, hspace=0.35)

    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(Dm, origin="lower", cmap="magma", aspect="auto")
    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    ax.plot(xs, ys, color="#38E1A0", lw=1.2, label=f"最优路径（{len(path)} 步）")
    ax.set_title(f"图 3.2a  累计代价矩阵 $D(i,j)$，$D(N,M)$={dist:.4f}\n"
                 f"查询 {test_path.name}（{len(q)} 帧）× 模板 {train_path.name}（{len(r)} 帧）",
                 fontsize=9.5)
    ax.set_xlabel("模板帧 j")
    ax.set_ylabel("查询帧 i")
    ax.legend(fontsize=8, loc="upper left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(q.T, origin="lower", aspect="auto", cmap="coolwarm")
    ax.set_title("查询 MFCC（13 阶）", fontsize=9)
    ax.set_ylabel("阶")

    ax = fig.add_subplot(gs[1, 1])
    dm = d.copy()
    ax.imshow(dm, origin="lower", aspect="auto", cmap="viridis")
    ax.plot(xs, ys, color="#FF3B3B", lw=0.9)
    ax.set_title("帧距矩阵 $d(i,j)$（最优路径叠在上面）", fontsize=9)
    ax.set_xlabel("模板帧 j")
    ax.set_ylabel("查询帧 i")

    fig.savefig(FIG_DIR / "fig3_2_dtw_grid.png", dpi=300)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt = _setup_matplotlib()
    fig_system(plt)
    fig_grid(plt)
    print(f"→ 已写 {FIG_DIR}\\fig3_1_system.png、fig3_2_dtw_grid.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
