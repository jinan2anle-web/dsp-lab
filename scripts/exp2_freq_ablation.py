"""任务一 P1：频域变体消融 —— 换一档频域参数，下游 DTW 识别率值不值。

跑法（一条命令重跑）：
    python -m scripts.exp2_freq_ablation

判据只能是**下游 DTW 识别率**（单看 MFCC 数值没有可比性）。
每个条件只动一个量；模板与测试用同一套参数。输出：
    report/tables/exp2_freq_ablation.csv
列：条件,参数,识别率,正确条数,总条数,耗时s,备注

依赖 A 周三交付的 `mfcc.extract(form=, use_delta=)`（logmel / ΔΔ 两行）。
"""

from __future__ import annotations

import csv
import sys
import time

import numpy as np

from dsp import config
from scripts import exp3_recognize as rec

TABLES_DIR = config.REPO_ROOT / "report" / "tables"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def accuracy_of(**extract_kw):
    """换一组提取参数，重算 train/test 特征并跑识别，返回 (acc, n_ok, rows, dt)。"""
    t0 = time.perf_counter()
    train = rec.featurize("train", **extract_kw)
    test = rec.featurize("test", **extract_kw)
    rows = rec.recognize(train, test, n_templates=5)
    dt = time.perf_counter() - t0
    return rec.accuracy(rows), sum(1 for r in rows if r[4]), rows, dt


# (条件, 参数, kwargs, 备注)  —— 基线必须复现 0.800/40
CONDS = [
    ("基线 24Mel/13维/汉明/0.97", "base", {}, "主口径，= 主结果 0.800"),
    ("Mel 数 16", "16", {"n_mels": 16}, "MFCC 维数仍取 13"),
    ("Mel 数 32", "32", {"n_mels": 32}, "MFCC 维数仍取 13"),
    ("无 DCT（log-Mel）", "logmel", {"form": "logmel"}, "24 维直接进 DTW（A 的 form 参数）"),
    ("加 Δ/ΔΔ", "delta", {"use_delta": True}, "39 维 = 13 静态 + 13 Δ + 13 ΔΔ（A 的 use_delta）"),
    ("MFCC 20 维", "20", {"n_mfcc": 20}, "n_mels 仍 24"),
]


def main(argv=None):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for name, param, kw, note in CONDS:
        acc, n_ok, _rows, dt = accuracy_of(**kw)
        out.append((name, param, acc, n_ok, len(_rows), dt, note))
        print(f"  {name:24s} acc={acc:.3f} ({n_ok}/50)  {dt:.1f}s", flush=True)

    path = TABLES_DIR / "exp2_freq_ablation.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["条件", "参数", "识别率", "正确条数", "总条数", "耗时s", "备注"])
        for name, param, acc, n_ok, n, dt, note in out:
            w.writerow([name, param, f"{acc:.4f}", n_ok, n, f"{dt:.1f}", note])
    print(f"→ 已写 {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
