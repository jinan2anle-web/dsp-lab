"""任务三 T1/T2：鲁棒性 —— 加性白噪声 + 整体增益，本周重点。

跑法（一条命令重跑）：
    python -m scripts.exp5_robustness

协议见 docs/测试实验设计.md 第 3/4 节：
    T1 噪声：SNR ∈ {inf, 20, 10, 5, 0} dB × seed ∈ {0,1,2} = 15 行
             （inf 档不重复 3 次，只 1 行；再加 1 行 seed=avg 写 3 次平均 ± 极差）
    T2 增益：factor ∈ {0.5, 1.0, 2.0} × normalize ∈ {none, cmn} = 6 行
输出 report/tables/exp5_robustness.csv，列：
    实验,条件,参数,重复次数,识别率,正确条数,总条数,耗时s,seed,备注

依赖 A 周三交付的 dsp/augment.py（add_white_noise / apply_gain）。
开跑前必须通过 _snr_selftest()：设定与实测 SNR 偏差 < 0.5 dB。
"""

from __future__ import annotations

import csv
import sys
import time

import numpy as np

from dsp import augment, config, io_wav, mfcc, templates
from scripts import exp3_recognize as rec

TABLES_DIR = config.REPO_ROOT / "report" / "tables"
SNR_TIERS = (float("inf"), 20.0, 10.0, 5.0, 0.0)
SEEDS = (0, 1, 2)
GAINS = (0.5, 1.0, 2.0)
NORM_MODES = ("none", "cmn")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def _snr_selftest():
    """开跑前核 SNR 口径（3 行），偏差 > 0.5 dB 直接抛错，不白跑 750 次。"""
    p = config.wav_paths()[0]
    x0, fs = io_wav.read_wav(p)
    # 语音段 5600:23760（= 端点检测区间，见任务单），与 add_white_noise 内部口径一致
    s0, s1 = 5600, 23760
    p_s = float(np.mean(x0[s0:s1] ** 2))
    lines = []
    ok = True
    for snr in (20.0, 10.0, 0.0):
        x1 = augment.add_white_noise(x0, snr, seed=0)
        measured = 10.0 * np.log10(p_s / float(np.mean((x1 - x0) ** 2)))
        delta = abs(measured - snr)
        ok = ok and delta < 0.5
        lines.append((snr, measured, delta))
    print("  SNR 自测：")
    for snr, measured, delta in lines:
        print(f"    设定 {snr:>4} dB  实测 {measured:6.3f} dB  偏差 {delta:.4f} dB"
              f"  {'OK' if delta < 0.5 else 'FAIL'}")
    if not ok:
        raise RuntimeError("SNR 实测偏差 ≥ 0.5 dB，噪声口径错误，终止（避免白跑 750 次）")
    return lines


def _clear_featurize(part, **extract_kw):
    """干净特征（训练模板或基线测试）。"""
    return rec.featurize(part, **extract_kw)


def _noisy_featurize(part, snr_db, seed, **extract_kw):
    """对该部分每条录音加噪后再提特征（训练模板恒干净，只动测试）。"""
    out = []
    for p in templates.files(part):
        x, fs = io_wav.read_wav(p)
        x1 = augment.add_white_noise(x, snr_db, fs=fs, seed=seed)
        feat = mfcc.extract(x1, fs, **extract_kw)
        out.append((p, config.label_of(p), feat))
    return out


def _gain_featurize(part, factor, **extract_kw):
    out = []
    for p in templates.files(part):
        x, fs = io_wav.read_wav(p)
        x1 = augment.apply_gain(x, factor)
        feat = mfcc.extract(x1, fs, **extract_kw)
        out.append((p, config.label_of(p), feat))
    return out


def _acc_rows(train, test):
    rows = rec.recognize(train, test, n_templates=5)
    n_ok = sum(1 for r in rows if r[4])
    return rec.accuracy(rows), n_ok, len(rows)


def run_t1(train):
    """T1 噪声：训练模板干净，测试加噪。按 seed 分行，另补 seed=avg。"""
    out = []
    for snr in SNR_TIERS:
        # 5 档 × 3 seed = 15 行：∞ 档也写满 3 行（无噪，三次结果必然相同）
        label = "SNR ∞（无噪）" if np.isinf(snr) else f"SNR {snr:.0f} dB"
        param = "inf" if np.isinf(snr) else f"{snr:.0f}"
        per_seed = []
        for seed in SEEDS:
            t0 = time.perf_counter()
            test = _noisy_featurize("test", snr, seed)
            acc, n_ok, n = _acc_rows(train, test)
            dt = time.perf_counter() - t0
            per_seed.append((acc, n_ok))
            note = ("基线：无噪，seed 无影响（3 行同值）" if np.isinf(snr)
                    else "噪声按语音段功率定，整条加噪")
            out.append(["T1_noise", label, param, 1, f"{acc:.4f}", n_ok, n,
                        f"{dt:.1f}", str(seed), note])
            print(f"  [T1] {label:12s} seed={seed}  acc={acc:.3f} ({n_ok}/50)  {dt:.1f}s",
                  flush=True)
        # 每档另补一行 seed=avg：3 次平均识别率 ± 极差
        accs = np.array([a for a, _ in per_seed])
        noks = np.array([k for _, k in per_seed])
        avg, rng = float(accs.mean()), float(accs.max() - accs.min())
        note = f"3次平均±极差: {avg:.3f}±{rng:.3f}; 正确条数 {noks.min()}-{noks.max()}/50"
        out.append(["T1_noise", label + "（3次平均）", param, 3,
                    f"{avg:.4f}", int(round(noks.mean())), 50, "", "avg", note])
    return out


def run_t2():
    """T2 增益：factor × normalize，训练与测试同口径提取，只给测试乘增益。"""
    out = []
    for factor in GAINS:
        for mode in NORM_MODES:
            t0 = time.perf_counter()
            kw = {"normalize": mode}
            train = _clear_featurize("train", **kw)
            test = _gain_featurize("test", factor, **kw)
            acc, n_ok, n = _acc_rows(train, test)
            dt = time.perf_counter() - t0
            out.append(["T2_gain", f"gain ×{factor:.1f} / {mode}", f"{factor:.1f}", 1,
                        f"{acc:.4f}", n_ok, n, f"{dt:.1f}", "0",
                        f"normalize={mode}; 整体乘常数"])
            print(f"  [T2] gain x{factor:.1f} {mode:4s}  acc={acc:.3f} ({n_ok}/50)  {dt:.1f}s",
                  flush=True)
    return out


def main(argv=None):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print("T1/T2 鲁棒性实验：")
    _snr_selftest()
    train = _clear_featurize("train")
    rows = run_t1(train) + run_t2()

    path = TABLES_DIR / "exp5_robustness.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["实验", "条件", "参数", "重复次数", "识别率", "正确条数",
                    "总条数", "耗时s", "seed", "备注"])
        w.writerows(rows)
    print(f"→ 已写 {path}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
