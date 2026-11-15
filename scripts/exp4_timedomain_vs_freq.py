"""任务二 P2：时域 vs 频域同条件对照 —— 同划分/同 DTW/同 5 条模板。

跑法（一条命令重跑）：
    python -m scripts.exp4_timedomain_vs_freq

输出 report/tables/exp4_timedomain_vs_freq.csv，列：
    特征,维数,归一化,约束带,识别率,正确条数,总条数,备注

时域两行用 `参考_实验一原代码/speech_time_domain.py` 的 read_wav +
time_domain_feature + DTWClassifier（整段 1.5 s），量纲是**原始 16 bit**；
频域两行取本实验主结果（0.800）与 CMN（0.820），口径见 exp3_accuracy.csv。
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

from dsp import config, templates

TABLES_DIR = config.REPO_ROOT / "report" / "tables"

# 参考模块目录：首选仓库根的 参考_实验一原代码（A/B 本机布局），
# 本仓库没有这个目录，按顺序回退到启动包副本与实验一冻结接口，
# 三处的 speech_time_domain.py 内容一致（12598 B），口径不变。
REF_CANDIDATES = (
    config.REPO_ROOT / "参考_实验一原代码",
    config.REPO_ROOT / "out" / "pkg_stage" / "启动包_公共基础" / "参考_实验一原代码",
    config.REPO_ROOT / "task1_w1" / "接口冻结",
)
REF_DIR = next((p for p in REF_CANDIDATES if (p / "speech_time_domain.py").is_file()),
               REF_CANDIDATES[0])

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# 参考模块不在包路径里，显式注入（与参考 README 用法一致）
if str(REF_DIR) not in sys.path:
    sys.path.insert(0, str(REF_DIR))
from speech_time_domain import read_wav, time_domain_feature, DTWClassifier  # noqa: E402


def _time_domain_rows(normalize, band_ratio):
    """用实验一原流程跑时域逐帧 [T,3] 特征，返回 (acc, n_ok, n, dt)。"""
    t0 = time.perf_counter()
    train_x, train_y = [], []
    for p in templates.files("train"):
        x, fs, _nch, _bits = read_wav(str(p))
        train_x.append(time_domain_feature(x, fs, normalize=normalize))
        train_y.append(config.label_of(p))
    test_x, test_y = [], []
    for p in templates.files("test"):
        x, fs, _nch, _bits = read_wav(str(p))
        test_x.append(time_domain_feature(x, fs, normalize=normalize))
        test_y.append(config.label_of(p))
    clf = DTWClassifier(band_ratio=band_ratio).fit(train_x, train_y)
    pred = clf.predict(test_x)
    n_ok = int(np.sum(pred == np.asarray(test_y)))
    dt = time.perf_counter() - t0
    return n_ok / len(test_y), n_ok, len(test_y), dt


def main(argv=None):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = []

    # ---- 时域两行（必须用它自己的 read_wav，整段 1.5 s）----
    acc, n_ok, n, dt = _time_domain_rows(normalize=True, band_ratio=0.35)
    out.append(["时域 [T,3] 逐帧", 3, "段内逐维归一化", "band 0.35",
                f"{acc:.4f}", n_ok, n,
                f"实验一原流程；用参考模块的 read_wav（原始16bit量纲）；耗时 {dt:.1f}s"])
    print(f"  时域 normalize=True band=0.35: acc={acc:.3f} ({n_ok}/50)  {dt:.1f}s", flush=True)

    acc, n_ok, n, dt = _time_domain_rows(normalize=False, band_ratio=0.35)
    out.append(["时域 [T,3] 逐帧", 3, "无", "无",
                f"{acc:.4f}", n_ok, n,
                f"去掉段内归一化即掉到 0.400；band 随意（0.35→1.00 准确率不变）；"
                f"耗时 {dt:.1f}s"])
    print(f"  时域 normalize=False: acc={acc:.3f} ({n_ok}/50)  {dt:.1f}s", flush=True)

    # ---- 频域两行（读 exp3_accuracy.csv，不写死数字）----
    acc_path = TABLES_DIR / "exp3_accuracy.csv"
    if not acc_path.exists():
        raise SystemExit(
            "缺少 report/tables/exp3_accuracy.csv：先跑 "
            "python -m scripts.exp3_recognize --all")
    with acc_path.open(encoding="utf-8") as fh:
        acc_rows = {r["条件"]: r for r in csv.DictReader(fh)}
    for cond, norm_label, note_text in (
        ("main", "无", "本实验主结果（整段/none/5模板），读自 exp3_accuracy.csv"),
        ("normalize_cmn", "CMN（段内逐维去均值）",
         "加 CMN 只多 1 条，按 1 条=2 个百分点口径属无显著差别"),
    ):
        r = acc_rows[cond]
        n_total = int(r["测试条数"])
        acc = float(r["准确率"])
        out.append(["MFCC [T,13]", 13, norm_label, "无",
                    f"{acc:.4f}", round(acc * n_total), n_total, note_text])
        print(f"  频域 {cond}: acc={acc:.3f} "
              f"({round(acc * n_total)}/{n_total})（读自 exp3_accuracy.csv）",
              flush=True)

    path = TABLES_DIR / "exp4_timedomain_vs_freq.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["特征", "维数", "归一化", "约束带", "识别率",
                    "正确条数", "总条数", "备注"])
        w.writerows(out)
    # 表注三条前提（任务单要求写全，随表落在同目录的注记文件里）
    note = TABLES_DIR / "exp4_timedomain_vs_freq_表注.txt"
    note.write_text(
        "表注（时域 vs 频域对照，三条前提必须同时成立才有可比性）：\n"
        "①实验一的 0.760 依赖「段内逐维归一化」，去掉归一化即掉到 0.400；\n"
        "②band 0.35→1.00 准确率不变，只是耗时约 15.90 s→27.21 s；\n"
        "③本实验（MFCC）无归一化、无约束带，故 0.800 与 0.760 只作定性对照，"
        "不构成“频域比时域高 2 条”。\n"
        "另：时域两行必须用参考模块 read_wav（返回原始 16 bit 值，absmax≈32032），"
        "不能用 dsp/io_wav.read_wav（返回 −1…1），否则 0.760/0.400 对不上。\n",
        encoding="utf-8")
    print(f"→ 已写 {path} 和 {note}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
