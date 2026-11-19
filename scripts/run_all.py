"""一条命令复现全部数字与图：自测 → 实验二消融+图 → 频域图/频域消融 → 实验三识别+消融 → 实验三图。

跑法：

    python -m scripts.run_all            # 全跑（本机实测 477 s，约 8 分钟；跨机波动 ±50%）
    python -m scripts.run_all --fast     # 跳过消融类，只跑主结果与图（本机实测 48 s）

产物位置（与 docs/实验二三_分工与交接单.md §5.2 一致）：
    report/evidence/*.txt   自测证据
    report/tables/*.csv     报告里每个数字的出处
    report/fig/*.png        7 张自绘图 + 界面截图
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time

from dsp import config

TABLES_DIR = config.REPO_ROOT / "report" / "tables"
EVIDENCE_DIR = config.REPO_ROOT / "report" / "evidence"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# (描述, 命令, --fast 处理, 额外参数)
#   keep = --fast 照跑；figs = --fast 只跳过消融表、图照出；skip = --fast 整步跳过
STEPS = [
    ("自测：FFT 对拍 / MFCC 形状 / DTW 手算表", ["-m", "scripts.selftest"], "keep", []),
    ("实验二：窗口与预加重消融 + 图 2.1–2.3", ["-m", "scripts.exp2_ablation"], "figs",
     ["--no-ablation"]),
    ("实验二：语谱图与功率谱（图 2.4–2.5）", ["-m", "scripts.exp2_freq_figs"], "keep", []),
    ("实验二：频域变体消融（表 4）", ["-m", "scripts.exp2_freq_ablation"], "skip", []),
    ("实验三 T1/T2：噪声与增益鲁棒性", ["-m", "scripts.exp5_robustness"], "skip", []),
    ("实验三：模板数 1–5 曲线", ["-m", "scripts.exp3_template_curve"], "skip", []),
    ("实验三：识别率 + 混淆矩阵 + 图 3.2", ["-m", "scripts.exp3_recognize", "--all"], "keep", []),
    ("实验三：系统框图 + DTW 网格", ["-m", "scripts.exp3_figs"], "keep", []),
]

TRACE = [
    ("FFT 与 np.fft 的最大误差", "report/evidence/exp2_fft_selftest.txt",
     "python -m scripts.selftest"),
    ("148 帧 / [T,13] 形状", "report/evidence/exp2_fft_selftest.txt; report/evidence/exp2_mfcc_shape.txt",
     "python -m scripts.selftest"),
    ("覆盖范围内 Σ_m H_m(k) = 1（k=3…230）", "report/evidence/exp2_fft_selftest.txt",
     "python -m scripts.selftest"),
    ("三窗 × 预加重的识别率", "report/tables/exp2_ablation.csv", "python -m scripts.exp2_ablation"),
    ("图 2.4 语谱图（0_01.wav，帧 5/40/95）", "report/fig/fig2_4_spectrogram.png",
     "python -m scripts.exp2_freq_figs"),
    ("图 2.5 功率谱（三帧 × 4–8 kHz 抬升）", "report/fig/fig2_5_power_spectrum.png",
     "python -m scripts.exp2_freq_figs"),
    ("频域变体消融（Mel 16/32、log-Mel、ΔΔ、20 维）", "report/tables/exp2_freq_ablation.csv",
     "python -m scripts.exp2_freq_ablation"),
    ("时域 [T,3] vs 频域 MFCC [T,13] 同条件对照", "report/tables/exp4_timedomain_vs_freq.csv",
     "python -m scripts.exp4_timedomain_vs_freq"),
    ("鲁棒性 T1 噪声（SNR ∞/20/10/5/0 dB × seed 0–2）与 T2 增益（×0.5/1.0/2.0 × none/cmn）",
     "report/tables/exp5_robustness.csv", "python -m scripts.exp5_robustness"),
    ("DTW 3×3 手算表", "report/evidence/exp3_dtw_handcalc.txt", "python -m scripts.selftest"),
    ("端点门限扫描（表 5）", "report/tables/exp3_endpoint_sweep.csv",
     "python -m scripts.exp3_recognize --all"),
    ("各条件耗时（表 7 注）", "report/tables/exp3_accuracy.csv（耗时s 列）",
     "python -m scripts.exp3_recognize --all"),
    ("主结果识别率 40/50", "report/tables/exp3_accuracy.csv; report/tables/exp3_predictions.csv",
     "python -m scripts.exp3_recognize --all"),
    ("混淆矩阵", "report/tables/exp3_confusion.csv", "python -m scripts.exp3_recognize --all"),
    ("模板数 1/3/5 对照", "report/tables/exp3_accuracy.csv; report/tables/exp3_template_curve.csv",
     "python -m scripts.exp3_recognize --all; python -m scripts.exp3_template_curve"),
    ("normalize none/cmn 对照", "report/tables/exp3_accuracy.csv", "python -m scripts.exp3_recognize --all"),
    ("端点段 vs 整段", "report/tables/exp3_accuracy.csv; report/tables/exp3_predictions_endpoint.csv",
     "python -m scripts.exp3_recognize --all"),
    ("7 张自绘图", "report/fig/fig2_1_melbank.png … fig3_2_dtw_grid.png",
     "python -m scripts.exp2_ablation; python -m scripts.exp2_freq_figs; python -m scripts.exp3_figs"),
    ("界面截图", "report/fig/app_main.png", "python -m dsp.app"),
]


def run(argv):
    cmd = [sys.executable] + argv
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(config.REPO_ROOT))
    dt = time.perf_counter() - t0
    print(f"[{proc.returncode}] 用时 {dt:.1f}s", flush=True)
    return proc.returncode, dt


def write_traceability(rows):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / "traceability.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["报告里的数字", "出处文件", "复现命令"])
        w.writerows(rows)
    print(f"→ 已写 {path}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    fast = "--fast" in argv

    log = []
    skipped = []
    for desc, cmd, fast_mode, extra in STEPS:
        if fast and fast_mode == "skip":
            print(f"\n== --fast：{desc}（消融类，整步跳过）")
            skipped.append(desc)
            continue
        if fast and fast_mode == "figs":
            # --fast 只跳过消融表，图 2.1–2.3 照出（图也在这一步的 make_figs 里）
            cmd = cmd + extra
            print(f"\n== --fast：{desc}（跳过消融表，图照出）")
        print(f"\n================ {desc}")
        code, dt = run(cmd)
        log.append((desc, code, round(dt, 1)))
        if code != 0:
            print(f"!! 上面这一步失败（exit {code}），后面的数字不可信，先修它。")
            break

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    with (EVIDENCE_DIR / "run_all.log").open("w", encoding="utf-8") as fh:
        fh.write(f"python {sys.version.split()[0]}\n\n")
        for desc, code, dt in log:
            fh.write(f"[{code}] {dt:>6.1f}s  {desc}\n")
        for desc in skipped:
            fh.write(f"[skip]        --fast 跳过  {desc}\n")
    print(f"\n→ 已写 {EVIDENCE_DIR / 'run_all.log'}")

    ok = (len(log) + len(skipped) == len(STEPS)
          and all(c == 0 for _, c, _ in log))
    if ok:
        write_traceability(TRACE)
    else:
        print("!! 有步骤没跑完或失败：本次不写 traceability.csv，"
              "否则会把未生成的产物登记成「有出处」。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
