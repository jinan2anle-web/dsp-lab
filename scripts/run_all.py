"""一条命令复现全部数字与图：自测 → 实验二消融+图 → 实验三识别+消融 → 实验三图。

跑法：

    python -m scripts.run_all            # 全跑（约 3 分钟）
    python -m scripts.run_all --fast     # 跳过消融，只跑主结果与图

产物位置（与 docs/实验二三_分工与交接单.md §5.2 一致）：
    report/evidence/*.txt   自测证据
    report/tables/*.csv     报告里每个数字的出处
    report/fig/*.png        5 张自绘图 + 界面截图
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

STEPS = [
    ("自测：FFT 对拍 / MFCC 形状 / DTW 手算表", ["-m", "scripts.selftest"], []),
    ("实验二：窗口与预加重消融 + 图 2.1–2.3", ["-m", "scripts.exp2_ablation"], ["--no-ablation"]),
    ("实验三：识别率 + 混淆矩阵 + 图 3.2", ["-m", "scripts.exp3_recognize", "--all"], []),
    ("实验三：系统框图 + DTW 网格", ["-m", "scripts.exp3_figs"], []),
]

TRACE = [
    ("FFT 与 np.fft 的最大误差", "report/evidence/exp2_fft_selftest.txt",
     "python -m scripts.selftest"),
    ("148 帧 / [T,13] 形状", "report/evidence/exp2_fft_selftest.txt; report/evidence/exp2_mfcc_shape.txt",
     "python -m scripts.selftest"),
    ("Σ_m H_m(k) 恒为 1", "report/evidence/exp2_fft_selftest.txt", "python -m scripts.selftest"),
    ("三窗 × 预加重的识别率", "report/tables/exp2_ablation.csv", "python -m scripts.exp2_ablation"),
    ("DTW 3×3 手算表", "report/evidence/exp3_dtw_handcalc.txt", "python -m scripts.selftest"),
    ("主结果识别率 40/50", "report/tables/exp3_accuracy.csv; report/tables/exp3_predictions.csv",
     "python -m scripts.exp3_recognize --all"),
    ("混淆矩阵", "report/tables/exp3_confusion.csv", "python -m scripts.exp3_recognize --all"),
    ("模板数 1/3/5 对照", "report/tables/exp3_accuracy.csv", "python -m scripts.exp3_recognize --all"),
    ("normalize none/cmn 对照", "report/tables/exp3_accuracy.csv", "python -m scripts.exp3_recognize --all"),
    ("端点段 vs 整段", "report/tables/exp3_accuracy.csv; report/tables/exp3_predictions_endpoint.csv",
     "python -m scripts.exp3_recognize --all"),
    ("5 张自绘图", "report/fig/fig2_1_melbank.png … fig3_2_dtw_grid.png",
     "python -m scripts.exp2_ablation; python -m scripts.exp3_figs"),
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
    for desc, cmd, skip_flags in STEPS:
        if fast and "--no-ablation" in skip_flags and "exp2_ablation" in cmd:
            print(f"\n== 跳过（--fast）：{desc}")
            continue
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
    print(f"\n→ 已写 {EVIDENCE_DIR / 'run_all.log'}")

    write_traceability(TRACE)
    return 0 if all(c == 0 for _, c, _ in log) else 1


if __name__ == "__main__":
    sys.exit(main())
