"""任务四 T3：模板数曲线 —— 每类模板 1…5 条各跑一行。

跑法（一条命令重跑）：
    python -m scripts.exp3_template_curve

模板固定取训练集 01–05 的前 n 条（顺序固定、不随机挑）。
输出 report/tables/exp3_template_curve.csv，列：
    模板数,识别率,正确条数,总条数,耗时s
1/3/5 三行须与 exp3_accuracy.csv 完全一致（0.600/0.740/0.800）。
"""

from __future__ import annotations

import csv
import sys
import time

from dsp import config
from scripts import exp3_recognize as rec

TABLES_DIR = config.REPO_ROOT / "report" / "tables"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def main(argv=None):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    # 训练/测试特征只算一次（默认口径），模板数在 recognize 里截断，避免重复提特征
    train = rec.featurize("train")
    test = rec.featurize("test")
    out = []
    for n in range(1, 6):
        t0 = time.perf_counter()
        rows = rec.recognize(train, test, n_templates=n)
        dt = time.perf_counter() - t0
        acc = rec.accuracy(rows)
        n_ok = sum(1 for r in rows if r[4])
        out.append((n, acc, n_ok, len(rows), dt))
        print(f"  模板 {n} 条: acc={acc:.3f} ({n_ok}/50)  {dt:.1f}s", flush=True)

    path = TABLES_DIR / "exp3_template_curve.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["模板数", "识别率", "正确条数", "总条数", "耗时s"])
        for n, acc, n_ok, total, dt in out:
            w.writerow([n, f"{acc:.4f}", n_ok, total, f"{dt:.1f}"])
    print(f"→ 已写 {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
