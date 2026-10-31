"""实验三：DTW 孤立字识别（模板 5 条/类，最近模板判决）。

用法：

    python -m scripts.exp3_recognize --all          # 主结果 + 全部消融，数字写进 report/
    python -m scripts.exp3_recognize                # 只跑主结果

口径（与 C01–C10 一致，改一处就要重跑全部）：
    训练 = 编号 01–05（每类 5 条模板），测试 = 编号 06–10（每类 5 条，共 50 条）
    主结果 = 整段输入 + normalize="none" + 5 条模板
    消融  = 端点段输入 / normalize="cmn" / 模板数 1、3
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from dsp import config, dtw, endpoint, io_wav, mfcc, templates

TABLES_DIR = config.REPO_ROOT / "report" / "tables"
EVIDENCE_DIR = config.REPO_ROOT / "report" / "evidence"


def featurize(part, *, use_endpoint=False, normalize="none", **kw):
    """把一部分录音算成特征，返回 ``[(path, label, mfcc), ...]``（按文件名序）。"""
    out = []
    for p in templates.files(part):
        x, fs = io_wav.read_wav(p)
        if use_endpoint:
            x, _ = endpoint.segment(x, fs)
        feat = mfcc.extract(x, fs, normalize=normalize, **kw)
        out.append((p, config.label_of(p), feat))
    return out


def recognize(train_feats, test_feats, n_templates=5, verbose=False):
    """最近模板判决，返回 ``[(path, true, pred, dist, ok), ...]``。"""
    table = {}
    for p, label, feat in train_feats:
        table.setdefault(label, []).append(feat)
    for label in table:
        table[label] = table[label][:n_templates]

    rows = []
    t0 = time.perf_counter()
    for i, (p, true_label, feat) in enumerate(test_feats, 1):
        best_label, best_d = -1, float("inf")
        for label, seqs in table.items():
            d = min(dtw.distance(feat, s) for s in seqs)
            if d < best_d:
                best_label, best_d = label, d
        rows.append((p, true_label, best_label, best_d, best_label == true_label))
        if verbose and i % 10 == 0:
            acc = np.mean([r[4] for r in rows])
            print(f"  {i}/{len(test_feats)}  累计 acc={acc:.3f}  "
                  f"{time.perf_counter()-t0:.1f}s", flush=True)
    return rows


def accuracy(rows):
    return float(np.mean([r[4] for r in rows])) if rows else 0.0


def confusion(rows, n_labels=10):
    cm = np.zeros((n_labels, n_labels), dtype=int)
    for _, true, pred, _, _ in rows:
        cm[true, pred] += 1
    return cm


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def save_confusion(path, cm):
    _write_csv(path, ["true\\pred"] + [str(i) for i in range(cm.shape[0])],
               [[i] + list(row) for i, row in enumerate(cm)])


def save_predictions(path, rows):
    _write_csv(path, ["file", "true", "pred", "distance", "correct"],
               [[p.name, t, pr, f"{d:.6f}", int(ok)] for p, t, pr, d, ok in rows])


def condition(name, *, use_endpoint=False, normalize="none", n_templates=5, verbose=True):
    """跑一个条件，返回 (acc, rows)。"""
    t0 = time.perf_counter()
    train = featurize("train", use_endpoint=use_endpoint, normalize=normalize)
    test = featurize("test", use_endpoint=use_endpoint, normalize=normalize)
    rows = recognize(train, test, n_templates=n_templates, verbose=verbose)
    acc = accuracy(rows)
    if verbose:
        n_ok = sum(1 for r in rows if r[4])
        print(f"[{name}] acc={acc:.3f} ({n_ok}/{len(rows)})  用时 {time.perf_counter()-t0:.1f}s",
              flush=True)
    return acc, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="实验三 DTW 识别")
    ap.add_argument("--all", action="store_true", help="跑全部消融并写 report/")
    ap.add_argument("--normalize", default="none", choices=["none", "cmn", "cmvn"])
    ap.add_argument("--templates", type=int, default=5, choices=[1, 3, 5])
    ap.add_argument("--endpoint", action="store_true", help="用端点段而不是整段")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    verbose = not args.quiet

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    if not args.all:
        acc, rows = condition("main", use_endpoint=args.endpoint,
                              normalize=args.normalize, n_templates=args.templates,
                              verbose=verbose)
        save_predictions(TABLES_DIR / "exp3_predictions.csv", rows)
        save_confusion(TABLES_DIR / "exp3_confusion.csv", confusion(rows))
        print(f"准确率 {acc:.3f}")
        return 0

    summary = []

    # 主结果：整段 + none + 5 条模板
    acc, rows = condition("main 整段/none/5模板", verbose=verbose)
    summary.append(("main", "整段", "none", 5, acc, len(rows)))
    save_predictions(TABLES_DIR / "exp3_predictions.csv", rows)
    save_confusion(TABLES_DIR / "exp3_confusion.csv", confusion(rows))

    # 模板数 1 / 3
    for n in (1, 3):
        acc_n, _ = condition(f"模板数 {n}", n_templates=n, verbose=verbose)
        summary.append((f"templates_{n}", "整段", "none", n, acc_n, 50))

    # normalize 消融
    acc_cmn, _ = condition("normalize=cmn", normalize="cmn", verbose=verbose)
    summary.append(("normalize_cmn", "整段", "cmn", 5, acc_cmn, 50))

    # 端点段消融
    acc_ep, rows_ep = condition("端点段", use_endpoint=True, verbose=verbose)
    summary.append(("endpoint", "端点段", "none", 5, acc_ep, 50))
    save_predictions(TABLES_DIR / "exp3_predictions_endpoint.csv", rows_ep)
    save_confusion(TABLES_DIR / "exp3_confusion_endpoint.csv", confusion(rows_ep))

    _write_csv(TABLES_DIR / "exp3_accuracy.csv",
               ["条件", "输入", "normalize", "模板数", "准确率", "测试条数"],
               [[n, seg, nz, k, f"{a:.4f}", m] for n, seg, nz, k, a, m in summary])
    for n, seg, nz, k, a, m in summary:
        print(f"  {n:16s} 输入={seg:4s} normalize={nz:4s} 模板={k}  acc={a:.3f}", flush=True)
    print("已写 report/tables/exp3_accuracy.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
