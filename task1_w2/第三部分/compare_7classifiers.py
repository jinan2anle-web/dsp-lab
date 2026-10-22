# -*- coding: utf-8 -*-
r"""第三部分：七类分类器在同一固定划分上的比较。

交接版 —— 放在 task1_w2\第三部分\ 下直接运行，结果就地写回本目录。
划分与 run_week2.py 完全一致：每类 01--05 训练、06--10 测试；标准化与选参只在
训练集五折内进行（StratifiedKFold(5, shuffle=True, random_state=42)），测试集只评一次。
DTW 一支沿用冻结件 DTWClassifier(band_ratio=0.35)，输入为整段波形的 [T,3] 逐帧特征。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent          # task1_w2\第三部分
W2 = HERE.parent                                # task1_w2
PART1 = W2 / "第一部分" / "第一部分"
RESULTS = PART1 / "results"
WAV_DIR = PART1 / "A"
INTERFACE = W2 / "第一部分" / "接口冻结_1"
OUT = HERE

sys.path.insert(0, str(INTERFACE))
sys.path.insert(0, str(PART1))

from speech_time_domain import DTWClassifier, read_wav, time_domain_feature  # noqa: E402
from utils.plot_style import (  # noqa: E402
    PALETTE,
    apply_publication_style,
    export_figure,
    publication_subplots,
)

color = PALETTE
apply_publication_style(language="zh", width="double")

SEED = 42
FRAME_MS = 25.0
HOP_MS = 10.0
FEAT_WIN = "hamming"
BAND_RATIO = 0.35


FEATURE_COLUMNS = [
    "log_energy_mean", "log_energy_std", "log_energy_max", "log_energy_q25", "log_energy_q75",
    "log_amplitude_mean", "log_amplitude_std", "log_amplitude_max", "log_amplitude_q25", "log_amplitude_q75",
    "zcr_mean", "zcr_std", "zcr_max", "zcr_q25", "zcr_q75", "speech_duration_s",
]


def load_feature_table() -> pd.DataFrame:
    """取 hamming 窗那一份（三种窗各 100 行，共 300 行；分类主流程用的是 hamming）。"""
    table = pd.read_csv(RESULTS / "时域特征汇总.csv")
    table = table[table["window"] == FEAT_WIN].reset_index(drop=True)
    assert len(table) == 100, f"预期 100 条，实得 {len(table)}"
    return table


def make_models() -> list[tuple[str, object, dict]]:
    cv_pipe = lambda est, grid: (Pipeline([("scaler", StandardScaler()), ("model", est)]),
                                 {f"model__{k}": v for k, v in grid.items()})
    knn = cv_pipe(KNeighborsClassifier(), {"n_neighbors": [1, 3, 5, 7, 9],
                                           "weights": ["uniform", "distance"]})
    svm = cv_pipe(SVC(), {"C": [0.1, 1.0, 10.0, 100.0], "kernel": ["linear", "rbf"],
                          "gamma": ["scale", 0.01, 0.1, 1.0]})
    lr = cv_pipe(LogisticRegression(max_iter=2000, random_state=SEED), {"C": [0.01, 0.1, 1.0, 10.0, 100.0]})
    nb = cv_pipe(GaussianNB(), {"var_smoothing": [1e-9, 1e-8, 1e-7]})
    dt = ("决策树", DecisionTreeClassifier(random_state=SEED),
          {"max_depth": [None, 3, 5, 7], "min_samples_leaf": [1, 2]})
    rf = ("随机森林", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
          {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 2]})
    return [("KNN", *knn), ("SVM", *svm), ("逻辑回归", *lr), ("高斯朴素贝叶斯", *nb), dt, rf]


def whole_sequences() -> dict[str, np.ndarray]:
    seqs = {}
    for path in sorted(WAV_DIR.glob("*.wav")):
        x, fs, _, _ = read_wav(path)
        seqs[path.name] = time_domain_feature(x, fs, frame_ms=FRAME_MS, hop_ms=HOP_MS,
                                              win=FEAT_WIN, normalize=True)
    return seqs


def main() -> None:
    table = load_feature_table()
    names = FEATURE_COLUMNS
    train = table[table["index"] <= 5]
    test = table[table["index"] > 5]
    X_train, y_train = train[names].to_numpy(), train["label"].to_numpy()
    X_test, y_test = test[names].to_numpy(), test["label"].to_numpy()
    print(f"训练 {len(train)} 条 / 测试 {len(test)} 条；特征 {len(names)} 维")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, predictions = [], []
    for name, est, grid in make_models():
        t0 = time.perf_counter()
        search = GridSearchCV(est, grid, scoring="accuracy", cv=cv, n_jobs=1)
        search.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = search.predict(X_test)
        predict_s = time.perf_counter() - t0
        rows.append({"classifier": name, "correct": int((pred == y_test).sum()),
                     "test_count": int(len(y_test)), "accuracy": accuracy_score(y_test, pred),
                     "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
                     "train_cv_best": float(search.best_score_), "fit_time_s": fit_s,
                     "predict_time_s": predict_s,
                     "selected_params": str(search.best_params_)})
        predictions.append(pd.DataFrame({"file": test["file"].to_numpy(), "true": y_test,
                                         "pred": pred, "classifier": name}))
        print(f"  {name}: {rows[-1]['correct']}/{len(y_test)} = {rows[-1]['accuracy']:.3f}")

    seqs = whole_sequences()
    tr = list(train["file"])
    te = list(test["file"])
    t0 = time.perf_counter()
    dtw = DTWClassifier(band_ratio=BAND_RATIO).fit([seqs[f] for f in tr], y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = dtw.predict([seqs[f] for f in te])
    predict_s = time.perf_counter() - t0
    rows.append({"classifier": "DTW 最近邻（整段）", "correct": int((pred == y_test).sum()),
                 "test_count": int(len(y_test)), "accuracy": accuracy_score(y_test, pred),
                 "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
                 "train_cv_best": float("nan"), "fit_time_s": fit_s, "predict_time_s": predict_s,
                 "selected_params": f"{{'band_ratio': {BAND_RATIO}}}"})
    predictions.append(pd.DataFrame({"file": te, "true": y_test, "pred": pred,
                                     "classifier": "DTW 最近邻（整段）"}))
    print(f"  DTW 最近邻（整段）: {rows[-1]['correct']}/{len(y_test)} = {rows[-1]['accuracy']:.3f}")

    metrics = pd.DataFrame(rows).sort_values(["accuracy", "macro_f1"], ascending=False)
    metrics.to_csv(OUT / "7类分类器比较.csv", index=False, encoding="utf-8-sig")
    pd.concat(predictions).to_csv(OUT / "7类分类器逐条预测.csv", index=False, encoding="utf-8-sig")

    best = metrics.iloc[0]["classifier"]
    best_pred = pd.concat(predictions)
    best_pred = best_pred[best_pred["classifier"] == best]
    conf = confusion_matrix(best_pred["true"], best_pred["pred"], labels=range(10))
    pd.DataFrame(conf, index=[f"真{d}" for d in range(10)],
                 columns=[f"预{d}" for d in range(10)]).to_csv(
        OUT / "7类分类器最佳混淆矩阵.csv", encoding="utf-8-sig")

    table_md = ["| 排名 | 分类器 | 正确/总数 | 准确率 | macro-F1 | 训练集五折 | 拟合 s | 预测 s | 选定参数 |",
                "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(metrics.itertuples(), 1):
        cv_txt = "—" if pd.isna(r.train_cv_best) else f"{r.train_cv_best:.3f}"
        table_md.append(f"| {i} | {r.classifier} | {int(r.correct)}/{int(r.test_count)} | "
                        f"{r.accuracy:.3f} | {r.macro_f1:.3f} | {cv_txt} | "
                        f"{r.fit_time_s:.4f} | {r.predict_time_s:.4f} | `{r.selected_params}` |")
    per_class = []
    for d in range(10):
        total = int((best_pred["true"] == d).sum())
        hit = int(((best_pred["true"] == d) & (best_pred["pred"] == d)).sum())
        per_class.append(f"| {d} | {hit}/{total} | {hit / total:.2f} |" if total else f"| {d} | 0/0 | — |")

    md = f"""# 七类分类器比较（真实运行数字）

- 划分：每类 01--05 训练（{len(train)} 条）、06--10 测试（{len(test)} 条），与 `run_week2.py` 完全一致。
- 特征：`results\\时域特征汇总.csv` 的 16 维汇总（逐维 mean/std/max/q25/q75 + 语音段时长）。
- 选参：仅在训练集上做 `StratifiedKFold(5, shuffle=True, random_state=42)` + `GridSearchCV`；标准化封装在同一 Pipeline 内，避免测试集信息泄漏。
- DTW 一支输入为整段波形的 `[T,3]` 逐帧特征（`normalize=True`），`band_ratio=0.35` 固定，无参数搜索。
- 预测耗时只统计 `predict` 阶段，不含特征提取。

## 结果

{chr(10).join(table_md)}

## 最佳分类器 {best} 的逐类召回

| 数字 | 正确/总数 | 召回率 |
|---|---|---|
{chr(10).join(per_class)}

## 结论口径

1. 数值是**同一固定划分下的描述性比较**，测试集只有 {len(test)} 条且被七个分类器共同使用，不作显著性结论，也不据此反向调参。
2. 样本来自同一批次、说话人独立性未经元数据证实 → 存在同说话人跨训练/测试的泄漏风险，准确率偏乐观。
3. DTW 只看形状不看绝对尺度，16 维汇总量纲差异大但不含帧序；两类方法的差异应这样解释，而不是"谁更先进"。

生成脚本：本目录 `compare_7classifiers.py`
"""
    (OUT / "7类分类器比较.md").write_text(md, encoding="utf-8")

    fig, axes = publication_subplots(1, 2, width="double", aspect=0.46, width_ratios=[1.25, 1])
    order = metrics.iloc[::-1]
    axes[0].barh(np.arange(len(order)), order["accuracy"],
                 color=[color["primary"], color["positive"], color["secondary"],
                        color["neutral"], color["contrast"], color["dark"], color["primary"]])
    axes[0].set(xlabel="测试集准确率", xlim=(0, 1.02), yticks=np.arange(len(order)),
                yticklabels=order["classifier"])
    for i, v in enumerate(order["accuracy"]):
        axes[0].text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=7)
    axes[0].set_title("七类分类器准确率")
    image = axes[1].imshow(conf, cmap="Blues", vmin=0, vmax=max(1, conf.max()))
    for i in range(10):
        for j in range(10):
            axes[1].text(j, i, str(conf[i, j]), ha="center", va="center", fontsize=6,
                         color="white" if conf[i, j] > conf.max() / 2 else color["dark"])
    axes[1].set(title="混淆矩阵（最佳）", xlabel="预测类别", ylabel="真实类别",
                xticks=range(10), yticks=range(10))
    fig.colorbar(image, ax=axes[1], label="样本数")
    export_figure(fig, OUT / "figures" / "q3_7class_comparison")
    plt.close(fig)
    print(f"\n最佳：{best} {int(metrics.iloc[0]['correct'])}/{len(y_test)} = {metrics.iloc[0]['accuracy']:.3f}")


if __name__ == "__main__":
    main()
