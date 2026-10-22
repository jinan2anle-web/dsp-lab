# -*- coding: utf-8 -*-
"""实验一第二周：数据预处理、时域特征与分类器比较的一键流程。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WAV_DIR = ROOT / "data" / "A"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
PROCESSED = HERE / "processed" / "endpoint"
INTERFACE = ROOT / "接口冻结_1"
sys.path.insert(0, str(INTERFACE))
sys.path.insert(0, str(HERE))

from speech_time_domain import (  # noqa: E402
    DTWClassifier,
    endpoint_detect,
    read_wav,
    remove_dc,
    time_domain_feature,
    write_wav,
)
from utils.plot_style import (  # noqa: E402
    PALETTE,
    add_panel_labels,
    apply_publication_style,
    export_figure,
    publication_subplots,
)

SEED = 42
WINDOWS = ("rect", "hamming", "hanning")
DIGITS = tuple(str(i) for i in range(10))
FILENAME_RE = re.compile(r"^(?P<label>[0-9])_(?P<index>[0-9]{2})\.wav$", re.I)
FEATURE_NAMES = (
    "log_energy_mean", "log_energy_std", "log_energy_max", "log_energy_q25", "log_energy_q75",
    "log_amplitude_mean", "log_amplitude_std", "log_amplitude_max", "log_amplitude_q25", "log_amplitude_q75",
    "zcr_mean", "zcr_std", "zcr_max", "zcr_q25", "zcr_q75", "speech_duration_s",
)


def ensure_dirs() -> None:
    for directory in (RESULTS, FIGURES, PROCESSED):
        directory.mkdir(parents=True, exist_ok=True)


def discover_wavs(smoke: bool = False) -> list[Path]:
    paths = sorted(WAV_DIR.glob("*.wav"))
    parsed = []
    for path in paths:
        match = FILENAME_RE.match(path.name)
        if match:
            parsed.append((path, int(match.group("label")), int(match.group("index"))))
    if smoke:
        parsed = [item for item in parsed if item[2] in {1, 6}]
    expected = 20 if smoke else 100
    if len(parsed) != expected:
        raise RuntimeError(f"期望读取 {expected} 个命名合规 WAV，实际 {len(parsed)} 个")
    if not smoke:
        actual = {(label, index) for _, label, index in parsed}
        required = {(label, index) for label in range(10) for index in range(1, 11)}
        if actual != required:
            raise RuntimeError("WAV 类别或序号不完整，要求每类严格包含 01--10")
    return [item[0] for item in parsed]


def summarize_sequence(seq: np.ndarray, duration_s: float) -> dict[str, float]:
    output: dict[str, float] = {}
    prefixes = ("log_energy", "log_amplitude", "zcr")
    for column, prefix in enumerate(prefixes):
        values = seq[:, column]
        output.update({
            f"{prefix}_mean": float(np.mean(values)),
            f"{prefix}_std": float(np.std(values)),
            f"{prefix}_max": float(np.max(values)),
            f"{prefix}_q25": float(np.quantile(values, 0.25)),
            f"{prefix}_q75": float(np.quantile(values, 0.75)),
        })
    output["speech_duration_s"] = float(duration_s)
    return output


def collect_records(paths: list[Path], write_segments: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    audit_rows, feature_rows = [], []
    endpoint_sequences: dict[str, np.ndarray] = {}
    whole_sequences: dict[str, np.ndarray] = {}
    for path in paths:
        match = FILENAME_RE.match(path.name)
        assert match is not None
        label, index = match.group("label"), int(match.group("index"))
        split = "train" if index <= 5 else "test"
        x, fs, channels, bits = read_wav(path)
        centered = remove_dc(x)
        det = endpoint_detect(x, fs, frame_ms=25.0, hop_ms=10.0, win="hamming")
        if det is None:
            raise RuntimeError(f"端点检测失败: {path.name}")
        peak = float(np.max(np.abs(x)))
        rms = float(np.sqrt(np.mean(np.square(centered))))
        clipped = int(np.sum(np.abs(x) >= 32767))
        duration = len(x) / fs
        speech_duration = det["end"] - det["start"]
        low_signal = peak < 3000
        audit_rows.append({
            "file": path.name, "label": int(label), "index": index, "split": split,
            "sample_rate_hz": fs, "channels": channels, "bits_per_sample": bits,
            "samples": len(x), "duration_s": duration, "dc_mean": float(np.mean(x)),
            "peak_abs": peak, "rms_centered": rms, "clipped_samples": clipped,
            "endpoint_start_s": det["start"], "endpoint_end_s": det["end"],
            "speech_duration_s": speech_duration, "low_signal_review": bool(low_signal),
        })
        if write_segments:
            write_wav(PROCESSED / path.name, det["seg"] / 32768.0, fs=fs, bits=16)
        endpoint_sequences[path.name] = time_domain_feature(det["seg"], fs, win="hamming", normalize=True)
        whole_sequences[path.name] = time_domain_feature(x, fs, win="hamming", normalize=True)
        for window in WINDOWS:
            seq = time_domain_feature(det["seg"], fs, win=window, normalize=False)
            row = {"file": path.name, "label": int(label), "index": index, "split": split, "window": window}
            row.update(summarize_sequence(seq, speech_duration))
            feature_rows.append(row)
    audit = pd.DataFrame(audit_rows).sort_values(["label", "index"]).reset_index(drop=True)
    features = pd.DataFrame(feature_rows).sort_values(["window", "label", "index"]).reset_index(drop=True)
    return audit, features, endpoint_sequences, whole_sequences


def feature_matrix(features: pd.DataFrame, window: str, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    subset = features[(features["window"] == window) & (features["split"] == split)].sort_values(["label", "index"])
    return subset.loc[:, FEATURE_NAMES].to_numpy(float), subset["label"].to_numpy(int), subset["file"].tolist()


def evaluate_fixed_classifier(name: str, estimator, x_train, y_train, x_test, y_test, test_files) -> tuple[dict, pd.DataFrame]:
    t0 = time.perf_counter()
    estimator.fit(x_train, y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = estimator.predict(x_test)
    predict_s = time.perf_counter() - t0
    row = {
        "classifier": name, "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "fit_time_s": fit_s, "predict_time_s": predict_s,
        "correct": int(np.sum(pred == y_test)), "test_count": len(y_test),
    }
    predictions = pd.DataFrame({"file": test_files, "true": y_test, "pred": pred, "classifier": name})
    return row, predictions


def tune_and_compare(features: pd.DataFrame, endpoint_sequences: dict[str, np.ndarray], whole_sequences: dict[str, np.ndarray], smoke: bool):
    x_train, y_train, train_files = feature_matrix(features, "hamming", "train")
    x_test, y_test, test_files = feature_matrix(features, "hamming", "test")
    if smoke:
        knn = Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=1))])
        svm = Pipeline([("scale", StandardScaler()), ("model", SVC(C=1.0, kernel="linear"))])
        knn_cv = pd.DataFrame([{"mode": "smoke", "params": "n_neighbors=1"}])
        svm_cv = pd.DataFrame([{"mode": "smoke", "params": "C=1,kernel=linear"}])
        estimators = [("KNN", knn), ("SVM", svm)]
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        knn_pipe = Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier())])
        knn = GridSearchCV(
            knn_pipe,
            {"model__n_neighbors": [1, 3, 5, 7, 9], "model__weights": ["uniform", "distance"]},
            scoring="accuracy", cv=cv, n_jobs=1, return_train_score=True,
        ).fit(x_train, y_train)
        svm_pipe = Pipeline([("scale", StandardScaler()), ("model", SVC())])
        svm = GridSearchCV(
            svm_pipe,
            [
                {"model__kernel": ["linear"], "model__C": [0.1, 1, 10, 100]},
                {"model__kernel": ["rbf"], "model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.01, 0.1, 1]},
            ],
            scoring="accuracy", cv=cv, n_jobs=1, return_train_score=True,
        ).fit(x_train, y_train)
        knn_cv = pd.DataFrame(knn.cv_results_)
        svm_cv = pd.DataFrame(svm.cv_results_)
        estimators = [("KNN", knn.best_estimator_), ("SVM", svm.best_estimator_)]

    metrics, all_predictions = [], []
    for name, estimator in estimators:
        row, pred = evaluate_fixed_classifier(name, estimator, x_train, y_train, x_test, y_test, test_files)
        if not smoke:
            search = knn if name == "KNN" else svm
            row["train_cv_best_accuracy"] = float(search.best_score_)
            row["selected_params"] = json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True)
        metrics.append(row)
        all_predictions.append(pred)

    dtw_train_endpoint = [endpoint_sequences[name] for name in train_files]
    dtw_test_endpoint = [endpoint_sequences[name] for name in test_files]
    dtw_train_whole = [whole_sequences[name] for name in train_files]
    dtw_test_whole = [whole_sequences[name] for name in test_files]
    for name, train_seq, test_seq in (
        ("DTW_endpoint", dtw_train_endpoint, dtw_test_endpoint),
        ("DTW_whole_baseline", dtw_train_whole, dtw_test_whole),
    ):
        t0 = time.perf_counter()
        classifier = DTWClassifier(band_ratio=0.35).fit(train_seq, y_train)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = classifier.predict(test_seq)
        predict_s = time.perf_counter() - t0
        metrics.append({
            "classifier": name, "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro"), "fit_time_s": fit_s,
            "predict_time_s": predict_s, "correct": int(np.sum(pred == y_test)),
            "test_count": len(y_test), "train_cv_best_accuracy": np.nan,
            "selected_params": json.dumps({"band_ratio": 0.35}),
        })
        all_predictions.append(pd.DataFrame({"file": test_files, "true": y_test, "pred": pred, "classifier": name}))
    return pd.DataFrame(metrics), pd.concat(all_predictions, ignore_index=True), knn_cv, svm_cv


def compare_windows(features: pd.DataFrame, smoke: bool) -> pd.DataFrame:
    rows = []
    for window in WINDOWS:
        x_train, y_train, _ = feature_matrix(features, window, "train")
        if smoke:
            score_mean, score_std = np.nan, np.nan
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            model = Pipeline([("scale", StandardScaler()), ("model", SVC(C=1.0, kernel="linear"))])
            scores = cross_val_score(model, x_train, y_train, scoring="accuracy", cv=cv, n_jobs=1)
            score_mean, score_std = float(scores.mean()), float(scores.std(ddof=1))
        rows.append({"window": window, "train_cv_accuracy_mean": score_mean, "train_cv_accuracy_sd": score_std})
    return pd.DataFrame(rows)


def save_tables(audit, features, metrics, predictions, knn_cv, svm_cv, windows) -> None:
    audit.to_csv(RESULTS / "数据质检与端点.csv", index=False, encoding="utf-8-sig")
    features.to_csv(RESULTS / "时域特征汇总.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(RESULTS / "分类器比较.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULTS / "测试集逐条预测.csv", index=False, encoding="utf-8-sig")
    windows.to_csv(RESULTS / "窗函数训练集比较.csv", index=False, encoding="utf-8-sig")
    knn_cv.to_csv(RESULTS / "KNN训练集交叉验证.csv", index=False, encoding="utf-8-sig")
    svm_cv.to_csv(RESULTS / "SVM训练集交叉验证.csv", index=False, encoding="utf-8-sig")


def representative_data(audit: pd.DataFrame, features: pd.DataFrame, filename: str, window: str = "hamming"):
    row = audit[audit["file"] == filename].iloc[0]
    x, fs, _, _ = read_wav(WAV_DIR / filename)
    det = endpoint_detect(x, fs, win=window)
    assert det is not None
    seq = time_domain_feature(det["seg"], fs, win=window, normalize=False)
    return row, x, fs, det, seq


def plot_all(audit, features, metrics, predictions, knn_cv, svm_cv) -> None:
    apply_publication_style(language="zh", width="report")
    color = PALETTE

    # raw_q1: 每类首条录音的原始波形，核对输入信号与录音时长。
    fig, axes = publication_subplots(5, 2, width="report", aspect=1.18, squeeze=False)
    for digit, ax in zip(DIGITS, axes.ravel()):
        x, fs, _, _ = read_wav(WAV_DIR / f"{digit}_01.wav")
        t = np.arange(len(x)) / fs
        ax.plot(t, x, color=color["primary"], linewidth=0.65)
        ax.set(title=f"数字 {digit}", xlim=(0, t[-1]))
        if digit in {"8", "9"}:
            ax.set_xlabel("时间（s）")
        if digit in {"0", "2", "4", "6", "8"}:
            ax.set_ylabel("幅值")
    add_panel_labels(axes.ravel())
    export_figure(fig, FIGURES / "raw_q1_digit_waveforms")
    plt.close(fig)

    # raw_q2: 单条真实录音的三维时域特征原始轨迹。
    _, _, _, det, seq = representative_data(audit, features, "0_01.wav")
    t = np.arange(len(seq)) * 0.01
    labels = (("对数能量", "ln(E)"), ("对数幅度和", "ln(M)"), ("过零率", "ZCR"))
    fig, axes = publication_subplots(3, 1, width="report", aspect=0.95, squeeze=False)
    for i, (title, ylabel) in enumerate(labels):
        ax = axes[i, 0]
        ax.plot(t, seq[:, i], color=(color["primary"], color["contrast"], color["positive"])[i])
        ax.set(title=title, ylabel=ylabel)
    axes[-1, 0].set_xlabel("语音段时间（s）")
    add_panel_labels(axes.ravel())
    export_figure(fig, FIGURES / "raw_q2_feature_traces")
    plt.close(fig)

    # raw_q3: 固定长度特征的二维 PCA，仅用于观察可分性，不用于训练。
    ham = features[features["window"] == "hamming"].sort_values(["label", "index"])
    z = StandardScaler().fit_transform(ham.loc[:, FEATURE_NAMES])
    pca = PCA(n_components=2).fit(z)
    xy = pca.transform(z)
    fig, ax = publication_subplots(1, 1, width="report", aspect=0.68)
    marker_map = {"train": "o", "test": "^"}
    for split, marker in marker_map.items():
        mask = ham["split"].to_numpy() == split
        ax.scatter(xy[mask, 0], xy[mask, 1], s=14, marker=marker, facecolors="none",
                   edgecolors=color["neutral"] if split == "train" else color["contrast"], label=split)
    for digit in range(10):
        mask = ham["label"].to_numpy() == digit
        center = xy[mask].mean(axis=0)
        ax.text(center[0], center[1], str(digit), color=color["primary"], fontweight="bold", ha="center", va="center")
    ax.set(title="固定长度特征分布", xlabel=f"PC1（{pca.explained_variance_ratio_[0]*100:.1f}%）",
           ylabel=f"PC2（{pca.explained_variance_ratio_[1]*100:.1f}%）")
    ax.legend()
    export_figure(fig, FIGURES / "raw_q3_feature_pca")
    plt.close(fig)

    # process_q1: 端点检测的波形覆盖与双能量门限证据。
    _, x, fs, det, _ = representative_data(audit, features, "4_10.wav")
    tx = np.arange(len(x)) / fs
    tf = np.arange(len(det["E"])) * 0.01
    fig, axes = publication_subplots(2, 1, width="report", aspect=0.8, height_ratios=[1.1, 1])
    axes[0].plot(tx, x, color=color["neutral"], linewidth=0.65)
    axes[0].axvspan(det["start"], det["end"], color=color["primary"], alpha=0.18)
    axes[0].set(title="低幅值样本 4_10", ylabel="幅值")
    axes[1].plot(tf, det["E"], color=color["primary"], label="短时能量")
    axes[1].axhline(det["T1"], color=color["secondary"], linestyle="--", label="T1")
    axes[1].axhline(det["T2"], color=color["contrast"], linestyle=":", label="T2")
    axes[1].axvspan(det["start"], det["end"], color=color["primary"], alpha=0.12)
    axes[1].set(title="能量门限与端点", xlabel="时间（s）", ylabel="能量")
    axes[1].legend(ncols=3)
    add_panel_labels(axes)
    export_figure(fig, FIGURES / "process_q1_endpoint_detection")
    plt.close(fig)

    # process_q2: 同一语音段使用三种窗的特征曲线。
    _, x, fs, det, _ = representative_data(audit, features, "0_01.wav")
    fig, axes = publication_subplots(3, 1, width="report", aspect=0.95, squeeze=False)
    line_styles = {"rect": "-", "hamming": "--", "hanning": ":"}
    for window in WINDOWS:
        seq_w = time_domain_feature(det["seg"], fs, win=window, normalize=False)
        tw = np.arange(len(seq_w)) * 0.01
        for col in range(3):
            axes[col, 0].plot(tw, seq_w[:, col], linestyle=line_styles[window], label=window)
    for col, (title, ylabel) in enumerate(labels):
        axes[col, 0].set(title=title, ylabel=ylabel)
    axes[0, 0].legend(ncols=3)
    axes[-1, 0].set_xlabel("语音段时间（s）")
    add_panel_labels(axes.ravel())
    export_figure(fig, FIGURES / "process_q2_window_comparison")
    plt.close(fig)

    # process_q3: 所有超参数只在训练集五折交叉验证中选择。
    kdf = knn_cv.copy()
    sdf = svm_cv.copy()
    fig, axes = publication_subplots(1, 2, width="report", aspect=0.52, width_ratios=[1, 1.15])
    for weight, linestyle in (("uniform", "-"), ("distance", "--")):
        part = kdf[kdf["param_model__weights"] == weight].sort_values("param_model__n_neighbors")
        axes[0].plot(part["param_model__n_neighbors"], part["mean_test_score"], marker="o", linestyle=linestyle, label=weight)
    axes[0].set(title="KNN 训练集验证", xlabel="邻居数 k", ylabel="五折准确率", ylim=(0, 1.02))
    axes[0].legend()
    sdf["kernel"] = sdf["param_model__kernel"].astype(str)
    for kernel, linestyle in (("linear", "-"), ("rbf", "--")):
        part = sdf[sdf["kernel"] == kernel].groupby("param_model__C", as_index=False)["mean_test_score"].max()
        axes[1].plot(part["param_model__C"], part["mean_test_score"], marker="s", linestyle=linestyle, label=kernel)
    axes[1].set_xscale("log")
    axes[1].set(title="SVM 训练集验证", xlabel="惩罚参数 C", ylabel="五折准确率", ylim=(0, 1.02))
    axes[1].legend()
    add_panel_labels(axes)
    export_figure(fig, FIGURES / "process_q3_train_only_cv")
    plt.close(fig)

    # result_q1: 采集质量和端点结果的全量分布。
    fig, axes = publication_subplots(1, 3, width="report", aspect=0.43)
    axes[0].hist(audit["peak_abs"], bins=12, color=color["primary"], edgecolor="white")
    axes[0].axvline(3000, color=color["contrast"], linestyle="--")
    axes[0].set(title="峰值幅度", xlabel="绝对峰值", ylabel="文件数")
    axes[1].hist(audit["speech_duration_s"], bins=12, color=color["positive"], edgecolor="white")
    axes[1].set(title="语音段时长", xlabel="时长（s）", ylabel="文件数")
    axes[2].scatter(audit["label"], audit["dc_mean"], s=13, facecolors="none", edgecolors=color["neutral"])
    axes[2].axhline(0, color=color["dark"], linewidth=0.7)
    axes[2].set(title="直流偏置", xlabel="数字类别", ylabel="均值")
    add_panel_labels(axes)
    export_figure(fig, FIGURES / "result_q1_quality_summary")
    plt.close(fig)

    # result_q2: Hamming 窗特征按类别的均值与样本标准差（n=10）。
    grouped = ham.groupby("label")
    selected = (("speech_duration_s", "语音段时长", "s"), ("log_energy_mean", "平均对数能量", "ln(E)"), ("zcr_mean", "平均过零率", "ZCR"))
    fig, axes = publication_subplots(1, 3, width="report", aspect=0.44)
    for ax, (column, title, ylabel) in zip(axes, selected):
        mean = grouped[column].mean()
        std = grouped[column].std(ddof=1)
        ax.errorbar(mean.index, mean, yerr=std, marker="o", capsize=2.5, color=color["primary"], linestyle="-")
        ax.set(title=title, xlabel="数字类别", ylabel=ylabel, xticks=range(10))
    add_panel_labels(axes)
    export_figure(fig, FIGURES / "result_q2_class_feature_summary")
    plt.close(fig)

    # result_q3: 独立测试集准确率、最佳分类器混淆矩阵与预测耗时。
    ordered = metrics.sort_values("accuracy", ascending=False).reset_index(drop=True)
    best = ordered.iloc[0]["classifier"]
    best_pred = predictions[predictions["classifier"] == best]
    conf = confusion_matrix(best_pred["true"], best_pred["pred"], labels=range(10))
    fig, axes = publication_subplots(1, 3, width="double", aspect=0.42, width_ratios=[1, 1.2, 0.85])
    names = ordered["classifier"].str.replace("_", "\n", regex=False)
    axes[0].bar(np.arange(len(ordered)), ordered["accuracy"], color=[color["primary"], color["positive"], color["secondary"], color["neutral"]])
    axes[0].set(title="测试集准确率", ylabel="准确率", xticks=np.arange(len(ordered)), xticklabels=names, ylim=(0, 1.02))
    image = axes[1].imshow(conf, cmap="Blues", vmin=0, vmax=max(1, conf.max()))
    for i in range(10):
        for j in range(10):
            axes[1].text(j, i, str(conf[i, j]), ha="center", va="center", fontsize=6,
                         color="white" if conf[i, j] > conf.max() / 2 else color["dark"])
    axes[1].set(title=f"最佳：{best}", xlabel="预测类别", ylabel="真实类别", xticks=range(10), yticks=range(10))
    fig.colorbar(image, ax=axes[1], label="样本数")
    axes[2].plot(ordered["predict_time_s"], np.arange(len(ordered)), marker="o", linestyle="none", color=color["contrast"])
    axes[2].set_xscale("log")
    axes[2].set(title="预测耗时", xlabel="秒（对数轴）", yticks=np.arange(len(ordered)), yticklabels=names)
    add_panel_labels(axes)
    export_figure(fig, FIGURES / "result_q3_classifier_comparison")
    plt.close(fig)


def write_contracts_and_report(audit, features, metrics, windows) -> None:
    contracts = [
        {"figure": "raw_q1_digit_waveforms", "conclusion": "100 条录音均为固定 1.5 s，十类真实波形形态可核查", "source": "data/A/*_01.wav", "n": 10},
        {"figure": "raw_q2_feature_traces", "conclusion": "能量、幅度与过零率提供互补的逐帧描述", "source": "data/A/0_01.wav", "n": 1},
        {"figure": "raw_q3_feature_pca", "conclusion": "固定长度时域特征存在类间重叠，分类难度不可忽略", "source": "results/时域特征汇总.csv", "n": len(audit)},
        {"figure": "process_q1_endpoint_detection", "conclusion": "双能量门限仍能定位低幅值样本 4_10 的有效段", "source": "data/A/4_10.wav", "n": 1},
        {"figure": "process_q2_window_comparison", "conclusion": "三种窗改变能量尺度，但主要时序轮廓一致", "source": "data/A/0_01.wav", "n": 1},
        {"figure": "process_q3_train_only_cv", "conclusion": "KNN 与 SVM 参数仅由 50 条训练样本五折验证选择", "source": "results/*训练集交叉验证.csv", "n": 50},
        {"figure": "result_q1_quality_summary", "conclusion": "全量格式一致且无削波，低峰值样本需人工复听", "source": "results/数据质检与端点.csv", "n": len(audit)},
        {"figure": "result_q2_class_feature_summary", "conclusion": "不同数字在时长、能量与过零率上存在分布差异", "source": "results/时域特征汇总.csv", "n": len(audit)},
        {"figure": "result_q3_classifier_comparison", "conclusion": "四条流程在同一独立测试集上比较准确率与耗时", "source": "results/分类器比较.csv", "n": int(metrics["test_count"].iloc[0])},
    ]
    (RESULTS / "图表契约.json").write_text(json.dumps(contracts, ensure_ascii=False, indent=2), encoding="utf-8")
    best = metrics.sort_values(["accuracy", "macro_f1"], ascending=False).iloc[0]
    dtw_rows = metrics.loc[metrics["classifier"] == "DTW_endpoint", "accuracy"]
    dtw_endpoint_acc = float(dtw_rows.iloc[0]) if len(dtw_rows) else float("nan")
    low = audit[audit["low_signal_review"]]["file"].tolist()
    window_lines = "\n".join(
        f"- {r.window}: 训练集五折准确率 {r.train_cv_accuracy_mean:.3f} ± {r.train_cv_accuracy_sd:.3f}"
        for r in windows.itertuples()
    )
    metric_lines = "\n".join(
        f"- {r.classifier}: {int(r.correct)}/{int(r.test_count)}，准确率 {r.accuracy:.3f}，宏平均 F1 {r.macro_f1:.3f}，预测 {r.predict_time_s:.6f} s"
        for r in metrics.sort_values("accuracy", ascending=False).itertuples()
    )
    report = f"""# 实验一第二周结果摘要

## 1. 数据采集与预处理

- 使用现有 `data/A` 的 100 条真实录音：数字 0--9 每类 10 条；未生成或冒充新录音。
- 格式审核：{len(audit)}/100 为 16 kHz、单声道、16 bit PCM、1.5 s；削波样点总数 {int(audit['clipped_samples'].sum())}。
- 去直流后执行 25 ms 帧长、10 ms 帧移、Hamming 窗的双门限端点检测；{len(audit)}/{len(audit)} 条均返回非空候选段。由于没有人工端点标注，这不等同于端点精度达到 100%。
- 峰值小于 3000 的人工复听候选：{', '.join(low) if low else '无'}。这只是信号幅度提示，不等同于错标或无效样本。

## 2. 时域特征

- 逐帧特征为对数短时能量 `ln(E)`、对数短时幅度和 `ln(M)`、归一化短时过零率 `ZCR`。
- 分类器的定长输入由每维均值、标准差、最大值、25%/75% 分位数及语音段时长组成，共 {len(FEATURE_NAMES)} 维。
- Rectangular、Hamming、Hann 三种窗只在训练集上作比较，避免用测试集选方案：
{window_lines}
- PCA 图使用全部 100 条样本，仅用于观察特征空间的类间重叠，不参与训练、调参或性能估计。

## 3. 分类器比较

固定划分为每类 01--05 训练、06--10 测试，即 50/50。KNN 与 SVM 的标准化和参数选择全部封装在训练集五折交叉验证内；DTW 保留固定 `band_ratio=0.35` 的最近邻基线。预测耗时只统计已完成特征提取后的 `predict` 阶段，不是端到端性能基准。

{metric_lines}

本次最佳为 **{best['classifier']}**，测试集 {int(best['correct'])}/{int(best['test_count'])}，准确率 {best['accuracy']:.3f}。整段 DTW 保留了完整时间顺序以及录音首尾背景，而 KNN/SVM 使用的 16 维汇总会丢失帧序列顺序，这与当前结果一致，但还不能证明是唯一原因。端点后 DTW 降至 {dtw_endpoint_acc:.3f}，说明当前自动裁剪、序列归一化和 DTW 组合没有带来收益；它不表示端点处理在一般情况下无效，仍需人工复听端点候选并做参数敏感性分析。

样本来自同一批次且说话人独立性未经元数据证实；测试集也只有 50 条，并被四条流程共同使用。因此这些数值是当前固定划分下的描述性比较，不能外推为跨说话人性能，也不作显著性结论或按测试集反向调参。
"""
    (RESULTS / "结果摘要.md").write_text(report, encoding="utf-8")


def write_run_metadata(smoke: bool, elapsed_s: float) -> None:
    metadata = {
        "seed": SEED,
        "mode": "smoke" if smoke else "full",
        "input_directory": str(WAV_DIR),
        "split": "每类 01-05 训练，06-10 测试",
        "frame_ms": 25.0, "hop_ms": 10.0,
        "endpoint_window": "hamming", "compared_windows": list(WINDOWS),
        "elapsed_s": elapsed_s,
    }
    (RESULTS / "运行参数.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_repro_manifest(paths: list[Path]) -> None:
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn")
    dependencies = {}
    for package in packages:
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": SEED,
        "input_files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in paths
        ],
        "runtime": {
            "name": "python", "version": sys.version.split()[0],
            "platform": platform.platform(), "dependencies": dependencies,
        },
        "key_parameters": {
            "split": "每类 01-05 训练，06-10 测试",
            "frame_ms": 25.0, "hop_ms": 10.0,
            "endpoint_window": "hamming", "windows": list(WINDOWS),
            "dtw_band_ratio": 0.35, "training_cv": "StratifiedKFold(5, shuffle=True, random_state=42)",
        },
        "reproduce_command": "python run_week2.py --full",
    }
    (RESULTS / "复现清单.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="每类使用 1 条训练和 1 条测试，仅验证纵向闭环")
    mode.add_argument("--full", action="store_true", help="运行全部 100 条、交叉验证与正式图表")
    args = parser.parse_args()
    smoke = args.smoke and not args.full
    ensure_dirs()
    np.random.seed(SEED)
    start = time.perf_counter()
    paths = discover_wavs(smoke=smoke)
    audit, features, endpoint_sequences, whole_sequences = collect_records(paths, write_segments=not smoke)
    metrics, predictions, knn_cv, svm_cv = tune_and_compare(features, endpoint_sequences, whole_sequences, smoke)
    windows = compare_windows(features, smoke)
    save_tables(audit, features, metrics, predictions, knn_cv, svm_cv, windows)
    if not smoke:
        plot_all(audit, features, metrics, predictions, knn_cv, svm_cv)
        write_contracts_and_report(audit, features, metrics, windows)
    elapsed = time.perf_counter() - start
    write_run_metadata(smoke, elapsed)
    if not smoke:
        write_repro_manifest(paths)
    print(metrics[["classifier", "correct", "test_count", "accuracy", "macro_f1", "predict_time_s"]].to_string(index=False))
    print(f"完成：{'smoke' if smoke else 'full'}，耗时 {elapsed:.2f} s，结果目录 {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
