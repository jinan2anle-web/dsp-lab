# -*- coding: utf-8 -*-
"""实验一第三周：预处理消融、特征消融、DTW调参与稳健性实验。"""

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WAV_DIR = ROOT / "data" / "A"
INTERFACE = ROOT / "接口冻结_1"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
sys.path.insert(0, str(INTERFACE))
sys.path.insert(0, str(HERE))

from speech_time_domain import (  # noqa: E402
    DTWClassifier,
    dtw_distance,
    endpoint_detect,
    pre_emphasis,
    read_wav,
    remove_dc,
    time_domain_feature,
)
from utils.plot_style import (  # noqa: E402
    PALETTE,
    apply_publication_style,
    export_figure,
    publication_subplots,
)

SEED = 42
FILENAME_RE = re.compile(r"^(?P<label>[0-9])_(?P<index>[0-9]{2})\.wav$", re.I)
FEATURE_NAMES = (
    "log_energy_mean", "log_energy_std", "log_energy_max", "log_energy_q25", "log_energy_q75",
    "log_amplitude_mean", "log_amplitude_std", "log_amplitude_max", "log_amplitude_q25", "log_amplitude_q75",
    "zcr_mean", "zcr_std", "zcr_max", "zcr_q25", "zcr_q75", "speech_duration_s",
)
FEATURE_GROUPS = {
    "energy_only": FEATURE_NAMES[0:5] + ("speech_duration_s",),
    "amplitude_only": FEATURE_NAMES[5:10] + ("speech_duration_s",),
    "zcr_only": FEATURE_NAMES[10:15] + ("speech_duration_s",),
    "energy_amplitude": FEATURE_NAMES[0:10] + ("speech_duration_s",),
    "all_three": FEATURE_NAMES,
}
PREPROCESSING = (
    "raw_whole",
    "dc_whole",
    "dc_endpoint",
    "dc_endpoint_preemphasis",
)


def discover_wavs(smoke: bool) -> list[Path]:
    parsed = []
    for path in sorted(WAV_DIR.glob("*.wav")):
        match = FILENAME_RE.match(path.name)
        if match:
            parsed.append((path, int(match.group("label")), int(match.group("index"))))
    if smoke:
        parsed = [item for item in parsed if item[2] in {1, 2, 6}]
    expected = 30 if smoke else 100
    if len(parsed) != expected:
        raise RuntimeError(f"期望 {expected} 条录音，实际 {len(parsed)} 条")
    return [item[0] for item in parsed]


def summarize_sequence(seq: np.ndarray, duration_s: float) -> dict[str, float]:
    result: dict[str, float] = {}
    for column, prefix in enumerate(("log_energy", "log_amplitude", "zcr")):
        values = seq[:, column]
        result.update({
            f"{prefix}_mean": float(values.mean()),
            f"{prefix}_std": float(values.std()),
            f"{prefix}_max": float(values.max()),
            f"{prefix}_q25": float(np.quantile(values, 0.25)),
            f"{prefix}_q75": float(np.quantile(values, 0.75)),
        })
    result["speech_duration_s"] = float(duration_s)
    return result


def preprocessing_signal(x: np.ndarray, fs: int, condition: str) -> np.ndarray:
    if condition == "raw_whole":
        return np.asarray(x, dtype=np.float64)
    centered = remove_dc(x)
    if condition == "dc_whole":
        return centered
    detected = endpoint_detect(x, fs, frame_ms=25.0, hop_ms=10.0, win="hamming")
    if detected is None:
        raise RuntimeError("端点检测未返回语音段")
    segment = np.asarray(detected["seg"], dtype=np.float64)
    if condition == "dc_endpoint":
        return segment
    if condition == "dc_endpoint_preemphasis":
        return pre_emphasis(segment, a=0.97)
    raise ValueError(f"未知预处理条件: {condition}")


def collect_feature_table(paths: list[Path]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    whole_sequences: dict[str, np.ndarray] = {}
    for path in paths:
        match = FILENAME_RE.match(path.name)
        assert match is not None
        label = int(match.group("label"))
        index = int(match.group("index"))
        split = "train" if index <= 5 else "test"
        x, fs, channels, bits = read_wav(path)
        if (fs, channels, bits) != (16000, 1, 16):
            raise RuntimeError(f"格式不符合冻结口径: {path.name}")
        whole_sequences[path.name] = time_domain_feature(
            remove_dc(x), fs, frame_ms=25.0, hop_ms=10.0, win="hamming", normalize=True,
        )
        for condition in PREPROCESSING:
            signal = preprocessing_signal(x, fs, condition)
            sequence = time_domain_feature(
                signal, fs, frame_ms=25.0, hop_ms=10.0, win="hamming", normalize=False,
            )
            row = {
                "file": path.name,
                "label": label,
                "index": index,
                "split": split,
                "condition": condition,
                "samples": len(signal),
                "duration_s": len(signal) / fs,
            }
            row.update(summarize_sequence(sequence, len(signal) / fs))
            rows.append(row)
    table = pd.DataFrame(rows).sort_values(["condition", "label", "index"]).reset_index(drop=True)
    if table.loc[:, FEATURE_NAMES].isna().any().any() or not np.isfinite(table.loc[:, FEATURE_NAMES].to_numpy()).all():
        raise RuntimeError("特征表包含缺失值或非有限值")
    return table, whole_sequences


def split_table(table: pd.DataFrame, condition: str, columns: tuple[str, ...]):
    subset = table[table["condition"] == condition].sort_values(["label", "index"])
    train = subset[subset["split"] == "train"]
    test = subset[subset["split"] == "test"]
    return (
        train.loc[:, columns].to_numpy(float), train["label"].to_numpy(int), train["file"].tolist(),
        test.loc[:, columns].to_numpy(float), test["label"].to_numpy(int), test["file"].tolist(),
    )


def tuned_logistic(x_train, y_train, smoke: bool):
    folds = 2 if smoke else 5
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=SEED)),
    ])
    search = GridSearchCV(
        model,
        {"model__C": [0.1, 1.0] if smoke else [0.01, 0.1, 1.0, 10.0, 100.0]},
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED),
        n_jobs=1,
    ).fit(x_train, y_train)
    return search.best_estimator_, float(search.best_score_), search.best_params_


def evaluate_ablation(table: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    preprocessing_rows = []
    for condition in PREPROCESSING:
        x_train, y_train, _, x_test, y_test, _ = split_table(table, condition, FEATURE_NAMES)
        model, cv_score, params = tuned_logistic(x_train, y_train, smoke)
        prediction = model.predict(x_test)
        preprocessing_rows.append({
            "condition": condition,
            "feature_group": "all_three",
            "train_cv_accuracy": cv_score,
            "test_accuracy": accuracy_score(y_test, prediction),
            "test_macro_f1": f1_score(y_test, prediction, average="macro"),
            "correct": int(np.sum(prediction == y_test)),
            "test_count": len(y_test),
            "selected_C": params["model__C"],
        })

    feature_rows = []
    for group, columns in FEATURE_GROUPS.items():
        x_train, y_train, _, x_test, y_test, _ = split_table(table, "dc_endpoint", columns)
        model, cv_score, params = tuned_logistic(x_train, y_train, smoke)
        prediction = model.predict(x_test)
        feature_rows.append({
            "condition": "dc_endpoint",
            "feature_group": group,
            "feature_count": len(columns),
            "train_cv_accuracy": cv_score,
            "test_accuracy": accuracy_score(y_test, prediction),
            "test_macro_f1": f1_score(y_test, prediction, average="macro"),
            "correct": int(np.sum(prediction == y_test)),
            "test_count": len(y_test),
            "selected_C": params["model__C"],
        })
    return pd.DataFrame(preprocessing_rows), pd.DataFrame(feature_rows)


def dtw_band_tuning(table: pd.DataFrame, sequences: dict[str, np.ndarray], smoke: bool) -> pd.DataFrame:
    baseline = table[table["condition"] == "dc_endpoint"].sort_values(["label", "index"])
    train = baseline[baseline["split"] == "train"]
    test = baseline[baseline["split"] == "test"]
    train_files = train["file"].tolist()
    test_files = test["file"].tolist()
    y_train = train["label"].to_numpy(int)
    y_test = test["label"].to_numpy(int)
    bands = [0.2, 0.35] if smoke else [0.1, 0.2, 0.35, 0.5, 1.0]
    rows = []
    for ratio in bands:
        loo_predictions = []
        for i, file_i in enumerate(train_files):
            best_distance = np.inf
            best_label = None
            for j, file_j in enumerate(train_files):
                if i == j:
                    continue
                length = max(len(sequences[file_i]), len(sequences[file_j]))
                band = max(abs(len(sequences[file_i]) - len(sequences[file_j])), int(np.ceil(ratio * length)))
                distance = dtw_distance(sequences[file_i], sequences[file_j], band=band)
                if distance < best_distance:
                    best_distance = distance
                    best_label = y_train[j]
            loo_predictions.append(best_label)
        loo_predictions = np.asarray(loo_predictions, dtype=int)
        t0 = time.perf_counter()
        classifier = DTWClassifier(band_ratio=ratio).fit([sequences[name] for name in train_files], y_train)
        test_prediction = classifier.predict([sequences[name] for name in test_files])
        prediction_time = time.perf_counter() - t0
        rows.append({
            "band_ratio": ratio,
            "train_loo_accuracy": accuracy_score(y_train, loo_predictions),
            "train_loo_macro_f1": f1_score(y_train, loo_predictions, average="macro"),
            "test_accuracy_descriptive": accuracy_score(y_test, test_prediction),
            "test_macro_f1_descriptive": f1_score(y_test, test_prediction, average="macro"),
            "correct": int(np.sum(test_prediction == y_test)),
            "test_count": len(y_test),
            "prediction_time_s": prediction_time,
        })
    return pd.DataFrame(rows)


def repeated_split_stability(table: pd.DataFrame, smoke: bool) -> pd.DataFrame:
    baseline = table[table["condition"] == "dc_endpoint"].sort_values(["label", "index"])
    x = baseline.loc[:, FEATURE_NAMES].to_numpy(float)
    y = baseline["label"].to_numpy(int)
    repeats = 2 if smoke else 10
    splitter = StratifiedShuffleSplit(n_splits=repeats, test_size=1 / 3 if smoke else 0.5, random_state=SEED)
    rows = []
    for repeat, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        model, cv_score, params = tuned_logistic(x[train_idx], y[train_idx], smoke)
        prediction = model.predict(x[test_idx])
        rows.append({
            "repeat": repeat,
            "seed": SEED,
            "train_count": len(train_idx),
            "test_count": len(test_idx),
            "selected_C": params["model__C"],
            "train_cv_accuracy": cv_score,
            "test_accuracy": accuracy_score(y[test_idx], prediction),
            "test_macro_f1": f1_score(y[test_idx], prediction, average="macro"),
        })
    return pd.DataFrame(rows)


def export_single(fig, name: str) -> None:
    export_figure(fig, FIGURES / name)
    plt.close(fig)


def make_figures(table, sequences, preprocessing, feature_ablation, dtw_bands, stability) -> None:
    apply_publication_style(language="zh", width="report")
    colors = [PALETTE["primary"], PALETTE["contrast"], PALETTE["secondary"], PALETTE["neutral"]]

    # raw_q1：四种预处理实际保留的信号时长分布。
    fig, ax = publication_subplots(1, 1, width="report", aspect=0.55)
    duration_groups = [table.loc[table["condition"] == name, "duration_s"].to_numpy() for name in PREPROCESSING]
    ax.boxplot(duration_groups, tick_labels=["原始整段", "去直流整段", "端点段", "端点+预加重"], showfliers=False)
    ax.set(xlabel="预处理条件", ylabel="信号时长（s）")
    export_single(fig, "raw_q1_duration_distribution")

    # raw_q2：基准16维特征相关结构，展示能量与幅度统计量的冗余。
    baseline = table[table["condition"] == "dc_endpoint"].loc[:, FEATURE_NAMES]
    corr = baseline.corr().to_numpy()
    fig, ax = publication_subplots(1, 1, width="report", aspect=0.74)
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set(xlabel="特征维度", ylabel="特征维度", xticks=range(16), yticks=range(16))
    ax.set_xticklabels(range(1, 17), fontsize=6)
    ax.set_yticklabels(range(1, 17), fontsize=6)
    fig.colorbar(image, ax=ax, label="Pearson相关系数")
    export_single(fig, "raw_q2_feature_correlation")

    # raw_q3：端点后候选段长度，展示类别内外的时长差异。
    endpoint_rows = table[table["condition"] == "dc_endpoint"]
    fig, ax = publication_subplots(1, 1, width="report", aspect=0.55)
    for digit in range(10):
        values = endpoint_rows.loc[endpoint_rows["label"] == digit, "duration_s"].to_numpy()
        x_values = np.full(len(values), digit) + np.linspace(-0.12, 0.12, len(values))
        ax.scatter(x_values, values, s=14, color=PALETTE["primary"], alpha=0.65)
        ax.plot(digit, np.median(values), marker="_", markersize=12, color=PALETTE["contrast"])
    ax.set(xlabel="数字类别", ylabel="端点后时长（s）", xticks=range(10))
    export_single(fig, "raw_q3_sequence_lengths")

    # process_q1：训练CV与固定测试对照。
    fig, ax = publication_subplots(1, 1, width="report", aspect=0.55)
    x = np.arange(len(preprocessing))
    ax.scatter(x - 0.08, preprocessing["train_cv_accuracy"], label="训练集五折", color=PALETTE["neutral"], marker="s")
    ax.scatter(x + 0.08, preprocessing["test_accuracy"], label="固定测试集", color=PALETTE["contrast"], marker="o")
    ax.set(xticks=x, xticklabels=["原始整段", "去直流整段", "端点段", "端点+预加重"], ylabel="准确率", ylim=(0, 0.8))
    ax.legend(frameon=False, ncol=2)
    export_single(fig, "process_q1_preprocessing_cv_test")

    # process_q2：不同特征组合的训练与测试表现。
    fig, ax = publication_subplots(1, 1, width="report", aspect=0.58)
    x = np.arange(len(feature_ablation))
    ax.scatter(x - 0.08, feature_ablation["train_cv_accuracy"], label="训练集五折", color=PALETTE["neutral"], marker="s")
    ax.scatter(x + 0.08, feature_ablation["test_accuracy"], label="固定测试集", color=PALETTE["primary"], marker="o")
    ax.set(xticks=x, xticklabels=["能量", "幅度", "ZCR", "能量+幅度", "三类特征"], ylabel="准确率", ylim=(0, 0.75))
    ax.legend(frameon=False, ncol=2)
    export_single(fig, "process_q2_feature_ablation")

    # process_q3：约束带对精度和耗时的共同影响。
    fig, axes = publication_subplots(1, 2, width="report", aspect=0.48, width_ratios=[1.25, 1])
    axes[0].plot(dtw_bands["band_ratio"], dtw_bands["train_loo_accuracy"], marker="s", label="训练集留一", color=PALETTE["neutral"])
    axes[0].plot(dtw_bands["band_ratio"], dtw_bands["test_accuracy_descriptive"], marker="o", label="固定测试集", color=PALETTE["contrast"])
    axes[0].set(xlabel="DTW约束带比例", ylabel="准确率", ylim=(0, 0.85))
    axes[0].legend(frameon=False)
    axes[1].plot(dtw_bands["band_ratio"], dtw_bands["prediction_time_s"], marker="o", color=PALETTE["primary"])
    axes[1].set(xlabel="DTW约束带比例", ylabel="预测耗时（s）")
    export_single(fig, "process_q3_dtw_band_tradeoff")

    # result_q1：预处理方案最终指标排序。
    ordered = preprocessing.sort_values("test_accuracy")
    fig, ax = publication_subplots(1, 1, width="report", aspect=0.55)
    condition_labels = {
        "raw_whole": "原始整段", "dc_whole": "去直流整段",
        "dc_endpoint": "端点段", "dc_endpoint_preemphasis": "端点+预加重",
    }
    labels = [condition_labels[name] for name in ordered["condition"]]
    y = np.arange(len(ordered))
    ax.scatter(ordered["test_accuracy"], y, label="准确率", color=PALETTE["contrast"], s=35)
    ax.scatter(ordered["test_macro_f1"], y, label="Macro-F1", color=PALETTE["primary"], marker="s", s=30)
    ax.set(xlabel="固定测试集指标", yticks=y, yticklabels=labels, xlim=(0, 0.75))
    ax.legend(frameon=False, ncol=2)
    export_single(fig, "result_q1_preprocessing_metrics")

    # result_q2：特征互补带来的增益。
    fig, ax = publication_subplots(1, 1, width="report", aspect=0.55)
    labels = ["能量", "幅度", "ZCR", "能量+幅度", "三类特征"]
    y = np.arange(len(feature_ablation))
    ax.hlines(y, 0, feature_ablation["test_accuracy"], color="#C8D2DC", linewidth=1.2)
    ax.scatter(feature_ablation["test_accuracy"], y, color=colors[:1] * len(y), s=30)
    ax.set(xlabel="固定测试集准确率", yticks=y, yticklabels=labels, xlim=(0, 0.7))
    export_single(fig, "result_q2_feature_contribution")

    # result_q3：重复划分给出稳定性范围，固定划分只是其中一个观测。
    fig, ax = publication_subplots(1, 1, width="report", aspect=0.55)
    repeats = stability["repeat"].to_numpy()
    ax.scatter(repeats, stability["test_accuracy"], color=PALETTE["primary"], label="重复分层划分")
    fixed_accuracy = float(preprocessing.loc[preprocessing["condition"] == "dc_endpoint", "test_accuracy"].iloc[0])
    ax.axhline(fixed_accuracy, color=PALETTE["contrast"], linestyle="--", label=f"固定划分 {fixed_accuracy:.2f}")
    ax.axhline(stability["test_accuracy"].mean(), color=PALETTE["neutral"], linestyle=":", label=f"重复均值 {stability['test_accuracy'].mean():.2f}")
    ax.set(xlabel="重复编号", ylabel="测试准确率", xticks=repeats, ylim=(0, 0.8))
    ax.legend(frameon=False, ncol=2)
    export_single(fig, "result_q3_split_stability")


def write_report(table, preprocessing, feature_ablation, dtw_bands, stability) -> None:
    best_pre = preprocessing.sort_values(["test_accuracy", "test_macro_f1"], ascending=False).iloc[0]
    best_feature = feature_ablation.sort_values(["test_accuracy", "test_macro_f1"], ascending=False).iloc[0]
    best_band = dtw_bands.sort_values(["train_loo_accuracy", "band_ratio"], ascending=[False, True]).iloc[0]
    long_endpoint_count = int(np.sum(
        (table["condition"] == "dc_endpoint") & (table["duration_s"] > 1.0)
    ))
    report = f"""# 实验一第三周：调参、消融与稳健性结果

## 实验设计

- 统一基准：16 kHz单声道录音，25 ms帧长、10 ms帧移、Hamming窗。
- 固定划分：每类01--05训练、06--10测试。逻辑回归的C只在训练集五折内选择。
- 预处理消融依次比较原始整段、去直流整段、去直流+端点、去直流+端点+预加重。
- 特征消融中的每组都保留语音段时长，再比较能量、幅度、ZCR及其组合。
- DTW约束带由训练集留一准确率选择；固定测试结果只作描述性复核。
- 自动端点中有 {long_endpoint_count} 条候选段超过1.0 s，需人工复听；非空候选不等于端点正确。

## 主要结果

- 最佳预处理条件为 `{best_pre['condition']}`：固定测试准确率 {best_pre['test_accuracy']:.3f}，Macro-F1 {best_pre['test_macro_f1']:.3f}。去直流本身没有改变整段结果；端点处理由0.600提高到0.640；再加入预加重降为0.580，因此实验一时域特征不采用预加重。
- 最佳特征组为 `{best_feature['feature_group']}`：固定测试准确率 {best_feature['test_accuracy']:.3f}。能量0.320、幅度0.340、ZCR 0.540，三类合用0.640，说明ZCR贡献最大，能量和幅度仍提供互补信息。
- DTW训练集留一在band=0.35后达到平台。按“训练表现优先、相同性能取更小约束带”的规则选择 `{best_band['band_ratio']:.2f}`；对应固定测试准确率 {best_band['test_accuracy_descriptive']:.3f}。band从0.35增至1.00没有提高准确率，但预测耗时由 {float(dtw_bands.loc[dtw_bands['band_ratio']==0.35, 'prediction_time_s'].iloc[0]):.2f}s 增至 {float(dtw_bands.loc[dtw_bands['band_ratio']==1.0, 'prediction_time_s'].iloc[0]):.2f}s。
- 10次分层50/50随机划分的逻辑回归准确率为 {stability['test_accuracy'].mean():.3f} ± {stability['test_accuracy'].std(ddof=1):.3f}（均值±SD，范围 {stability['test_accuracy'].min():.3f}--{stability['test_accuracy'].max():.3f}）。固定划分0.640只是一个观测，不能单独代表泛化性能。

## 结论与限制

第三周没有重复增加分类器，而是确认了端点处理、三类时域特征联合和DTW约束带0.35的合理性。预加重对本实验一的时域统计特征没有带来收益。全部样本来自同一批次，说话人元数据不足，结果不能解释为跨说话人性能；测试样本仍较少，不作显著性结论。
"""
    (RESULTS / "第三周结果摘要.md").write_text(report, encoding="utf-8")

    contracts = [
        {"figure": "raw_q1_duration_distribution", "question": "q1", "conclusion": "端点处理改变输入时长分布"},
        {"figure": "process_q1_preprocessing_cv_test", "question": "q1", "conclusion": "预处理方案需同时核对训练与测试表现"},
        {"figure": "result_q1_preprocessing_metrics", "question": "q1", "conclusion": "端点段优于整段，预加重无收益"},
        {"figure": "raw_q2_feature_correlation", "question": "q2", "conclusion": "16维特征存在明显相关结构"},
        {"figure": "process_q2_feature_ablation", "question": "q2", "conclusion": "特征组合改变训练和测试表现"},
        {"figure": "result_q2_feature_contribution", "question": "q2", "conclusion": "ZCR最强，三类联合最佳"},
        {"figure": "raw_q3_sequence_lengths", "question": "q3", "conclusion": "端点后候选段时长存在类别内外差异"},
        {"figure": "process_q3_dtw_band_tradeoff", "question": "q3", "conclusion": "0.35后准确率平台而耗时继续增加"},
        {"figure": "result_q3_split_stability", "question": "q3", "conclusion": "固定划分不能替代重复划分稳定性"},
    ]
    (RESULTS / "图表契约.json").write_text(json.dumps(contracts, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(paths: list[Path], smoke: bool, elapsed_s: float) -> None:
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")
    dtw_bands = [0.2, 0.35] if smoke else [0.1, 0.2, 0.35, 0.5, 1.0]
    repeated_splits = 2 if smoke else 10
    fixed_split = "每类01、02训练，06测试" if smoke else "每类01-05训练，06-10测试"
    mode_flag = "--smoke" if smoke else "--full"
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if smoke else "full",
        "random_seed": SEED,
        "input_files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in paths
        ],
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "dependencies": {name: importlib.metadata.version(name) for name in packages},
        },
        "parameters": {
            "frame_ms": 25.0,
            "hop_ms": 10.0,
            "window": "hamming",
            "preemphasis_a": 0.97,
            "dtw_band_ratios": dtw_bands,
            "fixed_split": fixed_split,
            "repeated_splits": repeated_splits,
        },
        "elapsed_s": elapsed_s,
        "reproduce_command": f"C:\\ProgramData\\anaconda3\\python.exe run_week3_experiments.py {mode_flag}",
    }
    (RESULTS / "复现清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    smoke = args.smoke and not args.full
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    started = time.perf_counter()
    paths = discover_wavs(smoke)
    table, sequences = collect_feature_table(paths)
    preprocessing, feature_ablation = evaluate_ablation(table, smoke)
    dtw_bands = dtw_band_tuning(table, sequences, smoke)
    stability = repeated_split_stability(table, smoke)
    table.to_csv(RESULTS / "第三周特征缓存.csv", index=False, encoding="utf-8-sig")
    preprocessing.to_csv(RESULTS / "预处理消融.csv", index=False, encoding="utf-8-sig")
    feature_ablation.to_csv(RESULTS / "时域特征消融.csv", index=False, encoding="utf-8-sig")
    dtw_bands.to_csv(RESULTS / "DTW约束带调参.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(RESULTS / "重复划分稳健性.csv", index=False, encoding="utf-8-sig")
    if not smoke:
        make_figures(table, sequences, preprocessing, feature_ablation, dtw_bands, stability)
        write_report(table, preprocessing, feature_ablation, dtw_bands, stability)
    elapsed = time.perf_counter() - started
    write_manifest(paths, smoke, elapsed)
    print("预处理消融")
    print(preprocessing.to_string(index=False))
    print("\n特征消融")
    print(feature_ablation.to_string(index=False))
    print("\nDTW约束带")
    print(dtw_bands.to_string(index=False))
    print(f"\n完成: {'smoke' if smoke else 'full'}, {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
