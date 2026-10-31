"""模板库：训练集 5 条/类 → MFCC 模板，测试集 5 条/类 → 逐条 DTW 取最近模板。

划分口径写死（讲义 p14「每人 5 条做模板，另 5 条做识别」）：

    文件名 `L_XX.wav` 的 XX ∈ 01..05 → 训练（建模板）
                       XX ∈ 06..10 → 测试（识别）

缓存落在 `out/templates/`：每条录音一个 `.npy` + 一个 `index.csv`。
`out/` 在 `.gitignore` 里，所以缓存不入库，换机器用 `build()` 一条命令重建。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config, endpoint, io_wav, mfcc

TRAIN_IDS = tuple(f"{i:02d}" for i in range(1, 6))    # 01..05
TEST_IDS = tuple(f"{i:02d}" for i in range(6, 11))    # 06..10
TEMPLATE_DIR = config.OUT_DIR / "templates"


def split_of(path):
    """返回 `"train"` / `"test"` / `None`（编号不在 01..10 内）。"""
    idx = Path(path).stem.split("_")[-1]
    if idx in TRAIN_IDS:
        return "train"
    if idx in TEST_IDS:
        return "test"
    return None


def files(part=None):
    """`part` 取 `"train"`/`"test"`/`None`(全部)，返回排序后的 wav 路径列表。"""
    paths = config.wav_paths()
    if part is None:
        return paths
    return [p for p in paths if split_of(p) == part]


def _extract(path, *, use_endpoint=False, **mfcc_kwargs):
    x, fs = io_wav.read_wav(path)
    if use_endpoint:
        x, _ = endpoint.segment(x, fs)
    return mfcc.extract(x, fs, **mfcc_kwargs)


def build(part="train", *, use_endpoint=False, save=True, **mfcc_kwargs):
    """建模板库，返回 ``{label: [mfcc[T,13], ...]}``。

    同一次调用里所有参数一致（``normalize``、``n_mfcc``、``n_mels`` 等），
    避免模板与查询用两套参数——那是 DTW 结果对不上的头号原因。
    """
    table = {}
    rows = []
    for p in files(part):
        feat = _extract(p, use_endpoint=use_endpoint, **mfcc_kwargs)
        label = config.label_of(p)
        table.setdefault(label, []).append(feat)
        rows.append((label, p.stem, feat.shape[0], feat.shape[1], p.name))

    if save:
        TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
        (TEMPLATE_DIR / "index.csv").write_text(
            "label,stem,n_frames,n_mfcc,file\n"
            + "".join(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n" for r in rows),
            encoding="utf-8",
        )
        for label, seqs in table.items():
            for k, feat in enumerate(seqs):
                np.save(TEMPLATE_DIR / f"{label}_{k:02d}.npy", feat)
    return table


def load():
    """读回 `build()` 存下的模板库，返回 ``{label: [mfcc, ...]}``（按文件名序）。"""
    index = TEMPLATE_DIR / "index.csv"
    if not index.exists():
        raise FileNotFoundError(f"{index} 不存在，先跑 templates.build()")
    table = {}
    lines = index.read_text(encoding="utf-8").splitlines()[1:]
    for i, row in enumerate(lines):
        label = int(row.split(",")[0])
        table.setdefault(label, []).append(np.load(TEMPLATE_DIR / f"{label}_{len(table.get(label, [])):02d}.npy"))
    return table


def distances(query, table):
    """查询序列对每个类别的最近模板距离，返回 ``{label: min_dist}``。"""
    from . import dtw

    out = {}
    for label, seqs in table.items():
        out[label] = min(dtw.distance(query, s) for s in seqs)
    return out


def classify(query, table):
    """返回 ``(label, {label: min_dist})``，最近距离胜。"""
    d = distances(query, table)
    best = min(d, key=d.get)
    return best, d
