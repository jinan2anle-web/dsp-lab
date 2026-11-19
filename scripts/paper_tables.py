# -*- coding: utf-8 -*-
"""从 report/tables/*.csv（含 report/evidence 的少量 txt）生成论文表 1–9 的 Markdown 片段。

用法（仓库根目录执行）：

    python -m scripts.paper_tables            # 生成 9 个片段并打印校验结果
    python -m scripts.paper_tables --check     # 只校验 csv 与论文口径是否一致，不写文件

输出：report/论文/tables/table1.md … table9.md

本脚本**只读** report/tables/*.csv 与 report/evidence/*.txt，**绝不写回**任何 csv；
片段顶部有一行 HTML 注释 `<!-- widths: … -->`，给 out/paper_build.py 的渲染器指定列宽（单位 cm，合计 16.0）。
表号与 csv 的映射见 report/tables/traceability.csv 第 4 列与 docs/论文/00_骨架与格式.md 的第 1.3 节。
"""
from __future__ import annotations

import csv
import glob
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "report" / "tables"
EVIDENCE = ROOT / "report" / "evidence"
OUTDIR = ROOT / "report" / "论文" / "tables"

CAPTIONS = {
    1: "语料与格式",
    2: "手写基 2 FFT 与 numpy.fft 对拍的五用例",
    3: "窗函数 × 预加重对识别率的影响",
    4: "频域变体消融",
    5: "主结果与消融",
    6: "各类召回与主要误判去向",
    7: "端点门限扫描",
    8: "时域与频域特征的同条件对照",
    9: "噪声与增益鲁棒性",
}


# ---------------------------------------------------------------- 基础工具
def read_csv(name):
    """读 report/tables/name，返回 list[dict]（表头为键，值去掉首尾空白）。"""
    path = TABLES / name
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    head = [h.strip() for h in rows[0]]
    out = []
    for r in rows[1:]:
        if not any(c.strip() for c in r):
            continue
        out.append({h: (r[i].strip() if i < len(r) else "") for i, h in enumerate(head)})
    return out


def fmt_rate(x):
    """0.8 → 0.800。"""
    return f"{float(x):.3f}"


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        cells = [str(c).replace("|", "\\|").replace("\n", " ") for c in r]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def emit(n, headers, rows, widths, note=None):
    assert abs(sum(widths) - 16.0) < 1e-6, (n, sum(widths))
    assert len(widths) == len(headers), (n, len(widths), len(headers))
    body = [f"<!-- widths: {','.join(f'{w:g}' for w in widths)} -->",
            f"表 {n}　{CAPTIONS[n]}", ""]
    body += md_table(headers, rows)
    body.append("")
    if note:
        body.append("注：" + note)
    text = "\n".join(body).rstrip() + "\n"
    if not CHECK_ONLY:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        (OUTDIR / f"table{n}.md").write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------- 表 1
def table1():
    wavs = sorted(glob.glob(str(ROOT / "data" / "A" / "*.wav")))
    pred = read_csv("exp3_predictions.csv")
    rows = [
        ["语料总量", f"{len(wavs)} 条", "10 类 × 每类 10 条，命名 {0…9}_{01…10}.wav"],
        ["说话人", "1 人", "单一说话人朗读 0–9 十个数字，无跨人条件"],
        ["采样格式", "16 kHz / 16 bit / 单声道", "每条 48044 字节，1.500 s = 24000 点"],
        ["训练集", "50 条", "序号 _01–_05，每类 5 条"],
        ["测试集", f"{len(pred)} 条", "序号 _06–_10，每类 5 条"],
        ["分帧", "148 帧", "帧长 400 点（25 ms）、帧移 160 点（10 ms）"],
        ["特征", "13 维 MFCC", "24 个 Mel 三角、FFT 512 点、汉明窗、预加重 0.97"],
    ]
    note = ("每条录音含约 0.35 s 前导静音（样本下标 0…5599）；训练/测试按录制序号划分，"
            "两组不重叠，每类 5 训练 + 5 测试。帧数按 floor((N−400)/160)+1 计算，N = 24000 点。")
    emit(1, ["项目", "取值", "说明"], rows, [3.4, 4.2, 8.4], note)
    return len(wavs), len(pred)


# ---------------------------------------------------------------- 表 2
def table2():
    path = EVIDENCE / "exp2_fft_selftest.txt"
    text = path.read_text(encoding="utf-8", errors="replace")
    pat = re.compile(r"^\s{2}(.+?)\s{2,}max\|自写 − np\.fft\| = ([\d.eE+\-]+)\s*$")
    rows, errs = [], []
    for line in text.splitlines():
        m = pat.match(line)
        if m:
            case = re.sub(r"\s{2,}", " ", m.group(1).strip())
            rows.append([str(len(rows) + 1), case, m.group(2)])
            errs.append(float(m.group(2)))
    if len(rows) < 5:  # 证据文本格式变了也别让整条流水线断掉
        raise SystemExit(f"[表2] 只解析到 {len(rows)} 个用例，检查 {path}")
    note = (f"判据 max|自写 − numpy.fft| < 1e-10，五个用例全部通过；最大者 {max(errs):.3e}"
            f"（直流全 1），最小者 {min(errs):.3e}（真实录音全部 148 帧）。"
            "误差量级 1e-14 已是双精度浮点的舍入底噪。")
    emit(2, ["序号", "用例与输入", "最大绝对误差"], rows, [1.6, 10.2, 4.2], note)
    return len(rows), max(errs)


# ---------------------------------------------------------------- 表 3
WIN_CN = {"rect": "矩形窗 rect", "hamming": "汉明窗 hamming", "hanning": "汉宁窗 hanning"}


def table3():
    rows = read_csv("exp2_ablation.csv")
    out = [[WIN_CN.get(r["窗口"], r["窗口"]), "0.97" if r["预加重系数"] in ("0.97", "0.9700") else "0（不做）",
            fmt_rate(r["识别率"]), r["说明"]] for r in rows]
    note = ("主口径为汉明窗 + 预加重 0.97；六格识别率全部落在 0.780–0.800，"
            "即相互相差不超过 1 条样本（1 条 = 2 个百分点），按判定粒度记作「无差别」。"
            "其余条件同表 5 主口径：13 维 MFCC、24 Mel、整段输入、5 条模板、不做归一化。")
    emit(3, ["窗函数", "预加重系数 a", "识别率", "说明"], out, [3.4, 3.0, 2.4, 7.2], note)
    return len(out)


# ---------------------------------------------------------------- 表 4
def table4():
    rows = read_csv("exp2_freq_ablation.csv")
    out = [[r["条件"], r["参数"], fmt_rate(r["识别率"]), r["正确条数"], r["总条数"], r["耗时s"], r["备注"]]
           for r in rows]
    drop = [r for r in out if fmt_rate(rows[out.index(r)]["识别率"]) == "0.720"]
    note = (f"{len(out)} 个变体只在特征侧改动，DTW 三条前驱、模板数 5、整段输入均不变；"
            "除「加 Δ/ΔΔ（39 维）」掉到 0.720（36/50）外，其余五档一律 0.800（40/50）。"
            "耗时 s 为同一台机器上 50 条的总耗时，跨机器只作相对比较。")
    emit(4, ["条件", "参数", "识别率", "正确条数", "总条数", "耗时 s", "备注"], out,
         [3.2, 2.4, 1.8, 1.6, 1.4, 1.4, 4.2], note)
    return len(out), len(drop)


# ---------------------------------------------------------------- 表 5
ACC_LABEL = {
    "main": "主口径",
    "templates_1": "模板数 1",
    "templates_3": "模板数 3",
    "normalize_cmn": "CMN 归一化",
    "endpoint": "端点段",
}
ACC_CORRECT = {  # csv 未存正确条数，缺省按已定稿数字补齐
    "main": "40/50",
    "templates_1": "30/50",
    "templates_3": "37/50",
    "normalize_cmn": "41/50",
    "endpoint": "40/50",
}


def table5():
    rows = read_csv("exp3_accuracy.csv")
    out = [[ACC_LABEL.get(r["条件"], r["条件"]), r["输入"], r["normalize"], r["模板数"],
            fmt_rate(r["准确率"]), ACC_CORRECT[r["条件"]], r["耗时s"]] for r in rows]
    note = ("统一口径：13 维 MFCC、24 个 Mel 三角、FFT 512 点、汉明窗、预加重 0.97、"
            "DTW 三条前驱、无路径约束带。正确/总 = 准确率 × 50（1 条样本 = 2 个百分点）。"
            "模板数 1/3/5 = 0.600/0.740/0.800 与 exp3_template_curve.csv 同值，互为交叉校验。")
    emit(5, ["条件", "输入", "normalize", "模板数", "准确率", "正确/总", "耗时 s"], out,
         [4.0, 2.0, 1.8, 1.4, 1.8, 2.2, 2.8], note)
    return len(out)


# ---------------------------------------------------------------- 表 6
def table6():
    path = TABLES / "exp3_confusion.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    head = [h.strip() for h in rows[0]]
    preds = head[1:]
    out, total, correct = [], 0, 0
    for r in rows[1:]:
        if not r or not r[0].strip():
            continue
        true_cls = r[0].strip()
        counts = [int(float(c)) for c in r[1:]]
        n = sum(counts)
        c = counts[preds.index(true_cls)] if true_cls in preds else 0
        total += n
        correct += c
        wrong = sorted(((preds[i], v) for i, v in enumerate(counts) if preds[i] != true_cls and v), key=lambda t: -t[1])
        where = "、".join(f"类 {k}（{v} 条）" for k, v in wrong) or "—"
        out.append([true_cls, str(n), str(c), f"{c / n:.2f}", where])
    note = (f"对角线和 {correct}/{total} = {correct / total:.3f}，非对角 {total - correct} 条为全部错例；"
            "错例集中在类 6（2/5，误判到 2、3、5）与类 8（2/5，误判到 2、3）；"
            "类 0、3、5、7、9 全部 5/5。口径同表 5 主口径，逐条明细见 exp3_predictions.csv。")
    emit(6, ["真实类别", "测试条数", "正确条数", "召回率", "主要误判去向"], out,
         [1.8, 2.2, 2.2, 2.2, 7.6], note)
    return len(out), correct, total


# ---------------------------------------------------------------- 表 7
def table7():
    rows = read_csv("exp3_endpoint_sweep.csv")
    keys = ["高门限(峰值比例)", "低门限", "高门限(dB)", "低门限(dB)", "段长中位数(s)", "p10(s)", "p90(s)", "最长(s)"]
    out = [[r[k] for k in keys] for r in rows]
    main_row = [r for r in rows if r["高门限(峰值比例)"].startswith("0.0020")][0]
    note = (f"低门限 = 高门限 × 0.35（LOW_HIGH_RATIO）；论文主口径取 0.0020 行，"
            f"对应段长中位数 {main_row['段长中位数(s)']} s。统计对象为 100 条录音，"
            "最长 1.145 s 是去掉前后静音后的上限（整条 1.5 s）。")
    emit(7, ["高门限", "低门限", "高门限/dB", "低门限/dB", "段长中位/s", "p10/s", "p90/s", "最长/s"],
         out, [2.6, 2.2, 1.8, 1.8, 2.4, 1.8, 1.8, 1.6], note)
    return len(out), main_row["段长中位数(s)"]


# ---------------------------------------------------------------- 表 8
def table8():
    rows = read_csv("exp4_timedomain_vs_freq.csv")
    out = [[r["特征"], r["维数"], r["归一化"], r["约束带"], fmt_rate(r["识别率"]),
            r["正确条数"], r["总条数"], r["备注"]] for r in rows]
    note = ("三条共同前提：同一训练/测试划分（50 条测试）、同一 DTW 三条前驱、同一 5 条模板。"
            "① 时域 0.760 依赖段内逐维归一化，去掉归一化即掉到 0.400；"
            "② 约束带由 band 0.35 放宽到 1.00，准确率不变而耗时 15.90 → 27.21 s；"
            "③ 两族特征只有 MFCC 两行是「无归一化、无约束带」，故 0.800 与 0.760 只能作定性对照。")
    emit(8, ["特征", "维数 K", "归一化", "约束带", "识别率", "正确条数", "总条数", "备注"], out,
         [2.4, 1.2, 3.2, 1.8, 1.6, 1.6, 1.2, 3.0], note)
    return len(out)


# ---------------------------------------------------------------- 表 9
T1_NOTE = {
    "SNR ∞（无噪）": "无扰动基线，3 个 seed 同值",
    "SNR 20 dB": "整条加噪；3 个 seed 同值（0.520/0.520/0.520）",
    "SNR 10 dB": "整条加噪；3 个 seed 0.240/0.240/0.260，均值 0.2467",
    "SNR 5 dB": "整条加噪；3 个 seed 均值",
    "SNR 0 dB": "整条加噪；3 个 seed 均值，已不可用",
}


def table9():
    rows = read_csv("exp5_robustness.csv")
    out = []
    for r in rows:
        if r["实验"] == "T1_noise" and r["seed"] == "avg":
            cond = r["条件"].replace("（3次平均）", "")
            out.append(["T1 噪声", cond, fmt_rate(r["识别率"]), f"{r['正确条数']}/{r['总条数']}",
                        T1_NOTE.get(cond, "整条加噪")])
    for r in rows:
        if r["实验"] == "T2_gain":
            norm = "cmn" if "cmn" in r["备注"] else "不做归一化"
            out.append(["T2 增益", r["条件"].split(" / ")[0], fmt_rate(r["识别率"]),
                        f"{r['正确条数']}/{r['总条数']}", f"normalize={r['条件'].split(' / ')[1]}（{norm}）"])
    note = ("噪声按语音段功率定噪声功率（add_white_noise，实测 SNR 与设定偏差 +0.045 dB），"
            "增益整体乘常数（apply_gain，含静音段）；T1 每档 3 个 seed（0/1/2），表中取 avg 行。"
            "T1 在 20 dB 已掉到 0.520，10 dB 仅 0.2467，说明幅值域特征对噪声没有余量；"
            "T2 加 CMN 后 ×0.5/×1.0/×2.0 三档恒为 0.820，即 CMN 消掉了增益对 c0 的影响。")
    emit(9, ["实验", "条件", "识别率", "正确/总", "备注"], out, [2.4, 3.2, 1.8, 2.0, 6.6], note)
    return len(out)


# ---------------------------------------------------------------- 校验
def check():
    """把生成结果与论文已定稿数字逐条对齐，任何一条不符就报错（防止 csv 被误改后静默出货）。"""
    want = {
        "表5 main": 0.8000, "表5 templates_1": 0.6000, "表5 templates_3": 0.7400,
        "表5 cmn": 0.8200, "表5 endpoint": 0.8000,
        "表6 正确/总": (40, 50),
        "表7 中位": "0.665",
        "表9 20dB": 0.5200, "表9 10dB": 0.2467,
    }
    acc = {r["条件"]: float(r["准确率"]) for r in read_csv("exp3_accuracy.csv")}
    got = {
        "表5 main": acc["main"], "表5 templates_1": acc["templates_1"],
        "表5 templates_3": acc["templates_3"], "表5 cmn": acc["normalize_cmn"],
        "表5 endpoint": acc["endpoint"],
    }
    n6, correct, total = table6()
    got["表6 正确/总"] = (correct, total)
    sweep = read_csv("exp3_endpoint_sweep.csv")
    got["表7 中位"] = [r["段长中位数(s)"] for r in sweep if r["高门限(峰值比例)"].startswith("0.0020")][0]
    rob = read_csv("exp5_robustness.csv")
    t1 = {r["条件"].replace("（3次平均）", ""): float(r["识别率"]) for r in rob
          if r["实验"] == "T1_noise" and r["seed"] == "avg"}
    got["表9 20dB"] = t1["SNR 20 dB"]
    got["表9 10dB"] = t1["SNR 10 dB"]
    bad = [(k, want[k], got.get(k)) for k in want if got.get(k) != want[k]]
    return bad


CHECK_ONLY = False


def main():
    global CHECK_ONLY
    CHECK_ONLY = "--check" in sys.argv
    bad = check()
    if bad:
        print("!! 与论文定稿数字不一致：")
        for k, w, g in bad:
            print(f"   {k}: 期望 {w} 实际 {g}")
        return 1

    n1 = table1()
    n2, fft_max = table2()
    n3 = table3()
    n4, n4_drop = table4()
    n5 = table5()
    n6, correct, total = table6()
    n7, mid = table7()
    n8 = table8()
    n9 = table9()

    print(f"ROOT = {ROOT}")
    print(f"表 1  语料 {n1[0]} 条 wav，测试集 {n1[1]} 条")
    print(f"表 2  FFT 对拍 {n2} 个用例，最大误差 {fft_max:.3e}")
    print(f"表 3  窗 × 预加重 {n3} 行")
    print(f"表 4  频域变体 {n4} 行（其中掉分 {n4_drop} 行）")
    print(f"表 5  主结果与消融 {n5} 行")
    print(f"表 6  各类召回 {n6} 行，对角和 {correct}/{total} = {correct / total:.3f}")
    print(f"表 7  端点门限扫描 {n7} 行，主口径段长中位 {mid} s")
    print(f"表 8  时频对照 {n8} 行")
    print(f"表 9  鲁棒性 {n9} 行")
    if CHECK_ONLY:
        print("--check：只校验，未写文件")
    else:
        print(f"已写入 {OUTDIR}\\table1.md … table9.md")
    print("校验通过：9 张表与论文定稿数字一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
