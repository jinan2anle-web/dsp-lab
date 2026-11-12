# 本周任务单 · B 师瑞颖（`exp3:` / `exp4:` / `exp5:` 前缀）

> 第六周（11/09–11/15）。本周老师要求：「继续完成实验二、三的实验内容，并开始设计相关测试实验」。
> 你负责**四张表**：频域变体消融、时域/频域对照、噪声与增益鲁棒性、模板数曲线。数字都从现有 100 条录音算出来，不需要新录音。

## 0 先做这三步（10 分钟）

```bash
python -m scripts.selftest                 # 看到「通过」
python -m scripts.exp3_recognize --all     # 约 2 分钟，产出基线识别率表
python -m scripts.run_all --fast           # 约 40 秒，确认图能出
```

基线必须对得上：主结果 **40/50 = 0.800**、模板 1/3/5 → **0.600/0.740/0.800**、CMN → **0.820**、端点段 → **0.800（7.5 s）**。对不上先别往下做，先查环境。

## 1 任务一：`scripts/exp2_freq_ablation.py` → `report/tables/exp2_freq_ablation.csv`

用途：回答"换一档频域参数，识别率值不值"。**判据只能是下游 DTW 识别率**（单看 MFCC 数值没有可比性）。

至少 6 行，建议这 6 个条件（每个都只动一个量）：

| 条件 | 变量 | 做法 |
|---|---|---|
| 基线 | — | 24 Mel / 13 维 / 汉明窗 / 预加重 0.97（= 主结果 0.800） |
| Mel 数 16 | `n_mels` | `extract(x, n_mels=16)`（对应 MFCC 维数仍取 13） |
| Mel 数 32 | `n_mels` | `extract(x, n_mels=32)` |
| 无 DCT（log-Mel） | `form="logmel"` | 24 维直接进 DTW（等 A 的 `form` 参数，见交接点） |
| 加 Δ/ΔΔ | `use_delta=True` | 39 维（等 A 的 `use_delta`） |
| 不用预加重 | `preemph=0.0` | 与已有 `exp2_ablation.csv` 的 0.00 行区别是这里同时固定 hamming |

csv 列：`条件,参数,识别率,正确条数,总条数,耗时s,说明`

> 注意：3、4、5 行依赖 A 的两个新参数，A 周三晚交付。**没交付就先跑 1、2、6 三行**，别干等；缺的行在 csv 里留空并写"待 A 交付"。

## 2 任务二：`report/tables/exp4_timedomain_vs_freq.csv`（时域 vs 频域）

用途：回答"同样条件下，频域特征比时域强多少"。

| 特征 | 维数 | 归一化 | 约束带 | 识别率 | 正确/50 | 说明 |
|---|---|---|---|---|---|---|
| 时域 `[T,3]` 逐帧 | 3 | 段内逐维归一化 | band 0.35 | 0.760 | 38/50 | 实验一原流程（必须写这条口径） |
| 时域 `[T,3]` 逐帧 | 3 | 无 | 无 | 0.400 | 20/50 | 去掉归一化就掉到 0.400 |
| MFCC `[T,13]` | 13 | 无 | 无 | 0.800 | 40/50 | 本实验主结果 |
| MFCC `[T,13]` | 13 | CMN | 无 | 0.820 | 41/50 | 加归一化只多 1 条 |

- 时域两行用包里自带的 `参考_实验一原代码/speech_time_domain.py` 的 `read_wav` + `time_domain_feature(normalize=...)` + `DTWClassifier(band_ratio=...)`，**整段 1.5 s**输入（用法见 `参考_实验一原代码/README.md`）。
- 表注必须写全三条：①实验一的 0.760 依赖段内逐维归一化（去掉即 0.400）；②band 0.35→1.00 准确率不变、只是耗时 15.90→27.21 s；③本实验无归一化、无约束带，所以 0.800 与 0.760 只作定性对照，不算"高 2 条"。
- **别用 `task1_w2/第二部分/frame_sequences.pkl` 做对照**——那是"端点段 + 已归一化"的缓存，跑出来只有 0.400，会得出错误结论。

## 3 任务三：`scripts/exp5_robustness.py` → `report/tables/exp5_robustness.csv`（本周重点）

先读 `docs/测试实验设计.md` 的 T1/T2 两节（协议已经写好，照着实现，不要自己改设计）。

**T1 噪声**：SNR ∈ {∞, 20, 10, 5, 0} dB × seed ∈ {0,1,2}，共 15 行。

```python
from dsp.augment import add_white_noise          # A 周三晚交付
x = add_white_noise(x0, snr_db, seed=seed)       # 内部按语音段功率定噪声
feats = mfcc.extract(x)                          # 其余参数全部用默认
```

**T2 增益**：factor ∈ {0.5, 1.0, 2.0} × normalize ∈ {none, cmn}，共 6 行。用 `apply_gain(x0, factor)`，加完再 `extract(..., normalize=...)`。

csv 列（与设计文档第 4 节一致）：

```
实验,条件,参数,重复次数,识别率,正确条数,总条数,耗时s,seed,备注
```

- `实验` ∈ {`T1_noise`,`T2_gain`}；`条件` 用中文标签（`SNR 10 dB` / `gain ×2.0 / cmn`）。
- 噪声 15 行按 seed 分行保留，另外在 `说明` 里写「3 次平均 ± 极差」。
- T2 **必须同时给 none 与 cmn**——这一实验要验证的正是"整体增益只让 c0 平移，CMN 能消掉"这个推论。

**写报告时对照预期**（设计文档第 3 节已写死）：20 dB 基本不掉、10 dB 掉 1–3 条、0 dB 接近随机；T2 的 `none` 掉、`cmn` 不掉。**不符合预期就如实写并解释**，这比符合预期更有价值。

## 4 任务四：`report/tables/exp3_template_curve.csv`（顺手）

每类模板 1…5 条各一行（模板固定取训练集 01–05 的前 n 条，不随机挑）：

```
模板数,识别率,正确条数,总条数,耗时s
```

1/3/5 三行应与现有 `exp3_accuracy.csv` 完全一致（0.600/0.740/0.800），不一致说明取模板的方式不同，先查再交。

## 5 交接与等待

| 时间 | 事项 |
|---|---|
| 周一晚 | C 发接口冻结（`augment` 签名 + `extract(form=, use_delta=)`），收到后回一句"确认" |
| 周三晚 | A 交 `dsp/augment.py`，你才能跑 T1/T2；他没交就先做任务一、二、四 |
| 周四晚 | 四张 csv 全部落盘（这是关键路径，报告和周报都等它） |
| 周五 | 与 A 交叉复核：他查你的四张表数值与口径，你查他的两张图 |

## 6 验收（周日逐条打勾）

- [ ] `exp2_freq_ablation.csv` ≥6 行，每行有识别率 + 正确条数/50 + 耗时
- [ ] `exp4_timedomain_vs_freq.csv` 4 行齐全，表注三条前提写全
- [ ] `exp5_robustness.csv` 噪声 15 行 + 增益 6 行，`seed` 列有值
- [ ] `exp3_template_curve.csv` 5 行，与旧表 1/3/5 对齐
- [ ] 四张表都能用**一条命令**重跑（把命令写在 csv 同名 `.txt` 或脚本头部注释里）
- [ ] 每条结论都用"条数"说话（1 条 = 2 个百分点），差 ≤1 条写"无差别"
- [ ] 提交信息前缀 `exp5:` / `exp3:`，正文带关键数字

## 7 你的接口边界

- 只改/新建：`scripts/exp2_freq_ablation.py`、`scripts/exp5_robustness.py`、四张 csv。
- 不改：`dsp/*`（A 和 C 的地盘）、`scripts/run_all.py`（C）、`report/实验二三报告.md`（C）。
- 需要新参数就找 C 冻结，别自己往 `mfcc.py` 里塞。

## 8 常见坑

1. 耗时只有单次值、波动可达 ±50% → 报"同一次运行的整组数"，别跨次拼。
2. CMN 下的 0.820 与 none 下的 0.800 只差 1 条 → 只能写"无显著差别"，不能写"CMN 更好"。
3. `0 dB` 那档如果识别率异常高（>0.30），先怀疑噪声没加进去：自查 `10*log10(P_s/P_n)`。
4. 中文方块 → 抄 `_scripts/exp2_ablation.py` 的 `_setup_matplotlib()`。
5. 脚本必须能 `python -m scripts.xxx` 跑（`scripts/__init__.py` 已处理依赖注入）。

## 9 交付

打包 `交付_exp3_师瑞颖.zip`（两个新脚本 + 四张 csv + 自测输出）发群里，我代提交。
提交信息例：`exp5: robustness: 加噪声 5 档 × 3 种子与增益 3 档 × 2 归一化 — 10 dB 掉 2 条、0 dB 随机`。
