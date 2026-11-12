# 本周任务单 · A 梁博强（`exp2:` 前缀）

> 第六周（11/09–11/15）。本周老师要求：「继续完成实验二、三的实验内容，并开始设计相关测试实验」。
> 你负责实验二那条线，三件必须做完的：**两张新图**、**一个特征变体**、**一个扰动工具**。

## 0 先做这三步（10 分钟）

```bash
python -m scripts.selftest           # 看到「通过」（只要有 numpy 就能跑）
python -m scripts.exp3_figs --help   # 可选：出图步骤需要 matplotlib，装不上看 README 末节兜底
```

跑完先看 `report/fig/fig2_3_mfcc_flow.png`——你这周要出的两张图，风格尽量跟它一致（标题 11pt 左右、网格淡灰、300 dpi）。

## 1 任务一：`scripts/exp2_freq_figs.py`（出图 2.4 / 2.5）

**图 2.4 语谱图** `report/fig/fig2_4_spectrogram.png`

- 输入：`data/A/0_01.wav`（1.5 s，前 0.35 s 是数字静音）。
- 内容：`[148, 257]` 的谱能量矩阵（帧 × 频点），`imshow` + 对数色标（`dB` 或 `log`），横轴时间 / 秒（0–1.5），纵轴频率 / Hz（0–8000）。
- 必须在图上标出：**静音前导区（0–0.35 s 用阴影或竖线标出）**、语音起点、语音终点。
- 验收：一眼能看出"0.35 s 之前几乎没有能量，之后才有横条纹"。

**图 2.5 功率谱对比** `report/fig/fig2_5_power_spectrum.png`

- 取**同一条录音的三个时刻**：静音帧（约第 5 帧）、语音起始帧（约第 40 帧）、语音中段帧（约第 95 帧）。
- 每帧画 `|X[k]|²`（0–8000 Hz，对数纵轴），三条线画在同一张图上或做三个子图，标注帧号与时间。
- 必须在图上体现：**预加重把高频抬起来**——画"预加重前 / 预加重后"两组曲线（可以每帧两条线，或 2×3 子图）。
- 验收：能指出同一帧预加重前后在 4–8 kHz 的差。

**写法建议**：直接抄 `scripts/exp2_ablation.py` 的骨架（`_setup_matplotlib()` **只负责挑中文字体**，字号/配色要自己设；现有三处 `savefig` 都是 `dpi=300`、**没有** `bbox_inches` 参数）。跑法：`python -m scripts.exp2_freq_figs`（`scripts/__init__.py` 会自动注入依赖路径）；**不要改 `scripts/run_all.py`**，集成由 C 做。

自测输出写到 `report/evidence/exp2_freq_figs.txt`：图的两条尺寸、三个帧号、每帧能量最大值。

## 2 任务二：`dsp/mfcc.py` 加 `form="logmel"`（只加可选参数）

```python
extract(x, fs=16000.0, *, n_mfcc=13, n_mels=24, nfft=512, win_len=400, hop_len=160,
        window="hamming", preemph=0.97, f_low=0.0, f_high=None,
        normalize="none", form="mfcc", use_delta=False)
```

- `form="mfcc"`（默认，不动）→ `[T, 13]`。
- `form="logmel"` → `[T, 24]`，即第 6 步"Mel 滤波 + 取对数"之后、DCT 之前的结果。
- `use_delta=True` 且 `form="mfcc"` → `[T, 39]`，顺序固定：静态 13 在前、Δ 13 居中、ΔΔ 13 在后。
- **不许改默认值和位置参数**，否则全仓 20 多处调用全炸。

自测单独写一个脚本 `scripts/exp2_selftest_extra.py`（**允许你新建的第 4 个文件**，不要改公共的 `scripts/selftest.py`），断言：

- `extract(x, form="logmel").shape == (148, 24)`
- `extract(x, use_delta=True).shape == (148, 39)`
- 默认调用与改动前一致：先 `np.savez` 到临时目录存一份现值、再改代码、再读回，`np.array_equal` 必须为 True（**不要依赖把 npy 入库**，`*.npy` 是 gitignore 的）
- Δ/ΔΔ 的口径（**本周新定，照这个实现**）：`Δ[t] = c[t] − c[t−1]`，`ΔΔ[t] = Δ[t] − Δ[t−1]`；`t=0` 时用前向差分补齐（`c[1]−c[0]`），保证输出仍 `[T,13]`；拼接顺序 `[静态13 | Δ13 | ΔΔ13]`。自测里贴一组数值：`x=0_01.wav` 的 `c[0,0]`、`Δ[0,0]`、`ΔΔ[0,0]`。

## 3 任务三：`dsp/augment.py`（新文件，B 要用它跑测试实验）

```python
add_white_noise(x, snr_db, *, fs=16000, seed=0) -> np.ndarray   # 按语音段功率定噪声功率
apply_gain(x, factor)                           -> np.ndarray   # 整条乘常数（含静音段）
```

- `add_white_noise`：先用 `dsp.endpoint.detect()` 取 `[s0, s1)`，算语音段均方值 `P_s`，噪声方差 `σ² = P_s / 10**(snr_db/10)`；`np.random.default_rng(seed).normal(0, sqrt(σ²), len(x))` 后**整条**相加（静音段也加，更接近真实）。
- `snr_db=float("inf")` 时原样返回。
- `apply_gain`：`x * factor`，长度不变。
- 自测（写 `report/evidence/exp2_augment.txt`）：实测 SNR 与设定值的偏差（用 `10*log10(P_s/P_n)` 复算，20/10/5 dB 三档各一条，偏差必须 < 0.5 dB）；`apply_gain` 前后 `np.abs(x).mean()` 的比值等于 `factor`。

- **`detect()` 返回 None 时的口径**（`dsp/endpoint.py:47` 有 4 条 `return None`）：退化为「整条均方值」当 `P_s`，在 txt 里记一行 `fallback_to_whole_file = N 条`；100 条里正常应为 0 次。
- 噪声功率公式写进 docstring，B 会照抄核对。

## 4 验收（周日逐条打勾）

- [ ] 图 2.4 有静音区/起点/终点标注，横轴秒、纵轴 Hz，300 dpi
- [ ] 图 2.5 三个时刻 + 预加重前后对比，能指出 4–8 kHz 的抬升
- [ ] `extract(form="logmel")` → `[148,24]`；`use_delta=True` → `[148,39]`；默认输出不变
- [ ] `add_white_noise` 实测 SNR 偏差 < 0.5 dB（三档）
- [ ] `apply_gain` 前后均值比等于 factor
- [ ] 四个自测 txt 都在 `report/evidence/`：`exp2_freq_figs.txt`、`exp2_mfcc_logmel.txt`、`exp2_augment.txt`、`exp2_selftest_extra.txt`
- [ ] 提交信息前缀是 `exp2:`，正文带关键数字

## 5 你的接口边界（别越界）

- 只改：`scripts/exp2_freq_figs.py`（新建）、`dsp/augment.py`（新建）、`dsp/mfcc.py`（只加可选参数）。
- 不改：`dsp/config.py`、`scripts/run_all.py`、`report/实验二三报告.md`（这三样是 C 的）。
- 需要 `extract` 之外的参数就找 C 冻结，不要自己在 `mfcc.py` 里加默认值。

## 6 常见坑（与公共包第 4 节同一张表，这里只列你会踩的）

1. 中文方块 → 抄 `_setup_matplotlib()`。
2. 三反引号脚本用 PowerShell 拼会炸 → 用编辑器写 .py。
3. 图里写 `**粗体**` 会原样画出星号（matplotlib 不认 markdown）→ 要粗体用 `fontweight="bold"`。
4. 语谱图的色标范围不同图不一样 → 固定 `vmin`/`vmax`，在 `report/evidence/exp2_freq_figs.txt` 里记下来。

## 7 交付

打包 `交付_exp2_梁博强.zip`（含两个新 py、一个改过的 `mfcc.py`、4 个证据 txt、两张 PNG）发群里，或直接把这几个文件丢给我代提交。
提交信息例：`exp2: figs: 加语谱图与三时刻功率谱 — 静音前导 0.35 s 可见，预加重后 4–8 kHz 抬升 6 dB`。
