# 先读我（公共基础包）

> 这个包是**自包含**的：代码、数据、图、表、文档都在里面，**不需要访问 GitHub**，也不需要联网装包（只要本机有 numpy/scipy/matplotlib，见第 2 节）。

## 1 包里有什么

```
dsp/                8 个算法模块（io_wav framing fft_radix2 mfcc endpoint dtw templates app）
scripts/            5 个脚本（selftest exp2_ablation exp3_recognize exp3_figs run_all）
data/A/             100 条测试用录音（10 个数字 × 10 条，16 kHz / 单声道 / 16 bit / 1.5 s）
docs/               技术路线、内容讲解、零基础解释、验收清单、测试实验设计、第六周任务与分工
report/fig/         已有 6 张图
report/tables/      已有 8 张 csv（所有数字的出处）
report/evidence/    自测输出的 txt（数字的原始证据）
report/实验二三报告.md   合并报告正文（14 页 docx 的源文件）
requirements.txt    依赖版本
```

## 2 环境（三步）

```bash
python -V                      # 需要 3.10+
pip install -r requirements.txt
python -m scripts.selftest     # 看到「通过」才算环境对
```

- 如果你本机**已经装过** numpy/scipy/matplotlib，`pip install` 可以直接跳过。
- 报 `ModuleNotFoundError: No module named 'matplotlib'` 说明只装了 numpy；回到上一步装齐即可。
- 本仓自带的 `dsp/env.py: add_pylibs()` 是给"没有 pip 权限、只能用内置库目录"的机器准备的，你不需要调它。

## 3 先跑通这套基线（两条命令）

```bash
python -m scripts.selftest                 # 约 1 秒，输出 5 个 FFT 用例误差 + 滤波器组断言
python -m scripts.run_all                  # 约 4 分钟，重算全部数字与图
```

`run_all` 的产物直接覆盖 `report/fig`、`report/tables`、`report/evidence`。**跑之前先备份一份，方便对比你改动前后的差异。**

基线数字（跑完应完全一致）：

| 项 | 值 |
|---|---|
| FFT 最大对拍误差 | 5.86×10⁻¹⁴（批量 `[148,512]` 为 7.94×10⁻¹⁵） |
| 每类 Mel 权重和 | k=3…230 严格为 1（228 个频点），两端 0 |
| 帧数 | 1.5 s → 148 帧 |
| 识别主结果 | 40/50 = 0.800 |
| 模板 1/3/5 条 | 0.600 / 0.740 / 0.800 |
| CMN | 0.820 |
| 端点段 | 0.800，耗时 28.4 s → 7.5 s |

## 4 已知的坑（我们全都踩过）

| 现象 | 原因 | 解法 |
|---|---|---|
| 控制台 `UnicodeEncodeError: 'gbk' codec can't encode character '\u2212'` | Windows 控制台 GBK 编不了减号 `−` | 脚本里已加 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`；自己写脚本时照抄这一行 |
| 图里中文全是方块 | matplotlib 默认字体没有中文 | 抄 `scripts/exp2_ablation.py: _setup_matplotlib()`，它会挑 Microsoft YaHei / SimHei |
| 路径里有中文导致某个库报错 | 个别老版本库对非 ASCII 路径敏感 | 尽量在仓库根目录运行；报错时把路径贴给我，别自己改目录名 |
| `out/` 里的东西 git 里找不到 | `.gitignore` 忽略 `out/`、`*.npy`、`*.zip`、`_env/` | 交付物放 `report/`；临时文件放 `out/` |
| `PermissionError` 写不进 pptx/docx | 文件正被 PowerPoint/Word 打开 | 先关掉 Office 再跑脚本 |
| 用 PowerShell 写含三反引号的脚本报 `SyntaxError` | 双引号 here-string 会把反引号当转义 | 用编辑器直接写 .py 文件，别用 `python -c "…"` 拼长代码 |
| 耗时数字每次不一样 | 同机负载波动可达 ±50% | 报"同一次运行的一组数"，不要跨次比较 |

## 5 三条纪律

1. **不改别人的文件**：`dsp/mfcc.py` 的签名、`scripts/run_all.py` 由肖雨乐（C）统一改；发现 bug 在群里说，别自己动手。
2. **每个数字可复跑**：交付的同时给一条能重跑的命令；跑不出来就写"未测"，不许估。
3. **不动历史目录**：`task1_w1/`、`task1_w2/`、`task1_w3/` 是归档，改了旧数字就对不上。

## 6 怎么交付（组员 → 我）

1. 只交**自己那几行文件** + 自测输出 txt + 截图/图，不要传整个仓库。
2. 打包成 `交付_exp2_梁博强.zip` / `交付_exp3_师瑞颖.zip` / `交付_app_肖雨乐.zip` 发群里。
3. 我收到后用 `git commit --author="你的名字 <你的邮箱>"` 代提交，`git log` 里作者仍是你。
4. **提交信息格式**：`<前缀>: <模块>: <做了什么> — <关键数字>`，例：`exp2: mfcc: 加 form="logmel" 分支 — extract 返回 [148,24]`。
