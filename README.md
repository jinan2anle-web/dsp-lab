# 数字信号处理实验 · 课程项目仓库

语音识别系列实验的代码、报告与归档。原始要求见 [数字信号处理实验.pdf](数字信号处理实验.pdf)（扫描版，17 页，无文字层，逐条拆解见 [docs/验收清单.md](docs/验收清单.md)）。

## 一、实验与进度

| 实验 | 主题 | 报告 | 状态 |
| --- | --- | --- | --- |
| 实验一 | 时域分析 + 孤立字 0–9 识别 | 单独一份 | 已完成：代码（`task1_w1`/`task1_w2`）＋论文 `task1_w3/基于时域分析技术的孤立数字语音识别.docx` |
| 实验二 | 短时频域分析：自写 FFT + MFCC | **并入实验三报告** | 已完成：`dsp/fft_radix2.py`、`framing.py`、`mfcc.py`，与 `np.fft` 对拍最大误差 5.86e-14（五用例最大值；批量 `[148,512]` 为 7.94e-15）；图 2.1/2.2/2.3 与三窗消融已出 |
| 实验三 | DTW 模板匹配 + 必要的界面 | 实验二 + 三合并一份 | 已完成：`dsp/dtw.py`、`templates.py`、`endpoint.py`、`app.py`；识别率 40/50 = 0.800 |
| 实验四 | 独立于内容的说话人识别 | 选做、不计期末成绩 | 不做（本组只有一位说话人） |

合并报告：`report/实验二三报告.docx`（14 页，6 图 9 表；数字出处见 `report/tables/traceability.csv`）。
交接与分工：`docs/实验二三_分工与交接单.docx`，周报：`report/周报/`。

三条容易被忽略的硬要求：

1. 实验二要求「编写其函数，**自行实现 FFT 过程**」——不能用 `import scipy.fft` 或 `np.fft` 顶替。
2. 实验三要求「**包含必要的界面**」「**自动地**完成语音识别的完整过程」——与算法无关，是纯工作量，必须提前排期。
3. 报告须遵循**学术论文的一般格式**（摘要 / 引言 / 方法 / 实验 / 结果与讨论 / 结论 / 参考文献）。

## 二、目录结构

```
数字信号处理/
├─ README.md                  本文件：进度、结构、约定、怎么跑
├─ .gitignore .gitattributes
├─ 数字信号处理实验.pdf        课程要求原件（只读，不要改）
│
├─ data/A/                    ★ 原始录音唯一真源 0_01.wav … 9_10.wav，共 100 条
├─ dsp/                       ★ 主线代码（实验二、三都写这里，唯一真源）
├─ scripts/                   复现入口：一条命令跑完一个实验
├─ report/                    ★ 最终提交的三份报告 + 图
├─ docs/                      验收清单、技术路线、内部文档
├─ out/                       所有派生产物（图、csv、pkl、模板库）——不进 git
├─ tools/                     一次性脚本（初始化 git 等）
├─ task1_w1/                  ← 第一周归档（当周交付的 zip、周报、旧代码）
└─ task1_w2/                  ← 第二周归档（含 _env 运行环境 403 MB）
```

`task1_w1` / `task1_w2` 是**历史归档，只读**：里面是当时交上去的东西，不再修改、也不再被新代码 import。要复用其中的逻辑，**复制进 `dsp/` 再改**，不要跨目录引用——第二周的相对路径失效、复现清单指向队友机器，都是跨目录引用造成的。

## 三、真源约定（唯一副本原则）

| 内容 | 唯一真源 | 说明 |
| --- | --- | --- |
| 原始录音 | `data/A/` | 归档里的 3 份同名副本已用 SHA256 核对，字节完全相同；它们留在磁盘上但被 `.gitignore` 排除，git 里只有一份 |
| 端点 / 分帧 / 特征 / 识别 | `dsp/` | 归档里「第一部分」「第二部分」「第三部分」各写一套的老代码不再演进 |
| 派生产物 | `out/` | 可随时重建，不进 git |
| 报告 | `report/` | 三份最终报告 |

不变量：**同一份逻辑只能存在一处**。发现第二份时，删掉它，或者把它改成 `from dsp.xxx import ...`。

## 四、怎么跑

一条命令复现报告里的全部数字与图（约 2 分钟，需要 `_env/pylibs` 或已装 requirements）：

```powershell
& $py -m scripts.run_all          # 自测 + 实验二消融与图 + 实验三识别与消融 + 图 3.x
& $py -m scripts.run_all --fast   # 跳过消融表，图照出
& $py -m dsp.app                  # 起界面
```

单步：`-m scripts.selftest`、`-m scripts.exp2_ablation`、`-m scripts.exp3_recognize --all`、`-m scripts.exp3_figs`。

统一用仓库自带的 python（3.12.14）：

```powershell
$py = 'C:\Users\jinan2an\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe'
```

它自带 numpy 2.3.5 / pandas 3.0.1 / python-docx / python-pptx / Pillow，**没有 scipy、matplotlib、scikit-learn**。这三个在归档的 `task1_w2/_env/pylibs` 里（scipy 1.16.3 / matplotlib 3.10.6 / sklearn 1.7.2，已实测可加载）。新代码不要自己拼 `sys.path`，统一走一处：

```python
from dsp.env import add_pylibs
add_pylibs()                    # 之后才能 import scipy / matplotlib / sklearn
```

以后若把 `_env` 挪到仓库根目录，只改 `dsp/env.py` 里的候选路径即可，其余代码不动。

**换机器 / 全新环境**：`_env` 不进 git（403 MB），所以 clone 下来的人拿不到这三个库。用仓库根的 `requirements.txt` 装一遍即可：

```powershell
python -m venv .venv; .venv\Scripts\activate; pip install -r requirements.txt
```

装好正式依赖后，`add_pylibs()` 就不再是必需的。

验证骨架是否装好：

```powershell
& $py -c "import sys; sys.path.insert(0, '.'); from dsp import config as c; print(len(c.wav_paths()))"
& $py -m scripts.selftest        # 写 report/evidence/ 三份自测证据
& $py -m scripts.exp3_recognize  # 主结果，末行打印准确率 0.800
```

## 五、Git

**已就绪（2026-10-23）。** 装的是 **PortableGit 2.55.0**（免管理员权限的绿色版，`winget` 需要 UAC 提权装不上，故改用绿色版）：

```
C:\Users\jinan2an\tools\PortableGit\cmd\git.exe    ← 已写入用户级 PATH，新开终端即可用 git
```

仓库已 `git init`（分支 `main`）并推送到 GitHub：

**<https://github.com/jinan2anle-web/dsp-lab>**（Public）

| 提交 | 内容 | 文件数 |
| --- | --- | --- |
| `87fab81` | `chore: 建立仓库骨架与实验二三主线目录` | 11 |
| `017f91f` | `chore: 归档第一、二周成果与原始录音（音频仅保留 data/A 一份）` | 214 |
| `0590db6` | `docs: 更新 Git 状态为已就绪；补 requirements.txt` | 3 |

远端：`origin → https://github.com/jinan2anle-web/dsp-lab.git`

换一台机器要重建，仍可用同一条命令：

```powershell
git init -b main
powershell -ExecutionPolicy Bypass -File .\tools\init_repo.ps1
```

脚本会完成：`git init -b main` → 配置 `core.quotepath=false` → 补 `user.name/email` → 两个提交 → 打印 `git log`。可重复运行，已完成的步骤自动跳过。

日常约定：

- 每个工作单元一次提交，说明写「做了什么 + 为什么」，例如
  `feat(mfcc): 自写 radix-2 FFT 替换 scipy，与 np.fft 对拍最大误差 1.2e-16`。
- 交互式简单操作够用：`git status` / `git diff` / `git add -A` / `git commit -m`。
- 交上去的 zip 一律由 `git archive` 生成，不再手工复制目录。
- OneDrive 与 `.git` 同放偶发 `index.lock` 冲突：遇到时先暂停 OneDrive 同步再操作；若频繁发生，把仓库移出 OneDrive。

## 六、当前状态与下一步

已定稿（2026-11-08）：

1. **实验二**：自写 radix-2 FFT（与 `np.fft` 最大误差 2.3e-14）、分帧 400/160（1.5 s → 148 帧）、24 个 Mel 三角（覆盖范围内逐频点权重和恒为 1、两端为 0）、13 维 MFCC 含 c0；三窗 × 预加重消融全部落在 0.780–0.800。
2. **实验三**：DTW 三条前驱、K=13、越界 +inf；每类 5 条模板取最小距离，测试 50 条 **40/50 = 0.800**；模板数 1/3/5 → 0.600/0.740/0.800；CMN 0.820；端点段总数同为 40/50（其中 12 条判决互换）而判决时间 28.4 → 7.5 s。
3. **报告**：`report/实验二三报告.docx` 14 页 6 图 9 表，每个数字可在 `report/tables/traceability.csv` 查到出处。
4. **界面**：`python -m dsp.app`，左侧 100 条录音、右侧波形 + 端点段 + MFCC 热图 + 各类最近模板距离。

下一步（第五周）：按 `docs/大纲.md` 与 `docs/第五周_工作分配方案.md` 做频域特征扩展实验；实验一论文已收（`task1_w3/`）。

**实验一 0.760 的口径（已在本仓复现）**：用 `task1_w2/第一部分/接口冻结_1/speech_time_domain.py` 的 `time_domain_feature(...)` + `DTWClassifier(band_ratio=0.35)` 在整段 1.5 s 录音上跑 5 条模板，`normalize=True` 得 38/50 = 0.760、`normalize=False` 得 20/50 = 0.400；把约束带放到 1.00，两个结果都不变、只是预测耗时从 15.90 s 升到 27.21 s。可见 0.760 与 0.400 的差距完全由**段内逐维归一化**决定，约束带只影响耗时；引用 0.760 必须同时写明「归一化 + band 0.35」两个前提，不能与本次 0.800 直接比大小。注意别用 `task1_w2/第二部分/frame_sequences.pkl` 做这个对照——那份缓存是「端点段 + 已归一化」的数据，拿它跑只会得 0.400。

详细模块边界与里程碑见 [docs/实验二三技术路线.md](docs/实验二三技术路线.md)。
