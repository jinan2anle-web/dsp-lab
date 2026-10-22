# 数字信号处理实验 · 课程项目仓库

语音识别系列实验的代码、报告与归档。原始要求见 [数字信号处理实验.pdf](数字信号处理实验.pdf)（扫描版，17 页，无文字层，逐条拆解见 [docs/验收清单.md](docs/验收清单.md)）。

## 一、实验与进度

| 实验 | 主题 | 报告 | 状态 |
| --- | --- | --- | --- |
| 实验一 | 时域分析 + 孤立字 0–9 识别 | 单独一份 | 已完成算法（第一周 `task1_w1`、第二周 `task1_w2`），报告未成文 |
| 实验二 | 短时频域分析：自写 FFT + MFCC | **并入实验三报告** | 未开始（MFCC 目前 0 行） |
| 实验三 | DTW 模板匹配 + 必要的界面 | 实验二 + 三合并一份 | 仅有 DTW 基线 0.760；无 MFCC、无界面 |
| 实验四 | 独立于内容的说话人识别 | 选做、不计期末成绩 | 不做（本组只有一位说话人） |

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

验证骨架是否装好：

```powershell
& $py -c "import sys; sys.path.insert(0, '.'); from dsp import config as c; print(len(c.wav_paths()))"
& $py scripts\exp2_mfcc.py      # 实验二（待实现）
& $py scripts\exp3_recognize.py # 实验三（待实现）
```

## 五、Git

本机目前**没有安装 Git**（`git --version` 报找不到命令），且当前环境无外网，装不上去，所以仓库还没有 `git init`。装好后一条命令即可生效：

```powershell
# 1) 安装：https://git-scm.com/download/win  （一路默认即可）
# 2) 重开终端，在项目根目录执行：
powershell -ExecutionPolicy Bypass -File .\tools\init_repo.ps1
```

脚本会自动完成：`git init -b main` → 配置 `core.quotepath=false` → 补 `user.name/email` → 两个提交（①骨架 ②第一二周归档）→ 打印 `git log`。可重复运行，已完成的步骤自动跳过。

日常约定：

- 每个工作单元一次提交，说明写「做了什么 + 为什么」，例如
  `feat(mfcc): 自写 radix-2 FFT 替换 scipy，与 np.fft 对拍最大误差 1.2e-16`。
- 交互式简单操作够用：`git status` / `git diff` / `git add -A` / `git commit -m`。
- 交上去的 zip 一律由 `git archive` 生成，不再手工复制目录。
- OneDrive 与 `.git` 同放偶发 `index.lock` 冲突：遇到时先暂停 OneDrive 同步再操作；若频繁发生，把仓库移出 OneDrive。

## 六、待办（按阻断程度排序）

1. **修端点**：整段 0.760 / 端点段 0.540，差 22 个点。端点把判别信息剪掉了（均值仅 0.274 s），实验三主线不能被它拖累——定稿一种口径，全仓只留一处实现。
2. **实验二**：`dsp/fft_radix2.py`（自写，与 `np.fft` 对拍）+ `dsp/mfcc.py`（预加重 → 分帧加窗 → FFT → 谱能量 → Mel 滤波 → log → DCT）。
3. **实验三**：`dsp/dtw.py` 按教材递推式核对 + `dsp/app.py` 界面 + 端到端一键流程。
4. **报告**：实验一一份、实验二+三合并一份，均按学术论文格式写。

详细模块边界与里程碑见 [docs/实验二三技术路线.md](docs/实验二三技术路线.md)。
