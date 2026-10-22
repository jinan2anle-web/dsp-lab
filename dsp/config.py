"""全局常量与路径。只放事实，不放逻辑。

所有实验共用的采样率、帧长、MFCC 参数都从这里取，避免各脚本各写一套。
"""

from pathlib import Path

# ---------- 路径 ----------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "A"          # 原始录音唯一真源
OUT_DIR = REPO_ROOT / "out"                  # 派生产物，不进 git
DOCS_DIR = REPO_ROOT / "docs"
REPORT_DIR = REPO_ROOT / "report"

# ---------- 录音固有属性（100 条录音一致）----------
FS = 16000          # 采样率 Hz
CHANNELS = 1        # 单声道
SAMPLE_WIDTH = 2    # 16 bit PCM
DURATION_S = 1.5

# ---------- 分帧 ----------
FRAME_MS = 25.0
HOP_MS = 10.0
WIN_LEN = int(FS * FRAME_MS / 1000)   # 400 点
HOP_LEN = int(FS * HOP_MS / 1000)     # 160 点
NFFT = 512                            # 要求 2^L，400 补零到 512

# ---------- 频域 / MFCC（实验二）----------
PREEMPH = 0.97      # 预加重 H(z) = 1 - 0.97 z^-1
N_MELS = 24         # Mel 三角滤波器个数 M
N_MFCC = 13         # 倒谱阶数，课程要求 12~16
F_LOW = 0.0         # 滤波器组最低频率 Hz
F_HIGH = FS / 2     # 滤波器组最高频率 Hz

# ---------- 端点检测（实验一/三共用的唯一口径）----------
MIN_LEN_MS = 100    # 端点段最短长度，短于此值向两侧补长

# ---------- 标签 ----------
LABELS = [str(i) for i in range(10)]


def wav_paths():
    """按文件名排序返回 100 条原始录音的路径。"""
    return sorted(DATA_DIR.glob("*.wav"))


def label_of(path):
    """从 `3_07.wav` 这样的文件名取类别整数 3。"""
    return int(Path(path).stem.split("_")[0])


def ensure_out_dir():
    """确保派生产物目录存在，返回其路径。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR
