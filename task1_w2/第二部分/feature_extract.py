from pathlib import Path
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import pickle
from scipy.fft import fft

# =========路径配置=========
HERE = Path(__file__).resolve()
ROOT = HERE.parent
INTERFACE_PATH = ROOT / "../第一部分/接口冻结_1"
sys.path.insert(0, str(INTERFACE_PATH))

from speech_time_domain import read_wav, remove_dc, time_domain_feature, enframe

FS = 16000
FRAME_DURATION = 0.025
HOP_DURATION = 0.010
frame_len = int(FS * FRAME_DURATION)

# --------------------------
# 特征计算固定只用汉明窗
# --------------------------
FEAT_WIN = "hamming"
color_map = {"rect": "#e74c3c", "hamming": "#2980b9", "hanning": "#27ae60"}
label_map = {"rect": "矩形窗", "hamming": "汉明窗", "hanning": "汉宁窗"}

# 输出路径
wav_dir = ROOT / "../第一部分/第一部分/A"
endpoint_csv = ROOT / "../第一部分/第一部分/数据质检与端点.csv"
out_feature_csv = ROOT / "time_domain_feature_table.csv"
out_fig_sample = ROOT / "sample_feature_curve.png"
out_pkl = ROOT / "frame_sequences.pkl"

NFFT = 4096          # 窗函数幅频图 FFT 点数（零填充，让谱平滑）


def time2sample(time_sec: float, fs: int):
    """时间秒 → 采样点下标"""
    return int(np.round(time_sec * fs))


def summarize_feature_sequence(seq: np.ndarray):
    """16维统计汇总特征
    输入标准化后[T,3]特征序列，输出16维向量
    mean(3),std(3),max(3),q25(3),q75(3),duration(1) → 3*5+1=16
    """
    feat_mean = np.mean(seq, axis=0)
    feat_std = np.std(seq, axis=0)
    feat_max = np.max(seq, axis=0)
    feat_q25 = np.quantile(seq, 0.25, axis=0)
    feat_q75 = np.quantile(seq, 0.75, axis=0)
    duration = seq.shape[0] * HOP_DURATION
    vec_16 = np.concatenate([feat_mean, feat_std, feat_max, feat_q25, feat_q75, np.array([duration])])
    return vec_16


def plot_single_window(w, title_name, save_path):

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7))

    # ---- 时域波形 ----
    n_axis = np.arange(len(w))
    ax1.plot(n_axis, w, linewidth=1.2, color="#2980b9")
    ax1.set_title(f"{title_name}时域波形")
    ax1.set_xlabel("样点数")
    ax1.set_ylabel("幅度")
    ax1.grid(alpha=0.3)

    # ---- 幅频特性（零填充 + 只画正频率一半）----
    W = fft(w, NFFT)
    mag = np.abs(W)
    mag_db = 20 * np.log10(mag / mag.max() + 1e-12)
    half = NFFT // 2 + 1
    f_norm = np.linspace(0, 0.5, half)

    ax2.plot(f_norm, mag_db[:half], linewidth=1.0, color="#2980b9")
    ax2.set_title(f"{title_name}幅度特性")
    ax2.set_xlabel("归一化频率")
    ax2.set_ylabel("幅度(dB)")
    ax2.set_xlim(0, 0.5)
    ax2.set_ylim(-150, 5)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存：{save_path.resolve()}")


def plot_sample_curve(signal, fs, save_path):
    """绘制典型样本时域特征曲线"""
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(4, 1, figsize=(14,10), sharex=True)
    t_wave = np.arange(len(signal)) / fs
    axes[0].plot(t_wave, signal, c="#222222", linewidth=0.8)
    axes[0].set_title("有效语音波形")
    axes[0].set_ylabel("幅值")
    axes[0].grid(alpha=0.3)

    feat_seq = time_domain_feature(signal, fs, frame_ms=25.0, hop_ms=10.0, win=FEAT_WIN)
    t_frame = np.arange(feat_seq.shape[0]) * HOP_DURATION

    axes[1].plot(t_frame, feat_seq[:,0], color=color_map[FEAT_WIN])
    axes[2].plot(t_frame, feat_seq[:,1], color=color_map[FEAT_WIN])
    axes[3].plot(t_frame, feat_seq[:,2], color=color_map[FEAT_WIN])

    axes[1].set_title("对数短时能量（汉明窗）")
    axes[1].set_ylabel("log能量")
    axes[1].grid(alpha=0.3)

    axes[2].set_title("对数短时幅度和（汉明窗）")
    axes[2].set_ylabel("log幅度和")
    axes[2].grid(alpha=0.3)

    axes[3].set_title("归一化过零率（汉明窗）")
    axes[3].set_xlabel("时间(s)")
    axes[3].set_ylabel("ZCR")
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"样本特征曲线保存：{save_path.resolve()}")


def main():
    # =====  画三张窗函数图=====
    dummy = np.zeros(frame_len)
    for win_name, cn_name in [("rect", "矩形窗"),
                              ("hamming", "汉明窗"),
                              ("hanning", "汉宁窗")]:
        _, w = enframe(dummy, frame_len=frame_len, hop=frame_len, win=win_name)
        plot_single_window(w, cn_name, ROOT / f"{cn_name}_时域幅频.png")

    # ===== 读端点结果 =====
    df_endpoint = pd.read_csv(endpoint_csv, encoding="utf-8-sig")
   

    table_rows = []
    all_frame_sequences = {}

    # ===== 取第一条样本绘制时域特征曲线 =====
    sample_row = df_endpoint.iloc[0]
    sig_raw, fs, _, _ = read_wav(wav_dir / sample_row["file"])
    sig_raw = remove_dc(sig_raw)
    s0 = time2sample(sample_row["endpoint_start_s"], fs)
    s1 = time2sample(sample_row["endpoint_end_s"], fs)
    valid_speech = sig_raw[s0:s1]
    plot_sample_curve(valid_speech, fs, out_fig_sample)

    # ===== 逐条提取特征=====
    for _, row_info in df_endpoint.iterrows():
        filename = row_info["file"]
        start_s = row_info["endpoint_start_s"]
        end_s = row_info["endpoint_end_s"]

        sig, fs, _, _ = read_wav(wav_dir / filename)
        sig = remove_dc(sig)

        idx_start = time2sample(start_s, fs)
        idx_end = time2sample(end_s, fs)
        valid_speech = sig[idx_start:idx_end]

        # =========提取时域特征=========
        frame_seq = time_domain_feature(valid_speech, fs, frame_ms=25.0, hop_ms=10.0, win=FEAT_WIN)
        feat16 = summarize_feature_sequence(frame_seq)

        # 从原始文件名提取label，0_01.wav → label=0
        label = int(filename.split("_")[0])
        row_out = {"filename": filename, "label": label}
        for idx, val in enumerate(feat16):
            row_out[f"dim{idx}"] = val

        all_frame_sequences[filename] = frame_seq
        table_rows.append(row_out)
        print(f"已处理：{filename}")

    # 输出16维汇总csv
    df_out = pd.DataFrame(table_rows)
    df_out.to_csv(out_feature_csv, index=False, encoding="utf-8-sig")
    print(f"\n特征汇总表输出：{out_feature_csv.resolve()}")

    # 输出逐帧序列pkl，直接供给DTW/KNN/SVM分类器
    with open(out_pkl, "wb") as f:
        pickle.dump(all_frame_sequences, f)
    print(f"逐帧时序特征已保存：{out_pkl.resolve()}")


if __name__ == "__main__":
    main()