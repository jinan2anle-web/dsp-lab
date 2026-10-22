# -*- coding: utf-8 -*-
"""
实验1：基于时域分析技术的语音识别 —— 数据生成与算法功能验证

说明：工作区内没有真实录音文件，本脚本用"准周期脉冲源 + 随机噪声源 + 时变线性系统"
（参考 PDF 图 4.2 的语音产生模型）合成语音样本驱动全部算法，用于验证程序正确性并
产出报告插图。真实实验时只需把 data/ 目录换成录音文件，接口完全一致。

输出：data/*.wav、figures/*.png、results/summary.txt
"""
import os
import numpy as np

import speech_time_domain as std
import plotlite as pl
from synth_speech import synth_digit, synth_utterance, decimate, FS

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA, FIG, RES = (os.path.join(ROOT, x) for x in ('data', 'figures', 'results'))
for d in (DATA, FIG, RES):
    os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(20240607)
LOG = []


def log(s=''):
    LOG.append(s)
    print(s)


# ======================================================================
# 图1：整段"0 1 2"波形
# ======================================================================
x, marks = synth_utterance(('0', '1', '2'), speaker=1)
std.write_wav(os.path.join(DATA, 'demo_012.wav'), x, FS)
xs, ys = decimate(x)
fig = pl.Figure(880, 300)
p = fig.panel((118, 55, 840, 225), (0, len(x) / FS), (-0.95, 0.95),
              xlabel='时间 / s', ylabel='幅值',
              title='图1 模拟录音"0 1 2"的语音波形（16 kHz / 16 bit / 单声道）')
p.draw_axes()
p.line(xs, ys, color=pl.PALETTE[0], lw=0.9)
for d, t0, t1 in marks:
    p.vline(t0, (214, 39, 40), 0.9)
    p.vline(t1, (214, 39, 40), 0.9)
    p.text_at((t0 + t1) / 2, 0.72, f'"{d}"', 15, (214, 39, 40))
fig.save(os.path.join(FIG, 'fig1_waveform.png'))

# ======================================================================
# 图2：三种窗函数的时域波形与幅频响应
# ======================================================================
N = 256
idx = np.arange(N)
ws = {'矩形窗': np.ones(N),
      '汉明窗': 0.54 - 0.46 * np.cos(2 * np.pi * idx / (N - 1)),
      '海宁窗': 0.5 * (1 - np.cos(2 * np.pi * idx / (N - 1)))}
fig = pl.Figure(880, 580)
pa = fig.panel((118, 60, 840, 260), (0, N), (0, 1.08), xlabel='样点 n', ylabel='w(n)',
               title='图2 三种窗函数：时域波形(a)与归一化幅频响应(b)', yfmt='{:.1f}')
pa.draw_axes()
for k, v in ws.items():
    pa.line(idx, v, lw=1.4, label=k)
pa.draw_legend('upper left')
pb = fig.panel((118, 375, 840, 525), (0, 0.1), (-90, 5), xlabel='归一化频率 f/fs',
               ylabel='幅度 / dB', yfmt='{:.0f}')
pb.draw_axes()
for k, v in ws.items():
    W = np.abs(np.fft.rfft(v, 4096))
    W /= W.max()
    f = np.arange(len(W)) / 4096
    m = f <= 0.1
    pb.line(f[m], 20 * np.log10(W[m] + 1e-12), lw=1.3, label=k)
pb.draw_legend('upper right')
fig.save(os.path.join(FIG, 'fig2_window.png'))

# ======================================================================
# 图3：短时能量/平均幅度/过零率（三种窗）
# ======================================================================
y = synth_digit('0', speaker=1, seed=23)
col = {'rect': pl.PALETTE[1], 'hamming': pl.PALETTE[0], 'hanning': pl.PALETTE[2]}
feat = {}
for win in ('rect', 'hamming', 'hanning'):
    fr, _ = std.enframe(std.remove_dc(y), int(0.025 * FS), int(0.010 * FS), win)
    feat[win] = (std.short_time_energy(fr), std.short_time_amplitude(fr),
                 std.short_time_zcr_rate(fr))
fig = pl.Figure(880, 740)
titles = ['(a) 短时能量 En', '(b) 短时平均幅度 Mn', '(c) 归一化短时过零率 Zn']
for j in range(3):
    vmax = max(feat[w][j].max() for w in feat)
    top = 55 + j * 225
    pp = fig.panel((118, top, 840, top + 165), (0, 0.55), (0, pl.nice_ceil(vmax * 1.1)),
                   ylabel=titles[j], xlabel='时间 / s',
                   title=titles[j])
    pp.draw_axes()
    tt = np.arange(len(feat['rect'][j])) * 0.010
    for win in ('rect', 'hamming', 'hanning'):
        pp.line(tt, feat[win][j], color=col[win], lw=1.4, label=win)
    pp.draw_legend('upper right', dy=26)
fig.save(os.path.join(FIG, 'fig3_features.png'))

log('== 表1 不同窗型下短时域特征对比（"0" 单字，帧长25 ms / 帧移10 ms） ==')
log('窗型\tE_max\tE_mean\tM_max\tZ_max(归一化)')
Emax = {}
for win in ('rect', 'hamming', 'hanning'):
    E, M, Z = feat[win]
    Emax[win] = E.max()
    log(f'{win}\t{E.max():.5f}\t{E.mean():.5f}\t{M.max():.4f}\t{Z.max():.4f}')
log(f'汉明窗相对矩形窗的能量峰值偏差：{(Emax["hamming"]/Emax["rect"]-1)*100:+.2f}%')
log(f'海宁窗相对矩形窗的能量峰值偏差：{(Emax["hanning"]/Emax["rect"]-1)*100:+.2f}%')

# ======================================================================
# 图4：帧长影响
# ======================================================================
fig = pl.Figure(880, 400)
_emax = max(std.short_time_energy(std.enframe(std.remove_dc(y), int(L / 1000 * FS),
                                              int(0.010 * FS), 'hamming')[0]).max()
            for L in (10, 25, 50))
p = fig.panel((118, 60, 840, 330), (0, 0.6), (0, pl.nice_ceil(_emax * 1.08)),
              xlabel='时间 / s', ylabel='短时能量 En',
              title='图4 帧长对短时能量曲线的影响（汉明窗，帧移10 ms）')
p.draw_axes()
for L, c in ((10, pl.PALETTE[4]), (25, pl.PALETTE[0]), (50, pl.PALETTE[3])):
    fr, _ = std.enframe(std.remove_dc(y), int(L / 1000 * FS), int(0.010 * FS), 'hamming')
    p.line(np.arange(len(fr)) * 0.010, std.short_time_energy(fr), color=c, lw=1.5, label=f'帧长 {L} ms')
p.draw_legend('upper right')
fig.save(os.path.join(FIG, 'fig4_framelen.png'))

# ======================================================================
# 图5：双门限端点检测
# ======================================================================
det = std.endpoint_detect(x, FS, win='hamming')
xs2, ys2 = decimate(x)
fig = pl.Figure(880, 720)
p0 = fig.panel((118, 55, 840, 235), (0, len(x) / FS), (-0.95, 0.95), ylabel='幅值',
               title='图5 双门限法端点检测全过程（语音"0 1 2"）')
p0.draw_axes()
p0.line(xs2, ys2, color=pl.PALETTE[0], lw=0.9)
tf = np.arange(len(det['E'])) * 0.010
p1 = fig.panel((118, 340, 840, 490), (0, len(x) / FS), (0, pl.nice_ceil(det['E'].max() * 1.1)),
               ylabel='短时能量 En')
p1.draw_axes()
p1.line(tf, det['E'], color=pl.PALETTE[0], lw=1.4, label='短时能量 En')
p1.hline(det['T2'], (214, 39, 40), 1.1)
p1.hline(det['T1'], (255, 127, 14), 1.1)
p1.legend = [('短时能量 En', pl.PALETTE[0]), ('高门限 T2', (214, 39, 40)),
             ('低门限 T1', (255, 127, 14))]
p1.draw_legend('upper right')
for nm, i in (('N3', det['N3']), ('N4', det['N4']), ('N2', det['N2']), ('N5', det['N5'])):
    p1.text_at(i * 0.010, det['E'][i] * 1.06, nm, 14, (30, 30, 30))
p2 = fig.panel((118, 580, 840, 690), (0, len(x) / FS), (0, pl.nice_ceil(det['Z'].max() * 1.1)),
               ylabel='过零率 Zn', xlabel='时间 / s')
p2.draw_axes()
p2.line(tf, det['Z'], color=pl.PALETTE[2], lw=1.3, label='归一化过零率 Zn')
p2.hline(det['T3'], (200, 60, 160), 1.1)
p2.legend = [('归一化过零率 Zn', pl.PALETTE[2]), ('过零率门限 T3', (200, 60, 160))]
p2.draw_legend('upper right')
for pp in (p0, p1, p2):
    for d, t0, t1 in marks:
        pp.vline(t0, (214, 39, 40), 0.8)
        pp.vline(t1, (214, 39, 40), 0.8)
fig.save(os.path.join(FIG, 'fig5_endpoint.png'))

log()
log('== 表2 双门限法端点检测结果 ==')
log(f'门限：T2 = {det["T2"]:.5f}（高门限），T1 = {det["T1"]:.5f}（低门限），T3 = {det["T3"]:.5f}（过零率）')
log(f'帧序号：N1={det["N1"]}  N2={det["N2"]}  N3={det["N3"]}  N4={det["N4"]}  N5={det["N5"]}  N6={det["N6"]}')
log(f'检测到的语音段：{det["start"]*1000:.0f} ms ~ {det["end"]*1000:.0f} ms'
    f'（时长 {(det["end"]-det["start"])*1000:.0f} ms）')
for d, t0, t1 in marks:
    log(f'  人工标注 数字"{d}"：{t0*1000:.0f} ~ {t1*1000:.0f} ms')
log(f'首字起点绝对误差 {abs(det["start"]-marks[0][1])*1000:.0f} ms，'
    f'末字终点绝对误差 {abs(det["end"]-marks[-1][2])*1000:.0f} ms')

# ======================================================================
# 图6：清音/浊音过零率差异
# ======================================================================
sil = np.zeros(int(0.30 * FS))
voi = synth_digit('0', speaker=1, seed=77)
fric = rng.normal(0, 1, int(0.25 * FS))
b = np.exp(-2 * np.pi * 3500 / FS)
fric = np.convolve(fric, [1, -b], 'same')
fric *= np.sin(np.pi * np.linspace(0, 1, len(fric))) ** 0.6
fric = fric / np.max(np.abs(fric)) * 0.25
mix = np.concatenate([sil, fric, np.zeros(int(0.15 * FS)), voi])
mix = mix + rng.normal(0, 1, len(mix)) * 0.004      # 环境本底噪声
fr, _ = std.enframe(std.remove_dc(mix), int(0.025 * FS), int(0.010 * FS), 'hamming')
tm = np.arange(len(fr)) * 0.010
Zc = std.short_time_zcr_rate(fr)
Ec = std.short_time_energy(fr)
Ec = Ec / Ec.max()
fig = pl.Figure(880, 500)
p0 = fig.panel((118, 60, 840, 215), (0, len(mix) / FS), (-0.9, 0.9), ylabel='幅值',
               title='图6 清音（摩擦噪声）与浊音（元音）的过零率差异')
p0.draw_axes()
xsm, ysm = decimate(mix)
p0.line(xsm, ysm, color=(70, 70, 70), lw=0.9)
p0.text_at(0.42, 0.62, '清音段', 15, (200, 60, 160))
p0.text_at(0.95, 0.62, '浊音段', 15, pl.PALETTE[0])
p1 = fig.panel((118, 310, 840, 440), (0, len(mix) / FS), (0, 1.15),
               ylabel='归一化值', xlabel='时间 / s', yfmt='{:.1f}')
p1.draw_axes()
p1.line(tm, Zc / Zc.max(), color=pl.PALETTE[2], lw=1.4, label='归一化过零率')
p1.line(tm, Ec, color=pl.PALETTE[0], lw=1.4, label='归一化短时能量')
p1.draw_legend('upper left')
fig.save(os.path.join(FIG, 'fig6_zcr.png'))

zf = Zc[int(0.35 / 0.01):int(0.53 / 0.01)]
zv = Zc[int(0.80 / 0.01):]
log()
log('== 表3 清音 / 浊音过零率对比 ==')
log(f'清音段平均归一化过零率 = {zf.mean():.4f}，浊音段 = {zv.mean():.4f}，'
    f'比值 = {zf.mean()/zv.mean():.2f}')
log('结论：清音段能量低但过零率显著高于浊音段，验证"过零率可区分清音与浊音"。')

# ======================================================================
# 识别实验数据集：3 人 × 3 类数字 × 10 组，每人每类前 5 组做模板、后 5 组做测试
# ======================================================================
DIGITS = ['0', '1', '2']
N_PER = 10
SAMPLES, LABELS, FNAMES = [], [], []
for spk in range(3):
    for d in DIGITS:
        for k in range(N_PER):
            s, _ = synth_utterance((d,), speaker=spk,
                                   snr_db=float(rng.uniform(12, 26)),
                                   gain=float(rng.uniform(0.45, 0.9)),
                                   seed=1000 * (spk + 1) + 10 * (k + 1) + DIGITS.index(d))
            name = f'digit_{d}_spk{spk+1}_{k+1:02d}.wav'
            std.write_wav(os.path.join(DATA, name), s, FS)
            SAMPLES.append(s)
            LABELS.append(d)
            FNAMES.append(name)
LABELS = np.array(LABELS)
TR = np.array([int(n.split('_')[3].split('.')[0]) <= 5 for n in FNAMES])


def build_features(frame_ms=25.0, win='hamming', use_endpoint=False):
    """
    对全部样本提取时域特征序列。
    use_endpoint=False：对整段录音分帧（孤立字录音中前置/后置静音提供固定的时间参照）；
    use_endpoint=True ：先做双门限端点检测，仅对检测出的语音段分帧。
    两种方式在报告"结果与分析"中做了对比。
    """
    F = []
    for s in SAMPLES:
        seg = std.endpoint_detect(s, FS, frame_ms=frame_ms, win=win)['seg'] \
            if use_endpoint else s
        F.append(std.time_domain_feature(seg, FS, frame_ms=frame_ms, win=win))
    return F


def evaluate(F):
    """用给定特征集训练模板库并测试，返回 (预测结果, 真值)"""
    c = std.DTWClassifier()
    c.fit([F[i] for i in np.where(TR)[0]], LABELS[TR])
    return c.predict([F[i] for i in np.where(~TR)[0]]), LABELS[~TR]


yp, yte = evaluate(build_features())
acc = float(np.mean(yp == yte))
conf = np.zeros((3, 3), dtype=int)
for a_, b_ in zip(yte, yp):
    conf[DIGITS.index(a_), DIGITS.index(b_)] += 1
pl.confusion_figure(os.path.join(FIG, 'fig7_confusion.png'), DIGITS, conf,
                    '图7 时域特征+DTW 识别混淆矩阵（测试集45组）')

log()
log('== 表4 孤立字识别结果（时域特征 + DTW 模板匹配） ==')
log(f'样本总数 {len(LABELS)} 组 = 3 人 × 3 类 × 10 组；模板库 {int(TR.sum())} 组，测试 {int((~TR).sum())} 组')
log('类别\t正确/总数\t查准率\t查全率')
for i, d in enumerate(DIGITS):
    tp = conf[i, i]
    fn = conf[i].sum() - tp
    fp = conf[:, i].sum() - tp
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    log(f'"{d}"\t{tp}/{conf[i].sum()}\t\t{pr:.3f}\t{rc:.3f}')
log(f'总体识别正确率 = {acc*100:.2f}%，误识率 = {(1-acc)*100:.2f}%')

# ---- 参数消融：同一批样本，只改窗型与帧长 ----
ablation = {'矩形窗': [], '汉明窗': [], '海宁窗': []}
for win in ('rect', 'hamming', 'hanning'):
    for L in (20, 25, 30, 40):
        pred_w, gt = evaluate(build_features(frame_ms=L, win=win))
        ablation[{'rect': '矩形窗', 'hamming': '汉明窗', 'hanning': '海宁窗'}[win]].append(
            float(np.mean(pred_w == gt)))
        print(f'  [消融] {win} {L} ms -> {np.mean(pred_w==gt)*100:.2f}%', flush=True)
log()
log('== 表5 窗型与帧长对识别正确率的影响（消融实验，同一批样本） ==')
log('窗型\t20 ms\t25 ms\t30 ms\t40 ms')
for k, v in ablation.items():
    log(f'{k}\t' + '\t'.join(f'{a*100:.2f}%' for a in v))

# ---- 消融：是否先做端点检测（裁掉前后静音）----
pred_trim, gt_trim = evaluate(build_features(use_endpoint=True))
acc_trim = float(np.mean(pred_trim == gt_trim))
log(f'对照：先做端点检测、只对语音段提特征 -> {acc_trim*100:.2f}%')
log(f'      对整段录音提特征（本文最终方案） -> {acc*100:.2f}%')
pl.grouped_bar_figure(os.path.join(FIG, 'fig8_ablation.png'),
                      ['20 ms', '25 ms', '30 ms', '40 ms'], ablation,
                      '图8 窗型与帧长对识别正确率的影响', ylabel='识别正确率',
                      ylim=(0, 1.1), yfmt='{:.2f}')

with open(os.path.join(RES, 'summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG))
print('\n生成图片:', sorted(os.listdir(FIG)))
print('生成音频:', len([f for f in os.listdir(DATA) if f.endswith(".wav")]), '个')
