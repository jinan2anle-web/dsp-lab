# -*- coding: utf-8 -*-
"""
周报用的音频数据审核脚本
1) 按 wav_parse.py 的接口解析 task1_w1/A/A/*.wav 全部文件
2) 检查录制格式是否符合接口冻结的硬约束
3) 检查数据质量：直流偏置、削波、幅度、静音段、时长一致性
4) 输出 接口冻结/results/audio_audit.txt
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # task1_w1/接口冻结
BASE = os.path.dirname(HERE)                                          # task1_w1
WAV_DIR = os.path.join(BASE, 'A', 'A')
OUT = os.path.join(HERE, 'results')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join(BASE, 'wav_parse'))
from wav_parse import read_wav


def audit_one(path):
    x, fs, ch, bits = read_wav(path)
    xf = x.astype(np.float64)
    n = len(xf)
    dur = n / fs
    dc = xf.mean()
    peak = np.max(np.abs(xf))
    rms = np.sqrt(np.mean(xf ** 2))
    clipped = int(np.sum(np.abs(xf) >= 32767))
    # 有效语音段（|x| > 1% 满量程）占比
    active = float(np.mean(np.abs(xf) > 0.01 * 32768)) if peak > 0 else 0.0
    # 静音段（低于 0.5% 满量程）占比
    quiet = float(np.mean(np.abs(xf) < 0.005 * 32768))
    return dict(n=n, fs=fs, ch=ch, bits=bits, dur=dur, dc=dc, peak=peak, rms=rms,
                clipped=clipped, active=active, quiet=quiet,
                clip_ratio=clipped / n if n else 0.0,
                crest=peak / rms if rms > 0 else float('inf'))


def main():
    files = sorted([f for f in os.listdir(WAV_DIR) if f.lower().endswith('.wav')])
    rows = {}
    fails = []
    for f in files:
        rows[f] = audit_one(os.path.join(WAV_DIR, f))

    lines = []
    w = lines.append
    w('=' * 78)
    w('录音数据审核报告 —— task1_w1/A（成员A 录制）')
    w('=' * 78)
    w(f'文件总数：{len(files)} 个')

    # ---- 1. 格式一致性 ----
    fs_set = sorted({r["fs"] for r in rows.values()})
    ch_set = sorted({r["ch"] for r in rows.values()})
    bit_set = sorted({r["bits"] for r in rows.values()})
    n_set = sorted({r["n"] for r in rows.values()})
    w('')
    w('【1】格式与接口冻结硬约束的对照')
    w(f'  采样率 fs    : {fs_set}        -> 约定 16000 Hz')
    w(f'  通道数       : {ch_set}            -> 约定 1（单声道）')
    w(f'  量化位深     : {bit_set}           -> 约定 16 bit')
    w(f'  采样点数集合 : {n_set}   -> 即每段时长 {[round(v/16000,3) for v in n_set]} s')
    w(f'  格式检查结论 : ' + ('全部符合冻结约定' if (fs_set == [16000] and ch_set == [1]
                                          and bit_set == [16]) else '存在不符合项'))

    # ---- 2. 时长分布 ----
    durs = np.array([r['dur'] for r in rows.values()])
    w('')
    w('【2】时长分布')
    w(f'  期望：每个数字 10 组，共 100 组；实际 {len(files)} 组')
    w(f'  时长 min={durs.min():.4f}s  max={durs.max():.4f}s  '
      f'mean={durs.mean():.4f}s  std={durs.std():.6f}s')
    w(f'  时长种类数：{len(n_set)} 种'
      f'{"（所有文件时长完全相同）" if len(n_set) == 1 else ""}')

    # ---- 3. 质量指标 ----
    w('')
    w('【3】数据质量指标（全体统计）')
    dc = np.array([abs(r['dc']) for r in rows.values()])
    pk = np.array([r['peak'] for r in rows.values()])
    cr = np.array([r['clip_ratio'] for r in rows.values()])
    ac = np.array([r['active'] for r in rows.values()])
    qt = np.array([r['quiet'] for r in rows.values()])
    w(f'  |直流偏置| : mean={dc.mean():.1f}  max={dc.max():.1f}  (满量程 32768)')
    w(f'  峰值       : mean={pk.mean():.0f}  min={pk.min():.0f}  max={pk.max():.0f}')
    w(f'  削波样点数 : 总计={int(cr.sum()*np.array([r["n"] for r in rows.values()]).mean() if False else sum(r["clipped"] for r in rows.values()))}  '
      f'最高单文件削波占比={cr.max()*100:.4f}%')
    w(f'  有效样点占比(|x|>1%FS) : mean={ac.mean()*100:.1f}%  min={ac.min()*100:.1f}%  max={ac.max()*100:.1f}%')
    w(f'  静音样点占比(<0.5%FS)  : mean={qt.mean()*100:.1f}%  min={qt.min()*100:.1f}%  max={qt.max()*100:.1f}%')
    w(f'  峰均比(crest)          : mean={np.mean([r["crest"] for r in rows.values()]):.2f}')

    # ---- 4. 数字覆盖 ----
    w('')
    w('【4】数字覆盖情况')
    cover = {}
    for f in files:
        d = f.split('_')[0]
        cover.setdefault(d, 0)
        cover[d] += 1
    for d in sorted(cover):
        flag = 'OK' if cover[d] == 10 else f'数量异常({cover[d]})'
        w(f'  数字 {d} : {cover[d]} 组  {flag}')
    w(f'  覆盖数字种类：{len(cover)} 种 -> ' +
      ('覆盖 0~9 全部数字' if len(cover) == 10 else '未覆盖 0~9'))

    # ---- 5. 逐文件明细 ----
    w('')
    w('【5】逐文件明细（峰值/直流/削波/有效占比）')
    w('  文件\t采样点\t时长(s)\t峰值\t直流\t削波\t有效占比')
    for f in files:
        r = rows[f]
        w(f'  {f}\t{r["n"]}\t{r["dur"]:.3f}\t{r["peak"]:.0f}\t{r["dc"]:.1f}\t'
          f'{r["clipped"]}\t{r["active"]*100:.1f}%')

    txt = '\n'.join(lines)
    with open(os.path.join(OUT, 'audio_audit.txt'), 'w', encoding='utf-8') as fp:
        fp.write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
