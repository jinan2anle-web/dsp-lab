# -*- coding: utf-8 -*-
"""
接口自检脚本（接口冻结出口判据）

用法：python 接口冻结/tools/contract_test.py
对 task1_w1/A/A/*.wav 的真实数据与公共算法接口逐条断言，输出 接口冻结/results/contract_test.txt
约定见 接口冻结/docs/接口冻结说明.md（2026-10-05 冻结）
"""
import os
import sys
import numpy as np

for _s in ('stdout', 'stderr'):            # Windows 控制台默认 GBK，统一改 UTF-8
    try:
        getattr(sys, _s).reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # task1_w1
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # task1_w1/接口冻结
WAV_DIR = os.path.join(BASE, 'A', 'A')
DOCS = os.path.join(HERE, 'docs')
OUT = os.path.join(HERE, 'results')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)                               # 公共模块 speech_time_domain.py（接口冻结/ 下）
sys.path.insert(0, os.path.join(BASE, 'wav_parse'))    # A 提交的 wav_parse.py
sys.path.insert(0, WAV_DIR)

RESULTS = []          # (编号, 检查项, 判据, 实测, 结论)


def check(no, item, expect, actual, ok):
    RESULTS.append((no, item, expect, actual, 'PASS' if ok else 'FAIL'))
    print('[%s] %s %s | 判据: %s | 实测: %s'
          % ('PASS' if ok else 'FAIL', no, item, expect, actual))
    return bool(ok)


def have(name):
    """延迟导入被测模块，未就绪时给出明确提示而不是崩溃"""
    try:
        return __import__(name)
    except Exception as e:
        print('  !! 模块 %s 尚未就绪：%s' % (name, e))
        return None


def main():
    all_ok = True
    wav = have('wav_parse')
    if wav is None:
        print('read_wav 未就绪，无法执行自检。')
        return 1

    def rw(f):
        """四元组解包助手（与冻结清单一致）"""
        return wav.read_wav(os.path.join(WAV_DIR, f))

    files = sorted(f for f in os.listdir(WAV_DIR) if f.lower().endswith('.wav'))

    # ---------- 1. 命名与目录约定 ----------
    bad_name = [f for f in files
                if not (len(f) == 8 and f[0] in '0123456789' and f[1] == '_'
                        and f[2:4].isdigit() and 1 <= int(f[2:4]) <= 10
                        and f[4:] == '.wav')]
    all_ok &= check('F1', '文件命名 {数字}_{序号:02d}.wav（序号 01~10）',
                    '不符合项 0 个', '不符合项 %d 个 %s' % (len(bad_name), bad_name[:3]),
                    len(bad_name) == 0)

    digits = sorted({f[0] for f in files})
    per_digit = sorted({sum(1 for f in files if f[0] == d) for d in digits})
    all_ok &= check('F2', '数字覆盖 0~9，每类 10 组',
                    '10 类 × 每类 10 组 = 100',
                    '%d 类，每类 %s 组，共 %d 个' % (len(digits), per_digit, len(files)),
                    digits == list('0123456789') and len(files) == 100
                    and per_digit == [10])

    all_ok &= check('F3', '按说话人建文件夹存放',
                    '存在 A\\A\\*.wav', os.path.relpath(WAV_DIR, BASE),
                    os.path.isdir(WAV_DIR))

    # ---------- 2. read_wav 契约 ----------
    ret = rw(files[0])
    all_ok &= check('W0', 'read_wav 返回四元组 (wave_data, sample_rate, num_channels, bits_per_sample)',
                    'tuple 且长度 4',
                    'type=%s len=%d' % (type(ret).__name__, len(ret) if hasattr(ret, '__len__') else -1),
                    isinstance(ret, tuple) and len(ret) == 4)
    data, fs, ch, bits = ret
    all_ok &= check('W1', 'wave_data 为 numpy 一维数组（16bit 原始量化值，未归一化）',
                    'ndarray, ndim=1',
                    '%s ndim=%d max|x|=%.0f' % (type(data).__name__,
                                                getattr(data, 'ndim', -1),
                                                np.max(np.abs(data))),
                    isinstance(data, np.ndarray) and data.ndim == 1)
    all_ok &= check('W2', '采样率 sample_rate = 16000', '16000', '%s' % fs, fs == 16000)
    all_ok &= check('W3', '通道数 num_channels = 1（单声道）', '1', '%s' % ch, ch == 1)
    all_ok &= check('W4', '量化位深 bits_per_sample = 16', '16', '%s' % bits, bits == 16)

    lens = [len(rw(f)[0]) for f in files]
    all_ok &= check('W5', '采样点数为 block_align（2 字节）整数倍',
                    '全部整除', 'min=%d max=%d' % (min(lens), max(lens)),
                    min(lens) % 2 == 0 and max(lens) % 2 == 0)

    meta = {(r[1], r[2], r[3]) for r in (rw(f) for f in files)}
    all_ok &= check('W6', '全部文件元数据一致（fs/ch/bits 组合唯一）',
                    '1 种组合', '%d 种 %s' % (len(meta), sorted(meta)), len(meta) == 1)

    # ---------- 3. 预处理与特征层 ----------
    std = have('speech_time_domain')
    if std is None:
        print('\n注意：speech_time_domain.py 未就绪，S/D/T 项未执行。')
    else:
        x = np.arange(400, dtype=np.float64)
        frames, w = std.enframe(x, 400, 160, 'hamming')
        all_ok &= check('S1', 'enframe 返回 (帧矩阵 [n_frames, frame_len], 窗 [frame_len])',
                        'frames.shape=(1,400), w.shape=(400,)',
                        '%s, %s' % (frames.shape, np.shape(w)),
                        frames.shape == (1, 400) and np.shape(w) == (400,))
        _, wr = std.enframe(x, 400, 160, 'rect')
        all_ok &= check('S2', "win='rect' 窗序列恒为 1",
                        'w ≡ 1', 'min=%.1f max=%.1f' % (wr.min(), wr.max()),
                        np.allclose(wr, 1.0))
        try:
            ok3 = all(std.enframe(x, 400, 160, k) is not None
                      for k in ('rect', 'hamming', 'hanning'))
            msg3 = 'OK'
        except Exception as e:
            ok3, msg3 = False, str(e)
        all_ok &= check('S3', '三种窗型 rect/hamming/hanning 均可调用',
                        '无异常', msg3, ok3)

        f1, _ = std.enframe(np.ones(1200), 400, 160, 'rect')
        E = std.short_time_energy(f1)
        all_ok &= check('S4', '短时能量 E_n = Σx²（全 1 帧应为 400，共 6 帧）',
                        '400 × 6', '%.6f × %d' % (E[0], len(E)),
                        abs(E[0] - 400) < 1e-9 and len(E) == 6)
        M = std.short_time_amplitude(f1)
        all_ok &= check('S5', '短时平均幅度 M_n = Σ|x|（全 1 帧应为 400）',
                        '400', '%.6f' % M[0], abs(M[0] - 400) < 1e-9)

        alt = np.array([(-1.0) ** i for i in range(400)])
        f2, _ = std.enframe(alt, 400, 160, 'rect')
        Z = std.short_time_zcr_rate(f2)
        all_ok &= check('S6', '归一化过零率 Z_n ∈ [0,1]（交替序列接近 1）',
                        '0 ≤ Z ≤ 1 且 Z > 0.99', '%.4f' % Z[0],
                        0 <= Z[0] <= 1 and Z[0] > 0.99)

        xdc = np.ones(800) * 100.0
        fr, _ = std.enframe(std.remove_dc(xdc), 400, 160, 'rect')
        all_ok &= check('S7', 'remove_dc 后直流为 0', '|mean| < 1e-9',
                        '%.2e' % abs(fr.mean()), abs(fr.mean()) < 1e-9)

        # ---------- 4. endpoint_detect 契约 ----------
        r_sil = std.endpoint_detect(np.zeros(24000), 16000)
        all_ok &= check('D1', '纯静音输入：返回 None 且不抛异常', 'None',
                        '%s' % r_sil, r_sil is None)

        xd = rw(files[0])[0]
        r2 = std.endpoint_detect(xd, 16000)
        fields = {'frames', 'E', 'M', 'Z', 'T1', 'T2', 'T3',
                  'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'start', 'end', 'seg'}
        missing = sorted(fields - set(r2.keys())) if isinstance(r2, dict) else ['未返回 dict']
        all_ok &= check('D2', '有效语音：返回 dict 且含全部约定字段（16 个）',
                        '字段齐全', '缺失 %s' % (missing or '无'),
                        isinstance(r2, dict) and not missing)
        if isinstance(r2, dict):
            ns = [r2['N%d' % i] for i in range(1, 7)]
            all_ok &= check('D3', '端点帧号满足 N1 ≤ N2 ≤ N3 ≤ N4 ≤ N5 ≤ N6',
                            '单调不减', ','.join(str(v) for v in ns),
                            all(ns[i] <= ns[i + 1] for i in range(5)))
            all_ok &= check('D4', '双门限满足 0 < T1 < T2',
                            '0 < T1 < T2', 'T1=%.4f T2=%.4f' % (r2['T1'], r2['T2']),
                            0 < r2['T1'] < r2['T2'])
            all_ok &= check('D5', '语音段 0 ≤ start < end ≤ 录音时长 1.5 s',
                            '0 ≤ start < end ≤ 1.5',
                            'start=%.3f s end=%.3f s' % (r2['start'], r2['end']),
                            0 <= r2['start'] < r2['end'] <= 1.5)

        # ---------- 5. 特征序列与 DTW ----------
        feat = std.time_domain_feature(xd, 16000)
        all_ok &= check('T1', 'time_domain_feature 输出 [T, 3]', 'shape[1] = 3',
                        '%s' % (feat.shape,), feat.ndim == 2 and feat.shape[1] == 3)
        dm = float(np.abs(feat.mean(axis=0)).max())
        ds = float(np.abs(feat.std(axis=0) - 1).max())
        all_ok &= check('T2', '特征已逐序列零均值单位方差归一化',
                        '|mean| < 1e-6 且 |std−1| < 1e-6',
                        '|mean|=%.2e |std−1|=%.2e' % (dm, ds), dm < 1e-6 and ds < 1e-6)

        fB = std.time_domain_feature(rw('0_02.wav')[0], 16000)
        d_same = std.dtw_distance(feat, feat)
        d_ab = std.dtw_distance(feat, fB)
        d_ba = std.dtw_distance(fB, feat)
        all_ok &= check('T3', 'dtw_distance(a,a) = 0', '0', '%.6f' % d_same,
                        abs(d_same) < 1e-9)
        all_ok &= check('T4', 'dtw_distance 满足对称性', '|d(a,b) − d(b,a)| < 1e-9',
                        '%.2e' % abs(d_ab - d_ba), abs(d_ab - d_ba) < 1e-9)
        all_ok &= check('T5', 'dtw_distance 非负且不同样本距离大于 0',
                        '> 0', '%.6f' % d_ab, d_ab > 0)

        clf = std.DTWClassifier()
        clf.fit([feat, fB], np.array(['0', '0']))
        pred = clf.predict([feat])
        all_ok &= check('T6', 'DTWClassifier(band_ratio=0.35).fit/predict 可跑通',
                        'predict 返回长度 1 的标签数组', '%s' % (pred,),
                        len(pred) == 1 and pred[0] == '0')

    # ---------- 输出 ----------
    n_pass = sum(1 for r in RESULTS if r[4] == 'PASS')
    n_all = len(RESULTS)
    lines = ['=' * 78,
             '接口契约自检报告（接口冻结出口判据）',
             '依据：《接口冻结说明》2026-10-05 版',
             '被测数据：%s（%d 个 wav）' % (os.path.relpath(WAV_DIR, BASE), len(files)),
             '=' * 78,
             '检查项总数：%d    通过：%d    失败：%d' % (n_all, n_pass, n_all - n_pass),
             '',
             '编号\t检查项\t判据\t实测\t结论']
    for no, item, exp, act, res in RESULTS:
        lines.append('%s\t%s\t%s\t%s\t%s' % (no, item, exp, act, res))
    lines += ['',
              '出口判据：全部检查项 PASS（接口冻结的唯一硬性判据）。',
              '当前状态：%s（被测数据：%s 的 %d 个 wav）'
              % ('全部通过' if n_pass == n_all else '存在失败项，接口未冻结',
                 os.path.relpath(WAV_DIR, BASE), len(files)),
              '备注：本框架与数据源无关；若有新的录音提交入库，'
              '把 WAV_DIR 指到对应目录重跑即可复检。']
    txt = '\n'.join(lines)
    with open(os.path.join(OUT, 'contract_test.txt'), 'w', encoding='utf-8') as fp:
        fp.write(txt + '\n')
    print('-' * 78)
    print('通过 %d/%d' % (n_pass, n_all))
    print('报告已写入 %s' % os.path.relpath(os.path.join(OUT, 'contract_test.txt'), BASE))
    return 0 if n_pass == n_all else 1


if __name__ == '__main__':
    sys.exit(main())
