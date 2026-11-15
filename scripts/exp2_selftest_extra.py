"""实验二新增接口与扰动工具的独立自测。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from dsp import augment, config, endpoint, io_wav, mfcc

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

EVIDENCE_DIR = config.REPORT_DIR / "evidence"
BASELINE_PATH = Path(r"C:\Users\32375\AppData\Local\Temp\mfcc_baseline_before.npz")


def main():
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    x, fs = io_wav.read_wav(config.DATA_DIR / "0_01.wav")
    default = mfcc.extract(x, fs)
    explicit = mfcc.extract(x, fs, form="mfcc", use_delta=False)
    logmel = mfcc.extract(x, fs, form="logmel")
    with_delta = mfcc.extract(x, fs, use_delta=True)

    assert default.shape == (148, 13)
    assert logmel.shape == (148, 24)
    assert with_delta.shape == (148, 39)
    assert np.array_equal(default, explicit)
    before_equal = False
    if BASELINE_PATH.exists():
        before = np.load(BASELINE_PATH)["y"]
        before_equal = np.array_equal(default, before)
        assert before_equal

    c = default
    delta = with_delta[:, 13:26]
    delta2 = with_delta[:, 26:39]
    assert np.array_equal(delta[0], c[1] - c[0])
    assert np.array_equal(delta[1:], c[1:] - c[:-1])
    assert np.array_equal(delta2[0], delta[1] - delta[0])
    assert np.array_equal(delta2[1:], delta[1:] - delta[:-1])

    mfcc_lines = [
        "实验二 MFCC 新接口自测",
        f"default_shape = {default.shape}",
        f"logmel_shape = {logmel.shape}",
        f"delta_shape = {with_delta.shape}",
        f"default_equals_explicit_mfcc = {np.array_equal(default, explicit)}",
        f"default_equals_saved_prechange = {before_equal}",
        f"c[0,0] = {c[0, 0]:.15f}",
        f"delta[0,0] = {delta[0, 0]:.15f}",
        f"delta2[0,0] = {delta2[0, 0]:.15f}",
    ]
    (EVIDENCE_DIR / "exp2_mfcc_logmel.txt").write_text(
        "\n".join(mfcc_lines) + "\n", encoding="utf-8")

    got = endpoint.detect(x, fs)
    fallback_count = int(got is None)
    s0, s1 = (0, len(x)) if got is None else got
    signal_power = float(np.mean(x[s0:s1] ** 2))
    aug_lines = ["实验二扰动工具自测"]
    for snr_db in (20.0, 10.0, 5.0):
        noisy = augment.add_white_noise(x, snr_db, fs=fs, seed=0)
        noise = noisy - x
        noise_power = float(np.mean(noise[s0:s1] ** 2))
        measured = 10.0 * np.log10(signal_power / noise_power)
        error = measured - snr_db
        assert abs(error) < 0.5
        aug_lines.append(
            f"target_snr_db={snr_db:.1f}, measured_snr_db={measured:.6f}, error_db={error:+.6f}")
    for factor in (0.5, 1.5, 2.0):
        gained = augment.apply_gain(x, factor)
        ratio = float(np.abs(gained).mean() / np.abs(x).mean())
        assert len(gained) == len(x)
        assert np.isclose(ratio, factor, rtol=0.0, atol=1e-12)
        aug_lines.append(f"gain_factor={factor:.1f}, abs_mean_ratio={ratio:.12f}")
    assert np.array_equal(augment.add_white_noise(x, float("inf"), fs=fs), x)
    aug_lines.append(f"fallback_to_whole_file = {fallback_count} 条")
    aug_lines.append("snr_inf_returns_original = True")
    (EVIDENCE_DIR / "exp2_augment.txt").write_text(
        "\n".join(aug_lines) + "\n", encoding="utf-8")

    summary = mfcc_lines + [""] + aug_lines + ["", "全部新增断言通过"]
    (EVIDENCE_DIR / "exp2_selftest_extra.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"→ 已写 {EVIDENCE_DIR} 下 3 份新增自测证据")
    return 0


if __name__ == "__main__":
    sys.exit(main())
