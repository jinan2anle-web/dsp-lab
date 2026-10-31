"""纯 Python 解析 RIFF/WAVE 头并读出采样，不依赖 scipy。

为什么不直接用 ``scipy.io.wavfile``：
1. 仓库自带的 python 没有 scipy（见 ``dsp/env.py`` 说明）；
2. 实验二/三要求"各个功能模块均采用编程实现"，读 wav 也自己走一遍 chunk。

实验一踩过的坑写在这里，别再犯：RIFF 的 chunk 长度是**奇数时要补 1 字节对齐**，
漏掉这一步会让后面所有 chunk 整体错位。
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# 允许的音频格式编号：1 = PCM，3 = IEEE float，0xFFFE = WAVE_FORMAT_EXTENSIBLE
_PCM = 1
_IEEE_FLOAT = 3
_EXTENSIBLE = 0xFFFE


def parse_header(path):
    """只解析头，不读采样。返回 dict（格式、通道数、采样率、位深、数据块长度）。"""
    raw = Path(path).read_bytes()
    if len(raw) < 12 or raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"不是 RIFF/WAVE 文件：{path}")

    info = {"audio_format": None, "channels": None, "fs": None,
            "bits": None, "data_bytes": None, "data_offset": None}

    pos = 12
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        size = struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = raw[pos + 8:pos + 8 + size]

        if cid == b"fmt ":
            if len(body) < 16:
                raise ValueError(f"fmt 块长度异常：{size}")
            fmt, nch, fs, _byte_rate, _block_align, bits = struct.unpack("<HHIIHH", body[:16])
            if fmt == _EXTENSIBLE and len(body) >= 26:
                # WAVE_FORMAT_EXTENSIBLE 的真实格式在 SubFormat 的前 2 字节
                fmt = struct.unpack("<H", body[24:26])[0]
            info.update(audio_format=fmt, channels=nch, fs=fs, bits=bits)
        elif cid == b"data":
            info.update(data_bytes=size, data_offset=pos + 8)
            break  # data 之后不需要再看

        pos += 8 + size + (size & 1)  # ← 奇数块补 1 字节
    else:
        raise ValueError(f"没有找到 data 块：{path}")

    if info["fs"] is None:
        raise ValueError(f"没有找到 fmt 块：{path}")
    # 有些录音机的 data 长度字段不可信，按实际字节数兜底
    avail = len(raw) - info["data_offset"]
    if info["data_bytes"] is None or info["data_bytes"] > avail:
        info["data_bytes"] = avail
    return info


def read_wav(path, *, mono=True):
    """读一条 wav。

    返回 ``(x, fs)``：``x`` 是 float64、归一化到 [-1, 1)，形状 ``[n]``（mono=True）
    或 ``[n, ch]``（mono=False）；16 bit PCM 按 2^15 归一化。
    """
    info = parse_header(path)
    raw = Path(path).read_bytes()
    off, nbytes = info["data_offset"], info["data_bytes"]
    payload = raw[off:off + nbytes]
    nch, bits, fmt = info["channels"], info["bits"], info["audio_format"]

    if fmt == _PCM and bits == 16:
        x = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0
    elif fmt == _PCM and bits == 8:
        # 8 bit PCM 是无符号的，0..255 对应 -128..127
        x = (np.frombuffer(payload, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif fmt == _PCM and bits == 32:
        x = np.frombuffer(payload, dtype="<i4").astype(np.float64) / 2147483648.0
    elif fmt == _IEEE_FLOAT and bits == 32:
        x = np.frombuffer(payload, dtype="<f4").astype(np.float64)
    elif fmt == _IEEE_FLOAT and bits == 64:
        x = np.frombuffer(payload, dtype="<f8").astype(np.float64)
    else:
        raise ValueError(f"暂不支持的格式：format={fmt}, bits={bits}（{path}）")

    usable = (len(x) // nch) * nch
    x = x[:usable].reshape(-1, nch)
    if mono:
        x = x[:, 0]
    return x, info["fs"]


def summarize(path):
    """一个 wav 的一行摘要，给自测输出用。"""
    info = parse_header(path)
    dur = info["data_bytes"] / (info["channels"] * info["bits"] // 8) / info["fs"]
    return (f"{Path(path).name}: fs={info['fs']} Hz, ch={info['channels']}, "
            f"bits={info['bits']}, {dur:.3f} s")
