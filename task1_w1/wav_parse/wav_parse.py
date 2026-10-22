import numpy as np
import os

def read_wav(filepath):

    f = open(filepath, "rb")
    # ========= RIFF块 =========
    riff_id = f.read(4)
    riff_size = int.from_bytes(f.read(4), byteorder='little')
    riff_type = f.read(4)

    # ========= fmt子块 =========
    fmt_sub_id = f.read(4)
    fmt_sub_size = int.from_bytes(f.read(4), byteorder='little')
    fmt_data = f.read(fmt_sub_size)

    audio_format = int.from_bytes(fmt_data[0:2], byteorder='little')
    num_channels = int.from_bytes(fmt_data[2:4], byteorder='little')
    sample_rate = int.from_bytes(fmt_data[4:8], byteorder='little')
    byte_rate = int.from_bytes(fmt_data[8:12], byteorder='little')
    block_align = int.from_bytes(fmt_data[12:14], byteorder='little')
    bits_per_sample = int.from_bytes(fmt_data[14:16], byteorder='little')

    # ========= 循环查找data音频数据块 =========
    while True:
        sub_id = f.read(4)
        sub_size_bytes = f.read(4)
        sub_size = int.from_bytes(sub_size_bytes, byteorder='little')
        if sub_id == b'data':
            audio_raw_bytes = f.read(sub_size)
            break
        else:
            f.read(sub_size)
    f.close()

    # 根据位深设置数据类型
    if bits_per_sample == 8:
        dtype = np.uint8
    elif bits_per_sample == 16:
        dtype = np.int16
    elif bits_per_sample == 32:
        dtype = np.int32
    else:
        raise NotImplementedError("不支持该位深")

    wave_data = np.frombuffer(audio_raw_bytes, dtype=dtype)
    return wave_data, sample_rate, num_channels, bits_per_sample


if __name__ == "__main__":
    # 输出字段对照表，保存到txt
    with open("wav_field_table.txt", "w", encoding="utf-8") as fp:
        fp.write("====  WAV文件字段对照表 ====\n")
        fp.write("字段名\t\t含义\n")
        fp.write("riff_id\t\t文件标识，固定'RIFF'\n")
        fp.write("riff_size\t文件总字节数 - 8\n")
        fp.write("riff_type\t文件类型，固定'WAVE'\n")
        fp.write("fmt_sub_id\t子块标识，固定'fmt '\n")
        fp.write("fmt_sub_size\tfmt子块大小，PCM为16\n")
        fp.write("audio_format\t音频编码格式，PCM=1\n")
        fp.write("num_channels\t通道数，单声道=1\n")
        fp.write("sample_rate\t采样率，本实验16000Hz\n")
        fp.write("bits_per_sample\t量化位深，本实验16bit\n")
        fp.write("data\t\t音频采样数据\n\n")

        # 遍历当前文件夹wav
        for filename in os.listdir("."):
            if filename.lower().endswith(".wav"):
                info_text = f"\n======== {filename} ========\n"
                try:
                    wav, fs, ch, bit = read_wav(filename)
                    info_text += f"采样率 fs = {fs} Hz\n"
                    info_text += f"通道数 channel = {ch}\n"
                    info_text += f"位深 bits = {bit} bit\n"
                    info_text += f"采样数据长度 = {wav.shape[0]}\n"

                    print(info_text)
                except Exception as e:
                    info_text += f"解析失败：{e}\n"
                    print(info_text)
                fp.write(info_text)
    print("===== 任务执行完成！ =====")