#!/usr/bin/env python3
"""
将 C++ 保存的 .bin 校准数据转换为 .npy 格式，并生成 dataset_decoder.txt。
每个样本的 .bin 文件前缀相同（如 0000），包含：
    attn_XXXX.bin, y_mask_XXXX.bin, g_XXXX.bin, m_p_XXXX.bin, logs_p_XXXX.bin, noise_scale_XXXX.bin
转换后生成对应的 .npy 文件，并写入 dataset_decoder.txt。
"""

import numpy as np
import os
import glob

# 配置参数
BIN_DIR = "real_calib"            # .bin 文件所在目录
OUT_DIR = "real_calibration_data" # .npy 输出目录
DATASET_TXT = "dataset_decoder.txt"

# 固定形状（根据你的模型定义）
SHAPES = {
    'attn': (1, 512, 256),        # [batch, seq_len*2, attn_dim]
    'y_mask': (1, 1, 512),        # [batch, 1, seq_len*2]
    'g': (1, 256, 1),
    'm_p': (1, 192, 256),
    'logs_p': (1, 192, 256),
    'noise_scale': (1,)
}

def main():
    # 创建输出目录
    os.makedirs(OUT_DIR, exist_ok=True)

    # 获取所有 attn 文件前缀
    attn_files = glob.glob(os.path.join(BIN_DIR, "attn_*.bin"))
    if not attn_files:
        print(f"错误：在 {BIN_DIR} 中未找到 attn_*.bin 文件")
        return

    prefixes = sorted([os.path.basename(f).replace("attn_", "").replace(".bin", "") for f in attn_files])
    print(f"找到 {len(prefixes)} 个样本")

    with open(DATASET_TXT, "w") as f_out:
        for prefix in prefixes:
            # 构造文件路径
            files = {
                key: os.path.join(BIN_DIR, f"{key}_{prefix}.bin")
                for key in SHAPES.keys()
            }
            # 检查是否所有文件都存在
            missing = [k for k, v in files.items() if not os.path.exists(v)]
            if missing:
                print(f"跳过前缀 {prefix}，缺少文件: {missing}")
                continue

            # 读取并重塑数据
            data = {}
            for key, shape in SHAPES.items():
                raw = np.fromfile(files[key], dtype=np.float32)
                # 检查大小是否匹配
                expected_size = np.prod(shape)
                if raw.size != expected_size:
                    print(f"警告：{key}_{prefix}.bin 大小 {raw.size} 与预期 {expected_size} 不匹配，将截断/填充")
                    if raw.size > expected_size:
                        raw = raw[:expected_size]
                    else:
                        raw = np.pad(raw, (0, expected_size - raw.size), 'constant')
                data[key] = raw.reshape(shape)

            # 保存为 .npy
            for key, arr in data.items():
                np.save(os.path.join(OUT_DIR, f"{prefix}_{key}.npy"), arr)

            # 写入 dataset_decoder.txt
            line = " ".join([
                os.path.join(OUT_DIR, f"{prefix}_attn.npy"),
                os.path.join(OUT_DIR, f"{prefix}_y_mask.npy"),
                os.path.join(OUT_DIR, f"{prefix}_g.npy"),
                os.path.join(OUT_DIR, f"{prefix}_m_p.npy"),
                os.path.join(OUT_DIR, f"{prefix}_logs_p.npy"),
                os.path.join(OUT_DIR, f"{prefix}_noise_scale.npy")
            ])
            f_out.write(line + "\n")

    print(f"完成！共处理 {len(prefixes)} 个样本")
    print(f".npy 文件保存在 {OUT_DIR}/")
    print(f"数据集索引文件已生成：{DATASET_TXT}")

if __name__ == "__main__":
    main()