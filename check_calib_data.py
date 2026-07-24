#!/usr/bin/env python3
"""
检查校准数据文件是否全部符合预期形状（attn 为 (1,512,256)），并统计样本数量。
"""

import numpy as np
import os
import glob

DATA_DIR = "real_calibration_data"  # .npy 文件目录
EXPECTED_ATTN_SHAPE = (1, 512, 256)

def main():
    attn_files = glob.glob(os.path.join(DATA_DIR, "*_attn.npy"))
    if not attn_files:
        print(f"在 {DATA_DIR} 中未找到 *_attn.npy 文件")
        return

    valid_count = 0
    invalid_count = 0
    prefix_list = []

    for f in sorted(attn_files):
        prefix = os.path.basename(f).replace("_attn.npy", "")
        attn = np.load(f)
        if attn.shape == EXPECTED_ATTN_SHAPE:
            valid_count += 1
            prefix_list.append(prefix)
        else:
            invalid_count += 1
            print(f"样本 {prefix} 形状错误: {attn.shape}，预期 {EXPECTED_ATTN_SHAPE}")

    print(f"\n统计结果：")
    print(f"有效样本（形状正确）: {valid_count}")
    print(f"无效样本（形状错误）: {invalid_count}")
    print(f"总计: {valid_count + invalid_count}")

    if valid_count == 0:
        print("没有有效样本，请检查数据生成过程。")
    else:
        print(f"\n有效样本前缀（前 20 个）: {prefix_list[:20]}")
        # 可选：检查是否存在对应的其他文件
        missing_files = []
        for prefix in prefix_list:
            for suffix in ["_y_mask", "_g", "_m_p", "_logs_p", "_noise_scale"]:
                fname = os.path.join(DATA_DIR, f"{prefix}{suffix}.npy")
                if not os.path.exists(fname):
                    missing_files.append(f"{prefix}{suffix}")
        if missing_files:
            print(f"\n警告：以下样本缺少配套文件: {missing_files[:10]}...")

if __name__ == "__main__":
    main()