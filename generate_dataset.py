import numpy as np
import os

def generate_encoder_dataset(num_samples=5, x_length=256):
    os.makedirs("calibration_data_encoder", exist_ok=True)
    with open("dataset_encoder.txt", "w") as f:
        for i in range(num_samples):
            # 整型输入保持 int64
            x = np.zeros((1, x_length), dtype=np.int64)
            x_lengths = np.array([x_length], dtype=np.int64)
            sid = np.array([0], dtype=np.int64)
            tone = np.random.randint(1, 5, size=(1, x_length)).astype(np.int64)
            lang_ids = np.zeros((1, x_length), dtype=np.int64)
            # 新增 ja_bert：float32 零张量，形状 [1, 768, x_length]
            ja_bert = np.zeros((1, 768, x_length), dtype=np.float32)
            noise_scale_w = np.array([0.8], dtype=np.float32)
            sdp_ratio = np.array([0.2], dtype=np.float32)

            prefix = f"calibration_data_encoder/sample_{i:02d}"

            # 保存所有 8 个输入
            np.save(f"{prefix}_x.npy", x)
            np.save(f"{prefix}_x_lengths.npy", x_lengths)
            np.save(f"{prefix}_sid.npy", sid)
            np.save(f"{prefix}_tone.npy", tone)
            np.save(f"{prefix}_lang_ids.npy", lang_ids)
            np.save(f"{prefix}_ja_bert.npy", ja_bert)              # ← 新增
            np.save(f"{prefix}_noise_scale_w.npy", noise_scale_w)
            np.save(f"{prefix}_sdp_ratio.npy", sdp_ratio)

            # dataset_encoder.txt 每行写 8 个文件
            line = (f"{prefix}_x.npy {prefix}_x_lengths.npy {prefix}_sid.npy "
                    f"{prefix}_tone.npy {prefix}_lang_ids.npy {prefix}_ja_bert.npy "
                    f"{prefix}_noise_scale_w.npy {prefix}_sdp_ratio.npy")
            f.write(line + "\n")
    print("dataset_encoder.txt 已生成（8 个输入，包含 ja_bert）。")

def generate_decoder_dataset(num_samples=10, seq_len=256):
    os.makedirs("calibration_data_decoder", exist_ok=True)
    with open("dataset_decoder.txt", "w") as f:
        for i in range(num_samples):
            # 使用更稳定的随机范围，避免数值溢出
            attn = np.random.uniform(-1, 1, (1, 2*seq_len, 256)).astype(np.float32)
            y_mask = np.random.uniform(-1, 1, (1, 1, 2*seq_len)).astype(np.float32)
            g = np.random.uniform(-1, 1, (1, 256, 1)).astype(np.float32)
            m_p = np.random.uniform(-1, 1, (1, 192, seq_len)).astype(np.float32)
            logs_p = np.random.uniform(-1, 1, (1, 192, seq_len)).astype(np.float32)
            noise_scale = np.array([0.6], dtype=np.float32)

            # 确保没有 NaN
            assert not np.isnan(attn).any(), "NaN in attn"
            assert not np.isnan(m_p).any(), "NaN in m_p"

            prefix = f"calibration_data_decoder/sample_{i:02d}"
            np.save(f"{prefix}_attn.npy", attn)
            np.save(f"{prefix}_y_mask.npy", y_mask)
            np.save(f"{prefix}_g.npy", g)
            np.save(f"{prefix}_m_p.npy", m_p)
            np.save(f"{prefix}_logs_p.npy", logs_p)
            np.save(f"{prefix}_noise_scale.npy", noise_scale)

            line = (f"{prefix}_attn.npy {prefix}_y_mask.npy {prefix}_g.npy "
                    f"{prefix}_m_p.npy {prefix}_logs_p.npy {prefix}_noise_scale.npy")
            f.write(line + "\n")
    print("dataset_decoder.txt 已重新生成，使用均匀分布数据。")

if __name__ == "__main__":
    generate_encoder_dataset(num_samples=5)   # 5个样本足够了
    generate_decoder_dataset(num_samples=5)