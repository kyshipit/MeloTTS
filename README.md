## MeloTTS-RKNN

> MeloTTS ONNX export + RKNN deployment for Rockchip NPU (RK3588/RK3576)

基于 [mmontol/MeloTTS](https://github.com/mmontol/MeloTTS) 社区版，增加 RKNN 转换脚本、INT8 量化支持和完整部署文档。


### 📌 本仓做了什么

| 功能 | 说明 |
|------|------|
| ONNX 导出 | 社区版已有，本仓库优化了导出配置（常量折叠、动态轴命名等） |
| RKNN 转换 | 新增 `convert_dynamic.py`，自动识别 Encoder/Decoder |
| INT8 量化 | 新增 `generate_dataset.py`和 `convert_bin_to_npy.py`，支持 `--do_quant` 量化转换 |
| 日语词典报错修复 | 修复 MeCab 初始化失败问题 |

### 📁 文件说明

| 文件 | 说明 |
|------|------|
| `export_onnx.py` | ONNX 导出入口 |
| `convert_dynamic.py` | RKNN 转换脚本，支持 `--do_quant` 和 `--seq_lens` |
| `generate_dataset.py` | INT8 量化校准数据生成（随机数据，仅供功能验证） |
| `convert_bin_to_npy.py` | 将开发板 C++ 程序导出的 `.bin` 校准数据转换为 `.npy` 格式 |
| `melo/api.py` | ONNX 导出方法（优化了常量折叠和动态轴配置） |

### 🚀 快速开始

#### 1. 导出 ONNX
```bash
conda activate 3.9-melotts
pip install -r requirements.txt
python export_onnx.py
```

#### 2. 生成校准数据（可选，用于 INT8 量化）
```bash
python generate_dataset.py
```

#### 3. 转换为 RKNN
```bash
conda activate py3.11-tk2-2.3.2

# FP16（推荐，稳定）
python convert_dynamic.py encoder-ZH_MIX_EN.onnx rk3588
python convert_dynamic.py decoder-ZH_MIX_EN.onnx rk3588 --seq_lens 256

# INT8 量化（模型更小，速度略快）
python convert_dynamic.py encoder-ZH_MIX_EN.onnx rk3588 --do_quant
python convert_dynamic.py decoder-ZH_MIX_EN.onnx rk3588 --do_quant --seq_lens 256
```

## 关键技术说明

### 1. 关于动态 Shape

MeloTTS 的 ONNX 导出基于 Torch Tracing 机制，`dynamic_axes` 仅为 ONNX 元数据标记，实际计算图已被固化。因此 RKNN 转换时只能使用单长度 `seq_lens=256`，多长度配置（如 128/256/512）会导致 `MatMul dimension mismatch`。

### 2. INT8 量化校准数据说明

本仓库提供 `generate_dataset.py` 用于生成随机校准数据（仅供功能验证使用）。如需实际部署 INT8 模型，建议使用真实推理数据：

1. 在开发板 C++ 推理程序中，在 `middle_process` 后保存 Decoder 的六个输入张量（`attn`、`y_mask`、`g`、`m_p`、`logs_p`、`noise_scale`）为 `.bin` 文件

2. 使用 `convert_bin_to_npy.py` 将 `.bin` 转换为 `.npy` 格式，生成 `dataset_decoder.txt`

3. 使用真实校准数据重新执行 INT8 量化

> 注意：随机数校准可通过 `rknn.build` 但模型输出静音；校准数据 shape 必须与模型固化维度一致（Decoder 的 `attn` 统一为 `[1, 512, 256]`）。

### 3. 已知限制

- Decoder 仅支持 `seq_len=256`（多长度转换在 RKNN 上不稳定）

- Encoder INT8 量化因 `Less` 算子类型冲突可能失败，建议使用 FP16

- INT8 量化后 TTS 音质可能有轻微下降，这是生成式任务 INT8 量化的固有限制

### 致谢

本仓库基于 mmontol/MeloTTS 开发，感谢 mmontol 提供的 ONNX 导出实现。

### 📄 License

MIT
