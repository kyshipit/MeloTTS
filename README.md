🌐 Language: **English** | [中文](README.zh.md)

## MeloTTS-RKNN

> MeloTTS ONNX export + RKNN deployment for Rockchip NPU (RK3588/RK3576)

Based on the community edition [mmontol/MeloTTS](https://github.com/mmontol/MeloTTS), with added RKNN conversion scripts, INT8 quantization support, and complete deployment documentation.


### 📌 What This Repository Does

| Feature | Description |
|------|------|
| ONNX export | Already available in the community edition; this repository optimizes the export configuration (constant folding, dynamic axis naming, etc.) |
| RKNN conversion | Added `convert_dynamic.py`, which automatically detects Encoder/Decoder |
| INT8 quantization | Added `generate_dataset.py` (random calibration data) and `convert_bin_to_npy.py` (real calibration data conversion), supporting `--do_quant` quantized conversion |
| Japanese dictionary error fix | Fixes the MeCab initialization failure issue |


### 📁 File Description

| File | Description |
|------|------|
| `export_onnx.py` | ONNX export entry point |
| `convert_dynamic.py` | RKNN conversion script, supports `--do_quant` and `--seq_lens` |
| `generate_dataset.py` | Generates INT8 quantization calibration data (random data, for functional verification only) |
| `convert_bin_to_npy.py` | Converts `.bin` calibration data exported by the board-side C++ program into `.npy` format |
| `melo/api.py` | ONNX export method (optimized constant folding and dynamic axis configuration) |


### 🚀 Quick Start

#### 1. Export ONNX

```
conda activate 3.9-melotts
pip install -r requirements.txt
python export_onnx.py
```

#### 2. Generate Calibration Data (Optional, for INT8 Quantization)

```
python generate_dataset.py
```



#### 3. Convert to RKNN

```
conda activate py3.11-tk2-2.3.2

# FP16 (recommended, clear audio quality)
python convert_dynamic.py encoder-ZH_MIX_EN.onnx rk3588
python convert_dynamic.py decoder-ZH_MIX_EN.onnx rk3588 --seq_lens 256

# INT8 quantization (smaller size, lossy audio quality)
python convert_dynamic.py encoder-ZH_MIX_EN.onnx rk3588 --do_quant
python convert_dynamic.py decoder-ZH_MIX_EN.onnx rk3588 --do_quant --seq_lens 256
```


## Key Technical Notes

### 1. About Dynamic Shape

MeloTTS ONNX export is based on the Torch Tracing mechanism. `dynamic_axes` is only an ONNX metadata marker, and the actual computation graph has already been fixed. Therefore, during RKNN conversion, only a single length `seq_lens=256` can be used. Multi-length configurations (such as 128/256/512) will cause `MatMul dimension mismatch`, because the ONNX internals have already been fixed to 256.

### 2. INT8 Quantization Calibration Data Notes

This repository provides `generate_dataset.py` to generate random calibration data (for functional verification only). For actual INT8 model deployment, it is recommended to use real inference data:

1. In the board-side C++ inference program, after `middle_process`, save the Decoder's six input tensors (`attn`, `y_mask`, `g`, `m_p`, `logs_p`, `noise_scale`) as `.bin` files
2. Use `convert_bin_to_npy.py` to convert the `.bin` files into `.npy` format, generating `dataset_decoder.txt`
3. Re-run INT8 quantization using the real calibration data

> Note: Random-number calibration can pass `rknn.build`, but the model outputs silence; the calibration data shape must match the model's fixed dimensions (Decoder's `attn` is uniformly `[1, 512, 256]`).

### 3. Known Limitations

- Decoder only supports seq_len=256 (multi-length configurations will directly report MatMul dimension mismatch)
- Encoder INT8 quantization fails due to the `Less` operator type conflict
- After INT8 quantization, Decoder audio quality degrades noticeably (noise), and it is not recommended for production environments

### 4. Summary Document

- [MeloTTS RKNN Quantization Deployment Pitfall Record](https://github.com/kyshipit/tech-notes/blob/main/en/deploy/MeloTTS-RKNN-Quantization-Deployment-Pitfalls.md) — Full deployment workflow from PyTorch → ONNX → RKNN, recording core pitfalls such as dynamic dimensions, pseudo-dynamic dimensions, INT8 quantization, and Less operator conflicts.

### Acknowledgements

This repository is developed based on mmontol/MeloTTS. Thanks to mmontol for providing the ONNX export implementation.

### 📄 License

MIT
