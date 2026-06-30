## MeloTTS-RKNN

> MeloTTS ONNX export + RKNN deployment for Rockchip NPU (RK3588/RK3576)

基于 [mmontol/MeloTTS](https://github.com/mmontol/MeloTTS) 社区版，增加 RKNN 转换脚本、INT8 量化支持和完整部署文档。


### 📌 本仓做了什么

| 功能 | 说明 |
|------|------|
| ONNX 导出 | 社区版已有，本仓库优化了导出配置（常量折叠、动态轴命名等） |
| RKNN 转换 | 新增 `convert_dynamic.py`，自动识别 Encoder/Decoder |
| INT8 量化 | 新增 `generate_dataset.py`，支持 `--do_quant` 量化转换 |
| 日语词典报错修复 | 修复 MeCab 初始化失败问题 |

### 📁 文件说明

| 文件                  | 说明                                                         |
| --------------------- | ------------------------------------------------------------ |
| export_onnx.py        | ONNX 导出入口                                                 |
| convert_dynamic.py    | RKNN 转换脚本，支持 --do_quant 和 --seq_lens                  |
| generate_dataset.py   | INT8 量化校准数据生成                                         |
| melo/api.py           | ONNX 导出方法（优化了常量折叠和动态轴配置）|

### ⚠️ 已知限制

- Decoder 仅支持 seq_len=256（多长度转换在 RKNN 上不稳定）

- Encoder INT8 量化可能失败，建议使用 FP16


### 🚀 快速开始

#### 1. 安装依赖
```bash
pip install -r requirements.txt
pip install rknn-toolkit2  # 需要 RK3588 开发环境
```

#### 2. 导出 ONNX
```bash
conda activate 3.9-melotts
python export_onnx.py
```

#### 3. 生成校准数据（可选，用于 INT8 量化）
```bash
python generate_dataset.py
```

#### 4. 转换为 RKNN
```bash
conda activate py3.11-tk2-2.3.2

# FP16（推荐，稳定）
python convert_dynamic.py encoder-ZH_MIX_EN.onnx rk3588
python convert_dynamic.py decoder-ZH_MIX_EN.onnx rk3588 --seq_lens 256

# INT8 量化（模型更小，速度略快）
python convert_dynamic.py encoder-ZH_MIX_EN.onnx rk3588 --do_quant
python convert_dynamic.py decoder-ZH_MIX_EN.onnx rk3588 --do_quant --seq_lens 256
```

### 📄 License

MIT