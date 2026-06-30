import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 打印当前路径，确认优先级
print("sys.path[0] =", sys.path[0])

import melo
print("melo.__file__ =", melo.__file__)  # 应该显示 MeloTTS-main/melo/__init__.py

from melo.api import TTS

# Speed is adjustable
speed = 1.0
device = 'cpu'  # or cuda:0

# 这里 text 变量虽然定义但未使用，因为 export_onnx 内部使用预设示例
text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"

model = TTS(language='ZH', device=device, use_hf=False,
            config_path='./MeloTTS-Chinese/config.json',
            ckpt_path='./MeloTTS-Chinese/checkpoint.pth')
speaker_ids = model.hps.data.spk2id

# 导出 ONNX，只传 speaker_id，内部使用固定示例文本
model.export_onnx(speaker_ids['ZH'])