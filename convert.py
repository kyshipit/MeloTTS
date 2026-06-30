import os
import argparse
from rknn.api import RKNN

def parse_arg():
    parser = argparse.ArgumentParser()
    # 必传参数：onnx路径、芯片平台
    parser.add_argument("model_path")
    parser.add_argument("platform")
    # 可选：开启量化
    parser.add_argument("--quant", action="store_true")
    args = parser.parse_args()

    # 自动把 xxx.onnx → xxx.rknn
    base_name = os.path.splitext(args.model_path)[0]
    args.output = base_name + ".rknn"

    return args.model_path, args.platform, args.quant, args.output

if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=False)
    # Pre-process config
    print('--> Config model')
    rknn.config(target_platform=platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()