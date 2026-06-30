import sys
import argparse
from rknn.api import RKNN

def build_dynamic_input(model_path, seq_lens=[256]):
    """
    仅为 Decoder 生成 dynamic_input 列表。
    对于 Encoder，此函数不会被调用。
    """
    if 'encoder' in model_path.lower():
        # 这个分支实际上不会被执行，因为 Encoder 走的是静态转换路径
        # 但为了完整性保留
        dynamic_list = []
        for seq_len in seq_lens:
            shapes = [
                [1, seq_len],   # x
                [1],            # x_lengths
                [1],            # sid
                [1, seq_len],   # tone
                [1, seq_len],   # lang_ids
                [1],            # noise_scale_w
                [1]             # sdp_ratio
            ]
            dynamic_list.append(shapes)
        return dynamic_list
    elif 'decoder' in model_path.lower():
        dynamic_list = []
        for seq_len in seq_lens:
            shapes = [
                [1, 2*seq_len, 256],
                [1, 1, 2*seq_len],
                [1, 256, 1],
                [1, 192, seq_len],
                [1, 192, seq_len],
                [1]
            ]
            dynamic_list.append(shapes)
        return dynamic_list
    else:
        raise ValueError("Unknown model type")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('platform', type=str)
    parser.add_argument('--do_quant', action='store_true', default=False)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--seq_lens', type=int, nargs='+', default=[128, 256, 512],
                        help='Supported sequence lengths (only for decoder)')
    args = parser.parse_args()

    output_path = args.output_path or args.model_path.replace('.onnx', '.rknn')

    rknn = RKNN(verbose=True)

    # ========== 根据模型类型选择转换方式 ==========
    if 'encoder' in args.model_path.lower():
        print("[INFO] Encoder: static conversion (no dynamic_input)")
        rknn.config(target_platform=args.platform)
        dataset_file = 'dataset_encoder.txt'  # ✅ 为 Encoder 指定数据集文件
    else:
        print("[INFO] Decoder: dynamic conversion")
        dynamic_input = build_dynamic_input(args.model_path, seq_lens=args.seq_lens)
        print("\n[INFO] Generated dynamic_input (first entry):")
        for idx, shapes in enumerate(dynamic_input[0]):
            print(f"  Input {idx}: {shapes}")
        rknn.config(
            target_platform=args.platform,
            dynamic_input=dynamic_input,
            optimization_level=0  # 仅对 Decoder 设置
        )
        dataset_file = 'dataset_decoder.txt'  # ✅ 为 Decoder 指定数据集文件

    # ========== 加载 ONNX ==========
    ret = rknn.load_onnx(model=args.model_path)
    if ret != 0:
        print('Load model failed!')
        sys.exit(1)

    # ========== 构建模型（量化） ==========
    dataset = dataset_file if args.do_quant else None
    ret = rknn.build(do_quantization=args.do_quant, dataset=dataset)
    if ret != 0:
        print('Build model failed!')
        sys.exit(1)
        
    # ========== 导出 RKNN ==========
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export failed!')
        sys.exit(1)

    rknn.release()
    print(f"Successfully exported to {output_path}")