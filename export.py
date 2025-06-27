"""
YOLOv7 简化版模型导出脚本
仅支持基本的TorchScript导出
"""

import argparse
import sys
import time

sys.path.append('./')

import torch

from models.experimental import attempt_load
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device

def export_torchscript(model, img, file):
    """导出TorchScript格式"""
    print(f'\n开始导出TorchScript格式到 {file}...')
    try:
        ts = torch.jit.trace(model, img, strict=False)
        ts.save(file)
        print(f'TorchScript导出成功，保存为 {file}')
        return True
    except Exception as e:
        print(f'TorchScript导出失败: {e}')
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./best.pt', help='权重文件路径')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='图像尺寸 [高度, 宽度]')
    parser.add_argument('--batch-size', type=int, default=1, help='批次大小')
    parser.add_argument('--device', default='cpu', help='设备选择: cpu')
    
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1
    print(opt)
    
    set_logging()
    t = time.time()

    # 加载PyTorch模型
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)
    labels = model.names

    # 检查图像尺寸
    gs = int(max(model.stride))
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]
    
    # 创建测试输入
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)
    
    # 设置模型为评估模式
    model.eval()
    
    # 预热
    for _ in range(2):
        y = model(img)
    
    shape = tuple(y[0].shape) if isinstance(y, tuple) else tuple(y.shape)
    print(f"\n模型摘要: {len(list(model.modules()))} 层, "
          f"{sum(x.numel() for x in model.parameters()):,} 参数, "
          f"{shape} 输出形状")

    # 导出TorchScript
    output_file = opt.weights.replace('.pt', '.torchscript.pt')
    success = export_torchscript(model, img, output_file)
    
    if success:
        print(f'\n导出完成 ({time.time() - t:.1f}s)')
        print(f'导出文件: {output_file}')
    else:
        print('\n导出失败！')
