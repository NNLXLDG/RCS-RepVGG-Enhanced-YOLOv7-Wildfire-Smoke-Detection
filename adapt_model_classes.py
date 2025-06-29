#!/usr/bin/env python3
"""
模型类别数适配工具
用于将预训练权重适配到不同类别数的数据集
"""

import torch
import torch.nn as nn
from pathlib import Path
import yaml
import argparse
from models.yolo import Model
from utils.general import check_file

def adapt_model_classes(weights_path, target_nc, output_path=None, data_config=None):
    """
    将预训练模型适配到目标类别数
    
    Args:
        weights_path: 预训练权重路径
        target_nc: 目标类别数
        output_path: 输出权重路径
        data_config: 数据配置文件路径
    """
    print(f"🔧 开始适配模型类别数...")
    print(f"   源权重: {weights_path}")
    print(f"   目标类别数: {target_nc}")
    
    # 加载权重
    try:
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        model = ckpt['model']
        print(f"✅ 成功加载源权重")
        print(f"   原始类别数: {model.model[-1].nc}")
        print(f"   模型类别名称: {model.names}")
        
    except Exception as e:
        print(f"❌ 加载权重失败: {e}")
        return False
    
    # 检查是否需要适配
    if model.model[-1].nc == target_nc:
        print(f"✅ 模型类别数已匹配，无需适配")
        return True
    
    # 适配检测头
    detection_layer = model.model[-1]  # 获取检测层
    
    # 保存原始的锚框设置
    anchors = detection_layer.anchors.clone()
    stride = detection_layer.stride.clone()
    
    # 重新初始化检测层
    detection_layer.nc = target_nc  # 设置新的类别数
    detection_layer.no = target_nc + 5  # 输出数量 = 类别数 + 5 (x,y,w,h,conf)
    
    # 重新初始化输出层权重
    for i, (conv, bn) in enumerate(zip(detection_layer.m, detection_layer.m_bn)):
        # 重新初始化卷积层
        old_weight = conv.weight.data
        old_bias = conv.bias.data if conv.bias is not None else None
        
        # 创建新的卷积层
        new_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=anchors.shape[1] * detection_layer.no,  # 新的输出通道数
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None
        )
        
        # 复制部分权重（前5个通道：x,y,w,h,conf）
        with torch.no_grad():
            # 计算每个锚框的输出维度
            anchor_outputs = anchors.shape[1]  # 每个位置的锚框数
            old_no = old_weight.shape[0] // anchor_outputs  # 原始的每个锚框输出数
            
            for a in range(anchor_outputs):
                # 复制位置和置信度权重 (x,y,w,h,conf)
                old_start = a * old_no
                new_start = a * detection_layer.no
                
                # 复制前5个通道的权重
                new_conv.weight.data[new_start:new_start+5] = old_weight[old_start:old_start+5]
                if new_conv.bias is not None and old_bias is not None:
                    new_conv.bias.data[new_start:new_start+5] = old_bias[old_start:old_start+5]
                
                # 随机初始化新的类别权重
                nn.init.normal_(new_conv.weight.data[new_start+5:new_start+5+target_nc], 0, 0.01)
                if new_conv.bias is not None:
                    nn.init.constant_(new_conv.bias.data[new_start+5:new_start+5+target_nc], 0)
        
        # 替换原始卷积层
        detection_layer.m[i] = new_conv
        
        # 更新批归一化层
        if bn is not None:
            new_bn = nn.BatchNorm2d(new_conv.out_channels)
            detection_layer.m_bn[i] = new_bn
    
    # 更新模型的类别名称
    if data_config:
        with open(data_config, 'r') as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)
        if 'names' in data_dict:
            model.names = data_dict['names']
    else:
        # 生成默认类别名称
        model.names = [f'class{i}' for i in range(target_nc)]
    
    print(f"✅ 成功适配检测头")
    print(f"   新类别数: {detection_layer.nc}")
    print(f"   新类别名称: {model.names}")
    
    # 保存适配后的权重
    if output_path is None:
        output_path = weights_path.replace('.pt', f'_adapted_{target_nc}classes.pt')
    
    # 更新checkpoint
    ckpt['model'] = model
    ckpt['epoch'] = -1  # 重置epoch
    if 'optimizer' in ckpt:
        del ckpt['optimizer']  # 删除优化器状态
    if 'training_results' in ckpt:
        del ckpt['training_results']  # 删除训练结果
    
    torch.save(ckpt, output_path)
    print(f"✅ 适配后权重保存到: {output_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='模型类别数适配工具')
    parser.add_argument('--weights', type=str, required=True, help='预训练权重路径')
    parser.add_argument('--nc', type=int, required=True, help='目标类别数')
    parser.add_argument('--data', type=str, help='数据配置文件路径')
    parser.add_argument('--output', type=str, help='输出权重路径')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.weights).exists():
        print(f"❌ 权重文件不存在: {args.weights}")
        return
    
    if args.data and not Path(args.data).exists():
        print(f"❌ 数据配置文件不存在: {args.data}")
        return
    
    # 执行适配
    success = adapt_model_classes(
        weights_path=args.weights,
        target_nc=args.nc,
        output_path=args.output,
        data_config=args.data
    )
    
    if success:
        print(f"🎉 模型适配完成！")
    else:
        print(f"❌ 模型适配失败！")

if __name__ == '__main__':
    main()
