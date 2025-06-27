#!/usr/bin/env python3
"""
YOLOv7 项目配置验证脚本
检查配置文件、数据集、模型架构的一致性和合理性
"""

import os
import sys
import yaml
import torch
from pathlib import Path

def check_config_consistency():
    """检查配置文件的一致性"""
    print("🔍 检查配置文件一致性...")
    
    config_files = [
        'cfg/training/yolov7.yaml',
        'cfg/training/yolov7-repvgg.yaml', 
        'cfg/training/yolov7-rcsosa.yaml',
        'cfg/training/yolov7-repvgg-rcsosa.yaml'
    ]
    
    configs = {}
    for config_file in config_files:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                configs[config_file] = yaml.safe_load(f)
        else:
            print(f"❌ 配置文件不存在: {config_file}")
            return False
    
    # 检查关键参数一致性
    base_nc = configs['cfg/training/yolov7.yaml']['nc']
    base_anchors = configs['cfg/training/yolov7.yaml']['anchors']
    
    for config_file, config in configs.items():
        if config['nc'] != base_nc:
            print(f"❌ {config_file} 类别数不一致: {config['nc']} != {base_nc}")
            return False
        if config['anchors'] != base_anchors:
            print(f"⚠️  {config_file} 锚框配置不同，这可能是正常的")
    
    print("✅ 配置文件一致性检查通过")
    return True

def check_dataset_consistency():
    """检查数据集配置和实际文件的一致性"""
    print("🔍 检查数据集一致性...")
    
    dataset_config = 'datasets_smokefire/smokefire.yaml'
    if not os.path.exists(dataset_config):
        print(f"❌ 数据集配置文件不存在: {dataset_config}")
        return False
    
    with open(dataset_config, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # 检查数据集路径
    train_path = data_config['train']
    val_path = data_config['val']
    
    if not os.path.exists(train_path):
        print(f"❌ 训练集路径不存在: {train_path}")
        return False
    
    if not os.path.exists(val_path):
        print(f"❌ 验证集路径不存在: {val_path}")
        return False
    
    # 检查图片和标签数量
    train_images = len(list(Path(train_path).glob('*.jpg'))) + len(list(Path(train_path).glob('*.png')))
    train_labels = len(list(Path(train_path.replace('images', 'labels')).glob('*.txt')))
    
    val_images = len(list(Path(val_path).glob('*.jpg'))) + len(list(Path(val_path).glob('*.png')))
    val_labels = len(list(Path(val_path.replace('images', 'labels')).glob('*.txt')))
    
    print(f"📊 训练集: {train_images} 图片, {train_labels} 标签")
    print(f"📊 验证集: {val_images} 图片, {val_labels} 标签")
    
    if train_images != train_labels:
        print(f"⚠️  训练集图片和标签数量不匹配: {train_images} != {train_labels}")
    
    if val_images != val_labels:
        print(f"⚠️  验证集图片和标签数量不匹配: {val_images} != {val_labels}")
    
    # 检查类别数量
    nc = data_config['nc']
    names = data_config['names']
    
    if len(names) != nc:
        print(f"❌ 类别名称数量与nc不匹配: {len(names)} != {nc}")
        return False
    
    print("✅ 数据集一致性检查通过")
    return True

def check_training_scripts():
    """检查训练脚本的参数设置"""
    print("🔍 检查训练脚本参数...")
    
    train_scripts = [
        'train.py',
        'train-repvgg.py',
        'train-rcsosa.py', 
        'train-repvgg-rcsosa.py'
    ]
    
    all_good = True
    for script in train_scripts:
        if not os.path.exists(script):
            print(f"❌ 训练脚本不存在: {script}")
            all_good = False
            continue
        
        print(f"✅ {script} 存在")
    
    return all_good

def check_hyperparameters():
    """检查超参数文件"""
    print("🔍 检查超参数配置...")
    
    hyp_file = 'data/hyp.scratch.p5.yaml'
    if not os.path.exists(hyp_file):
        print(f"❌ 超参数文件不存在: {hyp_file}")
        return False
    
    with open(hyp_file, 'r') as f:
        hyp = yaml.safe_load(f)
    
    # 检查关键超参数
    required_params = ['lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 
                      'box', 'cls', 'obj', 'iou_t', 'anchor_t']
    
    for param in required_params:
        if param not in hyp:
            print(f"❌ 缺少关键超参数: {param}")
            return False
    
    # 检查合理性
    if hyp['lr0'] <= 0 or hyp['lr0'] > 1:
        print(f"⚠️  学习率可能不合理: {hyp['lr0']}")
    
    if hyp['weight_decay'] < 0:
        print(f"❌ 权重衰减不能为负: {hyp['weight_decay']}")
        return False
    
    print("✅ 超参数配置检查通过")
    return True

def check_pytorch_compatibility():
    """检查PyTorch兼容性"""
    print("🔍 检查PyTorch兼容性...")
    
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查CPU支持
    if torch.backends.mkldnn.is_available():
        print("✅ MKLDNN优化可用")
    else:
        print("⚠️  MKLDNN优化不可用")
    
    # 测试基本张量操作
    try:
        x = torch.randn(1, 3, 640, 640)
        y = torch.nn.Conv2d(3, 64, 3, 1, 1)(x)
        print("✅ 基本张量操作正常")
    except Exception as e:
        print(f"❌ 张量操作失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("🚀 YOLOv7 项目配置验证开始...")
    print("="*50)
    
    checks = [
        check_config_consistency,
        check_dataset_consistency, 
        check_training_scripts,
        check_hyperparameters,
        check_pytorch_compatibility
    ]
    
    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
        print()
    
    print("="*50)
    if all_passed:
        print("🎉 所有检查通过！项目配置正确，可以开始训练。")
        print("\n推荐的训练命令:")
        print("python train.py --data datasets_smokefire/smokefire.yaml --cfg cfg/training/yolov7.yaml --epochs 50 --batch-size 4")
    else:
        print("❌ 存在配置问题，请根据上述提示进行修复。")
        sys.exit(1)

if __name__ == '__main__':
    main()
