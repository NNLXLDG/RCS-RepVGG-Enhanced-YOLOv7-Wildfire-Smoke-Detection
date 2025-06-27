#!/usr/bin/env python3
"""
简化的训练测试脚本
用于验证修复后的代码是否正常工作
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

def test_model_loading():
    """测试模型加载"""
    print("🔍 测试模型加载...")
    try:
        from models.yolo import Model
        
        # 测试基本模型创建
        model = Model('cfg/training/yolov7.yaml', ch=3, nc=2)
        print("✅ 模型创建成功")
        
        # 测试前向传播
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            y = model(x)
        print("✅ 前向传播成功")
        
        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def test_dataloader():
    """测试数据加载器"""
    print("\n🔍 测试数据加载器...")
    try:
        from utils.datasets import create_dataloader
        from utils.torch_utils import select_device
        
        # 检查数据集路径
        if not os.path.exists('datasets_smokefire/train/images'):
            print("❌ 训练数据集不存在")
            return False
            
        # 创建简单的opt对象
        class SimpleOpt:
            def __init__(self):
                self.single_cls = False
                
        opt = SimpleOpt()
        device = select_device('')
        
        # 创建数据加载器
        dataloader, dataset = create_dataloader(
            'datasets_smokefire/train/images',
            640, 2, 32, opt,
            augment=False, cache=False, rect=False,
            rank=-1, world_size=1, workers=2,
            image_weights=False, quad=False, prefix='test: '
        )
        
        print(f"✅ 数据加载器创建成功，数据集大小: {len(dataset)}")
        
        # 测试获取一个batch
        for i, (imgs, targets, paths, _) in enumerate(dataloader):
            print(f"✅ 成功加载batch，图像形状: {imgs.shape}")
            break
            
        return True
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        return False

def test_hyperparameters():
    """测试超参数加载"""
    print("\n🔍 测试超参数加载...")
    try:
        with open('data/hyp.scratch.p5.yaml') as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)
        print("✅ 超参数加载成功")
        print(f"   学习率: {hyp.get('lr0', 'N/A')}")
        print(f"   批次大小相关: {hyp.get('box', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ 超参数加载失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试修复后的YOLOv7代码...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"设备: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    # 检查MPS支持
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 Apple MPS 设备可用")
    
    tests = [
        test_hyperparameters,
        test_model_loading,
        test_dataloader,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！代码修复成功！")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
