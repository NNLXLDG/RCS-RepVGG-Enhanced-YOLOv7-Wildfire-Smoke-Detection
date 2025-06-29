#!/usr/bin/env python3
"""
训练脚本输出格式测试
用于验证优化后的终端输出效果
"""

import logging
import sys
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_training_info_display():
    """测试训练开始信息显示效果"""
    
    def print_training_info(save_dir, model_cfg, dataset_cfg, epochs, batch_size):
        """
        打印训练配置信息和保存路径 - 优化版本，排版更清晰
        """
        # 使用更美观的分隔线和emoji图标
        separator = "━" * 80
        logger.info(f"\n{separator}")
        logger.info(f"🔥 YOLOv7 烟火检测模型训练 - 开始")
        logger.info(f"{separator}")
        
        # 使用表格式排版，左右对齐
        logger.info(f"┌─ 📁 保存目录      │ {save_dir}")
        logger.info(f"├─ 🏗️ 模型配置      │ {model_cfg}")
        logger.info(f"├─ 📊 数据集配置    │ {dataset_cfg}")
        logger.info(f"├─ 🔄 训练轮数      │ {epochs} epochs")
        logger.info(f"└─ 📦 批次大小      │ {batch_size} images/batch")
        
        logger.info(f"{separator}")
        logger.info(f"🚀 开始训练... 请耐心等待模型收敛")
        logger.info(f"{separator}\n")
    
    # 测试训练开始信息
    print("="*50)
    print("测试 1: 训练开始信息显示")
    print("="*50)
    
    print_training_info(
        save_dir="runs/train/exp_test",
        model_cfg="cfg/training/yolov7.yaml", 
        dataset_cfg="datasets/smokefire.yaml",
        epochs=50,
        batch_size=8
    )
    
def test_epoch_header_display():
    """测试epoch表头显示效果"""
    
    print("="*50)
    print("测试 2: Epoch 表头显示")
    print("="*50)
    
    epoch = 0
    epochs = 50
    
    # 打印美观的训练表头
    header_separator = "─" * 100
    logger.info(f"\n{header_separator}")
    logger.info(f"📊 Epoch {epoch+1:3d}/{epochs} 训练详情")
    logger.info(f"{header_separator}")
    
    # 优化的表头，使用更清晰的列名和对齐
    header_format = '%12s' * 8
    logger.info(header_format % (
        'Epoch', 'Memory', 'Box_Loss', 'Obj_Loss', 'Cls_Loss', 'Total_Loss', 'Targets', 'Img_Size'
    ))
    logger.info(header_format % (
        '轮次', '内存', '边界损失', '目标损失', '分类损失', '总损失', '目标数', '图像尺寸'
    ))
    logger.info(f"{header_separator}")

def test_training_progress_display():
    """测试训练过程中的信息显示"""
    
    print("="*50)
    print("测试 3: 训练过程信息显示")
    print("="*50)
    
    # 模拟一些训练数据
    epoch, epochs = 5, 50
    mloss = [0.0234, 0.0567, 0.0891, 0.1692]  # [box, obj, cls, total]
    targets_count = 12
    img_size = 640
    mem = "2.1MB"
    
    # 优化的格式化显示信息
    display_format = '%12s' * 2 + '%12.4f' * 4 + '%12d' * 2
    s = display_format % (
        f'{epoch+1}/{epochs}',          # 当前轮次/总轮次
        mem,                            # 内存使用
        mloss[0],                       # 边界框损失
        mloss[1],                       # 目标性损失  
        mloss[2],                       # 分类损失
        mloss[3],                       # 总损失
        targets_count,                  # 当前批次的目标数量
        img_size                        # 图像尺寸
    )
    
    logger.info(s)
    
    # Epoch结束总结
    epoch_separator = "┄" * 60
    logger.info(f"\n{epoch_separator}")
    logger.info(f"📈 Epoch {epoch+1}/{epochs} 完成 | 平均损失: {mloss[3]:.4f}")
    logger.info(f"   Box: {mloss[0]:.4f} | Obj: {mloss[1]:.4f} | Cls: {mloss[2]:.4f}")
    logger.info(f"{epoch_separator}")

def test_validation_results_display():
    """测试验证结果显示"""
    
    print("="*50)
    print("测试 4: 验证结果显示")
    print("="*50)
    
    # 模拟验证结果
    epoch = 10
    results = [0.856, 0.743, 0.824, 0.691, 0.0234, 0.0567, 0.0123]  # P, R, mAP@.5, mAP@.5-.95, val_loss
    times = [5.2, 12.3, 8.1]  # 推理时间
    
    # 优化验证结果显示
    val_separator = "┈" * 80
    logger.info(f"\n{val_separator}")
    logger.info(f"🔍 Epoch {epoch+1} 验证结果")
    logger.info(f"{val_separator}")
    logger.info(f"┌─ 📊 精度指标")
    logger.info(f"├─ 🎯 Precision    │ {results[0]:.4f}")
    logger.info(f"├─ 🔍 Recall       │ {results[1]:.4f}")
    logger.info(f"├─ 📈 mAP@0.5      │ {results[2]:.4f}")
    logger.info(f"├─ 📊 mAP@0.5:0.95 │ {results[3]:.4f}")
    logger.info(f"└─ ⚡ 推理速度    │ {times[1]:.1f}ms")
    
    logger.info(f"┌─ 📉 验证损失")
    logger.info(f"├─ 📦 Box Loss     │ {results[4]:.4f}")
    logger.info(f"├─ 🎯 Obj Loss     │ {results[5]:.4f}")
    logger.info(f"└─ 🏷️  Cls Loss     │ {results[6]:.4f}")
    logger.info(f"{val_separator}")

def test_training_completion_display():
    """测试训练完成信息显示"""
    
    print("="*50)
    print("测试 5: 训练完成信息显示")
    print("="*50)
    
    # 模拟训练完成数据
    training_time = 2.35
    epoch, start_epoch, epochs = 49, 0, 50
    best_fitness = 0.8234
    save_dir = Path("runs/train/exp_test")
    best = save_dir / "weights/best.pt"
    last = save_dir / "weights/last.pt"
    
    # 美观的训练完成信息显示
    completion_separator = "━" * 80
    logger.info(f"\n{completion_separator}")
    logger.info(f"🎉 训练完成！恭喜您成功训练了 YOLOv7 烟火检测模型")
    logger.info(f"{completion_separator}")
    
    # 训练总结信息，使用表格式排版
    logger.info(f"┌─ 📊 训练总结")
    logger.info(f"├─ ⏱️  总耗时        │ {training_time:.2f} 小时")
    logger.info(f"├─ 🔄 完成轮数      │ {epoch - start_epoch + 1} epochs")
    logger.info(f"├─ 🎯 最佳性能      │ {best_fitness:.4f}")
    logger.info(f"├─ 💾 模型保存路径  │ {save_dir}")
    logger.info(f"├─ 🏆 最佳权重      │ {best}")
    logger.info(f"└─ 📈 最新权重      │ {last}")
    logger.info(f"{completion_separator}")
    
    # 友好的后续使用提示
    logger.info(f"🚀 下一步操作建议:")
    logger.info(f"   1. 使用 test.py 评估模型性能")
    logger.info(f"   2. 使用 detect.py 进行推理检测")
    logger.info(f"   3. 查看 {save_dir}/results.png 分析训练曲线")
    logger.info(f"{completion_separator}\n")

def test_model_saving_display():
    """测试模型保存信息显示"""
    
    print("="*50)
    print("测试 6: 模型保存信息显示")
    print("="*50)
    
    # 模拟模型保存
    epoch = 25
    fi = 0.8456
    best_fitness = 0.8234
    wdir = Path("runs/train/exp_test/weights")
    best = wdir / "best.pt"
    last = wdir / "last.pt"
    
    if fi > best_fitness:
        logger.info(f"🏆 发现更好的模型！fitness: {fi:.4f} -> 已保存到 {best}")
    
    checkpoint_path = wdir / f'epoch_{epoch:03d}.pt'
    logger.info(f"💾 Epoch {epoch+1} 检查点已保存: {checkpoint_path}")
    logger.info(f"💾 最新模型已保存: {last}")

if __name__ == "__main__":
    print("🧪 YOLOv7 训练脚本输出格式测试")
    print("="*80)
    
    # 运行所有测试
    test_training_info_display()
    test_epoch_header_display()
    test_training_progress_display()
    test_validation_results_display()
    test_training_completion_display()
    test_model_saving_display()
    
    print("\n" + "="*80)
    print("✅ 所有输出格式测试完成！")
    print("现在您可以运行 train.py 查看优化后的输出效果")
    print("="*80)
