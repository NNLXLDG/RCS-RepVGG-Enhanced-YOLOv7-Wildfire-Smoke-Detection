#!/usr/bin/env python3
"""
超参数文件验证脚本
验证所有超参数文件是否可以正确加载和解析
"""
import yaml
import os
from pathlib import Path

def validate_hyperparameter_file(file_path):
    """验证单个超参数文件"""
    try:
        with open(file_path, 'r') as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)
        
        # 检查必需的参数
        required_params = [
            'lr0', 'lrf', 'momentum', 'weight_decay',
            'box', 'cls', 'obj', 'iou_t', 'anchor_t',
            'hsv_h', 'hsv_s', 'hsv_v', 'fliplr', 'mosaic'
        ]
        
        missing_params = []
        for param in required_params:
            if param not in hyp:
                missing_params.append(param)
        
        if missing_params:
            return False, f"缺少参数: {missing_params}"
        
        # 检查参数值范围
        checks = [
            ('lr0', 0.001, 0.1, "学习率应在0.001-0.1之间"),
            ('lrf', 0.01, 1.0, "最终学习率比例应在0.01-1.0之间"),
            ('momentum', 0.8, 0.99, "动量应在0.8-0.99之间"),
            ('box', 0.01, 0.1, "box损失权重应在0.01-0.1之间"),
            ('cls', 0.1, 1.0, "cls损失权重应在0.1-1.0之间"),
            ('obj', 0.1, 2.0, "obj损失权重应在0.1-2.0之间"),
        ]
        
        warnings = []
        for param, min_val, max_val, msg in checks:
            if not (min_val <= hyp[param] <= max_val):
                warnings.append(f"{param}={hyp[param]} - {msg}")
        
        return True, warnings
        
    except yaml.YAMLError as e:
        return False, f"YAML解析错误: {e}"
    except FileNotFoundError:
        return False, "文件不存在"
    except Exception as e:
        return False, f"未知错误: {e}"

def main():
    """主函数"""
    hyperparams_dir = Path("hyperparameters")
    
    if not hyperparams_dir.exists():
        print("❌ hyperparameters 目录不存在")
        return
    
    # 获取所有超参数文件
    hyp_files = list(hyperparams_dir.glob("*.yaml"))
    
    if not hyp_files:
        print("❌ 未找到超参数文件")
        return
    
    print("🔍 验证超参数文件...")
    print("=" * 60)
    
    valid_count = 0
    
    for hyp_file in sorted(hyp_files):
        print(f"\n📄 验证: {hyp_file.name}")
        print("-" * 40)
        
        is_valid, result = validate_hyperparameter_file(hyp_file)
        
        if is_valid:
            print("✅ 文件格式正确")
            valid_count += 1
            
            if result:  # 有警告
                print("⚠️  注意事项:")
                for warning in result:
                    print(f"   - {warning}")
            else:
                print("🎯 所有参数都在推荐范围内")
        else:
            print(f"❌ 验证失败: {result}")
    
    print("\n" + "=" * 60)
    print(f"📊 验证结果: {valid_count}/{len(hyp_files)} 个文件有效")
    
    if valid_count == len(hyp_files):
        print("🎉 所有超参数文件验证通过！")
        
        # 显示文件用途
        print("\n📋 文件用途说明:")
        file_descriptions = {
            "hyp.scratch.p5.yaml": "标准P5模型 - 通用目标检测",
            "hyp.scratch.p6.yaml": "P6大模型 - 高精度检测",
            "hyp.scratch.tiny.yaml": "Tiny模型 - 轻量级快速推理",
            "hyp.scratch.custom.yaml": "自定义配置 - 用户自定义场景",
            "hyp.scratch.smokefire.yaml": "🔥 烟火检测专用 - 针对烟火特征优化"
        }
        
        for hyp_file in sorted(hyp_files):
            desc = file_descriptions.get(hyp_file.name, "未知用途")
            print(f"   {hyp_file.name:<30} - {desc}")
        
        print("\n🚀 推荐使用命令:")
        print("python train.py --data datasets/smokefire.yaml \\")
        print("                --cfg cfg/training/yolov7.yaml \\")
        print("                --hyp hyperparameters/hyp.scratch.smokefire.yaml")
    else:
        print(f"⚠️  {len(hyp_files) - valid_count} 个文件需要修复")

if __name__ == "__main__":
    main()
