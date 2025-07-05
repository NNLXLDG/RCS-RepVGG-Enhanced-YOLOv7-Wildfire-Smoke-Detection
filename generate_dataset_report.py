#!/usr/bin/env python3
"""
快速数据集报告生成脚本
专为当前项目的数据集配置优化

使用方法:
python generate_dataset_report.py
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

try:
    from dataset_report import DatasetAnalyzer
except ImportError:
    print("❌ 错误: 无法导入dataset_report模块")
    print("请确保dataset_report.py文件在同一目录下")
    sys.exit(1)

def find_dataset_configs():
    """查找项目中的数据集配置文件"""
    current_dir = Path(__file__).parent
    dataset_configs = []
    
    # 常见的数据集配置文件位置
    search_paths = [
        current_dir / "data",
        current_dir / "datasets", 
        current_dir / "cfg",
        current_dir
    ]
    
    for search_path in search_paths:
        if search_path.exists():
            # 查找.yaml文件
            for yaml_file in search_path.rglob("*.yaml"):
                if any(keyword in yaml_file.name.lower() for keyword in ['data', 'dataset', 'coco', 'custom']):
                    dataset_configs.append(yaml_file)
    
    return dataset_configs

def create_sample_dataset_config():
    """创建示例数据集配置文件"""
    sample_config = """# 示例数据集配置文件
# 根据您的实际数据集路径修改以下配置

# 数据集基本信息
nc: 1  # 类别数量 (根据yolov4-csp-IDetect.yaml中的nc: 1)
names: 
  0: 'object'  # 替换为您的实际类别名称，如 'fire', 'smoke' 等

# 数据集路径 (请根据实际情况修改)
train: datasets/train/images  # 训练集图像路径
val: datasets/val/images      # 验证集图像路径
test: datasets/test/images    # 测试集图像路径 (可选)

# 数据集根目录 (可选)
path: datasets/

# 下载信息 (可选)
download: |
  # 在这里可以添加数据集下载脚本
  echo "请手动下载并配置数据集"
"""
    
    config_path = Path("data/custom_dataset.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(sample_config)
    
    return config_path

def main():
    print("🔍 YOLOv7 数据集报告生成器")
    print("=" * 50)
    
    # 查找数据集配置文件
    dataset_configs = find_dataset_configs()
    
    if not dataset_configs:
        print("⚠️ 未找到数据集配置文件")
        print("\n创建示例配置文件...")
        sample_config = create_sample_dataset_config()
        print(f"✅ 示例配置文件已创建: {sample_config}")
        print("\n请编辑该文件，设置正确的数据集路径和类别信息，然后重新运行此脚本。")
        return
    
    print(f"📁 找到 {len(dataset_configs)} 个数据集配置文件:")
    for i, config in enumerate(dataset_configs, 1):
        print(f"  {i}. {config}")
    
    # 选择配置文件
    if len(dataset_configs) == 1:
        selected_config = dataset_configs[0]
        print(f"\n🎯 自动选择: {selected_config}")
    else:
        print(f"\n请选择要分析的数据集配置 (1-{len(dataset_configs)}):")
        try:
            choice = int(input("输入选项编号: ")) - 1
            if 0 <= choice < len(dataset_configs):
                selected_config = dataset_configs[choice]
            else:
                print("❌ 无效选择，使用第一个配置文件")
                selected_config = dataset_configs[0]
        except ValueError:
            print("❌ 无效输入，使用第一个配置文件")
            selected_config = dataset_configs[0]
    
    # 设置输出目录
    output_dir = Path("reports") / f"dataset_analysis_{selected_config.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 开始分析数据集: {selected_config}")
    print(f"📂 报告将保存到: {output_dir}")
    
    try:
        # 创建分析器
        analyzer = DatasetAnalyzer(selected_config, output_dir)
        
        # 执行分析
        print("\n" + "="*50)
        analyzer.analyze_dataset()
        
        # 生成可视化图表
        print("\n" + "="*50)
        analyzer.generate_plots()
        
        # 生成报告
        print("\n" + "="*50)
        analyzer.generate_report()
        
        print("\n" + "="*50)
        print("🎉 数据集分析完成!")
        print(f"\n📊 查看报告:")
        print(f"  🌐 HTML报告: {output_dir / 'dataset_report.html'}")
        print(f"  📄 JSON报告: {output_dir / 'dataset_report.json'}")
        print(f"  📋 CSV报告: {output_dir / 'class_statistics.csv'}")
        print(f"  📈 可视化图表: {output_dir}")
        
        # 提供打开报告的建议
        html_report = output_dir / 'dataset_report.html'
        if html_report.exists():
            print(f"\n💡 提示: 可以在浏览器中打开 {html_report.absolute()} 查看完整报告")
        
        # 输出关键统计信息
        print(f"\n📈 关键统计:")
        print(f"  📸 总图像数: {analyzer.stats['total_images']:,}")
        print(f"  🏷️ 总标注数: {analyzer.stats['total_annotations']:,}")
        print(f"  📂 类别数量: {len(analyzer.stats['classes'])}")
        print(f"  📊 平均每图标注数: {sum(analyzer.stats['annotations_per_image'])/len(analyzer.stats['annotations_per_image']):.2f}" 
              if analyzer.stats['annotations_per_image'] else "  📊 平均每图标注数: 0")
        
        if analyzer.stats['empty_images'] > 0:
            print(f"  ⚠️ 空图像: {analyzer.stats['empty_images']}")
        if analyzer.stats['corrupted_images'] > 0:
            print(f"  ⚠️ 损坏图像: {analyzer.stats['corrupted_images']}")
            
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("请检查数据集配置文件中的路径是否正确")
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 常见解决方案:")
        print("  1. 检查数据集路径是否正确")
        print("  2. 确保图像和标注文件存在")
        print("  3. 检查标注文件格式是否正确 (YOLO格式)")
        print("  4. 确保安装了所需的Python包 (matplotlib, seaborn, opencv-python等)")

if __name__ == "__main__":
    main()
