#!/usr/bin/env python3
"""
检查和安装数据集报告生成所需的依赖包
"""

import subprocess
import sys
from pathlib import Path

def check_and_install_packages():
    """检查并安装必要的Python包"""
    required_packages = [
        'matplotlib',
        'seaborn', 
        'opencv-python',
        'pandas',
        'numpy',
        'tqdm',
        'pyyaml'
    ]
    
    print("🔍 检查所需的Python包...")
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")
    
    if missing_packages:
        print(f"\n📦 需要安装 {len(missing_packages)} 个包:")
        for package in missing_packages:
            print(f"  - {package}")
        
        print("\n🚀 开始安装缺失的包...")
        try:
            for package in missing_packages:
                print(f"安装 {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} 安装完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 安装失败: {e}")
            print("请手动运行以下命令:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    print("\n✅ 所有依赖包检查完成!")
    return True

def create_requirements_file():
    """创建requirements.txt文件"""
    requirements = [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0", 
        "opencv-python>=4.5.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "tqdm>=4.60.0",
        "PyYAML>=6.0"
    ]
    
    requirements_path = Path("requirements_dataset_report.txt")
    
    with open(requirements_path, 'w') as f:
        f.write("# 数据集报告生成工具所需依赖包\n")
        f.write("# 安装命令: pip install -r requirements_dataset_report.txt\n\n")
        for req in requirements:
            f.write(f"{req}\n")
    
    print(f"📄 创建依赖文件: {requirements_path}")
    print("💡 您也可以使用以下命令安装所有依赖:")
    print(f"pip install -r {requirements_path}")

if __name__ == "__main__":
    print("🛠️ 数据集报告工具依赖检查")
    print("=" * 40)
    
    # 检查和安装包
    success = check_and_install_packages()
    
    # 创建requirements文件
    create_requirements_file()
    
    if success:
        print("\n🎉 环境准备完成!")
        print("现在可以运行数据集报告生成工具了:")
        print("python generate_dataset_report.py")
    else:
        print("\n⚠️ 环境准备遇到问题，请手动安装缺失的包")
