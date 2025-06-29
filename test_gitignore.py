#!/usr/bin/env python3
"""
测试 .gitignore 文件是否正确忽略数据集相关文件
"""

import os
import subprocess
import tempfile
import shutil

def test_gitignore_rules():
    """测试 .gitignore 规则是否正常工作"""
    print("🧪 测试 .gitignore 数据集忽略规则...")
    
    # 测试目录和文件列表
    test_items = [
        # 数据集目录
        "datasets/new_dataset/",
        "data/custom_data/",
        "dataset/yolo_data/",
        "my_dataset/",
        "custom_datasets/",
        
        # 数据集文件
        "datasets/test.cache",
        "datasets/labels.cache",
        "datasets/images.cache",
        "custom_dataset.zip",
        "data.tar.gz",
        "dataset.dataset",
        
        # 配置文件备份
        "datasets/config.yaml.bak",
        "data/hyp.yaml.backup",
        
        # 常见数据目录
        "annotations/",
        "images/train/",
        "labels/valid/",
        "train/data/",
        "valid/images/",
        "test/labels/",
        "val/annotations/",
        
        # COCO数据集
        "coco/",
        "coco128/",
        "yolo_data/"
    ]
    
    created_items = []
    
    try:
        # 创建测试文件和目录
        for item in test_items:
            if item.endswith('/'):
                # 创建目录
                os.makedirs(item, exist_ok=True)
                # 在目录中创建一个测试文件
                test_file = os.path.join(item, 'test.txt')
                with open(test_file, 'w') as f:
                    f.write('test content')
                created_items.append(item)
                created_items.append(test_file)
            else:
                # 创建文件
                dir_path = os.path.dirname(item)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                with open(item, 'w') as f:
                    f.write('test content')
                created_items.append(item)
        
        # 检查 git status
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Git 命令执行失败")
            return False
        
        # 检查是否有测试文件出现在 git status 中
        git_tracked = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        ignored_correctly = True
        for line in git_tracked:
            if line.startswith('??'):  # 未跟踪的文件
                file_path = line[3:]  # 去掉前缀 "?? "
                
                # 检查是否是我们创建的测试文件
                for test_item in created_items:
                    if file_path.startswith(test_item.rstrip('/')):
                        print(f"❌ 文件/目录应该被忽略但未被忽略: {file_path}")
                        ignored_correctly = False
        
        if ignored_correctly:
            print("✅ 所有数据集相关文件都被正确忽略!")
            return True
        else:
            return False
            
    finally:
        # 清理创建的测试文件和目录
        for item in created_items:
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                elif os.path.isfile(item):
                    os.remove(item)
            except:
                pass

def main():
    """主函数"""
    print("=" * 60)
    print("📁 Git Ignore 测试工具")
    print("=" * 60)
    
    if not os.path.exists('.gitignore'):
        print("❌ 找不到 .gitignore 文件")
        return False
    
    # 显示当前的数据集忽略规则
    print("\n📄 当前的数据集忽略规则:")
    with open('.gitignore', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    in_dataset_section = False
    for line in lines:
        line = line.strip()
        if '数据集' in line or 'dataset' in line.lower():
            in_dataset_section = True
        elif line.startswith('#') and in_dataset_section and '数据集' not in line and 'dataset' not in line.lower():
            in_dataset_section = False
        elif in_dataset_section and line and not line.startswith('#'):
            print(f"  • {line}")
    
    # 运行测试
    print()
    success = test_gitignore_rules()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 所有测试通过! .gitignore 配置正确。")
    else:
        print("⚠️  部分测试失败，请检查 .gitignore 配置。")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    main()
