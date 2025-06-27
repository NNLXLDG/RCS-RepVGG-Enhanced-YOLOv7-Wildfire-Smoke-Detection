#!/usr/bin/env python3
"""
检查并修复数据集中的标签问题
"""
import os
import glob
from pathlib import Path

def check_and_fix_labels():
    """检查和修复标签文件中的类别索引"""
    dataset_path = "datasets_smokefire"
    max_valid_class = 1  # 对于2个类别，最大有效类别索引是1
    
    issues_found = []
    files_fixed = []
    
    # 检查训练和验证标签
    for split in ['train', 'val']:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        if not os.path.exists(labels_dir):
            continue
            
        label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
        
        for label_file in label_files:
            if os.path.basename(label_file) == 'classes.txt':
                continue  # 跳过类别定义文件
                
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                modified = False
                new_lines = []
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:  # 跳过空行
                        new_lines.append(line + '\n')
                        continue
                        
                    parts = line.split()
                    if len(parts) < 5:  # YOLO格式应该有5个值
                        issue = f"{label_file}:{line_num} - 格式错误: {line}"
                        issues_found.append(issue)
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        if class_id > max_valid_class:
                            issue = f"{label_file}:{line_num} - 无效类别 {class_id}, 修复为 {max_valid_class}"
                            issues_found.append(issue)
                            parts[0] = str(max_valid_class)
                            modified = True
                        elif class_id < 0:
                            issue = f"{label_file}:{line_num} - 负类别 {class_id}, 修复为 0"
                            issues_found.append(issue)
                            parts[0] = "0"
                            modified = True
                    except ValueError:
                        issue = f"{label_file}:{line_num} - 非数字类别: {parts[0]}"
                        issues_found.append(issue)
                        continue
                    
                    new_lines.append(' '.join(parts) + '\n')
                
                # 如果有修改，写回文件
                if modified:
                    with open(label_file, 'w') as f:
                        f.writelines(new_lines)
                    files_fixed.append(label_file)
                    
            except Exception as e:
                issue = f"{label_file} - 读取错误: {e}"
                issues_found.append(issue)
    
    # 打印结果
    print(f"检查完成!")
    print(f"发现问题: {len(issues_found)}")
    print(f"修复文件: {len(files_fixed)}")
    
    if issues_found:
        print("\n发现的问题:")
        for issue in issues_found[:10]:  # 只显示前10个
            print(f"  {issue}")
        if len(issues_found) > 10:
            print(f"  ... 还有 {len(issues_found) - 10} 个问题")
    
    if files_fixed:
        print("\n修复的文件:")
        for file in files_fixed:
            print(f"  {file}")
    
    # 统计类别分布
    print("\n类别分布统计:")
    class_counts = {}
    for split in ['train', 'val']:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        if not os.path.exists(labels_dir):
            continue
            
        label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
        
        for label_file in label_files:
            if os.path.basename(label_file) == 'classes.txt':
                continue
                
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 1:
                                try:
                                    class_id = int(parts[0])
                                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                                except ValueError:
                                    pass
            except:
                pass
    
    for class_id in sorted(class_counts.keys()):
        print(f"  类别 {class_id}: {class_counts[class_id]} 个标注")

if __name__ == "__main__":
    check_and_fix_labels()
