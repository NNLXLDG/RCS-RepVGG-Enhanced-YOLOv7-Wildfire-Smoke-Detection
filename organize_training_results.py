#!/usr/bin/env python3
"""
训练结果整理脚本
用于整理runs/train/目录下的训练结果，使文件夹结构井井有条
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
import yaml

def load_training_info(train_dir):
    """从训练目录中加载训练信息"""
    info = {
        'model': 'unknown',
        'date': 'unknown', 
        'epochs': 0,
        'dataset': 'unknown',
        'batch_size': 0
    }
    
    # 从opt.yaml读取配置信息
    opt_file = train_dir / 'opt.yaml'
    if opt_file.exists():
        try:
            with open(opt_file, 'r') as f:
                opt = yaml.safe_load(f)
                if opt:
                    info['model'] = Path(opt.get('cfg', 'unknown')).stem
                    info['epochs'] = opt.get('epochs', 0)
                    info['dataset'] = Path(opt.get('data', 'unknown')).stem
                    info['batch_size'] = opt.get('batch_size', 0)
        except:
            pass
    
    # 从文件夹修改时间推断训练日期
    try:
        timestamp = train_dir.stat().st_mtime
        info['date'] = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M')
    except:
        pass
    
    return info

def get_folder_size(folder_path):
    """计算文件夹大小（MB）"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except:
                    pass
    except:
        pass
    return total_size / (1024 * 1024)  # 转换为MB

def has_valid_weights(train_dir):
    """检查是否有有效的权重文件"""
    weights_dir = train_dir / 'weights'
    if not weights_dir.exists():
        return False
    
    # 检查是否有best.pt或last.pt
    has_best = (weights_dir / 'best.pt').exists()
    has_last = (weights_dir / 'last.pt').exists()
    
    return has_best or has_last

def organize_training_results():
    """整理训练结果"""
    runs_dir = Path('runs/train')
    if not runs_dir.exists():
        print("未找到runs/train目录")
        return
    
    # 创建整理后的目录结构
    organized_dir = Path('training_results')
    archived_dir = organized_dir / 'archived'
    current_dir = organized_dir / 'current'
    
    # 创建目录
    for dir_path in [organized_dir, archived_dir, current_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # 扫描所有训练文件夹
    train_folders = []
    for folder in runs_dir.iterdir():
        if folder.is_dir():
            info = load_training_info(folder)
            info['path'] = folder
            info['size'] = get_folder_size(folder)
            info['has_weights'] = has_valid_weights(folder)
            train_folders.append(info)
    
    # 按模型和日期分组
    model_groups = {}
    incomplete_runs = []
    
    for folder_info in train_folders:
        model_name = folder_info['model']
        
        # 如果没有有效权重文件，归类为不完整训练
        if not folder_info['has_weights'] or folder_info['size'] < 10:  # 小于10MB认为是不完整的
            incomplete_runs.append(folder_info)
            continue
        
        if model_name not in model_groups:
            model_groups[model_name] = []
        model_groups[model_name].append(folder_info)
    
    # 整理完整的训练结果
    summary = {
        'organized_at': datetime.now().isoformat(),
        'models': {},
        'archived': [],
        'removed': []
    }
    
    for model_name, folders in model_groups.items():
        # 按日期排序，最新的在前
        folders.sort(key=lambda x: x['date'], reverse=True)
        
        model_dir = current_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # 保留最新的2个训练结果到current，其余归档
        for i, folder_info in enumerate(folders):
            if i < 2:  # 保留最新的2个
                target_name = f"{folder_info['date']}_epochs{folder_info['epochs']}_bs{folder_info['batch_size']}"
                target_path = model_dir / target_name
                
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.move(str(folder_info['path']), str(target_path))
                
                print(f"✅ 保留: {model_name} -> {target_name}")
                
                if model_name not in summary['models']:
                    summary['models'][model_name] = []
                summary['models'][model_name].append({
                    'name': target_name,
                    'size_mb': folder_info['size'],
                    'epochs': folder_info['epochs']
                })
                
            else:  # 归档旧的训练结果
                archive_model_dir = archived_dir / model_name
                archive_model_dir.mkdir(exist_ok=True)
                
                target_name = f"{folder_info['date']}_epochs{folder_info['epochs']}"
                target_path = archive_model_dir / target_name
                
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.move(str(folder_info['path']), str(target_path))
                
                print(f"📦 归档: {model_name} -> archived/{target_name}")
                summary['archived'].append(f"{model_name}/{target_name}")
    
    # 处理不完整的训练
    print(f"\n🗑️  发现 {len(incomplete_runs)} 个不完整的训练结果:")
    for folder_info in incomplete_runs:
        folder_name = folder_info['path'].name
        size = folder_info['size']
        has_weights = folder_info['has_weights']
        
        print(f"   - {folder_name} (大小: {size:.1f}MB, 有权重: {has_weights})")
        
        # 询问是否删除
        response = input(f"     是否删除 {folder_name}? [y/N]: ").lower()
        if response == 'y':
            shutil.rmtree(folder_info['path'])
            print(f"     ✅ 已删除 {folder_name}")
            summary['removed'].append(folder_name)
        else:
            # 移动到archived目录
            target_path = archived_dir / 'incomplete' / folder_name
            target_path.parent.mkdir(exist_ok=True)
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.move(str(folder_info['path']), str(target_path))
            print(f"     📦 已归档到 archived/incomplete/{folder_name}")
            summary['archived'].append(f"incomplete/{folder_name}")
    
    # 保存整理摘要
    summary_file = organized_dir / 'organization_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 创建README
    readme_content = f"""# 训练结果整理

整理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 目录结构

```
training_results/
├── current/          # 当前保留的训练结果（每个模型最多2个最新版本）
│   ├── yolov7/
│   ├── yolov7-repvgg/
│   └── ...
├── archived/         # 归档的训练结果
│   ├── yolov7/
│   ├── incomplete/   # 不完整的训练
│   └── ...
└── organization_summary.json  # 整理摘要
```

## 当前保留的模型

"""
    
    for model_name, folders in summary['models'].items():
        readme_content += f"\n### {model_name}\n\n"
        for folder in folders:
            readme_content += f"- **{folder['name']}**: {folder['size_mb']:.1f}MB, {folder['epochs']} epochs\n"
    
    readme_content += f"""
## 整理统计

- 保留模型数量: {len(summary['models'])}
- 归档结果数量: {len(summary['archived'])}
- 删除结果数量: {len(summary['removed'])}

## 使用说明

1. `current/` 目录包含每个模型的最新训练结果
2. `archived/` 目录包含历史训练结果和不完整的训练
3. 权重文件位于各自目录的 `weights/` 子目录中
4. 可以通过 `organization_summary.json` 查看详细的整理信息
"""
    
    readme_file = organized_dir / 'README.md'
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n✅ 整理完成!")
    print(f"📁 整理后的结果保存在: {organized_dir}")
    print(f"📊 详细信息请查看: {summary_file}")
    print(f"📖 说明文档: {readme_file}")

if __name__ == '__main__':
    print("🗂️  开始整理训练结果...")
    organize_training_results()
