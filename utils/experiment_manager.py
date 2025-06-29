"""
训练结果管理模块
统一管理训练、测试、检测结果的保存路径和命名规范
"""

import os
import yaml
from pathlib import Path
from datetime import datetime


def get_experiment_name(cfg_path, data_path, epochs, batch_size, additional_info=""):
    """
    生成标准化的实验名称
    
    Args:
        cfg_path: 模型配置文件路径
        data_path: 数据集配置文件路径  
        epochs: 训练轮数
        batch_size: 批次大小
        additional_info: 额外信息，如 repvgg, rcsosa 等
    
    Returns:
        标准化的实验名称
    """
    # 获取模型名称
    model_name = Path(cfg_path).stem if cfg_path else "unknown"
    
    # 获取数据集名称
    dataset_name = Path(data_path).stem if data_path else "unknown"
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构建实验名称
    name_parts = [model_name]
    
    if additional_info:
        name_parts.append(additional_info)
    
    name_parts.extend([
        dataset_name,
        f"ep{epochs}",
        f"bs{batch_size}",
        timestamp
    ])
    
    return "_".join(name_parts)


def create_experiment_directory(project_root, experiment_name, exist_ok=False):
    """
    创建实验目录并返回路径
    
    Args:
        project_root: 项目根目录 (如 runs/train)
        experiment_name: 实验名称
        exist_ok: 是否允许覆盖已存在的目录
    
    Returns:
        实验目录路径
    """
    experiment_dir = Path(project_root) / experiment_name
    
    if experiment_dir.exists() and not exist_ok:
        # 如果目录已存在且不允许覆盖，添加递增编号
        counter = 1
        while (experiment_dir.parent / f"{experiment_name}_{counter}").exists():
            counter += 1
        experiment_dir = experiment_dir.parent / f"{experiment_name}_{counter}"
    
    # 创建目录结构
    experiment_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = experiment_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    
    return experiment_dir


def save_experiment_config(save_dir, opt, hyp=None):
    """
    保存实验配置信息
    
    Args:
        save_dir: 保存目录
        opt: 命令行参数
        hyp: 超参数字典 (可选)
    """
    save_dir = Path(save_dir)
    
    # 保存命令行参数
    with open(save_dir / 'opt.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(vars(opt), f, sort_keys=False, allow_unicode=True)
    
    # 保存超参数
    if hyp is not None:
        with open(save_dir / 'hyp.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(hyp, f, sort_keys=False, allow_unicode=True)
    
    # 创建实验信息文件
    experiment_info = {
        'experiment_name': save_dir.name,
        'created_at': datetime.now().isoformat(),
        'model_config': getattr(opt, 'cfg', 'unknown'),
        'dataset_config': getattr(opt, 'data', 'unknown'),
        'epochs': getattr(opt, 'epochs', 0),
        'batch_size': getattr(opt, 'batch_size', 0),
        'device': getattr(opt, 'device', 'unknown'),
        'img_size': getattr(opt, 'img_size', [640, 640])
    }
    
    with open(save_dir / 'experiment_info.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(experiment_info, f, sort_keys=False, allow_unicode=True)


def create_training_report(save_dir, opt, hyp, results, best_fitness, training_time):
    """
    创建详细的训练报告
    """
    save_dir = Path(save_dir)
    
    report_content = f"""# YOLOv7 训练报告

## 实验信息
- **实验名称**: `{save_dir.name}`
- **训练时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总用时**: {training_time:.2f} 小时
- **训练目录**: `{save_dir}`

## 模型配置
- **模型配置文件**: `{opt.cfg}`
- **预训练权重**: `{opt.weights if opt.weights else '从头训练'}`
- **训练设备**: `{opt.device}`

## 数据集配置
- **数据集配置文件**: `{opt.data}`
- **图像尺寸**: {opt.img_size}
- **批次大小**: {opt.batch_size}
- **工作进程数**: {opt.workers}

## 训练参数
- **训练轮数**: {opt.epochs}
- **学习率策略**: {'线性衰减' if getattr(opt, 'linear_lr', False) else '余弦衰减'}
- **优化器**: {'Adam' if getattr(opt, 'adam', False) else 'SGD'}
- **多尺度训练**: {'是' if getattr(opt, 'multi_scale', False) else '否'}

## 超参数
"""
    
    if hyp:
        for key, value in hyp.items():
            report_content += f"- **{key}**: {value}\n"
    
    report_content += f"""
## 训练结果
- **最佳适应度**: {best_fitness:.4f}
- **精度 (P)**: {results[0]:.4f}
- **召回率 (R)**: {results[1]:.4f}
- **mAP@0.5**: {results[2]:.4f}
- **mAP@0.5:0.95**: {results[3]:.4f}

## 文件说明
- `weights/best.pt`: 最佳模型权重
- `weights/last.pt`: 最后一轮模型权重
- `results.txt`: 详细训练指标记录
- `results.png`: 训练曲线图表
- `hyp.yaml`: 使用的超参数配置
- `opt.yaml`: 命令行参数配置
- `experiment_info.yaml`: 实验基本信息
- `train_batch*.jpg`: 训练样本可视化
- `training_report.md`: 本训练报告

## 使用方法

### 检测推理
```bash
# 使用最佳权重进行检测
python detect.py --weights {save_dir}/weights/best.pt --source your_image.jpg

# 使用最后权重进行检测  
python detect.py --weights {save_dir}/weights/last.pt --source your_image.jpg
```

### 继续训练
```bash
# 从最后保存的权重继续训练
python train.py --weights {save_dir}/weights/last.pt --cfg {opt.cfg} --data {opt.data} --resume
```

### 模型评估
```bash
# 评估最佳模型
python test.py --weights {save_dir}/weights/best.pt --data {opt.data}
```

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*生成脚本: YOLOv7 训练结果管理系统*
"""
    
    # 保存报告
    report_file = save_dir / 'training_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_file


def organize_runs_directory():
    """
    整理runs目录，创建标准的子目录结构
    """
    runs_dir = Path('runs')
    
    # 标准子目录
    subdirs = [
        'train',      # 训练结果
        'test',       # 测试结果  
        'detect',     # 检测结果
        'export',     # 模型导出结果
        'hub'         # 模型仓库
    ]
    
    for subdir in subdirs:
        (runs_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # 创建runs目录说明文件
    readme_content = """# YOLOv7 运行结果目录

这个目录包含YOLOv7各种操作的运行结果。

## 目录结构

- `train/` - 训练结果，包含模型权重、训练曲线、配置文件等
- `test/` - 测试和验证结果，包含评估指标和可视化
- `detect/` - 检测推理结果，包含检测图片和结果文件
- `export/` - 模型导出结果，包含ONNX、TensorRT等格式
- `hub/` - 模型仓库相关文件

## 命名规范

训练实验目录命名格式：
`{模型名}_{变体}_{数据集}_{epochs}ep_{batch_size}bs_{时间戳}`

例如：
- `yolov7_smokefire_ep50_bs4_20250628_143022`
- `yolov7_repvgg_smokefire_ep100_bs8_20250628_150133`

## 自动整理

定期运行以下命令整理训练结果：
```bash
python organize_training_results.py
```
"""
    
    readme_file = runs_dir / 'README.md'
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    return runs_dir


def get_script_variant(script_path):
    """
    从脚本路径获取变体信息
    """
    script_name = Path(script_path).stem
    
    # 提取变体信息
    if 'repvgg-rcsosa' in script_name:
        return 'repvgg-rcsosa'
    elif 'repvgg' in script_name:
        return 'repvgg'
    elif 'rcsosa' in script_name:
        return 'rcsosa'
    elif 'tiny' in script_name:
        return 'tiny'
    else:
        return ''


def setup_training_directory(opt, script_path, hyp=None):
    """
    统一设置训练目录的函数
    
    Args:
        opt: 命令行参数对象
        script_path: 训练脚本路径 (用于确定变体)
        hyp: 超参数字典
    
    Returns:
        配置好的保存目录路径
    """
    # 确保runs目录结构存在
    organize_runs_directory()
    
    # 获取脚本变体信息
    variant = get_script_variant(script_path)
    
    # 生成实验名称
    experiment_name = get_experiment_name(
        cfg_path=opt.cfg,
        data_path=opt.data,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        additional_info=variant
    )
    
    # 创建实验目录
    save_dir = create_experiment_directory(
        project_root=opt.project,
        experiment_name=experiment_name,
        exist_ok=opt.exist_ok
    )
    
    # 保存配置信息
    save_experiment_config(save_dir, opt, hyp)
    
    return save_dir
