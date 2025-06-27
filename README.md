# YOLOv7 四版本对比项目

本项目用于比较YOLOv7的四个不同优化版本在野火烟雾检测任务上的性能。

## 四个版本说明

### 1. YOLOv7 基础版本 (yolov7.yaml)
- 标准YOLOv7架构
- 训练脚本: `train.py`
- 检测脚本: `detect.py`

### 2. YOLOv7-RepVGG版本 (yolov7-repvgg.yaml)
- 集成RepVGG重参数化模块
- 训练时多分支，推理时合并为单分支
- 训练脚本: `train-repvgg.py`
- 检测脚本: `detect-repvgg.py`

### 3. YOLOv7-RCSOSA版本 (yolov7-rcsosa.yaml)
- 集成RCSOSA(Residual Channel Shuffle and Spatial Attention)模块
- 改进的注意力机制
- 训练脚本: `train-rcsosa.py`
- 检测脚本: `detect-rcsosa.py`

### 4. YOLOv7-RepVGG-RCSOSA版本 (yolov7-repvgg-rcsosa.yaml)
- 同时集成RepVGG和RCSOSA模块
- 结合两种优化策略
- 训练脚本: `train-repvgg-rcsosa.py`
- 检测脚本: `detect-repvgg-rcsosa.py`

## 使用方法

### 训练
```bash
# 基础版本
python train.py --data datasets/smokefire.yaml --cfg cfg/training/yolov7.yaml --batch-size 4 --epochs 100

# RepVGG版本
python train-repvgg.py --data datasets/smokefire.yaml --cfg cfg/training/yolov7-repvgg.yaml --batch-size 4 --epochs 100

# RCSOSA版本
python train-rcsosa.py --data datasets/smokefire.yaml --cfg cfg/training/yolov7-rcsosa.yaml --batch-size 4 --epochs 100

# RepVGG+RCSOSA版本
python train-repvgg-rcsosa.py --data datasets/smokefire.yaml --cfg cfg/training/yolov7-repvgg-rcsosa.yaml --batch-size 4 --epochs 100
```

### 检测
```bash
# 使用对应的检测脚本和权重文件
python detect.py --weights runs/train/yolov7-baseline/weights/best.pt --source path/to/test/images
python detect-repvgg.py --weights runs/train/yolov7-repvgg/weights/best.pt --source path/to/test/images
python detect-rcsosa.py --weights runs/train/yolov7-rcsosa/weights/best.pt --source path/to/test/images
python detect-repvgg-rcsosa.py --weights runs/train/yolov7-repvgg-rcsosa/weights/best.pt --source path/to/test/images
```

### 自动对比所有四个版本 ⭐
```bash
# 完整对比（推荐）
python compare_versions.py --epochs 50 --batch-size 4

# 只对比特定版本
python compare_versions.py --versions yolov7 yolov7-repvgg --epochs 30
```

## 项目结构
```
├── cfg/training/          # 四个版本的配置文件
├── hyperparameters/       # 训练超参数配置（hyp.scratch.p5.yaml）
├── datasets/    # 数据集
├── models/                # 模型定义
├── utils/                 # 工具函数
├── runs/                  # 训练和检测输出结果
│   ├── train/             # 训练输出（权重文件、日志、可视化）
│   └── detect/            # 检测输出（带框标注的图片）
├── train*.py              # 四个训练脚本
├── detect*.py             # 四个检测脚本
├── test.py                # 测试脚本
├── compare_versions.py    # 四版本自动对比脚本
├── README.md              # 项目说明
├── QUICK_START.md         # 快速开始指南
└── requirements.txt       # 依赖包列表
```

## runs文件夹说明

`runs/`文件夹包含所有训练和检测的输出结果：

### runs/train/
每次训练会在此创建一个实验文件夹，包含：
- `weights/` - 训练权重文件（best.pt, last.pt）
- `results.txt` - 训练指标记录  
- `hyp.yaml` - 使用的超参数
- `opt.yaml` - 使用的训练配置

### runs/detect/
每次检测会创建一个实验文件夹，保存检测结果图片。

## 注意事项
- 所有版本都已针对Apple Silicon Mac CPU训练进行优化
- 建议使用相同的超参数进行公平比较
- 训练结果会保存在`runs/train/`目录下
- 使用`compare_versions.py`可以自动依次训练四个版本并生成对比报告
- 项目已精简，只保留核心文件和必要的训练输出
- **推荐配置**: epochs=50, batch-size=4, workers=1 (适合CPU训练)
- 数据集已优化为野火烟雾检测专用（2类：fire, smoke）













