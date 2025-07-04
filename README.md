# YOLOv7 烟火检测项目 

**经过完整优化和注释的YOLOv7烟火检测解决方案**

## 使用
### 训练模型
```bash
# 基础训练
python train.py --data datasets/smokefire.yaml --cfg cfg/training/yolov7.yaml --weights yolov7.pt

# RepVGG架构
python train-repvgg.py --data datasets/smokefire.yaml --weights yolov7.pt

# RCSOSA架构
python train-rcsosa.py --data datasets/smokefire.yaml --weights yolov7.pt

# 混合架构
python train-repvgg-rcsosa.py --data datasets/smokefire.yaml --weights yolov7.pt
```

### 模型评估
```bash
# 基本评估
python test.py --weights runs/train/exp/weights/best.pt --data datasets/smokefire.yaml

# 详细评估(推荐)
python test.py --weights runs/train/exp/weights/best.pt --data datasets/smokefire.yaml --verbose --save-json

# 速度基准测试
python test.py --task speed --weights runs/train/exp/weights/best.pt --data datasets/smokefire.yaml
```

### 推理检测
```bash
# 图像检测
python detect.py --weights runs/train/exp/weights/best.pt --source inference/images/

# 视频检测
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/video.mp4

# 实时检测(摄像头)
python detect.py --weights runs/train/exp/weights/best.pt --source 0
```

## 项目结构

```
RCS_Yolo/
├── 核心脚本
│   ├── train.py                    # 主训练脚本(详细注释)
│   ├── train-repvgg.py            # RepVGG训练脚本
│   ├── train-rcsosa.py            # RCSOSA训练脚本
│   ├── train-repvgg-rcsosa.py     # 混合架构训练脚本
│   ├── test.py                    # 评估脚本(完整中文注释)
│   ├── detect.py                  # 主推理脚本
│   └── detect-*.py                # 各架构推理脚本
│
├── 工具脚本
│   ├── organize_training_results.py    # 训练结果整理工具
│   ├── adapt_model_classes.py         # 权重类别适配工具
│   ├── test_class_filtering.py        # 类别过滤测试工具
│   └── verify_test_script.py          # test.py验证脚本
│
├── 结果目录
│   ├── runs/                      # 当前训练/测试结果
│   │   ├── train/                # 训练结果
│   │   ├── detect/               # 推理结果
│   │   └── test/                 # 评估结果
│   └── training_results/          # 整理后的历史结果
│       ├── by_date/              # 按日期分类
│       ├── by_model/             # 按模型分类
│       └── README.md             # 结果说明
│
├── 项目组件
│   ├── models/                   # 模型定义
│   ├── utils/                    # 工具函数
│   │   └── experiment_manager.py # 实验管理器
│   ├── cfg/                      # 配置文件
│   └── data/                     # 数据集配置
│
└── 文档资料
    └── docs/                     # 技术文档归档
        ├── README.md             # 文档索引
        ├── ACCURATE_PROBLEM_ANALYSIS.md    # 权重适配问题分析
        ├── CLASS_INDEX_FIX.md              # 类别索引修复说明
        ├── HYPERPARAMETER_COMPARISON.md    # 超参数对比分析
        ├── HYPERPARAMETER_OPTIMIZATION_LOGIC.md  # 超参数优化逻辑
        ├── REFACTORING_SUMMARY.md          # 代码重构总结
        ├── TEST_SCRIPT_GUIDE.md            # test.py详细使用指南
        ├── QUICK_START.md                  # 快速开始指南
        └── TRAINING_MANAGEMENT.md          # 训练管理指南
```


## 工具使用

### 训练结果整理
```bash
# 自动整理所有训练结果
python organize_training_results.py

# 查看整理后的结果
ls training_results/
```

### 权重类别适配
```bash
# 手动适配权重文件类别数
python adapt_model_classes.py --weights yolov7.pt --nc 2 --save adapted_yolov7.pt
```

### 类别过滤测试
```bash
# 验证类别过滤逻辑
python test_class_filtering.py
```


## 详细文档

- **[技术文档目录](docs/README.md)**: 完整的技术文档索引
- **[快速开始指南](docs/QUICK_START.md)**: 新手入门指南
- **[权重适配分析](docs/ACCURATE_PROBLEM_ANALYSIS.md)**: 权重加载问题深度分析
- **[超参数优化](docs/HYPERPARAMETER_OPTIMIZATION_LOGIC.md)**: 烟火检测专用超参数优化
- **[代码重构总结](docs/REFACTORING_SUMMARY.md)**: test.py脚本优化过程
- **[卷积核参数详解](docs/YOLOv7-Kernel-Size-Visualization.md)**: 卷积核大小(kernel_size)参数详细解析和可视化
- **[通道数详解](docs/YOLOv7-Channels-Explained.md)**: 卷积通道数(channels)参数详细解析和计算
- **[特征图和滤波器详解](docs/YOLOv7-Feature-Maps-and-Filters.md)**: 特征图和卷积滤波器的工作原理与可视化
- **[YOLOv7架构说明](docs/YOLOv7-RepVGG-RCSOSA-README.md)**: RepVGG-RCSOSA架构详细说明和参数映射


## 使用建议

1. **首次使用**: 建议从 `train.py` 开始，使用默认配置训练基础模型
2. **性能优化**: 尝试不同架构的训练脚本，比较性能差异
3. **结果分析**: 使用 `test.py --verbose` 获得详细的性能分析
4. **实验管理**: 定期运行 `organize_training_results.py` 整理实验结果


---

