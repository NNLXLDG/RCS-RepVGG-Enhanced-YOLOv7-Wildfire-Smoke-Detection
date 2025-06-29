# YOLOv7 测试脚本使用指南 (test.py)

## 概述

`test.py` 是YOLOv7项目的核心评估脚本，用于测试和评估训练好的模型性能。经过详细注释和安全性增强，该脚本能够：

- 🔧 **自动适配类别数不匹配问题**：智能处理预训练权重与目标数据集的类别数差异
- 🛡️ **多层安全检查**：防止索引越界、类别不匹配等常见错误
- 📊 **全面的性能评估**：计算mAP、precision、recall等多种指标
- 💾 **多格式结果输出**：支持TXT、JSON、可视化图表等多种输出格式
- ⚡ **性能优化**：支持半精度推理、模型追踪等加速技术

## 主要功能流程

### 1. 模型加载与自适应 🤖
```python
# 自动检测并修复类别数不匹配
if model_nc != nc:
    print("⚠️ 检测到类别数不匹配，正在自动修复...")
    model.model[-1].nc = nc  # 强制适配检测层
    model.nc = nc           # 强制适配模型
```

### 2. 数据预处理 📊
- 图像归一化 (0-255 → 0.0-1.0)
- 批次处理优化
- 坐标系统转换

### 3. 模型推理 🚀
- 前向传播计算
- 可选的测试时增强(TTA)
- 推理时间统计

### 4. 后处理与过滤 🔍
```python
# 智能过滤无效类别预测
valid_mask = (pred[:, 5] >= 0) & (pred[:, 5] < nc)
if not valid_mask.all():
    print("Warning: 过滤掉无效类别索引的预测")
    pred = pred[valid_mask]
```

### 5. 评估指标计算 📈
- AP (Average Precision) @ 0.5:0.95
- mAP (mean Average Precision)
- Precision & Recall
- F1-score
- 混淆矩阵

### 6. 结果输出 💾
- 控制台详细报告
- TXT格式标签文件
- JSON格式预测结果
- 可视化图表和混淆矩阵

## 使用方法

### 基本使用
```bash
# 在验证集上测试模型
python test.py --weights yolov7.pt --data datasets/smokefire.yaml

# 详细输出每个类别的性能
python test.py --weights yolov7.pt --data datasets/smokefire.yaml --verbose

# 保存预测结果
python test.py --weights yolov7.pt --data datasets/smokefire.yaml --save-txt --save-json
```

### 高级使用
```bash
# 使用测试时增强提高准确性
python test.py --weights yolov7.pt --data datasets/smokefire.yaml --augment

# 调整置信度和NMS阈值
python test.py --weights yolov7.pt --data datasets/smokefire.yaml --conf-thres 0.25 --iou-thres 0.45

# 速度基准测试
python test.py --task speed --weights yolov7.pt --data datasets/smokefire.yaml

# 多尺度性能研究
python test.py --task study --weights yolov7.pt --data datasets/smokefire.yaml
```

## 参数说明

### 核心参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | yolov7.pt | 模型权重文件路径 |
| `--data` | datasets/smokefire.yaml | 数据集配置文件 |
| `--batch-size` | 32 | 批次大小 |
| `--img-size` | 640 | 输入图像尺寸 |
| `--conf-thres` | 0.001 | 置信度阈值 |
| `--iou-thres` | 0.65 | NMS IoU阈值 |

### 输出控制
| 参数 | 说明 |
|------|------|
| `--save-txt` | 保存TXT格式标签 |
| `--save-json` | 保存JSON格式结果 |
| `--save-conf` | 在标签中包含置信度 |
| `--verbose` | 显示每类别详细指标 |

### 运行模式
| 参数 | 说明 |
|------|------|
| `--task val` | 在验证集上测试(默认) |
| `--task test` | 在测试集上测试 |
| `--task speed` | 速度基准测试 |
| `--task study` | 多尺度性能研究 |

## 输出结果解读

### 控制台输出
```
                 Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95
                   all        500       1000      0.847      0.821      0.884      0.576
                 smoke        300        600      0.856      0.833      0.891      0.582
                  fire        200        400      0.838      0.809      0.877      0.570
```

### 指标含义
- **P (Precision)**: 精确率 = TP/(TP+FP)
- **R (Recall)**: 召回率 = TP/(TP+FN)  
- **mAP@.5**: IoU阈值0.5时的平均精度
- **mAP@.5:.95**: IoU阈值0.5-0.95的平均精度

### 保存文件
- `runs/test/exp/`: 默认结果目录
- `labels/`: TXT格式标签文件
- `*_predictions.json`: JSON格式预测结果
- `confusion_matrix.png`: 混淆矩阵图
- `test_batch*_*.jpg`: 测试样本可视化

## 常见问题与解决方案

### 1. 类别数不匹配
**问题**: 使用COCO预训练权重测试自定义数据集时出现索引错误

**解决**: 脚本已自动处理此问题
```python
# 自动检测并修复
if model_nc != nc:
    model.model[-1].nc = nc  # 自动适配
```

### 2. 显存不足
**解决方案**:
```bash
# 减小批次大小
python test.py --batch-size 16

# 减小图像尺寸  
python test.py --img-size 512
```

### 3. 推理速度慢
**解决方案**:
```bash
# 禁用可视化
python test.py --task speed

# 使用模型追踪优化
python test.py --trace
```

### 4. 精度异常
**检查项目**:
- 数据集标签格式是否正确
- 置信度阈值是否合适
- 是否需要使用TTA增强

## 技术特性

### 🛡️ 安全性增强
- 多层类别索引检查
- 无效预测自动过滤
- 数组边界安全验证

### ⚡ 性能优化
- 半精度推理支持(GPU)
- 模型追踪优化
- 批次并行处理

### 🔧 自适应能力
- 自动类别数适配
- 智能设备选择
- 动态内存管理

### 📊 评估全面性
- 多种mAP计算方式
- 详细的类别统计
- 完整的性能分析

## 开发者说明

该脚本经过详细注释和安全性增强，主要改进包括：

1. **详细的中文注释**: 每个功能模块都有清晰的说明
2. **类别适配机制**: 自动处理预训练权重与目标数据集的不匹配
3. **多层安全检查**: 防止各种索引越界和类型错误
4. **结果路径规范化**: 与训练脚本保持一致的目录结构
5. **用户友好的错误提示**: 详细的警告和解决建议

这使得该脚本更加稳定、易用，特别适合从其他数据集迁移学习的场景。
