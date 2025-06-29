# YOLOv7 训练脚本统一管理说明

## 📋 概述

所有训练脚本已统一使用 `utils/experiment_manager.py` 进行实验管理，确保训练结果目录结构、命名规范、配置保存和报告生成的一致性。

## 🗂️ 统一的目录命名规范

所有训练脚本现在使用统一的目录命名格式：

```
{模型名}_{变体}_{数据集}_{epochs}ep_{batch_size}bs_{时间戳}
```

### 示例

| 训练脚本 | 生成目录名 |
|---------|-----------|
| `train.py` | `yolov7_smokefire_ep50_bs4_20250628_143022` |
| `train-repvgg.py` | `yolov7-repvgg_repvgg_smokefire_ep100_bs8_20250628_150133` |
| `train-rcsosa.py` | `yolov7-rcsosa_rcsosa_smokefire_ep75_bs6_20250628_152245` |
| `train-repvgg-rcsosa.py` | `yolov7-repvgg-rcsosa_repvgg-rcsosa_smokefire_ep200_bs2_20250628_154357` |

## 📁 统一的目录结构

每个训练实验目录包含：

```
训练实验目录/
├── weights/                    # 模型权重
│   ├── best.pt                # 最佳模型权重
│   └── last.pt                # 最后保存的权重
├── results.txt                # 训练指标记录
├── results.png                # 训练曲线图
├── opt.yaml                   # 命令行参数配置
├── hyp.yaml                   # 超参数配置
├── experiment_info.yaml       # 实验基本信息
├── training_report.md         # 详细训练报告
└── train_batch*.jpg           # 训练样本可视化
```

## 🚀 训练脚本使用

所有训练脚本保持原有的命令行参数，但结果保存更加规范：

### 基础训练
```bash
# 标准YOLOv7
python train.py --epochs 50 --batch-size 4

# RepVGG变体
python train-repvgg.py --epochs 100 --batch-size 8

# RCS-OSA变体  
python train-rcsosa.py --epochs 75 --batch-size 6

# RepVGG + RCS-OSA变体
python train-repvgg-rcsosa.py --epochs 200 --batch-size 2
```

### 自定义配置
```bash
# 使用自定义数据集和配置
python train.py \
  --cfg cfg/training/yolov7.yaml \
  --data datasets/custom.yaml \
  --hyp hyperparameters/hyp.custom.yaml \
  --epochs 100 \
  --batch-size 8
```

## 📊 训练报告

每次训练完成后自动生成详细的训练报告 (`training_report.md`)，包含：

- 🔧 实验配置信息
- 📈 训练结果和指标
- 📂 文件说明和用途
- 🎯 推理和评估命令示例
- 🔄 继续训练的命令

## 🗂️ 结果整理

### 自动整理脚本

```bash
# 整理训练结果，归档旧实验，保留最新实验
python organize_training_results.py
```

整理后的目录结构：
```
training_results/
├── current/              # 当前保留的训练结果（每个模型最多2个最新版本）
│   ├── yolov7/
│   ├── yolov7-repvgg/
│   ├── yolov7-rcsosa/
│   └── yolov7-repvgg-rcsosa/
├── archived/             # 归档的训练结果
│   ├── yolov7/
│   ├── yolov7-repvgg/
│   ├── yolov7-rcsosa/
│   ├── yolov7-repvgg-rcsosa/
│   └── incomplete/       # 不完整的训练
└── organization_summary.json  # 整理摘要
```

## 🔧 技术实现

### 核心模块：`utils/experiment_manager.py`

提供以下统一功能：

1. **`get_experiment_name()`** - 生成标准化实验名称
2. **`setup_training_directory()`** - 创建统一的实验目录
3. **`save_experiment_config()`** - 保存实验配置信息
4. **`create_training_report()`** - 生成详细训练报告
5. **`organize_runs_directory()`** - 创建标准目录结构

### 脚本变更

所有训练脚本已更新：

- ✅ 导入 `experiment_manager` 模块
- ✅ 使用 `setup_training_directory()` 生成保存目录
- ✅ 调用 `create_training_report()` 生成训练报告
- ✅ 保持原有的命令行参数兼容性

## 🎯 优势

1. **命名规范** - 所有实验目录名称清晰、一致、包含关键信息
2. **结构统一** - 所有训练结果目录结构相同，便于管理
3. **信息完整** - 自动保存配置、生成报告，便于复现和分析
4. **易于查找** - 通过目录名即可快速识别实验配置
5. **自动归档** - 提供整理脚本，避免结果目录混乱

## 🔍 验证测试

运行统一系统测试：

```bash
python test_unified_training.py
```

测试包括：
- ✅ 实验命名系统验证
- ✅ 目录结构创建测试  
- ✅ 脚本导入验证
- ✅ 功能完整性检查

## 📋 使用建议

1. **定期整理** - 每周运行一次 `organize_training_results.py`
2. **查看报告** - 每次训练后查看生成的 `training_report.md`
3. **备份重要结果** - 将最佳模型权重备份到安全位置
4. **记录实验** - 在项目文档中记录重要实验的目录名和结果

---

