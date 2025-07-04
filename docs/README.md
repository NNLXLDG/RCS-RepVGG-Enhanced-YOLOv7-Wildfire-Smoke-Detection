# YOLOv7 烟火检测项目文档

本文件夹包含了YOLOv7烟火检测项目的所有技术文档和说明。

## 📋 文档目录

### 🔧 核心技术分析
- **[ACCURATE_PROBLEM_ANALYSIS.md](./ACCURATE_PROBLEM_ANALYSIS.md)** - 权重适配问题的深度分析与最佳实践
- **[CLASS_INDEX_FIX.md](./CLASS_INDEX_FIX.md)** - 类别索引适配问题的详细说明与解决方案
- **[ROOT_CAUSE_SOLUTION.md](./ROOT_CAUSE_SOLUTION.md)** - 根本原因分析与解决方案

### 📐 架构设计文档
- **[YOLOv7-RepVGG-RCSOSA-README.md](./YOLOv7-RepVGG-RCSOSA-README.md)** - YOLOv7 RepVGG-RCSOSA架构详细说明
- **[YOLOv7-RepVGG-RCSOSA-Architecture.md](./YOLOv7-RepVGG-RCSOSA-Architecture.md)** - 架构逐步解释
- **[YOLOv7-RepVGG-RCSOSA-Parameters-Mapping.md](./YOLOv7-RepVGG-RCSOSA-Parameters-Mapping.md)** - YAML参数到图示模块映射表
- **[YOLOv7-RepVGG-RCSOSA-Visualization.md](./YOLOv7-RepVGG-RCSOSA-Visualization.md)** - 配置层到架构的可视化映射
- **[YOLOv7-Kernel-Size-Visualization.md](./YOLOv7-Kernel-Size-Visualization.md)** - 卷积核大小参数详细解析和可视化(英文)
- **[YOLOv7-Kernel-Size-Visualization-CN.md](./YOLOv7-Kernel-Size-Visualization-CN.md)** - 卷积核大小参数详细解析和可视化(中文)
- **[YOLOv7-Channels-Explained.md](./YOLOv7-Channels-Explained.md)** - 通道数参数详细解析(英文)
- **[YOLOv7-Channels-Explained-CN.md](./YOLOv7-Channels-Explained-CN.md)** - 通道数参数详细解析(中文)
- **[YOLOv7-Feature-Maps-and-Filters.md](./YOLOv7-Feature-Maps-and-Filters.md)** - 特征图和滤波器详细解析(英文)
- **[YOLOv7-Feature-Maps-and-Filters-CN.md](./YOLOv7-Feature-Maps-and-Filters-CN.md)** - 特征图和滤波器详细解析(中文)
- **[YOLOv7-YAML-Format-Explained.md](./YOLOv7-YAML-Format-Explained.md)** - YAML配置文件格式详细解释
- **[YOLOv7-YAML-Parameters-Illustrated.md](./YOLOv7-YAML-Parameters-Illustrated.md)** - YAML参数图形化解析(英文)
- **[YOLOv7-YAML-Parameters-Illustrated-CN.md](./YOLOv7-YAML-Parameters-Illustrated-CN.md)** - YAML参数图形化解析(中文)

### 📈 超参数优化
- **[HYPERPARAMETER_COMPARISON.md](./HYPERPARAMETER_COMPARISON.md)** - 超参数文件对比分析
- **[HYPERPARAMETER_OPTIMIZATION_LOGIC.md](./HYPERPARAMETER_OPTIMIZATION_LOGIC.md)** - 烟火检测专用超参数优化逻辑

### 🛠️ 代码重构
- **[REFACTORING_SUMMARY.md](./REFACTORING_SUMMARY.md)** - test.py脚本重构总结
- **[TEST_SCRIPT_GUIDE.md](./TEST_SCRIPT_GUIDE.md)** - 测试脚本使用指南

### 🚀 使用指南
- **[QUICK_START.md](./QUICK_START.md)** - 快速开始指南
- **[TRAINING_MANAGEMENT.md](./TRAINING_MANAGEMENT.md)** - 训练管理指南

## 📖 文档分类

### 架构设计类
YOLOv7模型架构、参数和设计原理：
- YOLOv7-RepVGG-RCSOSA-README.md
- YOLOv7-RepVGG-RCSOSA-Architecture.md
- YOLOv7-RepVGG-RCSOSA-Parameters-Mapping.md
- YOLOv7-RepVGG-RCSOSA-Visualization.md
- YOLOv7-Kernel-Size-Visualization.md
- YOLOv7-Kernel-Size-Visualization-CN.md
- YOLOv7-Channels-Explained.md
- YOLOv7-Channels-Explained-CN.md
- YOLOv7-YAML-Format-Explained.md
- YOLOv7-YAML-Parameters-Illustrated.md
- YOLOv7-YAML-Parameters-Illustrated-CN.md

### 问题诊断类
解决项目中遇到的技术问题和疑难杂症：
- ACCURATE_PROBLEM_ANALYSIS.md
- CLASS_INDEX_FIX.md
- ROOT_CAUSE_SOLUTION.md

### 优化配置类
超参数配置和模型优化相关：
- HYPERPARAMETER_COMPARISON.md
- HYPERPARAMETER_OPTIMIZATION_LOGIC.md

### 代码改进类
代码重构和质量提升相关：
- REFACTORING_SUMMARY.md
- TEST_SCRIPT_GUIDE.md

### 操作指南类
项目使用和操作相关：
- QUICK_START.md
- TRAINING_MANAGEMENT.md

## 🔍 快速查找

### 遇到权重加载问题？
👉 查看 [ACCURATE_PROBLEM_ANALYSIS.md](./ACCURATE_PROBLEM_ANALYSIS.md)

### 需要优化训练效果？
👉 查看 [HYPERPARAMETER_OPTIMIZATION_LOGIC.md](./HYPERPARAMETER_OPTIMIZATION_LOGIC.md)

### 想了解代码改进过程？
👉 查看 [REFACTORING_SUMMARY.md](./REFACTORING_SUMMARY.md)

### 第一次使用项目？
👉 查看 [QUICK_START.md](./QUICK_START.md)

### 想了解卷积核大小(kernel_size)参数？
👉 查看 [YOLOv7-Kernel-Size-Visualization.md](./YOLOv7-Kernel-Size-Visualization.md) (英文)  
👉 查看 [YOLOv7-Kernel-Size-Visualization-CN.md](./YOLOv7-Kernel-Size-Visualization-CN.md) (中文)

### 想了解通道数(channels)参数？
👉 查看 [YOLOv7-Channels-Explained.md](./YOLOv7-Channels-Explained.md) (英文)  
👉 查看 [YOLOv7-Channels-Explained-CN.md](./YOLOv7-Channels-Explained-CN.md) (中文)

### 想了解YAML参数配置？
👉 查看 [YOLOv7-YAML-Format-Explained.md](./YOLOv7-YAML-Format-Explained.md)  
👉 查看 [YOLOv7-YAML-Parameters-Illustrated.md](./YOLOv7-YAML-Parameters-Illustrated.md) (英文)  
👉 查看 [YOLOv7-YAML-Parameters-Illustrated-CN.md](./YOLOv7-YAML-Parameters-Illustrated-CN.md) (中文)

---
