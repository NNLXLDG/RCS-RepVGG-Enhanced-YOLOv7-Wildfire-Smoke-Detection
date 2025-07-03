# YOLOv7 高性能GPU训练超参数配置完整指南

## 📋 训练脚本与超参数文件完整对应表

| 训练脚本 | 超参数文件 | 模型特点 | GPU优化策略 | 推荐用途 |
|---------|-----------|----------|------------|----------|
| `train.py` | `hyp.train.yaml` | 标准YOLOv7架构 | 平衡精度和速度 | 通用检测任务 |
| `train-tiny.py` | `hyp.train-tiny.yaml` | 轻量级架构 | 最大化训练精度 | 实时检测/高效推理 |
| `train-repvgg.py` | `hyp.train-repvgg.yaml` | RepVGG重参数化 | 重参数化性能优化 | 高精度检测 |
| `train-rcsosa.py` | `hyp.train-rcsosa.yaml` | RCSOSA注意力机制 | 注意力机制精度最大化 | 复杂场景检测 |
| `train-repvgg-rcsosa.py` | `hyp.train-repvgg-rcsosa.yaml` | RepVGG+RCSOSA混合 | 极致精度优化 | 最高精度需求 |
| `train-tiny-repvgg.py` | `hyp.train-tiny-repvgg.yaml` | Tiny+RepVGG轻量化 | 轻量化高精度 | 平衡性能场景 |
| `train-tiny-rcsosa.py` | `hyp.train-tiny-rcsosa.yaml` | Tiny+RCSOSA轻量化 | 轻量化注意力优化 | 快速复杂场景 |
| `train-tiny-repvgg-rcsosa.py` | `hyp.train-tiny-repvgg-rcsosa.yaml` | Tiny+RepVGG+RCSOSA全能 | 全功能性能优化 | 综合性能最优 |

## 🚀 使用方法

### 直接使用默认超参数（推荐）
```bash
# 每个训练脚本都会自动使用对应的超参数文件
python train.py                    # 自动使用 hyp.train.yaml
python train-tiny.py               # 自动使用 hyp.train-tiny.yaml
python train-repvgg.py             # 自动使用 hyp.train-repvgg.yaml
python train-rcsosa.py             # 自动使用 hyp.train-rcsosa.yaml
python train-repvgg-rcsosa.py      # 自动使用 hyp.train-repvgg-rcsosa.yaml
python train-tiny-repvgg.py        # 自动使用 hyp.train-tiny-repvgg.yaml
python train-tiny-rcsosa.py        # 自动使用 hyp.train-tiny-rcsosa.yaml
python train-tiny-repvgg-rcsosa.py # 自动使用 hyp.train-tiny-repvgg-rcsosa.yaml
```

### 手动指定超参数文件（实验对比）
```bash
# 也可以手动指定其他超参数文件进行对比实验
python train.py --hyp hyperparameters/hyp.train-tiny.yaml
python train-tiny.py --hyp hyperparameters/hyp.train.yaml
```

## 📊 模型复杂度与性能对比

| 模型架构 | 参数量 | 计算复杂度 | GPU训练速度 | GPU推理速度 | 精度等级 | 推荐batch-size(GPU) |
|---------|-------|-----------|------------|------------|---------|-------------------|
| YOLOv7-Tiny | 最小 | 最低 | 最快 | 最快 | 良好 | 32-64 |
| YOLOv7 | 中等 | 中等 | 快 | 快 | 优秀 | 16-32 |
| YOLOv7-RepVGG | 中高 | 中高 | 中等 | 快 | 优秀+ | 16-32 |
| YOLOv7-RCSOSA | 中高 | 中高 | 中等 | 中等 | 优秀+ | 16-32 |
| YOLOv7-RepVGG-RCSOSA | 最高 | 最高 | 较慢 | 中等 | 极致 | 8-16 |
| Tiny+RepVGG | 小 | 低 | 快 | 快 | 良好+ | 32-64 |
| Tiny+RCSOSA | 小 | 低 | 快 | 中等 | 良好+ | 32-64 |
| Tiny+RepVGG+RCSOSA | 中小 | 中低 | 中等 | 中等 | 优秀 | 16-32 |

## 🔧 超参数调优完整策略

### 1. 学习率调优策略
- **所有模型**: lr0=0.01 (统一GPU训练学习率)
- **最终学习率**: lrf=0.01 (深度收敛获得最高精度)
- **预热策略**: 复杂架构使用更长预热时间
- **学习率调度**: 推荐使用cosine annealing

### 2. 数据增强调优策略
- **强化增强**: GPU训练可使用更强的数据增强
- **Mosaic增强**: 所有模型全开(1.0)提升泛化能力
- **Scale增强**: 使用0.5大范围缩放提升鲁棒性
- **Mixup策略**: 注意力模型适度使用，其他关闭以提升稳定性

### 3. 损失函数权重调优
- **box权重**: 统一0.05 (边界框定位精度)
- **cls权重**: 0.5-0.8 (分类精度，注意力和混合模型更高)
- **obj权重**: 统一1.0 (最大化目标检测置信度)

### 4. GPU训练专项优化
- **batch_size**: 推荐16-64 (充分利用GPU并行能力)
- **workers**: 推荐4-8 (GPU多线程数据加载)
- **loss_ota**: 全部开启 (GPU训练可承受OTA计算开销)
- **copy_paste**: 适度使用提升数据多样性

### 5. 火灾/烟雾检测特定优化
- **HSV增强**: 适应火焰色彩变化
- **翻转策略**: 火灾场景不适合上下翻转
- **马赛克增强**: 最大化复杂场景适应性
- **IoU阈值**: 火灾检测标准阈值0.2

## 🎯 模型选择建议

### 🥇 精度优先场景
1. **train-repvgg-rcsosa.py** - 最高精度，适合离线分析
2. **train-rcsosa.py** - 注意力机制优势，复杂场景
3. **train-repvgg.py** - RepVGG重参数化优势，高精度

### ⚡ 速度优先场景  
1. **train-tiny.py** - 最快训练和推理，实时应用
2. **train-tiny-repvgg.py** - 平衡速度和重参数化优势
3. **train-tiny-rcsosa.py** - 轻量化注意力，快速复杂场景

### 🔄 平衡选择场景
1. **train.py** - 标准平衡选择，通用性强
2. **train-tiny-repvgg-rcsosa.py** - 轻量化全功能，最佳平衡

## 📊 超参数实验建议

### 实验流程
1. **基准实验**: 使用默认超参数进行训练
2. **学习率调优**: lr0 ± 20% 范围内测试
3. **数据增强调优**: 根据数据集特点调整增强强度
4. **损失权重调优**: 根据检测任务调整各损失权重
5. **架构对比**: 在相同超参数下对比不同架构

### 性能评估指标
- **mAP@0.5**: 主要精度指标
- **mAP@0.5:0.95**: 严格精度指标  
- **推理速度**: FPS (帧每秒)
- **模型大小**: 参数量和文件大小
- **训练时间**: 每个epoch的训练时间

### 超参数调优实验表格
| 参数 | 基准值 | 调优范围 | 调优步长 | 优化目标 |
|------|-------|----------|----------|----------|
| lr0 | 0.01 | 0.005-0.02 | 0.002 | 收敛速度+精度 |
| weight_decay | 0.0005 | 0.0001-0.001 | 0.0002 | 防止过拟合 |
| box权重 | 0.05 | 0.03-0.08 | 0.01 | 定位精度 |
| cls权重 | 0.3 | 0.2-0.5 | 0.05 | 分类精度 |
| obj权重 | 0.7 | 0.5-0.8 | 0.05 | 检测置信度 |

## 📈 训练监控与早停策略

### 推荐训练配置
1. **Tiny模型**: 50-100 epochs，快速收敛
2. **标准模型**: 100-200 epochs，稳定训练
3. **复杂模型**: 200-300 epochs，精细调优
4. **早停策略**: 监控validation mAP，连续20 epochs无提升时停止
5. **学习率调度**: 使用cosine annealing获得更好的收敛性

### TensorBoard监控指标
- 训练/验证损失曲线
- mAP@0.5 和 mAP@0.5:0.95 曲线
- 学习率变化曲线
- 各类别精度和召回率

## 📝 配置文件详细说明

每个超参数文件都针对特定模型架构和火灾/烟雾检测任务进行了深度优化：

### 核心配置文件
- **`hyp.train.yaml`**: 标准YOLOv7，平衡性能和准确度
- **`hyp.train-tiny.yaml`**: 轻量级模型，适合实时检测
- **`hyp.train-repvgg.yaml`**: RepVGG架构，高准确度训练推理分离
- **`hyp.train-rcsosa.yaml`**: RCSOSA架构，注意力机制增强
- **`hyp.train-repvgg-rcsosa.yaml`**: 组合架构，极致性能

### 轻量化变体配置
- **`hyp.train-tiny-repvgg.yaml`**: Tiny+RepVGG，轻量化重参数化
- **`hyp.train-tiny-rcsosa.yaml`**: Tiny+RCSOSA，轻量化注意力
- **`hyp.train-tiny-repvgg-rcsosa.yaml`**: 全功能轻量化，最佳平衡

---
*最后更新: 2025年7月3日*  
*支持8个训练脚本的完整超参数优化体系*
