# YOLOv4系列模型配置文件对应关系

## 完整的8个模型配置对应表

| 序号 | CFG配置文件 | 超参数文件 | 模型特点 |
|------|-------------|------------|----------|
| 1 | `yolov4-csp-IDetect.yaml` | `hyp.yolov4-csp-idetect-smokefire.yaml` | 基础CSP架构，稳定性能基准 |
| 2 | `yolov4-repvgg.yaml` | `hyp.yolov4-repvgg-smokefire.yaml` | RepVGG结构，训练推理效率优化 |
| 3 | `yolov4-rcsosa.yaml` | `hyp.yolov4-rcsosa-smokefire.yaml` | RCSOSA模块，增强特征提取能力 |
| 4 | `yolov4-repvgg-rcsosa.yaml` | `hyp.yolov4-repvgg-rcsosa-smokefire.yaml` | RepVGG+RCSOSA，效率与特征并重 |
| 5 | `yolov4-simAM.yaml` | `hyp.yolov4-simAM-smokefire.yaml` | SimAM注意力，无参数空间增强 |
| 6 | `yolov4-repvgg-simAM.yaml` | `hyp.yolov4-repvgg-simAM-smokefire.yaml` | RepVGG+SimAM，高效注意力组合 |
| 7 | `yolov4-rcsosa-simAM.yaml` | `hyp.yolov4-rcsosa-simAM-smokefire.yaml` | RCSOSA+SimAM，特征与注意力融合 |
| 8 | `yolov4-repvgg-rcsosa-simAM.yaml` | `hyp.yolov4-repvgg-rcsosa-simAM-smokefire.yaml` | 终极三重组合，最强性能配置 |

## 训练命令快速参考

```bash
# 1. 基础CSP模型
python train.py --cfg cfg/training/yolov4-csp-IDetect.yaml --hyp hyperparameters/hyp.yolov4-csp-idetect-smokefire.yaml --data datasets/smokefire.yaml

# 2. RepVGG模型  
python train.py --cfg cfg/training/yolov4-repvgg.yaml --hyp hyperparameters/hyp.yolov4-repvgg-smokefire.yaml --data datasets/smokefire.yaml

# 3. RCSOSA模型
python train.py --cfg cfg/training/yolov4-rcsosa.yaml --hyp hyperparameters/hyp.yolov4-rcsosa-smokefire.yaml --data datasets/smokefire.yaml

# 4. RepVGG+RCSOSA模型
python train.py --cfg cfg/training/yolov4-repvgg-rcsosa.yaml --hyp hyperparameters/hyp.yolov4-repvgg-rcsosa-smokefire.yaml --data datasets/smokefire.yaml

# 5. SimAM模型
python train.py --cfg cfg/training/yolov4-simAM.yaml --hyp hyperparameters/hyp.yolov4-simAM-smokefire.yaml --data datasets/smokefire.yaml

# 6. RepVGG+SimAM模型
python train.py --cfg cfg/training/yolov4-repvgg-simAM.yaml --hyp hyperparameters/hyp.yolov4-repvgg-simAM-smokefire.yaml --data datasets/smokefire.yaml

# 7. RCSOSA+SimAM模型
python train.py --cfg cfg/training/yolov4-rcsosa-simAM.yaml --hyp hyperparameters/hyp.yolov4-rcsosa-simAM-smokefire.yaml --data datasets/smokefire.yaml

# 8. 终极组合模型
python train.py --cfg cfg/training/yolov4-repvgg-rcsosa-simAM.yaml --hyp hyperparameters/hyp.yolov4-repvgg-rcsosa-simAM-smokefire.yaml --data datasets/smokefire.yaml
```

## 使用场景推荐

### 🚀 速度优先
- **推荐**: RepVGG系列 (#2, #6)
- **最佳选择**: `yolov4-repvgg.yaml`

### 🎯 精度优先  
- **推荐**: 终极组合 (#8)
- **最佳选择**: `yolov4-repvgg-rcsosa-simAM.yaml`

### ⚖️ 平衡性能
- **推荐**: RepVGG+RCSOSA (#4) 
- **最佳选择**: `yolov4-repvgg-rcsosa.yaml`

### 💡 注意力增强
- **推荐**: RepVGG+SimAM (#6)
- **最佳选择**: `yolov4-repvgg-simAM.yaml`

## 文件清理状态

✅ **已删除的多余文件**:
- `hyp.yolov4-csp-rcsosa-idetect-smokefire.yaml`
- `hyp.yolov4-csp-repvgg-idetect-smokefire.yaml` 
- `hyp.yolov4-csp-repvgg-rcsosa-idetect-smokefire.yaml`
- `hyp.yolov4-simAM-rcsosa-smokefire.yaml`
- `hyp.yolov4-simAM-repvgg-rcsosa-smokefire.yaml`
- `hyp.yolov4-simAM-repvgg-smokefire.yaml`

✅ **保留的有效文件**: 8个配置文件对，完全对应

## 验证方法

可以通过以下命令验证文件对应关系：
```bash
ls cfg/training/yolov4* | wc -l    # 应该输出: 8
ls hyperparameters/hyp.yolov4*smokefire* | wc -l    # 应该输出: 8
```
