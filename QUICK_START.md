# 快速开始指南

## 1. 单独训练某个版本

### 基础版本
```bash
python train.py --data datasets_smokefire/smokefire.yaml --cfg cfg/training/yolov7.yaml --epochs 50 --batch-size 4 --device cpu --workers 1
```

### RepVGG版本
```bash
python train-repvgg.py --data datasets_smokefire/smokefire.yaml --cfg cfg/training/yolov7-repvgg.yaml --epochs 50 --batch-size 4 --device cpu --workers 1
```

### RCSOSA版本
```bash
python train-rcsosa.py --data datasets_smokefire/smokefire.yaml --cfg cfg/training/yolov7-rcsosa.yaml --epochs 50 --batch-size 4 --device cpu --workers 1
```

### RepVGG+RCSOSA版本
```bash
python train-repvgg-rcsosa.py --data datasets_smokefire/smokefire.yaml --cfg cfg/training/yolov7-repvgg-rcsosa.yaml --epochs 50 --batch-size 4 --device cpu --workers 1
```

## 2. 自动对比所有版本
```bash
python compare_versions.py --epochs 50 --batch-size 4
```

## 3. 测试模型
```bash
# 测试某个训练好的模型
python test.py --data datasets_smokefire/smokefire.yaml --weights runs/train/yolov7/weights/best.pt --device cpu
```

## 4. 检测图像
```bash
# 使用训练好的模型检测图像
python detect.py --weights runs/train/yolov7/weights/best.pt --source inference/images/ --device cpu
```

## 注意事项
- 所有命令都已配置为CPU模式，适合Apple Silicon Mac
- 建议batch-size不要超过8，避免内存不足
- workers设为1避免多进程问题
- epochs可以根据需要调整，测试时可以用较小的值如10-20
