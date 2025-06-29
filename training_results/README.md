# 训练结果整理

整理时间: 2025-06-28 14:22:06

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


### yolov7

- **20250628_1053_epochs50_bs4**: 997.8MB, 50 epochs
- **20250628_0103_epochs2_bs4**: 997.7MB, 2 epochs

## 整理统计

- 保留模型数量: 1
- 归档结果数量: 7
- 删除结果数量: 0

## 使用说明

1. `current/` 目录包含每个模型的最新训练结果
2. `archived/` 目录包含历史训练结果和不完整的训练
3. 权重文件位于各自目录的 `weights/` 子目录中
4. 可以通过 `organization_summary.json` 查看详细的整理信息
