# YOLOv4 架构图使用指南

## 📋 架构图概览

本项目为8个YOLOv4变体生成了详细的DrawIO架构图，每个图都准确反映了对应YAML配置文件的模型结构。

## 🎨 架构图列表

### 基础模型
1. **yolov4-csp-IDetect_architecture.drawio**
   - 基础CSP-DarkNet backbone
   - IDetect检测头
   - 适合作为baseline模型

### RepVGG系列
2. **yolov4-repvgg_architecture.drawio**
   - RepVGG卷积替代标准卷积
   - 提升推理速度的重参数化结构

3. **yolov4-repvgg-simAM_architecture.drawio**
   - RepVGG + SimAM注意力机制
   - 平衡速度与精度

### RCS-OSA系列
4. **yolov4-rcsosa_architecture.drawio**
   - RCS-OSA模块替代CSP结构
   - 优化特征融合能力

5. **yolov4-repvgg-rcsosa_architecture.drawio**
   - RepVGG + RCS-OSA组合
   - 快速推理 + 高效特征融合

### SimAM注意力系列
6. **yolov4-simAM_architecture.drawio**
   - CSP backbone + SimAM注意力
   - 轻量级注意力机制

7. **yolov4-rcsosa-simAM_architecture.drawio**
   - RCS-OSA + SimAM组合
   - 特征融合 + 注意力机制

### 全能版
8. **yolov4-repvgg-rcsosa-simAM_architecture.drawio**
   - RepVGG + RCS-OSA + SimAM全集成
   - 最复杂的架构变体

## 🎨 颜色编码说明

| 模块类型 | 颜色 | 描述 |
|---------|------|------|
| Conv | 浅蓝色 (#E1F5FE) | 标准卷积层 |
| RepVGG | 浅紫色 (#F3E5F5) | 重参数化卷积 |
| BottleneckCSPC | 浅绿色 (#E8F5E8) | CSP瓶颈模块 |
| RCSOSA | 浅橙色 (#FFF3E0) | RCS-OSA模块 |
| SimAM | 浅红色 (#FFEBEE) | 注意力机制 |
| SPPF | 浅黄色 (#F0F4C3) | 空间金字塔池化 |
| Concat | 浅灰色 (#F5F5F5) | 特征拼接 |
| Upsample | 浅青色 (#E0F2F1) | 上采样 |
| IDetect | 浅粉色 (#FCE4EC) | 检测头 |

## 📖 如何使用

### 1. 在线查看
1. 访问 [draw.io](https://app.diagrams.net/)
2. 点击"打开现有图表"
3. 选择并上传对应的.drawio文件
4. 即可查看详细的模型架构

### 2. 本地编辑
1. 下载并安装 [draw.io Desktop](https://github.com/jgraph/drawio-desktop/releases)
2. 直接双击.drawio文件打开
3. 可进行编辑和修改

### 3. 导出其他格式
在draw.io中可以将架构图导出为：
- PNG/JPG图片
- PDF文档
- SVG矢量图
- HTML文件

## 🔍 架构特点分析

### Backbone对比
| 模型 | Backbone特征 | 优势 | 应用场景 |
|-----|-------------|------|---------|
| CSP-IDetect | 标准CSP结构 | 稳定性好，精度高 | 基准测试 |
| RepVGG | 重参数化卷积 | 推理速度快 | 实时检测 |
| RCSOSA | 残差跨阶段 | 特征融合强 | 复杂场景 |

### 注意力机制
- **SimAM**: 轻量级注意力，计算开销小
- **位置**: 通常在backbone末端和head关键位置
- **效果**: 提升特征表达能力

### Head结构
- **IDetect**: 改进的检测头
- **多尺度**: P4/16和P5/32两个检测尺度
- **锚点**: 针对烟火检测优化的锚点设置

## 📊 模型复杂度对比

| 模型 | 参数复杂度 | 计算复杂度 | 推理速度 | 检测精度 |
|-----|-----------|-----------|---------|---------|
| CSP-IDetect | 中等 | 中等 | 中等 | 基准 |
| RepVGG | 低 | 低 | 高 | 略低 |
| RCSOSA | 中等 | 中等 | 中等 | 高 |
| SimAM | 中等+ | 中等+ | 中等- | 高 |
| RepVGG-RCSOSA | 中等 | 中等 | 中等+ | 高 |
| RepVGG-SimAM | 中等+ | 中等+ | 中等 | 高+ |
| RCSOSA-SimAM | 高 | 高 | 中等- | 高+ |
| 全能版 | 最高 | 最高 | 低 | 最高 |

## 🛠️ 自定义架构图

如果需要修改架构图：

1. **添加新模块**：
   - 在`generate_architecture_diagrams.py`中的`color_scheme`添加新颜色
   - 在绘制逻辑中添加新模块的处理

2. **调整布局**：
   - 修改`x_pos`和`y_pos`参数
   - 调整元素大小和间距

3. **更新配置**：
   - 修改对应的YAML配置文件
   - 重新运行生成脚本

## 📁 文件结构

```
docs/architecture_diagrams/
├── yolov4-csp-IDetect_architecture.drawio
├── yolov4-repvgg_architecture.drawio
├── yolov4-rcsosa_architecture.drawio
├── yolov4-simAM_architecture.drawio
├── yolov4-repvgg-rcsosa_architecture.drawio
├── yolov4-repvgg-simAM_architecture.drawio
├── yolov4-rcsosa-simAM_architecture.drawio
└── yolov4-repvgg-rcsosa-simAM_architecture.drawio
```

## 💡 使用建议

1. **模型选择**：根据应用场景选择合适的架构
2. **性能调优**：参考架构图理解模型瓶颈
3. **论文写作**：架构图可直接用于学术论文
4. **团队协作**：统一的可视化便于团队理解

---

*生成时间: 2025年7月6日*  
*工具: 自动化架构图生成器*  
*版本: v1.0*
