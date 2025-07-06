# YOLOv4-RepVGG 架构图

**描述**: RepVGG卷积 + CSP backbone + IDetect检测头

## 📊 模型参数

- **类别数**: 2
- **深度倍数**: 1.0
- **宽度倍数**: 0.75
- **锚点配置**: 2 组
  - P4: [37, 49, 89, 110]
  - P5: [177, 372, 249, 110]

## 🏗️ 网络架构

```mermaid
flowchart TD

    title["YOLOv4-RepVGG<br/>RepVGG卷积 + CSP backbone + IDetect检测头"]

    Input["Input Image<br/>640×640×3"]

    %% Backbone
    RepVGG0["RepVGG\n64ch\nk=3, s=2"]
    RepVGG1["RepVGG\n128ch\nk=3, s=2"]
    BottleneckCSPC2["BottleneckCSPC\n128ch"]
    RepVGG3["RepVGG\n256ch\nk=3, s=2"]
    BottleneckCSPC4["BottleneckCSPC\n256ch"]
    RepVGG5["RepVGG\n512ch\nk=3, s=2"]
    BottleneckCSPC6["BottleneckCSPC\n512ch"]
    RepVGG7["RepVGG\n1024ch\nk=3, s=2"]
    BottleneckCSPC8["BottleneckCSPC\n1024ch"]
    SPPF9["SPPF\n1024ch\nk=5"]

    %% Head
    Conv10["Conv\n512ch\nk=1, s=1"]
    nn.Upsample11["Upsample\n×2"]
    BottleneckCSPC12["BottleneckCSPC\n512ch"]
    Concat13["Concat\nFeature Fusion"]
    BottleneckCSPC14["BottleneckCSPC\n512ch"]
    RepVGG15["RepVGG\n512ch\nk=3, s=2"]
    Concat16["Concat\nFeature Fusion"]
    BottleneckCSPC17["BottleneckCSPC\n768ch"]
    RepVGG18["RepVGG\n512ch\nk=3, s=1"]
    RepVGG19["RepVGG\n768ch\nk=3, s=1"]
    IDetect20["IDetect\nClasses: nc"]

    Output["Detection Results<br/>Bounding Boxes + Classes"]

    %% Connections
    Input --> RepVGG0
    RepVGG0 --> RepVGG1
    RepVGG1 --> BottleneckCSPC2
    BottleneckCSPC2 --> RepVGG3
    RepVGG3 --> BottleneckCSPC4
    BottleneckCSPC4 --> RepVGG5
    RepVGG5 --> BottleneckCSPC6
    BottleneckCSPC6 --> RepVGG7
    RepVGG7 --> BottleneckCSPC8
    BottleneckCSPC8 --> SPPF9
    SPPF9 --> Conv10
    Conv10 --> nn.Upsample11
    nn.Upsample11 --> BottleneckCSPC12
    BottleneckCSPC12 --> Concat13
    Concat13 --> BottleneckCSPC14
    BottleneckCSPC14 --> RepVGG15
    RepVGG15 --> Concat16
    Concat16 --> BottleneckCSPC17
    BottleneckCSPC17 --> RepVGG18
    RepVGG18 --> RepVGG19
    RepVGG19 --> IDetect20
    IDetect20 --> Output

    %% Styling
    classDef inputStyle fill:#FFF9C4,stroke:#F57F17,stroke-width:3px,color:#000
    classDef outputStyle fill:#FFCDD2,stroke:#C62828,stroke-width:3px,color:#000
    class Input inputStyle
    class Output outputStyle
    classDef repvggStyle fill:#F3E5F5,stroke:#4A148C,stroke-width:2px,color:#000
    classDef bottleneckcspcStyle fill:#E8F5E8,stroke:#1B5E20,stroke-width:2px,color:#000
    classDef sppfStyle fill:#F0F4C3,stroke:#827717,stroke-width:2px,color:#000
    classDef convStyle fill:#E1F5FE,stroke:#01579B,stroke-width:2px,color:#000
    classDef upsampleStyle fill:#E0F2F1,stroke:#00695C,stroke-width:2px,color:#000
    classDef concatStyle fill:#F5F5F5,stroke:#424242,stroke-width:2px,color:#000
    classDef idetectStyle fill:#FCE4EC,stroke:#AD1457,stroke-width:2px,color:#000
    class RepVGG0 repvggStyle
    class RepVGG1 repvggStyle
    class BottleneckCSPC2 bottleneckcspcStyle
    class RepVGG3 repvggStyle
    class BottleneckCSPC4 bottleneckcspcStyle
    class RepVGG5 repvggStyle
    class BottleneckCSPC6 bottleneckcspcStyle
    class RepVGG7 repvggStyle
    class BottleneckCSPC8 bottleneckcspcStyle
    class SPPF9 sppfStyle
    class Conv10 convStyle
    class nn.Upsample11 upsampleStyle
    class BottleneckCSPC12 bottleneckcspcStyle
    class Concat13 concatStyle
    class BottleneckCSPC14 bottleneckcspcStyle
    class RepVGG15 repvggStyle
    class Concat16 concatStyle
    class BottleneckCSPC17 bottleneckcspcStyle
    class RepVGG18 repvggStyle
    class RepVGG19 repvggStyle
    class IDetect20 idetectStyle
```


## 📈 模块统计

| 模块类型 | 数量 | 描述 |
|---------|------|------|
| BottleneckCSPC | 7 | CSP瓶颈模块 |
| Concat | 2 | 特征拼接 |
| Conv | 1 | 标准卷积层 |
| IDetect | 1 | 检测头 |
| RepVGG | 8 | 重参数化卷积 |
| SPPF | 1 | 空间金字塔池化 |
| nn.Upsample | 1 | 上采样层 |
| **总计** | **21** | **总层数** |

## 🔧 Backbone详细结构

| 层序号 | 模块类型 | 参数 | 描述 |
|-------|---------|------|------|
| 0 | RepVGG | [64, 3, 2] | 来自: -1 |
| 1 | RepVGG | [128, 3, 2] | 来自: -1 |
| 2 | BottleneckCSPC | [128] | 来自: -1 |
| 3 | RepVGG | [256, 3, 2] | 来自: -1 |
| 4 | BottleneckCSPC | [256] | 来自: -1 |
| 5 | RepVGG | [512, 3, 2] | 来自: -1 |
| 6 | BottleneckCSPC | [512, True] | 来自: -1 |
| 7 | RepVGG | [1024, 3, 2] | 来自: -1 |
| 8 | BottleneckCSPC | [1024, True] | 来自: -1 |
| 9 | SPPF | [1024, 5] | 来自: -1 |

## 🎯 Head详细结构

| 层序号 | 模块类型 | 参数 | 描述 |
|-------|---------|------|------|
| 10 | Conv | [512, 1, 1] | 来自: -1 |
| 11 | nn.Upsample | ['None', 2, 'nearest'] | 来自: -1 |
| 12 | BottleneckCSPC | [512] | 来自: -1 |
| 13 | Concat | [1] | 来自: [-1, 6] |
| 14 | BottleneckCSPC | [512] | 来自: -1 |
| 15 | RepVGG | [512, 3, 2] | 来自: -1 |
| 16 | Concat | [1] | 来自: [-1, 10] |
| 17 | BottleneckCSPC | [768] | 来自: -1 |
| 18 | RepVGG | [512, 3, 1] | 来自: 14 |
| 19 | RepVGG | [768, 3, 1] | 来自: 17 |
| 20 | IDetect | ['nc', 'anchors'] | 来自: [18, 19] |

## 💡 使用说明

### 训练命令
```bash
python train.py \
  --cfg cfg/training/yolov4-repvgg.yaml \
  --data datasets/smokefire.yaml \
  --hyp hyperparameters/hyp.universal.yaml \
  --epochs 150 \
  --batch-size 8
```

### 测试命令
```bash
python test.py \
  --cfg cfg/training/yolov4-repvgg.yaml \
  --data datasets/smokefire.yaml \
  --weights runs/train/exp/weights/best.pt
```

---
*生成时间: 2025年07月06日 22:25:30*  
*生成工具: YOLOMermaidGenerator v1.0*