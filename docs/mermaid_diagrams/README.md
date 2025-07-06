# YOLOv4 Mermaid架构图总览

本目录包含了8个YOLOv4变体的详细Mermaid架构图，每个图都准确反映了对应YAML配置文件的模型结构。

## 📋 架构图列表

| 序号 | 模型名称 | 文档链接 | 主要特点 |
|-----|---------|----------|---------|
| 1 | YOLOv4-CSP-IDetect | [yolov4-csp-IDetect_architecture.md](./yolov4-csp-IDetect_architecture.md) | 基础CSP-DarkNet backbone + IDetect检测头 |
| 2 | YOLOv4-RepVGG | [yolov4-repvgg_architecture.md](./yolov4-repvgg_architecture.md) | RepVGG卷积 + CSP backbone + IDetect检测头 |
| 3 | YOLOv4-RCSOSA | [yolov4-rcsosa_architecture.md](./yolov4-rcsosa_architecture.md) | RCS-OSA模块 + 标准卷积 + IDetect检测头 |
| 4 | YOLOv4-SimAM | [yolov4-simAM_architecture.md](./yolov4-simAM_architecture.md) | CSP backbone + SimAM注意力机制 + IDetect检测头 |
| 5 | YOLOv4-RepVGG-RCSOSA | [yolov4-repvgg-rcsosa_architecture.md](./yolov4-repvgg-rcsosa_architecture.md) | RepVGG卷积 + RCS-OSA模块 + IDetect检测头 |
| 6 | YOLOv4-RepVGG-SimAM | [yolov4-repvgg-simAM_architecture.md](./yolov4-repvgg-simAM_architecture.md) | RepVGG卷积 + CSP backbone + SimAM注意力 + IDetect检测头 |
| 7 | YOLOv4-RCSOSA-SimAM | [yolov4-rcsosa-simAM_architecture.md](./yolov4-rcsosa-simAM_architecture.md) | RCS-OSA模块 + SimAM注意力机制 + IDetect检测头 |
| 8 | YOLOv4-RepVGG-RCSOSA-SimAM | [yolov4-repvgg-rcsosa-simAM_architecture.md](./yolov4-repvgg-rcsosa-simAM_architecture.md) | RepVGG + RCS-OSA + SimAM全能版 + IDetect检测头 |

## 🎨 Mermaid特点

- **📊 交互式图表**: 可在GitHub、GitLab等平台直接渲染
- **🎯 精确结构**: 严格按照YAML配置生成，无遗漏
- **🎨 美观样式**: 使用颜色编码区分不同模块类型
- **📱 响应式**: 自动适应不同屏幕尺寸
- **🔍 可搜索**: 支持文本搜索和导航

## 🚀 在线查看

### GitHub渲染
直接在GitHub仓库中点击对应的`.md`文件即可查看渲染后的Mermaid图表。

### 在线编辑器
- [Mermaid Live Editor](https://mermaid.live/)
- [GitHub Mermaid支持](https://github.blog/2022-02-14-include-diagrams-markdown-files-mermaid/)

## 💻 本地查看

### VS Code
安装`Mermaid Markdown Syntax Highlighting`插件后可直接预览。

### Typora
原生支持Mermaid图表渲染。
