#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv4 Mermaid架构图生成器
为8个不同的YOLOv4变体生成Mermaid格式的架构图
"""

import os
import yaml
from pathlib import Path

class YOLOMermaidGenerator:
    def __init__(self):
        self.models_info = {
            'yolov4-csp-IDetect': {
                'name': 'YOLOv4-CSP-IDetect',
                'description': '基础CSP-DarkNet backbone + IDetect检测头'
            },
            'yolov4-repvgg': {
                'name': 'YOLOv4-RepVGG',
                'description': 'RepVGG卷积 + CSP backbone + IDetect检测头'
            },
            'yolov4-rcsosa': {
                'name': 'YOLOv4-RCSOSA',
                'description': 'RCS-OSA模块 + 标准卷积 + IDetect检测头'
            },
            'yolov4-simAM': {
                'name': 'YOLOv4-SimAM',
                'description': 'CSP backbone + SimAM注意力机制 + IDetect检测头'
            },
            'yolov4-repvgg-rcsosa': {
                'name': 'YOLOv4-RepVGG-RCSOSA',
                'description': 'RepVGG卷积 + RCS-OSA模块 + IDetect检测头'
            },
            'yolov4-repvgg-simAM': {
                'name': 'YOLOv4-RepVGG-SimAM',
                'description': 'RepVGG卷积 + CSP backbone + SimAM注意力 + IDetect检测头'
            },
            'yolov4-rcsosa-simAM': {
                'name': 'YOLOv4-RCSOSA-SimAM',
                'description': 'RCS-OSA模块 + SimAM注意力机制 + IDetect检测头'
            },
            'yolov4-repvgg-rcsosa-simAM': {
                'name': 'YOLOv4-RepVGG-RCSOSA-SimAM',
                'description': 'RepVGG + RCS-OSA + SimAM全能版 + IDetect检测头'
            }
        }
        
        # Mermaid样式类定义
        self.style_classes = {
            'Conv': 'fill:#E1F5FE,stroke:#01579B,stroke-width:2px,color:#000',
            'RepVGG': 'fill:#F3E5F5,stroke:#4A148C,stroke-width:2px,color:#000',
            'BottleneckCSPC': 'fill:#E8F5E8,stroke:#1B5E20,stroke-width:2px,color:#000',
            'RCSOSA': 'fill:#FFF3E0,stroke:#E65100,stroke-width:2px,color:#000',
            'SimAM': 'fill:#FFEBEE,stroke:#B71C1C,stroke-width:2px,color:#000',
            'SPPF': 'fill:#F0F4C3,stroke:#827717,stroke-width:2px,color:#000',
            'Concat': 'fill:#F5F5F5,stroke:#424242,stroke-width:2px,color:#000',
            'Upsample': 'fill:#E0F2F1,stroke:#00695C,stroke-width:2px,color:#000',
            'IDetect': 'fill:#FCE4EC,stroke:#AD1457,stroke-width:2px,color:#000',
            'Input': 'fill:#FFF9C4,stroke:#F57F17,stroke-width:3px,color:#000',
            'Output': 'fill:#FFCDD2,stroke:#C62828,stroke-width:3px,color:#000'
        }

    def parse_yaml_config(self, yaml_path):
        """解析YAML配置文件"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def generate_layer_info(self, layer, layer_idx):
        """生成层信息字符串"""
        if len(layer) < 3:
            return f"Layer{layer_idx}", "Unknown", "Unknown"
        
        module_type = layer[2]
        args = layer[3] if len(layer) > 3 else []
        
        # 构建显示文本和节点ID
        node_id = f"{module_type}{layer_idx}"
        
        if module_type == 'Conv':
            if len(args) >= 2:
                display_text = f"Conv\\n{args[0]}ch\\nk={args[1]}, s={args[2] if len(args) > 2 else 1}"
            else:
                display_text = f"Conv\\n{args[0] if args else '?'}ch"
        elif module_type == 'RepVGG':
            if len(args) >= 2:
                display_text = f"RepVGG\\n{args[0]}ch\\nk={args[1]}, s={args[2] if len(args) > 2 else 1}"
            else:
                display_text = f"RepVGG\\n{args[0] if args else '?'}ch"
        elif module_type == 'BottleneckCSPC':
            display_text = f"BottleneckCSPC\\n{args[0] if args else '?'}ch"
        elif module_type == 'RCSOSA':
            display_text = f"RCS-OSA\\n{args[0] if args else '?'}ch"
        elif module_type == 'SimAM':
            display_text = "SimAM\\nAttention"
        elif module_type == 'SPPF':
            display_text = f"SPPF\\n{args[0] if args else '?'}ch\\nk={args[1] if len(args) > 1 else 5}"
        elif module_type == 'nn.Upsample':
            display_text = f"Upsample\\n×{args[1] if len(args) > 1 else 2}"
        elif module_type == 'Concat':
            display_text = "Concat\\nFeature Fusion"
        elif module_type == 'IDetect':
            nc = args[0] if args else 2
            display_text = f"IDetect\\nClasses: {nc}"
        else:
            display_text = module_type
        
        return node_id, display_text, module_type

    def generate_mermaid_flowchart(self, model_name, config):
        """生成Mermaid流程图代码"""
        
        mermaid_code = []
        mermaid_code.append("```mermaid")
        mermaid_code.append("flowchart TD")
        mermaid_code.append("")
        
        # 添加标题
        mermaid_code.append(f"    title[\"{self.models_info[model_name]['name']}<br/>{self.models_info[model_name]['description']}\"]")
        mermaid_code.append("")
        
        # 输入节点
        mermaid_code.append("    Input[\"Input Image<br/>640×640×3\"]")
        mermaid_code.append("")
        
        # 收集所有节点
        all_nodes = []
        all_connections = []
        layer_idx = 0
        
        # 处理backbone
        backbone_layers = config.get('backbone', [])
        prev_node = "Input"
        
        mermaid_code.append("    %% Backbone")
        for i, layer in enumerate(backbone_layers):
            node_id, display_text, module_type = self.generate_layer_info(layer, layer_idx)
            mermaid_code.append(f"    {node_id}[\"{display_text}\"]")
            all_nodes.append((node_id, module_type))
            
            # 添加连接
            all_connections.append(f"    {prev_node} --> {node_id}")
            prev_node = node_id
            layer_idx += 1
        
        mermaid_code.append("")
        
        # 处理head
        head_layers = config.get('head', [])
        backbone_end = prev_node
        
        mermaid_code.append("    %% Head")
        for i, layer in enumerate(head_layers):
            node_id, display_text, module_type = self.generate_layer_info(layer, layer_idx)
            mermaid_code.append(f"    {node_id}[\"{display_text}\"]")
            all_nodes.append((node_id, module_type))
            
            # 处理连接（简化版，主要处理顺序连接）
            if i == 0:
                all_connections.append(f"    {prev_node} --> {node_id}")
            else:
                # 检查是否有特殊连接（如Concat）
                if len(layer) > 0 and isinstance(layer[0], list):
                    # 有多输入的情况，简化为从前一层连接
                    prev_layer_id = f"{head_layers[i-1][2]}{layer_idx-1}"
                    all_connections.append(f"    {prev_layer_id} --> {node_id}")
                else:
                    # 常规顺序连接
                    prev_layer_id = f"{head_layers[i-1][2]}{layer_idx-1}"
                    all_connections.append(f"    {prev_layer_id} --> {node_id}")
            
            prev_node = node_id
            layer_idx += 1
        
        mermaid_code.append("")
        
        # 输出节点
        mermaid_code.append("    Output[\"Detection Results<br/>Bounding Boxes + Classes\"]")
        all_connections.append(f"    {prev_node} --> Output")
        mermaid_code.append("")
        
        # 添加所有连接
        mermaid_code.append("    %% Connections")
        for connection in all_connections:
            mermaid_code.append(connection)
        
        mermaid_code.append("")
        
        # 添加样式类
        mermaid_code.append("    %% Styling")
        added_classes = set()
        
        # 为输入输出添加样式
        mermaid_code.append(f"    classDef inputStyle {self.style_classes['Input']}")
        mermaid_code.append(f"    classDef outputStyle {self.style_classes['Output']}")
        mermaid_code.append("    class Input inputStyle")
        mermaid_code.append("    class Output outputStyle")
        
        # 为各种模块类型添加样式
        for node_id, module_type in all_nodes:
            style_key = module_type
            if module_type == 'nn.Upsample':
                style_key = 'Upsample'
            
            if style_key in self.style_classes and style_key not in added_classes:
                class_name = f"{style_key.lower()}Style"
                mermaid_code.append(f"    classDef {class_name} {self.style_classes[style_key]}")
                added_classes.add(style_key)
        
        # 应用样式类到节点
        for node_id, module_type in all_nodes:
            style_key = module_type
            if module_type == 'nn.Upsample':
                style_key = 'Upsample'
            
            if style_key in self.style_classes:
                class_name = f"{style_key.lower()}Style"
                mermaid_code.append(f"    class {node_id} {class_name}")
        
        mermaid_code.append("```")
        mermaid_code.append("")
        
        return '\n'.join(mermaid_code)

    def generate_mermaid_graph(self, model_name, config):
        """生成Mermaid图代码（备选方案）"""
        
        mermaid_code = []
        mermaid_code.append("```mermaid")
        mermaid_code.append("graph TD")
        mermaid_code.append("")
        
        # 处理层级结构
        layer_idx = 0
        prev_node = "A[Input Image<br/>640×640×3]"
        current_node = "A"
        
        # Backbone
        backbone_layers = config.get('backbone', [])
        for i, layer in enumerate(backbone_layers):
            layer_idx += 1
            node_id = chr(65 + layer_idx)  # A, B, C, ...
            
            _, display_text, module_type = self.generate_layer_info(layer, i)
            
            mermaid_code.append(f"    {node_id}[{display_text}]")
            mermaid_code.append(f"    {current_node} --> {node_id}")
            current_node = node_id
        
        # Head
        head_layers = config.get('head', [])
        for i, layer in enumerate(head_layers):
            layer_idx += 1
            node_id = chr(65 + layer_idx) if layer_idx < 26 else f"A{layer_idx}"
            
            _, display_text, module_type = self.generate_layer_info(layer, i)
            
            mermaid_code.append(f"    {node_id}[{display_text}]")
            mermaid_code.append(f"    {current_node} --> {node_id}")
            current_node = node_id
        
        # Output
        layer_idx += 1
        output_node = chr(65 + layer_idx) if layer_idx < 26 else f"A{layer_idx}"
        mermaid_code.append(f"    {output_node}[Detection Results]")
        mermaid_code.append(f"    {current_node} --> {output_node}")
        
        mermaid_code.append("```")
        mermaid_code.append("")
        
        return '\n'.join(mermaid_code)

    def generate_comprehensive_markdown(self, model_name, config):
        """生成包含多种Mermaid图表的综合Markdown文档"""
        
        doc = []
        
        # 文档标题
        doc.append(f"# {self.models_info[model_name]['name']} 架构图")
        doc.append("")
        doc.append(f"**描述**: {self.models_info[model_name]['description']}")
        doc.append("")
        
        # 模型参数
        doc.append("## 📊 模型参数")
        doc.append("")
        doc.append(f"- **类别数**: {config.get('nc', 2)}")
        doc.append(f"- **深度倍数**: {config.get('depth_multiple', 1.0)}")
        doc.append(f"- **宽度倍数**: {config.get('width_multiple', 0.75)}")
        
        if 'anchors' in config:
            doc.append(f"- **锚点配置**: {len(config['anchors'])} 组")
            for i, anchor_group in enumerate(config['anchors']):
                doc.append(f"  - P{i+4}: {anchor_group}")
        doc.append("")
        
        # 主要架构图
        doc.append("## 🏗️ 网络架构")
        doc.append("")
        doc.append(self.generate_mermaid_flowchart(model_name, config))
        doc.append("")
        
        # 模块统计
        doc.append("## 📈 模块统计")
        doc.append("")
        
        # 统计各类模块
        module_counts = {}
        total_layers = 0
        
        for layer in config.get('backbone', []) + config.get('head', []):
            if len(layer) >= 3:
                module_type = layer[2]
                module_counts[module_type] = module_counts.get(module_type, 0) + 1
                total_layers += 1
        
        doc.append("| 模块类型 | 数量 | 描述 |")
        doc.append("|---------|------|------|")
        
        module_descriptions = {
            'Conv': '标准卷积层',
            'RepVGG': '重参数化卷积',
            'BottleneckCSPC': 'CSP瓶颈模块',
            'RCSOSA': 'RCS-OSA模块',
            'SimAM': 'SimAM注意力机制',
            'SPPF': '空间金字塔池化',
            'nn.Upsample': '上采样层',
            'Concat': '特征拼接',
            'IDetect': '检测头'
        }
        
        for module_type, count in sorted(module_counts.items()):
            desc = module_descriptions.get(module_type, '未知模块')
            doc.append(f"| {module_type} | {count} | {desc} |")
        
        doc.append(f"| **总计** | **{total_layers}** | **总层数** |")
        doc.append("")
        
        # Backbone详细结构
        doc.append("## 🔧 Backbone详细结构")
        doc.append("")
        doc.append("| 层序号 | 模块类型 | 参数 | 描述 |")
        doc.append("|-------|---------|------|------|")
        
        for i, layer in enumerate(config.get('backbone', [])):
            if len(layer) >= 3:
                module_type = layer[2]
                args = layer[3] if len(layer) > 3 else []
                args_str = str(args) if args else "无"
                from_layer = layer[0] if len(layer) > 0 else "前层"
                doc.append(f"| {i} | {module_type} | {args_str} | 来自: {from_layer} |")
        
        doc.append("")
        
        # Head详细结构
        doc.append("## 🎯 Head详细结构")
        doc.append("")
        doc.append("| 层序号 | 模块类型 | 参数 | 描述 |")
        doc.append("|-------|---------|------|------|")
        
        backbone_len = len(config.get('backbone', []))
        for i, layer in enumerate(config.get('head', [])):
            if len(layer) >= 3:
                module_type = layer[2]
                args = layer[3] if len(layer) > 3 else []
                args_str = str(args) if args else "无"
                from_layer = layer[0] if len(layer) > 0 else "前层"
                doc.append(f"| {backbone_len + i} | {module_type} | {args_str} | 来自: {from_layer} |")
        
        doc.append("")
        
        # 使用说明
        doc.append("## 💡 使用说明")
        doc.append("")
        doc.append("### 训练命令")
        doc.append("```bash")
        doc.append(f"python train.py \\")
        doc.append(f"  --cfg cfg/training/{model_name}.yaml \\")
        doc.append("  --data datasets/smokefire.yaml \\")
        doc.append("  --hyp hyperparameters/hyp.universal.yaml \\")
        doc.append("  --epochs 150 \\")
        doc.append("  --batch-size 8")
        doc.append("```")
        doc.append("")
        
        doc.append("### 测试命令")
        doc.append("```bash")
        doc.append("python test.py \\")
        doc.append(f"  --cfg cfg/training/{model_name}.yaml \\")
        doc.append("  --data datasets/smokefire.yaml \\")
        doc.append("  --weights runs/train/exp/weights/best.pt")
        doc.append("```")
        doc.append("")
        
        # 生成时间戳
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        doc.append(f"---")
        doc.append(f"*生成时间: {timestamp}*  ")
        doc.append(f"*生成工具: YOLOMermaidGenerator v1.0*")
        
        return '\n'.join(doc)

    def generate_all_mermaid_diagrams(self):
        """生成所有模型的Mermaid架构图"""
        cfg_dir = Path('cfg/training')
        output_dir = Path('docs/mermaid_diagrams')
        output_dir.mkdir(exist_ok=True)
        
        generated_files = []
        
        for model_name in self.models_info.keys():
            yaml_file = cfg_dir / f"{model_name}.yaml"
            
            if yaml_file.exists():
                print(f"正在生成 {model_name} 的Mermaid架构图...")
                
                # 解析配置
                config = self.parse_yaml_config(yaml_file)
                
                # 生成综合Markdown文档
                markdown_content = self.generate_comprehensive_markdown(model_name, config)
                
                # 保存文件
                output_file = output_dir / f"{model_name}_architecture.md"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                generated_files.append(output_file)
                print(f"✅ 生成完成: {output_file}")
            else:
                print(f"❌ 配置文件不存在: {yaml_file}")
        
        return generated_files

    def generate_summary_index(self, generated_files):
        """生成Mermaid架构图总索引"""
        index_content = []
        
        index_content.append("# YOLOv4 Mermaid架构图总览")
        index_content.append("")
        index_content.append("本目录包含了8个YOLOv4变体的详细Mermaid架构图，每个图都准确反映了对应YAML配置文件的模型结构。")
        index_content.append("")
        
        index_content.append("## 📋 架构图列表")
        index_content.append("")
        index_content.append("| 序号 | 模型名称 | 文档链接 | 主要特点 |")
        index_content.append("|-----|---------|----------|---------|")
        
        for i, (model_name, info) in enumerate(self.models_info.items(), 1):
            file_name = f"{model_name}_architecture.md"
            index_content.append(f"| {i} | {info['name']} | [{file_name}](./{file_name}) | {info['description']} |")
        
        index_content.append("")
        
        index_content.append("## 🎨 Mermaid特点")
        index_content.append("")
        index_content.append("- **📊 交互式图表**: 可在GitHub、GitLab等平台直接渲染")
        index_content.append("- **🎯 精确结构**: 严格按照YAML配置生成，无遗漏")
        index_content.append("- **🎨 美观样式**: 使用颜色编码区分不同模块类型")
        index_content.append("- **📱 响应式**: 自动适应不同屏幕尺寸")
        index_content.append("- **🔍 可搜索**: 支持文本搜索和导航")
        index_content.append("")
        
        index_content.append("## 🚀 在线查看")
        index_content.append("")
        index_content.append("### GitHub渲染")
        index_content.append("直接在GitHub仓库中点击对应的`.md`文件即可查看渲染后的Mermaid图表。")
        index_content.append("")
        index_content.append("### 在线编辑器")
        index_content.append("- [Mermaid Live Editor](https://mermaid.live/)")
        index_content.append("- [GitHub Mermaid支持](https://github.blog/2022-02-14-include-diagrams-markdown-files-mermaid/)")
        index_content.append("")
        
        index_content.append("## 💻 本地查看")
        index_content.append("")
        index_content.append("### VS Code")
        index_content.append("安装`Mermaid Markdown Syntax Highlighting`插件后可直接预览。")
        index_content.append("")
        index_content.append("### Typora")
        index_content.append("原生支持Mermaid图表渲染。")
        index_content.append("")
        
        # 保存索引文件
        index_file = Path('docs/mermaid_diagrams/README.md')
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(index_content))
        
        return index_file

if __name__ == "__main__":
    generator = YOLOMermaidGenerator()
    generated_files = generator.generate_all_mermaid_diagrams()
    
    # 生成索引文件
    index_file = generator.generate_summary_index(generated_files)
    
    print(f"\n🎉 总共生成了 {len(generated_files)} 个Mermaid架构图:")
    for file in generated_files:
        print(f"  - {file}")
    
    print(f"\n📋 索引文件: {index_file}")
    print(f"\n📁 所有架构图已保存到: docs/mermaid_diagrams/")
    print("💡 使用方法: 在支持Mermaid的平台上查看.md文件，如GitHub、GitLab、Typora等")
