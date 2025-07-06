#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv4 架构图生成器
为8个不同的YOLOv4变体生成DrawIO架构图
"""

import os
import yaml
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

class YOLOArchitectureDrawer:
    def __init__(self):
        self.color_scheme = {
            'Conv': '#E1F5FE',           # 浅蓝色 - 基础卷积
            'RepVGG': '#F3E5F5',         # 浅紫色 - RepVGG卷积
            'BottleneckCSPC': '#E8F5E8',  # 浅绿色 - CSP瓶颈
            'RCSOSA': '#FFF3E0',         # 浅橙色 - RCS-OSA模块
            'SimAM': '#FFEBEE',          # 浅红色 - 注意力机制
            'SPPF': '#F0F4C3',          # 浅黄色 - 空间金字塔
            'Concat': '#F5F5F5',        # 浅灰色 - 拼接操作
            'Upsample': '#E0F2F1',      # 浅青色 - 上采样
            'IDetect': '#FCE4EC'        # 浅粉色 - 检测头
        }
        
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

    def parse_yaml_config(self, yaml_path):
        """解析YAML配置文件"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def create_drawio_xml(self, model_name, config):
        """创建DrawIO XML格式的架构图"""
        
        # 创建根元素
        mxfile = ET.Element('mxfile', host="app.diagrams.net", modified="2025-07-06T00:00:00.000Z", agent="5.0", version="22.1.16")
        diagram = ET.SubElement(mxfile, 'diagram', name=model_name, id="unique_id")
        mxGraphModel = ET.SubElement(diagram, 'mxGraphModel', 
                                    dx="1422", dy="754", grid="1", gridSize="10", 
                                    guides="1", tooltips="1", connect="1", 
                                    arrows="1", fold="1", page="1", 
                                    pageScale="1", pageWidth="827", pageHeight="1169", 
                                    math="0", shadow="0")
        
        root = ET.SubElement(mxGraphModel, 'root')
        ET.SubElement(root, 'mxCell', id="0")
        ET.SubElement(root, 'mxCell', id="1", parent="0")
        
        # 生成架构图元素
        y_pos = 60
        cell_id = 2
        
        # 标题
        title_cell = ET.SubElement(root, 'mxCell', 
                                  id=str(cell_id), 
                                  value=f'<b>{self.models_info[model_name]["name"]}</b><br/>{self.models_info[model_name]["description"]}',
                                  style="rounded=1;whiteSpace=wrap;html=1;fontSize=16;fontStyle=1;fillColor=#d5e8d4;strokeColor=#82b366;",
                                  vertex="1", parent="1")
        ET.SubElement(title_cell, 'mxGeometry', x="50", y="20", width="700", height="60", **{"as": "geometry"})
        cell_id += 1
        y_pos += 100
        
        # Input
        input_cell = ET.SubElement(root, 'mxCell', 
                                  id=str(cell_id), 
                                  value="Input Image<br/>640×640×3",
                                  style="rounded=1;whiteSpace=wrap;html=1;fontSize=12;fillColor=#fff2cc;strokeColor=#d6b656;",
                                  vertex="1", parent="1")
        ET.SubElement(input_cell, 'mxGeometry', x="350", y=str(y_pos), width="100", height="40", **{"as": "geometry"})
        cell_id += 1
        y_pos += 60
        
        # Backbone
        backbone_layers = config.get('backbone', [])
        for i, layer in enumerate(backbone_layers):
            if len(layer) >= 3:
                module_type = layer[2]
                args = layer[3] if len(layer) > 3 else []
                
                # 构建显示文本
                if module_type == 'Conv':
                    if len(args) >= 2:
                        display_text = f"Conv {args[0]}ch<br/>k={args[1]}, s={args[2] if len(args) > 2 else 1}"
                    else:
                        display_text = f"Conv {args[0] if args else ''}ch"
                elif module_type == 'RepVGG':
                    if len(args) >= 2:
                        display_text = f"RepVGG {args[0]}ch<br/>k={args[1]}, s={args[2] if len(args) > 2 else 1}"
                    else:
                        display_text = f"RepVGG {args[0] if args else ''}ch"
                elif module_type == 'BottleneckCSPC':
                    display_text = f"BottleneckCSPC<br/>{args[0] if args else ''}ch"
                elif module_type == 'RCSOSA':
                    display_text = f"RCS-OSA<br/>{args[0] if args else ''}ch"
                elif module_type == 'SimAM':
                    display_text = "SimAM<br/>Attention"
                elif module_type == 'SPPF':
                    display_text = f"SPPF<br/>{args[0] if args else ''}ch, k={args[1] if len(args) > 1 else 5}"
                else:
                    display_text = module_type
                
                # 获取颜色
                color = self.color_scheme.get(module_type, '#F5F5F5')
                
                # 创建层元素
                layer_cell = ET.SubElement(root, 'mxCell', 
                                          id=str(cell_id), 
                                          value=display_text,
                                          style=f"rounded=1;whiteSpace=wrap;html=1;fontSize=10;fillColor={color};strokeColor=#666666;",
                                          vertex="1", parent="1")
                ET.SubElement(layer_cell, 'mxGeometry', x="350", y=str(y_pos), width="100", height="40", **{"as": "geometry"})
                
                # 添加连接线（如果不是第一个元素）
                if cell_id > 3:
                    edge_cell = ET.SubElement(root, 'mxCell', 
                                            id=str(cell_id + 1000), 
                                            value="",
                                            style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;",
                                            edge="1", parent="1", source=str(cell_id-1), target=str(cell_id))
                    ET.SubElement(edge_cell, 'mxGeometry', relative="1", **{"as": "geometry"})
                
                cell_id += 1
                y_pos += 50
        
        # Head部分
        head_layers = config.get('head', [])
        for i, layer in enumerate(head_layers):
            if len(layer) >= 3:
                module_type = layer[2]
                args = layer[3] if len(layer) > 3 else []
                
                # 构建显示文本
                if module_type == 'Conv':
                    if len(args) >= 2:
                        display_text = f"Conv {args[0]}ch<br/>k={args[1]}, s={args[2] if len(args) > 2 else 1}"
                    else:
                        display_text = f"Conv {args[0] if args else ''}ch"
                elif module_type == 'RepVGG':
                    if len(args) >= 2:
                        display_text = f"RepVGG {args[0]}ch<br/>k={args[1]}, s={args[2] if len(args) > 2 else 1}"
                    else:
                        display_text = f"RepVGG {args[0] if args else ''}ch"
                elif module_type == 'BottleneckCSPC':
                    display_text = f"BottleneckCSPC<br/>{args[0] if args else ''}ch"
                elif module_type == 'RCSOSA':
                    display_text = f"RCS-OSA<br/>{args[0] if args else ''}ch"
                elif module_type == 'SimAM':
                    display_text = "SimAM<br/>Attention"
                elif module_type == 'nn.Upsample':
                    display_text = f"Upsample<br/>×{args[1] if len(args) > 1 else 2}"
                elif module_type == 'Concat':
                    display_text = "Concat<br/>Feature Fusion"
                elif module_type == 'IDetect':
                    nc = args[0] if args else config.get('nc', 2)
                    display_text = f"IDetect<br/>Classes: {nc}"
                else:
                    display_text = module_type
                
                # 获取颜色
                color = self.color_scheme.get(module_type, '#F5F5F5')
                if module_type == 'nn.Upsample':
                    color = self.color_scheme.get('Upsample', '#E0F2F1')
                
                # 创建层元素
                layer_cell = ET.SubElement(root, 'mxCell', 
                                          id=str(cell_id), 
                                          value=display_text,
                                          style=f"rounded=1;whiteSpace=wrap;html=1;fontSize=10;fillColor={color};strokeColor=#666666;",
                                          vertex="1", parent="1")
                ET.SubElement(layer_cell, 'mxGeometry', x="350", y=str(y_pos), width="100", height="40", **{"as": "geometry"})
                
                # 添加连接线
                if i > 0 or len(backbone_layers) > 0:
                    edge_cell = ET.SubElement(root, 'mxCell', 
                                            id=str(cell_id + 1000), 
                                            value="",
                                            style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;",
                                            edge="1", parent="1", source=str(cell_id-1), target=str(cell_id))
                    ET.SubElement(edge_cell, 'mxGeometry', relative="1", **{"as": "geometry"})
                
                cell_id += 1
                y_pos += 50
        
        # 输出
        output_cell = ET.SubElement(root, 'mxCell', 
                                   id=str(cell_id), 
                                   value=f"Output<br/>Detection Results",
                                   style="rounded=1;whiteSpace=wrap;html=1;fontSize=12;fillColor=#f8cecc;strokeColor=#b85450;",
                                   vertex="1", parent="1")
        ET.SubElement(output_cell, 'mxGeometry', x="350", y=str(y_pos), width="100", height="40", **{"as": "geometry"})
        
        # 最后一条连接线
        edge_cell = ET.SubElement(root, 'mxCell', 
                                id=str(cell_id + 1000), 
                                value="",
                                style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;",
                                edge="1", parent="1", source=str(cell_id-1), target=str(cell_id))
        ET.SubElement(edge_cell, 'mxGeometry', relative="1", **{"as": "geometry"})
        
        return mxfile

    def save_drawio_file(self, xml_tree, output_path):
        """保存DrawIO文件"""
        # 格式化XML
        rough_string = ET.tostring(xml_tree, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # 移除空行
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        formatted_xml = '\n'.join(lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_xml)

    def generate_all_architectures(self):
        """生成所有8个模型的架构图"""
        cfg_dir = Path('cfg/training')
        output_dir = Path('docs/architecture_diagrams')
        output_dir.mkdir(exist_ok=True)
        
        generated_files = []
        
        for model_name in self.models_info.keys():
            yaml_file = cfg_dir / f"{model_name}.yaml"
            
            if yaml_file.exists():
                print(f"正在生成 {model_name} 的架构图...")
                
                # 解析配置
                config = self.parse_yaml_config(yaml_file)
                
                # 生成DrawIO XML
                xml_tree = self.create_drawio_xml(model_name, config)
                
                # 保存文件
                output_file = output_dir / f"{model_name}_architecture.drawio"
                self.save_drawio_file(xml_tree, output_file)
                
                generated_files.append(output_file)
                print(f"✅ 生成完成: {output_file}")
            else:
                print(f"❌ 配置文件不存在: {yaml_file}")
        
        return generated_files

if __name__ == "__main__":
    drawer = YOLOArchitectureDrawer()
    generated_files = drawer.generate_all_architectures()
    
    print(f"\n🎉 总共生成了 {len(generated_files)} 个架构图文件:")
    for file in generated_files:
        print(f"  - {file}")
    
    print(f"\n📁 所有架构图已保存到: docs/architecture_diagrams/")
    print("💡 使用方法: 在draw.io网站上打开这些.drawio文件即可查看和编辑架构图")
