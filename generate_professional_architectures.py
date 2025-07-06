#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv4 专业架构图生成器 v2.0
参考论文风格生成高质量的DrawIO架构图
"""

import os
import yaml
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

class ProfessionalYOLODrawer:
    def __init__(self):
        # 专业配色方案 - 参考学术论文风格
        self.color_scheme = {
            'Conv': '#E3F2FD',           # 浅蓝色 - 标准卷积
            'RepVGG': '#E8EAF6',         # 浅靛蓝色 - RepVGG卷积  
            'BottleneckCSPC': '#E8F5E8',  # 浅绿色 - CSP瓶颈
            'RCSOSA': '#FFE0B2',         # 浅橙色 - RCS-OSA模块
            'SimAM': '#FCE4EC',          # 浅粉色 - 注意力机制
            'SPPF': '#FFF9C4',           # 浅黄色 - 空间金字塔
            'Concat': '#F5F5F5',         # 浅灰色 - 拼接操作
            'Upsample': '#E0F2F1',       # 浅青色 - 上采样
            'IDetect': '#FFEBEE',        # 浅红色 - 检测头
            'Input': '#E1F5FE',          # 输入
            'Output': '#F3E5F5'          # 输出
        }
        
        # 边框颜色
        self.stroke_scheme = {
            'Conv': '#1976D2',
            'RepVGG': '#3F51B5', 
            'BottleneckCSPC': '#388E3C',
            'RCSOSA': '#F57C00',
            'SimAM': '#E91E63',
            'SPPF': '#FBC02D',
            'Concat': '#757575',
            'Upsample': '#00695C',
            'IDetect': '#D32F2F',
            'Input': '#0277BD',
            'Output': '#7B1FA2'
        }
        
        self.models_info = {
            'yolov4-csp-IDetect': {
                'name': 'YOLOv4-CSP-IDetect',
                'description': 'Baseline CSP-DarkNet + IDetect Head'
            },
            'yolov4-repvgg': {
                'name': 'YOLOv4-RepVGG',
                'description': 'RepVGG Reparameterization + CSP Backbone'
            },
            'yolov4-rcsosa': {
                'name': 'YOLOv4-RCSOSA', 
                'description': 'RCS-OSA Residual Cross Stage + Standard Conv'
            },
            'yolov4-simAM': {
                'name': 'YOLOv4-SimAM',
                'description': 'CSP Backbone + SimAM Attention Mechanism'
            },
            'yolov4-repvgg-rcsosa': {
                'name': 'YOLOv4-RepVGG-RCSOSA',
                'description': 'RepVGG Reparameterization + RCS-OSA Module'
            },
            'yolov4-repvgg-simAM': {
                'name': 'YOLOv4-RepVGG-SimAM',
                'description': 'RepVGG + CSP + SimAM Attention'
            },
            'yolov4-rcsosa-simAM': {
                'name': 'YOLOv4-RCSOSA-SimAM',
                'description': 'RCS-OSA Module + SimAM Attention'
            },
            'yolov4-repvgg-rcsosa-simAM': {
                'name': 'YOLOv4-RepVGG-RCSOSA-SimAM',
                'description': 'Ultimate: RepVGG + RCS-OSA + SimAM'
            }
        }

    def parse_yaml_config(self, yaml_path):
        """解析YAML配置文件"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def get_layer_info(self, layer):
        """提取层的详细信息"""
        if len(layer) < 3:
            return "Unknown", [], ""
            
        module_type = layer[2]
        args = layer[3] if len(layer) > 3 else []
        
        # 构建详细的显示文本
        if module_type == 'Conv':
            if len(args) >= 3:
                channels = args[0]
                kernel = args[1] 
                stride = args[2]
                text = f"Conv\n{channels}ch\nk={kernel}, s={stride}"
            else:
                text = f"Conv\n{args[0] if args else ''}ch"
                
        elif module_type == 'RepVGG':
            if len(args) >= 3:
                channels = args[0]
                kernel = args[1]
                stride = args[2] 
                text = f"RepVGG\n{channels}ch\nk={kernel}, s={stride}"
            else:
                text = f"RepVGG\n{args[0] if args else ''}ch"
                
        elif module_type == 'BottleneckCSPC':
            channels = args[0] if args else ''
            repeat = layer[1] if len(layer) > 1 else 1
            text = f"BottleneckCSPC\n{channels}ch\nn={repeat}"
            
        elif module_type == 'RCSOSA':
            channels = args[0] if args else ''
            repeat = layer[1] if len(layer) > 1 else 1
            text = f"RCS-OSA\n{channels}ch\nn={repeat}"
            
        elif module_type == 'SimAM':
            text = "SimAM\nAttention"
            
        elif module_type == 'SPPF':
            channels = args[0] if args else ''
            kernel = args[1] if len(args) > 1 else 5
            text = f"SPPF\n{channels}ch\nk={kernel}"
            
        elif module_type == 'nn.Upsample':
            scale = args[1] if len(args) > 1 else 2
            text = f"Upsample\n×{scale}"
            
        elif module_type == 'Concat':
            text = "Concat\nFeature Fusion"
            
        elif module_type == 'IDetect':
            nc = args[0] if args else 2
            text = f"IDetect\nClasses: {nc}"
            
        else:
            text = module_type
            
        return module_type, args, text

    def create_professional_drawio(self, model_name, config):
        """创建专业风格的DrawIO架构图"""
        
        # 创建根元素
        mxfile = ET.Element('mxfile', 
                           host="app.diagrams.net", 
                           modified="2025-07-06T00:00:00.000Z", 
                           agent="5.0", 
                           version="22.1.16")
        
        diagram = ET.SubElement(mxfile, 'diagram', 
                               name=f"{model_name}_architecture", 
                               id=f"{model_name}_id")
        
        mxGraphModel = ET.SubElement(diagram, 'mxGraphModel', 
                                    dx="2000", dy="1200", grid="1", gridSize="10", 
                                    guides="1", tooltips="1", connect="1", 
                                    arrows="1", fold="1", page="1", 
                                    pageScale="1", pageWidth="1169", pageHeight="827", 
                                    math="0", shadow="0")
        
        root = ET.SubElement(mxGraphModel, 'root')
        ET.SubElement(root, 'mxCell', id="0")
        ET.SubElement(root, 'mxCell', id="1", parent="0")
        
        cell_id = 2
        
        # 创建标题区域
        title_style = ("rounded=0;whiteSpace=wrap;html=1;fontSize=18;fontStyle=1;"
                      "fillColor=#f8f9fa;strokeColor=#dee2e6;strokeWidth=2;")
        
        title_cell = ET.SubElement(root, 'mxCell', 
                                  id=str(cell_id),
                                  value=f'<b>{self.models_info[model_name]["name"]}</b><br/>'
                                        f'<font size="3">{self.models_info[model_name]["description"]}</font>',
                                  style=title_style,
                                  vertex="1", parent="1")
        ET.SubElement(title_cell, 'mxGeometry', x="50", y="20", width="800", height="70", **{"as": "geometry"})
        cell_id += 1
        
        # 输入层
        input_style = (f"rounded=1;whiteSpace=wrap;html=1;fontSize=12;fontStyle=1;"
                      f"fillColor={self.color_scheme['Input']};strokeColor={self.stroke_scheme['Input']};"
                      f"strokeWidth=2;")
        
        input_cell = ET.SubElement(root, 'mxCell',
                                  id=str(cell_id),
                                  value="Input Image<br/>640×640×3",
                                  style=input_style,
                                  vertex="1", parent="1")
        ET.SubElement(input_cell, 'mxGeometry', x="100", y="120", width="120", height="60", **{"as": "geometry"})
        prev_cell_id = cell_id
        cell_id += 1
        
        # Backbone区域标签
        backbone_label = ET.SubElement(root, 'mxCell',
                                      id=str(cell_id),
                                      value="<b>Backbone</b>",
                                      style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=16;fontColor=#1976D2;fontStyle=1;",
                                      vertex="1", parent="1")
        ET.SubElement(backbone_label, 'mxGeometry', x="50", y="200", width="80", height="30", **{"as": "geometry"})
        cell_id += 1
        
        # 绘制Backbone层
        backbone_layers = config.get('backbone', [])
        y_pos = 220
        backbone_start_id = cell_id
        
        for i, layer in enumerate(backbone_layers):
            module_type, args, display_text = self.get_layer_info(layer)
            
            # 获取样式
            fill_color = self.color_scheme.get(module_type, '#F5F5F5')
            stroke_color = self.stroke_scheme.get(module_type, '#666666')
            
            layer_style = (f"rounded=1;whiteSpace=wrap;html=1;fontSize=10;fontStyle=1;"
                          f"fillColor={fill_color};strokeColor={stroke_color};"
                          f"strokeWidth=2;align=center;verticalAlign=middle;")
            
            # 创建层元素
            layer_cell = ET.SubElement(root, 'mxCell',
                                      id=str(cell_id),
                                      value=display_text,
                                      style=layer_style,
                                      vertex="1", parent="1")
            ET.SubElement(layer_cell, 'mxGeometry', x="150", y=str(y_pos), width="100", height="60", **{"as": "geometry"})
            
            # 添加连接线
            if i == 0:
                # 从输入连接到第一层
                edge_cell = ET.SubElement(root, 'mxCell',
                                        id=str(cell_id + 1000),
                                        value="",
                                        style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#1976D2;",
                                        edge="1", parent="1", source=str(prev_cell_id), target=str(cell_id))
                ET.SubElement(edge_cell, 'mxGeometry', relative="1", **{"as": "geometry"})
            else:
                # 层与层之间的连接
                edge_cell = ET.SubElement(root, 'mxCell',
                                        id=str(cell_id + 1000),
                                        value="",
                                        style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#1976D2;",
                                        edge="1", parent="1", source=str(cell_id-1), target=str(cell_id))
                ET.SubElement(edge_cell, 'mxGeometry', relative="1", **{"as": "geometry"})
            
            cell_id += 1
            y_pos += 80
        
        backbone_end_id = cell_id - 1
        
        # Head区域标签
        head_y_start = y_pos + 20
        head_label = ET.SubElement(root, 'mxCell',
                                  id=str(cell_id),
                                  value="<b>Head</b>",
                                  style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=16;fontColor=#F57C00;fontStyle=1;",
                                  vertex="1", parent="1")
        ET.SubElement(head_label, 'mxGeometry', x="50", y=str(head_y_start), width="80", height="30", **{"as": "geometry"})
        cell_id += 1
        
        # 绘制Head层
        head_layers = config.get('head', [])
        y_pos = head_y_start + 40
        head_start_id = cell_id
        
        # 处理跳跃连接的位置记录
        layer_positions = {}
        
        for i, layer in enumerate(head_layers):
            module_type, args, display_text = self.get_layer_info(layer)
            
            # 处理特殊的连接情况
            from_layer = layer[0] if len(layer) > 0 else [-1]
            if isinstance(from_layer, list) and len(from_layer) > 1:
                # 多输入层(如Concat)，需要特殊处理位置
                x_pos = 350 if 'Concat' in display_text else 150
            else:
                x_pos = 150
            
            # 获取样式
            fill_color = self.color_scheme.get(module_type, '#F5F5F5')
            stroke_color = self.stroke_scheme.get(module_type, '#666666')
            
            layer_style = (f"rounded=1;whiteSpace=wrap;html=1;fontSize=10;fontStyle=1;"
                          f"fillColor={fill_color};strokeColor={stroke_color};"
                          f"strokeWidth=2;align=center;verticalAlign=middle;")
            
            # 创建层元素
            layer_cell = ET.SubElement(root, 'mxCell',
                                      id=str(cell_id),
                                      value=display_text,
                                      style=layer_style,
                                      vertex="1", parent="1")
            ET.SubElement(layer_cell, 'mxGeometry', x=str(x_pos), y=str(y_pos), width="100", height="60", **{"as": "geometry"})
            
            # 记录层位置
            layer_positions[cell_id] = (x_pos, y_pos)
            
            # 添加连接线
            if i == 0:
                # Head第一层连接到Backbone最后一层
                edge_cell = ET.SubElement(root, 'mxCell',
                                        id=str(cell_id + 2000),
                                        value="",
                                        style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#F57C00;",
                                        edge="1", parent="1", source=str(backbone_end_id), target=str(cell_id))
                ET.SubElement(edge_cell, 'mxGeometry', relative="1", **{"as": "geometry"})
            elif not (isinstance(from_layer, list) and len(from_layer) > 1):
                # 普通的顺序连接
                edge_cell = ET.SubElement(root, 'mxCell',
                                        id=str(cell_id + 2000),
                                        value="",
                                        style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#F57C00;",
                                        edge="1", parent="1", source=str(cell_id-1), target=str(cell_id))
                ET.SubElement(edge_cell, 'mxGeometry', relative="1", **{"as": "geometry"})
            
            cell_id += 1
            y_pos += 80
        
        # 输出层
        output_style = (f"rounded=1;whiteSpace=wrap;html=1;fontSize=12;fontStyle=1;"
                       f"fillColor={self.color_scheme['Output']};strokeColor={self.stroke_scheme['Output']};"
                       f"strokeWidth=2;")
        
        output_cell = ET.SubElement(root, 'mxCell',
                                   id=str(cell_id),
                                   value="Detection Output<br/>Bounding Boxes<br/>Class Predictions",
                                   style=output_style,
                                   vertex="1", parent="1")
        ET.SubElement(output_cell, 'mxGeometry', x="650", y=str(y_pos), width="150", height="80", **{"as": "geometry"})
        
        # 最后的连接线
        edge_cell = ET.SubElement(root, 'mxCell',
                                id=str(cell_id + 3000),
                                value="",
                                style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#D32F2F;",
                                edge="1", parent="1", source=str(cell_id-1), target=str(cell_id))
        ET.SubElement(edge_cell, 'mxGeometry', relative="1", **{"as": "geometry"})
        
        # 添加尺度标注
        scale_labels = [
            ("P4/16", 500, y_pos - 200),
            ("P5/32", 500, y_pos - 100)
        ]
        
        for label, x, y in scale_labels:
            scale_cell = ET.SubElement(root, 'mxCell',
                                      id=str(cell_id + 100),
                                      value=label,
                                      style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=10;fontColor=#666666;",
                                      vertex="1", parent="1")
            ET.SubElement(scale_cell, 'mxGeometry', x=str(x), y=str(y), width="60", height="20", **{"as": "geometry"})
            cell_id += 1
        
        return mxfile

    def save_drawio_file(self, xml_tree, output_path):
        """保存DrawIO文件"""
        rough_string = ET.tostring(xml_tree, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # 移除空行并优化格式
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        formatted_xml = '\n'.join(lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_xml)

    def generate_all_professional_architectures(self):
        """生成所有专业风格的架构图"""
        cfg_dir = Path('cfg/training')
        output_dir = Path('docs/architecture_diagrams_v2')
        output_dir.mkdir(exist_ok=True)
        
        generated_files = []
        
        for model_name in self.models_info.keys():
            yaml_file = cfg_dir / f"{model_name}.yaml"
            
            if yaml_file.exists():
                print(f"🎨 正在生成专业版架构图: {model_name}")
                
                # 解析配置
                config = self.parse_yaml_config(yaml_file)
                
                # 生成专业DrawIO XML
                xml_tree = self.create_professional_drawio(model_name, config)
                
                # 保存文件
                output_file = output_dir / f"{model_name}_professional.drawio"
                self.save_drawio_file(xml_tree, output_file)
                
                generated_files.append(output_file)
                print(f"✅ 生成完成: {output_file}")
            else:
                print(f"❌ 配置文件不存在: {yaml_file}")
        
        return generated_files

if __name__ == "__main__":
    drawer = ProfessionalYOLODrawer()
    generated_files = drawer.generate_all_professional_architectures()
    
    print(f"\n🎉 专业版架构图生成完成!")
    print(f"总共生成了 {len(generated_files)} 个专业架构图文件:")
    for file in generated_files:
        print(f"  📊 {file}")
    
    print(f"\n📁 所有专业架构图已保存到: docs/architecture_diagrams_v2/")
    print("🎨 新版本特点:")
    print("  - 参考学术论文风格设计")
    print("  - 清晰的模块分组和标签")
    print("  - 专业的配色方案")
    print("  - 详细的参数标注")
    print("  - 更好的视觉层次")
