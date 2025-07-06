#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv4 论文级架构图生成器 v3.0
完全按照提供的参考图像风格设计
"""

import os
import yaml
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

class PaperStyleYOLODrawer:
    def __init__(self):
        # 完全按照参考图的配色方案
        self.colors = {
            'RepVGG': '#9FC5E8',        # 浅蓝色
            'RCSOSA': '#F9CB9C',        # 浅橙色  
            'Conv': '#D9D9D9',          # 浅灰色
            'SPPF': '#D9D9D9',          # 浅灰色
            'Upsample': '#D9D9D9',      # 浅灰色
            'Concat': '#FFFFFF',        # 白色
            'IDetect': '#D9D9D9',       # 浅灰色
            'SimAM': '#FFB6C1',         # 浅粉色
            'BottleneckCSPC': '#B6D7A8'  # 浅绿色
        }
        
        # 边框颜色统一为黑色
        self.stroke_color = '#000000'
        
    def parse_yaml_config(self, yaml_path):
        """解析YAML配置文件"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def get_layer_display_info(self, layer):
        """获取层的显示信息"""
        if len(layer) < 3:
            return "Unknown", "", self.colors.get('Conv', '#D9D9D9')
            
        module_type = layer[2]
        args = layer[3] if len(layer) > 3 else []
        repeat = layer[1] if len(layer) > 1 else 1
        
        # 按照参考图的标注风格
        if module_type == 'RepVGG':
            stride = args[2] if len(args) > 2 else 1
            display_text = f"RepVGG\nstride={stride}"
            
        elif module_type == 'RCSOSA':
            display_text = f"RCS-OSA\nn={repeat}"
            
        elif module_type == 'Conv':
            if len(args) >= 3:
                kernel = args[1]
                stride = args[2]
                display_text = f"Conv\n{kernel}×{kernel}, stride={stride}"
            else:
                display_text = "Conv"
                
        elif module_type == 'BottleneckCSPC':
            display_text = f"BottleneckCSPC\nn={repeat}"
            
        elif module_type == 'SPPF':
            kernel = args[1] if len(args) > 1 else 5
            display_text = f"SPPF\nk=({kernel},{kernel},{kernel})"
            
        elif module_type == 'nn.Upsample':
            display_text = "Upsample"
            
        elif module_type == 'Concat':
            display_text = "C\nConcat"
            
        elif module_type == 'IDetect':
            display_text = "IDetect"
            
        elif module_type == 'SimAM':
            display_text = "SimAM"
            
        else:
            display_text = module_type
            
        color = self.colors.get(module_type, '#D9D9D9')
        return module_type, display_text, color

    def create_paper_style_drawio(self, model_name, config):
        """创建论文风格的DrawIO架构图"""
        
        # 创建XML结构
        mxfile = ET.Element('mxfile', 
                           host="app.diagrams.net", 
                           modified="2025-07-06T00:00:00.000Z", 
                           agent="5.0", 
                           version="22.1.16")
        
        diagram = ET.SubElement(mxfile, 'diagram', 
                               name=f"{model_name}_paper_style", 
                               id=f"{model_name}_paper")
        
        mxGraphModel = ET.SubElement(diagram, 'mxGraphModel', 
                                    dx="2500", dy="1500", grid="1", gridSize="10", 
                                    guides="1", tooltips="1", connect="1", 
                                    arrows="1", fold="1", page="1", 
                                    pageScale="1", pageWidth="1654", pageHeight="1169", 
                                    math="0", shadow="0")
        
        root = ET.SubElement(mxGraphModel, 'root')
        ET.SubElement(root, 'mxCell', id="0")
        ET.SubElement(root, 'mxCell', id="1", parent="0")
        
        cell_id = 2
        
        # 输入图像
        input_cell = ET.SubElement(root, 'mxCell',
                                  id=str(cell_id),
                                  value="",
                                  style="shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;verticalAlign=top;image=data:image/png,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mPk+89QDwADhgGAWjR9awAAAABJRU5ErkJggg==;",
                                  vertex="1", parent="1")
        ET.SubElement(input_cell, 'mxGeometry', x="50", y="600", width="80", height="60", **{"as": "geometry"})
        
        # 输入标签
        input_label = ET.SubElement(root, 'mxCell',
                                   id=str(cell_id + 1),
                                   value="640×640×3",
                                   style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=10;",
                                   vertex="1", parent="1")
        ET.SubElement(input_label, 'mxGeometry', x="50", y="670", width="80", height="20", **{"as": "geometry"})
        cell_id += 2
        
        # Backbone区域框架
        backbone_frame = ET.SubElement(root, 'mxCell',
                                      id=str(cell_id),
                                      value="",
                                      style="rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#4472C4;strokeWidth=2;dashed=1;",
                                      vertex="1", parent="1")
        ET.SubElement(backbone_frame, 'mxGeometry', x="180", y="100", width="200", height="600", **{"as": "geometry"})
        
        # Backbone标签
        backbone_label = ET.SubElement(root, 'mxCell',
                                      id=str(cell_id + 1),
                                      value="Backbone",
                                      style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=12;fontColor=#4472C4;fontStyle=1;",
                                      vertex="1", parent="1")
        ET.SubElement(backbone_label, 'mxGeometry', x="220", y="710", width="80", height="20", **{"as": "geometry"})
        cell_id += 2
        
        # Head区域框架
        head_frame = ET.SubElement(root, 'mxCell',
                                  id=str(cell_id),
                                  value="",
                                  style="rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#E1A91A;strokeWidth=2;dashed=1;",
                                  vertex="1", parent="1")
        ET.SubElement(head_frame, 'mxGeometry', x="420", y="200", width="400", height="400", **{"as": "geometry"})
        
        # Head标签
        head_label = ET.SubElement(root, 'mxCell',
                                  id=str(cell_id + 1),
                                  value="Head",
                                  style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=12;fontColor=#E1A91A;fontStyle=1;",
                                  vertex="1", parent="1")
        ET.SubElement(head_label, 'mxGeometry', x="600", y="620", width="40", height="20", **{"as": "geometry"})
        cell_id += 2
        
        # 绘制Backbone层
        backbone_layers = config.get('backbone', [])
        y_pos = 120
        x_pos = 200
        
        for i, layer in enumerate(backbone_layers):
            module_type, display_text, color = self.get_layer_display_info(layer)
            
            # 创建层
            layer_style = (f"rounded=1;whiteSpace=wrap;html=1;fontSize=9;fontStyle=1;"
                          f"fillColor={color};strokeColor={self.stroke_color};"
                          f"strokeWidth=1;align=center;verticalAlign=middle;")
            
            layer_cell = ET.SubElement(root, 'mxCell',
                                      id=str(cell_id),
                                      value=display_text,
                                      style=layer_style,
                                      vertex="1", parent="1")
            
            # 调整宽度
            width = 140 if module_type == 'SPPF' else 120
            ET.SubElement(layer_cell, 'mxGeometry', x=str(x_pos), y=str(y_pos), width=str(width), height="40", **{"as": "geometry"})
            
            # 添加箭头连接
            if i > 0:
                arrow_cell = ET.SubElement(root, 'mxCell',
                                         id=str(cell_id + 5000),
                                         value="",
                                         style="endArrow=classic;html=1;strokeWidth=2;strokeColor=#000000;",
                                         edge="1", parent="1", source=str(cell_id-1), target=str(cell_id))
                ET.SubElement(arrow_cell, 'mxGeometry', width="50", height="50", relative="1", **{"as": "geometry"})
            
            cell_id += 1
            y_pos += 60
        
        # 绘制Head层
        head_layers = config.get('head', [])
        head_y_start = 220
        head_x_start = 440
        
        # 处理Head的复杂连接
        current_y = head_y_start
        
        for i, layer in enumerate(head_layers[:3]):  # 只处理前几层作为示例
            module_type, display_text, color = self.get_layer_display_info(layer)
            
            layer_style = (f"rounded=1;whiteSpace=wrap;html=1;fontSize=9;fontStyle=1;"
                          f"fillColor={color};strokeColor={self.stroke_color};"
                          f"strokeWidth=1;align=center;verticalAlign=middle;")
            
            layer_cell = ET.SubElement(root, 'mxCell',
                                      id=str(cell_id),
                                      value=display_text,
                                      style=layer_style,
                                      vertex="1", parent="1")
            ET.SubElement(layer_cell, 'mxGeometry', x=str(head_x_start), y=str(current_y), width="80", height="40", **{"as": "geometry"})
            
            cell_id += 1
            current_y += 60
        
        # 添加检测输出
        for i, scale in enumerate(['40×40×3(nc+5)', '80×80×3(nc+5)']):
            output_cell = ET.SubElement(root, 'mxCell',
                                       id=str(cell_id),
                                       value="",
                                       style="shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;fillColor=#f0f0f0;strokeColor=#000000;",
                                       vertex="1", parent="1")
            
            cube_y = 300 + i * 120
            ET.SubElement(output_cell, 'mxGeometry', x="750", y=str(cube_y), width="60", height="50", **{"as": "geometry"})
            
            # 输出标签
            output_label = ET.SubElement(root, 'mxCell',
                                        id=str(cell_id + 1),
                                        value=scale,
                                        style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=9;",
                                        vertex="1", parent="1")
            ET.SubElement(output_label, 'mxGeometry', x="820", y=str(cube_y + 15), width="80", height="20", **{"as": "geometry"})
            
            cell_id += 2
        
        return mxfile

    def save_paper_style_drawio(self, xml_tree, output_path):
        """保存论文风格的DrawIO文件"""
        rough_string = ET.tostring(xml_tree, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        formatted_xml = '\n'.join(lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_xml)

    def generate_paper_style_architectures(self):
        """生成论文风格的架构图"""
        cfg_dir = Path('cfg/training')
        output_dir = Path('docs/architecture_diagrams_paper')
        output_dir.mkdir(exist_ok=True)
        
        # 重点生成几个主要模型
        key_models = [
            'yolov4-csp-IDetect',
            'yolov4-repvgg-rcsosa',
            'yolov4-repvgg-rcsosa-simAM'
        ]
        
        generated_files = []
        
        for model_name in key_models:
            yaml_file = cfg_dir / f"{model_name}.yaml"
            
            if yaml_file.exists():
                print(f"📄 正在生成论文风格架构图: {model_name}")
                
                config = self.parse_yaml_config(yaml_file)
                xml_tree = self.create_paper_style_drawio(model_name, config)
                
                output_file = output_dir / f"{model_name}_paper_style.drawio"
                self.save_paper_style_drawio(xml_tree, output_file)
                
                generated_files.append(output_file)
                print(f"✅ 完成: {output_file}")
            else:
                print(f"❌ 文件不存在: {yaml_file}")
        
        return generated_files

if __name__ == "__main__":
    drawer = PaperStyleYOLODrawer()
    generated_files = drawer.generate_paper_style_architectures()
    
    print(f"\n📄 论文风格架构图生成完成!")
    print(f"生成了 {len(generated_files)} 个文件:")
    for file in generated_files:
        print(f"  📊 {file}")
    
    print(f"\n🎨 特点:")
    print("  - 完全参考提供的图像风格")
    print("  - 使用虚线框划分Backbone和Head区域") 
    print("  - 精确的颜色配色方案")
    print("  - 清晰的参数标注")
    print("  - 3D输出立方体显示")
