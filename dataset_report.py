#!/usr/bin/env python3
"""
数据集分析与报告生成工具
用于分析YOLO格式数据集并生成详细报告

使用方法:
python dataset_report.py --data path/to/dataset.yaml --output reports/
"""

import os
import yaml
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import defaultdict, Counter
from datetime import datetime
import pandas as pd
from tqdm import tqdm

class DatasetAnalyzer:
    def __init__(self, data_path, output_dir="reports"):
        """
        初始化数据集分析器
        
        Args:
            data_path: 数据集YAML配置文件路径
            output_dir: 报告输出目录
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据集配置
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data_config = yaml.safe_load(f)
        
        self.stats = {
            'total_images': 0,
            'total_annotations': 0,
            'classes': {},
            'image_sizes': [],
            'bbox_sizes': [],
            'bbox_ratios': [],
            'bbox_areas': [],
            'images_per_class': defaultdict(int),
            'annotations_per_image': [],
            'empty_images': 0,
            'corrupted_images': 0
        }
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def load_class_names(self):
        """加载类别名称"""
        if 'names' in self.data_config:
            if isinstance(self.data_config['names'], dict):
                return self.data_config['names']
            elif isinstance(self.data_config['names'], list):
                return {i: name for i, name in enumerate(self.data_config['names'])}
        
        # 如果没有names字段，根据nc生成默认名称
        nc = self.data_config.get('nc', 1)
        return {i: f'class_{i}' for i in range(nc)}
    
    def analyze_dataset(self):
        """分析整个数据集"""
        print("🔍 开始分析数据集...")
        
        class_names = self.load_class_names()
        self.stats['class_names'] = class_names
        
        # 分析训练集和验证集
        for split in ['train', 'val', 'test']:
            if split in self.data_config:
                print(f"\n📊 分析 {split} 集...")
                self._analyze_split(split)
        
        self._calculate_statistics()
        print("\n✅ 数据集分析完成!")
        
    def _analyze_split(self, split):
        """分析特定数据集分割"""
        split_path = Path(self.data_config[split])
        
        # 如果路径是相对路径，相对于当前工作目录而不是YAML文件所在目录
        if not split_path.is_absolute():
            # 首先尝试相对于当前工作目录
            if split_path.exists():
                pass  # 路径正确
            else:
                # 如果不存在，尝试相对于YAML文件所在目录
                yaml_relative_path = self.data_path.parent / split_path
                if yaml_relative_path.exists():
                    split_path = yaml_relative_path
                else:
                    # 尝试移除路径中重复的部分
                    path_parts = split_path.parts
                    if len(path_parts) > 1 and path_parts[0] == 'datasets':
                        # 移除开头的'datasets'部分，因为我们已经在datasets目录中
                        new_path = Path(*path_parts[1:])
                        if new_path.exists():
                            split_path = new_path
        
        if not split_path.exists():
            print(f"⚠️ 警告: {split} 路径不存在: {split_path}")
            return
        
        # 获取图像文件列表
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        image_files = []
        
        if split_path.is_file():
            # 如果是文件列表
            with open(split_path, 'r') as f:
                image_files = [line.strip() for line in f if line.strip()]
        else:
            # 如果是目录
            for ext in image_extensions:
                image_files.extend(split_path.glob(f'*{ext}'))
                image_files.extend(split_path.glob(f'*{ext.upper()}'))
        
        # 分析每个图像
        split_stats = {
            'images': 0,
            'annotations': 0,
            'empty_images': 0,
            'corrupted_images': 0
        }
        
        for img_file in tqdm(image_files, desc=f"分析{split}集"):
            try:
                self._analyze_image(img_file, split_stats)
            except Exception as e:
                print(f"⚠️ 处理图像失败 {img_file}: {e}")
                split_stats['corrupted_images'] += 1
        
        # 更新总体统计
        self.stats['total_images'] += split_stats['images']
        self.stats['total_annotations'] += split_stats['annotations']
        self.stats['empty_images'] += split_stats['empty_images']
        self.stats['corrupted_images'] += split_stats['corrupted_images']
        
        print(f"{split}集统计: {split_stats['images']}张图像, {split_stats['annotations']}个标注")
    
    def _analyze_image(self, img_path, split_stats):
        """分析单个图像"""
        img_path = Path(img_path)
        
        # 读取图像获取尺寸
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                split_stats['corrupted_images'] += 1
                return
            
            h, w = img.shape[:2]
            self.stats['image_sizes'].append((w, h))
            split_stats['images'] += 1
            
        except Exception as e:
            split_stats['corrupted_images'] += 1
            return
        
        # 查找对应的标注文件
        label_path = img_path.with_suffix('.txt')
        if img_path.parent.name == 'images':
            # 如果图像在images文件夹，标注可能在labels文件夹
            label_path = img_path.parent.parent / 'labels' / img_path.with_suffix('.txt').name
        
        if not label_path.exists():
            # 没有标注文件，认为是空图像
            self.stats['annotations_per_image'].append(0)
            split_stats['empty_images'] += 1
            return
        
        # 读取标注
        annotations = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            annotations.append((cls, x_center, y_center, width, height))
        except Exception as e:
            print(f"⚠️ 读取标注文件失败 {label_path}: {e}")
            return
        
        # 更新统计信息
        num_annotations = len(annotations)
        self.stats['annotations_per_image'].append(num_annotations)
        split_stats['annotations'] += num_annotations
        
        if num_annotations == 0:
            split_stats['empty_images'] += 1
        
        # 分析每个标注
        for cls, x_center, y_center, bbox_w, bbox_h in annotations:
            # 转换为像素坐标
            pixel_w = bbox_w * w
            pixel_h = bbox_h * h
            
            # 更新类别统计
            if cls not in self.stats['classes']:
                self.stats['classes'][cls] = 0
            self.stats['classes'][cls] += 1
            self.stats['images_per_class'][cls] += 1
            
            # 边界框统计
            self.stats['bbox_sizes'].append((pixel_w, pixel_h))
            self.stats['bbox_ratios'].append(pixel_w / pixel_h if pixel_h > 0 else 1.0)
            self.stats['bbox_areas'].append(pixel_w * pixel_h)
    
    def _calculate_statistics(self):
        """计算统计指标"""
        if self.stats['image_sizes']:
            widths, heights = zip(*self.stats['image_sizes'])
            self.stats['image_width_stats'] = {
                'mean': np.mean(widths),
                'std': np.std(widths),
                'min': np.min(widths),
                'max': np.max(widths),
                'median': np.median(widths)
            }
            self.stats['image_height_stats'] = {
                'mean': np.mean(heights),
                'std': np.std(heights),
                'min': np.min(heights),
                'max': np.max(heights),
                'median': np.median(heights)
            }
        
        if self.stats['bbox_sizes']:
            bbox_widths, bbox_heights = zip(*self.stats['bbox_sizes'])
            self.stats['bbox_width_stats'] = {
                'mean': np.mean(bbox_widths),
                'std': np.std(bbox_widths),
                'min': np.min(bbox_widths),
                'max': np.max(bbox_widths),
                'median': np.median(bbox_widths)
            }
            self.stats['bbox_height_stats'] = {
                'mean': np.mean(bbox_heights),
                'std': np.std(bbox_heights),
                'min': np.min(bbox_heights),
                'max': np.max(bbox_heights),
                'median': np.median(bbox_heights)
            }
        
        if self.stats['bbox_ratios']:
            self.stats['bbox_ratio_stats'] = {
                'mean': np.mean(self.stats['bbox_ratios']),
                'std': np.std(self.stats['bbox_ratios']),
                'min': np.min(self.stats['bbox_ratios']),
                'max': np.max(self.stats['bbox_ratios']),
                'median': np.median(self.stats['bbox_ratios'])
            }
        
        if self.stats['bbox_areas']:
            self.stats['bbox_area_stats'] = {
                'mean': np.mean(self.stats['bbox_areas']),
                'std': np.std(self.stats['bbox_areas']),
                'min': np.min(self.stats['bbox_areas']),
                'max': np.max(self.stats['bbox_areas']),
                'median': np.median(self.stats['bbox_areas'])
            }
    
    def generate_plots(self):
        """生成可视化图表"""
        print("📊 生成可视化图表...")
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. 类别分布图
        if self.stats['classes']:
            self._plot_class_distribution()
        
        # 2. 图像尺寸分布
        if self.stats['image_sizes']:
            self._plot_image_size_distribution()
        
        # 3. 边界框尺寸分布
        if self.stats['bbox_sizes']:
            self._plot_bbox_size_distribution()
        
        # 4. 边界框长宽比分布
        if self.stats['bbox_ratios']:
            self._plot_bbox_ratio_distribution()
        
        # 5. 每张图像的标注数量分布
        if self.stats['annotations_per_image']:
            self._plot_annotations_per_image()
        
        print("✅ 图表生成完成!")
    
    def _plot_class_distribution(self):
        """绘制类别分布图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        class_names = self.stats['class_names']
        classes = list(self.stats['classes'].keys())
        counts = list(self.stats['classes'].values())
        labels = [class_names.get(cls, f'Class {cls}') for cls in classes]
        
        # 条形图
        bars = ax1.bar(labels, counts, color='skyblue', alpha=0.8)
        ax1.set_title('类别标注数量分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('类别')
        ax1.set_ylabel('标注数量')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom')
        
        # 饼图
        ax2.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('类别分布比例', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_image_size_distribution(self):
        """绘制图像尺寸分布图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        widths, heights = zip(*self.stats['image_sizes'])
        
        # 宽度分布
        ax1.hist(widths, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('图像宽度分布', fontsize=12, fontweight='bold')
        ax1.set_xlabel('宽度 (像素)')
        ax1.set_ylabel('频次')
        ax1.axvline(np.mean(widths), color='red', linestyle='--', label=f'均值: {np.mean(widths):.0f}')
        ax1.legend()
        
        # 高度分布
        ax2.hist(heights, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.set_title('图像高度分布', fontsize=12, fontweight='bold')
        ax2.set_xlabel('高度 (像素)')
        ax2.set_ylabel('频次')
        ax2.axvline(np.mean(heights), color='red', linestyle='--', label=f'均值: {np.mean(heights):.0f}')
        ax2.legend()
        
        # 宽高比分布
        ratios = [w/h for w, h in self.stats['image_sizes']]
        ax3.hist(ratios, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax3.set_title('图像宽高比分布', fontsize=12, fontweight='bold')
        ax3.set_xlabel('宽高比')
        ax3.set_ylabel('频次')
        ax3.axvline(np.mean(ratios), color='red', linestyle='--', label=f'均值: {np.mean(ratios):.2f}')
        ax3.legend()
        
        # 尺寸散点图
        ax4.scatter(widths, heights, alpha=0.6, s=20)
        ax4.set_title('图像尺寸分布散点图', fontsize=12, fontweight='bold')
        ax4.set_xlabel('宽度 (像素)')
        ax4.set_ylabel('高度 (像素)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'image_size_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_bbox_size_distribution(self):
        """绘制边界框尺寸分布图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        bbox_widths, bbox_heights = zip(*self.stats['bbox_sizes'])
        
        # 边界框宽度分布
        ax1.hist(bbox_widths, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax1.set_title('边界框宽度分布', fontsize=12, fontweight='bold')
        ax1.set_xlabel('宽度 (像素)')
        ax1.set_ylabel('频次')
        ax1.axvline(np.mean(bbox_widths), color='red', linestyle='--', 
                   label=f'均值: {np.mean(bbox_widths):.0f}')
        ax1.legend()
        
        # 边界框高度分布
        ax2.hist(bbox_heights, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax2.set_title('边界框高度分布', fontsize=12, fontweight='bold')
        ax2.set_xlabel('高度 (像素)')
        ax2.set_ylabel('频次')
        ax2.axvline(np.mean(bbox_heights), color='red', linestyle='--',
                   label=f'均值: {np.mean(bbox_heights):.0f}')
        ax2.legend()
        
        # 边界框面积分布
        ax3.hist(self.stats['bbox_areas'], bins=50, alpha=0.7, color='cyan', edgecolor='black')
        ax3.set_title('边界框面积分布', fontsize=12, fontweight='bold')
        ax3.set_xlabel('面积 (像素²)')
        ax3.set_ylabel('频次')
        bbox_area_mean = np.mean(self.stats['bbox_areas'])
        ax3.axvline(bbox_area_mean, color='red', linestyle='--',
                   label=f'均值: {bbox_area_mean:.0f}')
        ax3.legend()
        
        # 边界框尺寸散点图
        ax4.scatter(bbox_widths, bbox_heights, alpha=0.6, s=20, color='magenta')
        ax4.set_title('边界框尺寸散点图', fontsize=12, fontweight='bold')
        ax4.set_xlabel('宽度 (像素)')
        ax4.set_ylabel('高度 (像素)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bbox_size_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_bbox_ratio_distribution(self):
        """绘制边界框长宽比分布"""
        plt.figure(figsize=(10, 6))
        
        plt.hist(self.stats['bbox_ratios'], bins=50, alpha=0.7, color='gold', edgecolor='black')
        plt.title('边界框长宽比分布', fontsize=14, fontweight='bold')
        plt.xlabel('长宽比 (宽度/高度)')
        plt.ylabel('频次')
        bbox_ratio_mean = np.mean(self.stats['bbox_ratios'])
        plt.axvline(bbox_ratio_mean, color='red', linestyle='--',
                   label=f'均值: {bbox_ratio_mean:.2f}')
        plt.axvline(1.0, color='green', linestyle='--', label='正方形 (1:1)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bbox_ratio_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_annotations_per_image(self):
        """绘制每张图像标注数量分布"""
        plt.figure(figsize=(12, 6))
        
        # 统计每张图像的标注数量
        annotation_counts = Counter(self.stats['annotations_per_image'])
        counts = list(annotation_counts.keys())
        frequencies = list(annotation_counts.values())
        
        plt.bar(counts, frequencies, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('每张图像标注数量分布', fontsize=14, fontweight='bold')
        plt.xlabel('每张图像的标注数量')
        plt.ylabel('图像数量')
        
        # 添加统计信息
        mean_annotations = np.mean(self.stats['annotations_per_image'])
        plt.axvline(mean_annotations, color='red', linestyle='--',
                   label=f'均值: {mean_annotations:.2f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'annotations_per_image.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """生成详细报告"""
        print("📝 生成数据集报告...")
        
        # 生成HTML报告
        self._generate_html_report()
        
        # 生成JSON报告
        self._generate_json_report()
        
        # 生成CSV报告
        self._generate_csv_report()
        
        print("✅ 报告生成完成!")
    
    def _generate_html_report(self):
        """生成HTML格式报告"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>数据集分析报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        .plot-section {{
            margin: 30px 0;
        }}
        .plot-image {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .info {{
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 数据集分析报告</h1>
        
        <div class="info">
            <strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <strong>数据集配置:</strong> {self.data_path.name}<br>
            <strong>类别数量:</strong> {self.data_config.get('nc', 'Unknown')}
        </div>

        <h2>📊 总体统计</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{self.stats['total_images']:,}</div>
                <div class="stat-label">总图像数量</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.stats['total_annotations']:,}</div>
                <div class="stat-label">总标注数量</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(self.stats['classes'])}</div>
                <div class="stat-label">类别数量</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.stats['empty_images']:,}</div>
                <div class="stat-label">空图像数量</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.stats['corrupted_images']:,}</div>
                <div class="stat-label">损坏图像数量</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{np.mean(self.stats['annotations_per_image']):.2f}</div>
                <div class="stat-label">平均每图标注数</div>
            </div>
        </div>

        <h2>🏷️ 类别分布</h2>
        <table>
            <tr>
                <th>类别ID</th>
                <th>类别名称</th>
                <th>标注数量</th>
                <th>占比</th>
            </tr>
        """
        
        total_annotations = sum(self.stats['classes'].values())
        for cls_id, count in self.stats['classes'].items():
            cls_name = self.stats['class_names'].get(cls_id, f'Class {cls_id}')
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            html_content += f"""
            <tr>
                <td>{cls_id}</td>
                <td>{cls_name}</td>
                <td>{count:,}</td>
                <td>{percentage:.2f}%</td>
            </tr>
            """
        
        html_content += """
        </table>
        """
        
        # 添加图像统计信息
        if 'image_width_stats' in self.stats:
            html_content += f"""
            <h2>🖼️ 图像尺寸统计</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{self.stats['image_width_stats']['mean']:.0f}</div>
                    <div class="stat-label">平均宽度 (像素)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{self.stats['image_height_stats']['mean']:.0f}</div>
                    <div class="stat-label">平均高度 (像素)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{self.stats['image_width_stats']['min']:.0f} - {self.stats['image_width_stats']['max']:.0f}</div>
                    <div class="stat-label">宽度范围 (像素)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{self.stats['image_height_stats']['min']:.0f} - {self.stats['image_height_stats']['max']:.0f}</div>
                    <div class="stat-label">高度范围 (像素)</div>
                </div>
            </div>
            """
        
        # 添加边界框统计信息
        if 'bbox_width_stats' in self.stats:
            html_content += f"""
            <h2>📦 边界框统计</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{self.stats['bbox_width_stats']['mean']:.1f}</div>
                    <div class="stat-label">平均宽度 (像素)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{self.stats['bbox_height_stats']['mean']:.1f}</div>
                    <div class="stat-label">平均高度 (像素)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{self.stats['bbox_ratio_stats']['mean']:.2f}</div>
                    <div class="stat-label">平均长宽比</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{self.stats['bbox_area_stats']['mean']:.0f}</div>
                    <div class="stat-label">平均面积 (像素²)</div>
                </div>
            </div>
            """
        
        # 添加警告信息
        if self.stats['empty_images'] > 0:
            html_content += f"""
            <div class="warning">
                <strong>⚠️ 注意:</strong> 发现 {self.stats['empty_images']} 张空图像（无标注）。
                这可能影响训练效果，建议检查这些图像是否需要标注或移除。
            </div>
            """
        
        if self.stats['corrupted_images'] > 0:
            html_content += f"""
            <div class="warning">
                <strong>⚠️ 注意:</strong> 发现 {self.stats['corrupted_images']} 张损坏图像。
                这些图像无法正常读取，建议修复或移除。
            </div>
            """
        
        # 添加可视化图表
        plot_files = [
            ('class_distribution.png', '类别分布'),
            ('image_size_distribution.png', '图像尺寸分布'),
            ('bbox_size_distribution.png', '边界框尺寸分布'),
            ('bbox_ratio_distribution.png', '边界框长宽比分布'),
            ('annotations_per_image.png', '每图标注数量分布')
        ]
        
        html_content += """
        <h2>📈 可视化图表</h2>
        """
        
        for plot_file, plot_title in plot_files:
            plot_path = self.output_dir / plot_file
            if plot_path.exists():
                html_content += f"""
                <div class="plot-section">
                    <h3>{plot_title}</h3>
                    <img src="{plot_file}" alt="{plot_title}" class="plot-image">
                </div>
                """
        
        html_content += """
        </div>
    </body>
    </html>
        """
        
        # 保存HTML报告
        with open(self.output_dir / 'dataset_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _convert_to_serializable(self, obj):
        """递归转换对象为JSON可序列化的格式"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(item) for item in obj)
        else:
            return obj
    
    def _generate_json_report(self):
        """生成JSON格式报告"""
        # 准备JSON数据（移除不能序列化的numpy数组）
        json_stats = {}
        for key, value in self.stats.items():
            if key in ['image_sizes', 'bbox_sizes', 'bbox_ratios', 'bbox_areas', 'annotations_per_image']:
                # 转换为基本统计信息
                if value:
                    json_stats[f'{key}_stats'] = {
                        'count': len(value),
                        'mean': float(np.mean(value)),
                        'std': float(np.std(value)),
                        'min': float(np.min(value)),
                        'max': float(np.max(value)),
                        'median': float(np.median(value))
                    }
            else:
                # 使用辅助函数确保所有数据都可序列化
                json_stats[key] = self._convert_to_serializable(value)
        
        # 添加生成时间
        json_stats['generated_at'] = datetime.now().isoformat()
        json_stats['dataset_config'] = self.data_config
        
        # 保存JSON报告
        with open(self.output_dir / 'dataset_report.json', 'w', encoding='utf-8') as f:
            json.dump(json_stats, f, ensure_ascii=False, indent=2)
    
    def _generate_csv_report(self):
        """生成CSV格式报告"""
        # 类别统计CSV
        if self.stats['classes']:
            class_data = []
            total_annotations = sum(self.stats['classes'].values())
            
            for cls_id, count in self.stats['classes'].items():
                cls_name = self.stats['class_names'].get(cls_id, f'Class {cls_id}')
                percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
                class_data.append({
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'annotation_count': count,
                    'percentage': percentage
                })
            
            df_classes = pd.DataFrame(class_data)
            df_classes.to_csv(self.output_dir / 'class_statistics.csv', index=False, encoding='utf-8-sig')
        
        # 图像统计CSV
        if self.stats['image_sizes']:
            image_data = []
            for i, (w, h) in enumerate(self.stats['image_sizes']):
                num_annotations = self.stats['annotations_per_image'][i] if i < len(self.stats['annotations_per_image']) else 0
                image_data.append({
                    'image_id': i,
                    'width': w,
                    'height': h,
                    'aspect_ratio': w/h if h > 0 else 1.0,
                    'area': w*h,
                    'num_annotations': num_annotations
                })
            
            df_images = pd.DataFrame(image_data)
            df_images.to_csv(self.output_dir / 'image_statistics.csv', index=False, encoding='utf-8-sig')

def main():
    parser = argparse.ArgumentParser(description='生成YOLO数据集分析报告')
    parser.add_argument('--data', type=str, required=True, help='数据集YAML配置文件路径')
    parser.add_argument('--output', type=str, default='reports', help='报告输出目录')
    parser.add_argument('--plots', action='store_true', help='生成可视化图表')
    
    args = parser.parse_args()
    
    # 检查数据集配置文件是否存在
    if not Path(args.data).exists():
        print(f"❌ 错误: 数据集配置文件不存在: {args.data}")
        return
    
    print("🚀 开始数据集分析...")
    print(f"📁 数据集配置: {args.data}")
    print(f"📂 输出目录: {args.output}")
    
    # 创建分析器并执行分析
    analyzer = DatasetAnalyzer(args.data, args.output)
    
    try:
        # 分析数据集
        analyzer.analyze_dataset()
        
        # 生成可视化图表
        if args.plots:
            analyzer.generate_plots()
        
        # 生成报告
        analyzer.generate_report()
        
        print(f"\n🎉 分析完成! 报告已保存到: {args.output}")
        print(f"📊 HTML报告: {Path(args.output) / 'dataset_report.html'}")
        print(f"📄 JSON报告: {Path(args.output) / 'dataset_report.json'}")
        print(f"📋 CSV报告: {Path(args.output) / 'class_statistics.csv'}")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
