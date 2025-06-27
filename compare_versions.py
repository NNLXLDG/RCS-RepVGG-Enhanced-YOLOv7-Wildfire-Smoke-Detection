#!/usr/bin/env python3
"""
YOLOv7 四版本性能对比脚本
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

# 四个版本的配置
VERSIONS = {
    'yolov7': {
        'config': 'cfg/training/yolov7.yaml',
        'train_script': 'train.py',
        'detect_script': 'detect.py',
        'description': 'YOLOv7 基础版本'
    },
    'yolov7-repvgg': {
        'config': 'cfg/training/yolov7-repvgg.yaml',
        'train_script': 'train-repvgg.py',
        'detect_script': 'detect-repvgg.py',
        'description': 'YOLOv7 + RepVGG'
    },
    'yolov7-rcsosa': {
        'config': 'cfg/training/yolov7-rcsosa.yaml',
        'train_script': 'train-rcsosa.py',
        'detect_script': 'detect-rcsosa.py',
        'description': 'YOLOv7 + RCSOSA'
    },
    'yolov7-repvgg-rcsosa': {
        'config': 'cfg/training/yolov7-repvgg-rcsosa.yaml',
        'train_script': 'train-repvgg-rcsosa.py',
        'detect_script': 'detect-repvgg-rcsosa.py',
        'description': 'YOLOv7 + RepVGG + RCSOSA'
    }
}

class YOLOv7Comparator:
    def __init__(self, data_path='datasets/smokefire.yaml', 
                 epochs=50, batch_size=4, img_size=640):
        self.data_path = data_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.results = {}
        
    def train_version(self, version_name, version_config):
        """训练指定版本"""
        print(f"\n{'='*50}")
        print(f"开始训练: {version_config['description']}")
        print(f"{'='*50}")
        
        # 构建训练命令
        cmd = [
            'python', version_config['train_script'],
            '--data', self.data_path,
            '--cfg', version_config['config'],
            '--epochs', str(self.epochs),
            '--batch-size', str(self.batch_size),
            '--img-size', str(self.img_size),
            '--device', 'cpu',
            '--workers', '1',
            '--name', version_name
        ]
        
        print(f"训练命令: {' '.join(cmd)}")
        
        # 记录训练开始时间
        start_time = time.time()
        
        try:
            # 执行训练
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
            
            # 记录训练结束时间
            end_time = time.time()
            training_time = end_time - start_time
            
            # 保存结果
            self.results[version_name] = {
                'config': version_config,
                'training_time': training_time,
                'success': result.returncode == 0,
                'stdout': result.stdout[-2000:] if result.stdout else '',  # 保存最后2000字符
                'stderr': result.stderr[-2000:] if result.stderr else ''
            }
            
            if result.returncode == 0:
                print(f"✅ {version_name} 训练成功，耗时: {training_time:.1f}秒")
                # 尝试提取mAP等指标
                self.extract_metrics(version_name)
            else:
                print(f"❌ {version_name} 训练失败")
                print(f"错误信息: {result.stderr[-500:]}")  # 只显示最后500字符
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {version_name} 训练超时（2小时）")
            self.results[version_name] = {
                'config': version_config,
                'training_time': 7200,  # 超时时间
                'success': False,
                'timeout': True,
                'stdout': '',
                'stderr': '训练超时'
            }
        except Exception as e:
            print(f"💥 {version_name} 训练出现异常: {str(e)}")
            self.results[version_name] = {
                'config': version_config,
                'training_time': time.time() - start_time,
                'success': False,
                'exception': str(e),
                'stdout': '',
                'stderr': str(e)
            }
            self.results[version_name] = {
                'config': version_config,
                'training_time': 3600*4,
                'success': False,
                'error': 'Training timeout'
            }
        except Exception as e:
            print(f"❌ {version_name} 训练异常: {e}")
            self.results[version_name] = {
                'config': version_config,
                'training_time': 0,
                'success': False,
                'error': str(e)
            }
    
    def extract_metrics(self, version_name):
        """从训练结果中提取指标"""
        try:
            # 查找结果文件
            results_file = Path(f'runs/train/{version_name}/results.txt')
            if results_file.exists():
                with open(results_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # 解析最后一行的结果
                        last_line = lines[-1].strip()
                        parts = last_line.split()
                        if len(parts) >= 10:
                            # 提取主要指标
                            epoch = parts[0]
                            precision = float(parts[4])
                            recall = float(parts[5])
                            map50 = float(parts[6])
                            map50_95 = float(parts[7])
                            
                            self.results[version_name].update({
                                'final_epoch': epoch,
                                'precision': precision,
                                'recall': recall,
                                'mAP@0.5': map50,
                                'mAP@0.5:0.95': map50_95
                            })
        except Exception as e:
            print(f"警告: 无法提取 {version_name} 的指标: {e}")
    
    def compare_all_versions(self):
        """比较所有版本"""
        print("开始YOLOv7四版本对比实验...")
        print(f"数据集: {self.data_path}")
        print(f"训练轮数: {self.epochs}")
        print(f"批次大小: {self.batch_size}")
        print(f"图像尺寸: {self.img_size}")
        
        # 按顺序训练每个版本
        for version_name, version_config in VERSIONS.items():
            self.train_version(version_name, version_config)
        
        # 生成对比报告
        self.generate_report()
    
    def generate_report(self):
        """生成对比报告"""
        print(f"\n{'='*70}")
        print("YOLOv7 四版本对比结果")
        print(f"{'='*70}")
        
        # 表格标题
        print(f"{'版本':<20} {'状态':<8} {'训练时间':<12} {'mAP@0.5':<10} {'mAP@0.5:0.95':<12} {'精确度':<8} {'召回率':<8}")
        print(f"{'-'*70}")
        
        successful_versions = []
        
        for version_name, result in self.results.items():
            status = "成功" if result.get('success', False) else "失败"
            training_time = f"{result.get('training_time', 0):.1f}s"
            map50 = f"{result.get('mAP@0.5', 0):.3f}" if result.get('success') else "N/A"
            map50_95 = f"{result.get('mAP@0.5:0.95', 0):.3f}" if result.get('success') else "N/A"
            precision = f"{result.get('precision', 0):.3f}" if result.get('success') else "N/A"
            recall = f"{result.get('recall', 0):.3f}" if result.get('success') else "N/A"
            
            print(f"{version_name:<20} {status:<8} {training_time:<12} {map50:<10} {map50_95:<12} {precision:<8} {recall:<8}")
            
            if result.get('success'):
                successful_versions.append(version_name)
        
        # 保存详细结果到JSON
        with open('comparison_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细结果已保存到: comparison_results.json")
        
        # 推荐最佳版本
        if successful_versions:
            best_version = self.find_best_version(successful_versions)
            if best_version:
                print(f"\n🏆 推荐版本: {best_version}")
                print(f"   描述: {VERSIONS[best_version]['description']}")
                print(f"   mAP@0.5: {self.results[best_version].get('mAP@0.5', 'N/A')}")
        
    def find_best_version(self, successful_versions):
        """找到最佳版本（基于mAP@0.5:0.95）"""
        best_version = None
        best_map = 0
        
        for version in successful_versions:
            map_score = self.results[version].get('mAP@0.5:0.95', 0)
            if map_score > best_map:
                best_map = map_score
                best_version = version
        
        return best_version

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv7四版本对比实验')
    parser.add_argument('--data', type=str, default='datasets/smokefire.yaml', 
                        help='数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--img-size', type=int, default=640, help='图像尺寸')
    parser.add_argument('--versions', nargs='+', choices=list(VERSIONS.keys()),
                        default=list(VERSIONS.keys()), help='要比较的版本')
    
    args = parser.parse_args()
    
    # 过滤版本
    global VERSIONS
    if args.versions != list(VERSIONS.keys()):
        VERSIONS = {k: v for k, v in VERSIONS.items() if k in args.versions}
    
    # 创建比较器并运行
    comparator = YOLOv7Comparator(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    comparator.compare_all_versions()

if __name__ == '__main__':
    main()
