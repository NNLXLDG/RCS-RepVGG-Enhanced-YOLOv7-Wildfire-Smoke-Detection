#!/usr/bin/env python3
"""
YOLOv7 超参数可视化工具
生成所有训练配置的超参数对比图表
"""

import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_hyperparameters():
    """加载所有超参数文件"""
    hyperparams = {}
    hyp_dir = Path("hyperparameters")
    
    # 训练配置文件映射
    train_configs = {
        'hyp.train.yaml': 'YOLOv7基础版',
        'hyp.train-tiny.yaml': 'YOLOv7-Tiny',
        'hyp.train-repvgg.yaml': 'YOLOv7-RepVGG',
        'hyp.train-rcsosa.yaml': 'YOLOv7-RCSOSA',
        'hyp.train-repvgg-rcsosa.yaml': 'YOLOv7-RepVGG+RCSOSA',
        'hyp.train-tiny-repvgg.yaml': 'YOLOv7-Tiny-RepVGG',
        'hyp.train-tiny-rcsosa.yaml': 'YOLOv7-Tiny-RCSOSA',
        'hyp.train-tiny-repvgg-rcsosa.yaml': 'YOLOv7-Tiny-RepVGG+RCSOSA'
    }
    
    for file_name, display_name in train_configs.items():
        file_path = hyp_dir / file_name
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if data:
                        hyperparams[display_name] = data
                        print(f"✅ 加载成功: {display_name}")
                    else:
                        print(f"⚠️  文件为空: {file_name}")
            except Exception as e:
                print(f"❌ 加载失败: {file_name} - {e}")
        else:
            print(f"❌ 文件不存在: {file_name}")
    
    return hyperparams

def create_comparison_dataframe(hyperparams):
    """创建对比数据框"""
    # 核心超参数列表
    core_params = [
        'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs',
        'box', 'cls', 'obj', 'cls_pw', 'obj_pw', 'iou_t', 'anchor_t',
        'hsv_h', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fliplr',
        'mosaic', 'mixup', 'copy_paste', 'loss_ota'
    ]
    
    df_data = []
    for config_name, params in hyperparams.items():
        row = {'配置': config_name}
        for param in core_params:
            row[param] = params.get(param, 0)
        df_data.append(row)
    
    return pd.DataFrame(df_data)

def plot_learning_rate_comparison(df):
    """学习率相关参数对比"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('学习率与优化器参数对比', fontsize=16, fontweight='bold')
    
    # 学习率参数
    lr_params = ['lr0', 'lrf', 'momentum', 'weight_decay']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, param in enumerate(lr_params):
        ax = axes[i//2, i%2]
        bars = ax.bar(df['配置'], df[param], color=colors[i], alpha=0.8)
        ax.set_title(f'{param} 对比', fontweight='bold')
        ax.set_ylabel('数值')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('hyperparameters/learning_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_loss_weights_comparison(df):
    """损失权重参数对比"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('损失函数权重参数对比', fontsize=16, fontweight='bold')
    
    loss_params = ['box', 'cls', 'obj', 'cls_pw', 'obj_pw', 'iou_t']
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF']
    
    for i, param in enumerate(loss_params):
        ax = axes[i//3, i%3]
        bars = ax.bar(df['配置'], df[param], color=colors[i], alpha=0.8)
        ax.set_title(f'{param} 权重对比', fontweight='bold')
        ax.set_ylabel('权重值')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('hyperparameters/loss_weights_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_augmentation_comparison(df):
    """数据增强参数对比"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('数据增强参数对比', fontsize=16, fontweight='bold')
    
    aug_params = ['hsv_h', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fliplr', 'mosaic', 'mixup']
    colors = plt.cm.Set3(np.linspace(0, 1, len(aug_params)))
    
    for i, param in enumerate(aug_params):
        ax = axes[i//4, i%4]
        bars = ax.bar(df['配置'], df[param], color=colors[i], alpha=0.8)
        ax.set_title(f'{param} 增强强度', fontweight='bold')
        ax.set_ylabel('强度值')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('hyperparameters/augmentation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_heatmap_comparison(df):
    """超参数热力图对比"""
    # 准备数据
    heatmap_data = df.set_index('配置').T
    
    # 标准化数据到0-1范围
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(heatmap_data.values)
    normalized_df = pd.DataFrame(normalized_data, 
                                index=heatmap_data.index, 
                                columns=heatmap_data.columns)
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(normalized_df, 
                annot=True, 
                fmt='.2f', 
                cmap='RdYlBu_r',
                cbar_kws={'label': '标准化数值 (0-1)'},
                square=True)
    plt.title('YOLOv7 各配置超参数热力图对比\n(数值已标准化到0-1范围)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('模型配置', fontweight='bold')
    plt.ylabel('超参数', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('hyperparameters/hyperparameter_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_architecture_performance_radar(df):
    """架构性能雷达图"""
    from math import pi
    
    # 定义性能指标
    metrics = {
        '学习效率': ['lr0', 'momentum'],
        '精度潜力': ['cls', 'obj', 'iou_t'],
        '增强强度': ['mosaic', 'mixup', 'hsv_s'],
        '稳定性': ['weight_decay', 'lrf'],
        '复杂度': ['copy_paste', 'loss_ota']
    }
    
    # 计算每个配置的综合得分
    scores = {}
    for _, row in df.iterrows():
        config = row['配置']
        config_scores = []
        
        for metric, params in metrics.items():
            # 计算该指标的平均得分 (标准化)
            values = [row[param] for param in params if param in row]
            if values:
                if metric == '稳定性':  # 稳定性指标反向
                    score = 1 - np.mean(values)
                else:
                    score = np.mean(values)
                config_scores.append(max(0, min(1, score)))  # 限制在0-1范围
            else:
                config_scores.append(0)
        
        scores[config] = config_scores
    
    # 创建雷达图
    categories = list(metrics.keys())
    N = len(categories)
    
    # 计算角度
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # 闭合图形
    
    # 创建子图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), subplot_kw=dict(projection='polar'))
    fig.suptitle('YOLOv7 各配置架构性能雷达图', fontsize=16, fontweight='bold')
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(scores)))
    
    for i, (config, values) in enumerate(scores.items()):
        ax = axes[i//4, i%4]
        
        # 闭合数据
        values += values[:1]
        
        # 绘制雷达图
        ax.plot(angles, values, 'o-', linewidth=2, label=config, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title(config, fontweight='bold', pad=20)
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('hyperparameters/architecture_performance_radar.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report(df):
    """生成超参数总结报告"""
    report = {
        '总配置数': len(df),
        '平均学习率': df['lr0'].mean(),
        '最高学习率配置': df.loc[df['lr0'].idxmax(), '配置'],
        '最强增强配置': df.loc[(df['mosaic'] + df['mixup']).idxmax(), '配置'],
        '最高精度潜力配置': df.loc[(df['cls'] + df['obj']).idxmax(), '配置'],
        '使用OTA的配置数': df['loss_ota'].sum()
    }
    
    print("\n" + "="*60)
    print("📊 YOLOv7 超参数配置总结报告")
    print("="*60)
    
    for key, value in report.items():
        print(f"{key}: {value}")
    
    print("\n🎯 推荐配置:")
    print("• 高精度任务: YOLOv7-RepVGG+RCSOSA")
    print("• 快速推理: YOLOv7-Tiny")
    print("• 平衡性能: YOLOv7基础版")
    print("• 复杂场景: YOLOv7-RCSOSA")
    
    return report

def main():
    """主函数"""
    print("🚀 开始生成YOLOv7超参数可视化...")
    
    # 加载超参数
    hyperparams = load_hyperparameters()
    if not hyperparams:
        print("❌ 没有找到有效的超参数文件!")
        return
    
    # 创建对比数据框
    df = create_comparison_dataframe(hyperparams)
    print(f"\n📈 成功加载 {len(df)} 个配置")
    
    # 生成各种可视化图表
    print("\n📊 生成学习率参数对比图...")
    plot_learning_rate_comparison(df)
    
    print("📊 生成损失权重参数对比图...")
    plot_loss_weights_comparison(df)
    
    print("📊 生成数据增强参数对比图...")
    plot_augmentation_comparison(df)
    
    print("📊 生成超参数热力图...")
    plot_heatmap_comparison(df)
    
    print("📊 生成架构性能雷达图...")
    plot_architecture_performance_radar(df)
    
    # 生成总结报告
    report = generate_summary_report(df)
    
    # 保存数据到CSV
    csv_path = 'hyperparameters/hyperparameter_comparison_updated.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 数据已保存到: {csv_path}")
    
    print("\n✅ 超参数可视化更新完成!")
    print("📁 生成的图表文件:")
    print("  • learning_rate_comparison.png")
    print("  • loss_weights_comparison.png")
    print("  • augmentation_comparison.png")
    print("  • hyperparameter_heatmap.png")
    print("  • architecture_performance_radar.png")

if __name__ == "__main__":
    main()
