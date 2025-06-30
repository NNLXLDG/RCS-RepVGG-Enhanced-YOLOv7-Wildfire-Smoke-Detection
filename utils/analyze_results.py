#!/usr/bin/env python3
"""
YOLOv7 训练结果分析工具
分析 results.txt 文件中的训练指标
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_training_results(results_file):
    """分析训练结果文件"""
    
    # 定义列名
    columns = [
        'epoch_info', 'gpu_mem', 'box_loss', 'obj_loss', 'cls_loss', 'total_loss',
        'labels', 'img_size', 'precision', 'recall', 'mAP_0.5', 'mAP_0.5_0.95',
        'val_box_loss', 'val_obj_loss', 'val_cls_loss'
    ]
    
    try:
        # 读取数据
        df = pd.read_csv(results_file, sep='\s+', header=None, names=columns)
        
        # 提取epoch编号
        df['epoch'] = df['epoch_info'].str.split('/').str[0].astype(int)
        total_epochs = df['epoch_info'].str.split('/').str[1].iloc[0]
        
        print(f"📊 训练结果分析 - {results_file}")
        print("=" * 60)
        print(f"📈 总训练轮次: {total_epochs}")
        print(f"🔄 已完成轮次: {len(df)}")
        print(f"⏱️  完成进度: {len(df)}/{total_epochs} ({len(df)/int(total_epochs)*100:.1f}%)")
        print()
        
        # 当前性能指标
        latest = df.iloc[-1]
        print("🎯 最新性能指标:")
        print(f"├─ 总损失:        {latest['total_loss']:.4f}")
        print(f"├─ 精确率:        {latest['precision']:.4f} ({latest['precision']*100:.1f}%)")
        print(f"├─ 召回率:        {latest['recall']:.4f} ({latest['recall']*100:.1f}%)")
        print(f"├─ mAP@0.5:      {latest['mAP_0.5']:.4f} ({latest['mAP_0.5']*100:.1f}%)")
        print(f"└─ mAP@0.5:0.95: {latest['mAP_0.5_0.95']:.4f} ({latest['mAP_0.5_0.95']*100:.1f}%)")
        print()
        
        # 改进情况
        if len(df) > 1:
            first = df.iloc[0]
            improvement = {
                'total_loss': (first['total_loss'] - latest['total_loss']) / first['total_loss'] * 100,
                'precision': (latest['precision'] - first['precision']) / first['precision'] * 100,
                'recall': (latest['recall'] - first['recall']) / first['recall'] * 100,
                'mAP_0.5': (latest['mAP_0.5'] - first['mAP_0.5']) / first['mAP_0.5'] * 100
            }
            
            print("📈 训练改进情况 (相对于第0轮):")
            print(f"├─ 总损失降低:    {improvement['total_loss']:.1f}%")
            print(f"├─ 精确率提升:    {improvement['precision']:.1f}%")
            print(f"├─ 召回率提升:    {improvement['recall']:.1f}%")
            print(f"└─ mAP@0.5提升:   {improvement['mAP_0.5']:.1f}%")
            print()
        
        # 最佳性能
        best_map = df['mAP_0.5'].max()
        best_epoch = df.loc[df['mAP_0.5'].idxmax(), 'epoch']
        print(f"🏆 最佳 mAP@0.5: {best_map:.4f} (第{best_epoch}轮)")
        
        # 趋势分析
        recent_epochs = min(5, len(df))
        recent_trend = df.tail(recent_epochs)
        
        if len(recent_trend) >= 2:
            loss_trend = "下降" if recent_trend['total_loss'].iloc[-1] < recent_trend['total_loss'].iloc[0] else "上升"
            map_trend = "上升" if recent_trend['mAP_0.5'].iloc[-1] > recent_trend['mAP_0.5'].iloc[0] else "下降"
            print(f"📊 最近{recent_epochs}轮趋势: 损失{loss_trend}, mAP{map_trend}")
        
        print()
        print("💡 建议:")
        
        # 基于数据给出建议
        if latest['mAP_0.5'] < 0.5:
            print("├─ mAP@0.5较低，建议继续训练")
        elif latest['mAP_0.5'] > 0.8:
            print("├─ mAP@0.5表现优秀！")
            
        if latest['precision'] > 0.8 and latest['recall'] > 0.7:
            print("├─ 精确率和召回率均表现良好")
        elif latest['precision'] > latest['recall']:
            print("├─ 召回率相对较低，可能存在漏检")
        else:
            print("├─ 精确率相对较低，可能存在误检")
            
        if len(df) > 10 and df.tail(3)['total_loss'].std() < 0.001:
            print("└─ 损失趋于稳定，可能接近收敛")
        else:
            print("└─ 继续训练以获得更好效果")
            
        return df
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None

def plot_training_curves(df, save_path=None):
    """绘制训练曲线"""
    if df is None:
        return
        
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('YOLOv7 训练过程分析', fontsize=16, fontweight='bold')
    
    # 1. 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['total_loss'], 'r-', label='总损失', linewidth=2)
    ax1.plot(df['epoch'], df['box_loss'], 'b-', label='边界框损失', alpha=0.7)
    ax1.plot(df['epoch'], df['obj_loss'], 'g-', label='目标损失', alpha=0.7)
    ax1.plot(df['epoch'], df['cls_loss'], 'm-', label='分类损失', alpha=0.7)
    ax1.set_title('训练损失变化')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 验证损失
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['val_box_loss'], 'b--', label='验证边界框损失')
    ax2.plot(df['epoch'], df['val_obj_loss'], 'g--', label='验证目标损失')
    ax2.plot(df['epoch'], df['val_cls_loss'], 'm--', label='验证分类损失')
    ax2.set_title('验证损失变化')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 精确率和召回率
    ax3 = axes[0, 2]
    ax3.plot(df['epoch'], df['precision'], 'b-', label='精确率', linewidth=2)
    ax3.plot(df['epoch'], df['recall'], 'r-', label='召回率', linewidth=2)
    ax3.set_title('精确率和召回率')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # 4. mAP指标
    ax4 = axes[1, 0]
    ax4.plot(df['epoch'], df['mAP_0.5'], 'g-', label='mAP@0.5', linewidth=2)
    ax4.plot(df['epoch'], df['mAP_0.5_0.95'], 'orange', label='mAP@0.5:0.95', linewidth=2)
    ax4.set_title('平均精度 (mAP)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('mAP')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # 5. 训练vs验证损失对比
    ax5 = axes[1, 1]
    ax5.plot(df['epoch'], df['total_loss'], 'b-', label='训练总损失', linewidth=2)
    val_total = df['val_box_loss'] + df['val_obj_loss'] + df['val_cls_loss']
    ax5.plot(df['epoch'], val_total, 'r--', label='验证总损失', linewidth=2)
    ax5.set_title('训练 vs 验证损失')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Total Loss')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 综合性能指标
    ax6 = axes[1, 2]
    # F1 score calculation
    f1_score = 2 * (df['precision'] * df['recall']) / (df['precision'] + df['recall'] + 1e-8)
    ax6.plot(df['epoch'], f1_score, 'purple', label='F1-Score', linewidth=2)
    ax6.plot(df['epoch'], df['mAP_0.5'], 'g-', label='mAP@0.5', linewidth=2)
    ax6.set_title('综合性能指标')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Score')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 训练曲线已保存到: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # 分析当前训练结果
    results_file = "runs/train/yolov7-tiny_tiny_smokefire_ep100_bs8_20250629_214219/results.txt"
    
    if Path(results_file).exists():
        print("🔍 分析训练结果文件...")
        df = analyze_training_results(results_file)
        
        if df is not None:
            print("\n📊 生成训练曲线图...")
            save_path = Path(results_file).parent / "training_analysis.png"
            plot_training_curves(df, save_path)
    else:
        print(f"❌ 找不到结果文件: {results_file}")
        print("请确认文件路径是否正确")
