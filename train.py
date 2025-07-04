import argparse          
import logging          
import math         
import os           
import random           
import time             
from copy import deepcopy    
from datetime import datetime
from pathlib import Path     
from threading import Thread 


import numpy as np      


import torch.distributed as dist      
import torch.nn as nn                  
import torch.nn.functional as F       
import torch.optim as optim           
import torch.optim.lr_scheduler as lr_scheduler  
import torch.utils.data          
import yaml          

# 自动混合精度训练 - 已为CPU训练禁用
# AMP (Automatic Mixed Precision) 自动混合精度训练可以加速GPU训练并减少显存使用
# 但在CPU训练中不需要，因此已注释掉以避免兼容性问题
# from torch.cuda import amp  # CUDA AMP (已为CPU训练注释)

# PyTorch扩展功能
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.tensorboard import SummaryWriter     
from tqdm import tqdm         

# 项目模块导入
import test  # 导入test.py用于每个epoch后计算mAP
from models.experimental import attempt_load    
from models.yolo import Model                   
from utils.autoanchor import check_anchors      
from utils.datasets import create_dataloader    
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr  
from utils.google_utils import attempt_download     # Google云下载工具
from utils.loss import ComputeLoss, ComputeLossOTA  # 损失函数计算
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution  # 绘图工具
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel  # PyTorch工具
from utils.experiment_manager import create_training_report  # 训练报告生成

# 创建日志记录器
logger = logging.getLogger(__name__)


def print_training_info(save_dir, model_cfg, dataset_cfg, epochs, batch_size):
    """
    Print training configuration information and save path - optimized version with cleaner formatting
    """
    # Use more beautiful separators and emoji icons
    separator = "━" * 80
    logger.info(f"\n{separator}")
    logger.info(f"🔥 YOLOv7 Fire/Smoke Detection Model Training - Starting")
    logger.info(f"{separator}")
    
    # Use table-style formatting with left-right alignment
    logger.info(f"┌─ 📁 Save Directory │ {save_dir}")
    logger.info(f"├─ 🏗️ Model Config   │ {model_cfg}")
    logger.info(f"├─ 📊 Dataset Config │ {dataset_cfg}")
    logger.info(f"├─ 🔄 Epochs         │ {epochs} epochs")
    logger.info(f"└─ 📦 Batch Size     │ {batch_size} images/batch")
    
    logger.info(f"{separator}")
    logger.info(f"🚀 Starting training... Please wait for model convergence")
    logger.info(f"{separator}\n")


def train(hyp, opt, device, tb_writer=None):
    """
    主训练函数
    
    参数:
        hyp: 超参数字典，包含学习率、权重衰减等训练参数
        opt: 命令行选项，包含模型配置、数据路径等
        device: 训练设备 (CPU/GPU)
        tb_writer: TensorBoard日志写入器
    """
    # 打印超参数信息
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    
    # 从选项中提取关键参数
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank, opt.freeze

    # 显示训练配置信息
    print_training_info(save_dir, opt.cfg, opt.data, epochs, batch_size)
    # 权重保存目录
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # 递归创建目录
    
    # 定义重要文件路径
    last = wdir / 'last.pt'          # 最新模型权重
    best = wdir / 'best.pt'          # 最佳模型权重  
    results_file = save_dir / 'results.txt'  # 训练结果记录文件

    # 保存运行设置到文件，便于后续复现实验
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)      # 保存超参数
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)  # 保存命令行选项

    # 基础配置
    plots = not opt.evolve          # 是否创建训练过程图表（超参数进化时不创建）
    cuda = False                    # 强制CPU模式，适配Apple Silicon Mac
    init_seeds(2 + rank)           # 初始化随机种子，确保结果可复现
    
    # 加载数据集配置
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # 数据集配置字典
    is_coco = opt.data.endswith('coco.yaml')  # 判断是否为COCO数据集

    # 日志记录配置 - 简化版本，不使用wandb
    if rank in [-1, 0]:  # 只在主进程中初始化日志
        logger.info("🚀 Starting training - Using TensorBoard for logging")

    # 数据集类别配置
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # 类别数量
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # 类别名称
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # 验证类别数量

    # 模型创建与加载
    pretrained = weights.endswith('.pt')  # 判断是否使用预训练权重
    if pretrained:
        # 分布式训练中只让主进程下载权重，避免重复下载
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # 如果本地不存在则下载权重文件
        
        # 加载检查点
        ckpt = torch.load(weights, map_location=device, weights_only=False)  # 加载权重文件
        
        # 检查预训练权重的类别数与数据集类别数是否匹配
        pretrained_nc = ckpt['model'].model[-1].nc if hasattr(ckpt['model'], 'model') else None
        if pretrained_nc is not None and pretrained_nc != nc:
            logger.info(f'⚠️  Pretrained weight classes ({pretrained_nc}) != Dataset classes ({nc})')
            logger.info(f'   Automatically adapting detection head to {nc} classes...')
            
            # 自动适配检测头
            model_to_adapt = ckpt['model']
            detection_layer = model_to_adapt.model[-1]
            
            # 保存原始设置
            anchors = detection_layer.anchors.clone()
            stride = detection_layer.stride.clone()
            
            # 更新类别数
            detection_layer.nc = nc
            detection_layer.no = nc + 5  # 输出数量 = 类别数 + 5 (x,y,w,h,conf)
            
            # 重新初始化输出层
            for i, conv in enumerate(detection_layer.m):
                old_weight = conv.weight.data.clone()
                old_bias = conv.bias.data.clone() if conv.bias is not None else None
                
                # 创建新的卷积层
                new_conv = nn.Conv2d(
                    in_channels=conv.in_channels,
                    out_channels=anchors.shape[1] * detection_layer.no,
                    kernel_size=conv.kernel_size,
                    stride=conv.stride,
                    padding=conv.padding,
                    bias=conv.bias is not None
                )
                
                # 智能权重复制
                with torch.no_grad():
                    anchor_outputs = anchors.shape[1]
                    old_no = old_weight.shape[0] // anchor_outputs
                    
                    for a in range(anchor_outputs):
                        old_start = a * old_no
                        new_start = a * detection_layer.no
                        
                        # 复制位置和置信度权重 (x,y,w,h,conf)
                        new_conv.weight.data[new_start:new_start+5] = old_weight[old_start:old_start+5]
                        if new_conv.bias is not None and old_bias is not None:
                            new_conv.bias.data[new_start:new_start+5] = old_bias[old_start:old_start+5]
                        
                        # 智能初始化类别权重
                        if pretrained_nc >= nc:
                            # 如果原模型类别数更多，复制前nc个类别的权重
                            new_conv.weight.data[new_start+5:new_start+5+nc] = old_weight[old_start+5:old_start+5+nc]
                            if new_conv.bias is not None and old_bias is not None:
                                new_conv.bias.data[new_start+5:new_start+5+nc] = old_bias[old_start+5:old_start+5+nc]
                        else:
                            # 如果原模型类别数更少，复制已有的并随机初始化新的
                            new_conv.weight.data[new_start+5:new_start+5+pretrained_nc] = old_weight[old_start+5:old_start+5+pretrained_nc]
                            if new_conv.bias is not None and old_bias is not None:
                                new_conv.bias.data[new_start+5:new_start+5+pretrained_nc] = old_bias[old_start+5:old_start+5+pretrained_nc]
                            # 随机初始化额外的类别
                            nn.init.normal_(new_conv.weight.data[new_start+5+pretrained_nc:new_start+5+nc], 0, 0.01)
                            if new_conv.bias is not None:
                                nn.init.constant_(new_conv.bias.data[new_start+5+pretrained_nc:new_start+5+nc], 0)
                
                detection_layer.m[i] = new_conv
            
            # 更新模型的类别名称
            model_to_adapt.names = names
            ckpt['model'] = model_to_adapt
            
            logger.info(f'✅ Detection head adapted: {pretrained_nc} classes → {nc} classes')
        
        # 创建模型：优先使用命令行指定的配置，否则使用权重文件中的配置
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
        
        # 确定需要排除的参数键（锚框相关）
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []
        
        # 加载预训练权重
        state_dict = ckpt['model'].float().state_dict()  # 转换为FP32精度
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # 获取交集
        model.load_state_dict(state_dict, strict=False)  # 加载权重（允许部分匹配）
        
        # 报告权重加载情况
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))
    else:
        # 从头训练：根据配置文件创建新模型
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    
    # 验证数据集配置
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # 检查数据集路径和配置
    train_path = data_dict['train']  # 训练集路径
    test_path = data_dict['val']     # 验证集路径

    # 冻结指定的模型层（用于迁移学习）
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # 构建要冻结的参数名前缀
    for k, v in model.named_parameters():
        v.requires_grad = True  # 默认所有层都参与训练
        if any(x in k for x in freeze):  # 如果参数名匹配冻结列表
            print('freezing %s' % k)    # 打印冻结的层名
            v.requires_grad = False     # 冻结该层，不参与梯度更新

    # 优化器配置
    nbs = 64  # 标准批次大小（用于缩放超参数）
    accumulate = max(round(nbs / total_batch_size), 1)  # 梯度累积步数
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # 根据实际批次大小缩放权重衰减
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # 优化器参数分组 - YOLOv7的重要特性
    # 不同类型的参数使用不同的优化策略
    pg0, pg1, pg2 = [], [], []  # 三个参数组
    
    for k, v in model.named_modules():
        # pg2: 偏置参数 - 不使用权重衰减
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        
        # pg0: BatchNorm权重 - 不使用权重衰减
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)
        # pg1: 卷积层权重 - 使用权重衰减
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
        
        # 以下是YOLOv7特有的隐式学习参数，都加入pg0（不使用权重衰减）
        
        # 隐式乘法参数 (Implicit Multiplication)
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):           
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        
        # 隐式通道参数 (Implicit Channel)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):           
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        
        # 隐式批处理参数 (Implicit Batch)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):           
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        
        # 隐式输出参数 (Implicit Output)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):           
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        
        # 隐式注意力参数 (Implicit Attention)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):           
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        
        # 注意力机制相关参数
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):   
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):   
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):  
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):  
                pg0.append(v.attn.relative_position_bias_table)
        
        # RepVGG结构相关参数 (Re-parameterization)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):   
                pg0.append(v.rbr_dense.vector)

    # 创建优化器
    if opt.adam:
        # Adam优化器：调整beta1为momentum值
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        # SGD优化器：使用Nesterov动量
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    # 添加其他参数组
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # 卷积权重组（应用权重衰减）
    optimizer.add_param_group({'params': pg2})  # 偏置组（不应用权重衰减）
    
    # 打印优化器参数组信息
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2  # 释放内存

    # 学习率调度器
    # 参考: https://arxiv.org/pdf/1812.01187.pdf
    # 参考: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        # 线性学习率衰减
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']
    else:
        # 余弦学习率衰减（One Cycle策略）
        lf = one_cycle(1, hyp['lrf'], epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 指数移动平均 (EMA) - 用于模型权重的平滑更新
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # 恢复训练设置
    start_epoch, best_fitness = 0, 0.0  # 起始轮数和最佳适应度
    if pretrained:
        # 恢复优化器状态
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode (Disabled for CPU-only)
    # if cuda and rank == -1 and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    # SyncBatchNorm (Disabled for CPU-only)
    # if opt.sync_bn and cuda and rank != -1:
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    #     logger.info('Using SyncBatchNorm()')

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                #plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # 分布式数据并行模式 (已为CPU训练禁用)
    # DDP允许在多个GPU上并行训练，但在CPU模式下不需要
    # if cuda and rank != -1:
    #     model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
    #                 # nn.MultiheadAttention与DDP不兼容的解决方案
    #                 find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # 模型参数缩放 - 根据检测层数量和图像尺寸调整损失权重
    hyp['box'] *= 3. / nl  # 边界框损失缩放到检测层数
    hyp['cls'] *= nc / 80. * 3. / nl  # 分类损失缩放到类别数和检测层数 (基准80类COCO)
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # 目标性损失缩放到图像尺寸和检测层数 (基准640像素)
    hyp['label_smoothing'] = opt.label_smoothing  # 标签平滑参数
    
    # 将关键信息附加到模型对象
    model.nc = nc  # 类别数量
    model.hyp = hyp  # 超参数
    model.gr = 1.0  # IoU损失比率 (obj_loss = 1.0 或 iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # 类别权重（处理不平衡数据集）
    model.names = names  # 类别名称

    # 开始训练
    t0 = time.time()  # 记录训练开始时间
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # 预热迭代次数，最少1000次，最多3个epoch
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # 限制预热不超过训练的一半
    
    # 初始化训练指标
    maps = np.zeros(nc)  # 每个类别的mAP
    results = (0, 0, 0, 0, 0, 0, 0)  # 训练结果: P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # 设置学习率调度器的起始epoch
    
    # 自动混合精度训练 (已为CPU训练禁用)
    # 在CPU上不需要AMP，因此设置为None
    scaler = None  # CPU训练不需要梯度缩放器
    
    # 初始化损失计算器
    compute_loss_ota = ComputeLossOTA(model)  # OTA (Optimal Transport Assignment) 损失
    compute_loss = ComputeLoss(model)  # 标准损失
    
    # 打印训练信息
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    torch.save(model, wdir / 'init.pt')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

    torch.save(model, wdir / 'init.pt')  # 保存初始模型
    
    # 主训练循环 - 逐epoch训练
    for epoch in range(start_epoch, epochs):  # epoch循环 ----------------------------------------
        model.train()  # 设置模型为训练模式

        # 更新图像权重（可选）- 用于处理数据不平衡
        if opt.image_weights:
            # 生成权重索引
            if rank in [-1, 0]:  # 只在主进程中生成
                # 基于类别权重和当前mAP计算图像权重
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # 类别权重
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # 图像权重
                # 根据权重随机选择图像索引
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)
            
            # 分布式训练时广播索引
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)  # 从主进程广播到所有进程
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # 更新马赛克增强边界（已注释）
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # 高度和宽度边界

        # 初始化损失累积器
        mloss = torch.zeros(4, device=device)  # 平均损失 [box, obj, cls, total]
        
        # 分布式训练设置
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)  # 设置采样器的epoch
        
        # 设置进度条和表头显示 - 优化版本，更清晰的排版
        pbar = enumerate(dataloader)
        
        # 打印美观的训练表头
        header_separator = "─" * 100
        logger.info(f"\n{header_separator}")
        logger.info(f"📊 Epoch {epoch+1:3d}/{epochs} Training Details")
        logger.info(f"{header_separator}")
        
        # Optimized headers with clearer column names and alignment
        header_format = '%12s' * 8
        logger.info(header_format % (
            'Epoch', 'Memory', 'Box_Loss', 'Obj_Loss', 'Cls_Loss', 'Total_Loss', 'Targets', 'Img_Size'
        ))
        logger.info(header_format % (
            'Round', 'Memory', 'Box_Loss', 'Obj_Loss', 'Cls_Loss', 'Total_Loss', 'Targets', 'Img_Size'
        ))
        logger.info(f"{header_separator}")
        
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb, 
                       desc=f'Epoch {epoch+1}/{epochs}',  # Clearer progress bar description
                       bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')  # Custom progress bar format
        
        optimizer.zero_grad()  # Clear gradients
        
        # 批次训练循环
        for i, (imgs, targets, paths, _) in pbar:  # 批次循环 --------------------------------
            ni = i + nb * epoch  # 累积批次数（从训练开始计算）
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # 转换数据类型和归一化

            # 预热阶段 - 前几个epoch使用较小的学习率逐渐升温
            if ni <= nw:
                xi = [0, nw]  # 插值范围
                # 动态调整梯度累积步数
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                
                # 为不同参数组设置不同的学习率预热策略
                for j, x in enumerate(optimizer.param_groups):
                    # 偏置参数学习率从0.1降到lr0，其他参数从0.0升到lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        # 动量参数也进行预热
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # 多尺度训练 - 随机改变输入图像尺寸增强模型鲁棒性
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # 随机尺寸
                sf = sz / max(imgs.shape[2:])  # 缩放因子
                if sf != 1:
                    # 计算新的形状（确保是grid size的倍数）
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    # 双线性插值调整图像尺寸
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # 前向传播 - CPU训练不使用autocast
            pred = model(imgs)  # 模型前向传播获得预测结果
            
            # 损失计算 - 选择OTA损失或标准损失
            if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                # OTA (Optimal Transport Assignment) 损失 - YOLOv7的改进
                loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)
            else:
                # 标准YOLO损失
                loss, loss_items = compute_loss(pred, targets.to(device))
            
            # 分布式训练损失处理
            if rank != -1:
                loss *= opt.world_size  # 在DDP模式下平均梯度
            if opt.quad:
                loss *= 4.  # 四倍数据加载器模式

            # 反向传播 - CPU训练不使用AMP
            loss.backward()  # 计算梯度

            # 优化器更新
            if ni % accumulate == 0:  # 梯度累积达到设定步数时更新
                optimizer.step()  # 更新模型参数
                optimizer.zero_grad()  # 清零梯度
                if ema:
                    ema.update(model)  # 更新EMA模型

            # 打印训练信息 - 优化版本，更清晰的格式化输出
            if rank in [-1, 0]:  # 只在主进程中打印
                # 更新平均损失
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = '%.1fMB' % (0)  # CPU模式 - 显示内存使用情况（这里显示为0，实际可以获取CPU内存使用）
                
                # 优化的格式化显示信息，使用更好的对齐和数值格式
                # 格式：轮次, 内存, box损失, obj损失, cls损失, 总损失, 标签数, 图像尺寸
                display_format = '%12s' * 2 + '%12.4f' * 4 + '%12d' * 2
                s = display_format % (
                    f'{epoch+1}/{epochs}',          # 当前轮次/总轮次
                    mem,                            # 内存使用
                    mloss[0],                       # 边界框损失
                    mloss[1],                       # 目标性损失  
                    mloss[2],                       # 分类损失
                    mloss[3],                       # 总损失
                    targets.shape[0],               # 当前批次的目标数量
                    imgs.shape[-1]                  # 图像尺寸
                )
                
                # 每10个批次打印一次详细信息到日志
                if i % 10 == 0:
                    logger.info(s)
                
                # 更新进度条描述，显示关键损失信息
                pbar.set_description(
                    f'Epoch {epoch+1}/{epochs} | '
                    f'Loss: {mloss[3]:.4f} | '
                    f'Box: {mloss[0]:.4f} | '
                    f'Obj: {mloss[1]:.4f} | '
                    f'Cls: {mloss[2]:.4f}'
                )

                # 绘制训练图像（前10个批次）
                if plots and ni < 10:
                    f = save_dir / f'train_batch{ni}.jpg'  # 文件名
                    # 在后台线程中绘制图像，避免阻塞训练
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    # TensorBoard可视化（已注释）
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])

            # end batch ------------------------------------------------------------------------------------------------
        
        # Epoch 结束总结 - 添加每个epoch的总结信息
        if rank in [-1, 0]:
            epoch_separator = "┄" * 60
            logger.info(f"\n{epoch_separator}")
            logger.info(f"📈 Epoch {epoch+1}/{epochs} Completed | Average Loss: {mloss[3]:.4f}")
            logger.info(f"   Box: {mloss[0]:.4f} | Obj: {mloss[1]:.4f} | Cls: {mloss[2]:.4f}")
            logger.info(f"{epoch_separator}")
        
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP计算
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # 计算mAP
                try:
                    results, maps, times = test.test(data_dict,
                                                     batch_size=batch_size * 2,
                                                     imgsz=imgsz_test,
                                                     model=ema.ema,
                                                     single_cls=opt.single_cls,
                                                     dataloader=testloader,
                                                     save_dir=save_dir,
                                                     verbose=nc < 50 and final_epoch,
                                                     plots=plots and final_epoch,
                                                     compute_loss=compute_loss,
                                                     v5_metric=opt.v5_metric)
                    
                    # Optimize validation result display - beautiful formatted output
                    val_separator = "┈" * 80
                    logger.info(f"\n{val_separator}")
                    logger.info(f"🔍 Epoch {epoch+1} Validation Results")
                    logger.info(f"{val_separator}")
                    logger.info(f"┌─ 📊 Accuracy Metrics")
                    logger.info(f"├─ 🎯 Precision    │ {results[0]:.4f}")
                    logger.info(f"├─ 🔍 Recall       │ {results[1]:.4f}")
                    logger.info(f"├─ 📈 mAP@0.5      │ {results[2]:.4f}")
                    logger.info(f"├─ 📊 mAP@0.5:0.95 │ {results[3]:.4f}")
                    logger.info(f"└─ ⚡ Inference Speed │ {times[1]:.1f}ms")
                    
                    logger.info(f"┌─ 📉 Validation Loss")
                    logger.info(f"├─ 📦 Box Loss     │ {results[4]:.4f}")
                    logger.info(f"├─ 🎯 Obj Loss     │ {results[5]:.4f}")
                    logger.info(f"└─ 🏷️  Cls Loss     │ {results[6]:.4f}")
                    logger.info(f"{val_separator}")
                    
                except Exception as e:
                    logger.warning(f"⚠️  Error occurred during testing phase, continuing training with default values: {e}")
                    # Use default values to continue training
                    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
                    maps = np.zeros(nc)  # Set mAP for each class to 0
                    times = (0, 0, 0)  # Set time statistics to 0

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # 日志记录
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard

            # 更新最佳mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi

            # 保存模型
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict()}

                # Save last, best and remove old files - optimize save info display
                torch.save(ckpt, last)
                
                # Check if it's the best model and provide clear save info
                if best_fitness == fi:
                    torch.save(ckpt, best)
                    logger.info(f"🏆 Found better model! fitness: {fi.item():.4f} -> Saved to {best}")
                    
                if (best_fitness == fi) and (epoch >= 200):
                    milestone_path = wdir / 'best_{:03d}.pt'.format(epoch)
                    torch.save(ckpt, milestone_path)
                    logger.info(f"📍 Milestone model saved: {milestone_path}")
                
                # Save weight files for each epoch using a more reasonable strategy with clear logs
                saved_checkpoint = False
                if epoch == 0:
                    # Always save the first epoch
                    checkpoint_path = wdir / 'epoch_{:03d}.pt'.format(epoch)
                    torch.save(ckpt, checkpoint_path)
                    saved_checkpoint = True
                    logger.info(f"💾 Initial epoch checkpoint: {checkpoint_path}")
                elif epochs <= 10:
                    # If total epochs ≤ 10, save every epoch
                    checkpoint_path = wdir / 'epoch_{:03d}.pt'.format(epoch)
                    torch.save(ckpt, checkpoint_path)
                    saved_checkpoint = True
                elif epochs <= 50:
                    # If total epochs ≤ 50, save every 5 epochs and last 5 epochs
                    if ((epoch+1) % 5) == 0 or epoch >= (epochs-5):
                        checkpoint_path = wdir / 'epoch_{:03d}.pt'.format(epoch)
                        torch.save(ckpt, checkpoint_path)
                        saved_checkpoint = True
                else:
                    # If total epochs > 50, save every 25 epochs and last 5 epochs
                    if ((epoch+1) % 25) == 0 or epoch >= (epochs-5):
                        checkpoint_path = wdir / 'epoch_{:03d}.pt'.format(epoch)
                        torch.save(ckpt, checkpoint_path)
                        saved_checkpoint = True
                
                # Display checkpoint save info
                if saved_checkpoint:
                    logger.info(f"💾 Epoch {epoch+1} checkpoint saved: {checkpoint_path}")
                
                # Display regular save info
                logger.info(f"💾 Latest model saved: {last}")
                
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # 绘制结果图表
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
        
        # Test best model - optimize training completion info display
        training_time = (time.time() - t0) / 3600
        
        # Beautiful training completion info display
        completion_separator = "━" * 80
        logger.info(f"\n{completion_separator}")
        logger.info(f"🎉 Training completed! Congratulations on successfully training YOLOv7 Fire/Smoke Detection Model")
        logger.info(f"{completion_separator}")
        
        # Training summary information with table-style formatting
        logger.info(f"┌─ 📊 Training Summary")
        logger.info(f"├─ ⏱️  Total Time     │ {training_time:.2f} hours")
        logger.info(f"├─ 🔄 Epochs Completed│ {epoch - start_epoch + 1} epochs")
        logger.info(f"├─ 🎯 Best Fitness    │ {best_fitness:.4f}")
        logger.info(f"├─ 💾 Model Save Path │ {save_dir}")
        logger.info(f"├─ 🏆 Best Weights    │ {best}")
        logger.info(f"└─ 📈 Latest Weights  │ {last}")
        logger.info(f"{completion_separator}")
        
        # Friendly next step suggestions
        logger.info(f"🚀 Next Step Recommendations:")
        logger.info(f"   1. Use test.py to evaluate model performance")
        logger.info(f"   2. Use detect.py for inference detection")
        logger.info(f"   3. Check {save_dir}/results.png to analyze training curves")
        logger.info(f"{completion_separator}\n")
        
        # 生成训练报告
        create_training_report(save_dir, opt, hyp, results, best_fitness, training_time)
        
        if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
            for m in (last, best) if best.exists() else (last):  # speed, mAP tests
                results, _, _ = test.test(opt.data,
                                          batch_size=batch_size * 2,
                                          imgsz=imgsz_test,
                                          conf_thres=0.001,
                                          iou_thres=0.7,
                                          model=attempt_load(m, device).half(),
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          save_dir=save_dir,
                                          save_json=True,
                                          plots=False,
                                          v5_metric=opt.v5_metric)

        # 优化器剥离
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
    else:
        # Distributed training cleanup
        dist.destroy_process_group()
    
    # Clean up resources and free memory
    if 'dataloader' in locals():
        del dataloader
    if 'testloader' in locals():
        del testloader
    # torch.cuda.empty_cache()  # Disabled for CPU training
    return results


if __name__ == '__main__':
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='YOLOv7 Training Script - Optimized for Apple Silicon Mac and CPU training')
    
    # Model and data related parameters
    parser.add_argument('--weights', type=str, default='', help='Pretrained weights path')
    parser.add_argument('--cfg', type=str, default='cfg/training/yolov7.yaml', help='Model configuration file path')
    parser.add_argument('--data', type=str, default='datasets/smokefire.yaml', help='Dataset configuration file path')
    parser.add_argument('--hyp', type=str, default='hyperparameters/hyp.train.yaml', help='Hyperparameters configuration file path')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=150, help='Training epochs (recommend smaller values for CPU training)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size (recommend smaller values for CPU training)')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='Rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='Resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='Only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='Only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='Disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='Evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='Google Cloud Storage bucket')
    parser.add_argument('--cache-images', action='store_true', help='Cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='Use weighted image selection for training')
    parser.add_argument('--device', default='cpu', help='Training device, using CPU')
    parser.add_argument('--multi-scale', action='store_true', help='Multi-scale training: vary img-size +/- 50%')
    parser.add_argument('--single-cls', action='store_true', help='Train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='Use Adam optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='Use SyncBatchNorm (only available in DDP mode)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=1, help='Maximum dataloader workers (recommend 1 for CPU training)')
    
    # Output and logging parameters
    parser.add_argument('--project', default='runs/train', help='Training results save root directory')
    parser.add_argument('--name', default='', help='Experiment name suffix (deprecated, now uses auto-naming)')
    parser.add_argument('--exist-ok', action='store_true', help='Allow overwriting existing training directory')
    parser.add_argument('--quad', action='store_true', help='Quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='Linear learning rate')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    
    # Model related parameters
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=50, first 3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='Assume maximum recall as 1.0 in AP calculation')
    
    # Parse command line arguments
    opt = parser.parse_args()

    # 设置分布式训练变量
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)  # 设置日志记录
    # 可选的Git状态检查和依赖检查（已注释）
    #if opt.global_rank in [-1, 0]:
    #    check_git_status()
    #    check_requirements()

    # 恢复训练
    if opt.resume and not isinstance(opt.resume, str):  # resume an interrupted run
        ckpt = get_latest_run()  # 获取最新的训练运行
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        
        # Generate clearer save directory names
        if opt.evolve:
            opt.save_dir = increment_path(Path(opt.project) / 'evolve', exist_ok=opt.exist_ok)
        else:
            # Use unified experiment directory management
            import __main__
            script_path = __main__.__file__ if hasattr(__main__, '__file__') else 'train.py'
            from utils.experiment_manager import setup_training_directory
            opt.save_dir = setup_training_directory(opt, script_path)

    # DDP mode (Disabled for CPU-only)
    opt.total_batch_size = opt.batch_size
    device = select_device('cpu', batch_size=opt.batch_size)  # Force CPU device
    # if opt.local_rank != -1:
    #     assert torch.cuda.device_count() > opt.local_rank
    #     torch.cuda.set_device(opt.local_rank)
    #     device = torch.device('cuda', opt.local_rank)
    #     dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    #     assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
    #     opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),   # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0),  # segment copy-paste (probability)
                'paste_in': (1, 0.0, 1.0)}    # segment copy-paste (probability)
        
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
                
        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Limit hyperparameters to reasonable ranges
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train with mutated hyperparameters
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot evolution results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
    
    # Final cleanup
    import gc
    gc.collect()  # 强制垃圾回收，释放内存

# 程序结束


