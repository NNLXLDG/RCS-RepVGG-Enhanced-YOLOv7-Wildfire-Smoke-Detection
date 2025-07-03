# 此版本为组员陈颢元提供的测试版本

**其中包括：**

1.新版9600张完整烟火数据集

2.inference **（未改动）**

3.yolov7-tiny的架构文件和其用于消融实验的优化的三版架构文件

4.**依据原版设计的新的V4改消融实验yaml文件**

4.**原版训练文件和测试文件**

5.旧的训练文件和推理文件已经放入previous_ver文件夹中

其中不包括：

1.旧版的所有数据集

# 指令

接下来是rcs-yolo原文章的训练和测试指令，使用时请记得替换 **对应的--cfg, --name, --weights**

这里我已经对数据路径进行替换，

# 如何使用指令？

IDE打开文件夹-打开终端-粘贴你编辑好的指令-回车  即可运行

# 如果找不到路径：

1.侧边栏找到对应文件复制相对路径

2.粘贴到原路径处

# 遇到了严重故障和报错，无法运行，以及对更新内容有疑问

如果chy不在睡觉，请尽情骚扰他

# 训练和测试指令前面的'python'一定一定一定要换成你对应的环境！

e.g.  **C:/Users/Lenovo/anaconda3/envs/yolov8/python.exe**

#### Training

The hyperparameter setting file is hyp_training.yaml in the directory [./data/](https://github.com/mkang315/RCS-YOLO/tree/main/data).

###### Single GPU training

```
python train.py --workers 8 --device 0 --batch-size 32 --data datasets_smokefire\data.yaml --img 640 640 --cfg cfg/training/rcs-yolo.yaml --weights '' --name rcs-yolo --hyp data/hyp_training.yaml
```

###### Multiple GPU training

```
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data datasets_smokefire\data.yaml --img 640 640 --cfg cfg/training/rcs-yolo.yaml --weights '' --name rcs-yolo --hyp data/hyp_training.yaml
```

#### Testing

The model weights we pretrained on the brain tumor detection was saved as best.pt in the directory [./runs/train/exp/weights/](https://github.com/mkang315/RCS-YOLO/tree/main/runs/train).

```
python test.py --data datasets_smokefire\data.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights runs/train/exp/weights/best.pt --name val
```
