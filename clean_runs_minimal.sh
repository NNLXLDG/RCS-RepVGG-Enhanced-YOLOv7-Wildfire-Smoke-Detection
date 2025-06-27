

echo "正在精简runs文件夹，保留关键训练结果..."

cd runs/train

# 保留有完整训练权重的实验（yolov714）
# 重命名为更有意义的名称
if [ -d "yolov714" ]; then
    if [ ! -d "yolov7-baseline" ]; then
        mv yolov714 yolov7-baseline
        echo "已将 yolov714 重命名为 yolov7-baseline"
    fi
fi

# 删除所有只有init.pt的未完成实验
echo "删除未完成的训练实验..."
for dir in yolov7*; do
    if [ -d "$dir" ] && [ "$dir" != "yolov7-baseline" ]; then
        # 检查是否只有init.pt文件
        if [ -f "$dir/weights/init.pt" ] && [ ! -f "$dir/weights/best.pt" ] && [ ! -f "$dir/weights/last.pt" ]; then
            rm -rf "$dir"
            echo "已删除未完成的实验: $dir"
        fi
    fi
done

cd ../..

# 创建四个版本对应的文件夹结构（为未来训练做准备）
mkdir -p runs/train/yolov7-baseline
mkdir -p runs/train/yolov7-repvgg  
mkdir -p runs/train/yolov7-rcsosa
mkdir -p runs/train/yolov7-repvgg-rcsosa

echo ""
echo "runs文件夹精简完成！"
echo ""
echo "当前保留的训练结果："
ls -la runs/train/
echo ""
echo "准备好进行四版本对比训练。使用以下命令："
echo "1. python train.py --name yolov7-baseline"
echo "2. python train-repvgg.py --name yolov7-repvgg"
echo "3. python train-rcsosa.py --name yolov7-rcsosa"
echo "4. python train-repvgg-rcsosa.py --name yolov7-repvgg-rcsosa"
