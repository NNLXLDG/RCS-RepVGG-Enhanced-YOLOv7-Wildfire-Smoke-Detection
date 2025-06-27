import os
import yaml
from pathlib import Path

def check_file_exists(file_path, description=""):
    """检查文件是否存在"""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (不存在)")
        return False

def check_dataset_config():
    """检查数据集配置"""
    print("\n📁 检查数据集配置...")
    
    # 检查数据集文件夹
    dataset_dir = "datasets"
    if not os.path.exists(dataset_dir):
        print(f"❌ 数据集文件夹不存在: {dataset_dir}")
        return False
    
    # 检查配置文件
    config_file = f"{dataset_dir}/smokefire.yaml"
    if not check_file_exists(config_file, "数据集配置文件"):
        return False
    
    # 读取配置文件
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"   类别数量: {config['nc']}")
        print(f"   类别名称: {config['names']}")
        print(f"   训练集路径: {config['train']}")
        print(f"   验证集路径: {config['val']}")
        
        # 检查训练和验证数据路径
        train_path = config['train'].replace('datasets/', f'{dataset_dir}/')
        val_path = config['val'].replace('datasets/', f'{dataset_dir}/')
        
        check_file_exists(train_path, "训练集目录")
        check_file_exists(val_path, "验证集目录")
        
        return True
        
    except Exception as e:
        print(f"❌ 读取数据集配置文件失败: {e}")
        return False

def check_model_configs():
    """检查模型配置文件"""
    print("\n🔧 检查模型配置文件...")
    
    configs = [
        "cfg/training/yolov7.yaml",
        "cfg/training/yolov7-repvgg.yaml", 
        "cfg/training/yolov7-rcsosa.yaml",
        "cfg/training/yolov7-repvgg-rcsosa.yaml"
    ]
    
    all_exist = True
    for config in configs:
        if not check_file_exists(config, "模型配置"):
            all_exist = False
    
    return all_exist

def check_hyperparameters():
    """检查超参数配置"""
    print("\n⚙️ 检查超参数配置...")
    
    hyp_file = "hyperparameters/hyp.scratch.p5.yaml"
    return check_file_exists(hyp_file, "超参数配置文件")

def check_training_scripts():
    """检查训练脚本"""
    print("\n🚀 检查训练脚本...")
    
    scripts = [
        "train.py",
        "train-repvgg.py",
        "train-rcsosa.py", 
        "train-repvgg-rcsosa.py"
    ]
    
    all_exist = True
    for script in scripts:
        if not check_file_exists(script, "训练脚本"):
            all_exist = False
    
    return all_exist

def check_detection_scripts():
    """检查检测脚本"""
    print("\n🔍 检查检测脚本...")
    
    scripts = [
        "detect.py",
        "detect-repvgg.py",
        "detect-rcsosa.py",
        "detect-repvgg-rcsosa.py"
    ]
    
    all_exist = True
    for script in scripts:
        if not check_file_exists(script, "检测脚本"):
            all_exist = False
    
    return all_exist

def check_test_script():
    """检查测试脚本"""
    print("\n✅ 检查测试/验证脚本...")
    
    return check_file_exists("test.py", "测试脚本")

def check_coco_logic():
    """检查是否已清理COCO相关代码"""
    print("\n🧹 检查COCO代码清理情况...")
    
    # 检查test.py中是否还有COCO相关代码
    try:
        with open("test.py", 'r') as f:
            content = f.read().lower()
        
        coco_references = content.count('coco')
        if coco_references == 0:
            print("✅ COCO相关代码已完全清理")
            return True
        else:
            print(f"❌ 仍有 {coco_references} 处COCO相关代码")
            return False
            
    except Exception as e:
        print(f"❌ 检查test.py文件失败: {e}")
        return False

def main():
    """主验证函数"""
    print("🔥 YOLOv7 野火烟雾检测项目验证")
    print("=" * 50)
    
    checks = [
        check_dataset_config,
        check_model_configs,
        check_hyperparameters,
        check_training_scripts,
        check_detection_scripts,
        check_test_script,
        check_coco_logic
    ]
    
    results = []
    for check in checks:
        results.append(check())
    
    print("\n" + "=" * 50)
    print("📊 验证结果汇总:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 所有检查通过! ({passed}/{total})")
        print("项目配置完整，可以开始训练!")
    else:
        print(f"⚠️  有 {total - passed} 项检查失败 ({passed}/{total})")
        print("请修复上述问题后再开始训练")
    
    print("\n💡 快速开始命令:")
    print("python train.py --data datasets/smokefire.yaml --cfg cfg/training/yolov7.yaml --epochs 50 --batch-size 4 --device cpu --workers 1")

if __name__ == "__main__":
    main()
