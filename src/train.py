"""
天气图像分类 —— 主训练脚本
=========================================
技术路线：ResNet18/34 迁移学习（ImageNet 预训练 + 替换分类头）

数据集目录结构（ImageFolder 自动按子文件夹名作为类别标签）：
    weather/
    ├── train/
    │   ├── cloudy/   ├── foggy/    ├── lightning/ ├── rainbow/
    │   ├── rainy/    ├── rime/     ├── sandstorm/ └── sunrise/
    ├── val/         （同上 8 个类别子文件夹）
    └── test/        （同上 8 个类别子文件夹）

数据获取（二选一）：
    1. 运行 ablation_experiment.py 开头的 kagglehub 下载+划分代码，
       会自动生成 ./weather/{train,val,test}；
    2. 自行从 Kaggle 下载 tamimresearch/weather-image-dataset 后按 8:1:1 划分。

用法：
    python train.py                       # 使用下方默认配置训练
    python train.py --model resnet34      # 换 ResNet34
"""
import argparse
import copy
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# 项目根目录（src/ 的上一级），数据与输出都相对它定位，
# 因此无论从哪个目录运行 `python src/train.py` 路径都正确
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ====================== 1. 超参数配置区 ======================
DATA_DIR = str(PROJECT_ROOT / "weather")  # 数据根目录（train/val/test 的上一级）
NUM_CLASSES = 8                 # 天气类别数
BATCH_SIZE = 32
NUM_EPOCHS = 15                 # 先跑 15 轮观察效果
LEARNING_RATE = 1e-3
# Windows 下 DataLoader 多进程需要 if __name__ == "__main__" 保护，
# 为避免新手踩坑这里直接设 0（数据量不大，速度差异不明显）；Linux 可调 4
NUM_WORKERS = 0


def build_transforms():
    """训练集做数据增强；验证/测试集只做确定性预处理。

    数据增强动机（结合天气场景）：
      - 随机水平翻转 / 旋转：提升姿态鲁棒性；
      - ColorJitter：模拟不同光照、雾霾、大气散射条件，
        与真实户外天气成像的光照波动相吻合。
    ImageNet 均值方差标准化：复用预训练权重时必须与预训练分布对齐。
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    return train_transforms, val_test_transforms


def create_model(model_name="resnet18", num_classes=8):
    """加载 ImageNet 预训练 ResNet 并替换最后的全连接分类头。"""
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("目前只支持 resnet18 / resnet34")

    # 原 fc 层输出 1000 类，替换为天气类别数
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_model(model, dataloaders, dataset_sizes, device,
                criterion, optimizer, scheduler, num_epochs=15):
    """标准训练循环：每个 epoch 先 train 后 val，保留验证集最优权重。"""
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}\n" + "-" * 20)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss, running_corrects = 0.0, 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # train 阶段才开梯度和反向传播
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.cpu().numpy())
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 只在验证集刷新最优时保存权重
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    print(f"训练完成，用时 {time.time() - since:.0f}s，最佳 Val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model, history


def plot_history(history):
    """绘制并保存 Loss / Accuracy 训练曲线。"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    curve_path = PROJECT_ROOT / "results" / "training_curve.png"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(curve_path, dpi=200, bbox_inches="tight")
    print(f"训练曲线已保存: {curve_path}")


def main():
    parser = argparse.ArgumentParser(description="天气图像分类训练")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf, val_tf = build_transforms()
    image_datasets = {
        "train": datasets.ImageFolder(os.path.join(args.data_dir, "train"), train_tf),
        "val": datasets.ImageFolder(os.path.join(args.data_dir, "val"), val_tf),
        "test": datasets.ImageFolder(os.path.join(args.data_dir, "test"), val_tf),
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                      shuffle=(x == "train"), num_workers=NUM_WORKERS)
        for x in ["train", "val", "test"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    class_names = image_datasets["train"].classes
    print("类别:", class_names, "| 数据量:", dataset_sizes)

    model = create_model(args.model, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 每 7 个 epoch 学习率衰减为 1/10
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model, history = train_model(
        model, dataloaders, dataset_sizes, device,
        criterion, optimizer, scheduler, num_epochs=args.epochs
    )

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / "best_weather_resnet18.pth"
    torch.save(model.state_dict(), save_path)
    print(f"最优权重已保存: {save_path}")
    plot_history(history)


if __name__ == "__main__":
    # Windows 下 DataLoader / matplotlib 多进程安全所必需
    main()
