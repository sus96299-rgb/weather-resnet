#%%
import kagglehub

path = kagglehub.dataset_download("tamimresearch/weather-image-dataset")
print('路径', path)

#%%
import os
import random
import numpy as np
import torch

def random_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

random_everything(42)

#%%
import shutil
from pathlib import Path

old_dir = Path(path)
new_dir = Path('./weather')

cata = [c for c in old_dir.iterdir() if c.is_dir()]

for c in cata:
    c_name = c.name
    images = [i for i in c.glob('*') if i.is_file()]
    random.shuffle(images)
    images_num = len(images)
    train_num = int(images_num * 0.8)
    val_num = train_num + int(images_num * 0.1)
    train_images = images[:train_num]
    val_images = images[train_num:val_num]
    test_images = images[val_num:]

    for name, image in [("train", train_images), ("val", val_images), ("test", test_images)]:
        cata_name = new_dir / name / c_name
        cata_name.mkdir(parents=True, exist_ok=True)
        for img in image:
            shutil.copy(img, cata_name / img.name)
    print(c.name, len(images), len(train_images), len(val_images), len(test_images))

#%%
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 20
num_workers = 0
num_classes = 8  # 天气类别数，从数据集自动获取更稳妥

# 有数据增强的训练transform
tra_rule_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 无数据增强的训练transform（对照实验用）
tra_rule_noaug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_rule = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 主实验数据集（有增强）
tra_dataset = datasets.ImageFolder(root=new_dir / "train", transform=tra_rule_aug)
val_dataset = datasets.ImageFolder(root=new_dir / "val", transform=val_rule)
test_dataset = datasets.ImageFolder(root=new_dir / "test", transform=val_rule)

# 对照实验：无增强训练集
tra_dataset_noaug = datasets.ImageFolder(root=new_dir / "train", transform=tra_rule_noaug)

tra_loader = DataLoader(tra_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
tra_loader_noaug = DataLoader(tra_dataset_noaug, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

class_names = tra_dataset.classes
print("类别:", class_names)
print("训练集:", len(tra_dataset), "验证集:", len(val_dataset), "测试集:", len(test_dataset))

#%%
def train_ing(model, dataloader, criterion, optimizer, device):
    model.train()
    total, correct, run_loss = 0, 0, 0.0
    for image, label in dataloader:
        image, label = image.to(device), label.to(device)
        output = model(image)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += label.size(0)
        run_loss += loss.item() * label.size(0)
        _, pred = torch.max(output, 1)
        correct += (pred == label).sum().item()
    return run_loss / total, correct / total

def val_ing(model, dataloader, criterion, device):
    model.eval()
    total, correct, run_loss = 0, 0, 0.0
    with torch.no_grad():
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = criterion(output, label)
            total += label.size(0)
            run_loss += loss.item() * label.size(0)
            _, pred = torch.max(output, 1)
            correct += (pred == label).sum().item()
    return run_loss / total, correct / total

def build_model(model_name, num_classes):
    """构建模型：resnet18或resnet34，替换分类头"""
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
    nn.init.constant_(model.fc.bias, 0)
    return model.to(device)

def train_two_stage(model, train_loader, val_loader, freeze_epochs=5, finetune_epochs=5, save_prefix='exp'):
    """两阶段训练：冻结backbone训fc → 解冻layer4微调，返回训练历史"""
    criterion = nn.CrossEntropyLoss()
    all_train_loss, all_train_acc = [], []
    all_val_loss, all_val_acc = [], []
    best_val_acc = 0.0

    # 阶段1：冻结，只训fc
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)

    for epoch in range(freeze_epochs):
        tr_l, tr_a = train_ing(model, train_loader, criterion, optimizer, device)
        va_l, va_a = val_ing(model, val_loader, criterion, device)
        all_train_loss.append(tr_l); all_train_acc.append(tr_a)
        all_val_loss.append(va_l); all_val_acc.append(va_a)
        if va_a > best_val_acc:
            best_val_acc = va_a
            torch.save(model.state_dict(), f'{save_prefix}_freeze.pth')
        print(f'[冻结 {epoch+1}/{freeze_epochs}] train_loss:{tr_l:.4f} train_acc:{tr_a:.4f} val_acc:{va_a:.4f}')

    # 阶段2：解冻layer4+fc微调
    model.load_state_dict(torch.load(f'{save_prefix}_freeze.pth', map_location=device, weights_only=True))
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    optimizer_final = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer_final, step_size=5, gamma=0.5)

    for epoch in range(finetune_epochs):
        tr_l, tr_a = train_ing(model, train_loader, criterion, optimizer_final, device)
        va_l, va_a = val_ing(model, val_loader, criterion, device)
        scheduler.step()
        all_train_loss.append(tr_l); all_train_acc.append(tr_a)
        all_val_loss.append(va_l); all_val_acc.append(va_a)
        if va_a > best_val_acc:
            best_val_acc = va_a
            torch.save(model.state_dict(), f'{save_prefix}_finetune.pth')
        print(f'[微调 {epoch+1}/{finetune_epochs}] train_loss:{tr_l:.4f} train_acc:{tr_a:.4f} val_acc:{va_a:.4f}')

    return all_train_loss, all_train_acc, all_val_loss, all_val_acc, best_val_acc

#%%
# ==================================================
# 主实验：ResNet18 + 有数据增强
# ==================================================
print("=" * 60)
print("主实验：ResNet18 + 数据增强")
print("=" * 60)
model_r18_aug = build_model('resnet18', len(class_names))
r18_aug_tl, r18_aug_ta, r18_aug_vl, r18_aug_va, r18_aug_best = train_two_stage(
    model_r18_aug, tra_loader, val_loader, freeze_epochs=5, finetune_epochs=5, save_prefix='r18_aug'
)

#%%
# ==================================================
# 对照实验1：ResNet34 + 有数据增强
# ==================================================
print("=" * 60)
print("对照实验1：ResNet34 + 数据增强")
print("=" * 60)
model_r34_aug = build_model('resnet34', len(class_names))
r34_aug_tl, r34_aug_ta, r34_aug_vl, r34_aug_va, r34_aug_best = train_two_stage(
    model_r34_aug, tra_loader, val_loader, freeze_epochs=5, finetune_epochs=5, save_prefix='r34_aug'
)

#%%
# ==================================================
# 对照实验2：ResNet18 + 无数据增强
# ==================================================
print("=" * 60)
print("对照实验2：ResNet18 + 无数据增强")
print("=" * 60)
model_r18_noaug = build_model('resnet18', len(class_names))
r18_noaug_tl, r18_noaug_ta, r18_noaug_vl, r18_noaug_va, r18_noaug_best = train_two_stage(
    model_r18_noaug, tra_loader_noaug, val_loader, freeze_epochs=5, finetune_epochs=5, save_prefix='r18_noaug'
)

#%%
# ==================================================
# 训练曲线可视化（主实验ResNet18+增强）
# ==================================================
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

freeze_epochs = 5
total_epochs = 10

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(total_epochs), r18_aug_tl, label="Train Loss", linewidth=1.5)
plt.plot(range(total_epochs), r18_aug_vl, label="Val Loss", linewidth=1.5)
plt.axvline(x=freeze_epochs - 1, color='r', linestyle='--', label="进入微调阶段")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Training & Validation Loss (ResNet18+Aug)")
plt.grid(alpha=0.3); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(total_epochs), r18_aug_ta, label="Train Acc", linewidth=1.5)
plt.plot(range(total_epochs), r18_aug_va, label="Val Acc", linewidth=1.5)
plt.axvline(x=freeze_epochs - 1, color='r', linestyle='--', label="进入微调阶段")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy (ResNet18+Aug)")
plt.grid(alpha=0.3); plt.legend()

plt.tight_layout()
plt.savefig("training_curve.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# ==================================================
# 对照实验结果对比
# ==================================================
print("\n" + "=" * 60)
print("对照实验结果汇总（验证集最佳准确率）")
print("=" * 60)
print(f"{'实验配置':<25} {'最佳Val Acc':<15}")
print("-" * 40)
print(f"{'ResNet18 + 数据增强':<25} {r18_aug_best:.4f} ({r18_aug_best*100:.2f}%)")
print(f"{'ResNet34 + 数据增强':<25} {r34_aug_best:.4f} ({r34_aug_best*100:.2f}%)")
print(f"{'ResNet18 + 无数据增强':<25} {r18_noaug_best:.4f} ({r18_noaug_best*100:.2f}%)")
print("-" * 40)
print(f"ResNet34 vs ResNet18 提升: {(r34_aug_best - r18_aug_best)*100:.2f}%")
print(f"有增强 vs 无增强 提升: {(r18_aug_best - r18_noaug_best)*100:.2f}%")

# 对照实验柱状图
plt.figure(figsize=(8, 5))
exp_names = ['ResNet18\n+Aug', 'ResNet34\n+Aug', 'ResNet18\nNoAug']
exp_accs = [r18_aug_best, r34_aug_best, r18_noaug_best]
bars = plt.bar(exp_names, exp_accs, color=['#4C72B0', '#55A868', '#C44E52'])
for bar, acc in zip(bars, exp_accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{acc*100:.2f}%', ha='center', fontsize=11)
plt.ylabel('Validation Accuracy')
plt.title('对照实验：模型结构 & 数据增强对比')
plt.ylim(0, 1.05)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("comparison_experiment.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# ==================================================
# 测试集最终评估（用主实验最佳模型）
# ==================================================
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, weight_path, data_loader, device, model_name):
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())
    all_true, all_pred = np.array(all_true), np.array(all_pred)
    total_acc = np.sum(all_true == all_pred) / len(all_true)
    print(f"\n===== {model_name} =====")
    print(f"测试集整体准确率：{total_acc:.4f} ({total_acc * 100:.2f}%)")
    return total_acc, all_true, all_pred

print("=" * 60)
print("测试集最终评估")
print("=" * 60)

test_acc_r18, true_r18, pred_r18 = evaluate_model(
    model_r18_aug, 'r18_aug_finetune.pth', test_loader, device, "ResNet18+增强(主实验)"
)
test_acc_r34, true_r34, pred_r34 = evaluate_model(
    model_r34_aug, 'r34_aug_finetune.pth', test_loader, device, "ResNet34+增强(对照)"
)
test_acc_noaug, true_noaug, pred_noaug = evaluate_model(
    model_r18_noaug, 'r18_noaug_finetune.pth', test_loader, device, "ResNet18无增强(对照)"
)

print(f"\n测试集对比：")
print(f"  ResNet18+增强: {test_acc_r18*100:.2f}%")
print(f"  ResNet34+增强: {test_acc_r34*100:.2f}%")
print(f"  ResNet18无增强: {test_acc_noaug*100:.2f}%")

# 主实验详细评估
print("\n【主实验分类别准确率】")
for i, class_name in enumerate(class_names):
    class_mask = true_r18 == i
    class_acc = np.sum(pred_r18[class_mask] == i) / np.sum(class_mask)
    print(f"  {class_name:10s} : {class_acc:.4f} ({class_acc*100:.2f}%)")

print("\n【详细分类报告】")
print(classification_report(true_r18, pred_r18, target_names=class_names))

# 混淆矩阵
cm = confusion_matrix(true_r18, pred_r18)
cm_norm = confusion_matrix(true_r18, pred_r18, normalize='true')

plt.figure(figsize=(16, 7))
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('预测类别'); plt.ylabel('真实类别'); plt.title('混淆矩阵(绝对数量)')

plt.subplot(1, 2, 2)
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('预测类别'); plt.ylabel('真实类别'); plt.title('混淆矩阵(归一化比例)')

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# ==================================================
# 模型落地：单图预测 + 批量预测重命名
# ==================================================
from PIL import Image

model_r18_aug.load_state_dict(torch.load('r18_aug_finetune.pth', map_location=device, weights_only=True))
model_r18_aug.eval()

pred_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(img_path, show=True):
    image = Image.open(img_path).convert('RGB')
    img_tensor = pred_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model_r18_aug(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probabilities, dim=1)
    pred_class = class_names[pred_idx.item()]
    confidence_score = confidence.item()
    print(f"预测类别：{pred_class}，置信度：{confidence_score*100:.2f}%")
    if show:
        plt.figure(figsize=(5, 5))
        plt.imshow(image); plt.axis('off'); plt.title(f'{pred_class} ({confidence_score*100:.1f}%)')
        plt.tight_layout(); plt.show()
    return pred_class, confidence_score

def batch_predict_rename(input_dir, output_dir):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_ext]
    print(f"检测到 {len(image_files)} 张图片")
    for img_path in image_files:
        image = Image.open(img_path).convert('RGB')
        img_tensor = pred_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model_r18_aug(img_tensor)
            _, pred_idx = torch.max(outputs, 1)
        pred_class = class_names[pred_idx.item()]
        new_name = f"{img_path.stem}_{pred_class}{img_path.suffix}"
        shutil.copy(img_path, output_dir / new_name)
        print(f"{img_path.name} → {new_name}")
    print(f"完成，保存至：{output_dir.resolve()}")

# 调用示例
# predict_image(r"预测图片1.jpg")
# batch_predict_rename("./weather/test/foggy", "./predicted")
