"""
天气图像分类 —— 推理脚本（单张预测 / 文件夹批量预测）
=========================================
加载训练好的 ResNet18 权重，对图片做 8 类天气识别：
cloudy(多云) / foggy(雾) / lightning(闪电) / rainbow(彩虹) /
rainy(雨) / rime(雾凇) / sandstorm(沙尘暴) / sunrise(日出)

用法示例：
    # 单张图片：打印 Top-3 概率，并在图上标注结果后另存
    python predict.py --image ../results/demo_rainy.jpg

    # 批量预测整个文件夹（不递归）
    python predict.py --image-dir D:/some_photos

    # 指定别的权重
    python predict.py --image xx.jpg --weights ../checkpoints/best_finetune.pth
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# 项目根目录（src/ 的上一级），权重、类别文件都相对它定位
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ImageFolder 按字母序生成类别索引，顺序固定如下
CLASS_NAMES = [
    "cloudy", "foggy", "lightning", "rainbow",
    "rainy", "rime", "sandstorm", "sunrise",
]
CLASS_NAMES_CN = {
    "cloudy": "多云", "foggy": "雾天", "lightning": "闪电",
    "rainbow": "彩虹", "rainy": "雨天", "rime": "雾凇",
    "sandstorm": "沙尘暴", "sunrise": "日出",
}
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_model(weights_path: Path, device: torch.device) -> nn.Module:
    """构建与训练时一致的 ResNet18：ImageNet 骨干 + 8 分类头，再加载权重。"""
    model = models.resnet18(weights=None)  # 推理无需下载预训练权重
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


# 评估时的预处理必须与训练/验证完全一致：Resize→ToTensor→ImageNet 标准化
predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict_one(model: nn.Module, image: Image.Image, device: torch.device,
                topk: int = 3):
    """对单张 PIL 图推理，返回 [(类别名, 概率), ...]（按概率降序，取前 topk）。"""
    tensor = predict_transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    top_probs, top_idx = torch.topk(probs, k=min(topk, len(CLASS_NAMES)))
    # tolist() 后已经是 Python float，直接用即可
    return [(CLASS_NAMES[i], float(p)) for i, p in zip(top_idx.tolist(), top_probs.tolist())]


def save_annotated(image: Image.Image, label: str, conf: float, out_path: Path):
    """把预测结果画到图上并保存（matplotlib 中文字体做了兼容）。"""
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境也能存图
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(6, 6))
    plt.imshow(image.convert("RGB"))
    plt.axis("off")
    plt.title(f"{label} {CLASS_NAMES_CN.get(label, '')}  ({conf * 100:.1f}%)",
              fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="天气图像分类推理")
    parser.add_argument("--image", help="单张图片路径")
    parser.add_argument("--image-dir", help="批量预测：图片文件夹路径")
    parser.add_argument("--weights", default=str(PROJECT_ROOT / "checkpoints" / "best_finetune.pth"),
                        help="模型权重路径")
    parser.add_argument("--topk", type=int, default=3, help="显示概率最高的前 K 类")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("必须指定 --image（单张）或 --image-dir（批量）之一")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_path = Path(args.weights)
    assert weights_path.exists(), f"找不到权重文件: {weights_path}"
    model = build_model(weights_path, device)
    print(f"模型加载完成: {weights_path.name} | device={device}\n")

    # ---------- 单张预测 ----------
    if args.image:
        img_path = Path(args.image)
        assert img_path.exists(), f"找不到图片: {img_path}"
        image = Image.open(img_path)
        results = predict_one(model, image, device, args.topk)
        print(f"📷 {img_path.name}")
        for name, p in results:
            print(f"   {name:10s} ({CLASS_NAMES_CN[name]}): {p * 100:5.2f}%")
        top_label, top_conf = results[0]
        out_path = img_path.with_name(f"{img_path.stem}_pred.jpg")
        save_annotated(image, top_label, top_conf, out_path)
        print(f"   标注图已保存: {out_path}\n")

    # ---------- 批量预测 ----------
    if args.image_dir:
        in_dir = Path(args.image_dir)
        assert in_dir.exists(), f"找不到文件夹: {in_dir}"
        files = sorted(f for f in in_dir.iterdir()
                       if f.is_file() and f.suffix.lower() in VALID_EXT)
        print(f"检测到 {len(files)} 张图片：")
        for f in files:
            image = Image.open(f)
            label, conf = predict_one(model, image, device, topk=1)[0]
            print(f"   {f.name:40s} → {label:10s} ({CLASS_NAMES_CN[label]}) {conf * 100:5.2f}%")


if __name__ == "__main__":
    main()
