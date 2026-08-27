"""推理脚本 src/predict.py 的逻辑测试。

用随机初始化模型验证推理链路，不依赖 checkpoints/ 下的真实权重，
因此 CI 中无需 GPU、无需下载大文件。
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

import predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_class_names_completeness():
    """类别表必须是 8 个互不重复的天气类别。"""
    assert len(predict.CLASS_NAMES) == 8
    assert len(set(predict.CLASS_NAMES)) == 8
    assert set(predict.CLASS_NAMES_CN) == set(predict.CLASS_NAMES)


def test_class_indices_json_matches():
    """class_indices.json 必须与脚本中的类别顺序一致（ImageFolder 字母序）。"""
    with open(PROJECT_ROOT / "class_indices.json", encoding="utf-8") as f:
        idx_map = json.load(f)
    for i, name in enumerate(predict.CLASS_NAMES):
        assert idx_map[str(i)] == name


def test_transform_output_shape():
    """预处理后必须是 3×224×224 的张量。"""
    img = Image.new("RGB", (300, 400), color=(100, 150, 200))
    tensor = predict.predict_transform(img)
    assert tensor.shape == (3, 224, 224)


def _random_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 8)
    return model.eval()


def test_predict_one_structure():
    """predict_one 应返回 Top-K 个 (类别名, 概率) 且概率降序、总和为 1。"""
    torch.manual_seed(0)
    model = _random_model()
    img = Image.new("RGB", (256, 256), color=(120, 120, 120))
    results = predict.predict_one(model, img, torch.device("cpu"), topk=3)

    assert len(results) == 3
    names = [name for name, _ in results]
    probs = [p for _, p in results]

    assert all(name in predict.CLASS_NAMES for name in names)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs == sorted(probs, reverse=True)  # 降序
    assert abs(sum(p for _, p in
                   predict.predict_one(model, img, torch.device("cpu"), topk=8)) - 1.0) < 1e-4


def test_build_model_loads_checkpoint(tmp_path):
    """build_model 应能从 state_dict 文件正确加载权重。"""
    model = _random_model()
    weight_file = tmp_path / "random.pth"
    torch.save(model.state_dict(), weight_file)

    loaded = predict.build_model(weight_file, torch.device("cpu"))
    assert loaded.fc.out_features == 8
    # 同一文件加载后参数应逐元素一致
    for p1, p2 in zip(model.parameters(), loaded.parameters()):
        assert torch.allclose(p1, p2)
