"""手写 ResNet（src/resnet_manual.py）的结构与前向传播测试。"""
import torch

from resnet_manual import resnet34, resnet50


def test_resnet34_output_shape():
    """resnet34 分类头输出维度应等于指定类别数。"""
    model = resnet34(num_classes=8)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 8)


def test_resnet34_default_1000_classes():
    """不传 num_classes 时应为 ImageNet 默认 1000 类。"""
    model = resnet34()
    assert model.fc.out_features == 1000


def test_resnet50_output_shape():
    """Bottleneck 结构的 resnet50 同样应输出正确维度。"""
    model = resnet50(num_classes=8)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 8)


def test_resnet_gradient_flows():
    """反向传播应能把梯度传到第一层卷积（验证残差连接无断流）。"""
    model = resnet34(num_classes=8)
    out = model(torch.randn(2, 3, 224, 224))
    out.sum().backward()
    assert model.conv1.weight.grad is not None
    assert model.conv1.weight.grad.abs().sum() > 0
