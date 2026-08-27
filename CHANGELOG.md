# 更新日志 Changelog

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/) 与[约定式提交](https://www.conventionalcommits.org/zh-hans/)。

## [0.1.0] - 2026-08-28

### 新增

- 基于 ResNet18/34 ImageNet 预训练权重的两阶段迁移学习（冻结骨干 → 解冻 Layer4 微调）
- 8 类天气图像分类，测试集准确率 96.18%
- 数据增强：随机翻转/旋转 + ColorJitter（模拟户外光照与大气条件）
- 三组对照实验：ResNet18+增强 / ResNet34+增强 / ResNet18 无增强
- 单张图片推理（Top-K 概率 + 结果可视化）与文件夹批量预测
- 测试集评估：分类报告 + 混淆矩阵
- 从零手写 ResNet18/34/50 与 ResNeXt（`src/resnet_manual.py`）
- 提供训练好的权重 `checkpoints/best_finetune.pth`，可直接体验推理
- GitHub Actions CI（Windows/Ubuntu × Python 3.10/3.12）与 pytest 测试套件
