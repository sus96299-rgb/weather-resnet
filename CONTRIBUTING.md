# 贡献指南 Contributing

欢迎提 Issue 和 PR！为了让协作更顺畅，请先花两分钟读完本指南。

## 🛠️ 开发环境搭建

```bash
# 1. 克隆仓库
git clone https://github.com/sus96299-rgb/weather-resnet.git
cd weather-resnet

# 2. 建议使用虚拟环境
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. 安装依赖（含开发/测试工具）
pip install -r requirements-dev.txt
```

> GPU 用户请按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择对应 CUDA 版本安装。

## 📦 准备数据（仅训练需要）

推理演示直接使用 `checkpoints/best_finetune.pth`，无需数据集。
训练 / 复现实验请按 README「数据准备」一节获取 Kaggle 数据集并组织成
`weather/{train,val,test}/<类别>/` 结构。**请勿把数据集提交到仓库。**

## ✅ 提交前请确保

```bash
# 代码风格检查
ruff check src tests

# 全部测试通过（测试不依赖数据集和 GPU，约 1 分钟内完成）
pytest
```

可选：安装 pre-commit 钩子，每次 commit 自动格式化：

```bash
pre-commit install
```

## 📝 提交信息规范

采用[约定式提交](https://www.conventionalcommits.org/zh-hans/)：

- `feat: 新增视频逐帧推理`
- `fix: 修复 Windows 下中文路径读取问题`
- `docs: 补充数据准备说明`
- `test: 为 predict 增加边界测试`
- `refactor: 抽离数据增强配置`

## 🔀 Pull Request 流程

1. Fork 仓库并从 `main` 切出分支：`git checkout -b feat/your-feature`
2. 完成改动并补充/更新测试
3. 确认本地 `pytest`、`ruff check` 均通过
4. 推送到你的 Fork 并发起 PR，按模板填写改动说明
5. CI 通过后等待 review，有问题及时跟进

## 💡 可以从哪里入手

- 欢迎在 Issues 中认领标有 `good first issue` 的问题
- 补充更多天气类别、尝试其他骨干网络（MobileNet / EfficientNet）
- 改进数据增强策略、补充消融实验
- 完善文档与注释

感谢你的贡献！🎉
