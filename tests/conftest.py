"""pytest 全局配置：把 src/ 加入导入路径，使测试可以直接 import 脚本模块。"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
