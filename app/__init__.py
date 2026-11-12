"""
物理约束嵌入大语言模型的地下水智能建模原型系统
Physical-LLM Groundwater Intelligent Modeling System
"""
from pathlib import Path

# 版本号只有一处来源：仓库根目录的 VERSION 文件。改版本请只改 VERSION。
try:
    __version__ = (Path(__file__).resolve().parent.parent
                   / "VERSION").read_text(encoding="utf-8").strip()
except Exception:                      # 文件缺失/编码异常时，不让 import 失败
    __version__ = "unknown"
