"""
全局配置管理

管理所有环境变量、API密钥、路径和模型参数。
支持从 .env 文件和环境变量加载配置。
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "outputs"
TEMPLATE_DIR = DATA_DIR / "templates"
TEST_DIR = PROJECT_ROOT / "tests"


@dataclass
class Settings:
    """全局设置"""

    # --- 项目路径 ---
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR
    template_dir: Path = TEMPLATE_DIR
    workspace_dir: Path = OUTPUT_DIR / "workspaces"

    # --- LLM配置 ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "deepseek")  # deepseek | openai | anthropic
    llm_model: str = os.getenv("LLM_MODEL", "deepseek/deepseek-chat")
    llm_api_key: Optional[str] = os.getenv("LLM_API_KEY", None)
    llm_api_base: Optional[str] = os.getenv("LLM_API_BASE", None)
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    llm_request_timeout: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))

    # --- 物理约束配置 ---
    max_validation_iterations: int = 5  # 闭环校验最大迭代轮次
    water_balance_tolerance: float = 0.01  # 水量平衡容忍度 (1%)
    mass_balance_tolerance: float = 0.02  # 质量守恒容忍度 (2%)
    courant_limit: float = 5.0  # 库朗数上限（隐式格式）
    peclet_limit: float = 10.0  # Peclet数上限

    # --- 建模配置 ---
    default_grid_rows: int = 50
    default_grid_cols: int = 50
    default_grid_layers: int = 1
    default_solver: str = "pcg"
    default_workspace: str = "model_workspace"

    # --- 语义库配置 ---
    semantic_db_path: Path = DATA_DIR / "semantic_db.sqlite"

    # --- 服务配置 ---
    fastapi_host: str = os.getenv("FASTAPI_HOST", "127.0.0.1")
    fastapi_port: int = int(os.getenv("FASTAPI_PORT", "8000"))
    gradio_port: int = int(os.getenv("GRADIO_PORT", "7860"))
    gradio_share: bool = os.getenv("GRADIO_SHARE", "false").lower() == "true"

    # --- 日志配置 ---
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: Path = PROJECT_ROOT / "logs"

    def __post_init__(self):
        """确保输出目录存在"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env_file(cls, env_path: Optional[str] = None) -> "Settings":
        """从 .env 文件加载配置"""
        if env_path is None:
            env_path = PROJECT_ROOT / ".env"
        if Path(env_path).exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip().strip('"').strip("'")
        return cls()


# 全局单例
settings = Settings.from_env_file()
