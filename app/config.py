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


def _env(key: str, default: str = None) -> Optional[str]:
    """延迟读取环境变量（每次调用时重新读取）"""
    return os.getenv(key, default)


@dataclass
class Settings:
    """全局设置"""

    # --- 项目路径 ---
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR
    template_dir: Path = TEMPLATE_DIR

    # --- LLM配置 (使用 __post_init__ 从环境变量读取) ---
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek/deepseek-v4-flash"
    llm_api_key: Optional[str] = None
    llm_api_base: Optional[str] = "https://api.deepseek.com/v1"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096
    llm_request_timeout: int = 120

    # --- 物理约束配置 ---
    max_validation_iterations: int = 5
    water_balance_tolerance: float = 0.01
    mass_balance_tolerance: float = 0.02
    courant_limit: float = 5.0
    peclet_limit: float = 10.0

    # --- 建模配置 ---
    default_grid_rows: int = 50
    default_grid_cols: int = 50
    default_grid_layers: int = 1
    default_solver: str = "pcg"
    default_workspace: str = "model_workspace"

    # --- 语义库配置 ---
    semantic_db_path: Path = field(default_factory=lambda: DATA_DIR / "semantic_db.sqlite")

    # --- 服务配置 ---
    fastapi_host: str = "127.0.0.1"
    fastapi_port: int = 8000
    gradio_port: int = 7860
    gradio_share: bool = False

    # --- 日志配置 ---
    log_level: str = "INFO"

    # --- 运行时路径 ---
    workspace_dir: Path = field(default_factory=lambda: OUTPUT_DIR / "workspaces")
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")

    def __post_init__(self):
        """从环境变量覆盖默认值 + 创建输出目录"""
        # LLM 配置 — 每次创建实例时从当前环境变量重新读取
        self.llm_provider = _env("LLM_PROVIDER") or self.llm_provider
        self.llm_model = _env("LLM_MODEL") or self.llm_model
        self.llm_api_key = _env("LLM_API_KEY") or self.llm_api_key
        self.llm_api_base = _env("LLM_API_BASE") or self.llm_api_base
        self.llm_temperature = float(_env("LLM_TEMPERATURE") or self.llm_temperature)
        self.llm_max_tokens = int(_env("LLM_MAX_TOKENS") or self.llm_max_tokens)
        self.llm_request_timeout = int(_env("LLM_REQUEST_TIMEOUT") or self.llm_request_timeout)

        # 服务配置
        self.fastapi_host = _env("FASTAPI_HOST") or self.fastapi_host
        self.fastapi_port = int(_env("FASTAPI_PORT") or self.fastapi_port)
        self.gradio_port = int(_env("GRADIO_PORT") or self.gradio_port)
        self.gradio_share = (_env("GRADIO_SHARE") or "").lower() == "true"

        # 日志
        self.log_level = _env("LOG_LEVEL") or self.log_level

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env_file(cls, env_path: Optional[str] = None) -> "Settings":
        """从 .env 文件加载环境变量并创建设置实例"""
        if env_path is None:
            env_path = PROJECT_ROOT / ".env"

        env_file = Path(env_path)
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if value and value != f"your_{key.lower()}_here":
                            os.environ[key] = value

        # 创建实例 — __post_init__ 会从 os.environ 读取最新值
        return cls()

    @classmethod
    def reload(cls) -> "Settings":
        """重新加载配置"""
        return cls.from_env_file()


# 全局单例
settings = Settings.from_env_file()
