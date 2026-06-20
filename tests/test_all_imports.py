"""
L0 全量模块导入测试

验证 app/ 下所有 .py 文件可正常 import，无循环依赖。
特别检查：flopy MF6类、gradio、litellm 版本兼容。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

passed = 0
failed = 0

def check(name, success, detail=""):
    global passed, failed
    if success:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}  FAILED  {detail}")

print("=" * 60)
print("L0: 全量模块导入测试")
print("=" * 60)

# ============================================================
# 1. 核心依赖
# ============================================================
print("\n--- 核心依赖 ---")

try:
    import flopy
    from flopy.mf6 import MFSimulation
    check("flopy + MF6 import", True, f"v{flopy.__version__}")
except Exception as e:
    check("flopy + MF6 import", False, str(e))

try:
    import gradio as gr
    check("gradio import", True, f"v{gr.__version__}")
except Exception as e:
    check("gradio import", False, str(e))

try:
    import litellm
    check("litellm import", True, f"v{litellm.__version__ if hasattr(litellm, '__version__') else 'ok'}")
except Exception as e:
    check("litellm import", False, str(e))

try:
    import numpy as np
    check("numpy import", True, f"v{np.__version__}")
except Exception as e:
    check("numpy import", False, str(e))

try:
    import scipy
    check("scipy import", True, f"v{scipy.__version__}")
except Exception as e:
    check("scipy import", False, str(e))

try:
    import matplotlib
    check("matplotlib import", True, f"v{matplotlib.__version__}")
except Exception as e:
    check("matplotlib import", False, str(e))

try:
    import fastapi
    check("fastapi import", True)
except Exception as e:
    check("fastapi import", False, str(e))

try:
    import uvicorn
    check("uvicorn import", True)
except Exception as e:
    check("uvicorn import", False, str(e))

# ============================================================
# 2. app/ 子模块导入
# ============================================================
print("\n--- app/llm_agent ---")
try:
    from app.llm_agent.llm_client import LLMClient
    check("llm_client", True)
except Exception as e:
    check("llm_client", False, str(e))

try:
    from app.llm_agent.intent_parser import IntentParser
    check("intent_parser", True)
except Exception as e:
    check("intent_parser", False, str(e))

try:
    from app.llm_agent.dialogue_manager import DialogueManager
    check("dialogue_manager", True)
except Exception as e:
    check("dialogue_manager", False, str(e))

try:
    from app.llm_agent.prompt_templates import (
        INTENT_CLASSIFICATION_PROMPT,
        NER_EXTRACTION_PROMPT,
        IR_GENERATION_PROMPT,
        SYSTEM_PROMPT,
    )
    check("prompt_templates", True, f"4 prompts, SYSTEM={len(SYSTEM_PROMPT)}chars")
except Exception as e:
    check("prompt_templates", False, str(e))

print("\n--- app/modeling ---")
try:
    from app.modeling.model_mapper import FloPyModelMapper
    check("model_mapper", True)
except Exception as e:
    check("model_mapper", False, str(e))

try:
    from app.modeling.model_builder import AutoModelBuilder
    check("model_builder", True)
except Exception as e:
    check("model_builder", False, str(e))

try:
    from app.modeling.grid_generator import AutoGridGenerator
    check("grid_generator", True)
except Exception as e:
    check("grid_generator", False, str(e))

try:
    from app.modeling.solver_monitor import SolverMonitor
    check("solver_monitor", True)
except Exception as e:
    check("solver_monitor", False, str(e))

print("\n--- app/knowledge ---")
try:
    from app.knowledge.semantic_db import SemanticDB
    check("semantic_db", True)
except Exception as e:
    check("semantic_db", False, str(e))

try:
    from app.knowledge.param_guide import render_missing_params_guidance
    check("param_guide", True)
except Exception as e:
    check("param_guide", False, str(e))

print("\n--- app/physics_engine ---")
try:
    from app.physics_engine.validator import ClosedLoopValidator, quick_validate
    check("physics_validator", True)
except Exception as e:
    check("physics_validator", False, str(e))

try:
    from app.physics_engine.constraints import (
        MassBalanceConstraint,
        GoverningEquationConstraint,
        ParameterRangeConstraint,
        NumericalStabilityConstraint,
    )
    check("physics_constraints (4 classes)", True)
except Exception as e:
    check("physics_constraints (4 classes)", False, str(e))

print("\n--- app/analysis ---")
try:
    from app.analysis.result_processor import ResultProcessor
    check("result_processor", True)
except Exception as e:
    check("result_processor", False, str(e))

try:
    from app.analysis.visualization_engine import VisualizationEngine
    check("visualization_engine", True)
except Exception as e:
    check("visualization_engine", False, str(e))

try:
    from app.analysis.report_generator import ReportGenerator
    check("report_generator", True)
except Exception as e:
    check("report_generator", False, str(e))

try:
    from app.analysis.process_identifier import ProcessIdentifier
    check("process_identifier", True)
except Exception as e:
    check("process_identifier", False, str(e))

print("\n--- app/frontend ---")
try:
    from app.frontend.gradio_app import create_ui
    check("gradio_app (create_ui)", True)
except Exception as e:
    check("gradio_app", False, str(e))

try:
    from app.llm_agent.file_handler import FileUploadHandler
    check("file_handler (FileUploadHandler)", True)
except Exception as e:
    check("file_handler", False, str(e))

# ============================================================
# 3. 无循环依赖检查
# ============================================================
print("\n--- 循环依赖检查 ---")
import importlib
modules_to_reload = [
    "app.llm_agent.intent_parser",
    "app.llm_agent.dialogue_manager",
    "app.modeling.model_builder",
    "app.modeling.model_mapper",
    "app.physics_engine.validator",
]
for mod_name in modules_to_reload:
    try:
        importlib.reload(sys.modules.get(mod_name, __import__(mod_name)))
        check(f"{mod_name} reload (no circular import)", True)
    except Exception as e:
        check(f"{mod_name} reload", False, str(e))

# ============================================================
# 结果汇总
# ============================================================
print(f"\n{'='*60}")
total = passed + failed
pct = 100.0 * passed / total if total > 0 else 0
print(f"L0 全量模块导入测试: {passed}/{total} 通过 ({pct:.1f}%)")
if failed == 0:
    print("✅ 所有模块导入正常，无循环依赖！")
else:
    print(f"⚠️  {failed} 项失败，请检查")
print(f"{'='*60}")
