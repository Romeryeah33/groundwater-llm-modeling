"""
验证 LLM Agent 模块导入和基础功能
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Test imports
from app.llm_agent.prompt_templates import (
    SYSTEM_PROMPT, INTENT_CLASSIFICATION_PROMPT,
    QUESTION_GENERATION_PROMPT, SUGGESTION_GENERATION_PROMPT,
)
from app.llm_agent.intent_parser import IntentParser
from app.llm_agent.dialogue_manager import DialogueManager
from app.llm_agent.file_handler import FileUploadHandler

print("1. Testing imports...")
print("   ✅ Prompt templates imported")

# Test rule-based missing check
parser = IntentParser()
test_ir = {
    "project": {"name": "test", "model_type": "groundwater_flow"},
    "domain": {"x_range": [0, 1000], "y_range": [0, 800]},
    "grid": {"nrow": 50, "ncol": 40, "nlay": 1, "top": 100, "bottom": 50},
    "aquifer": {"type": "confined", "parameters": {"hk": {"value": 15, "unit": "m/d"}}},
    # chd 边界**必须**带水头：缺了它模型根本建不起来，属 P1 阻断项。
    # 这里原本写成 {"type": "chd", "name": "west"}（没给 head），于是下面
    # `P1 == 0` 的断言一直红。**是 fixture 不完整，不是程序缺陷** ——
    # 规则引擎判它 P1 是对的。补齐后 P1 才真的为 0。
    "boundary_conditions": [{"type": "chd", "name": "west",
                             "parameters": {"head": 100}}],
}
missing = parser._rule_based_missing_check(test_ir)
print(f"   Missing: P1={len(missing['priority_1'])}, P2={len(missing['priority_2'])}, P3={len(missing['priority_3'])}")

# 三档都断**具体参数名**，而不只断个数 —— 只断个数是弱锁：引擎退化成
# 永远返回空表时，`len(...) == 0` 和 `len(...) == 恒定值` 都可能照样绿。
assert missing["priority_1"] == [], \
    f"一份完整的 P1 参数集不该报缺失，实际: {missing['priority_1']}"
assert missing["priority_2"] == [
    "stress_periods",              # 没给应力期 → 影响时间离散
    "initial_conditions",          # 没给初始条件
    "aquifer.parameters.ss",       # 承压含水层没给储水系数
], f"P2 与预期不符: {missing['priority_2']}"
assert missing["priority_3"] == [
    "aquifer.parameters.porosity",  # 孔隙度（影响运移）
    "aquifer.parameters.vka",       # 垂向各向异性比（单层可省）
], f"P3 与预期不符: {missing['priority_3']}"
print("   ✅ Rule-based missing check passed (P1/P2/P3 逐项核对)")

# Test DialogueManager
dm = DialogueManager()
assert dm.state["phase"] == "intake"
print("   ✅ DialogueManager imported")

# Test FileUploadHandler
fh = FileUploadHandler()
assert ".shp" in fh.SUPPORTED_EXTENSIONS
assert ".xlsx" in fh.SUPPORTED_EXTENSIONS
print("   ✅ FileUploadHandler imported")

# Verify prompt templates are non-empty
assert len(SYSTEM_PROMPT) > 100
assert "渗透系数" in SYSTEM_PROMPT
print("   ✅ SYSTEM_PROMPT verified (length={})".format(len(SYSTEM_PROMPT)))

print("\n✅ All LLM Agent modules verified successfully!")
