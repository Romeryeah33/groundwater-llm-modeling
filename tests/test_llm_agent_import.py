"""
验证 LLM Agent 模块导入和基础功能
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    "boundary_conditions": [{"type": "chd", "name": "west"}],
}
missing = parser._rule_based_missing_check(test_ir)
print(f"   Missing: P1={len(missing['priority_1'])}, P2={len(missing['priority_2'])}, P3={len(missing['priority_3'])}")
assert len(missing["priority_1"]) == 0
print("   ✅ Rule-based missing check passed")

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
