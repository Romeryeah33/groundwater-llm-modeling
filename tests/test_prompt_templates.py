"""
提示词模板渲染测试（脚本式，与仓库其它测试一致，不用 pytest）

背景：`IR_GENERATION_PROMPT` 等模板由 `str.format()` 渲染，正文里的 JSON few-shot
若写单层大括号（`{"nper": 10}`），`{` 会被当成替换字段 → `KeyError: '"nper"'`
→ 整条 IR 生成链路崩溃。

为什么原有测试抓不到：测试套件**全都在 mock LLM**，`generate_json_ir()` 里的那句
`IR_GENERATION_PROMPT.format(...)` 从未真正执行过。所以"模板能否渲染"这件事在
验证套件里是**空白**。这个文件就是补上那块空白 —— 它不需要网络，秒级。

约定：新增/修改任何 `*_PROMPT` 模板后，本文件必须仍然通过。
"""
import sys
import os
import string
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}" + (f"  {detail}" if detail else ""))


import app.llm_agent.prompt_templates as PT

# 每个模板的渲染参数。新增模板时要在这里登记，否则会被下面的"未登记"检查报出来。
RENDER_KWARGS = {
    "SYSTEM_PROMPT": None,                 # 不当模板用，只作为系统提示词原文
    "IR_GENERATION_PROMPT": {
        "conversation_history": "（无历史）",
        "current_json_ir": "{}",
        "extracted_entities": "[]",
        "user_message": "测试输入",
    },
    "NER_EXTRACTION_PROMPT": {
        "user_message": "测试输入",
        "context_info": "",
        "conversation_history": "（无历史）",
    },
    "INTENT_CLASSIFICATION_PROMPT": {
        "user_message": "测试输入",
        "current_ir_summary": "{}",
        "conversation_history": "（无历史）",
    },
    "MISSING_DETECTION_PROMPT": {"json_ir": "{}"},
    "QUESTION_GENERATION_PROMPT": {
        "missing_params_summary": "无",
        "conversation_context": "（无）",
    },
    "RANGE_VALIDATOR_PROMPT": {"param_checks": "[]"},
    "SUGGESTION_GENERATION_PROMPT": {
        "param_name": "hk", "standard_name": "hk", "display_name_cn": "渗透系数",
        "valid_range": "0.001~1000", "unit": "m/d", "typical_value": "10",
        "context_info": "",
    },
    "SIMULATION_REPORT_PROMPT": {
        "model_summary": "无", "head_range": "1~2", "budget_error": "0",
        "max_conc": "0", "plume_area": "0", "process_analysis": "无",
    },
    "FILE_ANALYSIS_PROMPT": {
        "file_name": "a.shp", "file_size": "1KB",
        "detected_format": "shp", "extracted_info": "无",
    },
    "FORMAT_VALIDATOR_PROMPT": {"text": "{}"},
}


def find_prompt_names():
    return [
        n for n in dir(PT)
        if not n.startswith("_") and isinstance(getattr(PT, n), str)
        and (n.endswith("_PROMPT") or n.endswith("_TEMPLATE"))
    ]


def section_1_all_templates_render():
    """核心：每个模板都必须能用登记的参数渲染成功。"""
    print("\n--- 1. 模板渲染（.format 不抛异常）---")
    names = find_prompt_names()
    check("发现提示词模板", len(names) >= 9, f"实际 {len(names)}: {names}")

    for name in names:
        check(f"模板已登记: {name}", name in RENDER_KWARGS,
              "新增模板请在 RENDER_KWARGS 登记其渲染参数")
        if name not in RENDER_KWARGS:
            continue
        kwargs = RENDER_KWARGS[name]
        if kwargs is None:
            continue
        try:
            out = getattr(PT, name).format(**kwargs)
            check(f"{name} 可渲染", isinstance(out, str) and len(out) > 0)
        except (KeyError, IndexError, ValueError) as e:
            check(f"{name} 可渲染", False,
                  f"{type(e).__name__}: {e} —— 正文里可能有未转义的 JSON 大括号，"
                  f"请把 {{ 写成 {{{{，}} 写成 }}}}")

    # 登记了但模块里不存在的（改名/删除后忘了同步）
    for name in RENDER_KWARGS:
        check(f"登记项仍存在: {name}", hasattr(PT, name))


def section_2_no_unregistered_fields():
    """
    渲染成功还不够 —— 用 Formatter.parse 直接看字段名。
    `{"nper": 10}` 这种"字段名"能通过 format（如果恰好传了同名 kwarg），
    但字段名带引号/冒号/空格一定是 JSON 字面量漏转义。
    """
    print("\n--- 2. 字段名合法性（引号/冒号/空格 = JSON 漏转义）---")
    for name in find_prompt_names():
        tmpl = getattr(PT, name)
        if RENDER_KWARGS.get(name) is None:
            continue
        offenders = []
        for _lit, fname, _spec, _conv in string.Formatter().parse(tmpl):
            if fname is None:
                continue
            root = fname.split(".")[0].split("[")[0]
            if not root.isidentifier():
                offenders.append(fname)
        check(f"{name} 无非法字段名", not offenders, f"非法字段: {offenders[:3]}")

    # SYSTEM_PROMPT 明确不参与 .format()（只作为系统消息原文），单独锁住这个前提：
    # 一旦有人开始 format 它，那处既有的 JSON 字面量就会炸，这里先红。
    src = ""
    try:
        import inspect
        src = inspect.getsource(PT)
    except Exception:
        pass
    check("SYSTEM_PROMPT 未被 .format() 调用",
          "SYSTEM_PROMPT.format(" not in src,
          "若确实要 format SYSTEM_PROMPT，需先转义其中的 JSON 大括号")


def section_3_ir_prompt_keeps_json_intact():
    """渲染后 JSON 必须是**单层**大括号 —— 转义不能漏进最终发给 LLM 的文本。"""
    print("\n--- 3. 渲染后的 JSON 形态 ---")
    out = PT.IR_GENERATION_PROMPT.format(**RENDER_KWARGS["IR_GENERATION_PROMPT"])
    check("渲染后无 {{ 残留", "{{" not in out, "转义写了两层")
    check("含 total_time 正确示例",
          '"stress_periods": {"nper": 10, "steady_state": false, "total_time": 15000'
          in out)
    check("含 period_data 示例",
          '"period_data": [{"perlen": 100, "nstp": 10}' in out)
    check("说明了 perlen 是单期长度", "单个应力期的长度" in out)
    check("说明了带浓度边界必须给位置", "必须能定位到具体网格" in out)
    import json as _json
    # 逐段取出 ```json 代码块，必须能被 json.loads 解析成对象
    n_ok = 0
    for block in out.split("```json")[1:]:
        body = block.split("```")[0].strip()
        try:
            _json.loads("{" + body + "}")
            n_ok += 1
        except Exception:
            pass
    check("few-shot JSON 块本身合法", n_ok >= 3, f"可解析 {n_ok} 块")


def section_4_ner_prompt_renders():
    """NER 模板含大段示例 JSON，是同类风险最高的另一处。"""
    print("\n--- 4. NER 模板 ---")
    out = PT.NER_EXTRACTION_PROMPT.format(**RENDER_KWARGS["NER_EXTRACTION_PROMPT"])
    check("NER 渲染后无 {{ 残留", "{{" not in out)
    check("NER 渲染后含用户输入", "测试输入" in out)


if __name__ == "__main__":
    print("=" * 62)
    print("提示词模板渲染测试")
    print("=" * 62)
    section_1_all_templates_render()
    section_2_no_unregistered_fields()
    section_3_ir_prompt_keeps_json_intact()
    section_4_ner_prompt_renders()
    print("\n" + "=" * 62)
    total = passed + failed
    print(f"提示词模板测试结果: {passed}/{total} 通过 "
          f"({passed / total * 100 if total else 0:.1f}%)")
    if failed:
        print(f"[FAIL] {failed} 项未通过")
    else:
        print("[OK] 全部通过")
    print("=" * 62)
    sys.exit(1 if failed else 0)
