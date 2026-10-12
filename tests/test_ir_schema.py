# -*- coding: utf-8 -*-
"""IR 协议（docs/ir_schema_v1.0.json）标注的回归锁。

**为什么需要这个文件**：协议里有若干字段是"祖先留下的、当前没人读"的。
把它们标出来是为了让下一个接手的人（很可能就是作者本人，几个月后）
不必重新 grep 一遍就能知道"这个字段写了有用吗"。

但**标错比不标更坏** —— 把活代码误标成 `not_implemented`，等于递了一把
刀：下一个人会照着标注把它删掉。这个风险不是假想的，本轮初稿就标错了两处
（`transport.species` 与 `solver.parameters.relax` 都被当成死字段，
实际 `model_mapper.py` 一直在读），后来又发现第三处
（`boundary_conditions[].stress_periods`）。

所以本文件锁四件事：

  1. 标注表**逐条硬编码**在这里。任何人改动 schema 的标注，都必须同时改这个
     文件 —— 于是改动是"有意的"，不会悄悄漂移。
  2. 被标 `not_implemented` 的字段，**代码里真的没有读取它的地方**。
     这是"标注是否属实"的检查：一旦有人接了线却没改标注，这里会红。
  3. 被标 `implemented` 的字段，标注里写的**消费方还在**（文件存在、行号在范围内、
     那一行确实提到该字段）。这是反向检查：一旦消费方被删掉，标注就成了过期的
     谎话，这里会红，逼着把标注改成 not_implemented。
  4. `IntentParser.VALID_TOP_KEYS` 认可的顶层键，schema 里都得有。
     防止"解析器收下了、协议却不认"的静默不一致 —— `source_sinks` /
     `initial_conditions` / `observation_points` 就是这么漏了三年的。

风格与其它测试一致：脚本式，不用 pytest。``python tests/test_ir_schema.py``
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCHEMA = ROOT / "docs" / "ir_schema_v1.0.json"

_passed = 0


def _ok(msg):
    global _passed
    _passed += 1
    print(f"   OK: {msg}")


# ================================================================
print("\n1. schema 能解析，且顶层结构完整...")
# ================================================================
_raw = SCHEMA.read_text(encoding="utf-8")
doc = json.loads(_raw)                      # 解析失败会直接抛，不必额外断言
assert doc.get("type") == "object", "schema 顶层 type 应为 object"
for key in ("$schema", "type", "required", "properties"):
    assert key in doc, f"schema 缺顶层键 {key}"
assert doc["properties"], "schema 的 properties 是空的"
_ok(f"docs/ir_schema_v1.0.json 合法（顶层 {len(doc['properties'])} 个属性）")

# ================================================================
print("\n2. 依赖方向：schema 不被任何程序加载（纯文档，改动零运行时风险）...")
# ================================================================
# 这条是**事实断言**，不是期望。若哪天代码开始加载 schema，这个测试会红，
# 提醒"它已经不是纯文档了，上面的'零运行时风险'结论作废"。
_loaders = []
for f in (ROOT / "app").rglob("*.py"):
    txt = f.read_text(encoding="utf-8", errors="replace")
    if "ir_schema_v1.0" in txt:
        _loaders.append(f.relative_to(ROOT).as_posix())
assert not _loaders, (
    f"有程序代码引用了 ir_schema_v1.0.json: {_loaders} —— "
    "那么它已不只是文档，本文件关于'零运行时风险'的说明需要重写")
_ok("没有程序加载 schema（它确实只是文档）")


# ================================================================
print("\n3. 标注表与 schema 完全一致（改动必须是有意的）...")
# ================================================================
def walk_status(node, path="", out=None):
    """收集所有带 x-gw-status 的路径。"""
    if out is None:
        out = {}
    if isinstance(node, dict):
        if "x-gw-status" in node:
            out[path] = node["x-gw-status"]
        for k, v in node.items():
            if k in ("properties", "definitions"):
                for kk, vv in v.items():
                    walk_status(vv, f"{path}.{kk}" if path else kk, out)
            elif k == "items":
                walk_status(v, path + "[]", out)
    return out


status = walk_status(doc)

# —— 期望表：这是本文件的核心，逐条硬编码 ——
EXPECTED_NOT_IMPLEMENTED = {
    "domain.rotation_angle":      "网格旋转，建模路径从未读取",
    "domain.crs":                 "IR 里的 CRS 无人读（file_handler 读的是 SHP 自带的 CRS）",
    "aquifer.zones":              "非均质分区未实现",
    "observation_points":         "顶层观测点：解析器收下，无建模代码读",
    "source_sinks":               "源汇项的另一种写法：井/补给实际走 boundary_conditions",
    "transport.reaction.rt3d_package": "RT3D 历史遗留字段，零命中",
    "output":                     "整段输出控制无读取（输出哪些量由程序自行决定）",
    "metadata":                   "整段溯源记录无读取",
}
EXPECTED_IMPLEMENTED = {
    "grid.refinement_zones":              "app/modeling/grid_generator.py",
    "boundary_conditions[].stress_periods": "app/modeling/model_mapper.py",
    "initial_conditions":                 "app/modeling/model_builder.py",
    "transport.species":                  "app/modeling/model_mapper.py",
    "transport.initial_concentration":    "app/modeling/model_mapper.py",
    "transport.advection.scheme":         "app/modeling/model_mapper.py",
    "solver.parameters.relax":            "app/modeling/model_mapper.py",
}

_got_not = {k: v for k, v in status.items() if v == "not_implemented"}
_got_impl = {k: v for k, v in status.items() if v == "implemented"}

assert set(_got_not) == set(EXPECTED_NOT_IMPLEMENTED), (
    "not_implemented 的标注集合变了。\n"
    f"  多余（schema 有、这里没有）: {sorted(set(_got_not) - set(EXPECTED_NOT_IMPLEMENTED))}\n"
    f"  缺失（这里要求、schema 没有）: {sorted(set(EXPECTED_NOT_IMPLEMENTED) - set(_got_not))}\n"
    "如果这是有意的改动，请同步改本文件的期望表 —— 那正是它存在的意义。")

assert set(_got_impl) == set(EXPECTED_IMPLEMENTED), (
    "implemented 的标注集合变了。\n"
    f"  多余: {sorted(set(_got_impl) - set(EXPECTED_IMPLEMENTED))}\n"
    f"  缺失: {sorted(set(EXPECTED_IMPLEMENTED) - set(_got_impl))}")

for path, why in EXPECTED_NOT_IMPLEMENTED.items():
    assert why.strip(), f"{path} 没写理由 —— 标注必须带依据"
_ok(f"标注表一致：{len(_got_not)} 条未实现、{len(_got_impl)} 条已实现（各带理由）")

# ================================================================
print("\n4. 标了 not_implemented 的，代码里真的没人读...")
# ================================================================
# 每个字段配一组"读了它就会命中"的正则。全部在 app/ 下扫，必须零命中。
# 注意：**必须匹配"从 IR 读"的形态**（`.get("x")` / `["x"]`），不能只匹配字段名 ——
# 否则 `source_sinks` 会因为出现在 VALID_TOP_KEYS 白名单里而假红。
IR_READ = {
    "domain.rotation_angle":              r"rotation_angle",
    "domain.crs":                         r"""(?:domain|ir|json_ir|_ir)\s*(?:\.get\(\s*|\[\s*)["']crs["']""",
    "aquifer.zones":                      r"""(?:aquifer|aqua)\s*(?:\.get\(\s*|\[\s*)["']zones["']""",
    "observation_points":                 r"""(?:json_ir|ir|_ir)\s*(?:\.get\(\s*|\[\s*)["']observation_points["']""",
    "source_sinks":                       r"""(?:json_ir|ir|_ir)\s*(?:\.get\(\s*|\[\s*)["']source_sinks["']""",
    "transport.reaction.rt3d_package":    r"rt3d",
    "output":                             r"""(?:json_ir|ir|_ir|self\.json_ir)\s*(?:\.get\(\s*|\[\s*)["']output["']""",
    "metadata":                           r"""["']metadata["']""",
}
_app_py = sorted((ROOT / "app").rglob("*.py"))
assert _app_py, "app/ 下没扫到 .py —— 路径写错了，这条检查等于没跑"

for path, pat in IR_READ.items():
    assert path in EXPECTED_NOT_IMPLEMENTED, \
        f"{path} 配了读取正则，却不在 not_implemented 清单里 —— 对不上"
    rx = re.compile(pat)
    hits = []
    for f in _app_py:
        txt = f.read_text(encoding="utf-8", errors="replace")
        for m in rx.finditer(txt):
            hits.append("%s:%d" % (f.relative_to(ROOT).as_posix(),
                                   txt[:m.start()].count("\n") + 1))
    assert not hits, (
        f"{path} 被标为 not_implemented，但代码里出现了读取它的地方：{hits}\n"
        "要么是标注过期了（请改成 implemented），要么是有人新接了线还没改标注。")
_ok(f"{len(IR_READ)} 个未实现字段在 app/ 下确实零读取（逐条正则核过）")

# ================================================================
print("\n5. 标了 implemented 的，标注里的消费方还在...")
# ================================================================
def _resolve(status_path):
    """按 walk_status 的路径语法定位到 schema 里的那个对象。

    路径段可能是 ``xxx[]``（表示进到数组的 items 里），例如
    ``boundary_conditions[].stress_periods`` —— 少了这层处理就取不到它。
    """
    node = doc
    for part in status_path.split("."):
        if part.endswith("[]"):
            node = node["properties"][part[:-2]]["items"]
        else:
            node = node["properties"][part]
    return node


def _consume_by(status_path):
    """取某个状态路径的 x-gw-consume_by。"""
    return _resolve(status_path).get("x-gw-consume_by")


def _parse_cites(spec):
    """'a/b.py:12, c/d.py:34,56' -> [('a/b.py',12), ('c/d.py',34), ('c/d.py',56)]"""
    out, cur = [], None
    for tok in spec.split(","):
        tok = tok.strip()
        if ":" in tok:
            cur, ln = tok.rsplit(":", 1)
            out.append((cur, int(ln)))
        else:
            assert cur, f"引用片段 {tok!r} 没有前导文件名"
            out.append((cur, int(tok)))
    return out


_checked = 0
for path in EXPECTED_IMPLEMENTED:
    spec = _consume_by(path)
    assert spec, f"{path} 标了 implemented 却没写 x-gw-consume_by —— 无法核对"
    leaf = path.split(".")[-1].replace("[]", "")
    for rel, ln in _parse_cites(spec):
        f = ROOT / rel
        assert f.exists(), f"{path} 引用的消费方文件不存在: {rel}"
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        assert 1 <= ln <= len(lines), f"{path} 引用的 {rel}:{ln} 超出文件行数 {len(lines)}"
        line = lines[ln - 1]
        # 只认**叶子字段名**，且必须按**词边界**匹配。两处都踩过坑：
        #   · 允许"父键出现就算过"→ 像 `transport` 这种父键遍地都是，等于没查；
        #   · 用朴素子串匹配 → 会被变量名骗过：把 `transport.get("species")`
        #     改成 `...("organisms")` 后，`has_species = ...` 里的 "species"
        #     仍然命中了子串，于是删除消费方也照样绿（实测）。
        # 词边界可解：`_` 属 \w，所以 `\bspecies\b` 不会匹配 `has_species`。
        assert re.search(r"\b" + re.escape(leaf) + r"\b", line), (
            f"{path} 引用的 {rel}:{ln} 里没有独立的 {leaf!r} —— 行号漂移了，"
            f"或该消费方已被删改。实际内容: {line.strip()!r}")
        _checked += 1
_ok(f"{len(EXPECTED_IMPLEMENTED)} 个已实现字段的 {_checked} 处消费方引用逐一核对通过")

# ================================================================
print("\n6. 解析器认可的顶层键，schema 里都得有...")
# ================================================================
from app.llm_agent.intent_parser import IntentParser

_valid = set(IntentParser.VALID_TOP_KEYS)
_in_schema = set(doc["properties"])
_missing = sorted(_valid - _in_schema)
assert not _missing, (
    f"IntentParser 认这些顶层键，schema 里却没有: {_missing}\n"
    "解析器会收下、协议却不认 —— 这正是 source_sinks / initial_conditions / "
    "observation_points 当年漏掉的方式。")
_ok(f"VALID_TOP_KEYS 的 {len(_valid)} 个键在 schema 顶层都存在")

# ================================================================
print("\n" + "=" * 60)
print(f"IR 协议标注锁: {_passed} 项通过")
print("=" * 60)
