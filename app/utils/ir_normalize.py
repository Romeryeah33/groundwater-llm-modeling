"""
IR 形状归一化 —— `stress_periods` 的防御性解析。

## 为什么单独成模块

`stress_periods` 在实际 IR 里至少有六种形态：

```
{"nper": 10, "period_data": [...]}     # 规范形态
{"nper": 10, "perlen": 1500}           # 平铺（未文档化，但 LLM 常写）
10                                      # 裸整数，表示"分 10 期"
"10"                                    # 字符串
[{"nper": 1, "steady": true}]           # 单元素列表
true / None / 其他非法值                 # 垃圾
```

`model_mapper._map_time_params` 早就有一组防御分支处理它们，但
`physics_engine.validator` 里**没有**，且 `physics_engine` 不能 import `modeling`
（分层约束）。两处各自实现必然会漂移，所以把唯一实现放在 `app/utils/`。

`validator._check_pre_run` 曾因此崩溃：`stress = json_ir.get("stress_periods", {})`
之后直接 `stress.get("nper", 1)`，而 East Texas 的顶层正是裸整数 `10` →
`AttributeError: 'int' object has no attribute 'get'`，从 `validate()` 可达
（`_check_stability` 有 isinstance 守卫，`_check_pre_run` 漏了）。
"""

from typing import Any, Dict

# 时间信息键：顶层只带 nper 而没有这些键时，需要回退到 transport.stress_periods
TIME_KEYS = ("perlen", "nstp", "tsmult", "period_data", "total_time")


def normalize_stress_periods(raw: Any) -> Dict[str, Any]:
    """
    把任意形态的 stress_periods 归一化成 dict。

    - `str` / `bool` / `None` / 其他非法值 → `{}`
    - `int` / `float` → `{"nper": int}`（"分 N 个应力期"这一约定俗成的写法）
    - `list` → 取首元素（再把首元素走一遍本函数）
    - `dict` → 原样返回
    """
    if raw is None:
        return {}
    if isinstance(raw, bool):
        # 注意：bool 是 int 的子类，必须先判，否则 True 会变成 nper=1
        return {}
    if isinstance(raw, str):
        return {}
    if isinstance(raw, (int, float)):
        # 裸数字按"应力期数量"读；否则会退化成"只算 1 个 1 天的应力期"
        try:
            nper = int(raw)
        except (TypeError, ValueError):
            return {}
        return {"nper": nper} if nper > 0 else {}
    if isinstance(raw, list):
        return normalize_stress_periods(raw[0]) if raw else {}
    if isinstance(raw, dict):
        return raw
    return {}


def merge_time_sources(json_ir: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并顶层与 ``transport`` 两处 ``stress_periods``，优先级与建模侧一致。

    为什么需要合并：时间信息会被 LLM 塞进两个地方。判定规则是
    "顶层缺 ``nper``、或缺 ``perlen``/``nstp``/``tsmult``/``period_data``/``total_time``
    时，用 ``transport.stress_periods`` 补齐；顶层已显式给出的键优先"。

    East Texas 案例的形态最能说明问题：顶层是**裸整数** ``10``（归一化后只剩
    ``{"nper": 10}``，没有时间键），真实时间信息 ``{nper:10, perlen:15000}``
    在 ``transport.stress_periods`` 里。只看顶层就什么都看不到 ——
    这正是校验层此前对"总时长差 10 倍"完全瞎掉的原因。

    与 ``model_mapper._map_time_params`` 共用，避免两处各判一次而漂移。
    """
    stress = normalize_stress_periods(json_ir.get("stress_periods"))
    fallback = normalize_stress_periods(
        (json_ir.get("transport") or {}).get("stress_periods")
    )
    if not fallback.get("nper"):
        return stress

    no_nper = not stress.get("nper")
    no_time = not any(stress.get(k) for k in TIME_KEYS)
    if not (no_nper or no_time):
        return stress

    merged = dict(fallback)
    # 顶层已显式给出的键优先
    merged.update({k: v for k, v in stress.items() if v})
    return merged
