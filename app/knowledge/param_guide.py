"""
参数指南数据库 — 每个缺失参数的详细说明

为 intent_parser, dialogue_manager, gradio_app 提供统一的参数元数据
和格式化的参数缺失指南渲染（Markdown / HTML）。
"""


def _range_text(param: str, fmt: str = "g") -> str:
    """
    从 `ParameterRangeConstraint.PHYSICAL_RANGES` 取上界，供 `valid_range` 文案使用。

    为什么惰性 import：`app.knowledge` 是叶子层，而 `app.physics_engine.constraints`
    已经 import 了 `app.knowledge.semantic_db`，在模块顶层反向 import 会形成
    包级循环。放到函数里，import 只在实际渲染指南时发生。

    为什么要取而不是抄：这里原先写死"瞬态: 1~100期, 每期1~3650天"，与实际生效的
    范围（见 `PHYSICAL_RANGES`）不一致，等于对用户说谎。抄一份常量必然会再漂移。
    """
    try:
        from app.physics_engine.constraints import ParameterRangeConstraint
        hi = ParameterRangeConstraint.PHYSICAL_RANGES[param][1]
        return f"{hi:.0e}" if fmt == "e" else f"{int(hi):d}"
    except Exception:  # pragma: no cover - 仅文案，取不到就不显示上界
        return "?"


# ================================================================
# 参数指南数据
# ================================================================
PARAM_GUIDE = {
    # ── P1 阻断性参数 ──
    "project.name": {
        "display_name_cn": "项目名称",
        "nature": "标识建模项目的名称，用于区分不同模拟方案",
        "priority": "P1（阻断性）— 缺少此参数无法构建模型",
        "valid_range": "任意英文字符串，建议使用有意义的名字",
        "format": "字符串，如 'example_project'",
        "unit": "无",
        "example": "我的项目名称是'承压含水层水流模拟'",
        "why_needed": "用于标识和保存模拟方案，是模型的基本元数据",
    },
    "project.model_type": {
        "display_name_cn": "模拟类型",
        "nature": "指定模拟类型：地下水水流、溶质运移或活性传输",
        "priority": "P1（阻断性）— 缺少此参数无法构建模型",
        "valid_range": "groundwater_flow（地下水流）/ solute_transport（溶质运移）/ reactive_transport（活性传输）",
        "format": "字符串，取值: groundwater_flow | solute_transport | reactive_transport",
        "unit": "无",
        "example": "我只需要模拟地下水流",
        "why_needed": "决定系统调用 MODFLOW 水流模型还是 MT3DMS/RT3D 溶质运移模型",
    },
    "domain.x_range": {
        "display_name_cn": "模型东西向范围（x_range）",
        "nature": "模型在东西方向（X轴）的范围，定义模拟区域的宽度",
        "priority": "P1（阻断性）— 缺少此参数无法生成网格",
        "valid_range": "建议 100 m ~ 50,000 m（1m ~ 50km）",
        "format": "列表 [xmin, xmax]，单位：米",
        "unit": "米 (m)",
        "example": "模型东西向范围从0到1000米",
        "why_needed": "定义模拟区域空间范围，是网格生成的基础",
    },
    "domain.y_range": {
        "display_name_cn": "模型南北向范围（y_range）",
        "nature": "模型在南北方向（Y轴）的范围，定义模拟区域的长度",
        "priority": "P1（阻断性）— 缺少此参数无法生成网格",
        "valid_range": "建议 100 m ~ 50,000 m",
        "format": "列表 [ymin, ymax]，单位：米",
        "unit": "米 (m)",
        "example": "模型南北向范围从0到800米",
        "why_needed": "定义模拟区域空间范围，与x_range共同确定模型面积",
    },
    "grid.nrow": {
        "display_name_cn": "网格行数 (nrow)",
        "nature": "模型在Y方向（南北）的网格行数，决定垂向分辨率",
        "priority": "P1（阻断性）— 缺少此参数无法生成网格",
        "valid_range": "建议 20 ~ 500 行（根据模型范围和所需精度）",
        "format": "整数，如 50",
        "unit": "无（行数）",
        "example": "模型划分50行",
        "why_needed": "决定模型的空间离散精度，影响计算结果准确性",
    },
    "grid.ncol": {
        "display_name_cn": "网格列数 (ncol)",
        "nature": "模型在X方向（东西）的网格列数，决定水平分辨率",
        "priority": "P1（阻断性）— 缺少此参数无法生成网格",
        "valid_range": "建议 20 ~ 500 列",
        "format": "整数，如 50",
        "unit": "无（列数）",
        "example": "模型划分50列",
        "why_needed": "与nrow共同定义网格规模，影响计算耗时与精度",
    },
    "aquifer.type": {
        "display_name_cn": "含水层类型",
        "nature": "含水层的水力特性类型：承压（confined）、潜水/非承压（unconfined）或可转换（convertible）",
        "priority": "P1（阻断性）— 缺少此参数无法选择控制方程",
        "valid_range": "confined（承压）| unconfined（非承压/潜水）| convertible（可转换）",
        "format": "字符串，取值: confined | unconfined | convertible",
        "unit": "无",
        "example": "这是一个承压含水层",
        "why_needed": "决定使用承压或非承压地下水流动方程，影响储水项处理",
    },
    "aquifer.parameters.hk": {
        "display_name_cn": "渗透系数 (hk)",
        "nature": "含水层的水平方向渗透系数，表示地下水通过多孔介质的能力",
        "priority": "P1（阻断性）— 缺少此参数无法求解水流方程",
        "valid_range": "亚粘土 0.05~0.5, 粉砂 1~1.5, 细砂 5~10, 中砂 10~25, 粗砂 25~50, 砾砂 50~100, 圆砾 75~150, 卵石 100~200 m/d（据《水文地质手册》第二版 表C.2）",
        "format": "浮点数，单位：m/d（米/天）",
        "unit": "m/d（米/天）",
        "example": "渗透系数是 15 m/d",
        "why_needed": "达西定律核心参数，控制地下水流速与流量",
    },
    "boundary_conditions": {
        "display_name_cn": "边界条件",
        "nature": "模型边界的物理条件，决定地下水在模型边界上的行为。包括：定水头(chd)、定流量(ghb/well)、河流(riv)、面状补给(rch)、排水沟(drn)等",
        "priority": "P1（阻断性）— 缺少此参数模型定义不完整",
        "valid_range": "取决于边界类型，定水头范围一般在 0~5000 m（高程），流量范围 ±10000 m³/d",
        "format": "列表，每个边界条件为字典 {type, name, coordinates, parameters}。\n"
                 "  常见类型: chd（定水头）, ghb（通用水头）, wel（井）, riv（河流）, rch（面状补给）, drn（排水沟）",
        "unit": "取决于边界类型（m, m³/d 等）",
        "example": "西侧定水头边界100m，东侧定水头边界95m，中心一口抽水井流量200m³/d",
        "why_needed": "边界条件是地下水流动的驱动力，定义了水从哪里来、往哪里去",
    },
    # ── P2 影响性参数 ──
    "grid.top": {
        "display_name_cn": "顶板高程 (top)",
        "nature": "含水层顶面的海拔高程，定义模型顶面",
        "priority": "P2（影响性）— 缺少则使用默认值100m",
        "valid_range": "一般 -100 m ~ 5000 m（海拔），应高于底板高程",
        "format": "浮点数或二维数组，单位：米",
        "unit": "米 (m)",
        "example": "含水层顶板高程为 100 m",
        "why_needed": "定义含水层的几何形态，影响水头分布",
    },
    "grid.bottom": {
        "display_name_cn": "底板高程 (bottom)",
        "nature": "含水层底面的海拔高程，定义模型底面",
        "priority": "P2（影响性）— 缺少则使用默认值50m",
        "valid_range": "一般 -200 m ~ 5000 m（海拔），应低于顶板高程",
        "format": "浮点数或二维数组，单位：米",
        "unit": "米 (m)",
        "example": "含水层底板高程为 50 m",
        "why_needed": "与顶板高程共同定义含水层厚度，决定过水断面",
    },
    "stress_periods": {
        "display_name_cn": "应力期设置",
        "nature": "模拟的时间离散方案，包括是稳态还是瞬态、模拟时长、应力期数量和步长",
        "priority": "P2（影响性）— 缺少则使用默认稳态单期",
        "valid_range": f"稳态: 1期; 瞬态: nper 1~{_range_text('nper')} 期, "
                       f"total_time 1e-3~{_range_text('total_time', 'e')} 天, "
                       f"perlen 1e-3~{_range_text('perlen', 'e')} 天/期, "
                       f"nstp 1~{_range_text('nstp')}",
        # 注意字段语义：**总时长用 total_time，不要用 perlen**。
        # `perlen` 按 MF6 原生语义是**单个**应力期的长度，用户说"模拟时长15000天、
        # 分为10个应力期"时若写成平铺 perlen=15000，总时长会变成 150000 天（差 10 倍）。
        # 早期这里唯一的示例正是平铺 perlen，是那个 bug 的误导来源之一。
        "format": "字典，包含 nper / steady_state / total_time / nstp 等字段。"
                 "瞬态示例: {\"nper\": 10, \"steady_state\": false, \"total_time\": 15000, \"nstp\": 1}。"
                 "total_time 是**总**模拟时长，程序按 total_time/nper 等分每期长度。",
        "unit": "天 (d)",
        "example": "模拟时长15000天，均匀分为10个应力期，每个应力期1个时间步"
                   "（→ {\"nper\": 10, \"steady_state\": false, \"total_time\": 15000, \"nstp\": 1}）",
        "why_needed": "决定模拟是稳态还是瞬态，以及时间分辨率",
    },
    "aquifer.parameters.sy": {
        "display_name_cn": "给水度 (sy)",
        "nature": "非承压/潜水含水层的给水度，表示重力排水条件下释放的水量",
        "priority": "P2（影响性）— 潜水含水层必填",
        "valid_range": "亚粘土 0.04~0.07, 亚砂土 0.07~0.10, 粉砂 0.10~0.15, 细砂 0.10~0.20, 中砂 0.15~0.30, 粗砂 0.20~0.35, 裂隙灰岩 0.008~0.10（据《水文地质手册》第二版 表C.3）",
        "format": "浮点数，无量纲",
        "unit": "无量纲 (-)",
        "example": "给水度为 0.15",
        "why_needed": "控制潜水含水层的释水能力，影响水位下降速率",
    },
    "aquifer.parameters.ss": {
        "display_name_cn": "储水系数 (ss)",
        "nature": "承压含水层的单位储水系数，表示单位水头下降时单位体积释放的水量",
        "priority": "P2（影响性）— 承压含水层必填",
        "valid_range": "塑性粘土 (2.6~20)×10⁻³, 硬质粘土 (1.3~2.6)×10⁻³, 松散砂层 (4.9~10)×10⁻⁴, 致密砂砾石 (4.9~10)×10⁻⁵, 基岩裂隙层 (3.3~69)×10⁻⁶ 1/m（据 Domenico 1996 表C.4）",
        "format": "浮点数，单位：1/m",
        "unit": "1/m",
        "example": "储水系数为 0.00001 /m",
        "why_needed": "控制承压含水层的弹性释水能力",
    },
    # ── P3 优化性参数 ──
    "aquifer.parameters.porosity": {
        "display_name_cn": "孔隙度",
        "nature": "含水层介质的有效孔隙度，用于计算实际流速",
        "priority": "P3（优化性）— 可使用默认值0.25",
        "valid_range": "粘土 40~70%, 粉砂 35~50%, 细砂 26~53%, 粗砂 31~46%, 砂岩 5~30%, 灰岩 0~20%, 裂隙结晶岩 0~10%（据《水文地质学基础》第六版 表C.10）",
        "format": "浮点数，无量纲",
        "unit": "无量纲 (-)",
        "example": "孔隙度为 0.30",
        "why_needed": "将达西流速转换为实际平均流速，用于溶质运移计算",
    },
    "aquifer.parameters.vka": {
        "display_name_cn": "垂向渗透系数 (vka)",
        "nature": "含水层的垂向渗透系数，通常小于水平渗透系数",
        "priority": "P3（优化性）— 默认按 Kh/Kv=10 自动推算",
        "valid_range": "各向异性比 Kh/Kv 通常 1~100；vka = hk / (Kh/Kv)",
        "format": "浮点数，单位：m/d",
        "unit": "m/d（米/天）",
        "example": "垂向渗透系数为 1.5 m/d",
        "why_needed": "控制垂向水流交换，影响三维流场形态",
    },
    "transport.dispersion.al": {
        "display_name_cn": "纵向弥散度 (al)",
        "nature": "溶质在流动方向上的机械弥散尺度，表征多孔介质中溶质的扩散混合程度",
        "priority": "P3（优化性）— 可使用默认经验值（约模型长度的1/10）",
        "valid_range": "0.01 ~ 1000 m；经验公式: al ≈ L/10（L为模型特征长度）",
        "format": "浮点数，单位：米",
        "unit": "米 (m)",
        "example": "纵向弥散度为 50 米",
        "why_needed": "控制溶质运移的对流-弥散过程，影响污染羽扩散形态",
    },
    # ── GHB / 边界参数 ──
    "ghb_stage": {
        "display_name_cn": "GHB 边界水头 (stage)",
        "nature": "通用水头边界 (GHB) 上的参考水头/水位，代表边界外侧水源的水力状态",
        "priority": "P1（阻断性）— GHB 边界缺少水头值无法计算交换量",
        "valid_range": "一般 0 ~ 5000 m（海拔高程），应与实际观测水位一致",
        "format": "浮点数或一维数组（沿弧段线性插值），单位：米 (m)",
        "unit": "米 (m)",
        "example": "河流水位为 212 m",
        "why_needed": "驱动 GHB 边界与含水层之间的水量交换",
    },
    "ghb_conductance": {
        "display_name_cn": "GHB 导水系数 (conductance)",
        "nature": "通用水头边界的导水系数，控制边界与含水层之间交换速率。Cond = K × L × W / M",
        "priority": "P2（影响性）— 可使用经验值 1~100 (m²/d)/m",
        "valid_range": "0.001 ~ 10000 (m²/d)/m；河流边界通常 1~100",
        "format": "浮点数或一维数组，单位：(m²/d)/m",
        "unit": "(m²/d)/m",
        "example": "河流导水系数为 1.0 (m²/d)/m",
        "why_needed": "控制 GHB 边界的渗透能力，影响交换量大小",
    },
    # ── 初始条件 ──
    "initial_conditions": {
        "display_name_cn": "初始条件",
        "nature": "模拟开始时的水头/浓度分布。稳态模拟中仅作为迭代初值；瞬态模拟中为必需初始状态",
        "priority": "P2（影响性）— 稳态可使用默认值（顶板高程）；瞬态必须提供",
        "valid_range": "水头：一般 0~5000 m（海拔）；浓度：≥0 mg/L",
        "format": "浮点数或二维数组。通常设 uniform 值或从文件导入",
        "unit": "水头: m；浓度: mg/L",
        "example": "初始水头为 230 m（等于顶板高程）",
        "why_needed": "稳态模拟的初始解迭代起点；瞬态模拟的历史状态",
    },
    # ── 网格 ──
    "grid.nlay": {
        "display_name_cn": "网格层数 (nlay)",
        "nature": "模型的垂向分层数，控制三维结构的复杂程度",
        "priority": "P3（优化性）— 单层含水层默认 1",
        "valid_range": "一般 1 ~ 50 层",
        "format": "整数，如 1",
        "unit": "无（层数）",
        "example": "模型为单层含水层",
        "why_needed": "决定模型在垂向上的离散精度",
    },
    "grid.delr": {
        "display_name_cn": "行宽 (delr)",
        "nature": "网格沿行方向（通常为东西向）的间距",
        "priority": "P3（优化性）— 默认按模型宽度/ncol自动计算",
        "valid_range": "建议 5 ~ 500 m，取决于模型尺度",
        "format": "浮点数或一维数组，单位：米",
        "unit": "米 (m)",
        "example": "网格间距为 20 米",
        "why_needed": "控制行方向上的空间分辨率",
    },
    "grid.delc": {
        "display_name_cn": "列宽 (delc)",
        "nature": "网格沿列方向（通常为南北向）的间距",
        "priority": "P3（优化性）— 默认按模型高度/nrow自动计算",
        "valid_range": "建议 5 ~ 500 m",
        "format": "浮点数或一维数组，单位：米",
        "unit": "米 (m)",
        "example": "网格间距为 20 米",
        "why_needed": "控制列方向上的空间分辨率",
    },
}


# ================================================================
# 统一的参数缺失指南渲染
# ================================================================

def render_missing_params_guidance(p1_missing: list, p2_missing: list, p3_missing: list,
                                   *, format: str = "markdown") -> str:
    """
    渲染缺失参数详细指南 — 所有消费者的单一入口

    Args:
        p1_missing: P1 阻断性缺失参数列表
        p2_missing: P2 影响性缺失参数列表
        p3_missing: P3 优化性缺失参数列表
        format: "markdown" (聊天) 或 "html" (结果面板)

    Returns:
        格式化的参数字南字符串
    """
    if format == "html":
        return _render_html(p1_missing, p2_missing, p3_missing)
    return _render_markdown(p1_missing, p2_missing, p3_missing)


def _guide_for(pname: str) -> dict:
    """获取参数指南（含默认值）。对于未知参数，智能推断元数据。"""
    if pname in PARAM_GUIDE:
        return PARAM_GUIDE[pname]

    # ── 智能推断：从参数路径中提取关键词，提供合理的默认指南 ──
    path_lower = pname.lower()

    # 判断参数类别
    if "stage" in path_lower or "head" in path_lower:
        return {
            "display_name_cn": pname,
            "nature": "边界上的水头/水位值，定义该位置的水力条件",
            "valid_range": "一般 0 ~ 5000 m（海拔高程），应根据实际水文资料确定",
            "format": "浮点数，单位：米 (m)",
            "unit": "米 (m)",
            "example": f"请输入{_extract_position(pname)}的水头值，例如 212.0",
        }
    if "conductance" in path_lower or "cond" in path_lower:
        return {
            "display_name_cn": pname,
            "nature": "边界的水力传导系数，控制边界与含水层之间的水量交换速率",
            "valid_range": "一般 0.01 ~ 10000 (m²/d)/m，河流边界通常取 1~100",
            "format": "浮点数，单位：(m²/d)/m",
            "unit": "(m²/d)/m",
            "example": f"请输入{_extract_position(pname)}的导水系数，例如 1.0",
        }
    if "rate" in path_lower or "pumping" in path_lower or "flow" in path_lower:
        return {
            "display_name_cn": pname,
            "nature": "井或边界的抽注水速率，正值表示注水，负值表示抽水",
            "valid_range": "一般 -10000 ~ 10000 m³/d",
            "format": "浮点数，单位：m³/d",
            "unit": "m³/d",
            "example": f"请输入{_extract_position(pname)}的流量，例如 -200",
        }
    if "boundary" in path_lower:
        return {
            "display_name_cn": pname,
            "nature": "边界条件参数，定义地下水在模型边界上的行为",
            "valid_range": "取决于边界类型（水头/流量/导水系数），请参考具体边界的物理含义",
            "format": "取决于参数类型，通常为浮点数或字符串",
            "unit": "取决于参数类型",
            "example": f"请描述{_extract_position(pname)}的边界条件类型和取值",
        }
    if "elevation" in path_lower or "elev" in path_lower or "top" in path_lower or "bottom" in path_lower or "rbot" in path_lower:
        return {
            "display_name_cn": pname,
            "nature": "高程值，定义含水层或边界的几何位置",
            "valid_range": "一般 -200 ~ 5000 m（海拔高程）",
            "format": "浮点数，单位：米 (m)",
            "unit": "米 (m)",
            "example": f"请输入{_extract_position(pname)}的高程，例如 175.0",
        }
    if "initial" in path_lower or "starting" in path_lower:
        return {
            "display_name_cn": pname,
            "nature": "模型的初始条件参数，用于设置模拟开始时的状态",
            "valid_range": "取决于参数类型（水头/浓度），水头一般 0~5000 m",
            "format": "取决于参数类型，通常为浮点数",
            "unit": "取决于参数类型（水头: m, 浓度: mg/L）",
            "example": f"请输入{_extract_position(pname)}的初始值，例如 230.0",
        }
    if "recharge" in path_lower or "rch" in path_lower:
        return {
            "display_name_cn": pname,
            "nature": "面状补给速率，表示降雨入渗等补给强度",
            "valid_range": "一般 0.0001 ~ 0.1 m/d（0.04~36.5 m/yr）",
            "format": "浮点数或二维数组，单位：m/d",
            "unit": "m/d",
            "example": "请输入补给速率，例如 0.001",
        }

    # 通用回退：根据路径结构提供基本线索
    parts = pname.split(".")
    last_part = parts[-1] if parts else pname
    return {
        "display_name_cn": pname,
        "nature": f"「{last_part}」参数 — 该参数尚未在系统参数库中注册，建议以标准 JSON IR 路径提供（如 aquifer.parameters.hk、grid.nrow 等）",
        "valid_range": f"请根据专业经验判断「{last_part}」的合理取值范围",
        "format": f"取决于「{last_part}」的参数类型，通常为浮点数、整数或字符串",
        "unit": "",
        "example": f"请输入「{last_part}」的取值",
    }


def _extract_position(pname: str) -> str:
    """从参数路径中提取位置/方向信息"""
    pos_hints = {
        "west": "西侧", "east": "东侧", "north": "北侧", "south": "南侧",
        "left": "左侧", "right": "右侧", "top": "顶部", "bottom": "底部",
    }
    for key, cn in pos_hints.items():
        if key in pname.lower():
            return cn + "边界"
    return "该边界"


def _render_markdown(p1: list, p2: list, p3: list) -> str:
    """渲染为聊天 Markdown"""
    lines = []

    if p1:
        lines.append(f"## ⛔ 无法开始建模！关键参数缺失（{len(p1)} 项）\n")
        lines.append("以下参数是建模所必需的，系统无法自动推断，需要您手动提供：\n")
        for i, pname in enumerate(p1, 1):
            g = _guide_for(pname)
            lines.append(f"### {i}. {g['display_name_cn']} (`{pname}`)")
            lines.append(f"- 📖 **性质**：{g['nature']}")
            lines.append(f"- 📏 **取值范围**：{g['valid_range']}")
            lines.append(f"- 📐 **格式**：{g['format']}")
            if g.get("unit") and g["unit"] != "无":
                lines.append(f"- 📊 **单位**：{g['unit']}")
            lines.append(f"- 💬 **输入示例**：_{g['example']}_")
            lines.append("")

    if p2:
        lines.append(f"## ⚠️ 重要参数缺失（{len(p2)} 项）\n")
        lines.append("以下参数影响模型完整性，建议补充：\n")
        for i, pname in enumerate(p2, 1):
            g = _guide_for(pname)
            lines.append(f"### {i}. {g['display_name_cn']} (`{pname}`)")
            lines.append(f"- 📖 **性质**：{g['nature']}")
            lines.append(f"- 📏 **取值范围**：{g['valid_range']}")
            lines.append(f"- 💬 **输入示例**：_{g['example']}_")
            lines.append("")

    if p3:
        lines.append(f"💡 另有 {len(p3)} 个可选参数缺失，可使用默认值。")

    lines.append("---")
    lines.append("**💡 操作建议：** 输入 **使用建议值** 自动补全可补的参数；手动输入上述关键参数的具体数值。")
    if p1:
        g = _guide_for(p1[0])
        if g.get("example"):
            lines.append(f"\n例如：_{g['example']}_")

    return "\n".join(lines)


def _render_html(p1: list, p2: list, p3: list) -> str:
    """渲染为结果面板 HTML（醒目的红色/橙色卡片）"""
    lines = []

    if p1:
        lines.append(
            f"<div style='background:#fff5f5; border:2px solid #d63031; border-radius:8px; padding:16px; margin-bottom:12px;'>"
            f"<p style='color:#d63031; font-size:18px; font-weight:bold; margin:0 0 8px 0;'>"
            f"⛔ 无法开始建模 — {len(p1)} 个关键参数缺失</p>"
            f"<p style='color:#636e72; font-size:13px; margin:0 0 12px 0;'>"
            f"以下参数是建模所必需的，系统无法自动推断，需手动提供：</p>"
        )
        for i, pname in enumerate(p1, 1):
            g = _guide_for(pname)
            fmt = g.get("format", "—").replace("<", "&lt;").replace(">", "&gt;")
            unit_cell = ""
            if g.get("unit") and g["unit"] != "无":
                unit_cell = (f"<tr><td style='padding:2px 8px 2px 0; white-space:nowrap;'><b>📊 单位</b></td>"
                             f"<td>{g['unit']}</td></tr>")
            lines.append(
                f"<div style='background:#ffeaa7; border-left:4px solid #d63031; padding:10px 14px; "
                f"margin-bottom:8px; border-radius:4px;'>"
                f"<p style='color:#2d3436; font-weight:bold; margin:0 0 4px 0;'>"
                f"{i}. {g['display_name_cn']} (<code>{pname}</code>)</p>"
                f"<table style='font-size:13px; color:#2d3436; margin:0;'>"
                f"<tr><td style='padding:2px 8px 2px 0; white-space:nowrap;'><b>📖 性质</b></td>"
                f"<td>{g['nature']}</td></tr>"
                f"<tr><td style='padding:2px 8px 2px 0; white-space:nowrap;'><b>📏 取值范围</b></td>"
                f"<td>{g['valid_range']}</td></tr>"
                f"<tr><td style='padding:2px 8px 2px 0; white-space:nowrap;'><b>📐 格式</b></td>"
                f"<td>{fmt}</td></tr>"
                f"{unit_cell}"
                f"<tr><td style='padding:2px 8px 2px 0; white-space:nowrap;'><b>💬 输入示例</b></td>"
                f"<td style='color:#0984e3;'><i>{g['example']}</i></td></tr>"
                f"</table></div>"
            )
        lines.append("</div>")

    if p2:
        lines.append(
            f"<div style='background:#fff8ee; border:2px solid #e17055; border-radius:8px; padding:14px; margin-bottom:12px;'>"
            f"<p style='color:#e17055; font-size:16px; font-weight:bold; margin:0 0 8px 0;'>"
            f"⚠️ {len(p2)} 个重要参数缺失 — 模型可能不完整</p>"
        )
        for i, pname in enumerate(p2, 1):
            g = _guide_for(pname)
            lines.append(
                f"<div style='background:#fab1a0; border-left:4px solid #e17055; padding:8px 12px; "
                f"margin-bottom:6px; border-radius:4px;'>"
                f"<p style='color:#2d3436; font-weight:bold; margin:0 0 2px 0;'>"
                f"{i}. {g['display_name_cn']} (<code>{pname}</code>)</p>"
                f"<p style='font-size:13px; margin:0;'>"
                f"📖 {g['nature']}<br>"
                f"📏 取值范围：{g['valid_range']}<br>"
                f"💬 示例：<i style='color:#0984e3;'>{g['example']}</i></p>"
                f"</div>"
            )
        lines.append("</div>")

    if p3:
        lines.append(
            f"<p style='color:#636e72; font-size:13px;'>"
            f"💡 另有 {len(p3)} 个可选参数缺失，可使用默认值</p>"
        )

    lines.append(
        "<div style='margin-top:16px; padding:12px; background:#dfe6e9; border-radius:6px;'>"
        "<b>💡 操作建议：</b><br>"
        "1️⃣ 在左侧对话框中输入参数信息（见上方示例）<br>"
        "2️⃣ 点击 <b>建议补全</b> 自动填充可推断的参数<br>"
        "3️⃣ 补全后再次点击 <b>开始建模</b>"
        "</div>"
    )

    return "\n".join(lines)
