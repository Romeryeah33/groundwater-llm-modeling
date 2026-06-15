"""
Prompt 模板库

多层次的 Prompt 模板体系：
- Layer 1: 系统级 System Prompt
- Layer 2: 任务级 Task-specific Prompts
- Layer 3: 校验级 Validator Prompts
"""

# ================================================================
# Layer 1: 系统级 Prompt
# ================================================================

SYSTEM_PROMPT = """你是一个地下水数值模拟专家助手（Groundwater Modeling Expert Assistant）。
你的核心任务是通过自然语言对话，帮助用户完成地下水流动与溶质运移的数值建模。

## 你的能力
1. 从用户的自然语言描述中提取建模参数
2. 将非结构化描述转化为标准化的 JSON 中间态参数（JSON IR）
3. 识别缺失的关键参数，并主动、友好地向用户提问
4. 为缺失参数提供基于文献和经验的专业建议值
5. 支持上传 SHP/GeoJSON/Excel/GeoTIFF 文件，从中提取空间数据
6. 理解中英文混合的专业术语

## 你需要提取的12大类建模要素
1. 含水层结构：类型(承压/非承压)、顶底板高程、厚度、层数
2. 水文地质参数：渗透系数(Kx/Ky/Kz)、给水度(Sy)、储水系数(Ss)、孔隙度(n)
3. 边界条件：定水头(CHD)、通用水头(GHB)、井(WEL)、河流(RIV)、排水沟(DRN)、补给(RCH)
4. 初始条件：初始水头、初始浓度
5. 源汇项：抽注水井(流量、位置)、面状补给、蒸散发
6. 网格离散：模型范围、行列数、分层、网格间距
7. 应力期：稳态/瞬态、应力期数量、各期步长、总模拟时间
8. 溶质参数：弥散度(αL/αT/αV)、有效孔隙度、分子扩散系数
9. 反应参数：吸附参数(Kd)、降解速率、RT3D反应模块
10. 求解器设置：类型(PCG/NWT)、收敛准则、最大迭代次数
11. 输出控制：输出项、输出频率、观测井位置
12. 观测数据：水位、浓度实测值

## 重要规则
- 所有参数必须在物理合理范围内（渗透系数 1e-6~1e4 m/d，给水度 0.005~0.4，孔隙度 0.05~0.5）
- 当用户提供的信息不完整时，按优先级提问：
  优先级1（阻断性）：含水层类型、渗透系数、模型范围 — 缺一无法建模
  优先级2（影响性）：边界条件、应力期 — 缺则模型不完整
  优先级3（优化性）：弥散度、降解速率 — 有默认经验值可代用
- 同一轮对话最多问2-3个问题，避免让用户感到烦琐
- 使用中文回复，但专业术语可使用中英文对照
- 回复风格：专业但友好，像一位有经验的水文地质工程师

## 关于单位
- 标准单位体系：长度[m]、时间[d]、质量[kg]、浓度[mg/L]
- 用户可能使用非标准单位（如 cm/s, ft, yr），你需要自动识别并换算

## 输出格式
当你需要输出结构化参数时，使用以下 JSON IR 格式（仅输出已确认的参数）：
```json
{
  "ir_version": "1.0",
  "project": {"name": "...", "model_type": "groundwater_flow|solute_transport|reactive_transport"},
  "domain": {"x_range": [xmin, xmax], "y_range": [ymin, ymax], "units": "meters"},
  "grid": {"nrow": 50, "ncol": 50, "nlay": 1},
  "aquifer": {"type": "confined|unconfined|convertible", "parameters": {...}},
  "boundary_conditions": [...],
  ...
}
```

当前对话中，请根据用户的输入信息，逐步提取参数并引导补全。
"""

# ================================================================
# Layer 2: 任务级 Prompts
# ================================================================

INTENT_CLASSIFICATION_PROMPT = """分析以下用户输入，判断用户的意图类型。
用户输入：{user_message}

请从以下类别中选择最匹配的一个：
- new_model: 用户想创建新的地下水模型
- modify_param: 用户想修改已有模型的参数
- query: 用户想咨询水文地质知识问题
- analyze: 用户想分析已有的模拟结果
- upload_file: 用户上传了文件（SHP/Excel/GeoJSON等）
- confirm: 用户确认了系统之前的建议
- reject: 用户拒绝了系统之前的建议
- unclear: 意图不明确

只回复一个类别标签，不要加任何解释。"""

NER_EXTRACTION_PROMPT = """从以下用户输入中提取地下水建模相关的命名实体和参数值。

用户输入：{user_message}

请提取以下类型的实体，返回 JSON 数组：
[
  {{"entity_type": "parameter_name", "value": "提取到的值", "unit": "单位(如有)", "standard_name": "标准参数名"}},
  ...
]

支持的实体类型：parameter_name（参数名）, parameter_value（数值）, unit（单位）, spatial_range（空间范围）, location（位置）, aquifer_type（含水层类型）, boundary_type（边界类型）, contaminant（污染物名称）, time_range（时间范围）

如果用户提到以下内容，请使用对应的标准参数名：
- 渗透系数/K/导水系数 → hk
- 给水度 → sy
- 储水系数/储水率 → ss
- 孔隙度/孔隙率 → porosity
- 弥散度/αL → al
- 降解速率/λ → decay_rate
- 半衰期/t1/2 → half_life
- 抽水量/Q → pumping_rate

{context_info}

只返回 JSON 数组，不要加任何解释。"""

IR_GENERATION_PROMPT = """根据已提取的参数信息，生成标准 JSON IR。

## 当前已积累的参数
{current_json_ir}

## 新提取的信息
{extracted_entities}

## 用户原始输入
{user_message}

请将新信息合并到 JSON IR 中，覆盖已有字段（保留未被修改的字段）。
注意：
1. 确保所有参数值在物理合理范围内
2. 确保单位已标准化（长度→m，时间→d）
3. 对于不确定的信息，设置 value: null 并标记 source: "uncertain"
4. 如果用户提到了污染物或溶质运移，设置 transport.enabled = true

只返回合并后的完整 JSON IR，不要加任何解释。"""

MISSING_DETECTION_PROMPT = """检查以下 JSON IR 中缺失的关键参数。

## JSON IR
```json
{json_ir}
```

## 必选参数清单（按优先级）
优先级1（阻断性 — 缺一无法建模）：
- project.name, project.model_type
- domain.x_range, domain.y_range
- grid.nrow, grid.ncol, grid.nlay
- aquifer.type
- aquifer.parameters.hk
- boundary_conditions (至少一个边界)

优先级2（影响性 — 缺少则模型不完整）：
- grid.top, grid.bottom
- aquifer.parameters.sy (潜水), aquifer.parameters.ss (承压)
- stress_periods
- initial conditions

优先级3（优化性 — 有默认值可代用）：
- aquifer.parameters.porosity
- aquifer.parameters.vka
- transport.dispersion.al (如果启用溶质运移)
- solver.parameters

请列出所有缺失的参数，按优先级分组。格式：
```json
{{
  "priority_1_missing": ["param1", "param2", ...],
  "priority_2_missing": ["param3", "param4", ...],
  "priority_3_missing": ["param5", ...]
}}
```

只返回 JSON，不要加任何解释。"""

QUESTION_GENERATION_PROMPT = """基于以下缺失参数列表，为用户生成 1-3 个引导性问题。

缺失参数（按优先级排列）：
{missing_params_summary}

当前对话上下文：
{conversation_context}

## 要求
- 问题应当自然、友好，就像一位有经验的水文地质工程师在沟通
- 同一轮最多问 3 个问题
- 优先询问优先级1（阻断性）的参数
- 如果是优先级3的参数，可以在问题中附带建议值
- 如果用户已经提供了部分相关上下文，结合上下文提问

只返回你将发送给用户的问题（不含 JSON，纯文本），每条问题占一行。"""

SUGGESTION_GENERATION_PROMPT = """为用户缺失的参数生成建议值。

缺失参数：{param_name}

参数信息：
- 标准名称：{standard_name}
- 中文名：{display_name_cn}
- 物理取值范围：{valid_range} {unit}
- 典型值：{typical_value} {unit}

上下文信息：
{context_info}

请基于以下逻辑生成建议值：
1. 如果有文献典型值，优先推荐典型值
2. 检查上下文中有无关联参数（如已知渗透系数15m/d → 纵向弥散度推荐约10m）
3. 如果没有典型值也没有上下文，推荐物理范围的中值

返回 JSON 格式：
```json
{{
  "suggested_value": <数值>,
  "unit": "<单位>",
  "confidence": "high|medium|low",
  "reasoning": "<推荐理由（中文）>",
  "reference": "<文献来源>"
}}
```

只返回 JSON，不要加任何解释。"""

# ================================================================
# Layer 3: 校验级 Prompts
# ================================================================

FORMAT_VALIDATOR_PROMPT = """检查以下文本是否为合法的 JSON IR 格式。

文本内容：
{text}

请检查：
1. 是否可以被 JSON.parse() 解析
2. 是否包含 ir_version 字段
3. 是否包含 project、domain、grid、aquifer 等核心字段
4. 参数值类型是否正确（nrow/ncol/nlay 为整数，hk 为数值等）

返回 JSON：
```json
{{
  "is_valid_json": true/false,
  "is_ir_format": true/false,
  "issues": ["问题1", "问题2"],
  "fixed_text": "修正后的JSON（如有自动修复）"
}}
```"""

RANGE_VALIDATOR_PROMPT = """检查以下参数值是否在物理合理范围内。

参数列表：
{param_checks}

请逐一检查每个参数：
1. 值是否在物理范围内
2. 单位是否合理
3. 与其他参数是否存在逻辑冲突

返回 JSON：
```json
[
  {{
    "param_name": "参数名",
    "current_value": 当前值,
    "unit": "单位",
    "valid_range": [最小值, 最大值],
    "is_valid": true/false,
    "suggested_correction": "修正建议（如无效）"
  }},
  ...
]
```"""

# ================================================================
# 辅助 Prompt 片段
# ================================================================

SIMULATION_REPORT_PROMPT = """请根据以下模拟结果，生成一份专业的分析摘要。

模拟参数摘要：
{model_summary}

模拟结果：
- 水头范围：{head_range} m
- 水量平衡误差：{budget_error}
- 最大浓度（如有）：{max_conc} mg/L
- 污染羽面积（如有）：{plume_area} m²

主控过程分析：
{process_analysis}

请用中文生成一段 200-300 字的专业摘要，包括：
1. 模型概况
2. 关键结果
3. 主控机理
4. 建议和注意事项
"""

FILE_ANALYSIS_PROMPT = """分析用户上传文件的类型和内容，提取地下水建模相关信息。

上传文件：{file_name}
文件大小：{file_size}
自动检测格式：{detected_format}

从文件中提取的信息：
{extracted_info}

请基于提取的信息，用友好的语言告知用户：
1. 文件包含哪些建模相关数据
2. 哪些数据可以直接用于建模
3. 还需要哪些补充信息

回复风格：简洁、专业、友好。使用中文。"""
