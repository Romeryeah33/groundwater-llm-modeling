-- ============================================================
-- 地下水建模专业语义库 — 初始化SQL脚本
-- ============================================================

-- 术语别名表（中文名 ↔ 英文名 ↔ 标准参数名）
CREATE TABLE IF NOT EXISTS term_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    standard_name TEXT NOT NULL,           -- 标准参数名 (如 "hk")
    display_name_cn TEXT NOT NULL,         -- 中文显示名 (如 "渗透系数")
    display_name_en TEXT,                  -- 英文显示名 (如 "Hydraulic Conductivity")
    aliases TEXT,                          -- 别名列表, JSON数组: ["K", "渗透率", "导水系数"]
    category TEXT NOT NULL,                -- 所属大类: aquifer|boundary|transport|solver|grid|source|initial|output|reaction
    subcategory TEXT,                      -- 子类别
    data_type TEXT DEFAULT 'float',       -- 数据类型
    unit_standard TEXT,                    -- 标准单位
    required INTEGER DEFAULT 0,            -- 是否必选 (0/1)
    priority INTEGER DEFAULT 3,            -- 问询优先级 (1=阻断, 2=影响, 3=优化)
    description TEXT                       -- 参数说明
);

-- 参数物理取值范围表
CREATE TABLE IF NOT EXISTS param_ranges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    param_name TEXT NOT NULL,              -- 标准参数名
    min_value REAL NOT NULL,               -- 物理最小值
    max_value REAL NOT NULL,               -- 物理最大值
    unit TEXT NOT NULL,                    -- 单位
    typical_value REAL,                    -- 典型值（用于建议值生成）
    aquifer_type TEXT DEFAULT 'all',       -- 适用含水层类型
    reference TEXT,                        -- 数据来源
    notes TEXT                             -- 备注
);

-- 单位换算表
CREATE TABLE IF NOT EXISTS unit_conversions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_unit TEXT NOT NULL,
    to_unit TEXT NOT NULL,
    factor REAL NOT NULL,                  -- to_unit = from_unit * factor
    dimension TEXT NOT NULL                -- 量纲: length|time|velocity|mass|dimensionless
);

-- 建模概念模板表（预定义建模场景）
CREATE TABLE IF NOT EXISTS concept_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    template_name TEXT NOT NULL,
    description TEXT,
    model_type TEXT,                       -- groundwater_flow|solute_transport|reactive_transport
    json_ir_template TEXT                  -- 预填充的JSON IR模板
);
