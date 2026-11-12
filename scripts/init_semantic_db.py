#!/usr/bin/env python
"""
初始化地下水建模专业语义库

运行方式：
    python scripts/init_semantic_db.py
"""

import sys
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "semantic_db.sqlite"
SQL_INIT_PATH = PROJECT_ROOT / "app" / "knowledge" / "db_init.sql"


def create_tables(conn: sqlite3.Connection):
    """创建表结构"""
    with open(SQL_INIT_PATH, 'r', encoding='utf-8') as f:
        sql = f.read()
    conn.executescript(sql)
    conn.commit()
    print("✅ 表结构创建完成")


def insert_term_aliases(conn: sqlite3.Connection):
    """插入术语别名数据"""
    terms = [
        # --- 含水层参数 (aquifer) ---
        ("hk", "渗透系数", "Hydraulic Conductivity",
         '["K", "导水系数", "hydraulic conductivity", "渗透率"]',
         "aquifer", "horizontal", "float", "m/d", 1, 1, "水平方向渗透系数，控制地下水流动速率"),
        ("vka", "垂向渗透系数", "Vertical Hydraulic Conductivity",
         '["Kv", "Kz", "垂向渗透率"]',
         "aquifer", "vertical", "float", "m/d", 1, 1, "垂向渗透系数"),
        ("hani", "水平各向异性比", "Horizontal Anisotropy",
         '["Kh/Kv", "anisotropy"]',
         "aquifer", "anisotropy", "float", "dimensionless", 0, 3, "水平渗透系数与垂向渗透系数之比"),
        ("sy", "给水度", "Specific Yield",
         '["Sy", "specific yield"]',
         "aquifer", "storage", "float", "dimensionless", 1, 1, "潜水含水层重力排水体积与总体积之比"),
        ("ss", "储水系数", "Specific Storage",
         '["Ss", "specific storage", "储水率"]',
         "aquifer", "storage", "float", "1/m", 0, 2, "单位体积含水层在水头降低1单位时释放的水量"),
        ("porosity", "孔隙度", "Porosity",
         '["n", "有效孔隙度", "effective porosity", "孔隙率"]',
         "aquifer", "physical", "float", "dimensionless", 1, 1, "孔隙体积与总体积之比"),

        # --- 网格离散 (grid) ---
        ("nrow", "行数", "Number of Rows",
         '["rows", "行"]', "grid", None, "int", "dimensionless", 1, 1, "模型网格行数"),
        ("ncol", "列数", "Number of Columns",
         '["cols", "columns", "列"]', "grid", None, "int", "dimensionless", 1, 1, "模型网格列数"),
        ("nlay", "层数", "Number of Layers",
         '["layers", "分层", "层"]', "grid", None, "int", "dimensionless", 1, 1, "模型垂向分层数"),
        ("delr", "行宽", "Row Width",
         '["cell width", "网格宽度"]', "grid", None, "float", "m", 1, 1, "沿行方向的单元宽度"),
        ("delc", "列宽", "Column Width",
         '["cell height", "网格高度"]', "grid", None, "float", "m", 1, 1, "沿列方向的单元高度"),
        ("x_range", "模型X范围", "Model X Extent",
         '["xmin xmax", "东西范围"]', "grid", None, "float", "m", 1, 1, "模型东西向范围"),
        ("y_range", "模型Y范围", "Model Y Extent",
         '["ymin ymax", "南北范围"]', "grid", None, "float", "m", 1, 1, "模型南北向范围"),

        # --- 边界条件 (boundary) ---
        ("chd", "定水头边界", "Constant Head",
         '["定水头", "constant head", "Dirichlet"]',
         "boundary", "head", "float", "m", 1, 1, "指定固定水头的边界（Dirichlet边界）"),
        ("ghb", "通用水头边界", "General Head Boundary",
         '["通用水头", "GHB", "Cauchy"]',
         "boundary", "head", "float", "m", 0, 2, "通过河床传导系数与外部水源交换的边界"),
        ("wel", "井", "Well",
         '["抽水井", "注水井", "pumping well", "injection well"]',
         "boundary", "well", "float", "m3/d", 0, 2, "抽水或注水井（Neumann边界）"),
        ("riv", "河流边界", "River",
         '["河流", "river"]',
         "boundary", "river", "float", "m", 0, 2, "河流-含水层交互边界"),
        ("drn", "排水沟边界", "Drain",
         '["排水沟", "drain", "排水"]',
         "boundary", "drain", "float", "m", 0, 2, "仅允许地下水排出的边界"),
        ("rch", "补给边界", "Recharge",
         '["面状补给", "降雨入渗", "recharge"]',
         "boundary", "recharge", "float", "m/d", 0, 2, "面状地下水补给（降雨入渗等）"),

        # --- 初始条件 (initial) ---
        ("strt", "初始水头", "Starting Head",
         '["初始水位", "initial head"]',
         "initial", "head", "float", "m", 1, 1, "模拟开始时的水头分布"),
        ("initial_conc", "初始浓度", "Initial Concentration",
         '["背景浓度", "初始污染物浓度"]',
         "initial", "concentration", "float", "mg/L", 0, 2, "模拟开始时的溶质浓度分布"),

        # --- 源汇项 (source) ---
        ("pumping_rate", "抽水速率", "Pumping Rate",
         '["抽水量", "Q", "开采量"]',
         "source", "well", "float", "m3/d", 0, 2, "井的抽水速率（负值）或注水速率（正值）"),
        ("recharge_rate", "补给速率", "Recharge Rate",
         '["降雨入渗量", "面状补给速率"]',
         "source", "recharge", "float", "mm/yr", 0, 3, "面状地下水补给速率"),

        # --- 溶质运移参数 (transport) ---
        ("al", "纵向弥散度", "Longitudinal Dispersivity",
         '["αL", "longitudinal dispersivity", "弥散度"]',
         "transport", "dispersion", "float", "m", 0, 2, "沿流向的弥散度（约为运移距离的1/10）"),
        ("trpt", "横向弥散度比", "Horizontal Transverse Dispersivity Ratio",
         '["αTH/αL", "横向弥散度"]',
         "transport", "dispersion", "float", "dimensionless", 0, 3, "横向弥散度与纵向弥散度之比（通常0.01~0.3）"),
        ("trpv", "垂向弥散度比", "Vertical Transverse Dispersivity Ratio",
         '["αTV/αL", "垂向弥散度"]',
         "transport", "dispersion", "float", "dimensionless", 0, 3, "垂向弥散度与纵向弥散度之比（通常0.001~0.1）"),
        ("diffc", "分子扩散系数", "Molecular Diffusion Coefficient",
         '["D*", "扩散系数"]',
         "transport", "diffusion", "float", "m2/d", 0, 3, "溶质在水中的分子扩散系数"),

        # --- 反应参数 (reaction) ---
        ("kd", "分配系数", "Distribution Coefficient",
         '["Kd", "吸附系数", "sorption coefficient"]',
         "reaction", "sorption", "float", "L/kg", 0, 3, "线性吸附等温线的分配系数"),
        ("decay_rate", "一级降解速率", "First-order Decay Rate",
         '["λ", "降解速率常数", "decay constant"]',
         "reaction", "decay", "float", "1/d", 0, 3, "一级降解反应速率常数"),
        ("half_life", "半衰期", "Half-life",
         '["t1/2", "降解半衰期"]',
         "reaction", "decay", "float", "d", 0, 3, "污染物降解至原始浓度一半所需时间"),

        # --- 求解器参数 (solver) ---
        ("solver_type", "求解器类型", "Solver Type",
         '["PCG", "NWT", "求解器"]',
         "solver", None, "str", "dimensionless", 0, 3, "MODFLOW求解器（PCG: 预共轭梯度, NWT: Newton）"),
        ("hclose", "水头收敛准则", "Head Closure Criterion",
         '["收敛精度", "收敛准则"]',
         "solver", None, "float", "m", 0, 3, "水头迭代收敛容差"),
        ("max_iter", "最大迭代次数", "Maximum Iterations",
         '["迭代上限"]',
         "solver", None, "int", "dimensionless", 0, 3, "求解器最大迭代次数"),

        # --- 输出控制 (output) ---
        ("save_head", "保存水头", "Save Head",
         '["输出水头"]', "output", None, "bool", "dimensionless", 0, 3, "是否输出水头结果"),
        ("save_budget", "保存水量平衡", "Save Budget",
         '["输出水量平衡"]', "output", None, "bool", "dimensionless", 0, 3, "是否输出水量平衡结果"),
        ("save_conc", "保存浓度", "Save Concentration",
         '["输出浓度"]', "output", None, "bool", "dimensionless", 0, 3, "是否输出溶质浓度结果"),
    ]

    conn.executemany(
        """INSERT OR IGNORE INTO term_aliases
           (standard_name, display_name_cn, display_name_en, aliases,
            category, subcategory, data_type, unit_standard, required, priority, description)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        terms
    )
    conn.commit()
    print(f"✅ 插入 {len(terms)} 条术语别名")


def insert_param_ranges(conn: sqlite3.Connection):
    """插入参数物理取值范围"""
    ranges = [
        # (参数名, 最小值, 最大值, 单位, 典型值, 含水层类型, 参考)
        ("hk", 1e-6, 1e4, "m/d", 15.0, "all", "Freeze & Cherry (1979); Domenico & Schwartz (1998)"),
        ("vka", 1e-6, 1e4, "m/d", 1.5, "all", "Kh/Kv 通常 1~100"),
        ("hani", 1.0, 100.0, "dimensionless", 10.0, "all", "Kh/Kv 各向异性比（水平/垂向，通常≥1）"),
        ("sy", 0.005, 0.4, "dimensionless", 0.15, "unconfined", "Johnson (1967); 典型值 0.1~0.3"),
        ("ss", 1e-7, 1e-2, "1/m", 1e-5, "confined", "Domenico & Mifflin (1965)"),
        ("porosity", 0.05, 0.5, "dimensionless", 0.25, "all", "Freeze & Cherry (1979)"),
        # 下界取 0：纯对流（αL=0）是合法物理情形，MT3DMS 问题1 即为该基准。
        # 若下界取 0.01，约束层会把正确的纯对流算例误判为越界并强行改值。
        ("al", 0.0, 1000.0, "m", 10.0, "all", "Gelhar et al. (1992); 约为运移距离的1/10"),
        ("trpt", 0.01, 0.5, "dimensionless", 0.1, "all", "Gelhar et al. (1992); 通常 0.01~0.3"),
        ("trpv", 0.001, 0.1, "dimensionless", 0.01, "all", "通常 0.001~0.1"),
        ("diffc", 1e-10, 1e-8, "m2/d", 1e-9, "all", "水溶液分子扩散系数"),
        ("kd", 0.0, 1e5, "L/kg", 1.0, "all", "EPA (1999); 随有机碳含量变化"),
        ("decay_rate", 1e-6, 10.0, "1/d", 0.001, "all", "取决于污染物类型和电子受体条件"),
        ("pumping_rate", -10000.0, 10000.0, "m3/d", 100.0, "all", "典型抽水速率范围"),
        ("recharge_rate", 0.0, 5000.0, "mm/yr", 200.0, "all", "中国年均降雨入渗补给量"),
        ("hclose", 1e-8, 0.1, "m", 1e-5, "all", "MODFLOW 默认收敛容差"),
        ("max_iter", 10, 10000, "dimensionless", 100, "all", "求解器迭代次数"),
    ]

    conn.executemany(
        """INSERT OR IGNORE INTO param_ranges
           (param_name, min_value, max_value, unit, typical_value, aquifer_type, reference)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ranges
    )
    conn.commit()
    print(f"✅ 插入 {len(ranges)} 条物理取值范围")


def insert_unit_conversions(conn: sqlite3.Connection):
    """插入单位换算"""
    conversions = [
        # 长度
        ("ft", "m", 0.3048, "length"),
        ("cm", "m", 0.01, "length"),
        ("km", "m", 1000.0, "length"),
        ("in", "m", 0.0254, "length"),
        # 时间
        ("s", "d", 1.0 / 86400.0, "time"),
        ("hr", "d", 1.0 / 24.0, "time"),
        ("yr", "d", 365.25, "time"),
        ("min", "d", 1.0 / 1440.0, "time"),
        # 渗透系数/流速
        ("cm/s", "m/d", 864.0, "velocity"),
        ("ft/d", "m/d", 0.3048, "velocity"),
        ("m/s", "m/d", 86400.0, "velocity"),
        # 补给速率
        ("mm/yr", "m/d", 1.0 / (365.25 * 1000.0), "velocity"),
        # 面积/体积
        ("m2", "ft2", 10.7639, "area"),
        ("m3", "ft3", 35.3147, "volume"),
        ("L", "m3", 0.001, "volume"),
    ]

    conn.executemany(
        "INSERT OR IGNORE INTO unit_conversions (from_unit, to_unit, factor, dimension) VALUES (?, ?, ?, ?)",
        conversions
    )
    conn.commit()
    print(f"✅ 插入 {len(conversions)} 条单位换算")


def verify_db(conn: sqlite3.Connection):
    """验证语义库完整性"""
    print("\n--- 语义库完整性报告 ---")
    cursor = conn.execute("SELECT COUNT(*) FROM term_aliases")
    print(f"  术语总数: {cursor.fetchone()[0]}")
    cursor = conn.execute("SELECT COUNT(*) FROM param_ranges")
    print(f"  参数范围: {cursor.fetchone()[0]}")
    cursor = conn.execute("SELECT COUNT(*) FROM unit_conversions")
    print(f"  单位换算: {cursor.fetchone()[0]}")

    cursor = conn.execute(
        "SELECT category, COUNT(*) FROM term_aliases GROUP BY category ORDER BY category"
    )
    print("\n  按类别统计:")
    for row in cursor:
        print(f"    {row[0]}: {row[1]} 条")

    cursor = conn.execute(
        "SELECT COUNT(*) FROM term_aliases WHERE required = 1"
    )
    print(f"\n  必选参数: {cursor.fetchone()[0]} 个")


if __name__ == "__main__":
    # 确保目录存在
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 删除旧库（如有）
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"🗑️  已删除旧语义库: {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))

    try:
        create_tables(conn)
        insert_term_aliases(conn)
        insert_param_ranges(conn)
        insert_unit_conversions(conn)
        verify_db(conn)

        print(f"\n✅ 语义库初始化完成: {DB_PATH}")
    finally:
        conn.close()
