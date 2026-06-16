"""
地下水建模专业语义库 — 查询与匹配API

提供术语模糊匹配、参数范围查询、单位换算等功能。
"""

import sqlite3
import json
import logging
import threading
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from app.config import settings

logger = logging.getLogger(__name__)


class SemanticDB:
    """
    地下水建模专业语义库

    线程安全：所有 SQLite 操作通过 _lock 序列化，
    check_same_thread=False 允许 Gradio worker 线程访问。
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.semantic_db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            # check_same_thread=False 允许跨线程访问（Gradio worker threads）
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    def _execute(self, query: str, params=()):
        """线程安全的 SQL 执行"""
        with self._lock:
            return self.conn.execute(query, params)

    # ================================================================
    # 术语查询
    # ================================================================

    def search_term(self, keyword: str, category: Optional[str] = None) -> List[dict]:
        """
        模糊搜索术语——支持中英文、别名匹配

        Args:
            keyword: 搜索关键词（如 "渗透系数"、"K"、"hydraulic conductivity"）
            category: 限定类别 (aquifer|boundary|transport|...)

        Returns:
            匹配的术语列表
        """
        keyword_lower = keyword.lower().strip()
        query = """
            SELECT * FROM term_aliases
            WHERE LOWER(standard_name) LIKE ?
               OR LOWER(display_name_cn) LIKE ?
               OR LOWER(display_name_en) LIKE ?
               OR LOWER(aliases) LIKE ?
        """
        like_pattern = f"%{keyword_lower}%"
        params = [like_pattern, like_pattern, like_pattern, like_pattern]

        if category:
            query += " AND category = ?"
            params.append(category)

        cursor = self._execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_term_by_name(self, standard_name: str) -> Optional[dict]:
        """按标准参数名精确查询"""
        cursor = self._execute(
            "SELECT * FROM term_aliases WHERE standard_name = ?",
            (standard_name,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_required_params(self, model_type: str = "groundwater_flow") -> List[dict]:
        """获取指定模型类型的必选参数列表"""
        cursor = self._execute(
            "SELECT * FROM term_aliases WHERE required = 1 ORDER BY priority, category"
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_params_by_category(self, category: str) -> List[dict]:
        """按类别获取参数列表"""
        cursor = self._execute(
            "SELECT * FROM term_aliases WHERE category = ? ORDER BY priority",
            (category,)
        )
        return [dict(row) for row in cursor.fetchall()]

    # ================================================================
    # 参数范围查询
    # ================================================================

    def get_param_range(self, param_name: str, aquifer_type: str = "all") -> Optional[dict]:
        """
        获取参数的物理取值范围

        Returns: {min_value, max_value, unit, typical_value, reference}
        """
        cursor = self._execute(
            """SELECT * FROM param_ranges
               WHERE param_name = ? AND (aquifer_type = ? OR aquifer_type = 'all')
               ORDER BY aquifer_type = 'all' LIMIT 1""",
            (param_name, aquifer_type)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def check_value_in_range(self, param_name: str, value: float,
                              aquifer_type: str = "all") -> Tuple[bool, str]:
        """
        检查参数值是否在物理合理范围内

        Returns:
            (is_valid, message)
        """
        rng = self.get_param_range(param_name, aquifer_type)
        if rng is None:
            return True, f"参数 '{param_name}' 未在物理范围表中定义，默认通过"
        lo, hi = rng['min_value'], rng['max_value']
        if lo <= value <= hi:
            return True, f"参数 {param_name}={value} 在物理范围 [{lo}, {hi}] {rng['unit']} 内"
        else:
            return False, f"参数 {param_name}={value} 超出物理范围 [{lo}, {hi}] {rng['unit']}"

    def get_all_ranges(self) -> List[dict]:
        """获取全部物理取值范围"""
        cursor = self._execute("SELECT * FROM param_ranges ORDER BY param_name")
        return [dict(row) for row in cursor.fetchall()]

    # ================================================================
    # 单位换算
    # ================================================================

    def convert_unit(self, value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """
        单位换算

        Example:
            convert_unit(15, 'm/d', 'cm/s') → 0.01736
        """
        if from_unit == to_unit:
            return value
        cursor = self._execute(
            "SELECT factor FROM unit_conversions WHERE from_unit = ? AND to_unit = ?",
            (from_unit, to_unit)
        )
        row = cursor.fetchone()
        if row:
            return value * row['factor']
        # 尝试反向转换
        cursor = self._execute(
            "SELECT factor FROM unit_conversions WHERE from_unit = ? AND to_unit = ?",
            (to_unit, from_unit)
        )
        row = cursor.fetchone()
        if row and row['factor'] != 0:
            return value / row['factor']
        return None

    def get_compatible_units(self, unit: str) -> List[str]:
        """获取可换算的单位列表"""
        cursor = self._execute(
            """SELECT DISTINCT to_unit FROM unit_conversions WHERE from_unit = ?
               UNION
               SELECT DISTINCT from_unit FROM unit_conversions WHERE to_unit = ?""",
            (unit, unit)
        )
        return [row[0] for row in cursor.fetchall()]

    # ================================================================
    # 建议值生成
    # ================================================================

    def get_suggested_value(self, param_name: str, aquifer_type: str = "all",
                             context: Optional[dict] = None) -> dict:
        """
        为缺失参数生成建议值

        Args:
            param_name: 标准参数名
            aquifer_type: 含水层类型 (confined|unconfined|all)
            context: 上下文参数（用于智能推理，如已知渗透系数推荐弥散度）

        Returns:
            {value, unit, confidence, reason}
        """
        term = self.get_term_by_name(param_name)
        rng = self.get_param_range(param_name, aquifer_type)

        if rng and rng['typical_value']:
            return {
                'value': rng['typical_value'],
                'unit': rng['unit'],
                'confidence': 'medium',
                'reason': f"基于 {aquifer_type} 含水层的文献典型值"
            }
        elif rng:
            mid = (rng['min_value'] + rng['max_value']) / 2
            return {
                'value': round(mid, 6),
                'unit': rng['unit'],
                'confidence': 'low',
                'reason': f"取物理范围中值 [{rng['min_value']}, {rng['max_value']}]"
            }
        else:
            return {
                'value': None,
                'unit': term['unit_standard'] if term else 'unknown',
                'confidence': 'none',
                'reason': f"无可用的物理范围数据"
            }

    def get_suggestions_for_missing(self, missing_param_names: List[str],
                                     aquifer_type: str = "all") -> Dict[str, dict]:
        """为缺失参数批量生成建议值"""
        suggestions = {}
        for pname in missing_param_names:
            suggestions[pname] = self.get_suggested_value(pname, aquifer_type)
        return suggestions

    # ================================================================
    # 统计信息
    # ================================================================

    def get_stats(self) -> dict:
        """获取语义库统计信息"""
        n_terms = self._execute("SELECT COUNT(*) FROM term_aliases").fetchone()[0]
        n_ranges = self._execute("SELECT COUNT(*) FROM param_ranges").fetchone()[0]
        n_units = self._execute("SELECT COUNT(*) FROM unit_conversions").fetchone()[0]
        return {
            'total_terms': n_terms,
            'total_ranges': n_ranges,
            'total_unit_conversions': n_units,
        }


# 全局实例
semantic_db = SemanticDB()
