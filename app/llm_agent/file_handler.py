"""
文件上传解析模块

支持解析：
- SHP (Shapefile) — 边界、河流、点位
- GeoJSON / KML — 空间矢量
- Excel (.xlsx/.xls) — 调查点位、井信息、地层数据
- CSV — 时间序列、点位数据
- GeoTIFF / ASCII Grid — 高程DEM、参数分布栅格
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict
from zipfile import ZipFile

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FileUploadHandler:
    """文件上传解析器"""

    # 支持的文件类型
    SUPPORTED_EXTENSIONS = {
        ".shp": "shapefile_polygon",
        ".geojson": "geojson",
        ".kml": "kml",
        ".xlsx": "excel",
        ".xls": "excel",
        ".csv": "csv",
        ".tif": "geotiff",
        ".tiff": "geotiff",
        ".asc": "ascii_grid",
        ".txt": "text",
        ".prj": "shapefile_aux",
        ".shx": "shapefile_aux",
        ".dbf": "shapefile_aux",
    }

    def __init__(self, output_dir: Optional[Path] = None):
        from app.config import settings
        self.output_dir = output_dir or settings.data_dir / "uploads"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # 主入口
    # ================================================================

    def parse_files(self, files: List) -> dict:
        """
        解析上传的文件列表

        Args:
            files: Gradio 上传的文件列表

        Returns:
            {
                "shapefiles": {"boundary": {...}, "lines": [...], "points": [...]},
                "excel_data": {...},
                "raster_data": {...},
                "summary": "文件解析摘要文本"
            }
        """
        result = {
            "shapefiles": {},
            "excel_data": {},
            "raster_data": {},
            "csv_data": {},
            "errors": [],
            "summary": "",
        }

        # 分组：将SHP相关文件(.shp/.shx/.dbf/.prj)组合在一起
        file_groups = self._group_shapefiles(files)

        for group_name, group_files in file_groups.items():
            if group_name.startswith("shapefile_"):
                parsed = self._parse_shapefile_group(group_files[0])
                if parsed:
                    result["shapefiles"][group_name] = parsed
            elif group_name == "excel":
                for f in group_files:
                    parsed = self._parse_excel(f)
                    if parsed:
                        result["excel_data"][f.name] = parsed
            elif group_name == "csv":
                for f in group_files:
                    parsed = self._parse_csv(f)
                    if parsed:
                        result["csv_data"][f.name] = parsed
            elif group_name == "geojson":
                for f in group_files:
                    parsed = self._parse_geojson(f)
                    if parsed:
                        result["shapefiles"][f.name] = parsed
            elif group_name == "geotiff":
                for f in group_files:
                    parsed = self._parse_geotiff(f)
                    if parsed:
                        result["raster_data"][f.name] = parsed

        result["summary"] = self._generate_summary(result)
        return result

    def _group_shapefiles(self, files: List) -> Dict[str, List]:
        """将SHP相关文件分组"""
        groups = {}
        shp_basenames = {}

        for f in files:
            if not hasattr(f, 'name'):
                continue
            name = f.name if hasattr(f, 'name') else str(f)
            ext = Path(name).suffix.lower()

            if ext == '.shp':
                base = Path(name).stem
                shp_basenames[base] = name
                groups[f"shapefile_{name}"] = [f]

        # 将辅助文件(.shx/.dbf/.prj)关联到对应SHP
        for f in files:
            name = f.name if hasattr(f, 'name') else str(f)
            ext = Path(name).suffix.lower()
            if ext in ('.shx', '.dbf', '.prj'):
                base = Path(name).stem
                for group_name, group_files in list(groups.items()):
                    if Path(group_files[0].name).stem == base:
                        group_files.append(f)
                        break

        # 其他文件按扩展名分组
        for f in files:
            name = f.name if hasattr(f, 'name') else str(f)
            ext = Path(name).suffix.lower()
            if ext not in ('.shp', '.shx', '.dbf', '.prj'):
                category = self.SUPPORTED_EXTENSIONS.get(ext, "other")
                if category not in groups:
                    groups[category] = []
                # 避免重复
                already_in = False
                for g in groups.get(category, []):
                    if hasattr(g, 'name') and g.name == name:
                        already_in = True
                        break
                if not already_in:
                    groups[category].append(f)

        return groups

    # ================================================================
    # SHP 解析
    # ================================================================

    def _parse_shapefile_group(self, shp_file) -> Optional[dict]:
        """解析 Shapefile"""
        try:
            import geopandas as gpd

            # 获取临时路径
            temp_path = self._get_temp_path(shp_file)
            gdf = gpd.read_file(temp_path)

            result = {
                "geometry_type": str(gdf.geometry.type.iloc[0]) if len(gdf) > 0 else "Unknown",
                "feature_count": len(gdf),
                "crs": str(gdf.crs),
                "bounds": gdf.total_bounds.tolist(),  # [xmin, ymin, xmax, ymax]
                "columns": gdf.columns.tolist(),
                "area_m2": None,
            }

            # 计算面积（如果可以投影）
            try:
                projected = gdf.to_crs("EPSG:3857")
                result["area_m2"] = float(projected.area.sum())
            except Exception:
                pass

            # 判断几何类型的建模用途
            geom_type = result["geometry_type"].lower()
            if "polygon" in geom_type:
                result["suggested_use"] = "model_boundary_or_zones"
            elif "line" in geom_type:
                result["suggested_use"] = "river_or_boundary_line"
            elif "point" in geom_type:
                result["suggested_use"] = "wells_or_observation_points"
            else:
                result["suggested_use"] = "unknown"

            return result

        except ImportError:
            logger.warning("geopandas 未安装，无法解析 SHP")
            return {"error": "geopandas 未安装"}
        except Exception as e:
            logger.error(f"SHP解析失败: {e}")
            return {"error": str(e)}

    # ================================================================
    # Excel 解析
    # ================================================================

    def _parse_excel(self, file) -> Optional[dict]:
        """解析 Excel 调查点位/井信息"""
        try:
            temp_path = self._get_temp_path(file)
            df = pd.read_excel(temp_path)

            # 自动识别关键列
            x_col = self._find_column(df, ["x", "经度", "lon", "longitude", "easting", "东坐标", "东"])
            y_col = self._find_column(df, ["y", "纬度", "lat", "latitude", "northing", "北坐标", "北"])
            depth_col = self._find_column(df, ["深度", "depth", "埋深", "孔深", "高程", "elevation", "elev"])
            conc_cols = self._find_conc_columns(df)

            # 判断用途
            if x_col and y_col:
                if conc_cols:
                    use_type = "survey_points_with_concentration"
                elif depth_col:
                    use_type = "borehole_data"
                else:
                    use_type = "point_locations"
            elif depth_col:
                use_type = "depth_series"
            else:
                use_type = "tabular_data"

            return {
                "use_type": use_type,
                "n_rows": len(df),
                "n_cols": len(df.columns),
                "columns": df.columns.tolist(),
                "x_col": x_col,
                "y_col": y_col,
                "depth_col": depth_col,
                "concentration_cols": conc_cols,
                "preview": df.head(5).to_dict("records"),
            }
        except Exception as e:
            logger.error(f"Excel解析失败: {e}")
            return {"error": str(e)}

    def _find_column(self, df, candidates: list) -> Optional[str]:
        """在DataFrame中查找匹配的列名"""
        cols_lower = {c.lower(): c for c in df.columns}
        for candidate in candidates:
            if candidate.lower() in cols_lower:
                return cols_lower[candidate.lower()]
        return None

    def _find_conc_columns(self, df) -> list:
        """查找可能的浓度列"""
        conc_keywords = ["浓度", "conc", "concentration", "含量", "检出", "苯", "甲苯",
                         "tce", "pce", "dce", "vc", "btEX", "cod", "氨氮", "硝酸盐"]
        result = []
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in conc_keywords):
                if df[col].dtype in ("float64", "int64"):
                    result.append(col)
        return result

    # ================================================================
    # GeoJSON 解析
    # ================================================================

    def _parse_geojson(self, file) -> Optional[dict]:
        """解析 GeoJSON 文件"""
        try:
            temp_path = self._get_temp_path(file)
            with open(temp_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            features = data.get("features", [data] if data.get("type") else [])
            feature_count = len(features)
            geom_types = set()
            for feat in features:
                geom = feat.get("geometry", {})
                geom_types.add(geom.get("type", "Unknown"))

            return {
                "format": "GeoJSON",
                "feature_count": feature_count,
                "geometry_types": list(geom_types),
                "crs": data.get("crs", {}).get("properties", {}).get("name", "WGS84"),
                "suggested_use": "boundary_or_features",
            }
        except Exception as e:
            logger.error(f"GeoJSON解析失败: {e}")
            return {"error": str(e)}

    # ================================================================
    # CSV / GeoTIFF 解析
    # ================================================================

    def _parse_csv(self, file) -> Optional[dict]:
        """解析 CSV 文件"""
        try:
            temp_path = self._get_temp_path(file)
            df = pd.read_csv(temp_path)
            return {
                "n_rows": len(df),
                "n_cols": len(df.columns),
                "columns": df.columns.tolist(),
                "preview": df.head(5).to_dict("records"),
            }
        except Exception as e:
            return {"error": str(e)}

    def _parse_geotiff(self, file) -> Optional[dict]:
        """解析 GeoTIFF 栅格文件"""
        try:
            import rasterio
            temp_path = self._get_temp_path(file)
            with rasterio.open(temp_path) as src:
                return {
                    "width": src.width,
                    "height": src.height,
                    "bands": src.count,
                    "crs": str(src.crs),
                    "bounds": list(src.bounds),
                    "resolution": [float(src.res[0]), float(src.res[1])],
                    "suggested_use": "dem_or_parameter_raster",
                }
        except ImportError:
            return {"error": "rasterio 未安装"}
        except Exception as e:
            return {"error": str(e)}

    # ================================================================
    # 工具方法
    # ================================================================

    def _get_temp_path(self, file) -> str:
        """从 Gradio 文件对象获取临时路径"""
        if hasattr(file, "name"):
            # Gradio 临时文件
            return file.name
        return str(file)

    def _generate_summary(self, result: dict) -> str:
        """生成文件解析摘要"""
        parts = []

        # Shapefiles
        for name, info in result.get("shapefiles", {}).items():
            if "error" in info:
                parts.append(f"⚠️ {name}: 解析失败 ({info['error']})")
            else:
                parts.append(
                    f"✅ {name}: {info.get('feature_count', '?')} 个"
                    f"{info.get('geometry_type', 'Unknown')}要素"
                    f" → 建议用作 {info.get('suggested_use', 'unknown')}"
                )

        # Excel
        for name, info in result.get("excel_data", {}).items():
            if "error" in info:
                parts.append(f"⚠️ {name}: 解析失败 ({info['error']})")
            else:
                parts.append(
                    f"✅ {name}: {info.get('n_rows', '?')} 行 × {info.get('n_cols', '?')} 列"
                    f" → 识别为 {info.get('use_type', 'tabular_data')}"
                )
                if info.get("concentration_cols"):
                    parts.append(f"   检测到浓度列: {', '.join(info['concentration_cols'])}")
                if info.get("x_col") and info.get("y_col"):
                    parts.append(f"   坐标列: {info['x_col']}, {info['y_col']}")

        # GeoJSON
        for name, info in result.get("shapefiles", {}).items():
            if info.get("format") == "GeoJSON":
                parts.append(
                    f"✅ {name}: GeoJSON, {info.get('feature_count', '?')} 要素")

        # Raster
        for name, info in result.get("raster_data", {}).items():
            if "error" in info:
                parts.append(f"⚠️ {name}: 解析失败 ({info['error']})")
            else:
                parts.append(
                    f"✅ {name}: 栅格 {info.get('width', '?')}×{info.get('height', '?')}"
                    f", 分辨率 {info.get('resolution', '?')}m")

        if not parts:
            return "📁 文件已收到，但无法识别的格式。"
        return "📁 **文件解析结果：**\n" + "\n".join(parts)


# 全局实例
file_handler = FileUploadHandler()
