"""
文件上传解析模块

支持解析：
- ZIP 压缩包 — 自动解压并识别内部所有支持的文件格式
- SHP (Shapefile) — 边界、河流、点位
- GeoJSON / KML — 空间矢量
- Excel (.xlsx/.xls) — 调查点位、井信息、地层数据
- CSV — 时间序列、点位数据
- GeoTIFF / ASCII Grid — 高程DEM、参数分布栅格
"""

import json
import logging
import os
import shutil
import tempfile
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
        ".zip": "zip_archive",
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
                "shapefiles": {...},
                "excel_data": {...},
                "raster_data": {...},
                "csv_data": {...},
                "summary": "文件解析摘要文本"
            }
        """
        result = {
            "shapefiles": {},
            "excel_data": {},
            "raster_data": {},
            "csv_data": {},
            "extracted_file_paths": [],  # ZIP 解压出的所有文件路径
            "errors": [],
            "summary": "",
        }

        # 分离 ZIP 文件和非 ZIP 文件
        zip_files = []
        non_zip_files = []
        for f in files:
            if not hasattr(f, 'name'):
                continue
            ext = Path(f.name).suffix.lower()
            if ext == '.zip':
                zip_files.append(f)
            else:
                non_zip_files.append(f)

        # 先处理 ZIP 文件：解压后提取所有可识别文件
        all_extracted_files = []
        for zf in zip_files:
            extracted = self._parse_zip(zf)
            if extracted.get("errors"):
                result["errors"].extend(extracted["errors"])
            # 将解压出的文件路径收集起来，统一解析
            for file_path in extracted.get("extracted_files", []):
                all_extracted_files.append(file_path)
            if extracted.get("extracted_files"):
                logger.info(f"从 {zf.name} 解压了 {len(extracted['extracted_files'])} 个文件")

        # 持久化解压文件路径
        result["extracted_file_paths"] = all_extracted_files

        # 解析解压出的文件
        if all_extracted_files:
            zip_result = self._parse_extracted_files(all_extracted_files)
            self._merge_results(result, zip_result)

        # 处理非 ZIP 文件（原有逻辑）
        file_groups = self._group_shapefiles(non_zip_files)

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

    @staticmethod
    def _merge_results(target: dict, source: dict):
        """将 source 结果合并到 target 中"""
        for key in ("shapefiles", "excel_data", "raster_data", "csv_data"):
            if key in source:
                target[key].update(source[key])
        if "errors" in source:
            target["errors"].extend(source["errors"])

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
    # ZIP 压缩包解析
    # ================================================================

    # 压缩包内可识别的文件扩展名（不含 .zip 本身）
    _RECOGNIZED_EXTS = {".shp", ".shx", ".dbf", ".prj", ".geojson",
                        ".kml", ".xlsx", ".xls", ".csv", ".tif", ".tiff",
                        ".asc", ".txt", ".png", ".jpg", ".jpeg", ".bmp"}

    def _parse_zip(self, zip_file) -> dict:
        """
        解析 ZIP 压缩包：解压到临时目录，扫描可识别的文件

        Returns:
            {"extracted_files": [path, ...], "errors": [...]}
        """
        result = {"extracted_files": [], "errors": []}
        try:
            temp_dir = tempfile.mkdtemp(prefix="gm_zip_")
            zip_path = self._get_temp_path(zip_file)

            with ZipFile(zip_path, 'r') as zf:
                # 安全检查：防止 zip slip 攻击
                for member in zf.namelist():
                    # 跳过目录和隐藏文件
                    if member.endswith('/') or member.startswith('__MACOSX') or member.startswith('.'):
                        continue
                    basename = os.path.basename(member)
                    if basename.startswith('.') or basename.startswith('~'):
                        continue

                    ext = Path(member).suffix.lower()
                    if ext not in self._RECOGNIZED_EXTS:
                        continue

                    try:
                        # 安全提取：只取文件名，丢弃路径前缀
                        dest_path = os.path.join(temp_dir, basename)
                        # 避免同名文件覆盖
                        if os.path.exists(dest_path):
                            stem = Path(basename).stem
                            dest_path = os.path.join(temp_dir, f"{stem}_{hash(member) & 0xFFFF:04x}{ext}")

                        source = zf.open(member)
                        with open(dest_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        result["extracted_files"].append(dest_path)
                        logger.debug(f"ZIP提取: {member} → {basename}")
                    except Exception as e:
                        logger.warning(f"ZIP提取失败: {member}: {e}")

            if not result["extracted_files"]:
                result["errors"].append(f"压缩包 {getattr(zip_file, 'name', '?')} 中未找到可识别的建模数据文件。支持: {', '.join(sorted(self._RECOGNIZED_EXTS))}")

        except Exception as e:
            logger.error(f"ZIP解析失败: {e}")
            result["errors"].append(f"压缩包解析失败: {e}")

        return result

    def _parse_extracted_files(self, file_paths: List[str]) -> dict:
        """
        解析从压缩包中提取的文件列表，按类型分组后调用对应解析器

        将文件路径封装为简单的 FileInfo 对象，复用现有的分组与解析逻辑
        """
        result = {
            "shapefiles": {},
            "excel_data": {},
            "raster_data": {},
            "csv_data": {},
            "errors": [],
        }

        # 封装为简单的类文件对象（兼容现有 _group_shapefiles）
        class _FileInfo:
            def __init__(self, path):
                self.name = path
                self._path = path

        file_objs = [_FileInfo(p) for p in file_paths]

        # 分组解析
        file_groups = self._group_shapefiles(file_objs)

        for group_name, group_files in file_groups.items():
            if group_name.startswith("shapefile_"):
                parsed = self._parse_shapefile_group(group_files[0])
                if parsed:
                    # 用原始文件名作为 key
                    display_name = Path(group_files[0].name).stem
                    result["shapefiles"][display_name] = parsed
            elif group_name == "excel":
                for f in group_files:
                    parsed = self._parse_excel(f)
                    if parsed:
                        key = Path(f.name).name
                        result["excel_data"][key] = parsed
            elif group_name == "csv":
                for f in group_files:
                    parsed = self._parse_csv(f)
                    if parsed:
                        key = Path(f.name).name
                        result["csv_data"][key] = parsed
            elif group_name == "geojson":
                for f in group_files:
                    parsed = self._parse_geojson(f)
                    if parsed:
                        key = Path(f.name).name
                        result["shapefiles"][key] = parsed
            elif group_name == "geotiff":
                for f in group_files:
                    parsed = self._parse_geotiff(f)
                    if parsed:
                        key = Path(f.name).name
                        result["raster_data"][key] = parsed

        return result

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

        # 错误
        for err in result.get("errors", []):
            parts.append(f"❌ {err}")

        if not parts:
            return ("📁 文件已收到，但无法识别的格式。\n\n"
                    "支持格式：SHP(.shp/.shx/.dbf/.prj) / GeoJSON(.geojson) / "
                    "KML(.kml) / Excel(.xlsx/.xls) / CSV(.csv) / "
                    "GeoTIFF(.tif) / ASCII(.asc) / ZIP压缩包(.zip)")
        return "📁 **文件解析结果：**\n" + "\n".join(parts)


# 全局实例
file_handler = FileUploadHandler()
