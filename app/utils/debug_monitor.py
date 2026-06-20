"""
调试监控器 — 实时追踪管道每一步执行状态和错误位置

用法：
    from app.utils.debug_monitor import DebugMonitor, trace

    monitor = DebugMonitor()

    @monitor.trace("Span1: NER提取")
    def extract_entities(...):
        ...

输出：logs/debug_<timestamp>.log 和 Gradio 状态面板
"""
import sys
import os
import time
import traceback
import functools
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable


class DebugMonitor:
    """管道调试监控器（单例）"""

    _instance: Optional["DebugMonitor"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._log_entries: list = []  # 最近的日志条目（用于 UI 展示）
        self._max_entries = 200
        self._log_file: Optional[Path] = None
        self._enabled = True
        self._indent = 0
        self._start_time = time.time()

        # 创建日志目录和文件
        logs_dir = Path(__file__).parent.parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = logs_dir / f"debug_{timestamp}.log"

        self._write_header()

    def _write_header(self):
        with open(self._log_file, "w", encoding="utf-8") as f:
            f.write(f"{'='*80}\n")
            f.write(f"调试监控日志 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Python: {sys.version}\n")
            f.write(f"{'='*80}\n\n")

    # ================================================================
    # 核心日志方法
    # ================================================================

    def _log(self, level: str, msg: str):
        elapsed = time.time() - self._start_time
        prefix = "  " * self._indent
        line = f"[{elapsed:7.3f}s] {level:6s} {prefix}{msg}"

        # 写入文件
        if self._log_file:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

        # 存入内存（用于 UI）
        self._log_entries.append(line)
        if len(self._log_entries) > self._max_entries:
            self._log_entries = self._log_entries[-self._max_entries:]

    def info(self, msg: str):
        self._log("INFO", msg)

    def step(self, msg: str):
        """管道步骤标记"""
        self._log("STEP", f"▶ {msg}")

    def ok(self, msg: str):
        self._log("OK", f"  ✓ {msg}")

    def warn(self, msg: str):
        self._log("WARN", f"  ⚠ {msg}")

    def error(self, msg: str, exc_info: bool = False):
        self._log("ERROR", f"  ✗ {msg}")
        if exc_info:
            tb = traceback.format_exc()
            for line in tb.splitlines():
                self._log("TRACE", f"    {line}")
            # 提取文件:行号信息到错误消息
            tb_lines = tb.splitlines()
            for line in tb_lines:
                stripped = line.strip()
                if stripped.startswith("File ") and ", line " in stripped:
                    self._log("LOCATION", f"  📍 {stripped}")
                    break

    def capture_error(self, e: Exception, context: str = ""):
        """捕获异常并输出详细位置信息"""
        tb = traceback.format_exc()
        # 提取最后一个有意义的 File 行作为错误位置
        tb_lines = tb.splitlines()
        location = ""
        for line in reversed(tb_lines):
            stripped = line.strip()
            if stripped.startswith("File "):
                # 提取文件名和行号
                parts = stripped.split(", line ")
                if len(parts) == 2:
                    fname = parts[0].replace('File "', "").replace('"', "")
                    lineno = parts[1].split(",")[0]
                    fname_short = fname.split("groundwater-llm-modeling")[-1] if "groundwater-llm-modeling" in fname else fname.split("/")[-1]
                    location = f"{fname_short}:{lineno}"
                break
        # 提取最后一行的异常类型和消息
        last_line = tb_lines[-1].strip() if tb_lines else str(e)
        if context:
            self._log("ERROR", f"  ✗ [{context}] {type(e).__name__}: {e}")
        else:
            self._log("ERROR", f"  ✗ {type(e).__name__}: {e}")
        if location:
            self._log("LOCATION", f"  📍 错误位置: {location}")
        self._log("DETAIL", f"  └─ {last_line}")
        self._log("HINT", f"  💡 完整堆栈见日志文件: {self._log_file}")

    # ================================================================
    # 装饰器：自动追踪函数执行
    # ================================================================

    def trace(self, label: str, log_args: bool = False):
        """装饰器：追踪函数进入/退出/异常"""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self._enabled:
                    return func(*args, **kwargs)

                # 进入
                self.step(f"进入 {label}")
                self._indent += 1

                if log_args:
                    arg_str = ", ".join(
                        [repr(a)[:80] for a in args[1:]]  # skip self
                        + [f"{k}={repr(v)[:80]}" for k, v in kwargs.items()]
                    )
                    self.info(f"参数: {arg_str}")

                t0 = time.time()
                try:
                    result = func(*args, **kwargs)
                    dt = time.time() - t0
                    self.ok(f"完成 {label} ({dt:.3f}s)")
                    return result
                except Exception as e:
                    dt = time.time() - t0
                    self.error(
                        f"异常 {label} ({dt:.3f}s): {type(e).__name__}: {e}",
                        exc_info=True,
                    )
                    raise
                finally:
                    self._indent -= 1

            return wrapper

        return decorator

    # ================================================================
    # 上下文管理器：追踪代码块
    # ================================================================

    def span(self, label: str):
        """上下文管理器：追踪代码块"""
        return _TraceSpan(self, label)

    # ================================================================
    # UI 集成
    # ================================================================

    def get_recent_logs(self, n: int = 50) -> str:
        """获取最近 N 条日志（用于 Gradio UI）"""
        return "\n".join(self._log_entries[-n:])

    def get_log_file_path(self) -> Optional[str]:
        """获取日志文件路径"""
        return str(self._log_file) if self._log_file else None

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False


class _TraceSpan:
    """上下文管理器：自动追踪代码块"""

    def __init__(self, monitor: DebugMonitor, label: str):
        self._monitor = monitor
        self._label = label
        self._t0 = 0.0

    def __enter__(self):
        if self._monitor._enabled:
            self._monitor.step(f"进入 {self._label}")
            self._monitor._indent += 1
            self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._monitor._enabled:
            dt = time.time() - self._t0
            if exc_type is None:
                self._monitor.ok(f"完成 {self._label} ({dt:.3f}s)")
            else:
                self._monitor.error(
                    f"异常 {self._label} ({dt:.3f}s): {exc_type.__name__}: {exc_val}",
                    exc_info=True,
                )
            self._monitor._indent -= 1
        return False  # 不抑制异常


# ================================================================
# 全局实例
# ================================================================

monitor = DebugMonitor()


# ================================================================
# 便捷函数：快速添加追踪到关键管道函数
# ================================================================

def install_pipeline_traces():
    """
    自动为关键管道函数安装追踪（仅在调试模式调用）
    用法：在 gradio_app.py 启动时调用 install_pipeline_traces()
    """
    try:
        # ---- LLM Agent 管道 ----
        from app.llm_agent.intent_parser import IntentParser

        _orig_classify = IntentParser.classify_intent
        @functools.wraps(_orig_classify)
        def _traced_classify(self, user_msg, conv_history="", **kw):
            with monitor.span(f"classify_intent: {user_msg[:40]}..."):
                return _orig_classify(self, user_msg, conv_history, **kw)
        IntentParser.classify_intent = _traced_classify

        _orig_extract = IntentParser.extract_entities
        @functools.wraps(_orig_extract)
        def _traced_extract(self, user_msg, current_ir=None, conv_history="", **kw):
            with monitor.span(f"extract_entities: {user_msg[:40]}..."):
                return _orig_extract(self, user_msg, current_ir, conv_history, **kw)
        IntentParser.extract_entities = _traced_extract

        _orig_gen_ir = IntentParser.generate_json_ir
        @functools.wraps(_orig_gen_ir)
        def _traced_gen_ir(self, user_msg, patched_ir=None, entities=None,
                           normalized=None, conv_history="", **kw):
            with monitor.span(f"generate_json_ir: {user_msg[:40]}..."):
                return _orig_gen_ir(self, user_msg, patched_ir, entities,
                                    normalized, conv_history, **kw)
        IntentParser.generate_json_ir = _traced_gen_ir

        # ---- 物理校验 ----
        from app.physics_engine.validator import quick_validate

        _orig_qv = quick_validate
        @functools.wraps(_orig_qv)
        def _traced_qv(json_ir):
            with monitor.span("quick_validate"):
                project = json_ir.get("project", {}).get("name", "?")
                monitor.info(f"项目: {project}")
                result = _orig_qv(json_ir)
                monitor.info(
                    f"结果: valid={result.get('valid')}, "
                    f"errors={result.get('error_count')}, "
                    f"warnings={result.get('warning_count')}"
                )
                return result
        # 替换模块级函数
        import app.physics_engine.validator as vmod
        vmod.quick_validate = _traced_qv
        # 更新 Gradio 中的引用
        import app.frontend.gradio_app as gmod
        gmod.quick_validate = _traced_qv

        # ---- FloPy 建模 ----
        from app.modeling.model_builder import AutoModelBuilder

        _orig_build = AutoModelBuilder.build_and_run
        @functools.wraps(_orig_build)
        def _traced_build(self, json_ir, sim_name="auto_model"):
            with monitor.span(f"build_and_run: {sim_name}"):
                result = _orig_build(self, json_ir, sim_name)
                monitor.info(
                    f"success={result.get('success')}, "
                    f"head_range={result.get('head_range')}"
                )
                return result
        AutoModelBuilder.build_and_run = _traced_build

        monitor.info("已安装管道追踪 — 9 个关键函数")
        return True
    except Exception as e:
        monitor.warn(f"安装追踪失败: {e}")
        return False
