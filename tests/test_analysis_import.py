"""快速导入测试 — 分析模块"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from app.analysis.result_processor import ResultProcessor
from app.analysis.visualizer import VisualizationEngine
from app.analysis.process_identifier import ProcessIdentifier
from app.analysis.report_generator import ReportGenerator

# 1. ResultProcessor
rp = ResultProcessor()
print("1. ResultProcessor OK")

# 2. VisualizationEngine (no display)
viz = VisualizationEngine()
head = np.random.rand(40, 50) * 5 + 95
fig = viz.plot_head_contour(head, np.full(50, 20.0), np.full(40, 20.0), "Test")
assert fig is not None
print("2. VisualizationEngine OK")

# 3. ProcessIdentifier
pi = ProcessIdentifier()
conc_hist = [np.random.rand(40, 50) * 10 for _ in range(5)]
vel_hist = [np.random.rand(40, 50) * 0.1 for _ in range(5)]
result = pi.identify_dominant_process(
    conc_hist, vel_hist,
    {"al": 10, "decay_rate": 0.001, "kd": 0, "porosity": 0.25}
)
dominant = result["dominant_process"]
print(f"3. ProcessIdentifier OK: dominant={dominant}")

# 4. ReportGenerator
rg = ReportGenerator()
test_ir = {
    "project": {"name": "test", "model_type": "groundwater_flow"},
    "grid": {}, "aquifer": {"parameters": {}}, "boundary_conditions": [],
    "domain": {"x_range": [0, 1000], "y_range": [0, 800]},
}
path = rg.generate(test_ir)
print(f"4. ReportGenerator OK: {path}")

print("\nAll analysis modules verified!")
