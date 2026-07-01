"""Headless GUI check for the Mag Lat vs Lon page.

Runs in an ISOLATED SUBPROCESS: the GUI module does ``matplotlib.use("QtAgg")``
at import, which would flip the global backend and break the in-process (Agg)
plot tests. The subprocess verifies the fragile combo-index -> parameter-stack
alignment invariant and that the magnetic handler runs end-to-end. Skips cleanly
when PySide6 / apexpy / an offscreen Qt platform is unavailable.
"""
import os
import subprocess
import sys

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("apexpy")

# Self-contained script executed in a fresh interpreter (clean matplotlib state).
_GUI_CHECK = r'''
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from gcmprocpy.gui.gcmprocpy import MainWindow
    win = MainWindow()
except Exception as exc:
    print("SKIP:", exc); raise SystemExit(3)

combo = win.plot_type_combo
entries = [combo.itemText(i) for i in range(combo.count())]
assert "Mag Lat vs Lon" in entries, "mag page missing from combo"
# every combo index must map positionally to its own stack page (no drift)
for i, name in enumerate(entries):
    combo.setCurrentIndex(i)
    assert win.param_stack.currentWidget() is win._pages[name]["widget"], name

w = win._pages["Mag Lat vs Lon"]["widgets"]
for key in ("variable", "level", "level_type", "unit", "grid", "clean",
            "mlat_min", "mlat_max", "mlon_min", "mlon_max",
            "contour_intervals", "cmap"):
    assert key in w, "missing widget " + key

# run the handler on the synthetic tiegcm dataset used by the test suite
import numpy as np, xarray as xr
from gcmprocpy.containers import ModelDataset
levs = np.array([-2.0, 0.0, 2.0, 5.0]); ilevs = levs + 0.25
lats = np.array([-60., -30., 0., 30., 60.]); lons = np.arange(-180., 180., 30.)
times = np.array(["2003-03-20T00:00:00"], dtype="datetime64[ns]")
shape = (1, len(levs), len(lats), len(lons))
rng = np.linspace(200., 1500., int(np.prod(shape))).reshape(shape)
zg = np.linspace(9.0e6, 1.0e8, int(np.prod(shape))).reshape(shape)   # cm -> ~90-1000 km
ds = xr.Dataset(
    {"TN": (["time","lev","lat","lon"], rng, {"units":"K","long_name":"NEUTRAL TEMPERATURE"}),
     "ZG": (["time","ilev","lat","lon"], zg, {"units":"cm","long_name":"GEOMETRIC HEIGHT"}),
     "mtime": (["time","mtimedim"], np.array([[79,0,0,0]]))},
    coords={"time": times, "lev": levs, "ilev": ilevs, "lat": lats, "lon": lons},
)
ds["lev"].attrs["units"] = "ln(p0/p)"
win.selected_dataset = [ModelDataset(ds=ds, filename="synthetic.nc", model="TIE-GCM")]
w["variable"].addItem("TN"); w["variable"].setCurrentText("TN")
w["level"].addItem("5.0"); w["level"].setCurrentText("5.0")
if "date" in w:
    w["date"].addItem("2003-03-20"); w["date"].setCurrentText("2003-03-20")
if "time_of_day" in w:
    w["time_of_day"].addItem("00:00:00"); w["time_of_day"].setCurrentText("00:00:00")
fig = win._plot_mag_lat_lon()
assert fig is not None and fig.axes[0].get_ylabel() == "Magnetic Latitude (Deg)"
print("GUI_MAG_OK")
'''


def test_mag_gui_page_isolated():
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, "-c", _GUI_CHECK],
                          capture_output=True, text=True, env=env, timeout=300)
    if proc.returncode == 3:
        pytest.skip("GUI could not initialise: " + proc.stdout.strip())
    assert proc.returncode == 0, (
        f"GUI check failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    assert "GUI_MAG_OK" in proc.stdout
