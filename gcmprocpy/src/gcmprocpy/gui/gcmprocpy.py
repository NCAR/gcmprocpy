import os
import sys
import logging
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import mplcursors
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QSplitter, QLineEdit, QPushButton, QCheckBox,
    QComboBox, QMessageBox, QFileDialog, QLabel, QSizePolicy, QStackedWidget,
    QScrollArea
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import xarray as xr
from datetime import datetime
from ..plot_gen import (plt_lev_var, plt_lat_lon, plt_lev_lat, plt_lev_lon,
                        plt_lev_time, plt_lat_time, plt_lon_time, plt_var_time)
from ..io import load_datasets, close_datasets, save_output
from ..data_parse import time_list, var_list, level_list, lon_list, lat_list

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: safe type conversion from QLineEdit / QComboBox text
# ---------------------------------------------------------------------------
def _float_or_none(text):
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None

def _int_or_none(text):
    text = text.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None

def _str_or_none(text):
    text = text.strip()
    return text if text else None


# ---------------------------------------------------------------------------
# Widget builder helpers – eliminates duplication across parameter pages
# ---------------------------------------------------------------------------
def _add_combo(layout, label):
    combo = QComboBox()
    layout.addRow(label, combo)
    return combo

def _add_line(layout, label, placeholder=""):
    line = QLineEdit()
    if placeholder:
        line.setPlaceholderText(placeholder)
    layout.addRow(label, line)
    return line

def _add_check(layout, label, default=False):
    cb = QCheckBox()
    cb.setChecked(default)
    layout.addRow(label, cb)
    return cb


# ---------------------------------------------------------------------------
# Reusable widget groups – common parameter sets shared across pages
# ---------------------------------------------------------------------------
def _add_contour_widgets(layout):
    """Add contour interval, value, symmetric, colormap, cmap limits, line color."""
    w = {}
    w['contour_intervals'] = _add_line(layout, "Contour Intervals:")
    w['contour_value'] = _add_line(layout, "Contour Value:")
    w['symmetric'] = _add_check(layout, "Symmetric Interval:")
    w['cmap'] = _add_line(layout, "Colormap:")
    w['cmap_min'] = _add_line(layout, "Colormap Min:")
    w['cmap_max'] = _add_line(layout, "Colormap Max:")
    w['line_color'] = _add_line(layout, "Line Color:")
    return w

def _add_level_bounds(layout):
    w = {}
    w['level_min'] = _add_line(layout, "Level Min:")
    w['level_max'] = _add_line(layout, "Level Max:")
    return w

def _add_lat_bounds(layout):
    w = {}
    w['lat_min'] = _add_line(layout, "Latitude Min:")
    w['lat_max'] = _add_line(layout, "Latitude Max:")
    return w

def _add_lon_bounds(layout):
    w = {}
    w['lon_min'] = _add_line(layout, "Longitude Min:")
    w['lon_max'] = _add_line(layout, "Longitude Max:")
    return w

def _add_mtime_bounds(layout):
    w = {}
    w['mtime_min'] = _add_line(layout, "mtime Min:")
    w['mtime_max'] = _add_line(layout, "mtime Max:")
    return w


def _get_contour_params(w):
    """Extract contour parameter dict from widget group."""
    return {
        'contour_intervals': _int_or_none(w['contour_intervals'].text()),
        'contour_value': _int_or_none(w['contour_value'].text()),
        'symmetric_interval': w['symmetric'].isChecked(),
        'cmap_color': _str_or_none(w['cmap'].text()),
        'cmap_lim_min': _float_or_none(w['cmap_min'].text()),
        'cmap_lim_max': _float_or_none(w['cmap_max'].text()),
        'line_color': _str_or_none(w['line_color'].text()),
    }

def _update_contour_placeholders(w, returned):
    """Update placeholder text from values returned by plot functions."""
    mapping = {
        'contour_intervals': 'contour_intervals',
        'contour_value': 'contour_value',
        'cmap': 'cmap_color',
        'cmap_min': 'cmap_lim_min',
        'cmap_max': 'cmap_lim_max',
        'line_color': 'line_color',
    }
    for widget_key, param_key in mapping.items():
        if param_key in returned and returned[param_key] is not None:
            w[widget_key].setPlaceholderText(str(returned[param_key]))
    if 'symmetric_interval' in returned:
        w['symmetric'].setChecked(returned['symmetric_interval'])


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    """GCMProcPy main application window."""

    PLOT_TYPES = [
        "Lat vs Lon",
        "Lev vs Var",
        "Lev vs Lon",
        "Lev vs Lat",
        "Lev vs Time",
        "Lat vs Time",
        "Lon vs Time",
        "Var vs Time",
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GCMProcPy")
        self.resize(1600, 600)

        # Main splitter: left (controls) | right (plot)
        self.splitter = QSplitter(Qt.Horizontal)

        # --- Left Panel: Controls (scrollable) ---
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)

        # Dataset Group
        self.dataset_group = QGroupBox("Dataset")
        ds_layout = QFormLayout()
        dir_row = QHBoxLayout()
        self.directory_input = QLineEdit()
        self.directory_input.setPlaceholderText("Enter directory or file path")
        dir_row.addWidget(self.directory_input)
        self.browse_dir_button = QPushButton("Browse Dir")
        self.browse_dir_button.clicked.connect(self.on_browse_directory)
        dir_row.addWidget(self.browse_dir_button)
        self.browse_file_button = QPushButton("Browse File")
        self.browse_file_button.clicked.connect(self.on_browse_file)
        dir_row.addWidget(self.browse_file_button)
        ds_layout.addRow("Directory:", dir_row)
        self.dataset_filter_input = QLineEdit()
        self.dataset_filter_input.setPlaceholderText("Optional filter (e.g., 'sech')")
        ds_layout.addRow("Dataset Filter:", self.dataset_filter_input)
        self.load_button = QPushButton("Load Datasets")
        self.load_button.clicked.connect(self.on_load_datasets)
        ds_layout.addRow(self.load_button)
        self.dataset_group.setLayout(ds_layout)
        self.controls_layout.addWidget(self.dataset_group)

        # Plot Type Group
        self.plot_type_group = QGroupBox("Plot Type")
        pt_layout = QHBoxLayout()
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(self.PLOT_TYPES)
        self.plot_type_combo.currentIndexChanged.connect(self.on_plot_type_changed)
        pt_layout.addWidget(self.plot_type_combo)
        self.plot_type_group.setLayout(pt_layout)
        self.controls_layout.addWidget(self.plot_type_group)

        # Parameter pages via QStackedWidget
        self.param_stack = QStackedWidget()
        self._pages = {}
        self._create_all_pages()
        self.controls_layout.addWidget(self.param_stack)

        # Buttons
        button_layout = QHBoxLayout()
        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.on_plot)
        button_layout.addWidget(self.plot_button)
        self.save_button = QPushButton("Save as Image")
        self.save_button.clicked.connect(self.on_save_image)
        button_layout.addWidget(self.save_button)
        self.controls_layout.addLayout(button_layout)
        self.controls_layout.addStretch()

        # Wrap controls in a scroll area
        scroll = QScrollArea()
        scroll.setWidget(self.controls_widget)
        scroll.setWidgetResizable(True)

        # --- Right Panel: Plot Display ---
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.fig = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_layout.addWidget(self.canvas)

        # Assemble splitter
        self.splitter.addWidget(scroll)
        self.splitter.addWidget(self.plot_widget)
        self.splitter.setSizes([300, 900])
        self.setCentralWidget(self.splitter)

        # State
        self.datasets = []
        self.selected_dataset = None

    # ------------------------------------------------------------------
    # Page creation – one method per plot type, using shared helpers
    # ------------------------------------------------------------------
    def _create_all_pages(self):
        creators = [
            ("Lat vs Lon",  self._create_lat_lon_page),
            ("Lev vs Var",  self._create_lev_var_page),
            ("Lev vs Lon",  self._create_lev_lon_page),
            ("Lev vs Lat",  self._create_lev_lat_page),
            ("Lev vs Time", self._create_lev_time_page),
            ("Lat vs Time", self._create_lat_time_page),
            ("Lon vs Time", self._create_lon_time_page),
            ("Var vs Time", self._create_var_time_page),
        ]
        for name, creator in creators:
            page = creator()
            self._pages[name] = page
            self.param_stack.addWidget(page['widget'])

    def _create_lat_lon_page(self):
        page = QWidget(); layout = QFormLayout(page); w = {}
        w['variable'] = _add_combo(layout, "Variable Name:")
        w['time'] = _add_combo(layout, "Time (ISO):")
        w['level'] = _add_combo(layout, "Level:")
        w['unit'] = _add_line(layout, "Variable Unit:")
        w['projection'] = QComboBox()
        w['projection'].addItems(['mercator', 'orthographic', 'mollweide', 'north_polar', 'south_polar', 'polar'])
        layout.addRow("Projection:", w['projection'])
        w['center_lon'] = _add_line(layout, "Center Longitude:")
        w['central_lat'] = _add_line(layout, "Central Latitude:")
        w.update(_add_contour_widgets(layout))
        w['coastlines'] = _add_check(layout, "Coastlines:")
        w['nightshade'] = _add_check(layout, "Nightshade:")
        w['gm_equator'] = _add_check(layout, "GM Equator:")
        w['wind'] = _add_check(layout, "Wind Vectors:")
        w['wind_density'] = _add_line(layout, "Wind Density:", "15")
        w['wind_scale'] = _add_line(layout, "Wind Scale:")
        w['wind_color'] = _add_line(layout, "Wind Color:", "black")
        w.update(_add_lat_bounds(layout))
        w.update(_add_lon_bounds(layout))
        w['clean'] = _add_check(layout, "Clean Plot:", True)
        return {'widget': page, 'widgets': w}

    def _create_lev_var_page(self):
        page = QWidget(); layout = QFormLayout(page); w = {}
        w['variable'] = _add_combo(layout, "Variable Name:")
        w['latitude'] = _add_combo(layout, "Latitude:")
        w['time'] = _add_combo(layout, "Time (ISO):")
        w['longitude'] = _add_combo(layout, "Longitude:")
        w['unit'] = _add_line(layout, "Variable Unit:")
        w.update(_add_level_bounds(layout))
        w['log_level'] = _add_check(layout, "Log Level:", True)
        w['clean'] = _add_check(layout, "Clean Plot:", True)
        return {'widget': page, 'widgets': w}

    def _create_lev_lon_page(self):
        page = QWidget(); layout = QFormLayout(page); w = {}
        w['variable'] = _add_combo(layout, "Variable Name:")
        w['latitude'] = _add_combo(layout, "Latitude:")
        w['time'] = _add_combo(layout, "Time (ISO):")
        w['log_level'] = _add_check(layout, "Log Level:", True)
        w['unit'] = _add_line(layout, "Variable Unit:")
        w.update(_add_contour_widgets(layout))
        w.update(_add_level_bounds(layout))
        w.update(_add_lon_bounds(layout))
        w['clean'] = _add_check(layout, "Clean Plot:", True)
        return {'widget': page, 'widgets': w}

    def _create_lev_lat_page(self):
        page = QWidget(); layout = QFormLayout(page); w = {}
        w['variable'] = _add_combo(layout, "Variable Name:")
        w['time'] = _add_combo(layout, "Time (ISO):")
        w['longitude'] = _add_combo(layout, "Longitude:")
        w['log_level'] = _add_check(layout, "Log Level:", True)
        w['unit'] = _add_line(layout, "Variable Unit:")
        w.update(_add_contour_widgets(layout))
        w.update(_add_level_bounds(layout))
        w.update(_add_lat_bounds(layout))
        w['clean'] = _add_check(layout, "Clean Plot:", True)
        return {'widget': page, 'widgets': w}

    def _create_lev_time_page(self):
        page = QWidget(); layout = QFormLayout(page); w = {}
        w['variable'] = _add_combo(layout, "Variable Name:")
        w['latitude'] = _add_combo(layout, "Latitude:")
        w['longitude'] = _add_combo(layout, "Longitude:")
        w['log_level'] = _add_check(layout, "Log Level:", True)
        w['unit'] = _add_line(layout, "Variable Unit:")
        w.update(_add_contour_widgets(layout))
        w.update(_add_level_bounds(layout))
        w.update(_add_mtime_bounds(layout))
        w['clean'] = _add_check(layout, "Clean Plot:", True)
        return {'widget': page, 'widgets': w}

    def _create_lat_time_page(self):
        page = QWidget(); layout = QFormLayout(page); w = {}
        w['variable'] = _add_combo(layout, "Variable Name:")
        w['level'] = _add_combo(layout, "Level:")
        w['longitude'] = _add_combo(layout, "Longitude:")
        w['unit'] = _add_line(layout, "Variable Unit:")
        w.update(_add_contour_widgets(layout))
        w.update(_add_lat_bounds(layout))
        w.update(_add_mtime_bounds(layout))
        w['clean'] = _add_check(layout, "Clean Plot:", True)
        return {'widget': page, 'widgets': w}

    def _create_lon_time_page(self):
        page = QWidget(); layout = QFormLayout(page); w = {}
        w['variable'] = _add_combo(layout, "Variable Name:")
        w['latitude'] = _add_combo(layout, "Latitude:")
        w['level'] = _add_combo(layout, "Level:")
        w['unit'] = _add_line(layout, "Variable Unit:")
        w.update(_add_contour_widgets(layout))
        w.update(_add_lon_bounds(layout))
        w.update(_add_mtime_bounds(layout))
        w['clean'] = _add_check(layout, "Clean Plot:", True)
        return {'widget': page, 'widgets': w}

    def _create_var_time_page(self):
        page = QWidget(); layout = QFormLayout(page); w = {}
        w['variable'] = _add_combo(layout, "Variable Name:")
        w['latitude'] = _add_combo(layout, "Latitude:")
        w['longitude'] = _add_combo(layout, "Longitude:")
        w['level'] = _add_combo(layout, "Level:")
        w['unit'] = _add_line(layout, "Variable Unit:")
        w.update(_add_mtime_bounds(layout))
        w['clean'] = _add_check(layout, "Clean Plot:", True)
        return {'widget': page, 'widgets': w}

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def on_browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if directory:
            self.directory_input.setText(directory)

    def on_browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select NetCDF File", "",
            "NetCDF Files (*.nc *.nc4 *.h5);;All Files (*)")
        if file_path:
            self.directory_input.setText(file_path)

    def on_plot_type_changed(self, index):
        self.param_stack.setCurrentIndex(index)

    def on_load_datasets(self):
        directory = self.directory_input.text().strip()
        dataset_filter = self.dataset_filter_input.text().strip() or None
        if not directory:
            QMessageBox.warning(self, "Input Error", "Please enter a valid directory or file path.")
            return
        try:
            # Close previously loaded datasets
            if self.datasets:
                try:
                    close_datasets(self.datasets)
                except Exception:
                    pass
            self.datasets = load_datasets(directory, dataset_filter=dataset_filter)
            if not self.datasets:
                QMessageBox.warning(self, "No Datasets", "No valid NetCDF datasets were found.")
                return
            self.selected_dataset = self.datasets
            QMessageBox.information(self, "Datasets Loaded", f"Loaded {len(self.datasets)} dataset(s).")
            self._populate_all_combos()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load datasets:\n{e}")

    def _populate_all_combos(self):
        """Populate combo boxes across all pages from loaded datasets."""
        valid_vars = var_list(self.selected_dataset)
        valid_times = time_list(self.selected_dataset)
        valid_lats = lat_list(self.selected_dataset)
        valid_lons = lon_list(self.selected_dataset)
        valid_levels = level_list(self.selected_dataset)

        str_vars = [str(v) for v in valid_vars]
        str_times = [str(t) for t in valid_times]
        str_lats = [str(lat) for lat in valid_lats]
        str_lons = [str(lon) for lon in valid_lons]
        str_levels = [str(lev) for lev in valid_levels]

        combo_map = {
            'variable': str_vars,
            'time': str_times,
            'latitude': str_lats,
            'longitude': str_lons,
            'level': str_levels,
        }

        for page_data in self._pages.values():
            w = page_data['widgets']
            for key, items in combo_map.items():
                if key in w and isinstance(w[key], QComboBox):
                    w[key].clear()
                    w[key].addItems(items)

    # ------------------------------------------------------------------
    # Plot dispatch
    # ------------------------------------------------------------------
    def on_plot(self):
        if not self.selected_dataset:
            QMessageBox.warning(self, "No Data", "Please load datasets first.")
            return

        plot_type = self.plot_type_combo.currentText()
        dispatch = {
            "Lat vs Lon":  self._plot_lat_lon,
            "Lev vs Var":  self._plot_lev_var,
            "Lev vs Lon":  self._plot_lev_lon,
            "Lev vs Lat":  self._plot_lev_lat,
            "Lev vs Time": self._plot_lev_time,
            "Lat vs Time": self._plot_lat_time,
            "Lon vs Time": self._plot_lon_time,
            "Var vs Time": self._plot_var_time,
        }

        handler = dispatch.get(plot_type)
        if handler is None:
            QMessageBox.warning(self, "Plot Type Error", "Unknown plot type selected.")
            return

        try:
            fig = handler()
        except Exception as e:
            logger.exception("Plot generation failed")
            QMessageBox.critical(self, "Plot Error", f"Could not generate plot:\n{e}")
            return

        if fig is not None:
            self._update_canvas(fig)

    def _update_canvas(self, fig):
        """Replace the canvas figure and redraw (reuse canvas widget)."""
        self.plot_layout.removeWidget(self.canvas)
        self.canvas.deleteLater()
        self.canvas = FigureCanvas(fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_layout.addWidget(self.canvas)
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Per-type plot handlers
    # ------------------------------------------------------------------
    def _plot_lat_lon(self):
        w = self._pages["Lat vs Lon"]['widgets']
        params = {
            "datasets": self.selected_dataset,
            "variable_name": w['variable'].currentText(),
            "time": w['time'].currentText(),
            "level": w['level'].currentText(),
            "variable_unit": _str_or_none(w['unit'].text()),
            "projection": w['projection'].currentText(),
            "center_longitude": _float_or_none(w['center_lon'].text()) or 0,
            "central_latitude": _float_or_none(w['central_lat'].text()) or 0,
            "coastlines": w['coastlines'].isChecked(),
            "nightshade": w['nightshade'].isChecked(),
            "gm_equator": w['gm_equator'].isChecked(),
            "wind": w['wind'].isChecked(),
            "wind_density": _int_or_none(w['wind_density'].text()) or 15,
            "wind_scale": _float_or_none(w['wind_scale'].text()),
            "wind_color": _str_or_none(w['wind_color'].text()) or 'black',
            "latitude_minimum": _float_or_none(w['lat_min'].text()),
            "latitude_maximum": _float_or_none(w['lat_max'].text()),
            "longitude_minimum": _float_or_none(w['lon_min'].text()),
            "longitude_maximum": _float_or_none(w['lon_max'].text()),
            "clean_plot": w['clean'].isChecked(),
        }
        params.update(_get_contour_params(w))

        result = plt_lat_lon(**params)

        # Non-mercator projections return just the figure; mercator returns a tuple
        if not isinstance(result, tuple):
            return result

        (fig, variable_unit, center_longitude, contour_intervals, contour_value,
         symmetric_interval, cmap_color, cmap_lim_min, cmap_lim_max, line_color,
         latitude_minimum, latitude_maximum, longitude_minimum, longitude_maximum,
         contour_filled, unique_lons, unique_lats, variable_values) = result

        # Update placeholders
        w['unit'].setPlaceholderText(str(variable_unit))
        w['center_lon'].setPlaceholderText(str(center_longitude))
        _update_contour_placeholders(w, {
            'contour_intervals': contour_intervals, 'contour_value': contour_value,
            'symmetric_interval': symmetric_interval, 'cmap_color': cmap_color,
            'cmap_lim_min': cmap_lim_min, 'cmap_lim_max': cmap_lim_max,
            'line_color': line_color,
        })
        w['lat_min'].setPlaceholderText(str(latitude_minimum))
        w['lat_max'].setPlaceholderText(str(latitude_maximum))
        w['lon_min'].setPlaceholderText(str(longitude_minimum))
        w['lon_max'].setPlaceholderText(str(longitude_maximum))

        # Interactive cursor
        cursor = mplcursors.cursor(contour_filled, hover=True)
        clon = params["center_longitude"]
        var_name = params["variable_name"]
        @cursor.connect("add")
        def on_add(sel):
            x, y = sel.target
            if (x + clon) > 180:
                adj = -(360 - x - clon)
            elif (x + clon) < -180:
                adj = x + 360 + clon
            else:
                adj = x + clon
            lon_idx = (np.abs(unique_lons - adj)).argmin()
            lat_idx = (np.abs(unique_lats - y)).argmin()
            value = variable_values[lat_idx, lon_idx]
            sel.annotation.set(
                text=f"Lon: {unique_lons[lon_idx]:.2f}\nLat: {unique_lats[lat_idx]:.2f}\n{var_name}: {value:.2e} {variable_unit}")
            sel.annotation.get_bbox_patch().set(alpha=0.9)

        return fig

    def _plot_lev_var(self):
        w = self._pages["Lev vs Var"]['widgets']
        params = {
            "datasets": self.selected_dataset,
            "variable_name": w['variable'].currentText(),
            "latitude": _float_or_none(w['latitude'].currentText()),
            "time": w['time'].currentText(),
            "longitude": _float_or_none(w['longitude'].currentText()),
            "log_level": w['log_level'].isChecked(),
            "variable_unit": _str_or_none(w['unit'].text()),
            "level_minimum": _float_or_none(w['level_min'].text()),
            "level_maximum": _float_or_none(w['level_max'].text()),
            "clean_plot": w['clean'].isChecked(),
        }
        fig, variable_unit, level_minimum, level_maximum = plt_lev_var(**params)
        w['unit'].setPlaceholderText(str(variable_unit))
        w['level_min'].setPlaceholderText(str(level_minimum))
        w['level_max'].setPlaceholderText(str(level_maximum))

        var_name = params["variable_name"]
        cursor = mplcursors.cursor(fig, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            x, y = sel.target
            sel.annotation.set(
                text=f"Level: {y:.2f}\n{var_name}: {x:.2e} {variable_unit}")
            sel.annotation.get_bbox_patch().set(alpha=0.9)

        return fig

    def _plot_lev_lon(self):
        w = self._pages["Lev vs Lon"]['widgets']
        params = {
            "datasets": self.selected_dataset,
            "variable_name": w['variable'].currentText(),
            "latitude": _float_or_none(w['latitude'].currentText()),
            "time": w['time'].currentText(),
            "log_level": w['log_level'].isChecked(),
            "variable_unit": _str_or_none(w['unit'].text()),
            "level_minimum": _float_or_none(w['level_min'].text()),
            "level_maximum": _float_or_none(w['level_max'].text()),
            "longitude_minimum": _float_or_none(w['lon_min'].text()),
            "longitude_maximum": _float_or_none(w['lon_max'].text()),
            "clean_plot": w['clean'].isChecked(),
        }
        params.update(_get_contour_params(w))

        (fig, variable_unit, latitude, time, contour_intervals, contour_value,
         symmetric_interval, cmap_color, cmap_lim_min, cmap_lim_max, line_color,
         level_minimum, level_maximum, longitude_minimum, longitude_maximum,
         contour_filled, unique_lons, unique_levs, variable_values) = plt_lev_lon(**params)

        w['unit'].setPlaceholderText(str(variable_unit))
        _update_contour_placeholders(w, {
            'contour_intervals': contour_intervals, 'contour_value': contour_value,
            'symmetric_interval': symmetric_interval, 'cmap_color': cmap_color,
            'cmap_lim_min': cmap_lim_min, 'cmap_lim_max': cmap_lim_max,
            'line_color': line_color,
        })
        w['level_min'].setPlaceholderText(str(level_minimum))
        w['level_max'].setPlaceholderText(str(level_maximum))
        w['lon_min'].setPlaceholderText(str(longitude_minimum))
        w['lon_max'].setPlaceholderText(str(longitude_maximum))

        var_name = params["variable_name"]
        cursor = mplcursors.cursor(contour_filled, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            x, y = sel.target
            lon_idx = (np.abs(unique_lons - x)).argmin()
            level_idx = (np.abs(unique_levs - y)).argmin()
            value = variable_values[level_idx, lon_idx]
            sel.annotation.set(
                text=f"Lon: {unique_lons[lon_idx]:.2f}\nLev: {unique_levs[level_idx]:.2f}\n{var_name}: {value:.2e} {variable_unit}")
            sel.annotation.get_bbox_patch().set(alpha=0.9)

        return fig

    def _plot_lev_lat(self):
        w = self._pages["Lev vs Lat"]['widgets']
        params = {
            "datasets": self.selected_dataset,
            "variable_name": w['variable'].currentText(),
            "time": w['time'].currentText(),
            "longitude": _float_or_none(w['longitude'].currentText()),
            "log_level": w['log_level'].isChecked(),
            "variable_unit": _str_or_none(w['unit'].text()),
            "level_minimum": _float_or_none(w['level_min'].text()),
            "level_maximum": _float_or_none(w['level_max'].text()),
            "latitude_minimum": _float_or_none(w['lat_min'].text()),
            "latitude_maximum": _float_or_none(w['lat_max'].text()),
            "clean_plot": w['clean'].isChecked(),
        }
        params.update(_get_contour_params(w))

        (fig, variable_unit, time, contour_intervals, contour_value,
         symmetric_interval, cmap_color, cmap_lim_min, cmap_lim_max, line_color,
         level_minimum, level_maximum, latitude_minimum, latitude_maximum,
         contour_filled, unique_lats, unique_levs, variable_values) = plt_lev_lat(**params)

        w['unit'].setPlaceholderText(str(variable_unit))
        _update_contour_placeholders(w, {
            'contour_intervals': contour_intervals, 'contour_value': contour_value,
            'symmetric_interval': symmetric_interval, 'cmap_color': cmap_color,
            'cmap_lim_min': cmap_lim_min, 'cmap_lim_max': cmap_lim_max,
            'line_color': line_color,
        })
        w['level_min'].setPlaceholderText(str(level_minimum))
        w['level_max'].setPlaceholderText(str(level_maximum))
        w['lat_min'].setPlaceholderText(str(latitude_minimum))
        w['lat_max'].setPlaceholderText(str(latitude_maximum))

        var_name = params["variable_name"]
        cursor = mplcursors.cursor(contour_filled, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            x, y = sel.target
            lat_idx = (np.abs(unique_lats - x)).argmin()
            level_idx = (np.abs(unique_levs - y)).argmin()
            value = variable_values[level_idx, lat_idx]
            sel.annotation.set(
                text=f"Lat: {unique_lats[lat_idx]:.2f}\nLev: {unique_levs[level_idx]:.2f}\n{var_name}: {value:.2e} {variable_unit}")
            sel.annotation.get_bbox_patch().set(alpha=0.9)

        return fig

    def _plot_lev_time(self):
        w = self._pages["Lev vs Time"]['widgets']
        params = {
            "datasets": self.selected_dataset,
            "variable_name": w['variable'].currentText(),
            "latitude": _float_or_none(w['latitude'].currentText()),
            "longitude": _float_or_none(w['longitude'].currentText()),
            "log_level": w['log_level'].isChecked(),
            "variable_unit": _str_or_none(w['unit'].text()),
            "level_minimum": _float_or_none(w['level_min'].text()),
            "level_maximum": _float_or_none(w['level_max'].text()),
            "mtime_minimum": _str_or_none(w['mtime_min'].text()),
            "mtime_maximum": _str_or_none(w['mtime_max'].text()),
            "clean_plot": w['clean'].isChecked(),
        }
        params.update(_get_contour_params(w))
        result = plt_lev_time(**params)

        if not isinstance(result, tuple):
            return result

        (fig, variable_unit, contour_intervals, contour_value,
         symmetric_interval, cmap_color, cmap_lim_min, cmap_lim_max, line_color,
         level_minimum, level_maximum, contour_filled, unique_levs, variable_values) = result

        w['unit'].setPlaceholderText(str(variable_unit))
        _update_contour_placeholders(w, {
            'contour_intervals': contour_intervals, 'contour_value': contour_value,
            'symmetric_interval': symmetric_interval, 'cmap_color': cmap_color,
            'cmap_lim_min': cmap_lim_min, 'cmap_lim_max': cmap_lim_max,
            'line_color': line_color,
        })
        w['level_min'].setPlaceholderText(str(level_minimum))
        w['level_max'].setPlaceholderText(str(level_maximum))

        var_name = params["variable_name"]
        cursor = mplcursors.cursor(contour_filled, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            x, y = sel.target
            level_idx = (np.abs(unique_levs - y)).argmin()
            sel.annotation.set(
                text=f"Lev: {unique_levs[level_idx]:.2f}\n{var_name}: {variable_values[level_idx, int(round(x))]:.2e} {variable_unit}")
            sel.annotation.get_bbox_patch().set(alpha=0.9)

        return fig

    def _plot_lat_time(self):
        w = self._pages["Lat vs Time"]['widgets']
        params = {
            "datasets": self.selected_dataset,
            "variable_name": w['variable'].currentText(),
            "level": w['level'].currentText(),
            "longitude": _float_or_none(w['longitude'].currentText()),
            "variable_unit": _str_or_none(w['unit'].text()),
            "latitude_minimum": _float_or_none(w['lat_min'].text()),
            "latitude_maximum": _float_or_none(w['lat_max'].text()),
            "mtime_minimum": _str_or_none(w['mtime_min'].text()),
            "mtime_maximum": _str_or_none(w['mtime_max'].text()),
            "clean_plot": w['clean'].isChecked(),
        }
        params.update(_get_contour_params(w))
        result = plt_lat_time(**params)

        if not isinstance(result, tuple):
            return result

        (fig, variable_unit, contour_intervals, contour_value,
         symmetric_interval, cmap_color, cmap_lim_min, cmap_lim_max, line_color,
         latitude_minimum, latitude_maximum, contour_filled, unique_lats, variable_values) = result

        w['unit'].setPlaceholderText(str(variable_unit))
        _update_contour_placeholders(w, {
            'contour_intervals': contour_intervals, 'contour_value': contour_value,
            'symmetric_interval': symmetric_interval, 'cmap_color': cmap_color,
            'cmap_lim_min': cmap_lim_min, 'cmap_lim_max': cmap_lim_max,
            'line_color': line_color,
        })
        w['lat_min'].setPlaceholderText(str(latitude_minimum))
        w['lat_max'].setPlaceholderText(str(latitude_maximum))

        var_name = params["variable_name"]
        cursor = mplcursors.cursor(contour_filled, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            x, y = sel.target
            lat_idx = (np.abs(unique_lats - y)).argmin()
            sel.annotation.set(
                text=f"Lat: {unique_lats[lat_idx]:.2f}\n{var_name}: {variable_values[lat_idx, int(round(x))]:.2e} {variable_unit}")
            sel.annotation.get_bbox_patch().set(alpha=0.9)

        return fig

    def _plot_lon_time(self):
        w = self._pages["Lon vs Time"]['widgets']
        params = {
            "datasets": self.selected_dataset,
            "variable_name": w['variable'].currentText(),
            "latitude": _float_or_none(w['latitude'].currentText()),
            "level": _float_or_none(w['level'].currentText()),
            "variable_unit": _str_or_none(w['unit'].text()),
            "longitude_minimum": _float_or_none(w['lon_min'].text()),
            "longitude_maximum": _float_or_none(w['lon_max'].text()),
            "mtime_minimum": _str_or_none(w['mtime_min'].text()),
            "mtime_maximum": _str_or_none(w['mtime_max'].text()),
            "clean_plot": w['clean'].isChecked(),
        }
        params.update(_get_contour_params(w))
        result = plt_lon_time(**params)

        if not isinstance(result, tuple):
            return result

        (fig, variable_unit, contour_intervals, contour_value,
         symmetric_interval, cmap_color, cmap_lim_min, cmap_lim_max, line_color,
         longitude_minimum, longitude_maximum, contour_filled, unique_lons, variable_values) = result

        w['unit'].setPlaceholderText(str(variable_unit))
        _update_contour_placeholders(w, {
            'contour_intervals': contour_intervals, 'contour_value': contour_value,
            'symmetric_interval': symmetric_interval, 'cmap_color': cmap_color,
            'cmap_lim_min': cmap_lim_min, 'cmap_lim_max': cmap_lim_max,
            'line_color': line_color,
        })
        w['lon_min'].setPlaceholderText(str(longitude_minimum))
        w['lon_max'].setPlaceholderText(str(longitude_maximum))

        var_name = params["variable_name"]
        cursor = mplcursors.cursor(contour_filled, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            x, y = sel.target
            lon_idx = (np.abs(unique_lons - y)).argmin()
            sel.annotation.set(
                text=f"Lon: {unique_lons[lon_idx]:.2f}\n{var_name}: {variable_values[lon_idx, int(round(x))]:.2e} {variable_unit}")
            sel.annotation.get_bbox_patch().set(alpha=0.9)

        return fig

    def _plot_var_time(self):
        w = self._pages["Var vs Time"]['widgets']
        params = {
            "datasets": self.selected_dataset,
            "variable_name": w['variable'].currentText(),
            "latitude": _float_or_none(w['latitude'].currentText()),
            "longitude": _float_or_none(w['longitude'].currentText()),
            "level": _float_or_none(w['level'].currentText()),
            "variable_unit": _str_or_none(w['unit'].text()),
            "mtime_minimum": _str_or_none(w['mtime_min'].text()),
            "mtime_maximum": _str_or_none(w['mtime_max'].text()),
            "clean_plot": w['clean'].isChecked(),
        }
        result = plt_var_time(**params)

        if not isinstance(result, tuple):
            return result

        fig, variable_unit = result
        w['unit'].setPlaceholderText(str(variable_unit))
        return fig

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def on_save_image(self):
        if self.canvas is None or self.canvas.figure is None:
            QMessageBox.warning(self, "Save Error", "No plot available to save.")
            return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot As Image", "",
            "PNG Image (*.png);;JPEG Image (*.jpg);;PDF (*.pdf);;All Files (*)",
            options=options)
        if file_path:
            try:
                self.canvas.figure.savefig(file_path)
                QMessageBox.information(self, "Save Successful", f"Plot saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Could not save plot:\n{e}")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
