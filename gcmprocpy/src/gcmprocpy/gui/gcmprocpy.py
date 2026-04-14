import os
import sys
import logging
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import mplcursors
from PySide6.QtCore import (
    Qt, QSettings, QCoreApplication, QTimer
)
from PySide6.QtGui import QKeySequence, QShortcut, QAction, QColor, QBrush
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QSplitter, QLineEdit, QPushButton, QCheckBox,
    QComboBox, QMessageBox, QFileDialog, QLabel, QSizePolicy, QStackedWidget,
    QScrollArea, QCompleter, QProgressBar, QSlider, QMenu, QToolButton
)
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
import xarray as xr
from datetime import datetime
from ..plot_gen import (plt_lev_var, plt_lat_lon, plt_lev_lat, plt_lev_lon,
                        plt_lev_time, plt_lat_time, plt_lon_time, plt_var_time)
from ..io import load_datasets, close_datasets, save_output
from ..data_parse import time_list, var_list, level_list, lon_list, lat_list
from ..containers import get_species_names, MODEL_DEFAULTS, clear_derived_cache

logger = logging.getLogger(__name__)

ORG_NAME = "NCAR"
APP_NAME = "gcmprocpy"
RECENT_FILES_MAX = 10

# Derived variables exposed in the variable dropdown.
# Requirements are species keys (resolved via get_species_names), wind_u/wind_v
# keys from MODEL_DEFAULTS, or the sentinel 'W_or_OMEGA' (either must exist).
_DERIVED_VARS = [
    ('NO53',     ['temp', 'o', 'no']),
    ('CO215',    ['temp', 'o', 'co2']),
    ('OH83',     ['temp', 'o', 'o2', 'n2']),
    ('OH_TOTAL', ['temp', 'o', 'o2', 'n2', 'h', 'o3', 'ho2']),
    ('OH_8_3',   ['temp', 'o', 'o2', 'n2', 'h', 'o3', 'ho2']),
    ('OH_9_4',   ['temp', 'o', 'o2', 'n2', 'h', 'o3', 'ho2']),
    ('EPVY',     ['temp', 'wind_u', 'wind_v']),
    ('EPVZ',     ['temp', 'wind_u', 'wind_v', 'W_or_OMEGA']),
    ('EPVDIV',   ['temp', 'wind_u', 'wind_v', 'W_or_OMEGA']),
]


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

def _get_level_params(w):
    """Read level and level_type from the level group widgets."""
    is_height = w['level_type'].isChecked()
    if is_height:
        return w['level_height'].text().strip() or None, 'height'
    return w['level'].currentText(), 'pressure'

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

def _add_filterable_combo(layout, label):
    """Editable combo with type-to-filter dropdown suggestions."""
    combo = QComboBox()
    combo.setEditable(True)
    combo.setInsertPolicy(QComboBox.NoInsert)
    layout.addRow(label, combo)
    return combo

def _setup_filterable(combo):
    """Configure the completer on a filterable combo after items are added."""
    completer = QCompleter(combo.model(), combo)
    completer.setFilterMode(Qt.MatchContains)
    completer.setCompletionMode(QCompleter.PopupCompletion)
    combo.setCompleter(completer)

def _add_time_group(layout):
    """Add date combo, filterable time combo, and summary label."""
    w = {}
    w['date'] = _add_combo(layout, "Date:")
    w['time_of_day'] = _add_filterable_combo(layout, "Time:")
    w['time_avail'] = QLabel("")
    w['time_avail'].setWordWrap(True)
    w['time_avail'].setStyleSheet("color: #838ba7; font-size: 11px; padding: 2px 0;")
    layout.addRow("", w['time_avail'])
    return w


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

def _add_level_group(layout):
    """Add level picker with toggle between pressure (filterable combo) and height (line edit)."""
    w = {}
    # Toggle switch: unchecked = pressure, checked = height
    w['level_type'] = QCheckBox("Height (km)")
    layout.addRow("Level Mode:", w['level_type'])
    # Pressure level: filterable combo + summary label
    w['level'] = QComboBox()
    w['level'].setEditable(True)
    w['level'].setInsertPolicy(QComboBox.NoInsert)
    w['_level_label'] = QLabel("Pressure Level:")
    layout.addRow(w['_level_label'], w['level'])
    w['level_avail'] = QLabel("")
    w['level_avail'].setWordWrap(True)
    w['level_avail'].setStyleSheet("color: #838ba7; font-size: 11px; padding: 2px 0;")
    layout.addRow("", w['level_avail'])
    # Height input: shown when toggle is checked
    w['level_height'] = QLineEdit()
    w['level_height'].setPlaceholderText("Height in km")
    w['_height_label'] = QLabel("Height (km):")
    layout.addRow(w['_height_label'], w['level_height'])
    w['level_height'].hide()
    w['_height_label'].hide()
    # Wire toggle
    def _on_toggle(checked):
        w['level'].setVisible(not checked)
        w['_level_label'].setVisible(not checked)
        w['level_avail'].setVisible(not checked)
        w['level_height'].setVisible(checked)
        w['_height_label'].setVisible(checked)
    w['level_type'].toggled.connect(_on_toggle)
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

        # Persistent user prefs (window geometry, last directory, recent files)
        self.settings = QSettings()
        self._plot_running = False

        # Accept file drops anywhere on the window
        self.setAcceptDrops(True)

        # HiDPI-aware figure DPI — pick from screen logical DPI, clamped
        screen = QApplication.primaryScreen()
        self._fig_dpi = int(max(80, min(screen.logicalDotsPerInch() if screen else 100, 150)))

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
        self.browse_button = QToolButton()
        self.browse_button.setText("Browse…")
        self.browse_button.setObjectName("browse_btn")
        self.browse_button.setPopupMode(QToolButton.InstantPopup)
        browse_menu = QMenu(self.browse_button)
        act_dir = browse_menu.addAction("Directory…")
        act_dir.triggered.connect(self.on_browse_directory)
        act_file = browse_menu.addAction("File…")
        act_file.triggered.connect(self.on_browse_file)
        self.browse_button.setMenu(browse_menu)
        dir_row.addWidget(self.browse_button)
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

        self.controls_layout.addStretch()

        # Wrap controls in a scroll area
        scroll = QScrollArea()
        scroll.setWidget(self.controls_widget)
        scroll.setWidgetResizable(True)

        # --- Right Panel: Plot Display ---
        self.plot_widget = QWidget()
        self.plot_widget.setObjectName("plot_pane")
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_layout.setContentsMargins(8, 8, 8, 8)
        self.fig = plt.figure(figsize=(10, 6), dpi=self._fig_dpi)
        self.fig.patch.set_facecolor('#e6e6e6')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self.plot_widget)
        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.canvas)

        # Timeline scrobber (shown only for single-time plot types)
        self._build_timeline()
        self.plot_layout.addWidget(self.timeline_widget)

        # Assemble splitter
        self.splitter.addWidget(scroll)
        self.splitter.addWidget(self.plot_widget)
        self.splitter.setSizes([300, 900])
        self.setCentralWidget(self.splitter)

        # Clipboard shortcuts — ensure Ctrl+V/C/X/A work in text fields
        self._setup_clipboard_shortcuts()

        # Menubar with File → Open / Recent / Save / Quit
        self._build_menubar()

        # Status bar with busy indicator for async plot dispatch
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)          # indeterminate
        self.progress.setMaximumWidth(140)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Ready")

        # State
        self.datasets = []
        self.selected_dataset = None

        # Restore persisted UI state
        self._restore_state()

    # ------------------------------------------------------------------
    # Clipboard shortcuts
    # ------------------------------------------------------------------
    def _setup_clipboard_shortcuts(self):
        """Wire up Ctrl+V/C/X/A so they reach the focused QLineEdit."""
        def _do_clipboard(action):
            w = QApplication.focusWidget()
            if isinstance(w, QLineEdit):
                getattr(w, action)()
        for key, action in [
            (QKeySequence.Paste, 'paste'),
            (QKeySequence.Copy, 'copy'),
            (QKeySequence.Cut, 'cut'),
            (QKeySequence.SelectAll, 'selectAll'),
        ]:
            sc = QShortcut(key, self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(lambda a=action: _do_clipboard(a))

    # ------------------------------------------------------------------
    # Menubar, recent files, persistence
    # ------------------------------------------------------------------
    def _build_menubar(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")

        open_dir = QAction("Open &Directory…", self)
        open_dir.setShortcut(QKeySequence.Open)
        open_dir.triggered.connect(self.on_browse_directory)
        file_menu.addAction(open_dir)

        open_file = QAction("Open &File…", self)
        open_file.triggered.connect(self.on_browse_file)
        file_menu.addAction(open_file)

        self.recent_menu = file_menu.addMenu("Open &Recent")
        self._update_recent_menu()

        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

    def _recent_files(self):
        val = self.settings.value("recent_files", [])
        if isinstance(val, str):
            val = [val]
        return [p for p in (val or []) if os.path.exists(p)]

    def _update_recent_menu(self):
        self.recent_menu.clear()
        files = self._recent_files()
        if not files:
            empty = QAction("(none)", self)
            empty.setEnabled(False)
            self.recent_menu.addAction(empty)
            return
        for path in files:
            act = QAction(path, self)
            act.triggered.connect(lambda _checked=False, p=path: self._load_from_path(p))
            self.recent_menu.addAction(act)
        self.recent_menu.addSeparator()
        clear = QAction("Clear Recent", self)
        clear.triggered.connect(self._clear_recent)
        self.recent_menu.addAction(clear)

    def _add_recent(self, path):
        files = self._recent_files()
        if path in files:
            files.remove(path)
        files.insert(0, path)
        self.settings.setValue("recent_files", files[:RECENT_FILES_MAX])
        self._update_recent_menu()

    def _clear_recent(self):
        self.settings.setValue("recent_files", [])
        self._update_recent_menu()

    def _load_from_path(self, path):
        """Populate directory field and trigger load. Shared by browse, drag-drop, recent."""
        self.directory_input.setText(path)
        self.on_load_datasets()

    def _restore_state(self):
        geom = self.settings.value("geometry")
        if geom is not None:
            self.restoreGeometry(geom)
        last_dir = self.settings.value("last_directory", "")
        if last_dir:
            self.directory_input.setText(last_dir)
        last_filter = self.settings.value("last_dataset_filter", "")
        if last_filter:
            self.dataset_filter_input.setText(last_filter)
        last_plot = self.settings.value("last_plot_type", "")
        if last_plot in self.PLOT_TYPES:
            self.plot_type_combo.setCurrentText(last_plot)

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("last_directory", self.directory_input.text().strip())
        self.settings.setValue("last_dataset_filter", self.dataset_filter_input.text().strip())
        self.settings.setValue("last_plot_type", self.plot_type_combo.currentText())
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Timeline scrobber
    # ------------------------------------------------------------------
    def _build_timeline(self):
        self.timeline_widget = QWidget()
        self.timeline_widget.setObjectName("timeline_bar")
        tl = QHBoxLayout(self.timeline_widget)
        tl.setContentsMargins(6, 4, 6, 4)

        self.tl_prev = QPushButton("◀")
        self.tl_prev.setObjectName("tl_btn")
        self.tl_prev.setFixedWidth(36)
        self.tl_prev.clicked.connect(lambda: self._step_time(-1))

        self.tl_next = QPushButton("▶")
        self.tl_next.setObjectName("tl_btn")
        self.tl_next.setFixedWidth(36)
        self.tl_next.clicked.connect(lambda: self._step_time(+1))

        self.tl_slider = QSlider(Qt.Horizontal)
        self.tl_slider.setRange(0, 0)
        self.tl_slider.setTracking(True)
        self.tl_slider.valueChanged.connect(self._on_timeline_changed)

        self.tl_label = QLabel("—")
        self.tl_label.setMinimumWidth(170)
        self.tl_label.setAlignment(Qt.AlignCenter)

        self.tl_autoplot = QCheckBox("Auto-plot")
        self.tl_autoplot.setToolTip("Replot automatically when the timeline changes")
        self.tl_autoplot.setChecked(True)

        self.plot_button = QPushButton("Plot")
        self.plot_button.setObjectName("plot_btn")
        self.plot_button.clicked.connect(self.on_plot)

        # Debounce: coalesce rapid slider ticks into a single plot call
        self._tl_plot_timer = QTimer(self)
        self._tl_plot_timer.setSingleShot(True)
        self._tl_plot_timer.setInterval(300)
        self._tl_plot_timer.timeout.connect(self._tl_fire_plot)

        # Scrubber controls: hidden for plot types without a single-time input
        self._scrubber_widgets = [self.tl_prev, self.tl_slider, self.tl_label,
                                  self.tl_next, self.tl_autoplot]
        tl.addWidget(self.tl_prev)
        tl.addWidget(self.tl_slider, 1)
        tl.addWidget(self.tl_label)
        tl.addWidget(self.tl_next)
        tl.addWidget(self.tl_autoplot)
        tl.addWidget(self.plot_button)

        # Left / Right arrow keys scrub the timeline (unless text widget is focused)
        for key, delta in ((Qt.Key_Left, -1), (Qt.Key_Right, +1)):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(lambda d=delta: self._keybd_step(d))

    def _keybd_step(self, delta):
        w = QApplication.focusWidget()
        if isinstance(w, (QLineEdit, QComboBox)):
            return
        if self.tl_slider.isVisible() and self.tl_slider.isEnabled():
            self._step_time(delta)

    def _step_time(self, delta):
        if self.tl_slider.maximum() <= 0:
            return
        new_val = max(0, min(self.tl_slider.maximum(), self.tl_slider.value() + delta))
        if new_val != self.tl_slider.value():
            self.tl_slider.setValue(new_val)

    def _current_page_has_time(self):
        page_name = self.plot_type_combo.currentText()
        w = self._pages.get(page_name, {}).get('widgets', {})
        return 'date' in w and 'time_of_day' in w

    def _update_timeline_visibility(self):
        vis = self._current_page_has_time()
        for w in self._scrubber_widgets:
            w.setVisible(vis)
        if vis and self.tl_slider.maximum() > 0:
            # Reflect the current slider value onto the newly shown page
            self._apply_timeline_to_page(self.tl_slider.value(), replot=False)

    def _refresh_timeline_range(self):
        """Call after datasets load. Rebuilds slider range; preserves index if possible."""
        n = len(getattr(self, '_all_times', []) or [])
        if n == 0:
            self.tl_slider.blockSignals(True)
            self.tl_slider.setRange(0, 0)
            self.tl_slider.setValue(0)
            self.tl_slider.blockSignals(False)
            self.tl_label.setText("—")
            for w in self._scrubber_widgets:
                w.setEnabled(False)
            return
        for w in self._scrubber_widgets:
            w.setEnabled(True)
        prev = self.tl_slider.value()
        self.tl_slider.blockSignals(True)
        self.tl_slider.setRange(0, n - 1)
        self.tl_slider.setValue(min(prev, n - 1))
        self.tl_slider.blockSignals(False)
        self._on_timeline_changed(self.tl_slider.value())

    def _on_timeline_changed(self, idx):
        times = getattr(self, '_all_times', []) or []
        if not (0 <= idx < len(times)):
            return
        ts = str(times[idx])
        self.tl_label.setText(f"{ts[:10]}  {ts[11:19]}  ({idx + 1}/{len(times)})")
        if self._current_page_has_time():
            self._apply_timeline_to_page(idx, replot=self.tl_autoplot.isChecked())

    def _apply_timeline_to_page(self, idx, replot):
        times = getattr(self, '_all_times', []) or []
        if not (0 <= idx < len(times)):
            return
        ts = str(times[idx])
        date_part, time_part = ts[:10], ts[11:19]
        page = self._pages.get(self.plot_type_combo.currentText(), {})
        w = page.get('widgets', {})
        date_combo = w.get('date')
        time_combo = w.get('time_of_day')
        if date_combo is None or time_combo is None:
            return
        d_idx = date_combo.findText(date_part)
        if d_idx >= 0 and d_idx != date_combo.currentIndex():
            date_combo.setCurrentIndex(d_idx)   # triggers _on_date_changed → repopulates time_combo
        t_idx = time_combo.findText(time_part)
        if t_idx >= 0:
            time_combo.setCurrentIndex(t_idx)
        else:
            time_combo.setCurrentText(time_part)
        if replot:
            self._tl_plot_timer.start()

    def _tl_fire_plot(self):
        if self._plot_running:
            # Try again after current plot completes
            self._tl_plot_timer.start()
            return
        if self.tl_autoplot.isChecked() and self._current_page_has_time():
            self.on_plot()

    # ------------------------------------------------------------------
    # Drag-and-drop of .nc files onto the window
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasUrls() and any(
            u.toLocalFile().lower().endswith((".nc", ".nc4", ".h5"))
            or os.path.isdir(u.toLocalFile())
            for u in md.urls()
        ):
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            self._load_from_path(path)
            event.acceptProposedAction()

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
        w.update(_add_time_group(layout))
        w.update(_add_level_group(layout))
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
        w.update(_add_time_group(layout))
        w['longitude'] = _add_combo(layout, "Longitude:")
        w['unit'] = _add_line(layout, "Variable Unit:")
        w.update(_add_level_bounds(layout))
        w['y_axis'] = QComboBox()
        w['y_axis'].addItems(['pressure', 'height'])
        layout.addRow("Y-Axis:", w['y_axis'])
        w['log_level'] = _add_check(layout, "Log Level:", True)
        w['clean'] = _add_check(layout, "Clean Plot:", True)
        return {'widget': page, 'widgets': w}

    def _create_lev_lon_page(self):
        page = QWidget(); layout = QFormLayout(page); w = {}
        w['variable'] = _add_combo(layout, "Variable Name:")
        w['latitude'] = _add_combo(layout, "Latitude:")
        w.update(_add_time_group(layout))
        w['log_level'] = _add_check(layout, "Log Level:", True)
        w['unit'] = _add_line(layout, "Variable Unit:")
        w.update(_add_contour_widgets(layout))
        w.update(_add_level_bounds(layout))
        w.update(_add_lon_bounds(layout))
        w['y_axis'] = QComboBox()
        w['y_axis'].addItems(['pressure', 'height'])
        layout.addRow("Y-Axis:", w['y_axis'])
        w['clean'] = _add_check(layout, "Clean Plot:", True)
        return {'widget': page, 'widgets': w}

    def _create_lev_lat_page(self):
        page = QWidget(); layout = QFormLayout(page); w = {}
        w['variable'] = _add_combo(layout, "Variable Name:")
        w.update(_add_time_group(layout))
        w['longitude'] = _add_combo(layout, "Longitude:")
        w['log_level'] = _add_check(layout, "Log Level:", True)
        w['unit'] = _add_line(layout, "Variable Unit:")
        w.update(_add_contour_widgets(layout))
        w.update(_add_level_bounds(layout))
        w.update(_add_lat_bounds(layout))
        w['y_axis'] = QComboBox()
        w['y_axis'].addItems(['pressure', 'height'])
        layout.addRow("Y-Axis:", w['y_axis'])
        w['clean'] = _add_check(layout, "Clean Plot:", True)
        return {'widget': page, 'widgets': w}

    def _create_lev_time_page(self):
        page = QWidget(); layout = QFormLayout(page); w = {}
        w['variable'] = _add_combo(layout, "Variable Name:")
        w['latitude'] = _add_combo(layout, "Latitude:")
        w['longitude'] = _add_combo(layout, "Longitude:")
        w['log_level'] = _add_check(layout, "Log Level:", True)
        w['y_axis'] = QComboBox()
        w['y_axis'].addItems(['pressure', 'height'])
        layout.addRow("Y-Axis:", w['y_axis'])
        w['unit'] = _add_line(layout, "Variable Unit:")
        w.update(_add_contour_widgets(layout))
        w.update(_add_level_bounds(layout))
        w.update(_add_mtime_bounds(layout))
        w['clean'] = _add_check(layout, "Clean Plot:", True)
        return {'widget': page, 'widgets': w}

    def _create_lat_time_page(self):
        page = QWidget(); layout = QFormLayout(page); w = {}
        w['variable'] = _add_combo(layout, "Variable Name:")
        w.update(_add_level_group(layout))
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
        w.update(_add_level_group(layout))
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
        w.update(_add_level_group(layout))
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
        self._update_timeline_visibility()

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
            clear_derived_cache()
            self.datasets = load_datasets(directory, dataset_filter=dataset_filter)
            if not self.datasets:
                QMessageBox.warning(self, "No Datasets", "No valid NetCDF datasets were found.")
                return
            self.selected_dataset = self.datasets
            self._add_recent(directory)
            self.statusBar().showMessage(f"Loaded {len(self.datasets)} dataset(s) from {directory}", 5000)
            self._populate_all_combos()
            self._refresh_timeline_range()
            self._update_timeline_visibility()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load datasets:\n{e}")

    def _derived_var_missing(self, requirements):
        """Return a list of raw dataset variable names that are needed but absent.
        Empty list means the derived variable is computable with the current dataset.
        """
        if not self.selected_dataset:
            return list(requirements)
        ds0 = self.selected_dataset[0]
        if not hasattr(ds0, 'ds') or not hasattr(ds0, 'model'):
            return []
        try:
            sp = get_species_names(ds0.model)
            defaults = MODEL_DEFAULTS[ds0.model]
        except Exception:
            return []
        all_vars = set(ds0.ds.variables)
        missing = []
        for req in requirements:
            if req == 'W_or_OMEGA':
                if 'W' not in all_vars and 'OMEGA' not in all_vars:
                    missing.append('W/OMEGA')
            elif req in sp:
                if sp[req] not in all_vars:
                    missing.append(sp[req])
            elif req in defaults:
                if defaults[req] not in all_vars:
                    missing.append(defaults[req])
            else:
                missing.append(req)
        return missing

    def _populate_variable_combo(self, combo, std_vars, derived_entries):
        """Fill a variable combo with standard vars + derived vars (disabled if missing inputs)."""
        combo.clear()
        combo.addItems(std_vars)
        if not derived_entries:
            return
        combo.insertSeparator(combo.count())
        model = combo.model()
        for name, missing in derived_entries:
            combo.addItem(name)
            item = model.item(combo.count() - 1)
            if item is None:
                continue
            if missing:
                item.setEnabled(False)
                item.setForeground(QBrush(QColor("#e78284")))  # Catppuccin red
                item.setToolTip(f"Missing inputs: {', '.join(missing)}")
            else:
                item.setToolTip("Derived variable (computed from model fields)")

    def _populate_all_combos(self):
        """Populate combo boxes across all pages from loaded datasets."""
        valid_vars = var_list(self.selected_dataset)
        self._all_times = time_list(self.selected_dataset)
        valid_lats = lat_list(self.selected_dataset)
        valid_lons = lon_list(self.selected_dataset)
        valid_levels = level_list(self.selected_dataset)

        # Only show ALL-CAPS variable names (e.g. TN, UN, not lat, mtime)
        str_vars = [str(v) for v in valid_vars if str(v).isupper()]
        str_lats = [str(lat) for lat in valid_lats]
        str_lons = [str(lon) for lon in valid_lons]
        str_levels = [str(lev) for lev in valid_levels]

        # Build date → times mapping for the date/time picker
        self._date_time_map = {}
        for t in self._all_times:
            ts = str(t)
            date_part = ts[:10]   # "YYYY-MM-DD"
            time_part = ts[11:19] # "HH:MM:SS" — strip nanosecond fractions
            self._date_time_map.setdefault(date_part, []).append(time_part)
        str_dates = sorted(self._date_time_map.keys())

        # Pre-compute which derived variables are available for this dataset
        derived_entries = [(name, self._derived_var_missing(reqs))
                           for name, reqs in _DERIVED_VARS]

        combo_map = {
            'latitude': str_lats,
            'longitude': str_lons,
            'level': str_levels,
        }

        for page_data in self._pages.values():
            w = page_data['widgets']
            if 'variable' in w and isinstance(w['variable'], QComboBox):
                self._populate_variable_combo(w['variable'], str_vars, derived_entries)
            for key, items in combo_map.items():
                if key in w and isinstance(w[key], QComboBox):
                    w[key].clear()
                    w[key].addItems(items)
            # Set up filterable level combo and summary label
            if 'level' in w and isinstance(w['level'], QComboBox):
                _setup_filterable(w['level'])
                if 'level_avail' in w and str_levels:
                    w['level_avail'].setText(
                        f"{len(str_levels)} levels ({str_levels[0]} \u2013 {str_levels[-1]})")
            # Populate date combos and wire up time combos
            if 'date' in w and isinstance(w['date'], QComboBox):
                date_combo = w['date']
                time_combo = w.get('time_of_day')
                avail_label = w.get('time_avail')
                # Block signals while repopulating to avoid premature firing
                date_combo.blockSignals(True)
                date_combo.clear()
                date_combo.addItems(str_dates)
                date_combo.blockSignals(False)
                # Disconnect old handler
                try:
                    date_combo.currentIndexChanged.disconnect()
                except TypeError:
                    pass
                if time_combo is not None:
                    date_combo.currentIndexChanged.connect(
                        lambda _idx, d=date_combo, tc=time_combo, lbl=avail_label:
                            self._on_date_changed(d.currentText(), tc, lbl))
                    # Manually trigger for initial date
                    self._on_date_changed(
                        date_combo.currentText(), time_combo, avail_label)

    def _on_date_changed(self, date_str, time_combo, avail_label=None):
        """Populate time combo with available times and update summary label."""
        times = self._date_time_map.get(date_str, [])
        time_combo.clear()
        time_combo.addItems(times)
        _setup_filterable(time_combo)
        if avail_label:
            if times:
                avail_label.setText(f"{len(times)} times ({times[0]} \u2013 {times[-1]})")
            else:
                avail_label.setText("No times available")

    def _get_selected_time(self, w):
        """Reconstruct ISO timestamp from date + time_of_day combos."""
        date_str = w['date'].currentText() if 'date' in w else ''
        if not date_str:
            return ''
        time_str = ''
        if 'time_of_day' in w and isinstance(w['time_of_day'], QComboBox):
            time_str = w['time_of_day'].currentText().strip()
        if date_str and time_str:
            return f"{date_str}T{time_str}"
        return date_str

    # ------------------------------------------------------------------
    # Plot dispatch
    # ------------------------------------------------------------------
    def on_plot(self):
        if not self.selected_dataset:
            QMessageBox.warning(self, "No Data", "Please load datasets first.")
            return
        if self._plot_running:
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

        # Synchronous dispatch — matplotlib + Qt + threadpool segfaulted on
        # rapid plot replacement. processEvents lets the busy UI paint first.
        self._set_busy(True, f"Generating {plot_type} plot…")
        QApplication.processEvents()
        try:
            fig = handler()
        except Exception as exc:
            logger.exception("Plot generation failed")
            self._set_busy(False)
            self.statusBar().showMessage("Plot failed", 4000)
            QMessageBox.critical(self, "Plot Error", f"Could not generate plot:\n{exc}")
            return
        try:
            if fig is not None:
                self._update_canvas(fig)
                self.statusBar().showMessage("Plot rendered", 4000)
            else:
                self.statusBar().showMessage("Plot returned no figure", 4000)
        finally:
            self._set_busy(False)

    def _set_busy(self, busy, msg=""):
        self._plot_running = busy
        self.plot_button.setEnabled(not busy)
        if busy:
            self.progress.show()
            self.statusBar().showMessage(msg)
            QApplication.setOverrideCursor(Qt.BusyCursor)
        else:
            self.progress.hide()
            QApplication.restoreOverrideCursor()

    def _update_canvas(self, fig):
        """Swap the figure on the existing canvas and redraw.

        Reuses the canvas + toolbar instead of rebuilding them so scrubbing
        feels snappy and Qt widget churn doesn't trigger double-free segfaults.
        Forces the new figure to match the canvas widget size (plot_gen creates
        figures at their own figsize) and uses a synchronous draw so hover
        paintEvents don't show the previous figure's buffer. The previous
        figure is detached onto a throwaway Agg canvas before plt.close so
        pyplot's cleanup doesn't destroy the live Qt canvas.
        """
        old_fig = self.fig
        if fig is old_fig:
            self.canvas.draw()
            return
        self.fig = fig
        fig.set_canvas(self.canvas)
        self.canvas.figure = fig
        w, h = self.canvas.width(), self.canvas.height()
        if w > 0 and h > 0 and fig.dpi > 0:
            fig.set_size_inches(w / fig.dpi, h / fig.dpi, forward=False)
        self.toolbar.update()
        self.canvas.draw()
        if old_fig is not None:
            try:
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                FigureCanvasAgg(old_fig)
                plt.close(old_fig)
            except Exception:
                logger.debug("plt.close on old figure failed", exc_info=True)

    # ------------------------------------------------------------------
    # Per-type plot handlers
    # ------------------------------------------------------------------
    def _plot_lat_lon(self):
        w = self._pages["Lat vs Lon"]['widgets']
        params = {
            "datasets": self.selected_dataset,
            "variable_name": w['variable'].currentText(),
            "time": self._get_selected_time(w),
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
        level_val, level_type = _get_level_params(w)
        params["level"] = level_val
        params["level_type"] = level_type
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
            "time": self._get_selected_time(w),
            "longitude": _float_or_none(w['longitude'].currentText()),
            "log_level": w['log_level'].isChecked(),
            "variable_unit": _str_or_none(w['unit'].text()),
            "level_minimum": _float_or_none(w['level_min'].text()),
            "level_maximum": _float_or_none(w['level_max'].text()),
            "clean_plot": w['clean'].isChecked(),
            "y_axis": w['y_axis'].currentText(),
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
            "time": self._get_selected_time(w),
            "log_level": w['log_level'].isChecked(),
            "variable_unit": _str_or_none(w['unit'].text()),
            "level_minimum": _float_or_none(w['level_min'].text()),
            "level_maximum": _float_or_none(w['level_max'].text()),
            "longitude_minimum": _float_or_none(w['lon_min'].text()),
            "longitude_maximum": _float_or_none(w['lon_max'].text()),
            "clean_plot": w['clean'].isChecked(),
            "y_axis": w['y_axis'].currentText(),
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
            "time": self._get_selected_time(w),
            "longitude": _float_or_none(w['longitude'].currentText()),
            "log_level": w['log_level'].isChecked(),
            "variable_unit": _str_or_none(w['unit'].text()),
            "level_minimum": _float_or_none(w['level_min'].text()),
            "level_maximum": _float_or_none(w['level_max'].text()),
            "latitude_minimum": _float_or_none(w['lat_min'].text()),
            "latitude_maximum": _float_or_none(w['lat_max'].text()),
            "clean_plot": w['clean'].isChecked(),
            "y_axis": w['y_axis'].currentText(),
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
            "y_axis": w['y_axis'].currentText(),
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
            "longitude": _float_or_none(w['longitude'].currentText()),
            "variable_unit": _str_or_none(w['unit'].text()),
            "latitude_minimum": _float_or_none(w['lat_min'].text()),
            "latitude_maximum": _float_or_none(w['lat_max'].text()),
            "mtime_minimum": _str_or_none(w['mtime_min'].text()),
            "mtime_maximum": _str_or_none(w['mtime_max'].text()),
            "clean_plot": w['clean'].isChecked(),
        }
        level_val, level_type = _get_level_params(w)
        params["level"] = level_val
        params["level_type"] = level_type
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
            "variable_unit": _str_or_none(w['unit'].text()),
            "longitude_minimum": _float_or_none(w['lon_min'].text()),
            "longitude_maximum": _float_or_none(w['lon_max'].text()),
            "mtime_minimum": _str_or_none(w['mtime_min'].text()),
            "mtime_maximum": _str_or_none(w['mtime_max'].text()),
            "clean_plot": w['clean'].isChecked(),
        }
        level_val, level_type = _get_level_params(w)
        params["level"] = _float_or_none(level_val) if level_val else None
        params["level_type"] = level_type
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
            "variable_unit": _str_or_none(w['unit'].text()),
            "mtime_minimum": _str_or_none(w['mtime_min'].text()),
            "mtime_maximum": _str_or_none(w['mtime_max'].text()),
            "clean_plot": w['clean'].isChecked(),
        }
        level_val, level_type = _get_level_params(w)
        params["level"] = _float_or_none(level_val) if level_val else None
        params["level_type"] = level_type
        result = plt_var_time(**params)

        if not isinstance(result, tuple):
            return result

        fig, variable_unit = result
        w['unit'].setPlaceholderText(str(variable_unit))
        return fig

MODERN_STYLESHEET = """
QMainWindow {
    background-color: #303446;
}
QWidget {
    background-color: #303446;
    color: #c6d0f5;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    background-color: #414559;
    border: 1px solid #51576d;
    border-radius: 8px;
    margin-top: 14px;
    padding: 14px 10px 10px 10px;
    font-weight: bold;
    font-size: 13px;
    color: #ca9ee6;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 10px;
    background-color: #414559;
    border-radius: 4px;
}
QLabel {
    color: #b5bfe2;
    font-size: 12px;
    padding: 1px 0;
}
QLineEdit {
    background-color: #51576d;
    border: 1px solid #626880;
    border-radius: 6px;
    padding: 5px 8px;
    color: #c6d0f5;
    selection-background-color: #8caaee;
    selection-color: #303446;
}
QLineEdit:focus {
    border: 1px solid #8caaee;
}
QLineEdit::placeholder {
    color: #838ba7;
}
QComboBox {
    background-color: #51576d;
    border: 1px solid #626880;
    border-radius: 6px;
    padding: 5px 8px;
    color: #c6d0f5;
    min-height: 20px;
}
QComboBox:hover {
    border: 1px solid #8caaee;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid #626880;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #c6d0f5;
    margin-right: 5px;
}
QComboBox QAbstractItemView {
    background-color: #414559;
    border: 1px solid #626880;
    border-radius: 4px;
    color: #c6d0f5;
    selection-background-color: #8caaee;
    selection-color: #303446;
    padding: 4px;
}
QPushButton {
    background-color: #8caaee;
    color: #303446;
    border: none;
    border-radius: 6px;
    padding: 7px 16px;
    font-weight: bold;
    font-size: 12px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #85c1dc;
}
QPushButton:pressed {
    background-color: #babbf1;
}
QPushButton#browse_btn {
    background-color: #626880;
    color: #c6d0f5;
    padding: 7px 10px;
    font-weight: normal;
}
QPushButton#browse_btn:hover {
    background-color: #838ba7;
}
QCheckBox {
    spacing: 8px;
    color: #b5bfe2;
}
QCheckBox::indicator {
    width: 36px;
    height: 20px;
    border-radius: 10px;
    border: none;
    background-color: #626880;
}
QCheckBox::indicator:checked {
    background-color: #8caaee;
}
QCheckBox::indicator:hover {
    background-color: #838ba7;
}
QCheckBox::indicator:checked:hover {
    background-color: #85c1dc;
}
QTimeEdit {
    background-color: #51576d;
    border: 1px solid #626880;
    border-radius: 6px;
    padding: 5px 8px;
    color: #c6d0f5;
    min-height: 20px;
}
QTimeEdit:focus {
    border: 1px solid #8caaee;
}
QTimeEdit::up-button, QTimeEdit::down-button {
    background-color: #626880;
    border-radius: 3px;
    width: 16px;
    margin: 1px;
}
QTimeEdit::up-button:hover, QTimeEdit::down-button:hover {
    background-color: #838ba7;
}
QTimeEdit::up-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #c6d0f5;
}
QTimeEdit::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #c6d0f5;
}
QScrollArea {
    border: none;
    background-color: #303446;
}
QScrollBar:vertical {
    background-color: #303446;
    width: 10px;
    margin: 0;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background-color: #626880;
    min-height: 30px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background-color: #838ba7;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background-color: #303446;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background-color: #626880;
    min-width: 30px;
    border-radius: 5px;
}
QSplitter::handle {
    background-color: #51576d;
    width: 2px;
}
QSplitter::handle:hover {
    background-color: #8caaee;
}
QMessageBox {
    background-color: #414559;
}
QMessageBox QLabel {
    color: #c6d0f5;
    font-size: 13px;
}
QMenuBar {
    background-color: #292c3c;
    color: #c6d0f5;
    padding: 2px;
}
QMenuBar::item:selected {
    background-color: #51576d;
    border-radius: 4px;
}
QMenu {
    background-color: #414559;
    border: 1px solid #51576d;
    color: #c6d0f5;
    padding: 4px;
}
QMenu::item {
    padding: 5px 20px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #8caaee;
    color: #303446;
}
QMenu::item:disabled {
    color: #838ba7;
}
QMenu::separator {
    height: 1px;
    background-color: #51576d;
    margin: 4px 8px;
}
QStatusBar {
    background-color: #292c3c;
    color: #b5bfe2;
}
QProgressBar {
    background-color: #51576d;
    border: 1px solid #626880;
    border-radius: 4px;
    color: #c6d0f5;
    text-align: center;
    height: 14px;
}
QProgressBar::chunk {
    background-color: #8caaee;
    border-radius: 3px;
}
QToolBar {
    background-color: #e6e6e6;
    border: none;
    spacing: 2px;
    padding: 2px;
}
QToolBar QToolButton {
    background-color: #e6e6e6;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 3px;
    margin: 1px;
}
QToolBar QToolButton:hover {
    background-color: #c6d0f5;
    border: 1px solid #8caaee;
}
QToolBar QToolButton:pressed,
QToolBar QToolButton:checked {
    background-color: #babbf1;
    border: 1px solid #8caaee;
}
QToolBar QToolButton:disabled {
    background-color: #e6e6e6;
}
QToolBar QLabel {
    color: #303446;
    background-color: #e6e6e6;
    padding: 0 4px;
}
QWidget#plot_pane {
    background-color: #626880;
    border-radius: 8px;
}
QWidget#timeline_bar {
    background-color: #626880;
    border: none;
}
QWidget#timeline_bar QLabel {
    color: #c6d0f5;
    background-color: #626880;
    font-family: "Menlo", "Consolas", monospace;
    font-size: 12px;
    padding: 0 6px;
}
QWidget#timeline_bar QSlider {
    background-color: #626880;
}
QWidget#timeline_bar QCheckBox {
    background-color: #626880;
    color: #c6d0f5;
    padding: 0 6px;
}
QWidget#timeline_bar QCheckBox::indicator:unchecked {
    background-color: #51576d;
}
QPushButton#plot_btn {
    background-color: #e5c890;
    color: #232634;
    border: 1px solid #ef9f76;
    border-radius: 6px;
    padding: 6px 24px;
    font-weight: bold;
    font-size: 13px;
    min-height: 22px;
}
QPushButton#plot_btn:hover {
    background-color: #ef9f76;
    border-color: #e78284;
}
QPushButton#plot_btn:pressed {
    background-color: #e78284;
    color: #232634;
}
QPushButton#plot_btn:disabled {
    background-color: #626880;
    color: #838ba7;
}
QPushButton#tl_btn {
    background-color: #51576d;
    color: #c6d0f5;
    border: none;
    border-radius: 6px;
    padding: 4px 0;
    font-size: 14px;
    font-weight: bold;
}
QPushButton#tl_btn:hover {
    background-color: #8caaee;
    color: #303446;
}
QPushButton#tl_btn:pressed {
    background-color: #babbf1;
}
QPushButton#tl_btn:disabled {
    background-color: #414559;
    color: #838ba7;
}
QSlider::groove:horizontal {
    height: 6px;
    background-color: #51576d;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background-color: #8caaee;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background-color: #c6d0f5;
    border: 2px solid #8caaee;
    width: 14px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background-color: #8caaee;
    border-color: #babbf1;
}
"""


def main():
    QCoreApplication.setOrganizationName(ORG_NAME)
    QCoreApplication.setApplicationName(APP_NAME)
    app = QApplication(sys.argv)
    app.setStyleSheet(MODERN_STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
