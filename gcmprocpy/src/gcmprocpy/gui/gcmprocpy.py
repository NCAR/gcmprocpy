import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")  # Use the Qt5Agg backend for PyQt5
import matplotlib.pyplot as plt
import mplcursors  # For interactive annotations
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QSplitter, QLineEdit, QPushButton, QCheckBox,
    QComboBox, QMessageBox, QFileDialog, QLabel, QSizePolicy, QStackedWidget
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import xarray as xr
from datetime import datetime
from ..plot_gen import plt_lev_var, plt_lat_lon, plt_lev_lat, plt_lev_lon, plt_lev_time, plt_lat_time
from ..io import load_datasets, save_output
from ..data_parse import time_list, var_list, level_list, lon_list, lat_list



# ----------------------------------------------------------------------------
# Main GUI – organized into Dataset, Plot Type, and Plot Parameters sections.
# ----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GCMProcPy")
        self.resize(1600, 600)
        
        # Main splitter divides window into left (controls) and right (plot)
        self.splitter = QSplitter(Qt.Horizontal)
        
        # --- Left Panel: Controls ---
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        
        # Dataset Group
        self.dataset_group = QGroupBox("Dataset")
        ds_layout = QFormLayout()
        self.directory_input = QLineEdit()
        self.directory_input.setPlaceholderText("Enter directory or file path")
        ds_layout.addRow("Directory:", self.directory_input)
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
        self.plot_type_combo.addItems([
            "Lat vs Lon", 
            "Lev vs Var", 
            "Lev vs Lon", 
            "Lev vs Lat", 
            "Lev vs Time", 
            "Lat vs Time"
        ])
        self.plot_type_combo.currentIndexChanged.connect(self.on_plot_type_changed)
        pt_layout.addWidget(self.plot_type_combo)
        self.plot_type_group.setLayout(pt_layout)
        self.controls_layout.addWidget(self.plot_type_group)
        
        # Plot Parameters Group – use a QStackedWidget so parameters change with plot type
        self.param_stack = QStackedWidget()
        # Page for "Lat vs Lon" (using plt_lat_lon)
        self.page_lat_lon = self.create_lat_lon_page()
        # Page for "Lev vs Var" (using plt_lev_var)
        self.page_lev_var = self.create_lev_var_page()
        # For other plot types, add placeholder pages
        self.page_lev_lon = self.create_lev_lon_page()
        self.page_lev_lat = self.create_lev_lat_page()
        self.page_lev_time = self.create_lev_time_page()
        self.page_lat_time = self.create_lat_time_page()
        
        self.param_stack.addWidget(self.page_lat_lon)   # index 0
        self.param_stack.addWidget(self.page_lev_var)     # index 1
        self.param_stack.addWidget(self.page_lev_lon)     # index 2
        self.param_stack.addWidget(self.page_lev_lat)     # index 3
        self.param_stack.addWidget(self.page_lev_time)    # index 4
        self.param_stack.addWidget(self.page_lat_time)    # index 5
        
        self.controls_layout.addWidget(self.param_stack)
        
        # Buttons for Plot and Save as Image
        button_layout = QHBoxLayout()
        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.on_plot)
        button_layout.addWidget(self.plot_button)
        self.save_button = QPushButton("Save as Image")
        self.save_button.clicked.connect(self.on_save_image)
        button_layout.addWidget(self.save_button)
        self.controls_layout.addLayout(button_layout)

        self.controls_layout.addStretch()
        
        # --- Right Panel: Plot Display ---
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.canvas = FigureCanvas(plt.figure(figsize=(10,6)))
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_layout.addWidget(self.canvas)
        
        # Add panels to splitter
        self.splitter.addWidget(self.controls_widget)
        self.splitter.addWidget(self.plot_widget)
        self.splitter.setSizes([300, 900])
        self.setCentralWidget(self.splitter)
        
        # Dataset storage
        self.datasets = []  # list of dataset tuples
        self.selected_dataset = None  # will store the entire list
        
    # --- Parameter Pages ---
    def create_lat_lon_page(self):
        """Create parameters for plt_lat_lon."""
        page = QWidget()
        layout = QFormLayout(page)
        self.lat_lon_variable_combo = QComboBox()
        layout.addRow("Variable Name:", self.lat_lon_variable_combo)
        self.lat_lon_time_combo = QComboBox()
        layout.addRow("Time (ISO):", self.lat_lon_time_combo)
        self.lat_lon_level_combo = QComboBox()
        layout.addRow("Level:", self.lat_lon_level_combo)
        self.lat_lon_unit_input = QLineEdit()
        layout.addRow("Variable Unit:", self.lat_lon_unit_input)
        self.lat_lon_center_lon_input = QLineEdit()
        layout.addRow("Center Longitude:", self.lat_lon_center_lon_input)
        self.lat_lon_contour_intervals_input = QLineEdit()
        layout.addRow("Contour Intervals:", self.lat_lon_contour_intervals_input)
        self.lat_lon_contour_value_input = QLineEdit()
        layout.addRow("Contour Value:", self.lat_lon_contour_value_input)
        self.lat_lon_symmetric_checkbox = QCheckBox()
        layout.addRow("Symmetric Interval:", self.lat_lon_symmetric_checkbox)
        self.lat_lon_cmap_input = QLineEdit()
        layout.addRow("Colormap:", self.lat_lon_cmap_input)
        self.lat_lon_cmap_lim_min_input = QLineEdit()
        layout.addRow("Colormap Min:", self.lat_lon_cmap_lim_min_input)
        self.lat_lon_cmap_lim_max_input = QLineEdit()
        layout.addRow("Colormap Max:", self.lat_lon_cmap_lim_max_input)
        self.lat_lon_line_color_input = QLineEdit()
        layout.addRow("Line Color:", self.lat_lon_line_color_input)
        self.lat_lon_coastlines_checkbox = QCheckBox()
        layout.addRow("Coastlines:", self.lat_lon_coastlines_checkbox)
        self.lat_lon_nightshade_checkbox = QCheckBox()
        layout.addRow("Nightshade:", self.lat_lon_nightshade_checkbox)
        self.lat_lon_gm_equator_checkbox = QCheckBox()
        layout.addRow("GM Equator:", self.lat_lon_gm_equator_checkbox)
        self.lat_lon_lat_min_input = QLineEdit()
        layout.addRow("Latitude Min:", self.lat_lon_lat_min_input)
        self.lat_lon_lat_max_input = QLineEdit()
        layout.addRow("Latitude Max:", self.lat_lon_lat_max_input)
        self.lat_lon_lon_min_input = QLineEdit()
        layout.addRow("Longitude Min:", self.lat_lon_lon_min_input)
        self.lat_lon_lon_max_input = QLineEdit()
        layout.addRow("Longitude Max:", self.lat_lon_lon_max_input)
        self.lat_lon_clean_checkbox = QCheckBox()
        self.lat_lon_clean_checkbox.setChecked(True)
        layout.addRow("Clean Plot:", self.lat_lon_clean_checkbox)
        return page
        
    def create_lev_var_page(self):
        """Create parameters for plt_lev_var."""
        page = QWidget()
        layout = QFormLayout(page)
        self.lev_var_variable_combo = QComboBox()
        layout.addRow("Variable Name:", self.lev_var_variable_combo)
        self.lev_var_lat_combo = QComboBox()
        layout.addRow("Latitude:", self.lev_var_lat_combo)
        self.lev_var_time_combo = QComboBox()
        layout.addRow("Time (ISO):", self.lev_var_time_combo)
        self.lev_var_lon_combo = QComboBox()
        layout.addRow("Longitude:", self.lev_var_lon_combo)
        self.lev_var_unit_input = QLineEdit()
        layout.addRow("Variable Unit:", self.lev_var_unit_input)
        self.lev_var_level_min_input = QLineEdit()
        layout.addRow("Level Min:", self.lev_var_level_min_input)
        self.lev_var_level_max_input = QLineEdit()
        layout.addRow("Level Max:", self.lev_var_level_max_input)
        self.lev_var_log_checkbox = QCheckBox()
        self.lev_var_log_checkbox.setChecked(True)
        layout.addRow("Log Level:", self.lev_var_log_checkbox)
        self.lev_var_clean_checkbox = QCheckBox()
        self.lev_var_clean_checkbox.setChecked(True)
        layout.addRow("Clean Plot:", self.lev_var_clean_checkbox)
        return page

    def create_lev_lon_page(self):
        """Create parameters for plt_lev_lon."""
        page = QWidget()
        layout = QFormLayout(page)
        self.lev_lon_variable_combo = QComboBox()
        layout.addRow("Variable Name:", self.lev_lon_variable_combo)
        self.lev_lon_lat_combo = QComboBox()
        layout.addRow("Latitude:", self.lev_lon_lat_combo)
        self.lev_lon_time_combo = QComboBox()
        layout.addRow("Time (ISO):", self.lev_lon_time_combo)
        self.lev_lon_log_checkbox = QCheckBox()
        self.lev_lon_log_checkbox.setChecked(True)
        layout.addRow("Log Level:", self.lev_lon_log_checkbox)
        self.lev_lon_unit_input = QLineEdit()
        layout.addRow("Variable Unit:", self.lev_lon_unit_input)
        self.lev_lon_contour_intervals_input = QLineEdit()
        layout.addRow("Contour Intervals:", self.lev_lon_contour_intervals_input)
        self.lev_lon_contour_value_input = QLineEdit()
        layout.addRow("Contour Value:", self.lev_lon_contour_value_input)
        self.lev_lon_symmetric_checkbox = QCheckBox()
        layout.addRow("Symmetric Interval:", self.lev_lon_symmetric_checkbox)
        self.lev_lon_cmap_input = QLineEdit()
        layout.addRow("Colormap:", self.lev_lon_cmap_input)
        self.lev_lon_cmap_lim_min_input = QLineEdit()
        layout.addRow("Colormap Min:", self.lev_lon_cmap_lim_min_input)
        self.lev_lon_cmap_lim_max_input = QLineEdit()
        layout.addRow("Colormap Max:", self.lev_lon_cmap_lim_max_input)
        self.lev_lon_line_color_input = QLineEdit()
        layout.addRow("Line Color:", self.lev_lon_line_color_input)
        self.lev_lon_level_min_input = QLineEdit()
        layout.addRow("Level Min:", self.lev_lon_level_min_input)
        self.lev_lon_level_max_input = QLineEdit()
        layout.addRow("Level Max:", self.lev_lon_level_max_input)
        self.lev_lon_lon_min_input = QLineEdit()
        layout.addRow("Longitude Min:", self.lev_lon_lon_min_input)
        self.lev_lon_lon_max_input = QLineEdit()
        layout.addRow("Longitude Max:", self.lev_lon_lon_max_input)
        self.lev_lon_clean_checkbox = QCheckBox()
        self.lev_lon_clean_checkbox.setChecked(True)
        layout.addRow("Clean Plot:", self.lev_lon_clean_checkbox)
        return page
    

    def create_lev_lat_page(self):
        """Create parameters for plt_lev_lat."""
        page = QWidget()
        layout = QFormLayout(page)
        self.lev_lat_variable_combo = QComboBox()
        layout.addRow("Variable Name:", self.lev_lat_variable_combo)
        self.lev_lat_time_combo = QComboBox()
        layout.addRow("Time (ISO):", self.lev_lat_time_combo)
        self.lev_lat_lon_combo = QComboBox()
        layout.addRow("Longitude:", self.lev_lat_lon_combo)
        self.lev_lat_log_checkbox = QCheckBox()
        self.lev_lat_log_checkbox.setChecked(True)
        layout.addRow("Log Level:", self.lev_lat_log_checkbox)
        self.lev_lat_unit_input = QLineEdit()
        layout.addRow("Variable Unit:", self.lev_lat_unit_input)
        self.lev_lat_contour_intervals_input = QLineEdit()
        layout.addRow("Contour Intervals:", self.lev_lat_contour_intervals_input)
        self.lev_lat_contour_value_input = QLineEdit()
        layout.addRow("Contour Value:", self.lev_lat_contour_value_input)
        self.lev_lat_symmetric_checkbox = QCheckBox()
        layout.addRow("Symmetric Interval:", self.lev_lat_symmetric_checkbox)
        self.lev_lat_cmap_input = QLineEdit()
        layout.addRow("Colormap:", self.lev_lat_cmap_input)
        self.lev_lat_cmap_lim_min_input = QLineEdit()
        layout.addRow("Colormap Min:", self.lev_lat_cmap_lim_min_input)
        self.lev_lat_cmap_lim_max_input = QLineEdit()
        layout.addRow("Colormap Max:", self.lev_lat_cmap_lim_max_input)
        self.lev_lat_line_color_input = QLineEdit()
        layout.addRow("Line Color:", self.lev_lat_line_color_input)
        self.lev_lat_level_min_input = QLineEdit()
        layout.addRow("Level Min:", self.lev_lat_level_min_input)
        self.lev_lat_level_max_input = QLineEdit()
        layout.addRow("Level Max:", self.lev_lat_level_max_input)
        self.lev_lat_lat_min_input = QLineEdit()
        layout.addRow("Latitude Min:", self.lev_lat_lat_min_input)
        self.lev_lat_lat_max_input = QLineEdit()
        layout.addRow("Latitude Max:", self.lev_lat_lat_max_input)
        self.lev_lat_clean_checkbox = QCheckBox()
        self.lev_lat_clean_checkbox.setChecked(True)
        layout.addRow("Clean Plot:", self.lev_lat_clean_checkbox)
        return page


    def create_lev_time_page(self):
        """Create parameters for plt_lev_time."""
        page = QWidget()
        layout = QFormLayout(page)
        self.lev_time_variable_combo = QComboBox()
        layout.addRow("Variable Name:", self.lev_time_variable_combo)
        self.lev_time_lat_combo = QComboBox()
        layout.addRow("Latitude:", self.lev_time_lat_combo)
        self.lev_time_lon_combo = QComboBox()
        layout.addRow("Longitude:", self.lev_time_lon_combo)
        self.lev_time_log_checkbox = QCheckBox()
        self.lev_time_log_checkbox.setChecked(True)
        layout.addRow("Log Level:", self.lev_time_log_checkbox)
        self.lev_time_unit_input = QLineEdit()
        layout.addRow("Variable Unit:", self.lev_time_unit_input)
        self.lev_time_contour_intervals_input = QLineEdit()
        layout.addRow("Contour Intervals:", self.lev_time_contour_intervals_input)
        self.lev_time_contour_value_input = QLineEdit()
        layout.addRow("Contour Value:", self.lev_time_contour_value_input)
        self.lev_time_symmetric_checkbox = QCheckBox()
        layout.addRow("Symmetric Interval:", self.lev_time_symmetric_checkbox)
        self.lev_time_cmap_input = QLineEdit()
        layout.addRow("Colormap:", self.lev_time_cmap_input)
        self.lev_time_cmap_lim_min_input = QLineEdit()
        layout.addRow("Colormap Min:", self.lev_time_cmap_lim_min_input)
        self.lev_time_cmap_lim_max_input = QLineEdit()
        layout.addRow("Colormap Max:", self.lev_time_cmap_lim_max_input)
        self.lev_time_line_color_input = QLineEdit()
        layout.addRow("Line Color:", self.lev_time_line_color_input)
        self.lev_time_level_min_input = QLineEdit()
        layout.addRow("Level Min:", self.lev_time_level_min_input)
        self.lev_time_level_max_input = QLineEdit()
        layout.addRow("Level Max:", self.lev_time_level_max_input)
        self.lev_time_mtime_min_input = QLineEdit()
        layout.addRow("mtime Min:", self.lev_time_mtime_min_input)
        self.lev_time_mtime_max_input = QLineEdit()
        layout.addRow("mtime Max:", self.lev_time_mtime_max_input)
        self.lev_time_clean_checkbox = QCheckBox()
        self.lev_time_clean_checkbox.setChecked(True)
        layout.addRow("Clean Plot:", self.lev_time_clean_checkbox)
        return page
    
    def create_lat_time_page(self):
        """Create parameters for plt_lat_time."""
        page = QWidget()
        layout = QFormLayout(page)
        self.lat_time_variable_combo = QComboBox()
        layout.addRow("Variable Name:", self.lat_time_variable_combo)
        self.lat_time_level_combo = QComboBox()
        layout.addRow("Level:", self.lat_time_level_combo)
        self.lat_time_lon_combo = QComboBox()
        layout.addRow("Longitude:", self.lat_time_lon_combo)
        self.lat_time_unit_input = QLineEdit()
        layout.addRow("Variable Unit:", self.lat_time_unit_input)
        self.lat_time_contour_intervals_input = QLineEdit()
        layout.addRow("Contour Intervals:", self.lat_time_contour_intervals_input)
        self.lat_time_contour_value_input = QLineEdit()
        layout.addRow("Contour Value:", self.lat_time_contour_value_input)
        self.lat_time_symmetric_checkbox = QCheckBox()
        layout.addRow("Symmetric Interval:", self.lat_time_symmetric_checkbox)
        self.lat_time_cmap_input = QLineEdit()
        layout.addRow("Colormap:", self.lat_time_cmap_input)
        self.lat_time_cmap_lim_min_input = QLineEdit()
        layout.addRow("Colormap Min:", self.lat_time_cmap_lim_min_input)
        self.lat_time_cmap_lim_max_input = QLineEdit()
        layout.addRow("Colormap Max:", self.lat_time_cmap_lim_max_input)
        self.lat_time_line_color_input = QLineEdit()
        layout.addRow("Line Color:", self.lat_time_line_color_input)
        self.lat_time_lat_min_input = QLineEdit()
        layout.addRow("Latitude Min:", self.lat_time_lat_min_input)
        self.lat_time_lat_max_input = QLineEdit()
        layout.addRow("Latitude Max:", self.lat_time_lat_max_input)
        self.lat_time_mtime_min_input = QLineEdit()
        layout.addRow("mtime Min:", self.lat_time_mtime_min_input)
        self.lat_time_mtime_max_input = QLineEdit()
        layout.addRow("mtime Max:", self.lat_time_mtime_max_input)
        self.lat_time_clean_checkbox = QCheckBox()
        self.lat_time_clean_checkbox.setChecked(True)
        layout.addRow("Clean Plot:", self.lat_time_clean_checkbox)
        return page
    def create_placeholder_page(self, text):
        """Create a placeholder page for unimplemented plot types."""
        page = QWidget()
        layout = QVBoxLayout(page)
        label = QLabel(text)
        layout.addWidget(label)
        layout.addStretch()
        return page
        
    # --- Slots ---
    def on_plot_type_changed(self, index):
        self.param_stack.setCurrentIndex(index)
        
    def on_load_datasets(self):
        directory = self.directory_input.text().strip()
        dataset_filter = self.dataset_filter_input.text().strip() or None
        if not directory:
            QMessageBox.warning(self, "Input Error", "Please enter a valid directory or file path.")
            return
        try:
            self.datasets = load_datasets(directory, dataset_filter=dataset_filter)
            if not self.datasets:
                QMessageBox.warning(self, "No Datasets", "No valid NetCDF datasets were found.")
                return
            # Assign the entire list
            self.selected_dataset = self.datasets
            QMessageBox.information(self, "Datasets Loaded", f"Loaded {len(self.datasets)} dataset(s).")
            self.populate_common_lists()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load datasets:\n{e}")
            
    def populate_common_lists(self):
        # Use the list functions with self.selected_dataset
        valid_vars = var_list(self.selected_dataset)
        valid_times = time_list(self.selected_dataset)
        valid_lats = lat_list(self.selected_dataset)
        valid_lons = lon_list(self.selected_dataset)
        valid_levels = level_list(self.selected_dataset)
        
        # Populate fields in Lat vs Lon page
        self.lat_lon_variable_combo.clear()
        self.lat_lon_variable_combo.addItems([str(v) for v in valid_vars])
        self.lat_lon_time_combo.clear()
        self.lat_lon_time_combo.addItems([str(t) for t in valid_times])
        self.lat_lon_level_combo.clear()
        self.lat_lon_level_combo.addItems([str(lev) for lev in valid_levels])

        # Populate fields in Lev vs Var page
        self.lev_var_variable_combo.clear()
        self.lev_var_variable_combo.addItems([str(v) for v in valid_vars])
        self.lev_var_lat_combo.clear()
        self.lev_var_lat_combo.addItems([str(lat) for lat in valid_lats])
        self.lev_var_time_combo.clear()
        self.lev_var_time_combo.addItems([str(t) for t in valid_times])
        self.lev_var_lon_combo.clear()
        self.lev_var_lon_combo.addItems([str(lon) for lon in valid_lons])
        
        # Populate fields in Lev vs Lon page
        self.lev_lon_variable_combo.clear()
        self.lev_lon_variable_combo.addItems([str(v) for v in valid_vars])
        self.lev_lon_lat_combo.clear()
        self.lev_lon_lat_combo.addItems([str(lat) for lat in valid_lats])
        self.lev_lon_time_combo.clear()
        self.lev_lon_time_combo.addItems([str(t) for t in valid_times])

        # Populate fields in Lev vs Lat page
        self.lev_lat_variable_combo.clear()
        self.lev_lat_variable_combo.addItems([str(v) for v in valid_vars])
        self.lev_lat_time_combo.clear()
        self.lev_lat_time_combo.addItems([str(t) for t in valid_times])
        self.lev_lat_lon_combo.clear()
        self.lev_lat_lon_combo.addItems([str(lon) for lon in valid_lons])

        # Populate fields in Lev vs Time page
        self.lev_time_variable_combo.clear()
        self.lev_time_variable_combo.addItems([str(v) for v in valid_vars])
        self.lev_time_lat_combo.clear()
        self.lev_time_lat_combo.addItems([str(lat) for lat in valid_lats])
        self.lev_time_lon_combo.clear()
        self.lev_time_lon_combo.addItems([str(lon) for lon in valid_lons])

        # Populate fields in Lat vs Time page
        self.lat_time_variable_combo.clear()
        self.lat_time_variable_combo.addItems([str(v) for v in valid_vars])
        self.lat_time_level_combo.clear()
        self.lat_time_level_combo.addItems([str(lev) for lev in valid_levels])
        self.lat_time_lon_combo.clear()
        self.lat_time_lon_combo.addItems([str(lon) for lon in valid_lons])

        
    def on_plot(self):
        plot_type = self.plot_type_combo.currentText()
        fig = None
        if plot_type == "Lat vs Lon":
            params = {
                "datasets": self.selected_dataset,
                "variable_name": self.lat_lon_variable_combo.currentText(),
                "time": self.lat_lon_time_combo.currentText(),
                "level": self.lat_lon_level_combo.currentText(),
                "variable_unit": self.lat_lon_unit_input.text() or None,
                "center_longitude": float(self.lat_lon_center_lon_input.text()) if self.lat_lon_center_lon_input.text() else 0,
                "contour_intervals": int(self.lat_lon_contour_intervals_input.text()) if self.lat_lon_contour_intervals_input.text() else None,
                "contour_value": int(self.lat_lon_contour_value_input.text()) if self.lat_lon_contour_value_input.text() else None,
                "symmetric_interval": self.lat_lon_symmetric_checkbox.isChecked(),
                "cmap_color": self.lat_lon_cmap_input.text() or None,
                "cmap_lim_min": float(self.lat_lon_cmap_lim_min_input.text()) if self.lat_lon_cmap_lim_min_input.text() else None,
                "cmap_lim_max": float(self.lat_lon_cmap_lim_max_input.text()) if self.lat_lon_cmap_lim_max_input.text() else None,
                "line_color": self.lat_lon_line_color_input.text() or None,
                "coastlines": self.lat_lon_coastlines_checkbox.isChecked(),
                "nightshade": self.lat_lon_nightshade_checkbox.isChecked(),
                "gm_equator": self.lat_lon_gm_equator_checkbox.isChecked(),
                "latitude_minimum": float(self.lat_lon_lat_min_input.text()) if self.lat_lon_lat_min_input.text() else None,
                "latitude_maximum": float(self.lat_lon_lat_max_input.text()) if self.lat_lon_lat_max_input.text() else None,
                "longitude_minimum": float(self.lat_lon_lon_min_input.text()) if self.lat_lon_lon_min_input.text() else None,
                "longitude_maximum": float(self.lat_lon_lon_max_input.text()) if self.lat_lon_lon_max_input.text() else None,
                "clean_plot": self.lat_lon_clean_checkbox.isChecked()
            }
            fig, variable_unit, center_longitude, contour_intervals, contour_value, symmetric_interval, cmap_color, cmap_lim_min, cmap_lim_max, line_color, latitude_minimum, latitude_maximum, longitude_minimum, longitude_maximum, contour_filled, unique_lons, unique_lats, variable_values  = plt_lat_lon(**params)

            self.lat_lon_unit_input.setPlaceholderText(str(variable_unit))
            self.lat_lon_center_lon_input.setPlaceholderText(str(center_longitude))
            self.lat_lon_contour_intervals_input.setPlaceholderText(str(contour_intervals))
            self.lat_lon_contour_value_input.setPlaceholderText(str(contour_value))
            self.lat_lon_symmetric_checkbox.setChecked(symmetric_interval)
            self.lat_lon_cmap_input.setPlaceholderText(cmap_color)
            self.lat_lon_cmap_lim_min_input.setPlaceholderText(str(cmap_lim_min))
            self.lat_lon_cmap_lim_max_input.setPlaceholderText(str(cmap_lim_max))
            self.lat_lon_line_color_input.setPlaceholderText(line_color)
            self.lat_lon_lat_min_input.setPlaceholderText(str(latitude_minimum))
            self.lat_lon_lat_max_input.setPlaceholderText(str(latitude_maximum))
            self.lat_lon_lon_min_input.setPlaceholderText(str(longitude_minimum))
            self.lat_lon_lon_max_input.setPlaceholderText(str(longitude_maximum))

            # Attach the mplcursors cursor to the filled contour collections:
            cursor = mplcursors.cursor(contour_filled.collections, hover=True)
            @cursor.connect("add")
            def on_add(sel):
                # sel.target gives the coordinates where the cursor is
                x, y = sel.target
                # Adjust longitude using center_longitude
                if (x + params["center_longitude"]) > 180:
                    adjusted_lon = - (360 - x - params["center_longitude"])
                elif (x + params["center_longitude"]) < -180:
                    adjusted_lon = x + 360 + params["center_longitude"]
                else:
                    adjusted_lon = x + params["center_longitude"]
                # Find nearest indices in unique_lons and unique_lats
                lon_idx = (np.abs(unique_lons - adjusted_lon)).argmin()
                lat_idx = (np.abs(unique_lats - y)).argmin()
                # Retrieve corresponding value from variable_values array
                value = variable_values[lat_idx, lon_idx]
                # Set annotation text with details
                sel.annotation.set(
                    text=f"Lon: {unique_lons[lon_idx]:.2f}°\nLat: {unique_lats[lat_idx]:.2f}°\n{params['variable_name']}: {value:.2e} {variable_unit}"
                )
                sel.annotation.get_bbox_patch().set(alpha=0.9)
                
        elif plot_type == "Lev vs Var":
            params = {
                "datasets": self.selected_dataset,
                "variable_name": self.lev_var_variable_combo.currentText(),
                "latitude": float(self.lev_var_lat_combo.currentText()) if self.lev_var_lat_combo.currentText() else None,
                "time": self.lev_var_time_combo.currentText(),
                "longitude": float(self.lev_var_lon_combo.currentText()) if self.lev_var_lon_combo.currentText() else None,
                "log_level": self.lev_var_log_checkbox.isChecked(),
                "variable_unit": self.lev_var_unit_input.text() or None,
                "level_minimum": float(self.lev_var_level_min_input.text()) if self.lev_var_level_min_input.text() else None,
                "level_maximum": float(self.lev_var_level_max_input.text()) if self.lev_var_level_max_input.text() else None,
                "clean_plot": self.lev_var_clean_checkbox.isChecked()
            }
            fig, variable_unit, level_minimum, level_maximum = plt_lev_var(**params)
            self.lev_var_unit_input.setPlaceholderText(str(variable_unit))
            self.lev_var_level_min_input.setPlaceholderText(str(level_minimum))
            self.lev_var_level_max_input.setPlaceholderText(str(level_maximum))

            cursor = mplcursors.cursor(fig, hover=True)
            @cursor.connect("add")
            def on_add(sel):
                # Get the x (variable value) and y (level) from the cursor's target
                x, y = sel.target
                
                # Set annotation text to show level and variable value
                sel.annotation.set(
                    text=f"Level: {y:.2f} \n{params['variable_name']}: {x:.2e} {variable_unit}")
                
                # Customize the appearance of the annotation box
                sel.annotation.get_bbox_patch().set(alpha=0.9)

        elif plot_type == "Lev vs Lon":
            params = {
                "datasets": self.selected_dataset,
                "variable_name": self.lev_lon_variable_combo.currentText(),
                "latitude": float(self.lev_lon_lat_combo.currentText()) if self.lev_lon_lat_combo.currentText() else None,
                "time": self.lev_lon_time_combo.currentText(),
                "log_level": self.lev_lon_log_checkbox.isChecked(),
                "variable_unit": self.lev_lon_unit_input.text() or None,
                "contour_intervals": int(self.lev_lon_contour_intervals_input.text()) if self.lev_lon_contour_intervals_input.text() else None,
                "contour_value": int(self.lev_lon_contour_value_input.text()) if self.lev_lon_contour_value_input.text() else None,
                "symmetric_interval": self.lev_lon_symmetric_checkbox.isChecked(),
                "cmap_color": self.lev_lon_cmap_input.text() or None,
                "cmap_lim_min": float(self.lev_lon_cmap_lim_min_input.text()) if self.lev_lon_cmap_lim_min_input.text() else None,
                "cmap_lim_max": float(self.lev_lon_cmap_lim_max_input.text()) if self.lev_lon_cmap_lim_max_input.text() else None,
                "line_color": self.lev_lon_line_color_input.text() or None,
                "level_minimum": float(self.lev_lon_level_min_input.text()) if self.lev_lon_level_min_input.text() else None,
                "level_maximum": float(self.lev_lon_level_max_input.text()) if self.lev_lon_level_max_input.text() else None,
                "longitude_minimum": float(self.lev_lon_lon_min_input.text()) if self.lev_lon_lon_min_input.text() else None,
                "longitude_maximum": float(self.lev_lon_lon_max_input.text()) if self.lev_lon_lon_max_input.text() else None,
                "clean_plot": self.lev_lon_clean_checkbox.isChecked()
            }
            fig, variable_unit, latitude, time, contour_intervals, contour_value, symmetric_interval, cmap_color, cmap_lim_min, cmap_lim_max, line_color, level_minimum, level_maximum, longitude_minimum, longitude_maximum, contour_filled, unique_lons, unique_levs, variable_values = plt_lev_lon(**params)

            self.lev_lon_unit_input.setPlaceholderText(str(variable_unit))
            self.lev_lon_contour_intervals_input.setPlaceholderText(str(contour_intervals))
            self.lev_lon_contour_value_input.setPlaceholderText(str(contour_value))
            self.lev_lon_symmetric_checkbox.setChecked(symmetric_interval)
            self.lev_lon_cmap_input.setPlaceholderText(cmap_color)
            self.lev_lon_cmap_lim_min_input.setPlaceholderText(str(cmap_lim_min))
            self.lev_lon_cmap_lim_max_input.setPlaceholderText(str(cmap_lim_max))
            self.lev_lon_line_color_input.setPlaceholderText(line_color)
            self.lev_lon_level_min_input.setPlaceholderText(str(level_minimum))
            self.lev_lon_level_max_input.setPlaceholderText(str(level_maximum))
            self.lev_lon_lon_min_input.setPlaceholderText(str(longitude_minimum))
            self.lev_lon_lon_max_input.setPlaceholderText(str(longitude_maximum))
            center_longitude = 0
            cursor = mplcursors.cursor(contour_filled.collections, hover=True)
            @cursor.connect("add")
            def on_add(sel):
                # sel.target gives the coordinates where the cursor is
                x, y = sel.target
                # Find the nearest longitude index
                if (x + center_longitude) > 180:
                    adjusted_lon =  - (360 -x -center_longitude)
                elif (x + center_longitude) < -180:
                    adjusted_lon = x + 360 + center_longitude #180 + (x + center_longitude) 
                else:
                    adjusted_lon = x + center_longitude
                
                lon_idx = (np.abs(unique_lons - adjusted_lon)).argmin() 
                
                # Find the nearest latitude index
                level_idx = (np.abs(unique_levs - y)).argmin()
                
                # Retrieve the corresponding value
                value = variable_values[level_idx, lon_idx]
                
                # Set annotation text
                sel.annotation.set(
                    text=f"Lon: {unique_lons[lon_idx]:.2f}°\nLev: {unique_levs[level_idx]:.2f}°\n{params['variable_name']}: {value:.2e} {variable_unit}"
                )
                
                # Customize annotation appearance
                sel.annotation.get_bbox_patch().set(alpha=0.9)

        elif plot_type == "Lev vs Lat":
            params = {
                "datasets": self.selected_dataset,
                "variable_name": self.lev_lat_variable_combo.currentText(),
                "time": self.lev_lat_time_combo.currentText(),
                "longitude": float(self.lev_lat_lon_combo.currentText()) if self.lev_lat_lon_combo.currentText() else None,
                "log_level": self.lev_lat_log_checkbox.isChecked(),
                "variable_unit": self.lev_lat_unit_input.text() or None,
                "contour_intervals": int(self.lev_lat_contour_intervals_input.text()) if self.lev_lat_contour_intervals_input.text() else None,
                "contour_value": int(self.lev_lat_contour_value_input.text()) if self.lev_lat_contour_value_input.text() else None,
                "symmetric_interval": self.lev_lat_symmetric_checkbox.isChecked(),
                "cmap_color": self.lev_lat_cmap_input.text() or None,
                "cmap_lim_min": float(self.lev_lat_cmap_lim_min_input.text()) if self.lev_lat_cmap_lim_min_input.text() else None,
                "cmap_lim_max": float(self.lev_lat_cmap_lim_max_input.text()) if self.lev_lat_cmap_lim_max_input.text() else None,
                "line_color": self.lev_lat_line_color_input.text() or None,
                "level_minimum": float(self.lev_lat_level_min_input.text()) if self.lev_lat_level_min_input.text() else None,
                "level_maximum": float(self.lev_lat_level_max_input.text()) if self.lev_lat_level_max_input.text() else None,
                "latitude_minimum": float(self.lev_lat_lat_min_input.text()) if self.lev_lat_lat_min_input.text() else None,
                "latitude_maximum": float(self.lev_lat_lat_max_input.text()) if self.lev_lat_lat_max_input.text() else None,
                "clean_plot": self.lev_lat_clean_checkbox.isChecked()
            }
            fig,  variable_unit, time, contour_intervals, contour_value, symmetric_interval, cmap_color, cmap_lim_min, cmap_lim_max, line_color, level_minimum, level_maximum, latitude_minimum, latitude_maximum, contour_filled, unique_lats, unique_levs, variable_values = plt_lev_lat(**params)

            self.lev_lat_unit_input.setPlaceholderText(str(variable_unit))
            self.lev_lat_contour_intervals_input.setPlaceholderText(str(contour_intervals))
            self.lev_lat_contour_value_input.setPlaceholderText(str(contour_value))
            self.lev_lat_symmetric_checkbox.setChecked(symmetric_interval)
            self.lev_lat_cmap_input.setPlaceholderText(cmap_color)
            self.lev_lat_cmap_lim_min_input.setPlaceholderText(str(cmap_lim_min))
            self.lev_lat_cmap_lim_max_input.setPlaceholderText(str(cmap_lim_max))
            self.lev_lat_line_color_input.setPlaceholderText(line_color)
            self.lev_lat_level_min_input.setPlaceholderText(str(level_minimum))
            self.lev_lat_level_max_input.setPlaceholderText(str(level_maximum))
            self.lev_lat_lat_min_input.setPlaceholderText(str(latitude_minimum))
            self.lev_lat_lat_max_input.setPlaceholderText(str(latitude_maximum))

            cursor = mplcursors.cursor(contour_filled.collections, hover=True)
            @cursor.connect("add")
            def on_add(sel):
                # sel.target gives the coordinates where the cursor is
                x, y = sel.target
                
                lat_idx = (np.abs(unique_lats - x)).argmin() 
                
                # Find the nearest latitude index
                level_idx = (np.abs(unique_levs - y)).argmin()
                
                # Retrieve the corresponding value
                value = variable_values[level_idx, lat_idx]
                
                # Set annotation text
                sel.annotation.set(
                    text=f"Lat: {unique_lats[lat_idx]:.2f}°\nLev: {unique_levs[level_idx]:.2f}°\n{params['variable_name']}: {value:.2e} {variable_unit}"
                )
                
                # Customize annotation appearance
                sel.annotation.get_bbox_patch().set(alpha=0.9)

        elif plot_type == "Lev vs Time":
            params = {
                "datasets": self.selected_dataset,
                "variable_name": self.lev_time_variable_combo.currentText(),
                "latitude": float(self.lev_time_lat_combo.currentText()) if self.lev_time_lat_combo.currentText() else None,
                "longitude": float(self.lev_time_lon_combo.currentText()) if self.lev_time_lon_combo.currentText() else None,
                "log_level": self.lev_time_log_checkbox.isChecked(),
                "variable_unit": self.lev_time_unit_input.text() or None,
                "contour_intervals": int(self.lev_time_contour_intervals_input.text()) if self.lev_time_contour_intervals_input.text() else None,
                "contour_value": int(self.lev_time_contour_value_input.text()) if self.lev_time_contour_value_input.text() else None,
                "symmetric_interval": self.lev_time_symmetric_checkbox.isChecked(),
                "cmap_color": self.lev_time_cmap_input.text() or None,
                "cmap_lim_min": float(self.lev_time_cmap_lim_min_input.text()) if self.lev_time_cmap_lim_min_input.text() else None,
                "cmap_lim_max": float(self.lev_time_cmap_lim_max_input.text()) if self.lev_time_cmap_lim_max_input.text() else None,
                "line_color": self.lev_time_line_color_input.text() or None,
                "level_minimum": float(self.lev_time_level_min_input.text()) if self.lev_time_level_min_input.text() else None,
                "level_maximum": float(self.lev_time_level_max_input.text()) if self.lev_time_level_max_input.text() else None,
                "mtime_minimum": self.lev_time_mtime_min_input.text() or None,
                "mtime_maximum": self.lev_time_mtime_max_input.text() or None,
                "clean_plot": self.lev_time_clean_checkbox.isChecked()
            }
            fig = plt_lev_time(**params)
        elif plot_type == "Lat vs Time":
            params = {
                "datasets": self.selected_dataset,
                "variable_name": self.lat_time_variable_combo.currentText(),
                "level": self.lat_time_level_combo.currentText(),
                "longitude": float(self.lat_time_lon_combo.currentText()) if self.lat_time_lon_combo.currentText() else None,
                "variable_unit": self.lat_time_unit_input.text() or None,
                "contour_intervals": int(self.lat_time_contour_intervals_input.text()) if self.lat_time_contour_intervals_input.text() else None,
                "contour_value": int(self.lat_time_contour_value_input.text()) if self.lat_time_contour_value_input.text() else None,
                "symmetric_interval": self.lat_time_symmetric_checkbox.isChecked(),
                "cmap_color": self.lat_time_cmap_input.text() or None,
                "cmap_lim_min": float(self.lat_time_cmap_lim_min_input.text()) if self.lat_time_cmap_lim_min_input.text() else None,
                "cmap_lim_max": float(self.lat_time_cmap_lim_max_input.text()) if self.lat_time_cmap_lim_max_input.text() else None,
                "line_color": self.lat_time_line_color_input.text() or None,
                "latitude_minimum": float(self.lat_time_lat_min_input.text()) if self.lat_time_lat_min_input.text() else None,
                "latitude_maximum": float(self.lat_time_lat_max_input.text()) if self.lat_time_lat_max_input.text() else None,
                "mtime_minimum": self.lat_time_mtime_min_input.text() or None,
                "mtime_maximum": self.lat_time_mtime_max_input.text() or None,
                "clean_plot": self.lat_time_clean_checkbox.isChecked()
            }
            fig = plt_lat_time(**params)
        else:
            QMessageBox.warning(self, "Plot Type Error", "Unknown plot type selected.")
            return
        
        # Update the canvas with the new figure.
        self.plot_layout.removeWidget(self.canvas)
        self.canvas.deleteLater()
        self.canvas = FigureCanvas(fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_layout.addWidget(self.canvas)
        self.canvas.draw()
    
    def on_save_image(self):
        """Opens a file dialog to save the current figure as an image."""
        if self.canvas is None or self.canvas.figure is None:
            QMessageBox.warning(self, "Save Error", "No plot available to save.")
            return
        
        # Open file dialog for save location
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot As Image",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)",
            options=options
        )
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

# --- Main entry point ---
if __name__ == '__main__':
    main()