#!/usr/bin/env python3
"""
Comprehensive Real-time RSSI Spectrum GUI for Linux using PyQt5 + Matplotlib.

Features:
- Auto-detects serial ports (/dev/ttyACM*, /dev/ttyUSB*)
- Start/Stop scan
- Real-time RSSI vs Frequency graph (Matplotlib embedded)
- Adjustable frequency/rssi range and step size
- Logging to file with timestamps
- Configurable scan interval
- CSV export of latest RSSI dataset
- Error handling and status reporting

Dependencies:
    pip install pyqt5 matplotlib numpy pyairview

Run:
    python3 rssi_spectrum_gui.py

Notes:
- Expects pyairview API compatible with connect(), start_scan(callback), stop_scan(), is_scanning(), disconnect().
- Run with permissions to access /dev/tty* (dialout group or sudo).
"""

import sys
import glob
import csv
import os
import threading
import time
from datetime import datetime
from queue import Queue, Empty

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QFileDialog, QCheckBox
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# External scanning library
import pyairview

DEFAULT_START_FREQ = 2399.0
DEFAULT_END_FREQ = 2485.0
DEFAULT_FREQ_STEP = 0.5
DEFAULT_RSSI_MIN = -100
DEFAULT_RSSI_MAX = -40
DEFAULT_SCAN_INTERVAL = 2.5  # seconds


class SpectrumCanvas(FigureCanvas):
    def __init__(self, start_freq, end_freq, rssi_min, rssi_max, freq_step):
        self.fig = Figure(figsize=(8, 4))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.rssi_min = rssi_min
        self.rssi_max = rssi_max
        self.freq_step = freq_step
        self._init_plot()

    def _init_plot(self):
        self.ax.set_title("RSSI Spectrum Analyzer")
        self.ax.set_xlabel("Frequency (MHz)")
        self.ax.set_ylabel("RSSI Level (dBm)")
        self.ax.grid(True)
        self.ax.set_xlim(self.start_freq, self.end_freq)
        self.ax.set_ylim(self.rssi_min, self.rssi_max)
        self.draw()

    def update_limits(self, start_freq, end_freq, rssi_min, rssi_max, freq_step):
        self.start_freq, self.end_freq = start_freq, end_freq
        self.rssi_min, self.rssi_max = rssi_min, rssi_max
        self.freq_step = freq_step
        self.ax.set_xlim(start_freq, end_freq)
        self.ax.set_ylim(rssi_min, rssi_max)

    def plot_rssi(self, rssi_values):
        if not rssi_values:
            return
        freqs = np.linspace(self.start_freq, self.end_freq, len(rssi_values))
        self.ax.cla()
        self._init_plot()
        self.ax.plot(freqs, rssi_values, marker='o', linestyle='-', color='b', alpha=0.8)
        self.draw()


class ScannerThread(threading.Thread):
    def __init__(self, port, interval, callback, stop_event):
        super().__init__(daemon=True)
        self.port = port
        self.interval = interval
        self.callback = callback
        self.stop_event = stop_event

    def run(self):
        try:
            if not pyairview.connect(self.port):
                self.callback('__error__', f'Failed to connect to {self.port}')
                return

            self.callback('__status__', f'Connected to {self.port}')
            pyairview.start_scan(callback=self._rssi_callback)

            while not self.stop_event.is_set() and pyairview.is_scanning():
                time.sleep(self.interval)

            if pyairview.is_scanning():
                pyairview.stop_scan()
        except Exception as e:
            self.callback('__error__', f'Scanner error: {e}')
        finally:
            try:
                pyairview.disconnect()
            except Exception:
                pass
            self.callback('__status__', 'Disconnected')

    def _rssi_callback(self, rssi_list):
        self.callback('rssi', rssi_list)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('RSSI Spectrum — Extended GUI')
        self.setMinimumSize(1000, 600)

        self.plot_queue = Queue()
        self.scanner_thread = None
        self.scanner_stop_event = threading.Event()

        self.start_freq = DEFAULT_START_FREQ
        self.end_freq = DEFAULT_END_FREQ
        self.freq_step = DEFAULT_FREQ_STEP
        self.rssi_min = DEFAULT_RSSI_MIN
        self.rssi_max = DEFAULT_RSSI_MAX
        self.scan_interval = DEFAULT_SCAN_INTERVAL
        self.logging_enabled = False
        self.latest_data = []

        self._init_ui()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._update_plot)
        self.timer.start()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout()
        central.setLayout(layout)

        self.canvas = SpectrumCanvas(self.start_freq, self.end_freq, self.rssi_min, self.rssi_max, self.freq_step)
        layout.addWidget(self.canvas, stretch=3)

        controls = QVBoxLayout()
        layout.addLayout(controls, stretch=1)

        port_group = QGroupBox('Device')
        pg_layout = QVBoxLayout()
        self.port_combo = QComboBox()
        self._refresh_ports()
        refresh = QPushButton('Refresh')
        refresh.clicked.connect(self._refresh_ports)
        pg_layout.addWidget(self.port_combo)
        pg_layout.addWidget(refresh)
        port_group.setLayout(pg_layout)
        controls.addWidget(port_group)

        cfg_group = QGroupBox('Plot Settings')
        cfg_layout = QVBoxLayout()
        self.start_spin = QDoubleSpinBox(); self.start_spin.setRange(100,10000); self.start_spin.setValue(self.start_freq)
        self.end_spin = QDoubleSpinBox(); self.end_spin.setRange(100,10000); self.end_spin.setValue(self.end_freq)
        self.step_spin = QDoubleSpinBox(); self.step_spin.setRange(0.01,100); self.step_spin.setValue(self.freq_step)
        self.min_spin = QSpinBox(); self.min_spin.setRange(-200,0); self.min_spin.setValue(self.rssi_min)
        self.max_spin = QSpinBox(); self.max_spin.setRange(-200,0); self.max_spin.setValue(self.rssi_max)
        self.int_spin = QDoubleSpinBox(); self.int_spin.setRange(0.1,10); self.int_spin.setValue(self.scan_interval)
        for s in (self.start_spin,self.end_spin,self.step_spin,self.min_spin,self.max_spin,self.int_spin):
            s.valueChanged.connect(self._update_config)
        for label, w in [('Start Freq',self.start_spin),('End Freq',self.end_spin),('Step (MHz)',self.step_spin),('RSSI Min',self.min_spin),('RSSI Max',self.max_spin),('Interval (s)',self.int_spin)]:
            cfg_layout.addWidget(QLabel(label)); cfg_layout.addWidget(w)
        cfg_group.setLayout(cfg_layout)
        controls.addWidget(cfg_group)

        self.log_checkbox = QCheckBox('Enable logging')
        self.log_checkbox.stateChanged.connect(self._toggle_logging)
        self.export_btn = QPushButton('Export CSV')
        self.export_btn.clicked.connect(self._export_csv)
        controls.addWidget(self.log_checkbox)
        controls.addWidget(self.export_btn)

        self.start_btn = QPushButton('Start Scan')
        self.stop_btn = QPushButton('Stop Scan')
        self.start_btn.clicked.connect(self._start_scan)
        self.stop_btn.clicked.connect(self._stop_scan)
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)

        self.status_label = QLabel('Ready')
        controls.addStretch(1)
        controls.addWidget(QLabel('Status:'))
        controls.addWidget(self.status_label)

    def _refresh_ports(self):
        ports = sorted(set(glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')))
        self.port_combo.clear()
        self.port_combo.addItems(ports or ['/dev/ttyACM0'])

    def _update_config(self):
        self.start_freq = self.start_spin.value()
        self.end_freq = self.end_spin.value()
        self.freq_step = self.step_spin.value()
        self.rssi_min = self.min_spin.value()
        self.rssi_max = self.max_spin.value()
        self.scan_interval = self.int_spin.value()
        self.canvas.update_limits(self.start_freq, self.end_freq, self.rssi_min, self.rssi_max, self.freq_step)

    def _toggle_logging(self, state):
        self.logging_enabled = bool(state)
        if self.logging_enabled:
            self.status_label.setText('Logging enabled')
        else:
            self.status_label.setText('Logging disabled')

    def _export_csv(self):
        if not self.latest_data:
            self.status_label.setText('No data to export')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Export CSV', os.path.expanduser('~/rssi_export.csv'), 'CSV files (*.csv)')
        if not path:
            return
        freqs = np.linspace(self.start_freq, self.end_freq, len(self.latest_data))
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frequency (MHz)', 'RSSI (dBm)'])
            for fr, val in zip(freqs, self.latest_data):
                writer.writerow([fr, val])
        self.status_label.setText(f'Exported to {path}')

    def _start_scan(self):
        port = self.port_combo.currentText()
        self.scanner_stop_event.clear()
        self.scanner_thread = ScannerThread(port, self.scan_interval, self._scanner_callback, self.scanner_stop_event)
        self.scanner_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText('Scanning...')

    def _stop_scan(self):
        self.scanner_stop_event.set()
        try:
            if pyairview.is_scanning():
                pyairview.stop_scan()
        except Exception:
            pass
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText('Stopped')

    def _scanner_callback(self, tag, data):
        if tag == 'rssi':
            self.plot_queue.put(data)
        elif tag == '__status__':
            QtCore.QMetaObject.invokeMethod(self.status_label, 'setText', QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, data))
        elif tag == '__error__':
            QtCore.QMetaObject.invokeMethod(self.status_label, 'setText', QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, f'ERROR: {data}'))

    def _update_plot(self):
        try:
            while True:
                data = self.plot_queue.get_nowait()
                if isinstance(data, list) and data:
                    self.latest_data = data
                    self.canvas.plot_rssi(data)
                    if self.logging_enabled:
                        self._log_data(data)
        except Empty:
            pass

    def _log_data(self, data):
        freqs = np.linspace(self.start_freq, self.end_freq, len(data))
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_path = os.path.expanduser('~/rssi_log.csv')
        new_file = not os.path.exists(log_path)
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(['Timestamp', 'Frequency (MHz)', 'RSSI (dBm)'])
            for fr, val in zip(freqs, data):
                writer.writerow([timestamp, fr, val])

    def closeEvent(self, e):
        self.scanner_stop_event.set()
        try:
            if pyairview.is_scanning():
                pyairview.stop_scan()
            pyairview.disconnect()
        except Exception:
            pass
        e.accept()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
