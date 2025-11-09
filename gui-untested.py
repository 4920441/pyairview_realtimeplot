#!/usr/bin/env python3
"""
Real-time RSSI Spectrum GUI for Linux using PyQt5 + Matplotlib.

Save as rssi_spectrum_gui.py and run with: python3 rssi_spectrum_gui.py

Dependencies:
    pip install pyqt5 matplotlib numpy pyairview

Functionality:
- Select serial port (auto-detect /dev/ttyACM* and /dev/ttyUSB*)
- Start/Stop scanning
- Real-time plot of RSSI vs frequency (embedded Matplotlib)
- Configure RSSI min/max and frequency step via UI
- Status and basic error handling

Notes:
- The code expects the pyairview library API used in the original script
  (connect(port), start_scan(callback=...), stop_scan(), is_scanning(), disconnect()).
- Run on Linux with appropriate permissions for serial ports (e.g. add user to dialout or use sudo).
"""

import sys
import glob
import threading
import time
from queue import Queue, Empty

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# External scanning library used by original script
import pyairview

# Default constants (can be changed from UI)
DEFAULT_START_FREQ = 2399.0  # MHz
DEFAULT_END_FREQ = 2485.0    # MHz
DEFAULT_FREQ_STEP = 0.5      # MHz
DEFAULT_RSSI_MIN = -100      # dBm
DEFAULT_RSSI_MAX = -40       # dBm


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
        self.line, = self.ax.plot([], [], marker='o', linestyle='-')
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
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.rssi_min = rssi_min
        self.rssi_max = rssi_max
        self.freq_step = freq_step
        self.ax.set_xlim(self.start_freq, self.end_freq)
        self.ax.set_ylim(self.rssi_min, self.rssi_max)

    def plot_rssi(self, rssi_values):
        if not rssi_values:
            return
        num_points = len(rssi_values)
        freqs = np.array([self.start_freq + i * self.freq_step for i in range(num_points)])
        # Truncate or pad frequencies to match rssi_values length
        if freqs.size > num_points:
            freqs = freqs[:num_points]
        elif freqs.size < num_points:
            freqs = np.linspace(self.start_freq, self.end_freq, num_points)

        self.ax.cla()
        self.ax.set_title("RSSI Spectrum Analyzer")
        self.ax.set_xlabel("Frequency (MHz)")
        self.ax.set_ylabel("RSSI Level (dBm)")
        self.ax.grid(True)
        self.ax.set_xlim(self.start_freq, self.end_freq)
        self.ax.set_ylim(self.rssi_min, self.rssi_max)
        self.ax.plot(freqs, rssi_values, marker='o', linestyle='-')
        self.draw()


class ScannerThread(threading.Thread):
    """Thread that starts the scanner and monitors its running state.
    It calls pyairview.start_scan(callback=...) which is assumed to return immediately
    and use the callback to deliver scans. This thread monitors is_scanning() and
    stops cleanly when requested.
    """

    def __init__(self, port, callback, stop_event):
        super().__init__(daemon=True)
        self.port = port
        self.callback = callback
        self.stop_event = stop_event

    def run(self):
        try:
            connected = pyairview.connect(self.port)
            if not connected:
                # pyairview.connect may raise or return False
                self.callback('__error__', f'Failed to connect to {self.port}')
                return

            self.callback('__status__', f'Connected to {self.port} — starting scan')
            pyairview.start_scan(callback=self._pyairview_callback)

            # Monitor scanning until stop requested or pyairview reports not scanning
            while not self.stop_event.is_set() and pyairview.is_scanning():
                time.sleep(0.5)

            # If stop requested and still scanning, request stop
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

    def _pyairview_callback(self, rssi_list):
        # Deliver real RSSI data via the UI callback
        self.callback('rssi', rssi_list)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('RSSI Spectrum — GUI')
        self.setMinimumSize(900, 520)

        # Internal queue for incoming plot data
        self.plot_queue = Queue()

        # Default parameters
        self.start_freq = DEFAULT_START_FREQ
        self.end_freq = DEFAULT_END_FREQ
        self.freq_step = DEFAULT_FREQ_STEP
        self.rssi_min = DEFAULT_RSSI_MIN
        self.rssi_max = DEFAULT_RSSI_MAX

        # Scanner control
        self.scanner_thread = None
        self.scanner_stop_event = threading.Event()

        self._init_ui()

        # Timer to poll queue and update plot
        self.update_timer = QtCore.QTimer()
        self.update_timer.setInterval(200)  # ms
        self.update_timer.timeout.connect(self._process_plot_queue)
        self.update_timer.start()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        # Left: plot
        self.canvas = SpectrumCanvas(self.start_freq, self.end_freq, self.rssi_min, self.rssi_max, self.freq_step)
        main_layout.addWidget(self.canvas, stretch=3)

        # Right: controls
        controls = QVBoxLayout()
        main_layout.addLayout(controls, stretch=1)

        # Port selection
        port_group = QGroupBox('Device')
        port_layout = QVBoxLayout()
        port_group.setLayout(port_layout)
        self.port_combo = QComboBox()
        self._refresh_ports()
        refresh_btn = QPushButton('Refresh ports')
        refresh_btn.clicked.connect(self._refresh_ports)
        port_layout.addWidget(QLabel('Serial port'))
        port_layout.addWidget(self.port_combo)
        port_layout.addWidget(refresh_btn)
        controls.addWidget(port_group)

        # Frequency / RSSI settings
        cfg_group = QGroupBox('Plot settings')
        cfg_layout = QVBoxLayout()
        cfg_group.setLayout(cfg_layout)

        self.start_freq_spin = QDoubleSpinBox()
        self.start_freq_spin.setRange(100.0, 10000.0)
        self.start_freq_spin.setValue(self.start_freq)
        self.start_freq_spin.setDecimals(3)
        self.start_freq_spin.valueChanged.connect(self._update_limits_from_ui)

        self.end_freq_spin = QDoubleSpinBox()
        self.end_freq_spin.setRange(100.0, 10000.0)
        self.end_freq_spin.setValue(self.end_freq)
        self.end_freq_spin.setDecimals(3)
        self.end_freq_spin.valueChanged.connect(self._update_limits_from_ui)

        self.freq_step_spin = QDoubleSpinBox()
        self.freq_step_spin.setRange(0.001, 100.0)
        self.freq_step_spin.setValue(self.freq_step)
        self.freq_step_spin.setDecimals(3)
        self.freq_step_spin.valueChanged.connect(self._update_limits_from_ui)

        self.rssi_min_spin = QSpinBox()
        self.rssi_min_spin.setRange(-200, 0)
        self.rssi_min_spin.setValue(self.rssi_min)
        self.rssi_min_spin.valueChanged.connect(self._update_limits_from_ui)

        self.rssi_max_spin = QSpinBox()
        self.rssi_max_spin.setRange(-200, 0)
        self.rssi_max_spin.setValue(self.rssi_max)
        self.rssi_max_spin.valueChanged.connect(self._update_limits_from_ui)

        cfg_layout.addWidget(QLabel('Start frequency (MHz)'))
        cfg_layout.addWidget(self.start_freq_spin)
        cfg_layout.addWidget(QLabel('End frequency (MHz)'))
        cfg_layout.addWidget(self.end_freq_spin)
        cfg_layout.addWidget(QLabel('Frequency step (MHz)'))
        cfg_layout.addWidget(self.freq_step_spin)
        cfg_layout.addWidget(QLabel('RSSI min (dBm)'))
        cfg_layout.addWidget(self.rssi_min_spin)
        cfg_layout.addWidget(QLabel('RSSI max (dBm)'))
        cfg_layout.addWidget(self.rssi_max_spin)

        controls.addWidget(cfg_group)

        # Start/Stop buttons
        self.start_btn = QPushButton('Start scan')
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn = QPushButton('Stop scan')
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)

        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)

        # Status
        self.status_label = QLabel('Ready')
        controls.addStretch(1)
        controls.addWidget(QLabel('Status'))
        controls.addWidget(self.status_label)

    def _refresh_ports(self):
        # Find common serial devices on Linux
        ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
        ports = sorted(set(ports))
        self.port_combo.clear()
        if not ports:
            self.port_combo.addItem('/dev/ttyACM0')
        else:
            for p in ports:
                self.port_combo.addItem(p)

    def _update_limits_from_ui(self):
        self.start_freq = float(self.start_freq_spin.value())
        self.end_freq = float(self.end_freq_spin.value())
        self.freq_step = float(self.freq_step_spin.value())
        self.rssi_min = int(self.rssi_min_spin.value())
        self.rssi_max = int(self.rssi_max_spin.value())
        # Validate
        if self.start_freq >= self.end_freq:
            self.status_label.setText('Start frequency must be < end frequency')
            return
        if self.rssi_min >= self.rssi_max:
            self.status_label.setText('RSSI min must be < RSSI max')
            return
        self.canvas.update_limits(self.start_freq, self.end_freq, self.rssi_min, self.rssi_max, self.freq_step)

    def _on_start(self):
        port = self.port_combo.currentText()
        self.scanner_stop_event.clear()
        # Create and start scanner thread
        self.scanner_thread = ScannerThread(port=port, callback=self._scanner_callback_from_thread, stop_event=self.scanner_stop_event)
        self.scanner_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText('Starting...')

    def _on_stop(self):
        self.scanner_stop_event.set()
        # If pyairview exposes stop_scan directly, call it to speed up stop
        try:
            if pyairview.is_scanning():
                pyairview.stop_scan()
        except Exception:
            pass

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText('Stopping...')

    def _scanner_callback_from_thread(self, tag, payload):
        # This method is called from the scanner thread.
        # Forward messages to the Qt main thread via signals (use Qt's event loop)
        if tag == 'rssi':
            self.plot_queue.put(payload)
        elif tag == '__status__':
            QtCore.QMetaObject.invokeMethod(self.status_label, 'setText', QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, payload))
        elif tag == '__error__':
            QtCore.QMetaObject.invokeMethod(self.status_label, 'setText', QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, f'ERROR: {payload}'))

    def _process_plot_queue(self):
        # Called by QTimer in the GUI thread
        try:
            while True:
                data = self.plot_queue.get_nowait()
                # Expect data to be a list of integers
                if isinstance(data, list) and data:
                    # Plot values
                    self.canvas.plot_rssi(data)
                    self.status_label.setText(f'Last update: {len(data)} points')
        except Empty:
            pass

    def closeEvent(self, event):
        # Ensure scanner stops and resources cleaned
        self.scanner_stop_event.set()
        try:
            if pyairview.is_scanning():
                pyairview.stop_scan()
            pyairview.disconnect()
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
