# -*- coding: utf-8 -*-
"""
DAQ Dual-Gate Quantum Transport Mapping  —  v1.0
============================================
Author: Xie Jiao  <xequalux@gmail.com>
Data: 2026-04-15

Hardware setup this build is configured for
-------------------------------------------
For each measurement pair:

    sample electrode pair  ─┐
                            ├─ SR560 (Source = A−B, gain G)
    sample electrode pair  ─┘
                                       │
                                       ▼  OUTPUT (50 Ω)
                            SR830 (Signal Input = A, AC, Float)
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
                   CH1 OUTPUT (BNC)          CH2 OUTPUT (BNC)
                  Display 1 = R              Display 2 = θ
                          │                         │
                          ▼                         ▼
                     DAQ AI(2k)                DAQ AI(2k+1)
                       (RSE)                     (RSE)

Multiple SR830s feed multiple AI pairs in parallel; the program does not
care how many SR830s you have, only that for every "kind = R" channel
the corresponding SR830 has Display 1 set to R, and for every
"kind = Phase" channel the corresponding SR830 has Display 2 set to θ.

CRITICAL OPERATOR CHECKLIST (verify each cooldown):
  [ ] Every SR830 used for a kind=R channel:  Display 1 → R   (NOT X)
  [ ] Every SR830 used for a kind=Phase chan: Display 2 → θ   (NOT Y)
  [ ] Every SR830 SIGNAL INPUT: A,  AC coupling,  Float ground
  [ ] Every SR830 sensitivity dial matches the GUI sens field
      for its corresponding kind=R channel
  [ ] Every SR560 voltage gain dial matches the GUI gain field
  [ ] Every SR560 input coupling = AC, Source = A−B
"""

import sys
import os
import time
import csv
import json
import argparse
import traceback
import numpy as np
from dataclasses import dataclass

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QGroupBox, QGridLayout, QFileDialog,
    QMessageBox, QCheckBox, QComboBox, QSplitter, QPlainTextEdit, QProgressBar,
    QStatusBar, QAction, QScrollArea, QFrame, QToolButton, QMenu,
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QSettings, QRectF
from PyQt5.QtGui import QKeySequence, QTransform
import pyqtgraph as pg


# =====================================================================
# Globals & constants
# =====================================================================
APP_NAME       = "DAQ Dual-Gate Quantum Transport Mapping"
APP_VERSION    = "v1.0"
ORG_NAME       = "lab.transport"
SETTINGS_NAME  = "dual_gate_mapping_v1"   # kept = v1.0 so user settings persist

# Set from --demo command-line flag in main(). Default: real hardware.
DEMO_MODE      = False

NUM_AI         = 8
NUM_LINE_SLOTS = 4
NUM_MAP_SLOTS  = 4
DIRECTIONS     = ('fwd', 'bwd')

# Sweep modes
SWEEP_MODE_VTGVBG = 'vtg_vbg'
SWEEP_MODE_ND     = 'nd'

# DAQ defaults
DAQ_AI_RATE_HZ   = 2000.0       # Hz, AI sample-clock rate for finite acq
DAQ_DEFAULT_NAVG = 30           # samples averaged per point
DAQ_AI_TERMINAL  = 'RSE'        # 'DIFF' | 'RSE' | 'NRSE'
                                # 'RSE' matches the standard wiring:
                                # SR830 CH1/CH2 BNC center → DAQ AIx,
                                # BNC shield → DAQ AGND.
                                # If you re-wire all 8 BNCs differentially
                                # (center → AIx+, shield → AIx−) change
                                # this to 'DIFF' for ~10× CMRR improvement.

# SR830 input configuration enums (operator-recorded provenance)
SR830_SOURCE_OPTIONS   = ('A', 'A-B', 'I')
SR830_COUPLING_OPTIONS = ('AC', 'DC')
SR830_GROUND_OPTIONS   = ('Float', 'Ground')
SR830_RESERVE_OPTIONS  = ('Low Noise', 'Normal', 'High Reserve')

# Physical constants (SI)
ELEM_CHARGE = 1.602176634e-19   # C
EPS0        = 8.8541878128e-12  # F/m

# n-D display ↔ SI scaling
# User types n in 1e12 cm^-2; 1e12 cm^-2 = 1e12 × 1e4 m^-2 = 1e16 m^-2
N_DISPLAY_TO_M2 = 1e16
# User types D in V/nm; 1 V/nm = 1e9 V/m
D_DISPLAY_TO_VM = 1e9

KIND_OPTIONS  = ('R', 'Phase', 'Voltage')
KIND_UNIT     = {'R': 'Ω',   'Phase': 'deg', 'Voltage': 'V'}
KIND_CSV_UNIT = {'R': 'Ohm', 'Phase': 'deg', 'Voltage': 'V'}

# SR830 sin-out spec
SR830_VOSC_MIN = 0.004    # V
SR830_VOSC_MAX = 5.0      # V

# Safety: refuse to run with R_series below this
R_SERIES_MIN_OHM = 1e6

# Soft overload threshold on raw DAQ voltage
OVERLOAD_RAW_V = 9.5

# Catppuccin Mocha palette
CT_BASE     = "#1e1e2e"
CT_MANTLE   = "#181825"
CT_SURFACE0 = "#313244"
CT_SURFACE1 = "#45475a"
CT_SURFACE2 = "#585b70"
CT_OVERLAY0 = "#6c7086"
CT_TEXT     = "#cdd6f4"
CT_SUBTEXT0 = "#a6adc8"
CT_BLUE     = "#89b4fa"
CT_SAPPHIRE = "#74c7ec"
CT_GREEN    = "#a6e3a1"
CT_YELLOW   = "#f9e2af"
CT_RED      = "#f38ba8"
CT_MAUVE    = "#cba6f7"
CT_PEACH    = "#fab387"
CT_TEAL     = "#94e2d5"
CT_PINK     = "#f5c2e7"
CT_LAVENDER = "#b4befe"

CHANNEL_COLORS = [
    CT_BLUE, CT_GREEN, CT_PEACH, CT_MAUVE,
    CT_TEAL, CT_PINK, CT_YELLOW, CT_LAVENDER,
]

APP_QSS = f"""
QWidget {{
    background-color: {CT_BASE};
    color: {CT_TEXT};
    font-family: "Segoe UI", "Inter", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
    font-size: 10pt;
}}
QMainWindow, QDialog {{ background-color: {CT_BASE}; }}
QScrollArea {{ background-color: {CT_BASE}; border: none; }}

QGroupBox {{
    border: 1px solid {CT_SURFACE1};
    border-radius: 6px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: {CT_BLUE};
}}

QLineEdit, QComboBox, QPlainTextEdit, QToolButton {{
    background-color: {CT_SURFACE0};
    color: {CT_TEXT};
    border: 1px solid {CT_SURFACE1};
    border-radius: 4px;
    padding: 5px 8px;
    selection-background-color: {CT_SURFACE2};
}}
QLineEdit:focus, QComboBox:focus {{ border: 1px solid {CT_BLUE}; }}
QLineEdit:disabled, QComboBox:disabled {{ color: {CT_OVERLAY0}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {CT_SURFACE0};
    color: {CT_TEXT};
    selection-background-color: {CT_BLUE};
    selection-color: {CT_BASE};
    border: 1px solid {CT_SURFACE1};
}}
QToolButton {{ padding: 5px 10px; }}
QToolButton:hover {{ background-color: {CT_SURFACE1}; }}
QToolButton::menu-indicator {{ image: none; }}

QPushButton {{
    background-color: {CT_SURFACE1};
    color: {CT_TEXT};
    border: 1px solid {CT_SURFACE2};
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 500;
}}
QPushButton:hover  {{ background-color: {CT_SURFACE2}; }}
QPushButton:pressed{{ background-color: {CT_SURFACE1}; }}
QPushButton:disabled {{
    color: {CT_OVERLAY0};
    background-color: {CT_SURFACE0};
    border: 1px solid {CT_SURFACE1};
}}
QPushButton#primary {{
    background-color: {CT_BLUE};
    color: {CT_BASE};
    border: 1px solid {CT_BLUE};
    font-weight: 600;
    padding: 8px 16px;
}}
QPushButton#primary:hover    {{ background-color: {CT_SAPPHIRE}; }}
QPushButton#primary:disabled {{
    background-color: {CT_SURFACE0}; color: {CT_OVERLAY0};
    border: 1px solid {CT_SURFACE1};
}}
QPushButton#danger {{
    background-color: {CT_RED};
    color: {CT_BASE};
    border: 1px solid {CT_RED};
    font-weight: 600;
    padding: 8px 16px;
}}
QPushButton#danger:hover    {{ background-color: #ff9eb3; }}
QPushButton#danger:disabled {{
    background-color: {CT_SURFACE0}; color: {CT_OVERLAY0};
    border: 1px solid {CT_SURFACE1};
}}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border-radius: 3px;
    border: 1px solid {CT_OVERLAY0};
    background-color: {CT_SURFACE0};
}}
QCheckBox::indicator:checked {{
    background-color: {CT_BLUE};
    border: 1px solid {CT_BLUE};
}}

QMenu {{
    background-color: {CT_BASE};
    border: 1px solid {CT_SURFACE1};
    padding: 4px;
}}
QMenu::item {{ padding: 6px 22px 6px 22px; }}
QMenu::item:selected {{ background-color: {CT_SURFACE0}; }}
QMenu::indicator {{ width: 14px; height: 14px; left: 4px; }}

QLabel#sectionLabel {{ color: {CT_SUBTEXT0}; font-size: 9pt; }}
QLabel#metaLabel    {{ color: {CT_SUBTEXT0}; }}
QLabel#valueLabel   {{ color: {CT_YELLOW}; font-weight: 600; }}
QLabel#bigStat      {{ color: {CT_TEXT}; font-size: 13pt; font-weight: 600; }}
QLabel#hintLabel    {{ color: {CT_OVERLAY0}; font-size: 9pt; font-style: italic; }}

QProgressBar {{
    background-color: {CT_SURFACE0};
    border: 1px solid {CT_SURFACE1};
    border-radius: 4px;
    text-align: center;
    color: {CT_TEXT};
    height: 18px;
}}
QProgressBar::chunk {{
    background-color: {CT_BLUE};
    border-radius: 3px;
}}

QPlainTextEdit#eventLog {{
    background-color: {CT_MANTLE};
    border: 1px solid {CT_SURFACE1};
    border-radius: 4px;
    font-family: "Cascadia Code", "Consolas", "Menlo", monospace;
    font-size: 9pt;
}}

QMenuBar {{ background-color: {CT_MANTLE}; color: {CT_TEXT}; padding: 2px; }}
QMenuBar::item:selected {{ background-color: {CT_SURFACE0}; }}

QStatusBar {{
    background-color: {CT_MANTLE};
    color: {CT_SUBTEXT0};
    border-top: 1px solid {CT_SURFACE0};
}}
QStatusBar::item {{ border: none; }}

QSplitter::handle {{ background-color: {CT_SURFACE0}; }}
QSplitter::handle:horizontal {{ width: 4px; }}
QSplitter::handle:vertical   {{ height: 4px; }}

QFrame#topBar {{
    background-color: {CT_MANTLE};
    border-bottom: 1px solid {CT_SURFACE0};
}}
"""


# =====================================================================
# Utility
# =====================================================================
def fmt_dur(seconds):
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "—"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# =====================================================================
# Dataclasses
# =====================================================================
@dataclass
class ChannelConfig:
    """Per-AI channel configuration. ai_index is the physical AI number."""
    ai_index: int
    name:     str
    enabled:  bool
    kind:     str            # 'R' | 'Phase' | 'Voltage'
    sens:     float = 0.0    # only used if kind == 'R'
    gain:     float = 0.0    # only used if kind == 'R'

    @property
    def unit(self):
        return KIND_UNIT[self.kind]

    @property
    def csv_unit(self):
        return KIND_CSV_UNIT[self.kind]

    @property
    def csv_col_name(self):
        return f"{self.name}_{self.csv_unit}"


@dataclass
class VoltageLimits:
    """|Vtg| ≤ Vtg_max  AND  |Vbg| ≤ Vbg_max.

    Enforced in BOTH Vtg-Vbg AND n-D modes.  Target points whose
    physical gate voltages exceed these are SKIPPED (no DAQ write,
    NaN in CSV, transparent in 2D map).  Set to your hBN breakdown
    voltage with a safety margin.
    """
    Vtg_max: float
    Vbg_max: float

    def check(self, Vtg, Vbg):
        eps = 1e-9
        return (abs(Vtg) <= self.Vtg_max + eps and
                abs(Vbg) <= self.Vbg_max + eps)


@dataclass
class GeometryConfig:
    """hBN dual-gate device geometry — needed only for n-D mode.

    All thicknesses in nm, all voltages in V.  n is in m^-2 (SI), D is
    in V/m (SI, the ε₀-divided form), all SI internally.  GUI shows
    1e12 cm^-2 and V/nm display units.
    """
    d_t_nm:  float
    d_b_nm:  float
    eps_hBN: float
    Vtg0:    float       # CNP Vtg (V)
    Vbg0:    float       # CNP Vbg (V)

    @property
    def C_t(self):
        return EPS0 * self.eps_hBN / (self.d_t_nm * 1e-9)

    @property
    def C_b(self):
        return EPS0 * self.eps_hBN / (self.d_b_nm * 1e-9)

    def vgates_to_nD(self, Vtg, Vbg):
        """Forward: (Vtg, Vbg) → (n, D) in SI."""
        dVt = Vtg - self.Vtg0
        dVb = Vbg - self.Vbg0
        n = (self.C_t * dVt + self.C_b * dVb) / ELEM_CHARGE
        D = (self.C_t * dVt - self.C_b * dVb) / (2.0 * EPS0)
        return n, D

    def nD_to_vgates(self, n, D):
        """Inverse: (n, D) in SI → (Vtg, Vbg) in V.

        Matrix inversion of
            [ n·e   ]   [  C_t   C_b ] [ δVtg ]
            [       ] = [             ] [      ]
            [ 2D·ε₀ ]   [  C_t  -C_b ] [ δVbg ]

        det = -2·C_t·C_b, so:
            δVtg = (n·e + 2D·ε₀) / (2·C_t)
            δVbg = (n·e - 2D·ε₀) / (2·C_b)

        D sign convention: positive D points from bottom to top gate
        (i.e. δVtg > 0 → D > 0).  Verify against Wong et al. 2026 and
        the §2 field-theory derivation BEFORE running production data.
        """
        ne = n * ELEM_CHARGE
        twoDeps0 = 2.0 * D * EPS0
        Vtg = self.Vtg0 + (ne + twoDeps0) / (2.0 * self.C_t)
        Vbg = self.Vbg0 + (ne - twoDeps0) / (2.0 * self.C_b)
        return Vtg, Vbg


@dataclass
class LockInMetadata:
    """SR830 settings, recorded into the JSON sidecar for provenance.

    Nothing in the measurement code reads these — they exist purely for
    reviewer-grade traceability.  These fields apply to ALL SR830s in the
    measurement chain (we assume the operator runs them with consistent
    settings).  If different SR830s use different settings, write a free-
    form note in the run_name / device fields documenting which.

    Per-channel constraints NOT captured here (they're implicit in the
    channel table's `kind` column):
      * kind = R     channels    →   Display 1 must be R   on that SR830
      * kind = Phase channels    →   Display 2 must be θ   on that SR830
    The operator is responsible for matching SR830 front panels to GUI
    channel kinds before pressing START.  See the file-top CHECKLIST.
    """
    frequency_hz:        float
    time_constant_s:     float
    filter_slope_db_oct: int        # 6 / 12 / 18 / 24
    reserve:             str        # 'Low Noise' | 'Normal' | 'High Reserve'
    sync_filter:         bool
    reference_phase_deg: float
    input_source:        str        # 'A' | 'A-B' | 'I'
    input_coupling:      str        # 'AC' | 'DC'
    input_ground:        str        # 'Float' | 'Ground'


@dataclass
class PointInfo:
    """One measurement point, MeasurementThread → GUI payload."""
    slow_target: float        # display-coordinate slow target
    fast_target: float        # display-coordinate fast target
    Vtg:         float        # actual physical Vtg written (NaN if skipped)
    Vbg:         float        # actual physical Vbg written (NaN if skipped)
    values:      list         # length NUM_AI; physical means
    errors:      list         # length NUM_AI; per-point std (1σ)
    skipped:     bool
    j_idx:       int          # fast index in this row
    i_idx:       int          # slow index (row index)
    direction:   str          # 'fwd' or 'bwd'


# =====================================================================
# Sweep strategies
# =====================================================================
class VtgVbgStrategy:
    """Vtg-Vbg rectangular scan.

    fast_is_top selects whether the fast (inner-loop) axis is Vtg or Vbg.
    Voltage limits are checked per-point: out-of-limit targets are skipped
    (no DAQ write, NaN in CSV).  This eliminates the v1.0 bug where the
    DAQ silently clipped to ±10 V while the CSV recorded the un-clipped
    target.
    """
    mode_id = SWEEP_MODE_VTGVBG

    def __init__(self, limits: VoltageLimits, fast_is_top: bool,
                 fast_min, fast_max, num_fast,
                 slow_min, slow_max, num_slow):
        self.limits = limits
        self.fast_is_top = bool(fast_is_top)
        self.fast_min = float(fast_min)
        self.fast_max = float(fast_max)
        self.num_fast = int(num_fast)
        self.slow_min = float(slow_min)
        self.slow_max = float(slow_max)
        self.num_slow = int(num_slow)

    @property
    def fast_label(self):
        return 'Vtg' if self.fast_is_top else 'Vbg'

    @property
    def slow_label(self):
        return 'Vbg' if self.fast_is_top else 'Vtg'

    @property
    def fast_unit(self): return 'V'
    @property
    def slow_unit(self): return 'V'
    @property
    def fast_axis_label(self): return f'{self.fast_label} (V)'
    @property
    def slow_axis_label(self): return f'{self.slow_label} (V)'

    def fast_array(self):
        return np.linspace(self.fast_min, self.fast_max, self.num_fast)

    def slow_array(self):
        return np.linspace(self.slow_min, self.slow_max, self.num_slow)

    def target_to_gates(self, slow_target, fast_target):
        """Returns (Vtg, Vbg, skipped). Skipped if outside voltage limits."""
        if self.fast_is_top:
            Vtg, Vbg = fast_target, slow_target
        else:
            Vtg, Vbg = slow_target, fast_target
        if not self.limits.check(Vtg, Vbg):
            return (None, None, True)
        return (Vtg, Vbg, False)

    def precheck_skip_fraction(self, n_samples=21):
        slow_grid = np.linspace(self.slow_min, self.slow_max, n_samples)
        fast_grid = np.linspace(self.fast_min, self.fast_max, n_samples)
        skipped = 0
        total = 0
        for s in slow_grid:
            for f in fast_grid:
                _, _, sk = self.target_to_gates(s, f)
                if sk:
                    skipped += 1
                total += 1
        return skipped / max(total, 1)


class NDStrategy:
    """n-D rectangular scan.  fast_axis selects which of (n, D) is
    the inner loop.  The outer loop is the other one.

    Display units: n in 1e12 cm^-2, D in V/nm.  Internally everything
    runs in SI (m^-2 and V/m).  Per-point reverse map to (Vtg, Vbg) is
    done via GeometryConfig.nD_to_vgates(); points outside the voltage
    limits are skipped.
    """
    mode_id = SWEEP_MODE_ND

    def __init__(self, geometry: GeometryConfig, limits: VoltageLimits,
                 fast_axis: str,
                 fast_min, fast_max, num_fast,
                 slow_min, slow_max, num_slow):
        self.geom = geometry
        self.limits = limits
        if fast_axis not in ('n', 'D'):
            raise ValueError(f"fast_axis must be 'n' or 'D', got {fast_axis!r}")
        self.fast_axis = fast_axis
        self.slow_axis = 'D' if fast_axis == 'n' else 'n'
        self.fast_min = float(fast_min)        # display units
        self.fast_max = float(fast_max)
        self.num_fast = int(num_fast)
        self.slow_min = float(slow_min)
        self.slow_max = float(slow_max)
        self.num_slow = int(num_slow)

    @staticmethod
    def _disp_to_si(value, axis):
        if axis == 'n':
            return value * N_DISPLAY_TO_M2
        else:
            return value * D_DISPLAY_TO_VM

    @staticmethod
    def _axis_unit_label(axis):
        return '×10¹² cm⁻²' if axis == 'n' else 'V/nm'

    @property
    def fast_label(self): return self.fast_axis
    @property
    def slow_label(self): return self.slow_axis
    @property
    def fast_unit(self): return self._axis_unit_label(self.fast_axis)
    @property
    def slow_unit(self): return self._axis_unit_label(self.slow_axis)
    @property
    def fast_axis_label(self):
        return f'{self.fast_axis} ({self._axis_unit_label(self.fast_axis)})'
    @property
    def slow_axis_label(self):
        return f'{self.slow_axis} ({self._axis_unit_label(self.slow_axis)})'

    def fast_array(self):
        return np.linspace(self.fast_min, self.fast_max, self.num_fast)

    def slow_array(self):
        return np.linspace(self.slow_min, self.slow_max, self.num_slow)

    def target_to_gates(self, slow_target_disp, fast_target_disp):
        if self.fast_axis == 'n':
            n_si = self._disp_to_si(fast_target_disp, 'n')
            D_si = self._disp_to_si(slow_target_disp, 'D')
        else:
            D_si = self._disp_to_si(fast_target_disp, 'D')
            n_si = self._disp_to_si(slow_target_disp, 'n')
        Vtg, Vbg = self.geom.nD_to_vgates(n_si, D_si)
        if not self.limits.check(Vtg, Vbg):
            return (None, None, True)
        return (Vtg, Vbg, False)

    def precheck_skip_fraction(self, n_samples=21):
        slow_grid = np.linspace(self.slow_min, self.slow_max, n_samples)
        fast_grid = np.linspace(self.fast_min, self.fast_max, n_samples)
        skipped = 0
        total = 0
        for s in slow_grid:
            for f in fast_grid:
                _, _, sk = self.target_to_gates(s, f)
                if sk:
                    skipped += 1
                total += 1
        return skipped / max(total, 1)


# =====================================================================
# Small UI helpers
# =====================================================================
class StayOpenMenu(QMenu):
    """Menu that does not auto-close on checkable item triggers."""
    def mouseReleaseEvent(self, e):
        action = self.actionAt(e.pos())
        if action is not None and action.isCheckable() and action.isEnabled():
            action.trigger()
            return
        super().mouseReleaseEvent(e)


class NoWheelComboBox(QComboBox):
    """Disable scroll-wheel value change (prevents accidental edits)."""
    def wheelEvent(self, e):
        e.ignore()


# DAQ-related imports gated on DEMO_MODE — set in main() before any class
# is instantiated.  Module-level fallback values so the file imports cleanly.
nidaqmx = None
TerminalConfiguration = None
AcquisitionType = None


# =====================================================================
# 1. Hardware abstraction
# =====================================================================
class DAQHardware:
    """NI DAQ wrapper.

    AO tasks: one per channel (each task has one AO channel).
    AI task : one task with all AI channels, configured for FINITE
              N-sample acquisition.  For each measurement point we
              start the task, read N samples per channel, then stop.

    n_avg can be changed at run time via set_n_avg(); the AI task is
    reconfigured (cfg_samp_clk_timing) accordingly.
    """

    def __init__(self,
                 ao_chans=('Dev1/ao0', 'Dev1/ao1'),
                 ai_chans=('Dev1/ai0', 'Dev1/ai1', 'Dev1/ai2', 'Dev1/ai3',
                           'Dev1/ai4', 'Dev1/ai5', 'Dev1/ai6', 'Dev1/ai7'),
                 ai_terminal=DAQ_AI_TERMINAL,
                 ai_sample_rate=DAQ_AI_RATE_HZ,
                 n_avg=DAQ_DEFAULT_NAVG):
        self.ao_chans = list(ao_chans)
        self.ai_chans = list(ai_chans)
        self.ai_terminal_name = ai_terminal
        self.ai_sample_rate = float(ai_sample_rate)
        self.n_avg = max(1, int(n_avg))
        self._ao_tasks = {}
        self._ai_task = None
        self._last_ao = {ch: 0.0 for ch in self.ao_chans}
        self.connected = False

    # ---- lifecycle ----
    def open(self):
        if DEMO_MODE:
            self.connected = True
            return
        # AO tasks
        for ch in self.ao_chans:
            t = nidaqmx.Task()
            t.ao_channels.add_ao_voltage_chan(ch, min_val=-10.0, max_val=10.0)
            self._ao_tasks[ch] = t

        # AI task
        term_map = {
            'DIFF': TerminalConfiguration.DIFF,
            'RSE':  TerminalConfiguration.RSE,
            'NRSE': TerminalConfiguration.NRSE,
        }
        term_const = term_map.get(self.ai_terminal_name, TerminalConfiguration.DIFF)
        self._ai_task = nidaqmx.Task()
        for ch in self.ai_chans:
            self._ai_task.ai_channels.add_ai_voltage_chan(
                ch, terminal_config=term_const,
                min_val=-10.0, max_val=10.0)
        self._configure_ai_timing()
        self.connected = True

    def _configure_ai_timing(self):
        """Apply n_avg to the AI task's sample-clock timing.  No-op in demo.

        v2.1 fix: ALWAYS configure FINITE timing with samps_per_chan = n_avg
        (clamped to ≥ 1).  v2.0 had a divergent 'else' branch for n_avg ≤ 1
        that left stale timing config in place — which then made read_ai()'s
        n_avg == 1 fast path return the wrong number of samples.
        """
        if DEMO_MODE or self._ai_task is None:
            return
        # nidaqmx requires the task to be unreserved / stopped before
        # cfg_samp_clk_timing can be changed.
        try:
            self._ai_task.stop()
        except Exception:
            pass
        self._ai_task.timing.cfg_samp_clk_timing(
            rate=self.ai_sample_rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=max(int(self.n_avg), 1))

    def set_n_avg(self, n_avg):
        n = max(1, int(n_avg))
        if n == self.n_avg:
            return
        self.n_avg = n
        self._configure_ai_timing()

    def close(self):
        if DEMO_MODE:
            self._ao_tasks.clear()
            self._ai_task = None
            self.connected = False
            return
        for t in self._ao_tasks.values():
            try: t.close()
            except Exception: pass
        self._ao_tasks.clear()
        if self._ai_task is not None:
            try: self._ai_task.close()
            except Exception: pass
            self._ai_task = None
        self.connected = False

    # ---- AO ----
    def write_ao(self, channel, voltage):
        """Write AO channel.  No-op if target equals last-written value."""
        voltage = float(np.clip(voltage, -10.0, 10.0))
        if abs(voltage - self._last_ao.get(channel, 0.0)) < 1e-9:
            return
        if not DEMO_MODE:
            self._ao_tasks[channel].write(voltage)
        self._last_ao[channel] = voltage

    def ramp_ao(self, channel, v_target, rate_v_per_s=2.0, step=0.02):
        """Slow safe ramp at rate_v_per_s, in `step`-sized increments."""
        v0 = self._last_ao.get(channel, 0.0)
        v_target = float(np.clip(v_target, -10.0, 10.0))
        if abs(v_target - v0) < 1e-6:
            return
        n = max(2, int(np.ceil(abs(v_target - v0) / step)))
        dwell = abs(v_target - v0) / max(rate_v_per_s, 1e-3) / n
        for v in np.linspace(v0, v_target, n):
            self.write_ao(channel, v)
            time.sleep(dwell)

    # ---- AI ----
    def read_ai(self):
        """Read n_avg samples on every AI channel, return (means, stds).

        v2.1: unified code path — always FINITE acquisition with
        samps_per_chan = max(n_avg, 1).  No more divergent fast path.

        Returns:
            means: list of NUM_AI floats — sample mean of raw DAQ voltage
            stds : list of NUM_AI floats — per-point sample std (1σ);
                   0.0 if n_avg == 1
        """
        if DEMO_MODE:
            return self._demo_read_ai()

        n = max(int(self.n_avg), 1)
        try:
            self._ai_task.start()
        except Exception:
            # Some nidaqmx versions auto-start; ignore.
            pass
        try:
            data = self._ai_task.read(
                number_of_samples_per_channel=n,
                timeout=10.0)
        finally:
            try:
                self._ai_task.stop()
            except Exception:
                pass

        arr = np.asarray(data, dtype=float)
        # Single-channel edge case: nidaqmx returns a flat list
        if arr.ndim == 1:
            arr = arr[None, :]
        means = arr.mean(axis=1).tolist()
        if arr.shape[1] > 1:
            stds = arr.std(axis=1, ddof=1).tolist()
        else:
            stds = [0.0] * arr.shape[0]
        return means, stds

    def _demo_read_ai(self):
        """Synthesise plausible data from current AO state for DEMO_MODE.

        For 'kind=R' channels the raw voltage is shaped so that, after
        the inverse SR830-CH-output + SR560-gain + R_series chain in
        MeasurementThread._convert (sens=10mV, gain=100, I_ac≈1nA),
        the reported R lands in a few-kΩ to ~30 kΩ range with a
        2D pattern that depends on both AOs.

        Even AI indices → R-like signals (positive 0..few V on the BNC).
        Odd AI indices → phase-like small signals near 0 V.
        """
        ao0 = self._last_ao.get(self.ao_chans[0], 0.0)
        ao1 = self._last_ao.get(self.ao_chans[1], 0.0)
        n = max(int(self.n_avg), 1)
        means = []
        stds = []
        for ai in range(NUM_AI):
            if ai % 2 == 0:
                # R-like: SR830 R-output is 0..10 V mapping to 0..sens.
                # Generate a structured pattern with a peak (Dirac point)
                # that moves with AO sum — looks like a CNP feature.
                cnp_dist = (ao0 + ao1) * 0.5
                base = 1.5 + 1.5 * np.exp(-0.5 * cnp_dist**2 / 1.0**2)
                base += 0.4 * np.cos(0.7 * ao0) * np.sin(0.5 * ao1)
                base = max(base, 0.05)
            else:
                # phase-like: small ±0.2 V
                base = 0.05 * np.sin(ao0 + 0.3 * ai) + 0.02 * np.cos(1.7 * ao1)
            samples = base + 0.005 * np.random.randn(n)
            means.append(float(samples.mean()))
            if n > 1:
                stds.append(float(samples.std(ddof=1)))
            else:
                stds.append(0.0)
        return means, stds


# =====================================================================
# 2. Measurement worker thread
# =====================================================================
class MeasurementThread(QThread):
    point_ready    = pyqtSignal(object)         # PointInfo
    row_finished   = pyqtSignal(int)
    log_msg        = pyqtSignal(str, str)
    finished_ok    = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, params, hardware: DAQHardware):
        super().__init__()
        self.p = params
        self.hw = hardware
        self.is_running = True

        self.v_osc        = params['v_osc']
        self.r_series     = params['r_series']
        self.i_ac         = self.v_osc / self.r_series
        self.phase_factor = 18.0
        self.channels         = list(params['channels'])
        self.enabled_channels = [ch for ch in self.channels if ch.enabled]
        self.bidirectional    = params['bidirectional']

        self.strategy   = params['strategy']
        self.geometry   = params.get('geometry', None)
        self.limits     = params['limits']
        self.lockin_meta = params.get('lockin_meta', None)

        self.ao_top = params['ao_top']
        self.ao_bot = params['ao_bot']

        self.t_dwell            = params['t_dwell']
        self.t_settle_slow      = params['t_settle_slow']
        self.t_settle_after_fwd = params['t_settle_after_fwd']
        self.t_settle_after_bwd = params['t_settle_after_bwd']
        self.t_retrace_step     = params['t_retrace_step']
        self.n_avg              = params['n_avg']

        self.skip_count = 0
        self._started_at_iso = None
        self._completed_normally = False
        self._overload_warned = set()       # ai indices already warned

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def _interruptible_sleep(self, total_s):
        """Sleep up to total_s in 100 ms chunks; return False if aborted."""
        if total_s <= 0:
            return self.is_running
        n_chunks = max(1, int(round(total_s / 0.1)))
        chunk = total_s / n_chunks
        for _ in range(n_chunks):
            if not self.is_running:
                return False
            time.sleep(chunk)
        return self.is_running

    def _check_overload(self, raw_means):
        """One-shot warning per AI when |raw V| > OVERLOAD_RAW_V."""
        for ai, v in enumerate(raw_means):
            if abs(v) > OVERLOAD_RAW_V and ai not in self._overload_warned:
                ch_name = next(
                    (c.name for c in self.channels if c.ai_index == ai),
                    f"AI{ai}")
                self.log_msg.emit(
                    f"OVERLOAD: AI{ai} ({ch_name}) raw |V| = {abs(v):.2f} V "
                    f"> {OVERLOAD_RAW_V:.1f} V — likely SR830 sensitivity "
                    f"too low. Will not warn again for this channel.",
                    "warning")
                self._overload_warned.add(ai)

    def _convert(self, raw_means, raw_stds):
        """Convert raw DAQ voltages to physical units.

        For each enabled-or-disabled channel, returns the physical mean
        and the physical 1σ uncertainty propagated through the linear
        scaling.  Length NUM_AI (full 8 channels) so on_point can index
        by ai_index.  Disabled channels go through the same conversion
        for consistency; they're filtered out at CSV-write time.

        SR830 OUTPUT scaling reference (from SR830 manual §3):

          * X / Y / R outputs:  full scale ±10 V  ⇆  ±sensitivity
            (R is non-negative so it uses 0..10 V → 0..sensitivity)
          * θ output:           full scale ±10 V  ⇆  ±180°  (18 deg/V)

        IMPORTANT: kind='R' channels assume the SR830 front panel has
        DISPLAY 1 = R (not the default DISPLAY 1 = X).  Then CH1 OUTPUT
        outputs R in the 0..sens mapping.  If you forget to change the
        display setting, CH1 will output X instead, which can be
        negative AND only equals R when the phase is exactly 0° — your
        R values will be wrong in a way that does not raise any error.
        Verify the SR830 front panel before every cooldown.

        kind='Phase' channels assume DISPLAY 2 = θ.

        The R conversion chain:
            V_DAQ                                       (read at AI pin)
              ↓ × (sens / 10 V)                         (undo SR830 R-output scaling)
            R_at_lockin_input  [V]
              ↓ ÷ gain                                  (undo SR560 voltage gain)
            V_at_device  [V]
              ↓ ÷ I_ac                                  (Ohm's law)
            R_device  [Ω]
        """
        means_phys = []
        errs_phys  = []
        for ch in self.channels:
            v = raw_means[ch.ai_index]
            s = raw_stds[ch.ai_index]
            if ch.kind == 'R':
                scale = (ch.sens / 10.0) / ch.gain / self.i_ac
                means_phys.append(v * scale)
                errs_phys.append(abs(s * scale))
            elif ch.kind == 'Phase':
                # SR830 θ output: ±10 V → ±180 deg
                means_phys.append(v * self.phase_factor)
                errs_phys.append(abs(s * self.phase_factor))
            else:    # Voltage
                means_phys.append(v)
                errs_phys.append(abs(s))
        return means_phys, errs_phys

    # ------------------------------------------------------------
    # CSV / metadata
    # ------------------------------------------------------------
    def _write_csv_header(self, writer):
        """v2.3: write ONLY the column-name row.  All provenance metadata
        goes into the .txt and .json sidecars instead.

        This means downstream analysis can do simply:
            df = pd.read_csv(path)
        with no `comment='#'` or `skiprows=` arguments.
        """
        s = self.strategy
        cols = [f'{s.slow_label}_target',
                f'{s.fast_label}_target',
                'Vtg_actual', 'Vbg_actual',
                'skipped', 'direction']
        for ch in self.enabled_channels:
            cols.append(f'{ch.csv_col_name}_mean')
            cols.append(f'{ch.csv_col_name}_std')
        writer.writerow(cols)

    def _write_txt_sidecar(self):
        """Write a human-readable plain-text metadata file alongside the CSV.

        File path: <csv stem>.txt
        Sections are laid out for eyeball-scanning when you come back to
        a dataset months later.  Lines are wrapped to ~78 chars where
        practical.
        """
        p = self.p
        s = self.strategy
        txt_path = os.path.splitext(p['save_path'])[0] + '.txt'

        def _row(label, value, indent=2):
            return f"{' ' * indent}{label:<28} {value}"

        L = []
        L.append("=" * 78)
        L.append(f"  {APP_NAME}  ·  {APP_VERSION}")
        L.append(f"  Metadata sidecar for: {os.path.basename(p['save_path'])}")
        L.append("=" * 78)
        L.append("")

        # ---- run identification ----
        L.append("[ Run identification ]")
        L.append(_row("started_at",  self._started_at_iso))
        L.append(_row("finished_at", time.strftime('%Y-%m-%dT%H:%M:%S')))
        L.append(_row("completed_normally", self._completed_normally))
        L.append(_row("sample",   p.get('sample', '')))
        L.append(_row("device",   p.get('device', '')))
        L.append(_row("operator", p.get('operator', '')))
        L.append(_row("run_name", p.get('run_name', '')))
        L.append("")

        # ---- scan parameters ----
        L.append("[ Scan parameters ]")
        L.append(_row("sweep_mode",    s.mode_id))
        L.append(_row("bidirectional", self.bidirectional))
        L.append(_row("fast_axis",
                      f"{s.fast_label}  ∈ [{s.fast_min}, {s.fast_max}]  "
                      f"({s.num_fast} pts {s.fast_unit})"))
        L.append(_row("slow_axis",
                      f"{s.slow_label}  ∈ [{s.slow_min}, {s.slow_max}]  "
                      f"({s.num_slow} pts {s.slow_unit})"))
        # Effective steps
        if s.num_fast > 1:
            fstep = (s.fast_max - s.fast_min) / (s.num_fast - 1)
            L.append(_row("fast_step (effective)", f"{fstep:g} {s.fast_unit}"))
        if s.num_slow > 1:
            sstep = (s.slow_max - s.slow_min) / (s.num_slow - 1)
            L.append(_row("slow_step (effective)", f"{sstep:g} {s.slow_unit}"))
        L.append(_row("t_dwell_s",            self.t_dwell))
        L.append(_row("t_settle_slow_s",      self.t_settle_slow))
        L.append(_row("t_settle_after_fwd_s", self.t_settle_after_fwd))
        L.append(_row("t_settle_after_bwd_s", self.t_settle_after_bwd))
        L.append(_row("t_retrace_step_s",     self.t_retrace_step))
        L.append(_row("skip_count",           self.skip_count))
        L.append("")

        # ---- DAQ ----
        L.append("[ NI DAQ ]")
        L.append(_row("ai_terminal",       self.hw.ai_terminal_name))
        L.append(_row("ai_sample_rate_hz", self.hw.ai_sample_rate))
        L.append(_row("n_avg_per_point",   self.n_avg))
        avg_window = max(int(self.n_avg), 1) / self.hw.ai_sample_rate
        L.append(_row("avg_window_s", f"{avg_window:.4g}  "
                                       f"({avg_window*1e3:.2f} ms)"))
        L.append(_row("ao_top",  self.ao_top))
        L.append(_row("ao_bot",  self.ao_bot))
        L.append("")

        # ---- voltage limits ----
        L.append("[ Voltage limits  (hBN safety) ]")
        L.append(_row("|Vtg|_max_v", self.limits.Vtg_max))
        L.append(_row("|Vbg|_max_v", self.limits.Vbg_max))
        L.append("")

        # ---- lock-in source ----
        L.append("[ Lock-in current source ]")
        L.append(_row("V_osc_v",      f"{self.v_osc:.6g}"))
        L.append(_row("R_series_ohm", f"{self.r_series:.6e}"))
        L.append(_row("I_ac_a",       f"{self.i_ac:.6e}"))
        L.append(_row("phase_factor_deg_per_v", self.phase_factor))
        L.append("")

        # ---- SR830 settings ----
        if self.lockin_meta is not None:
            lm = self.lockin_meta
            L.append("[ SR830 settings (operator-recorded) ]")
            L.append(_row("frequency_hz",         lm.frequency_hz))
            L.append(_row("time_constant_s",      lm.time_constant_s))
            L.append(_row("filter_slope_db_oct",  lm.filter_slope_db_oct))
            L.append(_row("reserve",              lm.reserve))
            L.append(_row("sync_filter",          lm.sync_filter))
            L.append(_row("reference_phase_deg",  lm.reference_phase_deg))
            L.append(_row("input_source",         lm.input_source))
            L.append(_row("input_coupling",       lm.input_coupling))
            L.append(_row("input_ground",         lm.input_ground))
            L.append("")
            L.append("  OPERATOR ASSERTION:")
            L.append("    Every  kind=R    channel: its SR830 Display 1 = R")
            L.append("    Every  kind=Phase channel: its SR830 Display 2 = θ")
            L.append("")

        # ---- geometry (n-D mode only) ----
        if self.geometry is not None:
            g = self.geometry
            L.append("[ hBN geometry  (n-D mode) ]")
            L.append(_row("d_t_nm",  g.d_t_nm))
            L.append(_row("d_b_nm",  g.d_b_nm))
            L.append(_row("eps_hBN", g.eps_hBN))
            L.append(_row("Vtg0_v",  g.Vtg0))
            L.append(_row("Vbg0_v",  g.Vbg0))
            L.append(_row("C_t_F_per_m2", f"{g.C_t:.6e}"))
            L.append(_row("C_b_F_per_m2", f"{g.C_b:.6e}"))
            L.append(_row("n_display_unit", "1e12 cm^-2"))
            L.append(_row("D_display_unit", "V/nm"))
            L.append(_row("D_sign_convention",
                          "(C_t·dVtg - C_b·dVbg) / (2·eps0)  [+Vtg → +D]"))
            L.append("")

        # ---- channels ----
        L.append("[ Enabled channels ]")
        L.append(f"  {'AI':<4} {'name':<18} {'kind':<10} "
                 f"{'sens':<10} {'gain':<10} {'csv_unit':<8}")
        L.append(f"  {'-' * 70}")
        for ch in self.enabled_channels:
            sens = f"{ch.sens:g}" if ch.kind == 'R' else '—'
            gain = f"{ch.gain:g}" if ch.kind == 'R' else '—'
            L.append(f"  {ch.ai_index:<4} {ch.name:<18} {ch.kind:<10} "
                     f"{sens:<10} {gain:<10} {ch.csv_unit:<8}")
        L.append("")
        L.append("[ All AI channels (incl. disabled, for traceability) ]")
        for ch in self.channels:
            state = "ENABLED " if ch.enabled else "disabled"
            sens = f"{ch.sens:g}" if ch.kind == 'R' else '—'
            gain = f"{ch.gain:g}" if ch.kind == 'R' else '—'
            L.append(f"  AI{ch.ai_index} [{state}]  name={ch.name}  "
                     f"kind={ch.kind}  sens={sens}  gain={gain}")
        L.append("")

        # ---- CSV column reference ----
        L.append("[ CSV column reference ]")
        L.append(f"  data file: {os.path.basename(p['save_path'])}")
        L.append(f"  load with: pd.read_csv('{os.path.basename(p['save_path'])}')")
        L.append("")
        L.append("  Columns:")
        L.append(f"    {s.slow_label}_target   "
                 f"slow-axis target value (display units: {s.slow_unit})")
        L.append(f"    {s.fast_label}_target   "
                 f"fast-axis target value (display units: {s.fast_unit})")
        L.append("    Vtg_actual         physical top-gate voltage written to AO (V)")
        L.append("    Vbg_actual         physical bottom-gate voltage written to AO (V)")
        L.append("    skipped            1 if target was outside voltage limits, else 0")
        L.append("    direction          'fwd' or 'bwd'")
        for ch in self.enabled_channels:
            unit_pretty = ch.csv_unit
            L.append(f"    {ch.csv_col_name}_mean   "
                     f"AI{ch.ai_index} {ch.kind}, mean of n_avg samples "
                     f"({unit_pretty})")
            L.append(f"    {ch.csv_col_name}_std    "
                     f"AI{ch.ai_index} {ch.kind}, std of n_avg samples (1σ, "
                     f"{unit_pretty})")
        L.append("")
        L.append("=" * 78)

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(L) + '\n')
        return txt_path

    def _write_metadata_sidecar(self):
        p = self.p
        s = self.strategy
        sidecar = os.path.splitext(p['save_path'])[0] + '.json'
        meta = {
            'schema_version': 4,
            'software':    f'{APP_NAME} {APP_VERSION}',
            'started_at':  self._started_at_iso,
            'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'completed_normally': self._completed_normally,
            'sample_metadata': {
                'sample':   p.get('sample', ''),
                'device':   p.get('device', ''),
                'operator': p.get('operator', ''),
                'run_name': p.get('run_name', ''),
            },
            'sweep_mode': s.mode_id,
            'gates_physical': {
                'ao_top': self.ao_top,
                'ao_bot': self.ao_bot,
            },
            'voltage_limits': {
                'Vtg_max_v': self.limits.Vtg_max,
                'Vbg_max_v': self.limits.Vbg_max,
            },
            'sweep': {
                'fast_label': s.fast_label,
                'fast_unit':  s.fast_unit,
                'fast_min':   s.fast_min,
                'fast_max':   s.fast_max,
                'fast_N':     s.num_fast,
                'slow_label': s.slow_label,
                'slow_unit':  s.slow_unit,
                'slow_min':   s.slow_min,
                'slow_max':   s.slow_max,
                'slow_N':     s.num_slow,
                't_dwell_s':            self.t_dwell,
                't_settle_slow_s':      self.t_settle_slow,
                't_settle_after_fwd_s': self.t_settle_after_fwd,
                't_settle_after_bwd_s': self.t_settle_after_bwd,
                't_retrace_step_s':     self.t_retrace_step,
                'bidirectional':        self.bidirectional,
            },
            'daq': {
                'ai_terminal':         self.hw.ai_terminal_name,
                'ai_sample_rate_hz':   self.hw.ai_sample_rate,
                'n_avg_per_point':     self.n_avg,
                # v2.1: always n_avg/rate, even when n_avg == 1 (was 0.0).
                'effective_avg_window_s':
                    max(int(self.n_avg), 1) / self.hw.ai_sample_rate,
            },
            'geometry': (None if self.geometry is None else {
                'd_t_nm':   self.geometry.d_t_nm,
                'd_b_nm':   self.geometry.d_b_nm,
                'eps_hBN':  self.geometry.eps_hBN,
                'Vtg0_v':   self.geometry.Vtg0,
                'Vbg0_v':   self.geometry.Vbg0,
                'C_t_F_per_m2': self.geometry.C_t,
                'C_b_F_per_m2': self.geometry.C_b,
                'n_display_unit': '1e12 cm^-2',
                'D_display_unit': 'V/nm',
                'D_sign_convention':
                    '(C_t·dVtg - C_b·dVbg) / (2·eps0)  [+Vtg → +D]',
            }),
            'lockin_source': {
                'v_osc_v':      self.v_osc,
                'r_series_ohm': self.r_series,
                'i_ac_a':       self.i_ac,
                'phase_factor_deg_per_v': self.phase_factor,
            },
            'lockin_metadata': (None if self.lockin_meta is None else {
                'frequency_hz':         self.lockin_meta.frequency_hz,
                'time_constant_s':      self.lockin_meta.time_constant_s,
                'filter_slope_db_oct':  self.lockin_meta.filter_slope_db_oct,
                'reserve':              self.lockin_meta.reserve,
                'sync_filter':          self.lockin_meta.sync_filter,
                'reference_phase_deg':  self.lockin_meta.reference_phase_deg,
                'input_source':         self.lockin_meta.input_source,
                'input_coupling':       self.lockin_meta.input_coupling,
                'input_ground':         self.lockin_meta.input_ground,
                'operator_assertion':
                    'every kind=R channel: SR830 Display 1 = R; '
                    'every kind=Phase channel: SR830 Display 2 = theta',
            }),
            'channels': [
                {
                    'ai_index':      ch.ai_index,
                    'name':          ch.name,
                    'enabled':       bool(ch.enabled),
                    'kind':          ch.kind,
                    'unit':          ch.unit,
                    'sensitivity_v': ch.sens if ch.kind == 'R' else None,
                    'gain':          ch.gain if ch.kind == 'R' else None,
                } for ch in self.channels
            ],
            'skip_count': self.skip_count,
            'data_file': os.path.basename(p['save_path']),
        }
        with open(sidecar, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return sidecar

    # ------------------------------------------------------------
    # Per-point emit helpers
    # ------------------------------------------------------------
    def _emit_skip(self, writer, slow_t, fast_t, j_idx, i_idx, direction):
        nan_values = [float('nan')] * NUM_AI
        nan_errors = [float('nan')] * NUM_AI
        data_cols = []
        for _ch in self.enabled_channels:
            data_cols.append(float('nan'))
            data_cols.append(float('nan'))
        row = [slow_t, fast_t, float('nan'), float('nan'), 1, direction] + data_cols
        writer.writerow(row)
        info = PointInfo(
            slow_target=slow_t, fast_target=fast_t,
            Vtg=float('nan'), Vbg=float('nan'),
            values=nan_values, errors=nan_errors, skipped=True,
            j_idx=j_idx, i_idx=i_idx, direction=direction,
        )
        self.point_ready.emit(info)
        self.skip_count += 1

    def _emit_valid(self, writer, slow_t, fast_t, Vtg, Vbg,
                    values, errors, j_idx, i_idx, direction):
        data_cols = []
        for ch in self.enabled_channels:
            ai = ch.ai_index
            data_cols.append(values[ai])
            data_cols.append(errors[ai])
        row = [slow_t, fast_t, Vtg, Vbg, 0, direction] + data_cols
        writer.writerow(row)
        info = PointInfo(
            slow_target=slow_t, fast_target=fast_t,
            Vtg=Vtg, Vbg=Vbg,
            values=values, errors=errors, skipped=False,
            j_idx=j_idx, i_idx=i_idx, direction=direction,
        )
        self.point_ready.emit(info)

    # ------------------------------------------------------------
    # Inner loop — one direction along fast axis
    # ------------------------------------------------------------
    def _sweep_one_direction(self, writer, f, slow_target,
                             i_idx, direction, fast_array):
        """Inner loop, shared by Vtg-Vbg and n-D modes via strategy.

        Skip-safety: every entry into this function starts with
        last_was_skip=True, so the FIRST valid point uses ramp_ao
        (not write_ao).  Inside the loop, after a skipped point we
        also reset last_was_skip=True so the next valid point ramps.
        """
        ao_top = self.ao_top
        ao_bot = self.ao_bot
        last_was_skip = True
        for j, fast_target in enumerate(fast_array):
            if not self.is_running:
                f.flush()
                return
            Vtg, Vbg, skipped = self.strategy.target_to_gates(
                slow_target, fast_target)
            if skipped:
                self._emit_skip(writer, slow_target, fast_target,
                                j, i_idx, direction)
                last_was_skip = True
                continue
            if last_was_skip:
                # Safe ramp.  ramp_ao no-ops if already at target.
                self.hw.ramp_ao(ao_top, Vtg)
                self.hw.ramp_ao(ao_bot, Vbg)
                if not self._interruptible_sleep(self.t_settle_slow):
                    f.flush()
                    return
            else:
                self.hw.write_ao(ao_top, Vtg)
                self.hw.write_ao(ao_bot, Vbg)
            if not self._interruptible_sleep(self.t_dwell):
                f.flush()
                return
            raw_means, raw_stds = self.hw.read_ai()
            self._check_overload(raw_means)
            values, errors = self._convert(raw_means, raw_stds)
            self._emit_valid(writer, slow_target, fast_target,
                             Vtg, Vbg, values, errors, j, i_idx, direction)
            last_was_skip = False
        f.flush()

    def _safe_retrace(self, fast_array_bwd, slow_target):
        """Single-direction mode: ride the gates back to fast_min without
        recording.  Same safe-ramp logic, no AI read, configurable step
        time (was hardcoded 10 ms in v1.0)."""
        ao_top = self.ao_top
        ao_bot = self.ao_bot
        last_was_skip = True
        for fast_target in fast_array_bwd:
            if not self.is_running:
                return
            Vtg, Vbg, skipped = self.strategy.target_to_gates(
                slow_target, fast_target)
            if skipped:
                last_was_skip = True
                continue
            if last_was_skip:
                self.hw.ramp_ao(ao_top, Vtg)
                self.hw.ramp_ao(ao_bot, Vbg)
            else:
                self.hw.write_ao(ao_top, Vtg)
                self.hw.write_ao(ao_bot, Vbg)
            time.sleep(self.t_retrace_step)
            last_was_skip = False

    # ------------------------------------------------------------
    # run()
    # ------------------------------------------------------------
    def run(self):
        p = self.p
        s = self.strategy
        self._started_at_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

        # Reconfigure DAQ AI timing for this scan's n_avg.  If this fails
        # the AI task is in an undefined state and we should NOT proceed —
        # raise so the main except block catches it and reports cleanly.
        try:
            self.hw.set_n_avg(self.n_avg)
        except Exception as e:
            tb = traceback.format_exc()
            self.log_msg.emit(
                f"FATAL: DAQ AI timing reconfiguration failed: {e}.  "
                f"Aborting scan before any data is written.", "error")
            self.error_occurred.emit(tb)
            return

        slow_array     = s.slow_array()
        fast_array_fwd = s.fast_array()
        fast_array_bwd = fast_array_fwd[::-1]

        ao_top = self.ao_top
        ao_bot = self.ao_bot

        try:
            self.log_msg.emit(
                f"Scan started: sample={p.get('sample','')!r} "
                f"device={p.get('device','')!r} run={p.get('run_name','')!r}",
                "info")
            self.log_msg.emit(
                f"Mode = {s.mode_id}.  I_ac = {self.i_ac:.3e} A.  "
                f"Bidirectional = {self.bidirectional}", "info")
            self.log_msg.emit(
                f"Fast axis: {s.fast_label} ∈ [{s.fast_min}, {s.fast_max}] "
                f"({s.num_fast} pts {s.fast_unit})", "info")
            self.log_msg.emit(
                f"Slow axis: {s.slow_label} ∈ [{s.slow_min}, {s.slow_max}] "
                f"({s.num_slow} pts {s.slow_unit})", "info")
            self.log_msg.emit(
                f"Physical: top gate ← {ao_top}, bottom gate ← {ao_bot}.  "
                f"Limits |Vtg|≤{self.limits.Vtg_max}V, "
                f"|Vbg|≤{self.limits.Vbg_max}V", "info")
            self.log_msg.emit(
                f"DAQ: {self.hw.ai_terminal_name} terminal, "
                f"{self.n_avg} samples/point @ {self.hw.ai_sample_rate:.0f} Hz "
                f"(window = {self.n_avg/self.hw.ai_sample_rate*1000:.1f} ms)",
                "info")
            if s.mode_id == SWEEP_MODE_ND and self.geometry is not None:
                g = self.geometry
                self.log_msg.emit(
                    f"hBN geometry: d_t={g.d_t_nm}nm, d_b={g.d_b_nm}nm, "
                    f"ε={g.eps_hBN}, C_t={g.C_t:.3e}, C_b={g.C_b:.3e} F/m²",
                    "info")
            if self.lockin_meta is not None:
                lm = self.lockin_meta
                self.log_msg.emit(
                    f"SR830: f={lm.frequency_hz} Hz, τ={lm.time_constant_s} s, "
                    f"slope={lm.filter_slope_db_oct} dB/oct, "
                    f"reserve={lm.reserve}", "info")
            self.log_msg.emit(f"Saving to {p['save_path']}", "info")

            self.log_msg.emit("Ramping gates to safe start (0 V)...", "info")
            self.hw.ramp_ao(ao_top, 0.0)
            self.hw.ramp_ao(ao_bot, 0.0)
            time.sleep(0.3)

            with open(p['save_path'], mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                self._write_csv_header(writer)
                f.flush()

                for i, slow_target in enumerate(slow_array):
                    if not self.is_running:
                        break

                    self._sweep_one_direction(writer, f, slow_target,
                                              i, 'fwd', fast_array_fwd)
                    self.row_finished.emit(i)
                    if not self.is_running:
                        break
                    if not self._interruptible_sleep(self.t_settle_after_fwd):
                        break

                    if self.bidirectional:
                        self._sweep_one_direction(writer, f, slow_target,
                                                  i, 'bwd', fast_array_bwd)
                        self.row_finished.emit(i)
                    else:
                        self._safe_retrace(fast_array_bwd, slow_target)

                    if not self.is_running:
                        break
                    if not self._interruptible_sleep(self.t_settle_after_bwd):
                        break

            self._completed_normally = self.is_running
            if self._completed_normally:
                self.log_msg.emit(
                    f"Scan finished normally.  Skipped points: {self.skip_count}",
                    "success")
            else:
                self.log_msg.emit(
                    f"Scan aborted by user.  Skipped points: {self.skip_count}",
                    "warning")
            self.finished_ok.emit()

        except Exception:
            tb = traceback.format_exc()
            self.log_msg.emit(f"ERROR: {tb.splitlines()[-1]}", "error")
            self.error_occurred.emit(tb)
        finally:
            try:
                self.log_msg.emit("Ramping gates back to 0 V...", "info")
                self.hw.ramp_ao(ao_top, 0.0)
                self.hw.ramp_ao(ao_bot, 0.0)
                self.log_msg.emit("Gates safely at 0 V.", "success")
            except Exception as e:
                self.log_msg.emit(f"WARN: ramp-down failed: {e}", "warning")
            try:
                json_sidecar = self._write_metadata_sidecar()
                self.log_msg.emit(
                    f"JSON sidecar written: {os.path.basename(json_sidecar)}",
                    "info")
            except Exception as e:
                self.log_msg.emit(f"WARN: JSON sidecar write failed: {e}",
                                  "warning")
            try:
                txt_sidecar = self._write_txt_sidecar()
                self.log_msg.emit(
                    f"TXT sidecar written:  {os.path.basename(txt_sidecar)}",
                    "info")
            except Exception as e:
                self.log_msg.emit(f"WARN: TXT sidecar write failed: {e}",
                                  "warning")

    def stop(self):
        self.is_running = False


# =====================================================================
# 3. Main window
# =====================================================================
class QuantumTransportGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME}  ·  {APP_VERSION}"
                            + ("   [DEMO MODE]" if DEMO_MODE else ""))
        self.resize(1800, 1100)

        # pyqtgraph globals
        pg.setConfigOption('background', CT_BASE)
        pg.setConfigOption('foreground', CT_TEXT)
        pg.setConfigOption('imageAxisOrder', 'row-major')
        pg.setConfigOptions(antialias=True)

        # Runtime state
        self.thread = None
        self.curr_row_idx = -1
        self.curr_buf = self._make_empty_1d_buffers()
        self.map_data = None              # {ai: {direction: ndarray}}
        self.locked_channels = None
        self.locked_bidirectional = False
        self.locked_strategy = None
        self.locked_rect = None
        self._scan_started_at = None
        self._total_points = 0
        self._points_done = 0

        # Hardware
        self.hw = DAQHardware()
        self._daq_init_error = None
        try:
            self.hw.open()
        except Exception as e:
            self._daq_init_error = str(e)

        # UI assembly
        self._init_ui()
        self._init_menubar()
        self._init_statusbar()

        self._load_settings()

        # Clock
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)
        self._tick_clock()

        # Initial derived UI
        self._refresh_filename()
        self._update_iac_label()
        self._on_sweep_mode_changed()
        self._refresh_sweep_labels()
        self._refresh_axis_labels()
        self._refresh_effective_step_labels()
        self._rebuild_channel_selectors()
        for i in range(NUM_AI):
            self._refresh_kind_widgets(i)

        # Initial log
        self.log_event("━" * 60, "info")
        self.log_event(f"{APP_NAME}  ·  {APP_VERSION}", "success")
        self.log_event("━" * 60, "info")
        if DEMO_MODE:
            self.log_event("DEMO mode active — no hardware will be touched.",
                           "warning")
        if self._daq_init_error:
            self.log_event(f"DAQ init failed: {self._daq_init_error}", "error")
        elif self.hw.connected:
            self.log_event(
                f"DAQ connection ready ({self.hw.ai_terminal_name} terminal, "
                f"{self.hw.n_avg} samples @ {self.hw.ai_sample_rate:.0f} Hz).",
                "success")
            # ---- DAQ self-test: read all 8 AI once, log raw voltages ----
            # Lets the operator verify wiring + SR830 amplitude before
            # starting a long scan.  Catches:
            #   * floating AI pins (DIFF mode + single-ended wiring)
            #   * SR830 overload (channel railing at ±10 V)
            #   * SR830 not powered on / sin-out off
            try:
                means, _ = self.hw.read_ai()
                lines = []
                for ai in range(NUM_AI):
                    v = means[ai]
                    if abs(v) > OVERLOAD_RAW_V:
                        flag = " ⚠ OVERLOAD"
                    elif abs(v) < 1e-4:
                        flag = "  (≈0 V)"
                    else:
                        flag = ""
                    lines.append(f"AI{ai}={v:+8.4f}V{flag}")
                self.log_event(
                    "Self-test read: " + " | ".join(lines[:4]), "info")
                self.log_event(
                    "Self-test read: " + " | ".join(lines[4:]), "info")
            except Exception as e:
                self.log_event(
                    f"DAQ self-test read FAILED: {e}.  Check wiring before "
                    f"starting a scan.", "error")

    # ============================================================
    # state helpers
    # ============================================================
    @staticmethod
    def _make_empty_1d_buffers():
        return {
            d: {
                'fast': [],
                'vals': [[] for _ in range(NUM_AI)],
            } for d in DIRECTIONS
        }

    # ============================================================
    # Top-level UI assembly
    # ============================================================
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_top_bar())

        v_split = QSplitter(Qt.Vertical)
        v_split.setChildrenCollapsible(False)

        h_split = QSplitter(Qt.Horizontal)
        h_split.setChildrenCollapsible(False)
        h_split.addWidget(self._build_left_panel())
        h_split.addWidget(self._build_right_panel())
        h_split.setStretchFactor(0, 0)
        h_split.setStretchFactor(1, 1)
        h_split.setSizes([520, 1280])
        self.h_split = h_split

        v_split.addWidget(h_split)
        v_split.addWidget(self._build_bottom_panel())
        v_split.setStretchFactor(0, 1)
        v_split.setStretchFactor(1, 0)
        v_split.setSizes([800, 280])
        self.v_split = v_split

        root.addWidget(v_split, stretch=1)

        self._setup_plot_linkage_and_units()

    # ---- top metadata bar ----
    def _build_top_bar(self):
        bar = QFrame()
        bar.setObjectName("topBar")
        h = QHBoxLayout(bar)
        h.setContentsMargins(14, 8, 14, 8)
        h.setSpacing(18)

        def add_field(label_text, default, tooltip, width=160):
            lab = QLabel(label_text)
            lab.setObjectName("metaLabel")
            le = QLineEdit(default)
            le.setMinimumWidth(width)
            le.setToolTip(tooltip)
            h.addWidget(lab)
            h.addWidget(le)
            return le

        self.le_sample   = add_field("Sample",   "",
            "Sample / wafer / batch identifier.\nGoes into filename + JSON metadata.")
        self.le_device   = add_field("Device",   "",
            "Specific device / chip / Hall bar on this sample.")
        self.le_operator = add_field("Operator", "",
            "Initials or name of the person running this measurement.", width=80)
        self.le_run_name = add_field("Run",      "",
            "Short name for this particular scan (e.g. cooldown1_map1).")

        for le in (self.le_sample, self.le_device, self.le_operator, self.le_run_name):
            le.textChanged.connect(self._refresh_filename)

        h.addStretch()
        return bar

    # ---- left control panel ----
    def _build_left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(10)

        v.addWidget(self._build_mode_group())
        v.addWidget(self._build_gates_group())
        v.addWidget(self._build_voltage_limits_group())
        v.addWidget(self._build_geometry_group())
        v.addWidget(self._build_sweep_group())
        v.addWidget(self._build_lockin_group())
        v.addWidget(self._build_channels_group())
        v.addWidget(self._build_output_group())
        v.addStretch()
        v.addLayout(self._build_button_row())

        scroll.setWidget(inner)
        scroll.setMinimumWidth(500)
        return scroll

    # ---------------- mode group ----------------
    def _build_mode_group(self):
        gb = QGroupBox("Sweep mode")
        grid = QGridLayout()
        grid.setContentsMargins(8, 4, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        l = QLabel("Mode")
        l.setObjectName("sectionLabel")
        grid.addWidget(l, 0, 0)
        self.cb_sweep_mode = NoWheelComboBox()
        self.cb_sweep_mode.addItem("Vtg–Vbg  (gate voltages)",
                                   userData=SWEEP_MODE_VTGVBG)
        self.cb_sweep_mode.addItem("n–D  (carrier density × displacement field)",
                                   userData=SWEEP_MODE_ND)
        self.cb_sweep_mode.setToolTip(
            "Vtg–Vbg: classic dual-gate rectangle scan in gate voltage space.\n"
            "n–D: scan target points in (n, D) space; reverse-mapped to (Vtg, Vbg).\n"
            "Points outside hBN voltage limits are skipped (NaN in CSV, transparent in 2D).")
        self.cb_sweep_mode.currentIndexChanged.connect(self._on_sweep_mode_changed)
        grid.addWidget(self.cb_sweep_mode, 0, 1, 1, 2)

        l2 = QLabel("Fast axis")
        l2.setObjectName("sectionLabel")
        grid.addWidget(l2, 1, 0)
        self.cb_nd_fast_axis = NoWheelComboBox()
        self.cb_nd_fast_axis.addItem("n  (carrier density)", userData='n')
        self.cb_nd_fast_axis.addItem("D  (displacement field)", userData='D')
        self.cb_nd_fast_axis.setToolTip(
            "n–D mode only. Which axis is the inner-loop fast axis.\n"
            "For 'fix D, repeat n sweep' workflows pick n here.")
        self.cb_nd_fast_axis.currentIndexChanged.connect(self._refresh_sweep_labels)
        grid.addWidget(self.cb_nd_fast_axis, 1, 1, 1, 2)

        gb.setLayout(grid)
        return gb

    def _on_sweep_mode_changed(self, _idx=None):
        self._refresh_sweep_labels()
        if hasattr(self, 'geom_group'):
            mode = self.cb_sweep_mode.currentData()
            self.geom_group.setVisible(mode == SWEEP_MODE_ND)
        if hasattr(self, 'cb_nd_fast_axis'):
            mode = self.cb_sweep_mode.currentData()
            self.cb_nd_fast_axis.setEnabled(mode == SWEEP_MODE_ND)
        if hasattr(self, 'cb_fast_gate'):
            mode = self.cb_sweep_mode.currentData()
            self.cb_fast_gate.setEnabled(mode == SWEEP_MODE_VTGVBG)
        self._refresh_effective_step_labels()
        self._refresh_axis_labels()

    # ---------------- gates group (explicit top/bottom mapping) ----------------
    def _build_gates_group(self):
        gb = QGroupBox("Gates  (physical AO → top / bottom)")
        grid = QGridLayout()
        grid.setContentsMargins(8, 4, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        for col, h in enumerate(['Role', 'AO channel']):
            l = QLabel(h)
            l.setObjectName("sectionLabel")
            grid.addWidget(l, 0, col)

        ao_options = ['AO0', 'AO1']

        # Top gate
        grid.addWidget(QLabel("Top gate (Vtg)"), 1, 0)
        self.cb_top_ao = NoWheelComboBox()
        self.cb_top_ao.addItems(ao_options)
        self.cb_top_ao.setCurrentText('AO0')
        self.cb_top_ao.setToolTip(
            "Which physical DAQ AO drives the TOP gate (Vtg).\n"
            "Set this to match your wiring; the OTHER AO is automatically\n"
            "assigned to the bottom gate.")
        self.cb_top_ao.currentTextChanged.connect(self._on_top_ao_changed)
        grid.addWidget(self.cb_top_ao, 1, 1)

        # Bottom gate
        grid.addWidget(QLabel("Bottom gate (Vbg)"), 2, 0)
        self.cb_bot_ao = NoWheelComboBox()
        self.cb_bot_ao.addItems(ao_options)
        self.cb_bot_ao.setCurrentText('AO1')
        self.cb_bot_ao.setToolTip(
            "Auto-set to whichever AO is not used by the top gate.")
        self.cb_bot_ao.currentTextChanged.connect(self._on_bot_ao_changed)
        grid.addWidget(self.cb_bot_ao, 2, 1)

        # Vtg-Vbg only: which gate is the FAST (inner) axis?
        l3 = QLabel("Fast (inner)")
        l3.setObjectName("sectionLabel")
        grid.addWidget(l3, 3, 0)
        self.cb_fast_gate = NoWheelComboBox()
        self.cb_fast_gate.addItem("Vtg", userData='top')
        self.cb_fast_gate.addItem("Vbg", userData='bot')
        self.cb_fast_gate.setToolTip(
            "Vtg-Vbg mode only. Which gate is the inner-loop (fast) axis;\n"
            "the other becomes the outer (slow) axis.\n"
            "In n-D mode this is ignored — use the 'Fast axis' selector\n"
            "in the Sweep mode group.")
        self.cb_fast_gate.currentIndexChanged.connect(self._refresh_sweep_labels)
        grid.addWidget(self.cb_fast_gate, 3, 1)

        grid.setColumnStretch(1, 1)
        gb.setLayout(grid)
        return gb

    def _on_top_ao_changed(self, text):
        other = 'AO1' if text == 'AO0' else 'AO0'
        if self.cb_bot_ao.currentText() != other:
            self.cb_bot_ao.blockSignals(True)
            self.cb_bot_ao.setCurrentText(other)
            self.cb_bot_ao.blockSignals(False)

    def _on_bot_ao_changed(self, text):
        other = 'AO1' if text == 'AO0' else 'AO0'
        if self.cb_top_ao.currentText() != other:
            self.cb_top_ao.blockSignals(True)
            self.cb_top_ao.setCurrentText(other)
            self.cb_top_ao.blockSignals(False)

    # ---------------- voltage limits group (always visible) ----------------
    def _build_voltage_limits_group(self):
        """Voltage limits enforced in BOTH modes.  This was the v1.0
        Vtg-Vbg data-integrity bug — the DAQ silently clipped to ±10 V
        while the CSV recorded the un-clipped value."""
        gb = QGroupBox("Voltage limits  (hBN safety, both modes)")
        grid = QGridLayout()
        grid.setContentsMargins(8, 4, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.lim_inputs = {}
        rows = [
            ('Vtg_max', '|Vtg|_max (V)', '5.0',
             "Hard upper bound on |Vtg| for THIS scan, in BOTH modes.\n"
             "Target points whose physical Vtg exceeds this are SKIPPED\n"
             "(no DAQ write, NaN in CSV, transparent in 2D map).\n"
             "Set to your hBN breakdown voltage with a safety margin."),
            ('Vbg_max', '|Vbg|_max (V)', '5.0',
             "Hard upper bound on |Vbg|. Same logic."),
        ]
        for i, (key, label_text, default, tip) in enumerate(rows):
            l = QLabel(label_text)
            l.setToolTip(tip)
            grid.addWidget(l, i, 0)
            le = QLineEdit(default)
            le.setToolTip(tip)
            self.lim_inputs[key] = le
            grid.addWidget(le, i, 1)

        gb.setLayout(grid)
        return gb

    # ---------------- geometry group (n-D only) ----------------
    def _build_geometry_group(self):
        gb = QGroupBox("Geometry  (n–D mode)")
        self.geom_group = gb
        grid = QGridLayout()
        grid.setContentsMargins(8, 4, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.geom_inputs = {}
        rows = [
            ('d_t',  'd_t  (top hBN, nm)',    '30.0',
             "Top hBN dielectric thickness in nm. Determines C_t."),
            ('d_b',  'd_b  (bottom hBN, nm)', '30.0',
             "Bottom hBN dielectric thickness in nm. Determines C_b."),
            ('eps',  'ε_hBN (out-of-plane)',  '3.0',
             "hBN out-of-plane dielectric constant. Typical 3.0–3.9."),
            ('Vtg0', 'Vtg₀  (CNP, V)',        '0.0',
             "Charge neutrality point Vtg, found from a prior Vtg–Vbg scan."),
            ('Vbg0', 'Vbg₀  (CNP, V)',        '0.0',
             "Charge neutrality point Vbg."),
        ]
        for i, (key, label_text, default, tip) in enumerate(rows):
            l = QLabel(label_text)
            l.setToolTip(tip)
            grid.addWidget(l, i, 0)
            le = QLineEdit(default)
            le.setToolTip(tip)
            self.geom_inputs[key] = le
            grid.addWidget(le, i, 1)

        gb.setLayout(grid)
        return gb

    # ---------------- sweep group (now: min, max, N) ----------------
    def _build_sweep_group(self):
        gb = QGroupBox("Sweep")
        grid = QGridLayout()
        grid.setContentsMargins(8, 4, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.sweep_inputs          = {}
        self.sweep_label_widgets   = {}
        self.sweep_label_templates = {}
        self.sweep_step_labels     = {}   # 'fast' / 'slow' → QLabel hint

        # (key, default, tooltip, label_template)
        rows = [
            ('fast_min', '-5.0',
             "Fast axis lower bound.",
             '{fast} min ({funit})'),
            ('fast_max',  '5.0',
             "Fast axis upper bound.",
             '{fast} max ({funit})'),
            ('fast_N',  '801',
             "Number of points along the fast axis (inclusive of endpoints).\n"
             "Effective step = (max − min) / (N − 1).\n"
             "Tip: use this to make 'fix slow, repeat fast' workflows trivial.",
             '{fast} N (points)'),
            ('slow_min', '-5.0',
             "Slow axis lower bound.",
             '{slow} min ({sunit})'),
            ('slow_max',  '5.0',
             "Slow axis upper bound.",
             '{slow} max ({sunit})'),
            ('slow_N',   '101',
             "Number of points along the slow axis. For a fixed-slow scan,\n"
             "set min = max and N = 1.",
             '{slow} N (points)'),
        ]
        for i, (key, dv, tip, label_tpl) in enumerate(rows):
            lab = QLabel(label_tpl)
            lab.setToolTip(tip)
            self.sweep_label_widgets[key]   = lab
            self.sweep_label_templates[key] = label_tpl
            grid.addWidget(lab, i, 0)

            le = QLineEdit(dv)
            le.setToolTip(tip)
            le.textChanged.connect(self._refresh_effective_step_labels)
            self.sweep_inputs[key] = le
            grid.addWidget(le, i, 1)

        # Effective-step hint labels (next to N-points field)
        next_row = len(rows)

        l_fhint = QLabel("Effective fast step: —")
        l_fhint.setObjectName("hintLabel")
        grid.addWidget(l_fhint, 2, 2)
        self.sweep_step_labels['fast'] = l_fhint

        l_shint = QLabel("Effective slow step: —")
        l_shint.setObjectName("hintLabel")
        grid.addWidget(l_shint, 5, 2)
        self.sweep_step_labels['slow'] = l_shint

        # Timing fields
        timing_rows = [
            ('t_dwell', '1.5',
             "Dwell time at each fast point AFTER stepping the AO and BEFORE\n"
             "starting the n_avg-sample DAQ acquisition.\n"
             "MUST be ≥ 5 × SR830 time constant for full settling.\n"
             "Default 1.5 s = 5 × default TC (0.3 s) — adjust together.",
             't_dwell (s)'),
            ('t_settle_slow', '1.0',
             "Wait after each slow step / safe ramp before starting the fast sweep.",
             't_settle_slow (s)'),
            ('t_settle_after_fwd', '2.0',
             "Wait after the forward fast sweep finishes.",
             't_settle_after_fwd (s)'),
            ('t_settle_after_bwd', '2.0',
             "Wait after the reverse fast sweep / retrace finishes,\n"
             "before stepping the slow gate.",
             't_settle_after_bwd (s)'),
            ('t_retrace_step', '0.01',
             "Per-point dwell during retrace (single-direction mode).\n"
             "Default 10 ms.  Note: Windows time.sleep resolution is ~16 ms.",
             't_retrace_step (s)'),
            ('n_avg', str(DAQ_DEFAULT_NAVG),
             f"Number of DAQ samples averaged per measurement point.\n"
             f"AI sample rate is fixed at {DAQ_AI_RATE_HZ:.0f} Hz.\n"
             f"Effective averaging window = n_avg / {DAQ_AI_RATE_HZ:.0f} Hz.\n"
             f"Both the mean AND the per-point std are saved to CSV.",
             'n_avg (samples)'),
        ]
        base_row = next_row + 1
        for i, (key, dv, tip, label_tpl) in enumerate(timing_rows):
            r = base_row + i
            lab = QLabel(label_tpl)
            lab.setToolTip(tip)
            self.sweep_label_widgets[key]   = lab
            self.sweep_label_templates[key] = label_tpl
            grid.addWidget(lab, r, 0)
            le = QLineEdit(dv)
            le.setToolTip(tip)
            self.sweep_inputs[key] = le
            grid.addWidget(le, r, 1)

        self.cb_bidirectional = QCheckBox("Bidirectional  (record fwd & bwd)")
        self.cb_bidirectional.setToolTip(
            "Off: forward sweep recorded, reverse is per-point retrace at\n"
            "    t_retrace_step (NOT recorded).\n"
            "On:  both directions recorded at full t_dwell.\n"
            "Use ON for ferroelectric / hysteresis measurements, AND for\n"
            "fwd/bwd averaging which cancels the slow lock-in lag artefact.")
        self.cb_bidirectional.stateChanged.connect(self._rebuild_channel_selectors)
        grid.addWidget(self.cb_bidirectional, base_row + len(timing_rows), 0, 1, 3)

        gb.setLayout(grid)
        return gb

    def _refresh_sweep_labels(self):
        """Update sweep-group labels based on mode and which gate is fast."""
        if not hasattr(self, 'sweep_label_widgets'):
            return
        mode = self.cb_sweep_mode.currentData()
        if mode == SWEEP_MODE_ND:
            fast = self.cb_nd_fast_axis.currentData()  # 'n' or 'D'
            slow = 'D' if fast == 'n' else 'n'
            funit = '×10¹² cm⁻²' if fast == 'n' else 'V/nm'
            sunit = '×10¹² cm⁻²' if slow == 'n' else 'V/nm'
        else:
            # Vtg-Vbg
            fast_role = self.cb_fast_gate.currentData()  # 'top' or 'bot'
            fast = 'Vtg' if fast_role == 'top' else 'Vbg'
            slow = 'Vbg' if fast_role == 'top' else 'Vtg'
            funit = 'V'
            sunit = 'V'

        for key, tpl in self.sweep_label_templates.items():
            lab = self.sweep_label_widgets[key]
            txt = (tpl
                   .replace('{fast}', fast)
                   .replace('{slow}', slow)
                   .replace('{funit}', funit)
                   .replace('{sunit}', sunit))
            lab.setText(txt)

    def _refresh_effective_step_labels(self, _=None):
        """Show '(max-min)/(N-1)' under the N-points fields."""
        if not hasattr(self, 'sweep_step_labels'):
            return
        for axis in ('fast', 'slow'):
            try:
                vmin = float(self.sweep_inputs[f'{axis}_min'].text())
                vmax = float(self.sweep_inputs[f'{axis}_max'].text())
                N    = int(float(self.sweep_inputs[f'{axis}_N'].text()))
                if N <= 1:
                    if N == 1 and abs(vmax - vmin) < 1e-12:
                        self.sweep_step_labels[axis].setText(
                            f"Single-point ({axis}): N=1, value={vmin:g}")
                    else:
                        self.sweep_step_labels[axis].setText(
                            f"Effective {axis} step: N must be ≥ 1")
                else:
                    step = (vmax - vmin) / (N - 1)
                    self.sweep_step_labels[axis].setText(
                        f"Effective {axis} step: {step:.6g}")
            except (ValueError, ZeroDivisionError):
                self.sweep_step_labels[axis].setText(f"Effective {axis} step: —")

    # ---------------- lock-in source group ----------------
    def _build_lockin_group(self):
        gb = QGroupBox("Lock-in  /  Source")
        grid = QGridLayout()
        grid.setContentsMargins(8, 4, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self.hw_inputs = {}

        rows = [
            ('v_osc', 'V_osc (V_rms)', '0.5',
             f"SR830 sine-out amplitude.\nMust be in [{SR830_VOSC_MIN}, {SR830_VOSC_MAX}] V."),
            ('r_series', 'R_series (Ω)', '500e6',
             f"Series resistor for V→I conversion.\n"
             f"Refused if < {R_SERIES_MIN_OHM:.0e} Ω (would push too much current)."),
        ]
        for i, (key, lbl, dv, tip) in enumerate(rows):
            l = QLabel(lbl)
            l.setToolTip(tip)
            grid.addWidget(l, i, 0)
            le = QLineEdit(dv)
            le.setToolTip(tip)
            self.hw_inputs[key] = le
            grid.addWidget(le, i, 1)

        l_iac = QLabel('I_ac (computed)')
        l_iac.setToolTip("Computed AC excitation current. Read-only.")
        grid.addWidget(l_iac, 2, 0)
        self.lbl_iac = QLabel('— A')
        self.lbl_iac.setObjectName("valueLabel")
        grid.addWidget(self.lbl_iac, 2, 1)

        self.hw_inputs['v_osc'].textChanged.connect(self._update_iac_label)
        self.hw_inputs['r_series'].textChanged.connect(self._update_iac_label)

        # SR830 settings (provenance only — operator transcribes from front panel)
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {CT_SURFACE1};")
        grid.addWidget(sep, 3, 0, 1, 2)

        l_meta = QLabel("SR830 settings (for JSON metadata)")
        l_meta.setObjectName("sectionLabel")
        grid.addWidget(l_meta, 4, 0, 1, 2)

        self.sr830_inputs = {}
        # Plain text fields
        sr_rows = [
            ('freq',    'Frequency (Hz)',    '13.333',
             "SR830 reference frequency.\n"
             "Must match the front-panel display.\n"
             "Default 13.333 Hz is well clear of 50/60 Hz mains."),
            ('tc',      'Time constant (s)', '0.3',
             "SR830 time constant.\n"
             "t_dwell must be ≥ 5×TC for full settling.\n"
             "Default 0.3 s + t_dwell=1.5 s = 5τ → no warning fires."),
            ('slope',   'Slope (dB/oct)',    '24',
             "SR830 low-pass slope. Typical 12 / 18 / 24 dB/oct."),
            ('phase',   'Reference phase (deg)', '0.0',
             "SR830 reference phase offset (deg)."),
        ]
        for i, (key, lbl, dv, tip) in enumerate(sr_rows):
            r = 5 + i
            l = QLabel(lbl)
            l.setObjectName("sectionLabel")
            l.setToolTip(tip)
            grid.addWidget(l, r, 0)
            le = QLineEdit(dv)
            le.setToolTip(tip)
            self.sr830_inputs[key] = le
            grid.addWidget(le, r, 1)

        # Combo fields with enumerated choices (provenance only)
        sr_combo_rows = [
            ('reserve',  'Reserve',     SR830_RESERVE_OPTIONS,  'Low Noise',
             "SR830 dynamic reserve.\n"
             "  Low Noise   — lowest input-referred noise; small overload margin\n"
             "  Normal      — balanced\n"
             "  High Reserve — large overload margin; higher noise floor\n"
             "Default 'Low Noise' for clean signal chains "
             "(SR560 differential preamp upstream)."),
            ('sync',     'Sync filter', ('on', 'off'),          'on',
             "SR830 sync filter.\n"
             "Recommended ON for f < 200 Hz.\n"
             "ON adds a notch at the reference frequency to remove 2f leakage."),
            ('source',   'Input source', SR830_SOURCE_OPTIONS,  'A',
             "SR830 SIGNAL INPUT source.\n"
             "Standard wiring: SR560 OUTPUT → SR830 A → choose 'A' here.\n"
             "Use 'A-B' only if you wire SR830 A and B both to the device "
             "and skip the SR560.\n"
             "Use 'I' only for direct current input (rare for our chain)."),
            ('coupling', 'Input coupling', SR830_COUPLING_OPTIONS, 'AC',
             "SR830 SIGNAL INPUT coupling.\n"
             "AC: high-pass at 0.16 Hz; use for AC-modulated lock-in detection.\n"
             "DC: bypass the input AC coupler.\n"
             "Default 'AC' for standard f=13.33 Hz lock-in detection."),
            ('ground',   'Input ground', SR830_GROUND_OPTIONS,   'Float',
             "SR830 SIGNAL INPUT ground reference.\n"
             "Float : input shield decoupled from chassis (10 kΩ to chassis).\n"
             "Ground: input shield tied to chassis ground.\n"
             "Default 'Float' to break ground loops with the cryostat."),
            #
            # NOTE: display1 / display2 fields removed in v2.2.
            # In multi-SR830 setups a single global field cannot speak for
            # all instruments.  The constraint
            #     kind=R     ⇒ Display 1 must be R   on that channel's SR830
            #     kind=Phase ⇒ Display 2 must be θ   on that channel's SR830
            # is a per-channel responsibility of the operator.  See the
            # file-top CHECKLIST and the kind-column tooltip.
        ]
        base = 5 + len(sr_rows)
        for i, (key, lbl, choices, default, tip) in enumerate(sr_combo_rows):
            r = base + i
            l = QLabel(lbl)
            l.setObjectName("sectionLabel")
            l.setToolTip(tip)
            grid.addWidget(l, r, 0)
            cb = NoWheelComboBox()
            cb.addItems(list(choices))
            cb.setCurrentText(default)
            cb.setToolTip(tip)
            self.sr830_inputs[key] = cb
            grid.addWidget(cb, r, 1)

        gb.setLayout(grid)
        return gb

    # ---------------- channels group ----------------
    def _build_channels_group(self):
        gb = QGroupBox("Channels  (per-AI configuration)")
        grid = QGridLayout()
        grid.setContentsMargins(8, 4, 8, 8)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)

        for col, h in enumerate(['', 'Name', 'On', 'Kind', 'Sens (V)', 'Gain']):
            l = QLabel(h)
            l.setObjectName("sectionLabel")
            grid.addWidget(l, 0, col)

        kind_tip = (
            "Per-channel measurement type:\n"
            "  R       → resistance.  Reverse-maps SR830 CH1 OUTPUT (in 0..10 V\n"
            "            mapping to 0..sens) and SR560 voltage gain to give\n"
            "            R = V_device / I_ac in Ω.\n"
            "            ★ ASSUMES the SR830 driving this AI has Display 1 = R.\n"
            "            If Display 1 is left at the default 'X', CH1 OUTPUT\n"
            "            is X (in-phase), not R, and your resistance values\n"
            "            will be silently wrong.  Verify the SR830 front panel\n"
            "            of EVERY kind=R channel before pressing START.\n"
            "  Phase   → degrees.  V_AI × 18 deg/V from SR830 CH2 OUTPUT.\n"
            "            ★ ASSUMES the SR830 driving this AI has Display 2 = θ.\n"
            "  Voltage → raw DAQ voltage, no conversion."
        )
        sens_tip = ("SR830 sensitivity (V).\n"
                    "Used only when Kind = R. Disabled and shown as '—' otherwise.\n"
                    "Must match the front-panel dial of the SR830 driving this AI.")
        gain_tip = ("SR560 voltage preamp gain (dimensionless).\n"
                    "Used only when Kind = R.\n"
                    "Must match the front-panel dial of the SR560 driving this AI.")
        name_tip = ("Channel display name. Goes into 1D legend, 2D slot dropdowns,\n"
                    "CSV column headers, and JSON metadata. Pick something unique.")
        on_tip   = ("Enable this channel. ONLY enabled channels are written to CSV\n"
                    "(disabled channels do not produce dead columns in the output).")

        # Default channel layout: 4 active channels (AI0..AI3) for the
        # standard Rxx + Rxy two-SR830 setup.  AI4..AI7 are present for
        # expansion but disabled by default.
        defaults = [
            # name,         kind,   sens,    gain,   enabled
            ('Rxx_R',       'R',    '10e-3', '100',  True),
            ('Rxx_phase',   'Phase','—',     '—',    True),
            ('Rxy_R',       'R',    '1e-3',  '1000', True),
            ('Rxy_phase',   'Phase','—',     '—',    True),
            ('AI4',         'Voltage','—',   '—',    False),
            ('AI5',         'Voltage','—',   '—',    False),
            ('AI6',         'Voltage','—',   '—',    False),
            ('AI7',         'Voltage','—',   '—',    False),
        ]

        self.ch_name_inputs   = []
        self.ch_enable_checks = []
        self.ch_kind_combos   = []
        self.ch_sens_inputs   = []
        self.ch_gain_inputs   = []

        for ai, (nm, kd, sd, gd, en) in enumerate(defaults):
            row = ai + 1
            l = QLabel(f'AI{ai}')
            l.setObjectName("sectionLabel")
            grid.addWidget(l, row, 0)

            le_n = QLineEdit(nm)
            le_n.setToolTip(name_tip)
            le_n.editingFinished.connect(self._rebuild_channel_selectors)
            self.ch_name_inputs.append(le_n)
            grid.addWidget(le_n, row, 1)

            cb_e = QCheckBox()
            cb_e.setChecked(en)
            cb_e.setToolTip(on_tip)
            self.ch_enable_checks.append(cb_e)
            grid.addWidget(cb_e, row, 2)

            cb_k = NoWheelComboBox()
            cb_k.addItems(KIND_OPTIONS)
            cb_k.setCurrentText(kd)
            cb_k.setToolTip(kind_tip)
            cb_k.currentTextChanged.connect(
                lambda _t, i=ai: (self._refresh_kind_widgets(i),
                                  self._rebuild_channel_selectors()))
            self.ch_kind_combos.append(cb_k)
            grid.addWidget(cb_k, row, 3)

            le_s = QLineEdit(sd)
            le_s.setToolTip(sens_tip)
            self.ch_sens_inputs.append(le_s)
            grid.addWidget(le_s, row, 4)

            le_g = QLineEdit(gd)
            le_g.setToolTip(gain_tip)
            self.ch_gain_inputs.append(le_g)
            grid.addWidget(le_g, row, 5)

        grid.setColumnStretch(1, 1)
        gb.setLayout(grid)
        return gb

    def _refresh_kind_widgets(self, ai_idx):
        kind = self.ch_kind_combos[ai_idx].currentText()
        is_r = (kind == 'R')
        s_le = self.ch_sens_inputs[ai_idx]
        g_le = self.ch_gain_inputs[ai_idx]
        if is_r:
            s_le.setEnabled(True); g_le.setEnabled(True)
            if s_le.text().strip() == '—': s_le.setText('10e-3')
            if g_le.text().strip() == '—': g_le.setText('100')
        else:
            s_le.setEnabled(False); g_le.setEnabled(False)
            s_le.setText('—'); g_le.setText('—')

    # ---------------- output group ----------------
    def _build_output_group(self):
        gb = QGroupBox("Output")
        v = QVBoxLayout()
        v.setContentsMargins(8, 4, 8, 8)
        v.setSpacing(6)

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Folder:"))
        self.le_folder = QLineEdit(os.getcwd())
        self.le_folder.setToolTip("Where CSV + JSON sidecar will be written.")
        self.le_folder.textChanged.connect(self._refresh_filename)
        h1.addWidget(self.le_folder, stretch=1)
        btn = QPushButton("Browse")
        btn.clicked.connect(self._pick_folder)
        h1.addWidget(btn)
        v.addLayout(h1)

        h2 = QHBoxLayout()
        l = QLabel("File:")
        l.setObjectName("metaLabel")
        h2.addWidget(l)
        self.lbl_filename = QLabel("—")
        self.lbl_filename.setObjectName("valueLabel")
        self.lbl_filename.setWordWrap(True)
        h2.addWidget(self.lbl_filename, stretch=1)
        v.addLayout(h2)

        gb.setLayout(v)
        return gb

    def _build_button_row(self):
        h = QHBoxLayout()
        self.btn_start = QPushButton("▶  START   (F5)")
        self.btn_start.setObjectName("primary")
        self.btn_start.setToolTip("Start the scan with current settings (F5).")
        self.btn_start.clicked.connect(self.start_measurement)

        self.btn_stop = QPushButton("■  ABORT   (Esc)")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.setToolTip("Abort the current scan and ramp gates back to 0 V (Esc).")
        self.btn_stop.clicked.connect(self.stop_measurement)
        self.btn_stop.setEnabled(False)

        h.addWidget(self.btn_start)
        h.addWidget(self.btn_stop)
        return h

    # ============================================================
    # Right plotting panel
    # ============================================================
    def _build_right_panel(self):
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        v_split = QSplitter(Qt.Vertical)
        v_split.setChildrenCollapsible(False)
        v_split.addWidget(self._build_1d_area())
        v_split.addWidget(self._build_2d_area())
        v_split.setSizes([400, 700])
        v.addWidget(v_split)
        return wrap

    def _build_1d_area(self):
        wrap = QWidget()
        grid = QGridLayout(wrap)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        self.line_slot_buttons   = []
        self.line_slot_menus     = []
        self.line_slot_actions   = []
        self.line_slot_plots     = []
        self.line_slot_lines     = []
        self.line_slot_legends   = []

        for slot_idx in range(NUM_LINE_SLOTS):
            r = (slot_idx // 2) * 2
            c = slot_idx % 2

            btn = QToolButton()
            btn.setText(f"Slot {slot_idx + 1}:  (no channels)  ▾")
            btn.setPopupMode(QToolButton.InstantPopup)
            btn.setToolTip(
                "Click to pick which channels to overlay on this plot.\n"
                "Multi-select stays open until you click outside or press Esc.")
            menu = StayOpenMenu(btn)
            btn.setMenu(menu)
            self.line_slot_buttons.append(btn)
            self.line_slot_menus.append(menu)
            self.line_slot_actions.append({})
            grid.addWidget(btn, r, c)

            pw = pg.PlotWidget()
            pw.showGrid(x=True, y=True, alpha=0.3)
            pw.setLabel('bottom', 'fast')
            legend = pw.addLegend(offset=(10, 10))
            self.line_slot_plots.append(pw)
            self.line_slot_lines.append({})
            self.line_slot_legends.append(legend)
            grid.addWidget(pw, r + 1, c)

        grid.setRowStretch(1, 1)
        grid.setRowStretch(3, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        return wrap

    def _build_2d_area(self):
        wrap = QWidget()
        grid = QGridLayout(wrap)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        # Plasma-style colormap
        pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        col = np.array([
            [13,  8,  135, 255],
            [84,  2,  163, 255],
            [185, 50, 137, 255],
            [249, 142, 9,  255],
            [240, 249, 33, 255],
        ], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, col)

        self.map_slot_combos = []
        self.map_slot_plots  = []
        self.map_slot_images = []

        for slot_idx in range(NUM_MAP_SLOTS):
            r = (slot_idx // 2) * 2
            c = slot_idx % 2

            combo = NoWheelComboBox()
            combo.setToolTip(
                "Pick which (channel × direction) to display in this 2D map.\n"
                "All combinations are kept in memory so you can switch the\n"
                "view at any time during or after the scan.")
            combo.currentIndexChanged.connect(
                lambda _i, s=slot_idx: self._on_map_slot_changed(s))
            self.map_slot_combos.append(combo)
            grid.addWidget(combo, r, c)

            pw = pg.PlotWidget()
            pw.setLabel('bottom', 'fast')
            pw.setLabel('left',   'slow')
            pw.setTitle("(off)")
            img = pg.ImageItem()
            img.setColorMap(cmap)
            pw.addItem(img)
            self.map_slot_plots.append(pw)
            self.map_slot_images.append(img)
            grid.addWidget(pw, r + 1, c)

        grid.setRowStretch(1, 1)
        grid.setRowStretch(3, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        return wrap

    def _setup_plot_linkage_and_units(self):
        if not (hasattr(self, 'line_slot_plots') and hasattr(self, 'map_slot_plots')):
            return

        # 1) kill auto-SI-prefix on every relevant axis
        for pw in self.line_slot_plots + self.map_slot_plots:
            for axis_name in ('left', 'bottom'):
                try:
                    axis = pw.getAxis(axis_name)
                    if axis is not None:
                        axis.enableAutoSIPrefix(False)
                except Exception:
                    pass

        # 2) link x-axes
        x_master = self.line_slot_plots[0]
        for pw in (list(self.line_slot_plots[1:]) + list(self.map_slot_plots)):
            try:
                pw.setXLink(x_master)
            except Exception:
                pass

        # 3) link map y-axes
        y_master = self.map_slot_plots[0]
        for pw in self.map_slot_plots[1:]:
            try:
                pw.setYLink(y_master)
            except Exception:
                pass

    # ============================================================
    # Bottom panel: progress + event log
    # ============================================================
    def _build_bottom_panel(self):
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(12, 8, 12, 8)
        v.setSpacing(6)

        prog_box = QGroupBox("Progress")
        pg_layout = QGridLayout()
        pg_layout.setContentsMargins(8, 4, 8, 8)
        pg_layout.setHorizontalSpacing(20)
        pg_layout.setVerticalSpacing(4)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        pg_layout.addWidget(self.progress_bar, 0, 0, 1, 4)

        self.lbl_step    = QLabel("Step  —");      self.lbl_step.setObjectName("bigStat")
        self.lbl_elapsed = QLabel("Elapsed  —");   self.lbl_elapsed.setObjectName("bigStat")
        self.lbl_eta     = QLabel("ETA  —");       self.lbl_eta.setObjectName("bigStat")
        self.lbl_now     = QLabel("—");            self.lbl_now.setObjectName("bigStat")
        pg_layout.addWidget(self.lbl_step,    1, 0)
        pg_layout.addWidget(self.lbl_elapsed, 1, 1)
        pg_layout.addWidget(self.lbl_eta,     1, 2)
        pg_layout.addWidget(self.lbl_now,     1, 3)

        prog_box.setLayout(pg_layout)
        v.addWidget(prog_box)

        log_box = QGroupBox("Event log")
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(8, 4, 8, 8)
        self.event_log = QPlainTextEdit()
        self.event_log.setObjectName("eventLog")
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumBlockCount(2000)
        log_layout.addWidget(self.event_log)
        log_box.setLayout(log_layout)
        v.addWidget(log_box, stretch=1)

        return wrap

    # ============================================================
    # Menubar / statusbar
    # ============================================================
    def _init_menubar(self):
        mb = self.menuBar()

        m_file = mb.addMenu("&File")
        a_save = QAction("&Save settings now", self)
        a_save.setShortcut(QKeySequence("Ctrl+S"))
        a_save.triggered.connect(self._save_settings)
        m_file.addAction(a_save)
        a_browse = QAction("Choose output &folder...", self)
        a_browse.triggered.connect(self._pick_folder)
        m_file.addAction(a_browse)
        m_file.addSeparator()
        a_quit = QAction("&Quit", self)
        a_quit.setShortcut(QKeySequence("Ctrl+Q"))
        a_quit.triggered.connect(self.close)
        m_file.addAction(a_quit)

        m_run = mb.addMenu("&Run")
        self.act_start = QAction("▶  Start scan", self)
        self.act_start.setShortcut(QKeySequence("F5"))
        self.act_start.triggered.connect(self.start_measurement)
        m_run.addAction(self.act_start)
        self.act_stop = QAction("■  Abort scan", self)
        self.act_stop.setShortcut(QKeySequence("Esc"))
        self.act_stop.triggered.connect(self.stop_measurement)
        self.act_stop.setEnabled(False)
        m_run.addAction(self.act_stop)

        m_view = mb.addMenu("&View")
        a_clear_log = QAction("Clear event log", self)
        a_clear_log.triggered.connect(lambda: self.event_log.clear())
        m_view.addAction(a_clear_log)

        m_help = mb.addMenu("&Help")
        a_about = QAction("About", self)
        a_about.triggered.connect(self._show_about)
        m_help.addAction(a_about)

    def _init_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.lbl_state = QLabel("●  Idle")
        self.lbl_state.setStyleSheet(f"color: {CT_GREEN}; font-weight: 600;")
        sb.addWidget(self.lbl_state)
        sb.addWidget(QLabel("  |  "))
        daq_text = "DAQ: Dev1 connected" if self.hw.connected else "DAQ: not connected"
        self.lbl_daq = QLabel(daq_text)
        sb.addWidget(self.lbl_daq)
        sb.addPermanentWidget(QLabel(""))
        self.lbl_clock = QLabel("")
        sb.addPermanentWidget(self.lbl_clock)

    def _set_state(self, label, color):
        self.lbl_state.setText(f"●  {label}")
        self.lbl_state.setStyleSheet(f"color: {color}; font-weight: 600;")

    def _tick_clock(self):
        self.lbl_clock.setText(time.strftime("%Y-%m-%d  %H:%M:%S"))

    def _show_about(self):
        QMessageBox.about(self, f"About {APP_NAME}",
            f"<h3>{APP_NAME}</h3>"
            f"<p>{APP_VERSION}</p>"
            f"<p>Dual-gate quantum transport mapping with NI DAQ + SR830 + SR560.</p>"
            f"<p>Sweep parameterised by (min, max, N).<br>"
            f"Per-point n_avg averaging.<br>"
            f"Voltage limits enforced in both Vtg-Vbg and n-D modes.</p>")

    # ============================================================
    # Misc helpers
    # ============================================================
    @staticmethod
    def _sanitize(s):
        s = s.strip()
        for c in '<>:"/\\|?*':
            s = s.replace(c, '_')
        return s.replace(' ', '_')

    def _refresh_filename(self):
        date = time.strftime("%Y%m%d_%H%M%S")
        sample = self._sanitize(self.le_sample.text()) or "sample"
        device = self._sanitize(self.le_device.text()) or "device"
        run    = self._sanitize(self.le_run_name.text()) or "run"
        self.lbl_filename.setText(f"{date}_{sample}_{device}_{run}.csv")

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(
            self, "Choose output folder", self.le_folder.text() or os.getcwd())
        if d:
            self.le_folder.setText(d)

    def _update_iac_label(self):
        try:
            v = float(self.hw_inputs['v_osc'].text())
            r = float(self.hw_inputs['r_series'].text())
            i = v / r
            if abs(i) >= 1e-3:
                self.lbl_iac.setText(f'{i*1e3:.3f} mA')
            elif abs(i) >= 1e-6:
                self.lbl_iac.setText(f'{i*1e6:.3f} µA')
            elif abs(i) >= 1e-9:
                self.lbl_iac.setText(f'{i*1e9:.3f} nA')
            else:
                self.lbl_iac.setText(f'{i:.3e} A')
        except (ValueError, ZeroDivisionError):
            self.lbl_iac.setText('— A')

    def _refresh_axis_labels(self):
        if not hasattr(self, 'line_slot_plots'):
            return
        # Use locked strategy if scanning, else the live mode/labels
        if self.locked_strategy is not None:
            fast_lbl = self.locked_strategy.fast_axis_label
            slow_lbl = self.locked_strategy.slow_axis_label
        else:
            mode = self.cb_sweep_mode.currentData() if hasattr(self, 'cb_sweep_mode') else SWEEP_MODE_VTGVBG
            if mode == SWEEP_MODE_ND:
                fast = self.cb_nd_fast_axis.currentData()
                slow = 'D' if fast == 'n' else 'n'
                funit = '×10¹² cm⁻²' if fast == 'n' else 'V/nm'
                sunit = '×10¹² cm⁻²' if slow == 'n' else 'V/nm'
                fast_lbl = f'{fast} ({funit})'
                slow_lbl = f'{slow} ({sunit})'
            else:
                fast_role = self.cb_fast_gate.currentData() if hasattr(self, 'cb_fast_gate') else 'top'
                fast = 'Vtg' if fast_role == 'top' else 'Vbg'
                slow = 'Vbg' if fast_role == 'top' else 'Vtg'
                fast_lbl = f'{fast} (V)'
                slow_lbl = f'{slow} (V)'
        for pw in self.line_slot_plots:
            pw.setLabel('bottom', fast_lbl)
        for pw in self.map_slot_plots:
            pw.setLabel('bottom', fast_lbl)
            pw.setLabel('left',   slow_lbl)

    def _current_channel_name(self, ai_idx):
        return self.ch_name_inputs[ai_idx].text().strip() or f'AI{ai_idx}'

    def _current_channel_kind(self, ai_idx):
        return self.ch_kind_combos[ai_idx].currentText()

    def _current_channel_unit(self, ai_idx):
        return KIND_UNIT[self._current_channel_kind(ai_idx)]

    # ============================================================
    # Channel selectors (line slot multi-select + map slot combos)
    # ============================================================
    def _rebuild_channel_selectors(self):
        self._rebuild_line_slot_menus()
        self._rebuild_map_slot_combos()

    def _rebuild_line_slot_menus(self):
        if not hasattr(self, 'line_slot_menus'):
            return
        for slot_idx in range(NUM_LINE_SLOTS):
            menu = self.line_slot_menus[slot_idx]
            old_actions = self.line_slot_actions[slot_idx]
            prev_selected = {ai for ai, a in old_actions.items() if a.isChecked()}

            menu.clear()
            new_actions = {}
            for ai in range(NUM_AI):
                name = self._current_channel_name(ai)
                unit = self._current_channel_unit(ai)
                act = QAction(f'AI{ai}  ·  {name}  ({unit})', menu)
                act.setCheckable(True)
                act.setChecked(ai in prev_selected)
                act.triggered.connect(
                    lambda _checked=False, s=slot_idx: self._on_line_slot_changed(s))
                menu.addAction(act)
                new_actions[ai] = act
            self.line_slot_actions[slot_idx] = new_actions

            self._update_line_slot_button_text(slot_idx)
            self._on_line_slot_changed(slot_idx)

    def _update_line_slot_button_text(self, slot_idx):
        actions = self.line_slot_actions[slot_idx]
        names = [self._current_channel_name(ai)
                 for ai, a in actions.items() if a.isChecked()]
        if not names:
            txt = f"Slot {slot_idx + 1}:  (no channels)  ▾"
        elif len(names) <= 3:
            txt = f"Slot {slot_idx + 1}:  {', '.join(names)}  ▾"
        else:
            txt = f"Slot {slot_idx + 1}:  {len(names)} channels  ▾"
        self.line_slot_buttons[slot_idx].setText(txt)

    def _on_line_slot_changed(self, slot_idx):
        self._update_line_slot_button_text(slot_idx)
        self._rebuild_line_slot_lines(slot_idx)

    def _rebuild_line_slot_lines(self, slot_idx):
        pw = self.line_slot_plots[slot_idx]
        legend = self.line_slot_legends[slot_idx]
        for item in list(pw.listDataItems()):
            pw.removeItem(item)
        try: legend.clear()
        except Exception: pass

        self.line_slot_lines[slot_idx] = {}
        actions = self.line_slot_actions[slot_idx]
        selected_ais = [ai for ai, a in actions.items() if a.isChecked()]
        if not selected_ais:
            pw.setLabel('left', '')
            return

        units = {self._current_channel_unit(ai) for ai in selected_ais}
        ylabel = list(units)[0] if len(units) == 1 else '(mixed units)'
        pw.setLabel('left', ylabel)

        bidi = (self.locked_bidirectional if self.locked_channels
                else self.cb_bidirectional.isChecked())
        dirs = list(DIRECTIONS) if bidi else ['fwd']

        for ai in selected_ais:
            color = CHANNEL_COLORS[ai % len(CHANNEL_COLORS)]
            name = self._current_channel_name(ai)
            for d in dirs:
                style = Qt.SolidLine if d == 'fwd' else Qt.DashLine
                label = f'{name} ({d})' if bidi else name
                pen = pg.mkPen(color=color, width=2, style=style)
                line = pw.plot(name=label, pen=pen)
                self.line_slot_lines[slot_idx][(ai, d)] = line

        for (ai, d), line in self.line_slot_lines[slot_idx].items():
            buf = self.curr_buf[d]
            if buf['fast']:
                line.setData(buf['fast'], buf['vals'][ai])

    def _rebuild_map_slot_combos(self):
        if not hasattr(self, 'map_slot_combos'):
            return
        bidi = self.cb_bidirectional.isChecked()
        dirs = list(DIRECTIONS) if bidi else ['fwd']
        for slot_idx, combo in enumerate(self.map_slot_combos):
            prev = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem('(off)', userData=None)
            for ai in range(NUM_AI):
                name = self._current_channel_name(ai)
                unit = self._current_channel_unit(ai)
                for d in dirs:
                    combo.addItem(f'AI{ai} · {name} ({unit}) · {d}',
                                  userData=(ai, d))
            target = 0
            if prev is not None:
                for k in range(combo.count()):
                    if combo.itemData(k) == prev:
                        target = k
                        break
            combo.setCurrentIndex(target)
            combo.blockSignals(False)
            self._update_map_slot_title(slot_idx)

    def _update_map_slot_title(self, slot_idx):
        sel = self.map_slot_combos[slot_idx].currentData()
        if sel is None:
            self.map_slot_plots[slot_idx].setTitle("(off)")
        else:
            ai, d = sel
            self.map_slot_plots[slot_idx].setTitle(
                f"AI{ai} · {self._current_channel_name(ai)} "
                f"({self._current_channel_unit(ai)}) · {d}")

    def _on_map_slot_changed(self, slot_idx):
        self._update_map_slot_title(slot_idx)
        self._refresh_map_slot(slot_idx)

    def _refresh_map_slot(self, slot_idx):
        sel = self.map_slot_combos[slot_idx].currentData()
        img = self.map_slot_images[slot_idx]
        if sel is None or self.map_data is None:
            img.clear()
            return
        ai, d = sel
        try:
            data = self.map_data[ai][d]
        except (KeyError, IndexError):
            img.clear()
            return
        valid = data[~np.isnan(data)]
        if valid.size < 2:
            return
        lo, hi = np.percentile(valid, [2, 98])
        if hi - lo < 1e-12:
            hi = lo + 1e-12
        img.setImage(data, levels=(lo, hi), autoLevels=False)
        self._apply_image_voltage_transform(img, data.shape)

    def _apply_image_voltage_transform(self, img, data_shape):
        """QTransform-based image rect (independent of pyqtgraph version
        differences in setRect)."""
        if self.locked_rect is None or len(data_shape) < 2:
            return
        ny, nx = data_shape[-2], data_shape[-1]
        if nx <= 0 or ny <= 0:
            return
        xmin = self.locked_rect.left()
        ymin = self.locked_rect.top()
        w    = self.locked_rect.width()
        h    = self.locked_rect.height()
        tr = QTransform()
        tr.translate(xmin, ymin)
        tr.scale(w / nx, h / ny)
        img.setTransform(tr)

    # ============================================================
    # Event log
    # ============================================================
    def log_event(self, msg, level='info'):
        ts = time.strftime("%H:%M:%S")
        color = {
            'info':    CT_SUBTEXT0,
            'success': CT_GREEN,
            'warning': CT_YELLOW,
            'error':   CT_RED,
        }.get(level, CT_TEXT)
        line = (f'<span style="color:{CT_OVERLAY0};">{ts}</span>  '
                f'<span style="color:{color};">{msg}</span>')
        self.event_log.appendHtml(line)

    # ============================================================
    # Lock / unlock config widgets during a scan
    # ============================================================
    def _config_widgets(self):
        widgets = []
        widgets += [self.le_sample, self.le_device,
                    self.le_operator, self.le_run_name]
        widgets += [self.cb_sweep_mode, self.cb_nd_fast_axis]
        widgets += [self.cb_top_ao, self.cb_bot_ao, self.cb_fast_gate]
        widgets += list(self.lim_inputs.values())
        widgets += list(self.geom_inputs.values())
        widgets += list(self.sweep_inputs.values())
        widgets.append(self.cb_bidirectional)
        widgets += list(self.hw_inputs.values())
        widgets += list(self.sr830_inputs.values())
        widgets += list(self.ch_name_inputs)
        widgets += list(self.ch_enable_checks)
        widgets += list(self.ch_kind_combos)
        widgets += list(self.ch_sens_inputs)
        widgets += list(self.ch_gain_inputs)
        widgets.append(self.le_folder)
        return widgets

    def _set_config_enabled(self, enabled):
        for w in self._config_widgets():
            w.setEnabled(enabled)
        if enabled:
            for ai in range(NUM_AI):
                self._refresh_kind_widgets(ai)
            # Re-apply mode-dependent disabling
            self._on_sweep_mode_changed()

    # ============================================================
    # Param parsing
    # ============================================================
    def _parse_params(self):
        """Parse all GUI fields → params dict for MeasurementThread.

        Raises ValueError with a human-readable message on the first
        validation failure.  All parsing happens BEFORE any side effect,
        so a parse failure leaves the GUI state untouched.
        """
        # ---------- 1) sweep range + N points ----------
        try:
            fast_min = float(self.sweep_inputs['fast_min'].text())
            fast_max = float(self.sweep_inputs['fast_max'].text())
            num_fast = int(float(self.sweep_inputs['fast_N'].text()))
            slow_min = float(self.sweep_inputs['slow_min'].text())
            slow_max = float(self.sweep_inputs['slow_max'].text())
            num_slow = int(float(self.sweep_inputs['slow_N'].text()))
        except ValueError:
            raise ValueError("Sweep parameter parsing failed (must be numeric).")
        if num_fast < 1 or num_slow < 1:
            raise ValueError("N points must be ≥ 1 on both axes.")
        if num_fast == 1 and abs(fast_max - fast_min) > 1e-12:
            raise ValueError("fast_N = 1 requires fast_min == fast_max.")
        if num_slow == 1 and abs(slow_max - slow_min) > 1e-12:
            raise ValueError("slow_N = 1 requires slow_min == slow_max.")
        if num_fast > 1 and fast_max < fast_min:
            raise ValueError("fast_max must be ≥ fast_min.")
        if num_slow > 1 and slow_max < slow_min:
            raise ValueError("slow_max must be ≥ slow_min.")

        # ---------- 2) timing ----------
        try:
            t_dwell        = float(self.sweep_inputs['t_dwell'].text())
            t_settle_slow  = float(self.sweep_inputs['t_settle_slow'].text())
            t_after_fwd    = float(self.sweep_inputs['t_settle_after_fwd'].text())
            t_after_bwd    = float(self.sweep_inputs['t_settle_after_bwd'].text())
            t_retrace_step = float(self.sweep_inputs['t_retrace_step'].text())
            n_avg          = int(float(self.sweep_inputs['n_avg'].text()))
        except ValueError:
            raise ValueError("Timing parameter parsing failed (must be numeric).")
        for name, val in (('t_dwell', t_dwell),
                          ('t_settle_slow', t_settle_slow),
                          ('t_settle_after_fwd', t_after_fwd),
                          ('t_settle_after_bwd', t_after_bwd),
                          ('t_retrace_step', t_retrace_step)):
            if val < 0:
                raise ValueError(f"{name} must be ≥ 0.")
        if n_avg < 1:
            raise ValueError("n_avg must be ≥ 1.")

        # ---------- 3) lock-in source ----------
        try:
            v_osc    = float(self.hw_inputs['v_osc'].text())
            r_series = float(self.hw_inputs['r_series'].text())
        except ValueError:
            raise ValueError("Lock-in / source parsing failed.")
        if not (SR830_VOSC_MIN <= abs(v_osc) <= SR830_VOSC_MAX):
            raise ValueError(
                f"V_osc = {v_osc} V is outside the SR830 sin-out range "
                f"[{SR830_VOSC_MIN}, {SR830_VOSC_MAX}] V_rms.")
        if r_series < R_SERIES_MIN_OHM:
            raise ValueError(
                f"R_series = {r_series:.3e} Ω is below the safety floor "
                f"{R_SERIES_MIN_OHM:.0e} Ω. Refusing to push that much "
                f"current into the device.")

        # ---------- 4) SR830 metadata ----------
        def _sr_text(key):
            w = self.sr830_inputs[key]
            return w.currentText() if isinstance(w, NoWheelComboBox) \
                                   else w.text()
        try:
            lockin_meta = LockInMetadata(
                frequency_hz       = float(_sr_text('freq')),
                time_constant_s    = float(_sr_text('tc')),
                filter_slope_db_oct= int(float(_sr_text('slope'))),
                reserve            = _sr_text('reserve').strip(),
                sync_filter        = _sr_text('sync').strip().lower()
                                     in ('on', 'true', '1', 'yes'),
                reference_phase_deg= float(_sr_text('phase')),
                input_source       = _sr_text('source').strip(),
                input_coupling     = _sr_text('coupling').strip(),
                input_ground       = _sr_text('ground').strip(),
            )
        except ValueError:
            raise ValueError("SR830 metadata parsing failed.")
        if lockin_meta.time_constant_s <= 0:
            raise ValueError("SR830 time constant must be > 0.")

        # ---------- 5) channels ----------
        channels = []
        for ai in range(NUM_AI):
            kind = self.ch_kind_combos[ai].currentText()
            sens, gain = 0.0, 0.0
            if kind == 'R':
                try:
                    sens = float(self.ch_sens_inputs[ai].text())
                    gain = float(self.ch_gain_inputs[ai].text())
                except ValueError:
                    raise ValueError(f"AI{ai}: sens / gain parsing failed.")
                if sens <= 0 or gain <= 0:
                    raise ValueError(f"AI{ai}: sens and gain must be > 0.")
            channels.append(ChannelConfig(
                ai_index=ai,
                name=self._current_channel_name(ai),
                enabled=self.ch_enable_checks[ai].isChecked(),
                kind=kind,
                sens=sens,
                gain=gain,
            ))
        enabled = [c for c in channels if c.enabled]
        if not enabled:
            raise ValueError("No channels enabled. Enable at least one AI.")
        names = [c.csv_col_name for c in enabled]
        if len(set(names)) != len(names):
            raise ValueError("Enabled channels have duplicate CSV column names. "
                             "Give each enabled AI a unique name.")

        # ---------- 6) AO assignment (explicit top/bot) ----------
        if self.cb_top_ao.currentText() == self.cb_bot_ao.currentText():
            raise ValueError("Top and bottom gates cannot share the same AO.")
        ao_map = {'AO0': self.hw.ao_chans[0], 'AO1': self.hw.ao_chans[1]}
        ao_top = ao_map[self.cb_top_ao.currentText()]
        ao_bot = ao_map[self.cb_bot_ao.currentText()]

        # ---------- 7) voltage limits ----------
        try:
            vtg_max = float(self.lim_inputs['Vtg_max'].text())
            vbg_max = float(self.lim_inputs['Vbg_max'].text())
        except ValueError:
            raise ValueError("Voltage limit parsing failed.")
        if vtg_max <= 0 or vbg_max <= 0:
            raise ValueError("|Vtg|_max and |Vbg|_max must be > 0.")
        # The DAQ AO hardware itself caps at ±10 V; refuse to even pretend
        # we can go higher, because the CSV would record the un-clipped value.
        if vtg_max > 10.0 or vbg_max > 10.0:
            raise ValueError(
                "|Vtg|_max and |Vbg|_max cannot exceed 10 V (NI DAQ AO range).")
        limits = VoltageLimits(Vtg_max=vtg_max, Vbg_max=vbg_max)

        # ---------- 8) build strategy ----------
        mode = self.cb_sweep_mode.currentData()
        if mode == SWEEP_MODE_VTGVBG:
            fast_role = self.cb_fast_gate.currentData()  # 'top' or 'bot'
            fast_is_top = (fast_role == 'top')
            strategy = VtgVbgStrategy(
                limits=limits, fast_is_top=fast_is_top,
                fast_min=fast_min, fast_max=fast_max, num_fast=num_fast,
                slow_min=slow_min, slow_max=slow_max, num_slow=num_slow,
            )
            geometry = None

        elif mode == SWEEP_MODE_ND:
            try:
                d_t  = float(self.geom_inputs['d_t'].text())
                d_b  = float(self.geom_inputs['d_b'].text())
                eps  = float(self.geom_inputs['eps'].text())
                Vtg0 = float(self.geom_inputs['Vtg0'].text())
                Vbg0 = float(self.geom_inputs['Vbg0'].text())
            except ValueError:
                raise ValueError("Geometry parameter parsing failed.")
            if d_t <= 0 or d_b <= 0 or eps <= 0:
                raise ValueError("Geometry: d_t, d_b, eps must be > 0.")
            geometry = GeometryConfig(
                d_t_nm=d_t, d_b_nm=d_b, eps_hBN=eps,
                Vtg0=Vtg0, Vbg0=Vbg0,
            )
            fast_axis = self.cb_nd_fast_axis.currentData()
            strategy = NDStrategy(
                geometry=geometry, limits=limits, fast_axis=fast_axis,
                fast_min=fast_min, fast_max=fast_max, num_fast=num_fast,
                slow_min=slow_min, slow_max=slow_max, num_slow=num_slow,
            )
        else:
            raise ValueError(f"Unknown sweep mode: {mode!r}")

        # ---------- 9) output path ----------
        folder = self.le_folder.text().strip() or os.getcwd()
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Cannot create output folder: {e}")
        self._refresh_filename()
        base_name = self.lbl_filename.text()
        save_path = os.path.join(folder, base_name)
        # Filename collision safety net.  Two scans launched in the same
        # wall-clock second would otherwise share a stem and silently
        # overwrite each other's CSV/TXT/JSON triple.  Append _2, _3, ...
        # to the stem until an unused name is found.  Checks the CSV file
        # AND the .txt / .json sidecars so a partial leftover from an
        # aborted run doesn't trick us into reusing its stem.
        if (os.path.exists(save_path)
                or os.path.exists(os.path.splitext(save_path)[0] + '.txt')
                or os.path.exists(os.path.splitext(save_path)[0] + '.json')):
            stem, ext = os.path.splitext(save_path)
            for k in range(2, 1000):
                candidate = f"{stem}_{k}{ext}"
                stem_k = os.path.splitext(candidate)[0]
                if not (os.path.exists(candidate)
                        or os.path.exists(stem_k + '.txt')
                        or os.path.exists(stem_k + '.json')):
                    save_path = candidate
                    break
            else:
                raise ValueError(
                    "Could not find a free filename after 1000 attempts. "
                    "Clean up the output folder.")

        return dict(
            strategy=strategy,
            geometry=geometry,
            limits=limits,
            sweep_mode=mode,
            t_dwell=t_dwell,
            t_settle_slow=t_settle_slow,
            t_settle_after_fwd=t_after_fwd,
            t_settle_after_bwd=t_after_bwd,
            t_retrace_step=t_retrace_step,
            n_avg=n_avg,
            v_osc=v_osc, r_series=r_series,
            lockin_meta=lockin_meta,
            channels=channels,
            bidirectional=self.cb_bidirectional.isChecked(),
            ao_top=ao_top, ao_bot=ao_bot,
            save_path=save_path,
            sample=self.le_sample.text().strip(),
            device=self.le_device.text().strip(),
            operator=self.le_operator.text().strip(),
            run_name=self.le_run_name.text().strip(),
        )

    # ============================================================
    # Start / Stop
    # ============================================================
    def start_measurement(self):
        if self.thread and self.thread.isRunning():
            self.log_event("A scan is already running.", "warning")
            return

        try:
            params = self._parse_params()
        except Exception as e:
            self.log_event(f"Cannot start: {e}", "error")
            QMessageBox.warning(self, "Invalid parameters", str(e))
            return

        strategy = params['strategy']
        lockin_meta = params['lockin_meta']

        # ------ Reviewer-grade pre-flight: dwell vs SR830 TC ------
        tc = lockin_meta.time_constant_s
        if params['t_dwell'] < 5.0 * tc and not params['bidirectional']:
            reply = QMessageBox.warning(
                self, "Settling-time warning",
                f"t_dwell = {params['t_dwell']*1000:.0f} ms is less than\n"
                f"5 × SR830 TC ({5*tc*1000:.0f} ms).\n\n"
                f"In single-direction mode this leaves a one-sided lock-in "
                f"settling artefact in the data — the readings lag the gate "
                f"voltage along the fast axis.\n\n"
                f"Recommended fixes:\n"
                f"  • increase t_dwell to ≥ {5*tc:.2f} s, OR\n"
                f"  • enable bidirectional mode and average fwd/bwd.\n\n"
                f"Continue anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                self.log_event("Scan aborted by user (settling pre-flight).", "info")
                return

        # ------ Reverse-map pre-flight (n-D mode only) ------
        if strategy.mode_id == SWEEP_MODE_ND:
            skip_frac = strategy.precheck_skip_fraction(n_samples=21)
            if skip_frac >= 0.999:
                self.log_event(
                    f"Pre-flight FAIL: 100% of target points are outside "
                    f"hBN limits.", "error")
                QMessageBox.critical(self, "Pre-flight failed",
                    "Every target point in the n–D rectangle reverse-maps "
                    "outside the voltage limits.\n\n"
                    "Adjust geometry, voltage limits, or n / D range.")
                return
            if skip_frac > 0.001:
                pct = skip_frac * 100
                self.log_event(
                    f"Pre-flight: {pct:.1f}% of target points will be SKIPPED.",
                    "warning")
                reply = QMessageBox.question(
                    self, "Some points will be skipped",
                    f"About {pct:.1f}% of the (n, D) target points reverse-map "
                    f"to gate voltages outside the limits.\n\n"
                    f"These will be skipped (no DAQ write, NaN in CSV, "
                    f"transparent in the 2D map).\n\n"
                    f"Continue?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply != QMessageBox.Yes:
                    self.log_event("Scan aborted by user (n-D pre-flight).", "info")
                    return
            else:
                self.log_event("Pre-flight: 0% skip, all target points safe.",
                               "success")
        else:
            # In Vtg-Vbg mode, pre-flight too — catches the case where
            # the user typed a range outside the voltage limits.
            skip_frac = strategy.precheck_skip_fraction(n_samples=21)
            if skip_frac >= 0.999:
                self.log_event(
                    "Pre-flight FAIL: every target outside voltage limits.",
                    "error")
                QMessageBox.critical(self, "Pre-flight failed",
                    "Every target point is outside |Vtg|_max / |Vbg|_max.\n\n"
                    "Adjust the sweep range or the voltage limits.")
                return
            if skip_frac > 0.001:
                pct = skip_frac * 100
                self.log_event(
                    f"Pre-flight: {pct:.1f}% of Vtg-Vbg targets are outside "
                    f"voltage limits and will be skipped.", "warning")
                reply = QMessageBox.question(
                    self, "Some points will be skipped",
                    f"About {pct:.1f}% of the Vtg-Vbg target points are outside "
                    f"the voltage limits and will be skipped (NaN in CSV).\n\n"
                    f"Continue?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply != QMessageBox.Yes:
                    self.log_event("Scan aborted by user (Vtg-Vbg pre-flight).",
                                   "info")
                    return

        # ------ commit locked state ------
        self.locked_channels      = list(params['channels'])
        self.locked_bidirectional = params['bidirectional']
        self.locked_strategy      = strategy

        self.map_data = {
            ai: {
                d: np.full((strategy.num_slow, strategy.num_fast), np.nan)
                for d in DIRECTIONS
            } for ai in range(NUM_AI)
        }

        # locked rect — for single-point axes give it a small finite extent
        # so the QRectF doesn't degenerate.
        fmin = strategy.fast_min
        fmax = strategy.fast_max
        smin = strategy.slow_min
        smax = strategy.slow_max
        if strategy.num_fast == 1:
            half = max(abs(fmin) * 0.01, 1e-6)
            fmin -= half; fmax += half
        if strategy.num_slow == 1:
            half = max(abs(smin) * 0.01, 1e-6)
            smin -= half; smax += half

        self.locked_rect = QRectF(
            fmin, smin, fmax - fmin, smax - smin)
        for img in self.map_slot_images:
            img.clear()

        self.log_event(
            f"Locked map rect: {strategy.fast_label} ∈ "
            f"[{strategy.fast_min:.4g}, {strategy.fast_max:.4g}] {strategy.fast_unit}, "
            f"{strategy.slow_label} ∈ "
            f"[{strategy.slow_min:.4g}, {strategy.slow_max:.4g}] {strategy.slow_unit} "
            f"({strategy.num_slow}×{strategy.num_fast} pixels)",
            "info")

        self._rebuild_channel_selectors()
        self._refresh_axis_labels()

        # ------ lock plot ranges ------
        self.line_slot_plots[0].setXRange(fmin, fmax, padding=0)
        self.map_slot_plots[0].setYRange(smin, smax, padding=0)
        for pw in (list(self.line_slot_plots) + list(self.map_slot_plots)):
            pw.enableAutoRange(axis='x', enable=False)
        for pw in self.map_slot_plots:
            pw.enableAutoRange(axis='y', enable=False)
        for pw in self.line_slot_plots:
            pw.enableAutoRange(axis='y', enable=True)

        self.curr_row_idx = -1
        self.curr_buf = self._make_empty_1d_buffers()

        per_row = strategy.num_fast * (2 if params['bidirectional'] else 1)
        self._total_points = strategy.num_slow * per_row
        self._points_done  = 0
        self._scan_started_at = time.monotonic()
        self.progress_bar.setValue(0)
        self.lbl_step.setText(f"Step 0 / {self._total_points}")
        self.lbl_elapsed.setText("Elapsed  0s")
        self.lbl_eta.setText("ETA  —")
        self.lbl_now.setText("—")

        self.thread = MeasurementThread(params, self.hw)
        self.thread.point_ready.connect(self.on_point)
        self.thread.row_finished.connect(self.on_row_finished)
        self.thread.log_msg.connect(self.log_event)
        self.thread.finished_ok.connect(self.on_finished_ok)
        self.thread.error_occurred.connect(self.on_error)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.act_start.setEnabled(False)
        self.act_stop.setEnabled(True)
        self._set_config_enabled(False)
        self._set_state("Scanning", CT_YELLOW)
        self.thread.start()

    def stop_measurement(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.log_event("Abort requested by user.", "warning")
            self._set_state("Aborting", CT_PEACH)

    # ============================================================
    # Worker slots
    # ============================================================
    def on_point(self, info):
        i_idx = info.i_idx
        j_idx = info.j_idx
        direction = info.direction

        if i_idx != self.curr_row_idx:
            self.curr_row_idx = i_idx
            self.curr_buf = self._make_empty_1d_buffers()
            for slot_idx in range(NUM_LINE_SLOTS):
                for line in self.line_slot_lines[slot_idx].values():
                    line.setData([], [])

        buf = self.curr_buf[direction]
        buf['fast'].append(info.fast_target)
        for ai in range(NUM_AI):
            buf['vals'][ai].append(info.values[ai])
            self.map_data[ai][direction][i_idx, j_idx] = info.values[ai]

        for slot_idx in range(NUM_LINE_SLOTS):
            for (ai, d), line in self.line_slot_lines[slot_idx].items():
                if d == direction:
                    line.setData(buf['fast'], buf['vals'][ai])

        # Progress
        self._points_done += 1
        pct = int(100 * self._points_done / max(self._total_points, 1))
        self.progress_bar.setValue(pct)
        elapsed = time.monotonic() - self._scan_started_at
        eta = (elapsed * (self._total_points - self._points_done)
               / self._points_done) if self._points_done > 0 else None
        self.lbl_step.setText(f"Step  {self._points_done} / {self._total_points}")
        self.lbl_elapsed.setText(f"Elapsed  {fmt_dur(elapsed)}")
        self.lbl_eta.setText(f"ETA  {fmt_dur(eta)}")

        s = self.locked_strategy
        if info.skipped:
            self.lbl_now.setText(
                f"{s.slow_label}  {info.slow_target:+.4g}    "
                f"{s.fast_label}  {info.fast_target:+.4g}    "
                f"{direction}  [SKIP]")
        else:
            self.lbl_now.setText(
                f"{s.slow_label}  {info.slow_target:+.4g}    "
                f"{s.fast_label}  {info.fast_target:+.4g}    "
                f"{direction}    "
                f"Vtg={info.Vtg:+.3f}V  Vbg={info.Vbg:+.3f}V")

    def on_row_finished(self, _i_idx):
        for slot_idx in range(NUM_MAP_SLOTS):
            self._refresh_map_slot(slot_idx)

    def on_finished_ok(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.act_start.setEnabled(True)
        self.act_stop.setEnabled(False)
        self._set_config_enabled(True)
        if self._points_done >= self._total_points:
            self._set_state("Idle", CT_GREEN)
            self.progress_bar.setValue(100)
        else:
            self._set_state("Idle (aborted)", CT_PEACH)

    def on_error(self, tb):
        self.log_event("Scan crashed (see traceback below).", "error")
        for line in tb.strip().splitlines()[-6:]:
            self.log_event("    " + line, "error")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.act_start.setEnabled(True)
        self.act_stop.setEnabled(False)
        self._set_config_enabled(True)
        self._set_state("Error", CT_RED)
        print(tb, file=sys.stderr)

    # ============================================================
    # QSettings persistence
    # ============================================================
    def _persistent_widgets(self):
        items = []
        for k, le in self.sweep_inputs.items():
            items.append(('lineedit', f'sweep/{k}', le))
        items.append(('checkbox', 'sweep/bidirectional', self.cb_bidirectional))
        for k, le in self.lim_inputs.items():
            items.append(('lineedit', f'limits/{k}', le))
        for k, le in self.geom_inputs.items():
            items.append(('lineedit', f'geometry/{k}', le))
        for k, le in self.hw_inputs.items():
            items.append(('lineedit', f'lockin/{k}', le))
        # SR830 inputs are mixed: QLineEdit for free-form fields, QComboBox
        # for enumerated fields.  Detect at runtime.
        for k, w in self.sr830_inputs.items():
            kind = 'combo' if isinstance(w, NoWheelComboBox) else 'lineedit'
            items.append((kind, f'sr830/{k}', w))
        for ai in range(NUM_AI):
            items.append(('lineedit', f'channels/{ai}/name',    self.ch_name_inputs[ai]))
            items.append(('checkbox', f'channels/{ai}/enabled', self.ch_enable_checks[ai]))
            items.append(('combo',    f'channels/{ai}/kind',    self.ch_kind_combos[ai]))
            items.append(('lineedit', f'channels/{ai}/sens',    self.ch_sens_inputs[ai]))
            items.append(('lineedit', f'channels/{ai}/gain',    self.ch_gain_inputs[ai]))
        items.append(('combo',    'gates/top_ao',  self.cb_top_ao))
        items.append(('combo',    'gates/bot_ao',  self.cb_bot_ao))
        items.append(('lineedit', 'meta/sample',   self.le_sample))
        items.append(('lineedit', 'meta/device',   self.le_device))
        items.append(('lineedit', 'meta/operator', self.le_operator))
        items.append(('lineedit', 'meta/run_name', self.le_run_name))
        items.append(('lineedit', 'output/folder', self.le_folder))
        return items

    def _load_settings(self):
        s = QSettings(ORG_NAME, SETTINGS_NAME)
        for kind, key, w in self._persistent_widgets():
            v = s.value(key)
            if v is None:
                continue
            if kind == 'lineedit':
                w.setText(str(v))
            elif kind == 'checkbox':
                if isinstance(v, bool):
                    w.setChecked(v)
                else:
                    w.setChecked(str(v).lower() in ('true', '1', 'yes'))
            elif kind == 'combo':
                w.setCurrentText(str(v))

        # mode / fast_axis / fast_gate combos use userData
        for combo, key in (
            (self.cb_sweep_mode,    'mode/sweep_mode'),
            (self.cb_nd_fast_axis,  'mode/nd_fast_axis'),
            (self.cb_fast_gate,     'gates/fast_gate'),
        ):
            v = s.value(key)
            if v is not None:
                for k in range(combo.count()):
                    if combo.itemData(k) == str(v):
                        combo.setCurrentIndex(k)
                        break

        # line slots & map slots
        for slot_idx in range(NUM_LINE_SLOTS):
            v = s.value(f'line_slots/{slot_idx}/sel', '')
            if v:
                try:
                    sel = {int(x) for x in str(v).split(',') if x}
                except Exception:
                    sel = set()
                self._pending_line_slot_sel = getattr(self, '_pending_line_slot_sel', {})
                self._pending_line_slot_sel[slot_idx] = sel
        for slot_idx in range(NUM_MAP_SLOTS):
            v = s.value(f'map_slots/{slot_idx}/sel', '')
            if v:
                try:
                    ai_str, d = str(v).split(',')
                    self._pending_map_slot_sel = getattr(self, '_pending_map_slot_sel', {})
                    self._pending_map_slot_sel[slot_idx] = (int(ai_str), d)
                except Exception:
                    pass

        geom = s.value('window/geometry')
        if geom is not None:
            self.restoreGeometry(geom)
        for name, split in (('h_split', self.h_split), ('v_split', self.v_split)):
            sizes = s.value(f'window/{name}')
            if sizes:
                try:
                    split.setSizes([int(x) for x in sizes])
                except Exception:
                    pass

        # Refresh derived UI
        self._update_iac_label()
        self._refresh_filename()
        for ai in range(NUM_AI):
            self._refresh_kind_widgets(ai)
        self._on_sweep_mode_changed()
        self._refresh_sweep_labels()
        self._refresh_axis_labels()
        self._refresh_effective_step_labels()
        self._rebuild_channel_selectors()

        if hasattr(self, '_pending_line_slot_sel'):
            for slot_idx, sel in self._pending_line_slot_sel.items():
                actions = self.line_slot_actions[slot_idx]
                for ai, act in actions.items():
                    act.setChecked(ai in sel)
                self._on_line_slot_changed(slot_idx)
            del self._pending_line_slot_sel
        if hasattr(self, '_pending_map_slot_sel'):
            for slot_idx, (ai, d) in self._pending_map_slot_sel.items():
                combo = self.map_slot_combos[slot_idx]
                for k in range(combo.count()):
                    if combo.itemData(k) == (ai, d):
                        combo.setCurrentIndex(k)
                        break
            del self._pending_map_slot_sel

    def _save_settings(self):
        s = QSettings(ORG_NAME, SETTINGS_NAME)
        for kind, key, w in self._persistent_widgets():
            if kind == 'lineedit':
                s.setValue(key, w.text())
            elif kind == 'checkbox':
                s.setValue(key, w.isChecked())
            elif kind == 'combo':
                s.setValue(key, w.currentText())
        s.setValue('mode/sweep_mode',   self.cb_sweep_mode.currentData())
        s.setValue('mode/nd_fast_axis', self.cb_nd_fast_axis.currentData())
        s.setValue('gates/fast_gate',   self.cb_fast_gate.currentData())
        for slot_idx, actions in enumerate(self.line_slot_actions):
            sel = ','.join(str(ai) for ai, a in actions.items() if a.isChecked())
            s.setValue(f'line_slots/{slot_idx}/sel', sel)
        for slot_idx, combo in enumerate(self.map_slot_combos):
            data = combo.currentData()
            if data is None:
                s.setValue(f'map_slots/{slot_idx}/sel', '')
            else:
                s.setValue(f'map_slots/{slot_idx}/sel', f'{data[0]},{data[1]}')
        s.setValue('window/geometry', self.saveGeometry())
        s.setValue('window/h_split', self.h_split.sizes())
        s.setValue('window/v_split', self.v_split.sizes())
        self.log_event("Settings saved.", "success")

    # ============================================================
    # Shutdown
    # ============================================================
    def closeEvent(self, ev):
        try:
            if self.thread and self.thread.isRunning():
                self.log_event("Window closing — aborting active scan first.",
                               "warning")
                self.thread.stop()
                ok = self.thread.wait(15000)
                if not ok:
                    self.log_event(
                        "Worker did not stop within 15 s — forcing terminate. "
                        "DAQ state may be inconsistent.", "error")
                    self.thread.terminate()
                    self.thread.wait(2000)
        finally:
            # Worker is now guaranteed dead — safe to touch DAQ from GUI thread.
            try:
                for ch in self.hw.ao_chans:
                    self.hw.ramp_ao(ch, 0.0)
            except Exception:
                pass
            try:
                self._save_settings()
            except Exception:
                pass
            try:
                self.hw.close()
            except Exception:
                pass
        ev.accept()


# =====================================================================
# main
# =====================================================================
def main():
    global DEMO_MODE, nidaqmx, TerminalConfiguration, AcquisitionType

    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} {APP_VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="By default, real NI DAQ hardware is used. "
               "Pass --demo to run with synthetic data and no hardware.")
    parser.add_argument(
        '--demo', action='store_true',
        help="Run in DEMO mode (no hardware access; synthetic AI data).")
    args = parser.parse_args()

    DEMO_MODE = bool(args.demo)
    if not DEMO_MODE:
        try:
            import nidaqmx as _nidaqmx
            from nidaqmx.constants import (
                TerminalConfiguration as _Term,
                AcquisitionType as _Acq,
            )
            nidaqmx = _nidaqmx
            TerminalConfiguration = _Term
            AcquisitionType = _Acq
        except ImportError as e:
            sys.stderr.write(
                f"ERROR: nidaqmx not available ({e}).\n"
                f"Either install nidaqmx or run with --demo for testing.\n")
            sys.exit(1)

    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)

    win = QuantumTransportGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()