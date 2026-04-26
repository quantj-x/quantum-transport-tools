# -*- coding: utf-8 -*-
"""
Differential Conductance (dI/dV) Mapping  —  v1.0
=================================================
Author: Xie Jiao  <xequalux@gmail.com>
Revised: 2026-04-16  (first release, voltage-bias mode)

Publication-grade dI/dV measurement program using the AC modulation
lock-in technique.  Hardware stack:
    NI DAQ  (AI readout, 8× single-ended, RSE)
    Keithley 2636B  (DC bias via SMU A + optional gate via SMU B)
    Stanford Research SR830  (AC modulation via sine out, phase-
    sensitive detection via A input; X/Y/R/θ displayed on CH1/CH2
    BNC outputs and read by DAQ AI for high-speed averaging)

This v1.0 focuses on voltage-bias mode with a single V_DC axis and
an optional outer gate axis.  Current-bias mode and field/temperature
outer axes are designed into the data model but not in this release.

See accompanying docstring section for hardware topology, signal-
chain math, and the operator checklist.  Provenance of every
instrumentation parameter is preserved in the TXT and JSON sidecars.

Hardware topology  —  voltage-bias mode
---------------------------------------

                                  ┌─── K2636B SMU A (V-source, I-sense)
                                  │    V_DC_cmd = sweep variable
                                  ▼
    V_DC_cmd ─── R_series_dc ─────┐
                                  ├── summing node ── sample HI
    V_osc × sin(2π f t) ── R_s_ac ┘                    │
      (SR830 sine out)                                 ▼
                                                    sample
                                                       │
                                           sample LO ──┴── GND
                                 (separate inner V-pair, 4-wire)
                                                       │
                                            SR560 voltage preamp
                                         (gain g, AC couple, A-B)
                                                       │
                                                       ▼
                                            SR830 A input
                                      (X, Y, R, θ on CH1/CH2 BNC)
                                                       │
                                                       ▼
                                              NI DAQ AI (RSE)

Signal-chain math  (R_sample << R_series_{dc,ac} cryogenic regime):
    V_DC_sample ≈ V_DC_cmd × R_sample / R_series_dc
    V_AC_sample ≈ V_osc    × R_sample / R_series_ac
    I_AC        ≈ V_osc / R_series_ac   (independent of R_sample)
    dI/dV       ≈ I_AC / V_AC_sample
                = (V_osc / R_series_ac) × g / V_lockin_X_rms     [see note]

The SR560 gain is g; it amplifies V_AC_sample to V_lockin_X_rms.
Therefore dI/dV  ≈  (V_osc × g) / (R_series_ac × V_lockin_X_rms).
This is what the CSV dIdV_X column records; sidecars preserve all
constituents so analysis can recompute under other assumptions.

Operator checklist
------------------
  [ ] K2636B front panel: BOTH OUTPUTS OFF before pressing START
  [ ] K2636B compliance currents set for THIS sample on BOTH channels
  [ ] SR830 reference mode = INTERNAL
  [ ] SR830 SIGNAL INPUT = A,  coupling = AC,  ground = Float
  [ ] SR830 reserve = Low Noise, sync filter ON
  [ ] SR830 CH1 display = X,  CH2 display = Y
      (or R and θ — channel kind in GUI MUST match physical display)
  [ ] SR830 sensitivity dial matches GUI sens fields
  [ ] SR560 gain and coupling match GUI fields
  [ ] R_series_dc and R_series_ac GUI values match physical resistors
  [ ] V_DC_cmd range + sample R_est  →  estimated V_DC_sample is safe
  [ ] V_osc × R_sample_est / R_series_ac  < kT/e at your T
      (energy-resolution limit; GUI warns if violated)
  [ ] NI DAQ AI terminal config = RSE for all used channels

History
-------
v1.0 (2026-04-16):  first release.  Voltage-bias 1D V_DC + optional
                    gate axis.  Built on n_D_mapping / hysteresis_mapping
                    v2.5-2636B hardware + CSV + sidecar infrastructure.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Hardware modules imported lazily in main() after --demo decision
nidaqmx                = None
TerminalConfiguration  = None
AcquisitionType        = None
pyvisa                 = None
VisaIOError            = None

# ----- Qt / pyqtgraph -----
from PyQt5.QtCore    import (Qt, QThread, pyqtSignal, QTimer, QSettings,
                             QRectF, QSize)
from PyQt5.QtGui     import QFont, QColor, QIcon, QKeySequence, QTransform
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QLineEdit, QGroupBox, QGridLayout, QFileDialog,
                             QMessageBox, QCheckBox, QComboBox, QSplitter,
                             QPlainTextEdit, QProgressBar, QStatusBar,
                             QAction, QScrollArea, QFrame, QToolButton,
                             QMenu, QSizePolicy)
import pyqtgraph as pg


# =====================================================================
# Globals & constants
# =====================================================================
APP_NAME       = "dI/dV Mapping"
APP_VERSION    = "v1.0"
ORG_NAME       = "lab.transport"
SETTINGS_NAME  = "didv_mapping_v1"

DEMO_MODE = False

NUM_AI         = 8
DIRECTIONS     = ('fwd', 'bwd')

# DAQ defaults
DAQ_AI_RATE_HZ   = 2000.0       # Hz, AI sample-clock rate
DAQ_DEFAULT_NAVG = 50           # samples averaged per point
DAQ_AI_TERMINAL  = 'RSE'

# Bias modes
BIAS_MODE_VOLTAGE = 'voltage'    # V_DC source, sample = 2- or 4-wire V-sense
BIAS_MODE_CURRENT = 'current'    # I_DC source, sample V via preamp
BIAS_MODE_OPTIONS = (BIAS_MODE_VOLTAGE, BIAS_MODE_CURRENT)

# K2636B channels
K2636_CHANNEL_OPTIONS = ('smua', 'smub')
K2636_DEFAULT_BIAS_CHANNEL = 'smua'
K2636_DEFAULT_GATE_CHANNEL = 'smub'

# Default safety values
SMU_DEFAULT_BIAS_COMPLIANCE_A = 1e-6   # 1 µA default on the bias channel
SMU_DEFAULT_GATE_COMPLIANCE_A = 100e-9 # 100 nA default on gate
SMU_COMPLIANCE_ABORT_FRACTION = 0.95   # abort when |I| > 0.95 × compliance

# Software ramp parameters for the DC bias source (fast axis inner steps
# use linear stepping at SMU_RAMP_STEP_V resolution; between scans we
# ramp to 0 at SMU_RAMP_RATE_V_PER_S).
SMU_RAMP_RATE_V_PER_S = 2.0
SMU_RAMP_STEP_V       = 0.02

# Per-point dwell defaults (operator can override in GUI).  These
# values matter for publication-grade dI/dV: the dwell MUST exceed
# several lock-in time constants before sampling.
PREFLIGHT_TC_MULTIPLIER = 5.0    # warn if t_dwell < 5 × TC
PREFLIGHT_TC_MINIMUM    = 3.0    # refuse to start if < 3 × TC

# SR830 setup enums (mirrors the physical instrument)
SR830_SENSITIVITY_VALUES = (
    # SCPI index → full-scale rms voltage (V)
    2e-9, 5e-9, 10e-9, 20e-9, 50e-9, 100e-9, 200e-9, 500e-9,
    1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6,
    500e-6, 1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3,
    200e-3, 500e-3, 1.0,
)
SR830_TC_VALUES = (
    # SCPI index → time constant (s)
    10e-6, 30e-6, 100e-6, 300e-6, 1e-3, 3e-3, 10e-3, 30e-3,
    100e-3, 300e-3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0,
    1000.0, 3000.0, 10000.0, 30000.0,
)
SR830_SLOPE_DB_VALUES = (6, 12, 18, 24)
SR830_RESERVE_VALUES  = ('High Reserve', 'Normal', 'Low Noise')
SR830_COUPLING_VALUES = ('AC', 'DC')
SR830_GROUND_VALUES   = ('Float', 'Ground')
SR830_SOURCE_VALUES   = ('A', 'A-B', 'I(1MOhm)', 'I(100MOhm)')

# Boltzmann constant (for energy-resolution preflight)
K_B_EV_PER_K = 8.617333e-5        # eV / K

# Catppuccin Mocha palette (consistent with prior programs)
CT_BASE       = '#1e1e2e'
CT_MANTLE     = '#181825'
CT_SURFACE0   = '#313244'
CT_SURFACE1   = '#45475a'
CT_TEXT       = '#cdd6f4'
CT_SUBTEXT1   = '#bac2de'
CT_OVERLAY    = '#6c7086'
CT_RED        = '#f38ba8'
CT_GREEN      = '#a6e3a1'
CT_YELLOW     = '#f9e2af'
CT_BLUE       = '#89b4fa'
CT_MAUVE      = '#cba6f7'
CT_PEACH      = '#fab387'
CT_TEAL       = '#94e2d5'


# =====================================================================
# Data classes
# =====================================================================
@dataclass
class ResistorNetwork:
    """The passive resistors in the DC+AC summing/series network.
    All values in ohms.

    For voltage-bias mode:
      - r_series_dc:  series resistor from Keithley V-source to summing node.
                      Typical 1 MΩ.  Limits DC current injection and
                      provides a known source impedance.
      - r_series_ac:  series resistor from SR830 sine-out to summing node.
                      Typical 10 kΩ–1 MΩ.  Along with a small attenuator
                      (optional) sets the AC modulation amplitude at the
                      sample.  In the limit R_series_ac >> R_sample, the
                      AC EXCITATION is a CURRENT source with
                      I_AC = V_osc / R_series_ac, independent of sample R.
      - r_sample_estimate:  operator's best estimate of sample DC resistance.
                      Used ONLY for safety preflight (estimating the DC
                      voltage that will actually reach the sample) and for
                      the energy-resolution warning.  NOT used in the
                      reported dI/dV values.
      - ac_attenuator:  optional passive divider between SR830 sine-out and
                      r_series_ac; dimensionless ratio ≤ 1.  If you have a
                      real attenuator box (10×, 100×, ...), enter the
                      attenuation FACTOR here as e.g. 0.1 for 10×.  The
                      CSV/sidecar record the effective V_osc delivered
                      after attenuation.
    """
    r_series_dc: float = 1.0e6
    r_series_ac: float = 100.0e3
    r_sample_estimate: float = 1.0e3
    ac_attenuator: float = 1.0


@dataclass
class SR830Config:
    """SR830 setup recorded as metadata AND optionally pushed to the
    instrument at arm time."""
    frequency_hz: float = 13.333        # Hz, AC modulation freq
    v_osc_rms: float = 0.004            # V, sine-out amplitude (SR830
                                        # sine out value; the number in the
                                        # GUI V_osc field).  NOTE: SR830
                                        # amplitude is RMS — not peak.
    sensitivity_index: int = 20         # SCPI index into SR830_SENSITIVITY_VALUES
    time_constant_index: int = 10       # SCPI index into SR830_TC_VALUES
    slope_db_oct: int = 24              # 6/12/18/24
    reserve: str = 'Low Noise'
    sync_filter: bool = True
    input_source: str = 'A'
    input_coupling: str = 'AC'
    input_ground: str = 'Float'
    auto_configure: bool = True         # push these values to SR830 at arm time

    # Computed properties
    @property
    def sensitivity_v(self) -> float:
        return SR830_SENSITIVITY_VALUES[self.sensitivity_index]

    @property
    def time_constant_s(self) -> float:
        return SR830_TC_VALUES[self.time_constant_index]

    def validate(self):
        assert 0 <= self.sensitivity_index < len(SR830_SENSITIVITY_VALUES), \
            f"sensitivity_index {self.sensitivity_index} out of range"
        assert 0 <= self.time_constant_index < len(SR830_TC_VALUES), \
            f"time_constant_index {self.time_constant_index} out of range"
        assert self.slope_db_oct in SR830_SLOPE_DB_VALUES
        assert self.reserve in SR830_RESERVE_VALUES
        assert self.input_coupling in SR830_COUPLING_VALUES
        assert self.input_ground in SR830_GROUND_VALUES
        assert self.input_source in SR830_SOURCE_VALUES
        # SR830 sine-out range 0.004 – 5.0 V rms
        assert 0.004 <= self.v_osc_rms <= 5.0, \
            f"v_osc_rms {self.v_osc_rms} out of SR830 range [0.004, 5.0] V"
        # SR830 frequency range 0.001 – 102400 Hz
        assert 0.001 <= self.frequency_hz <= 102400.0


@dataclass
class BiasConfig:
    """K2636B SMU bias channel settings."""
    mode: str = BIAS_MODE_VOLTAGE        # 'voltage' or 'current'
    channel: str = K2636_DEFAULT_BIAS_CHANNEL
    compliance_a: float = SMU_DEFAULT_BIAS_COMPLIANCE_A
    voltage_range_v: float = 21.0        # source rangev for K2636B

    def validate(self):
        assert self.mode in BIAS_MODE_OPTIONS
        assert self.channel in K2636_CHANNEL_OPTIONS
        assert self.compliance_a > 0


@dataclass
class GateConfig:
    """K2636B SMU gate channel settings.  Optional; gate can be disabled."""
    enabled: bool = False
    channel: str = K2636_DEFAULT_GATE_CHANNEL
    compliance_a: float = SMU_DEFAULT_GATE_COMPLIANCE_A
    voltage_range_v: float = 21.0
    v_fixed: float = 0.0                 # V, fixed gate during 1D V_DC sweep
    v_outer_min: float = -3.0            # V, outer-axis min (if outer_axis='gate')
    v_outer_max: float = 3.0             # V, outer-axis max
    num_outer: int = 11                  # outer-axis points

    def validate(self):
        assert self.channel in K2636_CHANNEL_OPTIONS
        assert self.compliance_a > 0


@dataclass
class SweepConfig:
    """DC bias sweep parameters (the FAST axis)."""
    # Sweep range
    v_dc_min: float = -0.01              # V, sweep lower bound (cmd at source)
    v_dc_max: float = 0.01               # V, sweep upper bound
    num_points: int = 101                # number of points (inclusive)
    bidirectional: bool = True           # sweep fwd then bwd
    # Timing
    t_settle_point: float = 0.2          # s, dwell after each V_DC step,
                                         # BEFORE averaging window begins
                                         # (must exceed ~5× SR830 TC)
    n_avg: int = DAQ_DEFAULT_NAVG        # DAQ samples averaged per point
    # Outer axis
    outer_axis: str = 'none'             # 'none' | 'gate'
    # Voltage safety
    v_dc_limit_cmd_v: float = 5.0        # refuse sweep if |V_DC_cmd|_max > this
    v_dc_limit_sample_v: float = 0.05    # warn if estimated |V_DC_sample|_max > this
                                         # (given r_sample_estimate)

    def validate(self):
        assert self.outer_axis in ('none', 'gate')
        assert self.num_points >= 2
        assert self.v_dc_max > self.v_dc_min
        assert self.t_settle_point > 0
        assert self.n_avg >= 1
        assert self.v_dc_limit_cmd_v > 0

    def fast_array(self) -> np.ndarray:
        return np.linspace(self.v_dc_min, self.v_dc_max, self.num_points)


@dataclass
class ChannelConfig:
    """Per-DAQ-AI channel setup.  v1.0 kinds:

        'dIdV_X'    — in-phase component of dI/dV;  AI reads SR830 CH1
                      (displayed as X).  Report dI/dV in Siemens using
                      the full voltage-bias conversion chain.
        'dIdV_Y'    — quadrature; AI reads CH2 (displayed as Y).
        'dIdV_R'    — magnitude; AI reads CH1 (displayed as R).
        'theta'     — phase in degrees; AI reads CH2 (displayed as θ).
        'Voltage'   — generic AI voltage passthrough (no lock-in conversion).
        'disabled'  — skipped; not written to CSV.
    """
    ai_index: int
    name: str
    enabled: bool = True
    kind: str = 'disabled'
    sens: float = 0.0     # SR830 full-scale rms V for this channel's display
                          # (only used for kinds using sens, i.e. dIdV_X/Y/R)
    gain: float = 1.0     # voltage preamp gain g between sample and SR830 A input
                          # (usually the SR560 voltage gain)

    def validate(self):
        assert self.kind in (
            'disabled', 'dIdV_X', 'dIdV_Y', 'dIdV_R', 'theta', 'Voltage')
        if self.kind in ('dIdV_X', 'dIdV_Y', 'dIdV_R'):
            assert self.sens > 0, f"AI{self.ai_index} ({self.name}): sens must be > 0"
            assert self.gain > 0, f"AI{self.ai_index} ({self.name}): gain must be > 0"


@dataclass
class PointInfo:
    """Emitted to GUI for live plotting."""
    outer_index: int
    fast_index: int
    direction: str           # 'fwd' or 'bwd'
    outer_value: float       # V_gate (or NaN if outer_axis='none')
    v_dc_cmd: float          # V, commanded at K2636B
    v_dc_meas: float         # V, read back from K2636B
    i_dc_meas: float         # A, read back from K2636B
    v_dc_sample_est: float   # V, estimated at sample (via resistor divider)
    didv_values: Dict[str, float]   # kind → value (one per enabled channel)
    didv_errors: Dict[str, float]   # same keys; std of n_avg samples
    i_gate_compl: float      # A, read back on gate channel (NaN if disabled)


@dataclass
class DIDVConfig:
    """Master config object passed to MeasurementThread."""
    # ----- Provenance -----
    sample: str = ''
    device: str = ''
    operator: str = ''
    run_name: str = ''
    sample_T_k: float = 0.040   # K, sample temperature for energy-resolution check
    # ----- Instrument settings -----
    k2636b_visa_address: str = 'GPIB0::26::INSTR'
    bias: BiasConfig = field(default_factory=BiasConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    sr830_visa_address: str = 'GPIB0::8::INSTR'
    sr830: SR830Config = field(default_factory=SR830Config)
    resistors: ResistorNetwork = field(default_factory=ResistorNetwork)
    # ----- DAQ -----
    daq_device_name: str = 'Dev1'       # NI-MAX device name for the AI board
    daq_ai_rate_hz:  float = DAQ_AI_RATE_HZ
    # ----- Sweep -----
    sweep: SweepConfig = field(default_factory=SweepConfig)
    # ----- Output -----
    save_path: str = ''

    def validate(self):
        self.bias.validate()
        self.gate.validate()
        self.sr830.validate()
        self.sweep.validate()
        if self.gate.enabled and self.bias.channel == self.gate.channel:
            raise ValueError(
                f"K2636B bias channel ({self.bias.channel}) must differ from "
                f"gate channel ({self.gate.channel}) when gate is enabled.")
        if self.gate.enabled and self.sweep.outer_axis == 'gate':
            if self.gate.num_outer < 2:
                raise ValueError("Outer-axis gate sweep requires num_outer >= 2.")
            if self.gate.v_outer_max <= self.gate.v_outer_min:
                raise ValueError("Outer-axis gate: v_outer_max must exceed v_outer_min.")
        elif not self.gate.enabled and self.sweep.outer_axis == 'gate':
            raise ValueError(
                "Outer axis is 'gate' but gate is not enabled.  Enable gate "
                "first or set outer_axis='none'.")
        # Resistor network: positivity + sanity
        r = self.resistors
        if not (r.r_series_dc > 0 and r.r_series_ac > 0
                and r.r_sample_estimate > 0):
            raise ValueError(
                f"Resistor network values must be > 0 "
                f"(R_dc={r.r_series_dc}, R_ac={r.r_series_ac}, "
                f"R_sample={r.r_sample_estimate})")
        if not (0 < r.ac_attenuator <= 1.0):
            raise ValueError(
                f"ac_attenuator must be in (0, 1], got {r.ac_attenuator}")
        # DAQ AI rate: reasonable range (100 Hz — 100 kHz)
        if not (100.0 <= self.daq_ai_rate_hz <= 100_000.0):
            raise ValueError(
                f"daq_ai_rate_hz ({self.daq_ai_rate_hz}) out of reasonable "
                f"range [100, 100000] Hz")
        if not self.daq_device_name.strip():
            raise ValueError("daq_device_name is empty (e.g. 'Dev1')")

    def estimate_v_dc_sample_max(self) -> float:
        """Estimate |V_DC_sample|_max given R_sample_est and R_series_dc.
        Used for the safety preflight warning."""
        r = self.resistors.r_sample_estimate
        R = self.resistors.r_series_dc
        v_max_cmd = max(abs(self.sweep.v_dc_min), abs(self.sweep.v_dc_max))
        return v_max_cmd * r / (R + r) if (R + r) > 0 else v_max_cmd

    def estimate_v_ac_sample(self) -> float:
        """Estimate V_AC_sample given R_sample_est, R_series_ac, V_osc, ac_attenuator."""
        r = self.resistors.r_sample_estimate
        R = self.resistors.r_series_ac
        v_osc = self.sr830.v_osc_rms * self.resistors.ac_attenuator
        return v_osc * r / (R + r) if (R + r) > 0 else v_osc

    def energy_resolution_ok(self) -> Tuple[bool, float, float]:
        """Returns (ok, V_AC_sample, kT_over_e). ok is True iff
        V_AC_sample_peak < kT/e (the thermal energy resolution).
        Uses peak = V_AC_rms × sqrt(2)."""
        v_ac_rms = self.estimate_v_ac_sample()
        v_ac_peak = v_ac_rms * np.sqrt(2.0)
        kT_over_e = K_B_EV_PER_K * self.sample_T_k  # V
        return (v_ac_peak < kT_over_e), v_ac_rms, kT_over_e


# =====================================================================
# 1. DAQ hardware  (AI readout only)
# =====================================================================
class DAQHardware:
    """Thin wrapper around nidaqmx for the 8 AI channels.  Voltage-mode
    finite-acquisition reads of n_avg samples at DAQ_AI_RATE_HZ, then
    per-channel mean + std.  Same pattern as prior programs.
    """

    def __init__(self, device_name: str = 'Dev1',
                 ai_rate_hz: float = DAQ_AI_RATE_HZ,
                 n_avg: int = DAQ_DEFAULT_NAVG):
        self.device_name = device_name
        self.ai_chans = [f'{device_name}/ai{i}' for i in range(NUM_AI)]
        self.ai_sample_rate = float(ai_rate_hz)
        self.ai_terminal_name = DAQ_AI_TERMINAL
        self.n_avg = int(n_avg)
        self._task = None
        self.connected = False

    def set_n_avg(self, n: int):
        n = max(1, int(n))
        if n == self.n_avg:
            return
        self.n_avg = n
        if self.connected and not DEMO_MODE:
            self._reconfigure_timing()

    def _reconfigure_timing(self):
        try:
            self._task.timing.cfg_samp_clk_timing(
                rate=self.ai_sample_rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=self.n_avg)
        except Exception as e:
            raise RuntimeError(f"DAQ timing reconfigure failed: {e}")

    def open(self):
        if DEMO_MODE:
            self.connected = True
            return
        if nidaqmx is None:
            raise RuntimeError("nidaqmx not imported; run with --demo or install it.")
        self._task = nidaqmx.Task()
        term = getattr(TerminalConfiguration, DAQ_AI_TERMINAL, None)
        if term is None:
            # nidaqmx uses TerminalConfiguration.RSE (not 'RSE' string)
            term = TerminalConfiguration.RSE
        for chan in self.ai_chans:
            self._task.ai_channels.add_ai_voltage_chan(
                chan, terminal_config=term, min_val=-10.0, max_val=10.0)
        self._task.timing.cfg_samp_clk_timing(
            rate=self.ai_sample_rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=self.n_avg)
        self.connected = True

    def close(self):
        if DEMO_MODE:
            self.connected = False
            return
        try:
            if self._task is not None:
                self._task.close()
        except Exception:
            pass
        self._task = None
        self.connected = False

    def read_ai(self,
                demo_provider: Optional[Callable[[], Tuple[float, ...]]] = None
                ) -> Tuple[List[float], List[float]]:
        """Return (means, stds) for the NUM_AI channels.
        Means are per-channel arithmetic mean of n_avg DAQ samples;
        stds are the sample standard deviation.

        demo_provider: optional function that returns an 8-tuple of
        "true" values at the current instant.  In DEMO_MODE, white
        noise is added around these to simulate n_avg samples."""
        if DEMO_MODE:
            return self._demo_read(demo_provider)
        if not self.connected or self._task is None:
            raise RuntimeError("DAQ not connected.")
        self._task.start()
        try:
            data = self._task.read(number_of_samples_per_channel=self.n_avg)
        finally:
            try:
                self._task.stop()
            except Exception:
                pass
        arr = np.asarray(data)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        means = [float(np.mean(row)) for row in arr]
        stds = [float(np.std(row, ddof=1)) if arr.shape[1] > 1 else 0.0
                for row in arr]
        # pad if fewer than NUM_AI (shouldn't happen)
        while len(means) < NUM_AI:
            means.append(0.0); stds.append(0.0)
        return means[:NUM_AI], stds[:NUM_AI]

    def _demo_read(self, demo_provider):
        if demo_provider is not None:
            try:
                true_vals = list(demo_provider())
            except Exception:
                true_vals = [0.0] * NUM_AI
        else:
            true_vals = [0.0] * NUM_AI
        while len(true_vals) < NUM_AI:
            true_vals.append(0.0)
        # Add n_avg synthetic samples with white noise
        noise = 0.002  # 2 mV rms noise per sample
        means, stds = [], []
        for v in true_vals[:NUM_AI]:
            samples = v + noise * np.random.randn(self.n_avg)
            means.append(float(np.mean(samples)))
            stds.append(float(np.std(samples, ddof=1))
                        if self.n_avg > 1 else 0.0)
        return means, stds


# =====================================================================
# 2. K2636B dual-channel  (single VISA session, shared by bias + gate)
# =====================================================================
class ComplianceTrippedError(RuntimeError):
    def __init__(self, label, voltage, current, compliance):
        self.label      = label
        self.voltage    = voltage
        self.current    = current
        self.compliance = compliance
        super().__init__(
            f"Compliance tripped on {label} at V={voltage:+.4f} V, "
            f"I={current:+.3e} A (limit {compliance:.3e} A)")


class K2636B_DualChannel:
    """Owns the single VISA session to the K2636B.  Both bias and gate
    borrow this session and issue per-channel TSP commands.
    """

    def __init__(self, address: str, voltage_range_v: float = 21.0):
        self.address = address
        self.voltage_range_v = float(voltage_range_v)
        self._rm = None
        self._inst = None
        self._idn = ''
        self.connected = False
        self._armed = {'smua': False, 'smub': False}

    def open(self):
        if DEMO_MODE:
            self._idn = f'DEMO Keithley 2636B @ {self.address}'
            self.connected = True
            return
        if pyvisa is None:
            raise RuntimeError("pyvisa not available.")
        self._rm = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(self.address)
        self._inst.timeout = 5000
        self._inst.read_termination  = '\n'
        self._inst.write_termination = '\n'
        self._idn = self._inst.query('*IDN?').strip()
        if '2636' not in self._idn.upper():
            sys.stderr.write(
                f"WARNING: device at {self.address} returned {self._idn!r} "
                f"which does not match '2636'.  Proceeding anyway.\n")
        self.connected = True

    def close(self):
        try:
            if self.connected and not DEMO_MODE:
                for ch in ('smua', 'smub'):
                    if self._armed.get(ch, False):
                        try:
                            self._inst.write(f'{ch}.source.output = {ch}.OUTPUT_OFF')
                        except Exception:
                            pass
                        self._armed[ch] = False
        except Exception:
            pass
        try:
            if self._inst is not None:
                self._inst.close()
        except Exception:
            pass
        try:
            if self._rm is not None:
                self._rm.close()
        except Exception:
            pass
        self._inst = None
        self._rm   = None
        self.connected = False

    def write(self, cmd: str):
        if DEMO_MODE:
            return
        self._inst.write(cmd)

    def query(self, cmd: str) -> str:
        if DEMO_MODE:
            return '0'
        return self._inst.query(cmd)


class K2636B_BiasChannel:
    """DC bias source, K2636B mode-switchable between V-source and I-source.
    Always sourced + sensed simultaneously; write_bias returns the measured
    pair (V_meas, I_meas)."""

    def __init__(self, parent: K2636B_DualChannel, channel: str,
                 mode: str, compliance_a: float):
        assert channel in K2636_CHANNEL_OPTIONS
        assert mode in BIAS_MODE_OPTIONS
        self.parent = parent
        self.channel = channel
        self.mode = mode
        self.compliance_a = float(compliance_a)
        self._last_cmd = 0.0
        self._last_v = 0.0
        self._last_i = 0.0
        self._armed = False

    @property
    def label(self) -> str:
        return f"bias_{self.channel}_{self.mode}"

    def arm_output(self):
        if self._armed:
            return
        if DEMO_MODE:
            self._armed = True
            self.parent._armed[self.channel] = True
            return
        ch = self.channel
        p = self.parent
        p.write(f'{ch}.reset()')
        if self.mode == BIAS_MODE_VOLTAGE:
            p.write(f'{ch}.source.func = {ch}.OUTPUT_DCVOLTS')
            p.write(f'{ch}.source.autorangev = {ch}.AUTORANGE_OFF')
            p.write(f'{ch}.source.rangev = {p.voltage_range_v:.3f}')
            p.write(f'{ch}.source.levelv = 0')
            p.write(f'{ch}.source.limiti = {self.compliance_a:.6e}')
            p.write(f'{ch}.measure.autorangei = {ch}.AUTORANGE_ON')
        else:  # BIAS_MODE_CURRENT
            p.write(f'{ch}.source.func = {ch}.OUTPUT_DCAMPS')
            p.write(f'{ch}.source.autorangei = {ch}.AUTORANGE_ON')
            p.write(f'{ch}.source.leveli = 0')
            p.write(f'{ch}.source.limitv = {p.voltage_range_v:.3f}')
            p.write(f'{ch}.measure.autorangev = {ch}.AUTORANGE_ON')
        p.write(f'{ch}.source.output = {ch}.OUTPUT_ON')
        self._armed = True
        self.parent._armed[self.channel] = True

    def disarm_output(self):
        if DEMO_MODE:
            self._armed = False
            self.parent._armed[self.channel] = False
            return
        try:
            self.parent.write(
                f'{self.channel}.source.output = {self.channel}.OUTPUT_OFF')
        except Exception:
            pass
        self._armed = False
        self.parent._armed[self.channel] = False

    def write_bias(self, cmd: float) -> Tuple[float, float]:
        """Set source level to cmd (V or A depending on mode), then read
        back (V_meas, I_meas) from the SMU.  Raises ComplianceTrippedError
        if the NON-SOURCED variable exceeds compliance × abort_fraction."""
        cmd = float(np.clip(cmd, -self.parent.voltage_range_v,
                                  self.parent.voltage_range_v))
        if DEMO_MODE:
            v, i = self._demo_step(cmd)
        else:
            ch = self.channel
            if self.mode == BIAS_MODE_VOLTAGE:
                self.parent.write(f'{ch}.source.levelv = {cmd}')
                resp = self.parent.query(f'print({ch}.measure.iv())')
                # K2636B iv() returns "i, v" on some firmwares, "v i" on others.
                parts = resp.replace(',', ' ').split()
                # defensive: if exactly 2 numbers, interpret as i, v
                if len(parts) >= 2:
                    i = float(parts[0]); v = float(parts[1])
                else:
                    v = float(self.parent.query(
                        f'print({ch}.measure.v())'))
                    i = float(self.parent.query(
                        f'print({ch}.measure.i())'))
            else:
                self.parent.write(f'{ch}.source.leveli = {cmd}')
                resp = self.parent.query(f'print({ch}.measure.iv())')
                parts = resp.replace(',', ' ').split()
                if len(parts) >= 2:
                    i = float(parts[0]); v = float(parts[1])
                else:
                    v = float(self.parent.query(
                        f'print({ch}.measure.v())'))
                    i = float(self.parent.query(
                        f'print({ch}.measure.i())'))
        self._last_cmd = cmd
        self._last_v = v
        self._last_i = i
        # Compliance check on the NON-sourced variable:
        if self.mode == BIAS_MODE_VOLTAGE:
            # compliance is on current
            thresh = self.compliance_a * SMU_COMPLIANCE_ABORT_FRACTION
            if abs(i) > thresh:
                raise ComplianceTrippedError(self.label, cmd, i, self.compliance_a)
        else:
            # compliance (in volts) is self.parent.voltage_range_v, less stringent.
            # Here we treat compliance_a as the COMPLIANCE FOR CURRENT mode's
            # output limit; instead of current we monitor the compliance voltage.
            # For publication use we simply warn if |v| is near voltage_range.
            if abs(v) > self.parent.voltage_range_v * SMU_COMPLIANCE_ABORT_FRACTION:
                raise ComplianceTrippedError(self.label, cmd, v,
                                             self.parent.voltage_range_v)
        return v, i

    def _demo_step(self, cmd):
        """Synthetic response for DEMO_MODE: pretend the sample is a
        BCS-like superconductor with a gap at ±0.25 mV, normal resistance
        R_N ≈ 2 kΩ (chosen so that at ±1 mV sweep the current stays
        below the default 1 µA compliance).  Adds small gate-dependent
        scaling if gate V is set."""
        if self.mode == BIAS_MODE_VOLTAGE:
            V = cmd
            # BCS-like tunneling: very small I inside |V| < Δ/e, larger outside.
            gap = 0.25e-3  # V
            Rn = 2000.0    # Ω  (chosen for compatibility with typical
                           # compliance settings in the 1 µA range)
            if abs(V) < gap:
                I = V / Rn * 0.02   # strongly suppressed inside gap
            else:
                excess = np.sqrt(max(V**2 - gap**2, 0))
                I = np.sign(V) * excess / Rn
            I += 5e-12 * np.random.randn()  # 5 pA noise
            return V, I
        else:
            I = cmd
            # Josephson-like: zero voltage under Ic = 500 nA, then linear
            Ic = 500e-9
            Rn = 20.0
            if abs(I) < Ic:
                V = 0.0 + 1e-8 * np.random.randn()
            else:
                V = np.sign(I) * (abs(I) - Ic) * Rn + 1e-8 * np.random.randn()
            return V, I

    def ramp_to(self, target: float,
                rate_per_s: Optional[float] = None,
                step: Optional[float] = None,
                stop_flag: Optional[Callable[[], bool]] = None):
        """Linear ramp of the source level from current value to target.

        In voltage mode, target/rate/step are in V and V/s.
        In current mode, target/rate/step are in A and A/s.

        If rate_per_s / step are None, sensible mode-specific defaults
        are used:
          V-mode: rate = SMU_RAMP_RATE_V_PER_S, step = SMU_RAMP_STEP_V
          I-mode: rate = 1 µA/s, step = 10 nA  (conservative, adequate
                  for ramp-to-zero on shutdown; for active ramps in
                  I-mode, callers should pass explicit values).
        """
        if rate_per_s is None:
            rate_per_s = (SMU_RAMP_RATE_V_PER_S
                          if self.mode == BIAS_MODE_VOLTAGE else 1.0e-6)
        if step is None:
            step = (SMU_RAMP_STEP_V
                    if self.mode == BIAS_MODE_VOLTAGE else 1.0e-8)
        if abs(target - self._last_cmd) < 1e-12:
            return
        n = max(2, int(np.ceil(abs(target - self._last_cmd) / step)))
        dwell = abs(target - self._last_cmd) / max(rate_per_s, 1e-12) / n
        for v in np.linspace(self._last_cmd, target, n):
            if stop_flag is not None and stop_flag():
                return
            self.write_bias(v)
            time.sleep(dwell)

    def get_last(self):
        return self._last_cmd, self._last_v, self._last_i


class K2636B_GateChannel:
    """Voltage-source gate driver, very similar to v2.5-2636B's
    K2636B_GateChannel.  Always V-source / I-sense."""

    def __init__(self, parent: K2636B_DualChannel, channel: str,
                 compliance_a: float, label: str = 'gate'):
        assert channel in K2636_CHANNEL_OPTIONS
        self.parent = parent
        self.channel = channel
        self.compliance_a = float(compliance_a)
        self.label = label
        self._last_v = 0.0
        self._last_i = 0.0
        self._armed = False

    def arm_output(self):
        if self._armed:
            return
        if DEMO_MODE:
            self._armed = True
            self.parent._armed[self.channel] = True
            return
        ch = self.channel
        p = self.parent
        p.write(f'{ch}.reset()')
        p.write(f'{ch}.source.func = {ch}.OUTPUT_DCVOLTS')
        p.write(f'{ch}.source.autorangev = {ch}.AUTORANGE_OFF')
        p.write(f'{ch}.source.rangev = {p.voltage_range_v:.3f}')
        p.write(f'{ch}.source.levelv = 0')
        p.write(f'{ch}.source.limiti = {self.compliance_a:.6e}')
        p.write(f'{ch}.measure.autorangei = {ch}.AUTORANGE_ON')
        p.write(f'{ch}.source.output = {ch}.OUTPUT_ON')
        self._armed = True
        self.parent._armed[ch] = True

    def disarm_output(self):
        if DEMO_MODE:
            self._armed = False
            self.parent._armed[self.channel] = False
            return
        try:
            self.parent.write(
                f'{self.channel}.source.output = {self.channel}.OUTPUT_OFF')
        except Exception:
            pass
        self._armed = False
        self.parent._armed[self.channel] = False

    def write_voltage(self, v: float) -> float:
        v = float(np.clip(v, -self.parent.voltage_range_v,
                              self.parent.voltage_range_v))
        if DEMO_MODE:
            i = 1e-12 * v + 0.3e-12 * np.random.randn()
        else:
            ch = self.channel
            self.parent.write(f'{ch}.source.levelv = {v}')
            i = float(self.parent.query(f'print({ch}.measure.i())'))
        self._last_v = v
        self._last_i = i
        thresh = self.compliance_a * SMU_COMPLIANCE_ABORT_FRACTION
        if abs(i) > thresh:
            raise ComplianceTrippedError(self.label, v, i, self.compliance_a)
        return i

    def ramp_to(self, target: float,
                rate_v_per_s: float = SMU_RAMP_RATE_V_PER_S,
                step: float = SMU_RAMP_STEP_V,
                stop_flag: Optional[Callable[[], bool]] = None):
        if abs(target - self._last_v) < 1e-6:
            return
        n = max(2, int(np.ceil(abs(target - self._last_v) / step)))
        dwell = abs(target - self._last_v) / max(rate_v_per_s, 1e-3) / n
        for v in np.linspace(self._last_v, target, n):
            if stop_flag is not None and stop_flag():
                return
            self.write_voltage(v)
            time.sleep(dwell)

    def get_last(self):
        return self._last_v, self._last_i


# =====================================================================
# 3. SR830 lock-in controller
# =====================================================================
class SR830Controller:
    """Controls the SR830 via VISA (GPIB or USB-GPIB).  Can push a full
    setup (frequency, amplitude, sensitivity, TC, slope, etc.), and can
    read X/Y/R/θ over GPIB.  In the fast-acquisition path we DO NOT use
    GPIB reads — instead we route CH1/CH2 BNC outputs into the NI DAQ
    and convert voltages using the known SR830 sensitivity.  GPIB reads
    are used only for preflight and status queries."""

    def __init__(self, address: str, config: SR830Config):
        self.address = address
        self.config = config
        self._rm = None
        self._inst = None
        self._idn = ''
        self.connected = False

    def open(self):
        if DEMO_MODE:
            self._idn = f'DEMO SR830 @ {self.address}'
            self.connected = True
            return
        if pyvisa is None:
            raise RuntimeError("pyvisa not available.")
        self._rm = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(self.address)
        self._inst.timeout = 5000
        self._inst.read_termination  = '\n'
        self._inst.write_termination = '\n'
        self._idn = self._inst.query('*IDN?').strip()
        if 'SR830' not in self._idn.upper() and 'SR810' not in self._idn.upper():
            sys.stderr.write(
                f"WARNING: device at {self.address} returned {self._idn!r}; "
                f"not a recognised SR830/SR810.  Proceeding anyway.\n")
        self.connected = True

    def close(self):
        try:
            if self._inst is not None:
                self._inst.close()
        except Exception:
            pass
        try:
            if self._rm is not None:
                self._rm.close()
        except Exception:
            pass
        self._inst = None
        self._rm = None
        self.connected = False

    def push_config(self):
        """Push self.config to the SR830 (if auto_configure is True).

        CRITICAL for publication-grade correctness: this also RESETS the
        HARM (harmonic) setting to 1 and all OEXP (offset/expand) channels
        to (0%, 1×).  Without these resets, if a previous user had left
        the SR830 in 2nd-harmonic detection mode (for d²I/dV²) or with
        X-expand=10×, the dI/dV values coming out of CH1/CH2 BNC would
        be silently off by factors of 10 or more.  These are invisible
        on the SR830 front panel unless you look specifically at the
        HARM and OEXP indicators, and are a classic publication trap.

        After auto_configure this method sleeps for 5× TC to let the
        new TC filter reach steady-state before any measurement.
        """
        if DEMO_MODE:
            return
        if not self.config.auto_configure:
            return
        c = self.config
        inst = self._inst
        # Reference: FMOD 0 = external, FMOD 1 = internal
        inst.write('FMOD 1')                          # internal reference
        inst.write('HARM 1')                          # fundamental (NOT 2nd/3rd harmonic)
        inst.write(f'FREQ {c.frequency_hz:.6f}')
        inst.write(f'SLVL {c.v_osc_rms:.4f}')         # sine-out amplitude in V_rms
        inst.write(f'SENS {c.sensitivity_index}')
        inst.write(f'OFLT {c.time_constant_index}')
        inst.write(f'OFSL {SR830_SLOPE_DB_VALUES.index(c.slope_db_oct)}')
        # Reserve: SR830 RMOD 0 = high reserve, 1 = normal, 2 = low noise
        rmod_map = {'High Reserve': 0, 'Normal': 1, 'Low Noise': 2}
        inst.write(f'RMOD {rmod_map[c.reserve]}')
        # Sync filter (useful below 200 Hz)
        inst.write(f'SYNC {1 if c.sync_filter else 0}')
        # Input source ISRC: 0=A, 1=A-B, 2=I(1MΩ), 3=I(100MΩ)
        isrc_map = {'A': 0, 'A-B': 1, 'I(1MOhm)': 2, 'I(100MOhm)': 3}
        inst.write(f'ISRC {isrc_map[c.input_source]}')
        # Coupling ICPL: 0=AC, 1=DC
        inst.write(f'ICPL {0 if c.input_coupling == "AC" else 1}')
        # Ground IGND: 0=Float, 1=Ground
        inst.write(f'IGND {0 if c.input_ground == "Float" else 1}')
        # Reset expand + offset for X (1), Y (2), R (3).  OEXP i, x_percent, expand_idx
        # where expand_idx: 0=1×, 1=10×, 2=100×.  We force all to 0,0 so
        # CH1/CH2 BNC output scales as standard X/Y/R with full-scale = 10 V.
        for chan_idx in (1, 2, 3):
            inst.write(f'OEXP {chan_idx}, 0, 0')
        # Phase offset reference: don't touch (user may want a pre-nulled phase).
        # Let the new TC settle before anyone reads: 5 × TC, capped at 2 s.
        settle = min(5.0 * c.time_constant_s, 2.0)
        time.sleep(settle)

    def check_overload(self) -> Tuple[bool, bool]:
        """Query the LIA status byte.  Returns (input_overload, time_constant_overload)."""
        if DEMO_MODE:
            return False, False
        try:
            b = int(self._inst.query('LIAS?').strip())
        except Exception:
            return False, False
        # LIAS bit 0 = input/reserve overload; bit 2 = output overload (TC)
        return bool(b & 0x01), bool(b & 0x04)

    def read_xy(self) -> Tuple[float, float]:
        """GPIB read of X, Y directly.  Use only for preflight; fast-path
        uses DAQ AI of CH1/CH2 BNC."""
        if DEMO_MODE:
            return 0.0, 0.0
        s = self._inst.query('SNAP? 1,2').strip()
        a, b = s.split(',')[:2]
        return float(a), float(b)

    def read_rtheta(self) -> Tuple[float, float]:
        if DEMO_MODE:
            return 0.0, 0.0
        s = self._inst.query('SNAP? 3,4').strip()
        a, b = s.split(',')[:2]
        return float(a), float(b)


# =====================================================================
# 4. Composite hardware — holds DAQ + K2636B + SR830, enforces lifecycle
# =====================================================================
class DIDVCompositeHardware:
    """Bundles all instruments for a dI/dV scan.  Owns the bias SMU
    channel, the optional gate SMU channel, and the SR830 controller.
    Single VISA session to K2636B; separate VISA session to SR830.
    """

    def __init__(self,
                 k2636b: K2636B_DualChannel,
                 bias: BiasConfig,
                 gate: GateConfig,
                 sr830: SR830Controller,
                 daq: DAQHardware):
        self.k2636b = k2636b
        self.bias_cfg = bias
        self.gate_cfg = gate
        self.sr830 = sr830
        self.daq = daq
        self.bias_smu = K2636B_BiasChannel(
            parent=k2636b, channel=bias.channel,
            mode=bias.mode, compliance_a=bias.compliance_a)
        if gate.enabled:
            if gate.channel == bias.channel:
                raise ValueError(
                    f"Gate channel ({gate.channel}) must differ from bias "
                    f"channel ({bias.channel}).")
            self.gate_smu = K2636B_GateChannel(
                parent=k2636b, channel=gate.channel,
                compliance_a=gate.compliance_a, label='gate')
        else:
            self.gate_smu = None
        self.connected = False
        self._last_i_gate = float('nan')

    def open(self):
        """Open all instruments in the safe order:
          1. Open K2636B VISA session  (no SMU outputs yet)
          2. Open SR830 VISA + push config  (so V_osc, sens, HARM, OEXP, ...
             are known BEFORE SMU arms.  If the SR830 was left at a large
             V_osc from a previous user, push_config immediately brings
             it down to the user-set value.)
          3. Arm SMU bias channel (and gate channel if enabled) — outputs ON
             at 0 V.  Sample now sees the known-small V_osc from SR830 plus
             0 V DC.
          4. Open DAQ.
        This order is critical for protecting fragile samples (JJs, hBN-
        capped devices) against incidental AC amplitudes left over from
        earlier lock-in sessions.
        """
        opened = []
        try:
            self.k2636b.open(); opened.append(self.k2636b)
            self.sr830.open(); opened.append(self.sr830)
            self.sr830.push_config()
            self.bias_smu.arm_output()
            if self.gate_smu is not None:
                self.gate_smu.arm_output()
            self.daq.open(); opened.append(self.daq)
        except Exception:
            # Undo on failure: disarm SMU channels first, then close all VISA
            try:
                self.bias_smu.disarm_output()
            except Exception:
                pass
            if self.gate_smu is not None:
                try:
                    self.gate_smu.disarm_output()
                except Exception:
                    pass
            for h in reversed(opened):
                try: h.close()
                except Exception: pass
            raise
        self.connected = True

    def close(self, stop_flag: Optional[Callable[[], bool]] = None):
        # Tear down in reverse: DAQ, then ramp SMUs to 0, disarm, close.
        try: self.daq.close()
        except Exception: pass
        # Ramp bias to 0
        try:
            self.bias_smu.ramp_to(0.0, stop_flag=stop_flag)
        except Exception:
            pass
        try:
            self.bias_smu.disarm_output()
        except Exception:
            pass
        if self.gate_smu is not None:
            try:
                self.gate_smu.ramp_to(0.0, stop_flag=stop_flag)
            except Exception:
                pass
            try:
                self.gate_smu.disarm_output()
            except Exception:
                pass
        try:
            self.k2636b.close()
        except Exception:
            pass
        try:
            self.sr830.close()
        except Exception:
            pass
        self.connected = False

    def set_gate(self, v_gate: float):
        if self.gate_smu is None:
            return
        self.gate_smu.write_voltage(v_gate)
        self._last_i_gate = self.gate_smu._last_i

    def write_bias(self, cmd: float) -> Tuple[float, float]:
        """Set DC bias, return (V_meas, I_meas) from SMU."""
        return self.bias_smu.write_bias(cmd)

    def ramp_bias(self, target: float,
                  stop_flag: Optional[Callable[[], bool]] = None):
        self.bias_smu.ramp_to(target, stop_flag=stop_flag)

    def read_ai(self, demo_provider: Optional[Callable] = None):
        return self.daq.read_ai(demo_provider=demo_provider)

    def get_i_gate(self) -> float:
        return self._last_i_gate


# =====================================================================
# 5. Measurement worker thread
# =====================================================================
class MeasurementThread(QThread):
    """Executes a dI/dV sweep end-to-end.  Outer loop over gate (if
    enabled); inner loop over V_DC.  Forward sweep always; optional
    backward sweep (bidirectional=True).  Each fast point:
        1) write V_DC_cmd to bias SMU  (with compliance check)
        2) sleep t_settle_point          (must exceed ~5× SR830 TC)
        3) read SR830 overload status via GPIB (once per several points)
        4) DAQ finite acquisition of n_avg samples of CH1/CH2
        5) convert to physical units per channel kind
        6) write CSV row + emit PointInfo to GUI
    """

    point_ready      = pyqtSignal(object)         # PointInfo
    log_msg          = pyqtSignal(str, str)       # text, level
    progress         = pyqtSignal(int, int)       # current, total
    finished_ok      = pyqtSignal(str)            # save_path
    error_occurred   = pyqtSignal(str)

    def __init__(self, config: DIDVConfig,
                 channels: List[ChannelConfig], parent=None):
        super().__init__(parent)
        self.cfg = config
        self.channels = channels
        self.enabled_channels = [c for c in channels if c.enabled]
        self._stop = False
        self._composite: Optional[DIDVCompositeHardware] = None
        self._abort_cause: Optional[Dict[str, str]] = None
        self._completed_normally = False
        self._started_at_iso = ''
        self._overload_warned = False

    def stop(self):
        self._stop = True

    def is_stopping(self) -> bool:
        return self._stop

    # =================================================================
    # Public entry point
    # =================================================================
    def run(self):
        try:
            self._run_inner()
        except ComplianceTrippedError as e:
            # If the inner loop already set self._abort_cause with the
            # compliance fields, keep it; otherwise record now.  This
            # catches trips that happen in ramp_bias() between scan
            # blocks or in set_gate(), not just in the main loop.
            if self._abort_cause is None:
                self._abort_cause = {
                    'kind':          'compliance_tripped',
                    'label':         e.label,
                    'voltage_cmd_V': f'{e.voltage:+.4e}',
                    'current_A':     f'{e.current:+.4e}',
                    'compliance_A':  f'{e.compliance:.4e}',
                    'message':       str(e),
                }
            self.log_msg.emit(f"ABORT (compliance): {e}", "error")
        except Exception as e:
            tb = traceback.format_exc()
            self._abort_cause = {
                'kind': 'exception',
                'message': str(e),
                'traceback_last_line': tb.strip().split('\n')[-1],
            }
            self.log_msg.emit(f"FATAL: {e}", "error")
            self.error_occurred.emit(tb)
        finally:
            # Always teardown
            if self._composite is not None:
                try:
                    self._composite.close(stop_flag=self.is_stopping)
                except Exception as e:
                    self.log_msg.emit(
                        f"Teardown error: {e}  "
                        f"(CHECK K2636B FRONT PANEL MANUALLY)",
                        "error")
            # Write sidecars even on abort, so evidence is preserved.
            try:
                if self._started_at_iso:
                    self._write_txt_sidecar()
                    self._write_metadata_sidecar()
            except Exception as e:
                self.log_msg.emit(
                    f"Sidecar write failed: {e}", "warning")
            if self._completed_normally:
                self.finished_ok.emit(self.cfg.save_path)

    # =================================================================
    # Main routine
    # =================================================================
    def _run_inner(self):
        cfg = self.cfg
        cfg.validate()
        for ch in self.channels:
            ch.validate()

        # Preflight: energy resolution
        ok, v_ac_rms, kT = cfg.energy_resolution_ok()
        if not ok:
            self.log_msg.emit(
                f"WARNING: estimated V_AC_sample_peak "
                f"({v_ac_rms*np.sqrt(2)*1e6:.1f} µV_pk) exceeds thermal "
                f"energy kT/e ({kT*1e6:.1f} µV) at T={cfg.sample_T_k}K.  "
                f"dI/dV features below this scale will be broadened.", "warning")
        else:
            self.log_msg.emit(
                f"Energy resolution: V_AC_sample_peak ≈ "
                f"{v_ac_rms*np.sqrt(2)*1e6:.2f} µV_pk < kT/e = "
                f"{kT*1e6:.2f} µV at T={cfg.sample_T_k}K.  OK.", "success")

        # Preflight: t_settle_point vs TC
        tc = cfg.sr830.time_constant_s
        if cfg.sweep.t_settle_point < PREFLIGHT_TC_MINIMUM * tc:
            raise RuntimeError(
                f"t_settle_point ({cfg.sweep.t_settle_point:.3f} s) must "
                f"exceed {PREFLIGHT_TC_MINIMUM} × SR830 TC ({tc:.3f} s).  "
                f"Increase t_settle_point to at least "
                f"{PREFLIGHT_TC_MINIMUM*tc:.3f} s.")
        if cfg.sweep.t_settle_point < PREFLIGHT_TC_MULTIPLIER * tc:
            self.log_msg.emit(
                f"WARNING: t_settle_point ({cfg.sweep.t_settle_point:.3f} s) "
                f"< {PREFLIGHT_TC_MULTIPLIER} × TC ({tc:.3f} s).  Recommended "
                f"dwell is 5× TC for 99% settling.", "warning")

        # Preflight: V_DC_cmd limits
        v_dc_cmd_max = max(abs(cfg.sweep.v_dc_min), abs(cfg.sweep.v_dc_max))
        if v_dc_cmd_max > cfg.sweep.v_dc_limit_cmd_v:
            raise RuntimeError(
                f"|V_DC_cmd|_max ({v_dc_cmd_max:+.3f} V) exceeds safety "
                f"limit ({cfg.sweep.v_dc_limit_cmd_v:+.3f} V).  Reduce "
                f"sweep range or increase the limit if you really need it.")
        v_dc_sample_max = cfg.estimate_v_dc_sample_max()
        if v_dc_sample_max > cfg.sweep.v_dc_limit_sample_v:
            self.log_msg.emit(
                f"WARNING: estimated |V_DC_sample|_max "
                f"({v_dc_sample_max*1e3:+.3f} mV, using r_sample_est="
                f"{cfg.resistors.r_sample_estimate:.1e} Ω) exceeds the "
                f"sample safety limit ({cfg.sweep.v_dc_limit_sample_v*1e3:+.3f} mV).  "
                f"Double-check your resistor network and sample estimate.",
                "warning")

        # Preflight: line-frequency proximity.  If f_mod is within 2 Hz
        # of 50/60/100/120 Hz (or their low harmonics), mains pickup at
        # that frequency will leak through the SR830's 24 dB/oct filter
        # and contaminate the dI/dV.  A classic publication-grade trap —
        # one of the most common sources of "mystery peaks" in lock-in
        # spectra.  We warn rather than refuse, because a carefully-
        # shielded experiment can sometimes tolerate it.
        f_mod = cfg.sr830.frequency_hz
        LINE_FREQS = (50.0, 60.0, 100.0, 120.0, 150.0, 180.0, 200.0, 240.0)
        for f_line in LINE_FREQS:
            if abs(f_mod - f_line) < 2.0:
                self.log_msg.emit(
                    f"WARNING: f_mod = {f_mod:.3f} Hz is within 2 Hz of "
                    f"line-frequency or harmonic {f_line:.0f} Hz.  Mains "
                    f"pickup will contaminate dI/dV.  Pick an f_mod that "
                    f"is at least 2 Hz from every multiple of 50 Hz and "
                    f"60 Hz.  Good choices: 13.333, 17.777, 23.7, 77.77 Hz.",
                    "warning")
                break
        if f_mod < 1.0:
            self.log_msg.emit(
                f"NOTE: f_mod = {f_mod:.3f} Hz is very low.  Even with "
                f"a 24 dB/oct filter you will need at least 5–10 × TC "
                f"≈ {5*tc:.1f}–{10*tc:.1f} s dwell per point, and "
                f"total scan time may be hours.", "info")
        if f_mod > 2000.0:
            self.log_msg.emit(
                f"NOTE: f_mod = {f_mod:.1f} Hz is high; cable capacitance "
                f"in the series-AC resistor path will roll off the AC "
                f"excitation.  Verify V_AC at the sample.", "warning")

        # Construct composite hardware
        k2636b = K2636B_DualChannel(
            address=cfg.k2636b_visa_address,
            voltage_range_v=max(cfg.bias.voltage_range_v,
                                cfg.gate.voltage_range_v, 10.0))
        sr830  = SR830Controller(cfg.sr830_visa_address, cfg.sr830)
        daq    = DAQHardware(device_name=cfg.daq_device_name,
                             ai_rate_hz=cfg.daq_ai_rate_hz,
                             n_avg=cfg.sweep.n_avg)
        composite = DIDVCompositeHardware(
            k2636b=k2636b, bias=cfg.bias, gate=cfg.gate,
            sr830=sr830, daq=daq)
        self._composite = composite

        # Open
        self.log_msg.emit("Opening hardware...", "info")
        composite.open()
        self.log_msg.emit(
            f"  K2636B: {k2636b._idn}\n"
            f"  SR830:  {sr830._idn}\n"
            f"  DAQ:    {daq.device_name} ({NUM_AI} AI, {daq.ai_sample_rate:.0f} Hz, "
            f"n_avg={daq.n_avg})", "success")

        # Arm timestamp + CSV header + progress banner
        self._started_at_iso = datetime.datetime.now(
            datetime.timezone.utc).astimezone().strftime(
            '%Y-%m-%dT%H:%M:%S%z')
        self._overload_warned = False

        self.log_msg.emit(
            f"Bias mode: {cfg.bias.mode.upper()}  ({cfg.bias.channel})\n"
            f"  compliance = {cfg.bias.compliance_a:.3e} A\n"
            f"SR830: f={cfg.sr830.frequency_hz:.3f} Hz, "
            f"V_osc={cfg.sr830.v_osc_rms*1e3:.2f} mV_rms, "
            f"sens={cfg.sr830.sensitivity_v:.2e}, "
            f"TC={cfg.sr830.time_constant_s:.3f} s, "
            f"slope={cfg.sr830.slope_db_oct} dB/oct\n"
            f"Resistors: R_series_dc={cfg.resistors.r_series_dc:.2e} Ω, "
            f"R_series_ac={cfg.resistors.r_series_ac:.2e} Ω, "
            f"r_sample_est={cfg.resistors.r_sample_estimate:.2e} Ω, "
            f"ac_attenuator={cfg.resistors.ac_attenuator:.3f}", "info")

        fast_arr = cfg.sweep.fast_array()
        if cfg.sweep.outer_axis == 'gate':
            outer_arr = np.linspace(
                cfg.gate.v_outer_min, cfg.gate.v_outer_max,
                cfg.gate.num_outer)
        else:
            outer_arr = np.array([float('nan')])   # single-point outer
        n_dirs = 2 if cfg.sweep.bidirectional else 1
        total = len(outer_arr) * len(fast_arr) * n_dirs

        # Open CSV + write header
        os.makedirs(os.path.dirname(cfg.save_path) or '.', exist_ok=True)
        with open(cfg.save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            self._write_csv_header(writer)
            done = 0
            for i_outer, outer_v in enumerate(outer_arr):
                if self._stop: break
                # Set gate (if outer='gate' or even if static gate is requested)
                if composite.gate_smu is not None:
                    if cfg.sweep.outer_axis == 'gate':
                        composite.set_gate(float(outer_v))
                    else:
                        composite.set_gate(cfg.gate.v_fixed)
                    # Allow gate to settle
                    time.sleep(0.2)

                # Ramp bias from its current value to fast_arr[0]
                composite.ramp_bias(float(fast_arr[0]),
                                    stop_flag=self.is_stopping)
                # Let SR830 settle after the big jump
                time.sleep(max(5 * tc, 0.2))

                # ---- Forward sweep ----
                for direction_idx, direction in enumerate(
                        DIRECTIONS[:n_dirs]):
                    if direction == 'bwd':
                        sweep_order = list(range(len(fast_arr) - 1, -1, -1))
                    else:
                        sweep_order = list(range(len(fast_arr)))
                    for pos, i_fast in enumerate(sweep_order):
                        if self._stop:
                            break
                        v_dc_cmd = float(fast_arr[i_fast])
                        try:
                            v_dc_meas, i_dc_meas = composite.write_bias(v_dc_cmd)
                        except ComplianceTrippedError as e:
                            self._abort_cause = {
                                'kind': 'compliance_tripped',
                                'label': e.label,
                                'voltage_cmd_V': f'{e.voltage:+.4e}',
                                'current_A': f'{e.current:+.4e}',
                                'compliance_A': f'{e.compliance:.4e}',
                            }
                            self.log_msg.emit(
                                f"ABORT (compliance): {e}", "error")
                            raise

                        time.sleep(cfg.sweep.t_settle_point)

                        # Once every 16 points, query SR830 overload
                        if (not self._overload_warned) and (pos % 16 == 0):
                            ov_in, ov_tc = composite.sr830.check_overload()
                            if ov_in or ov_tc:
                                self.log_msg.emit(
                                    f"SR830 overload: input={ov_in}, "
                                    f"TC={ov_tc}. Consider lowering sens "
                                    f"or V_osc.", "warning")
                                self._overload_warned = True

                        # Fast DAQ read
                        means, stds = composite.read_ai(
                            demo_provider=lambda vc=v_dc_cmd, vg=(
                                outer_v if cfg.sweep.outer_axis=='gate'
                                else cfg.gate.v_fixed
                            ): self._demo_ai_provider(vc, vg))

                        # Convert to physical units per channel kind
                        didv_values, didv_errors = self._convert(
                            means, stds, cfg)

                        # Estimate sample V_DC (for logging/analysis)
                        r = cfg.resistors.r_sample_estimate
                        R = cfg.resistors.r_series_dc
                        v_dc_sample_est = (v_dc_meas * r / (R + r)
                                            if (R + r) > 0 else v_dc_meas)

                        # Emit PointInfo + CSV row
                        pinfo = PointInfo(
                            outer_index=i_outer,
                            fast_index=i_fast,
                            direction=direction,
                            outer_value=float(outer_v),
                            v_dc_cmd=v_dc_cmd,
                            v_dc_meas=v_dc_meas,
                            i_dc_meas=i_dc_meas,
                            v_dc_sample_est=v_dc_sample_est,
                            didv_values=didv_values,
                            didv_errors=didv_errors,
                            i_gate_compl=composite.get_i_gate(),
                        )
                        self._write_csv_row(writer, pinfo, means, stds)
                        self.point_ready.emit(pinfo)
                        done += 1
                        self.progress.emit(done, total)

            self._completed_normally = not self._stop
            if self._stop:
                self._abort_cause = {'kind': 'user_stop',
                                      'message': 'stop() called from GUI'}

    # =================================================================
    # Conversion from DAQ V to physical kinds
    # =================================================================
    def _convert(self, means: List[float], stds: List[float],
                 cfg: DIDVConfig) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Returns (values, errors) dicts keyed by the channel NAME
        (ChannelConfig.name), with the kind-specific conversion applied.

        For dIdV_X / dIdV_Y / dIdV_R:
            V_lockin_rms = V_AI × sens / 10.0         [10 V = SR830 full scale at sens]
            dI/dV = V_lockin_rms × V_osc_eff / (gain × R_series_ac × V_AC_sample_rms^2)
            Wait — that isn't right.  Re-derive:
              V_AC_sample_rms ≈ V_osc_eff × R_sample / R_series_ac   (but we don't know R_sample)
              I_AC_rms = V_osc_eff / R_series_ac     (independent of R_sample)
              SR560 reads V_AC_sample × gain = V_lockin_rms
              ⇒ V_AC_sample_rms = V_lockin_rms / gain
              ⇒ dI/dV = I_AC / V_AC_sample_rms
                      = (V_osc_eff / R_series_ac) × gain / V_lockin_rms
            For X (in-phase) this is the real part of the AC conductance; for Y the imaginary.
        For theta: V_AI × 18.0 degrees/volt (SR830 CH2 phase mapping standard: ±180° over ±10 V)
        For Voltage: passthrough V_AI."""
        v_osc_eff = cfg.sr830.v_osc_rms * cfg.resistors.ac_attenuator
        R_ac = cfg.resistors.r_series_ac
        sens = cfg.sr830.sensitivity_v

        values: Dict[str, float] = {}
        errors: Dict[str, float] = {}
        for ch in self.enabled_channels:
            V_AI = means[ch.ai_index]
            dV_AI = stds[ch.ai_index]
            if ch.kind == 'Voltage':
                values[ch.name] = V_AI
                errors[ch.name] = dV_AI
            elif ch.kind == 'theta':
                # SR830 CH2 display=θ maps θ_degrees → V_out: V_out = θ × (10/180) V
                # i.e. 180° ↔ 10 V.  Invert:
                values[ch.name] = V_AI * 18.0
                errors[ch.name] = dV_AI * 18.0
            elif ch.kind in ('dIdV_X', 'dIdV_Y', 'dIdV_R'):
                # V_lockin = V_AI × sens / 10 V
                V_lockin = V_AI * sens / 10.0
                dV_lockin = dV_AI * sens / 10.0
                # dI/dV = (V_osc_eff × gain) / (R_ac × V_lockin)     Hmm but this
                # diverges at V_lockin=0.  We instead store the FORWARD formula:
                #   dI/dV = V_lockin × (something?)
                # NO — re-derive carefully:
                #   I_AC = V_osc_eff / R_ac    (AC current into sample, fixed)
                #   V_AC_sample = V_lockin / gain  (sample AC voltage)
                #   dI/dV = I_AC / V_AC_sample = (V_osc_eff / R_ac) × (gain / V_lockin)
                # Yes, this is INVERSELY proportional to V_lockin.  Fine.
                #
                # But what if V_lockin is small?  Then dI/dV is LARGE (high
                # conductance, = low sample R).  That's the correct physics.
                # We do need to guard against V_lockin == 0 exactly.
                if abs(V_lockin) < 1e-15:
                    values[ch.name] = float('nan')
                    errors[ch.name] = float('nan')
                else:
                    didv = (v_osc_eff / R_ac) * ch.gain / V_lockin
                    # Error propagation: |∂(dI/dV)/∂V_lockin| = |didv / V_lockin|
                    didv_err = abs(didv / V_lockin) * dV_lockin
                    values[ch.name] = didv
                    errors[ch.name] = didv_err
            else:
                values[ch.name] = 0.0
                errors[ch.name] = 0.0
        return values, errors

    def _demo_ai_provider(self, v_dc_cmd: float, v_gate: float):
        """Synthetic AI voltages for DEMO_MODE.  Simulates a BCS-like
        dI/dV(V) spectrum with a superconducting gap at ±Δ, plus
        small gate-voltage dependence.  The demo uses V_DC_cmd
        directly as the energy axis (pretending R_series_dc gives an
        appropriate sample V_DC_sample), then works backwards to
        synthesise the AI voltage that would be observed.
        Channel layout assumption (matches default GUI):
          AI0 = CH1 (X display), AI1 = CH2 (Y display),
          AI2 = CH1 (R display), AI3 = CH2 (θ display).
        """
        # Energy axis: use the sweep range to pick a gap that's visible
        # across the scan.  Default target: gap at 30 % of the sweep's
        # half-range, so a ±V_max scan shows coherence peaks around ±0.3 V_max.
        v_max = max(abs(self.cfg.sweep.v_dc_min),
                    abs(self.cfg.sweep.v_dc_max), 1e-6)
        gap = 0.3 * v_max * (1.0 + 0.05 * v_gate)   # gate tunes gap slightly

        V_s = v_dc_cmd        # "sample" voltage for the demo spectrum
        # Piecewise BCS-like dI/dV(V) expressed as RATIO of normal-state
        # plateau conductance:
        #   |V| < gap   : 0.3  (strongly suppressed, subgap states)
        #   |V| = gap   : ~ gap/broaden ≈ 20  (coherence peak)
        #   |V| >> gap  : 1.0 (plateau)
        broaden = 0.05 * gap
        floor = 0.30
        peak_cap = 5.0
        if abs(V_s) < gap - broaden:
            didv_ratio_over_plateau = floor
        else:
            raw = abs(V_s) / np.sqrt(max(V_s**2 - gap**2, 0.0) + broaden**2)
            didv_ratio_over_plateau = min(raw, peak_cap)
            didv_ratio_over_plateau = max(didv_ratio_over_plateau, floor)

        # Compute the corresponding V_AC at sample and V_lockin_X.  Here
        # we use the SAME conversion formula as _convert, INVERTED:
        #     dI/dV = (V_osc_eff × gain) / (R_ac × V_lockin)
        # so V_lockin = (V_osc_eff × gain) / (R_ac × dI/dV)
        v_osc_eff = self.cfg.sr830.v_osc_rms * self.cfg.resistors.ac_attenuator
        R_ac = self.cfg.resistors.r_series_ac
        # Assume the enabled X channel's gain (fallback 100 if not found)
        gain = 100.0
        for ch in self.enabled_channels:
            if ch.kind == 'dIdV_X' and ch.gain > 0:
                gain = ch.gain
                break
        # Dynamically pick Rn so the PLATEAU signal is ~30 % of AI full
        # scale (3 V out of ±10 V).  This ensures the whole BCS-like
        # spectrum fits in the AI range regardless of user sens choice.
        sens = self.cfg.sr830.sensitivity_v
        v_lockin_plateau_target = 0.3 * sens    # 30% of sens, well within range
        didv_S_plateau = (v_osc_eff * gain) / (R_ac * v_lockin_plateau_target)
        didv_S = didv_S_plateau * didv_ratio_over_plateau
        V_lockin_X = (v_osc_eff * gain) / (R_ac * max(didv_S, 1e-12))
        # Add small quadrature signal (phase varies across the spectrum)
        V_lockin_Y = 0.03 * V_lockin_X * np.tanh(V_s / max(gap, 1e-9))
        V_lockin_R = np.sqrt(V_lockin_X**2 + V_lockin_Y**2)
        theta_deg  = np.degrees(np.arctan2(V_lockin_Y, V_lockin_X))

        sens = self.cfg.sr830.sensitivity_v
        V_AI_X = V_lockin_X / sens * 10.0
        V_AI_Y = V_lockin_Y / sens * 10.0
        V_AI_R = V_lockin_R / sens * 10.0
        V_AI_theta = theta_deg / 18.0
        # Clip to SR830 full-scale ±10 V AI range
        V_AI_X = float(np.clip(V_AI_X, -10.0, 10.0))
        V_AI_Y = float(np.clip(V_AI_Y, -10.0, 10.0))
        V_AI_R = float(np.clip(V_AI_R, -10.0, 10.0))
        V_AI_theta = float(np.clip(V_AI_theta, -10.0, 10.0))
        return (V_AI_X, V_AI_Y, V_AI_R, V_AI_theta,
                0.0, 0.0, 0.0, 0.0)

    # =================================================================
    # CSV header + row
    # =================================================================
    def _csv_base_columns(self) -> List[str]:
        return [
            'outer_index', 'fast_index', 'direction',
            'v_outer_cmd_V',             # gate V (NaN if outer_axis=='none')
            'v_dc_cmd_V',                # commanded V_DC at K2636B source
            'v_dc_smu_V',                # SMU measured V readback
            'i_dc_smu_A',                # SMU measured I readback
            'v_dc_sample_est_V',         # V at sample (via divider, using r_sample_est)
            'i_gate_compl_A',            # gate leakage current (NaN if no gate)
        ]

    def _csv_channel_columns(self) -> List[str]:
        cols: List[str] = []
        for ch in self.enabled_channels:
            unit = {
                'dIdV_X': 'S', 'dIdV_Y': 'S', 'dIdV_R': 'S',
                'theta':   'deg',
                'Voltage': 'V',
            }.get(ch.kind, '')
            base = f'{ch.name}_{ch.kind}' if ch.kind != 'Voltage' else ch.name
            cols.append(f'{base}_{unit}_mean')
            cols.append(f'{base}_{unit}_std')
            cols.append(f'{base}_AI_V_mean')    # raw DAQ voltage, for rederivation
            cols.append(f'{base}_AI_V_std')
        return cols

    def _write_csv_header(self, writer):
        writer.writerow(self._csv_base_columns() + self._csv_channel_columns())

    def _write_csv_row(self, writer, pinfo: PointInfo,
                       means: List[float], stds: List[float]):
        row: List = [
            pinfo.outer_index, pinfo.fast_index, pinfo.direction,
            f'{pinfo.outer_value:.8e}' if not np.isnan(pinfo.outer_value) else 'nan',
            f'{pinfo.v_dc_cmd:.8e}',
            f'{pinfo.v_dc_meas:.8e}',
            f'{pinfo.i_dc_meas:.8e}',
            f'{pinfo.v_dc_sample_est:.8e}',
            (f'{pinfo.i_gate_compl:.8e}'
             if not np.isnan(pinfo.i_gate_compl) else 'nan'),
        ]
        for ch in self.enabled_channels:
            v = pinfo.didv_values.get(ch.name, float('nan'))
            e = pinfo.didv_errors.get(ch.name, float('nan'))
            row.append(f'{v:.8e}' if not np.isnan(v) else 'nan')
            row.append(f'{e:.8e}' if not np.isnan(e) else 'nan')
            row.append(f'{means[ch.ai_index]:.8e}')
            row.append(f'{stds[ch.ai_index]:.8e}')
        writer.writerow(row)

    # =================================================================
    # Sidecars  (TXT + JSON)
    # =================================================================
    def _sidecar_paths(self):
        stem, _ = os.path.splitext(self.cfg.save_path)
        return stem + '.txt', stem + '.json'

    def _write_txt_sidecar(self):
        cfg = self.cfg
        txt_path, _ = self._sidecar_paths()
        L: List[str] = []
        def row(k, v, indent=2):
            L.append(' ' * indent + f'{k:<34} = {v}')

        L.append(f"# {APP_NAME}  ·  {APP_VERSION}")
        L.append(f"# Generated at {self._started_at_iso}")
        L.append(f"# CSV file: {os.path.basename(cfg.save_path)}")
        L.append("")

        L.append("[ Sample / run ]")
        row('sample',   cfg.sample)
        row('device',   cfg.device)
        row('operator', cfg.operator)
        row('run_name', cfg.run_name)
        row('sample_T_K', f'{cfg.sample_T_k:.3f}')
        row('started_at', self._started_at_iso)
        if self._completed_normally:
            row('status', 'COMPLETED_NORMALLY')
        elif self._abort_cause is not None:
            row('status', f'ABORTED ({self._abort_cause.get("kind", "?")})')
        else:
            row('status', 'INCOMPLETE')
        L.append("")

        L.append("[ Sweep ]")
        row('fast_axis',    f'V_DC_cmd,  {cfg.sweep.v_dc_min:+.4e} to '
                            f'{cfg.sweep.v_dc_max:+.4e} V  '
                            f'({cfg.sweep.num_points} points)')
        row('bidirectional', cfg.sweep.bidirectional)
        row('outer_axis',    cfg.sweep.outer_axis)
        if cfg.sweep.outer_axis == 'gate':
            row('outer_range',  f'V_gate {cfg.gate.v_outer_min:+.3f} to '
                                f'{cfg.gate.v_outer_max:+.3f} V  '
                                f'({cfg.gate.num_outer} points)')
        else:
            row('gate_fixed_V', f'{cfg.gate.v_fixed:+.3f}' if cfg.gate.enabled else 'n/a')
        row('t_settle_point_s',    f'{cfg.sweep.t_settle_point:.4f}')
        row('n_avg',               f'{cfg.sweep.n_avg}')
        row('v_dc_limit_cmd_V',    f'{cfg.sweep.v_dc_limit_cmd_v:+.3f}')
        row('v_dc_limit_sample_V', f'{cfg.sweep.v_dc_limit_sample_v*1e3:.3f} mV')
        L.append("")

        L.append("[ Bias source  (K2636B) ]")
        row('k2636b_visa_address', cfg.k2636b_visa_address)
        row('bias_channel',        cfg.bias.channel)
        row('bias_mode',           cfg.bias.mode)
        row('bias_compliance_a',   f'{cfg.bias.compliance_a:.3e}')
        row('voltage_range_v',     f'{cfg.bias.voltage_range_v:.1f}')
        row('ramp_rate_V_per_s',   f'{SMU_RAMP_RATE_V_PER_S}')
        row('ramp_step_v',         f'{SMU_RAMP_STEP_V}')
        row('compliance_abort_fraction', f'{SMU_COMPLIANCE_ABORT_FRACTION}')
        L.append("")

        L.append("[ Gate  (K2636B, optional) ]")
        row('gate_enabled', cfg.gate.enabled)
        if cfg.gate.enabled:
            row('gate_channel',       cfg.gate.channel)
            row('gate_compliance_a',  f'{cfg.gate.compliance_a:.3e}')
            if cfg.sweep.outer_axis != 'gate':
                row('gate_fixed_V',   f'{cfg.gate.v_fixed:+.3f}')
        L.append("")

        L.append("[ Lock-in  (SR830) ]")
        row('sr830_visa_address',  cfg.sr830_visa_address)
        row('frequency_Hz',        f'{cfg.sr830.frequency_hz:.4f}')
        row('v_osc_rms_V',         f'{cfg.sr830.v_osc_rms:.4f}')
        row('sensitivity_V',       f'{cfg.sr830.sensitivity_v:.3e}')
        row('time_constant_s',     f'{cfg.sr830.time_constant_s:.3e}')
        row('slope_dB_oct',        f'{cfg.sr830.slope_db_oct}')
        row('reserve',             cfg.sr830.reserve)
        row('sync_filter',         cfg.sr830.sync_filter)
        row('input_source',        cfg.sr830.input_source)
        row('input_coupling',      cfg.sr830.input_coupling)
        row('input_ground',        cfg.sr830.input_ground)
        row('auto_configured',     cfg.sr830.auto_configure)
        L.append("")

        L.append("[ Resistor network ]")
        row('r_series_dc_ohm', f'{cfg.resistors.r_series_dc:.3e}')
        row('r_series_ac_ohm', f'{cfg.resistors.r_series_ac:.3e}')
        row('r_sample_est_ohm', f'{cfg.resistors.r_sample_estimate:.3e}')
        row('ac_attenuator',    f'{cfg.resistors.ac_attenuator:.4f}')
        v_osc_eff = cfg.sr830.v_osc_rms * cfg.resistors.ac_attenuator
        row('v_osc_eff_rms_V', f'{v_osc_eff:.4e}')
        row('I_AC_to_sample_A', f'{v_osc_eff/cfg.resistors.r_series_ac:.3e}')
        ok, v_ac, kT = cfg.energy_resolution_ok()
        row('est_V_AC_sample_rms_V', f'{v_ac:.3e}')
        row('est_V_AC_sample_peak_V', f'{v_ac*np.sqrt(2):.3e}')
        row('kT_over_e_at_sample_V', f'{kT:.3e}')
        row('energy_resolution_OK', ok)
        L.append("")

        L.append("[ NI DAQ  (AI readout) ]")
        row('device_name',       cfg.daq_device_name)
        row('ai_terminal',       DAQ_AI_TERMINAL)
        row('ai_sample_rate_Hz', f'{cfg.daq_ai_rate_hz:.1f}')
        row('n_avg',             cfg.sweep.n_avg)
        row('avg_window_s',
            f'{cfg.sweep.n_avg/cfg.daq_ai_rate_hz:.4f}  '
            f'({cfg.sweep.n_avg/cfg.daq_ai_rate_hz*1e3:.2f} ms)')
        L.append("")

        L.append("[ Channels ]")
        for ch in self.channels:
            if ch.enabled:
                row(f'AI{ch.ai_index}',
                    f'name={ch.name!r}, kind={ch.kind}, '
                    f'sens={ch.sens:.2e}, gain={ch.gain:.2f}')
            else:
                row(f'AI{ch.ai_index}', f'name={ch.name!r}, DISABLED')
        L.append("")

        if self._abort_cause is not None:
            L.append("[ Abort cause ]")
            for k, v in self._abort_cause.items():
                row(k, v)
            L.append("")

        L.append("[ CSV columns ]")
        cols = self._csv_base_columns() + self._csv_channel_columns()
        for c in cols:
            L.append(f'    {c}')
        L.append("")

        L.append("# To load:  df = pd.read_csv('%s')" %
                 os.path.basename(cfg.save_path))

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(L) + '\n')
        return txt_path

    def _write_metadata_sidecar(self):
        _, json_path = self._sidecar_paths()
        cfg = self.cfg
        # Build a JSON-safe nested dict.  Use asdict() for dataclasses,
        # then add schema_version and runtime-only fields.
        def _ds(x):
            try:
                return asdict(x)
            except Exception:
                return x
        meta = {
            'schema_version': 'didv_v1.0',
            'software':       f'{APP_NAME} {APP_VERSION}',
            'started_at':     self._started_at_iso,
            'completed_normally': self._completed_normally,
            'abort_cause':    self._abort_cause,
            # Provenance
            'sample':   cfg.sample,
            'device':   cfg.device,
            'operator': cfg.operator,
            'run_name': cfg.run_name,
            'sample_T_K': cfg.sample_T_k,
            # Config
            'k2636b_visa_address': cfg.k2636b_visa_address,
            'bias':  _ds(cfg.bias),
            'gate':  _ds(cfg.gate),
            'sr830_visa_address':  cfg.sr830_visa_address,
            'sr830': {**_ds(cfg.sr830),
                       'sensitivity_v': cfg.sr830.sensitivity_v,
                       'time_constant_s': cfg.sr830.time_constant_s},
            'resistors': _ds(cfg.resistors),
            'sweep':  _ds(cfg.sweep),
            'daq': {
                'device_name':         cfg.daq_device_name,
                'ai_terminal':         DAQ_AI_TERMINAL,
                'ai_sample_rate_hz':   cfg.daq_ai_rate_hz,
                'n_avg':               cfg.sweep.n_avg,
                'avg_window_s':        cfg.sweep.n_avg/cfg.daq_ai_rate_hz,
            },
            'channels': [asdict(c) for c in self.channels],
            'smu_ramp': {
                'rate_v_per_s': SMU_RAMP_RATE_V_PER_S,
                'step_v':       SMU_RAMP_STEP_V,
                'compliance_abort_fraction': SMU_COMPLIANCE_ABORT_FRACTION,
            },
            'preflight': {
                'tc_multiplier':   PREFLIGHT_TC_MULTIPLIER,
                'tc_minimum':      PREFLIGHT_TC_MINIMUM,
                'energy_resolution_check':
                    list(cfg.energy_resolution_ok()),
            },
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, default=str)
        return json_path


# =====================================================================
# 6. GUI main window
# =====================================================================
class NoWheelComboBox(QComboBox):
    """Ignores mouse wheel events; prevents accidentally changing a
    setting while scrolling the config panel."""
    def wheelEvent(self, e):
        e.ignore()


class DIDVGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} · {APP_VERSION}")
        self.resize(1500, 950)

        self._apply_style()

        self._thread: Optional[MeasurementThread] = None
        self._scan_data: Dict[str, List[float]] = {
            'v_dc_cmd': [], 'v_dc_sample': [],
        }
        self._channel_series: Dict[str, List[float]] = {}

        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(6, 6, 6, 6)

        # Left: config panel inside a scroll area
        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True)
        left_panel = QWidget(); left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        # Build groups with category-specific accent colors:
        #   Teal      = provenance (sample/run, output)
        #   Blue      = instruments (K2636B, SR830, DAQ)
        #   Mauve     = measurement config (sweep, resistors, channels)
        #   Green     = controls
        def _accent(gb, color):
            gb.setStyleSheet(f"QGroupBox {{ border-left: 3px solid {color}; }}\n"
                             f"QGroupBox::title {{ color: {color}; }}")
            return gb

        left_layout.addWidget(_accent(self._build_meta_group(), CT_TEAL))
        left_layout.addWidget(_accent(self._build_bias_group(), CT_BLUE))
        left_layout.addWidget(_accent(self._build_gate_group(), CT_BLUE))
        left_layout.addWidget(_accent(self._build_sr830_group(), CT_BLUE))
        left_layout.addWidget(_accent(self._build_daq_group(), CT_BLUE))
        left_layout.addWidget(_accent(self._build_resistors_group(), CT_MAUVE))
        left_layout.addWidget(_accent(self._build_sweep_group(), CT_MAUVE))
        left_layout.addWidget(_accent(self._build_channels_group(), CT_MAUVE))
        left_layout.addWidget(_accent(self._build_output_group(), CT_TEAL))
        left_layout.addWidget(_accent(self._build_controls_group(), CT_GREEN))
        left_layout.addStretch(1)
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(580)

        # Right: plot + log
        right_split = QSplitter(Qt.Vertical)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(CT_BASE)
        self.plot.setLabel('bottom', 'V_DC_sample_est', units='V',
                           color=CT_TEXT)
        self.plot.setLabel('left',   'dI/dV', units='S',
                           color=CT_TEXT)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot_legend = self.plot.addLegend()
        self._plot_curves: Dict[str, pg.PlotDataItem] = {}

        log_container = QWidget(); log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setFont(QFont('Consolas', 10))
        log_label = QLabel("Event log")
        log_label.setStyleSheet(
            f"color: {CT_TEAL}; font-size: 14px; font-weight: bold; "
            f"padding: 2px 0px;")
        log_layout.addWidget(log_label)
        log_layout.addWidget(self.log)

        right_split.addWidget(self.plot)
        right_split.addWidget(log_container)
        right_split.setSizes([700, 250])

        root.addWidget(left_scroll, 1)
        root.addWidget(right_split, 2)

        self.status = QStatusBar(); self.setStatusBar(self.status)
        self._progress_bar = QProgressBar(); self._progress_bar.setMaximum(100)
        self.status.addPermanentWidget(self._progress_bar)

        self._load_settings()
        self._log_event(f"{APP_NAME} · {APP_VERSION} ready.", "success")
        if DEMO_MODE:
            self._log_event("DEMO mode — no real hardware.", "warning")

    # -----------------------------------------------------------------
    # Styling
    # -----------------------------------------------------------------
    def _apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {CT_BASE};
            }}
            QWidget {{
                color: {CT_TEXT};
                font-family: 'Segoe UI', 'Microsoft YaHei UI', sans-serif;
                font-size: 13px;
            }}

            /* --- Group boxes with left accent bar --- */
            QGroupBox {{
                border: 1px solid {CT_SURFACE1};
                border-left: 3px solid {CT_MAUVE};
                border-radius: 4px;
                margin-top: 14px;
                padding: 20px 8px 8px 8px;
                background: {CT_MANTLE};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 2px 8px;
                color: {CT_MAUVE};
                font-size: 14px;
                font-weight: bold;
                letter-spacing: 0.3px;
            }}

            /* --- Field labels: slightly subdued for hierarchy --- */
            QLabel {{
                color: {CT_SUBTEXT1};
                font-size: 12px;
            }}

            /* --- Inputs --- */
            QLineEdit {{
                background: {CT_SURFACE0};
                color: {CT_TEXT};
                border: 1px solid {CT_SURFACE1};
                border-radius: 3px;
                padding: 3px 6px;
                font-size: 13px;
                min-height: 20px;
                selection-background-color: {CT_MAUVE};
            }}
            QLineEdit:focus {{
                border: 1px solid {CT_BLUE};
            }}

            QComboBox {{
                background: {CT_SURFACE0};
                color: {CT_TEXT};
                border: 1px solid {CT_SURFACE1};
                border-radius: 3px;
                padding: 3px 6px;
                font-size: 13px;
                min-height: 20px;
            }}
            QComboBox:focus {{
                border: 1px solid {CT_BLUE};
            }}
            QComboBox QAbstractItemView {{
                background: {CT_SURFACE0};
                color: {CT_TEXT};
                selection-background-color: {CT_SURFACE1};
            }}

            QCheckBox {{
                spacing: 6px;
                font-size: 12px;
            }}
            QCheckBox::indicator {{
                width: 16px; height: 16px;
                border: 1px solid {CT_SURFACE1};
                border-radius: 3px;
                background: {CT_SURFACE0};
            }}
            QCheckBox::indicator:checked {{
                background: {CT_MAUVE};
                border-color: {CT_MAUVE};
            }}

            /* --- Buttons --- */
            QPushButton {{
                background: {CT_SURFACE0};
                color: {CT_TEXT};
                border: 1px solid {CT_SURFACE1};
                border-radius: 4px;
                padding: 5px 14px;
                font-size: 13px;
                font-weight: bold;
                min-height: 24px;
            }}
            QPushButton:hover {{
                background: {CT_SURFACE1};
                border-color: {CT_OVERLAY};
            }}
            QPushButton:pressed {{
                background: {CT_OVERLAY};
            }}
            QPushButton:disabled {{
                color: {CT_OVERLAY};
                border-color: {CT_SURFACE0};
            }}

            /* START button green accent */
            QPushButton#btn_start {{
                background: #2d4a3e;
                border-color: {CT_GREEN};
                color: {CT_GREEN};
            }}
            QPushButton#btn_start:hover {{
                background: #3a5c4d;
            }}
            /* STOP button red accent */
            QPushButton#btn_stop {{
                background: #4a2d3a;
                border-color: {CT_RED};
                color: {CT_RED};
            }}
            QPushButton#btn_stop:hover {{
                background: #5c3a4d;
            }}

            /* --- Log panel --- */
            QPlainTextEdit {{
                background: {CT_MANTLE};
                color: {CT_TEXT};
                border: 1px solid {CT_SURFACE1};
                border-radius: 3px;
                font-size: 12px;
            }}

            /* --- Progress bar --- */
            QProgressBar {{
                background: {CT_SURFACE0};
                border: 1px solid {CT_SURFACE1};
                border-radius: 3px;
                text-align: center;
                color: {CT_TEXT};
                font-size: 12px;
                min-height: 18px;
            }}
            QProgressBar::chunk {{
                background: {CT_GREEN};
                border-radius: 2px;
            }}

            /* --- Scroll area transparent background --- */
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollArea > QWidget > QWidget {{
                background: transparent;
            }}
            QScrollBar:vertical {{
                background: {CT_MANTLE};
                width: 10px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: {CT_SURFACE1};
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {CT_OVERLAY};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}

            /* --- Status bar --- */
            QStatusBar {{
                background: {CT_MANTLE};
                color: {CT_SUBTEXT1};
                font-size: 12px;
            }}

            /* --- Tooltips --- */
            QToolTip {{
                background: {CT_SURFACE0};
                color: {CT_TEXT};
                border: 1px solid {CT_SURFACE1};
                padding: 4px;
                font-size: 12px;
            }}
        """)

    # -----------------------------------------------------------------
    # GUI builders
    # -----------------------------------------------------------------
    def _build_meta_group(self):
        gb = QGroupBox("Sample / run"); grid = QGridLayout(); r = 0
        for label, attr, default in [
            ("Sample",     'le_sample',   ''),
            ("Device",     'le_device',   ''),
            ("Operator",   'le_operator', ''),
            ("Run name",   'le_run_name', ''),
        ]:
            grid.addWidget(QLabel(label), r, 0)
            le = QLineEdit(default); setattr(self, attr, le)
            grid.addWidget(le, r, 1); r += 1
        grid.addWidget(QLabel("Sample T (K)"), r, 0)
        self.le_sample_T = QLineEdit('0.040'); grid.addWidget(self.le_sample_T, r, 1); r += 1
        gb.setLayout(grid); return gb

    def _build_bias_group(self):
        gb = QGroupBox("Bias source (K2636B)"); grid = QGridLayout(); r = 0
        grid.addWidget(QLabel("K2636B VISA"), r, 0)
        self.le_k2636b_addr = QLineEdit('GPIB0::26::INSTR')
        grid.addWidget(self.le_k2636b_addr, r, 1, 1, 2); r += 1

        grid.addWidget(QLabel("Bias mode"), r, 0)
        self.cb_bias_mode = NoWheelComboBox()
        for m in BIAS_MODE_OPTIONS: self.cb_bias_mode.addItem(m)
        grid.addWidget(self.cb_bias_mode, r, 1)
        grid.addWidget(QLabel("channel"), r, 2)
        self.cb_bias_channel = NoWheelComboBox()
        self.cb_bias_channel.addItems(K2636_CHANNEL_OPTIONS)
        self.cb_bias_channel.setCurrentText(K2636_DEFAULT_BIAS_CHANNEL)
        grid.addWidget(self.cb_bias_channel, r, 3); r += 1

        grid.addWidget(QLabel("Compliance (A)"), r, 0)
        self.le_bias_compl = QLineEdit(f'{SMU_DEFAULT_BIAS_COMPLIANCE_A:g}')
        grid.addWidget(self.le_bias_compl, r, 1)
        grid.addWidget(QLabel("V range (V)"), r, 2)
        self.le_bias_vrange = QLineEdit('21.0')
        grid.addWidget(self.le_bias_vrange, r, 3); r += 1

        gb.setLayout(grid); return gb

    def _build_gate_group(self):
        gb = QGroupBox("Gate (K2636B, optional)"); grid = QGridLayout(); r = 0
        self.cb_gate_enabled = QCheckBox("Enable gate")
        grid.addWidget(self.cb_gate_enabled, r, 0, 1, 2); r += 1

        grid.addWidget(QLabel("channel"), r, 0)
        self.cb_gate_channel = NoWheelComboBox()
        self.cb_gate_channel.addItems(K2636_CHANNEL_OPTIONS)
        self.cb_gate_channel.setCurrentText(K2636_DEFAULT_GATE_CHANNEL)
        grid.addWidget(self.cb_gate_channel, r, 1)
        grid.addWidget(QLabel("compl (A)"), r, 2)
        self.le_gate_compl = QLineEdit(f'{SMU_DEFAULT_GATE_COMPLIANCE_A:g}')
        grid.addWidget(self.le_gate_compl, r, 3); r += 1

        grid.addWidget(QLabel("V fixed (V)"), r, 0)
        self.le_gate_vfixed = QLineEdit('0.0')
        grid.addWidget(self.le_gate_vfixed, r, 1)
        grid.addWidget(QLabel("V range (V)"), r, 2)
        self.le_gate_vrange = QLineEdit('21.0')
        grid.addWidget(self.le_gate_vrange, r, 3); r += 1

        grid.addWidget(QLabel("Outer V_min"), r, 0)
        self.le_gate_vmin = QLineEdit('-3.0'); grid.addWidget(self.le_gate_vmin, r, 1)
        grid.addWidget(QLabel("V_max"), r, 2)
        self.le_gate_vmax = QLineEdit('3.0');  grid.addWidget(self.le_gate_vmax, r, 3); r += 1
        grid.addWidget(QLabel("N outer"), r, 0)
        self.le_gate_nouter = QLineEdit('11'); grid.addWidget(self.le_gate_nouter, r, 1); r += 1

        gb.setLayout(grid); return gb

    def _build_sr830_group(self):
        gb = QGroupBox("Lock-in (SR830)"); grid = QGridLayout(); r = 0
        grid.addWidget(QLabel("SR830 VISA"), r, 0)
        self.le_sr830_addr = QLineEdit('GPIB0::8::INSTR')
        grid.addWidget(self.le_sr830_addr, r, 1, 1, 3); r += 1

        grid.addWidget(QLabel("Frequency (Hz)"), r, 0)
        self.le_sr830_freq = QLineEdit('13.333'); grid.addWidget(self.le_sr830_freq, r, 1)
        grid.addWidget(QLabel("V_osc_rms (V)"), r, 2)
        self.le_sr830_vosc = QLineEdit('0.004'); grid.addWidget(self.le_sr830_vosc, r, 3); r += 1

        grid.addWidget(QLabel("Sensitivity"), r, 0)
        self.cb_sr830_sens = NoWheelComboBox()
        for idx, s in enumerate(SR830_SENSITIVITY_VALUES):
            self.cb_sr830_sens.addItem(f'{s:.2e}', idx)
        self.cb_sr830_sens.setCurrentIndex(20)  # 10 mV
        grid.addWidget(self.cb_sr830_sens, r, 1)
        grid.addWidget(QLabel("Time const"), r, 2)
        self.cb_sr830_tc = NoWheelComboBox()
        for idx, t in enumerate(SR830_TC_VALUES):
            self.cb_sr830_tc.addItem(f'{t:.3e}', idx)
        self.cb_sr830_tc.setCurrentIndex(10)    # 1 s
        grid.addWidget(self.cb_sr830_tc, r, 3); r += 1

        grid.addWidget(QLabel("Slope (dB/oct)"), r, 0)
        self.cb_sr830_slope = NoWheelComboBox()
        for s in SR830_SLOPE_DB_VALUES: self.cb_sr830_slope.addItem(str(s))
        self.cb_sr830_slope.setCurrentText('24')
        grid.addWidget(self.cb_sr830_slope, r, 1)
        grid.addWidget(QLabel("Reserve"), r, 2)
        self.cb_sr830_reserve = NoWheelComboBox()
        self.cb_sr830_reserve.addItems(SR830_RESERVE_VALUES)
        self.cb_sr830_reserve.setCurrentText('Low Noise')
        grid.addWidget(self.cb_sr830_reserve, r, 3); r += 1

        grid.addWidget(QLabel("Input source"), r, 0)
        self.cb_sr830_src = NoWheelComboBox()
        self.cb_sr830_src.addItems(SR830_SOURCE_VALUES)
        grid.addWidget(self.cb_sr830_src, r, 1)
        grid.addWidget(QLabel("Coupling"), r, 2)
        self.cb_sr830_cpl = NoWheelComboBox()
        self.cb_sr830_cpl.addItems(SR830_COUPLING_VALUES)
        grid.addWidget(self.cb_sr830_cpl, r, 3); r += 1

        grid.addWidget(QLabel("Ground"), r, 0)
        self.cb_sr830_gnd = NoWheelComboBox()
        self.cb_sr830_gnd.addItems(SR830_GROUND_VALUES)
        grid.addWidget(self.cb_sr830_gnd, r, 1)
        self.cb_sr830_sync = QCheckBox("Sync filter"); self.cb_sr830_sync.setChecked(True)
        grid.addWidget(self.cb_sr830_sync, r, 2)
        self.cb_sr830_auto = QCheckBox("Auto-configure at start")
        self.cb_sr830_auto.setChecked(True)
        grid.addWidget(self.cb_sr830_auto, r, 3); r += 1

        gb.setLayout(grid); return gb

    def _build_resistors_group(self):
        gb = QGroupBox("Resistor network"); grid = QGridLayout(); r = 0
        grid.addWidget(QLabel("R_series_dc (Ω)"), r, 0)
        self.le_r_dc = QLineEdit('1e6'); grid.addWidget(self.le_r_dc, r, 1)
        grid.addWidget(QLabel("R_series_ac (Ω)"), r, 2)
        self.le_r_ac = QLineEdit('100e3'); grid.addWidget(self.le_r_ac, r, 3); r += 1
        grid.addWidget(QLabel("R_sample_est (Ω)"), r, 0)
        self.le_r_sample = QLineEdit('1e3'); grid.addWidget(self.le_r_sample, r, 1)
        grid.addWidget(QLabel("AC attenuator"), r, 2)
        self.le_ac_atten = QLineEdit('1.0'); grid.addWidget(self.le_ac_atten, r, 3); r += 1
        gb.setLayout(grid); return gb

    def _build_sweep_group(self):
        gb = QGroupBox("V_DC sweep"); grid = QGridLayout(); r = 0
        grid.addWidget(QLabel("V_DC min (V)"), r, 0)
        self.le_vdc_min = QLineEdit('-0.001'); grid.addWidget(self.le_vdc_min, r, 1)
        grid.addWidget(QLabel("V_DC max (V)"), r, 2)
        self.le_vdc_max = QLineEdit('0.001');  grid.addWidget(self.le_vdc_max, r, 3); r += 1
        grid.addWidget(QLabel("Num points"), r, 0)
        self.le_vdc_n = QLineEdit('101'); grid.addWidget(self.le_vdc_n, r, 1)
        self.cb_bidir = QCheckBox("Bidirectional (fwd + bwd)")
        self.cb_bidir.setChecked(True)
        grid.addWidget(self.cb_bidir, r, 2, 1, 2); r += 1
        grid.addWidget(QLabel("t_settle (s)"), r, 0)
        self.le_t_settle = QLineEdit('0.2'); grid.addWidget(self.le_t_settle, r, 1)
        grid.addWidget(QLabel("n_avg"), r, 2)
        self.le_n_avg = QLineEdit(str(DAQ_DEFAULT_NAVG))
        grid.addWidget(self.le_n_avg, r, 3); r += 1
        grid.addWidget(QLabel("Outer axis"), r, 0)
        self.cb_outer = NoWheelComboBox()
        self.cb_outer.addItems(['none', 'gate'])
        grid.addWidget(self.cb_outer, r, 1); r += 1
        grid.addWidget(QLabel("V_DC_cmd safety (V)"), r, 0)
        self.le_vdc_limit_cmd = QLineEdit('5.0')
        grid.addWidget(self.le_vdc_limit_cmd, r, 1)
        grid.addWidget(QLabel("V_DC_sample safety (V)"), r, 2)
        self.le_vdc_limit_sample = QLineEdit('0.05')
        grid.addWidget(self.le_vdc_limit_sample, r, 3); r += 1
        gb.setLayout(grid); return gb

    def _build_daq_group(self):
        gb = QGroupBox("NI DAQ (AI readout)"); grid = QGridLayout(); r = 0
        grid.addWidget(QLabel("Device name"), r, 0)
        self.le_daq_device = QLineEdit('Dev1')
        self.le_daq_device.setToolTip(
            "NI-MAX device name (e.g. 'Dev1', 'PXI1Slot2').  "
            "Open NI-MAX to check the name of your AI board.")
        grid.addWidget(self.le_daq_device, r, 1)
        grid.addWidget(QLabel("AI rate (Hz)"), r, 2)
        self.le_daq_rate = QLineEdit(f'{DAQ_AI_RATE_HZ:.0f}')
        self.le_daq_rate.setToolTip(
            "AI sample-clock rate.  n_avg samples will be averaged per "
            "measurement point, so the effective averaging window is "
            "n_avg / AI_rate.")
        grid.addWidget(self.le_daq_rate, r, 3); r += 1
        gb.setLayout(grid); return gb

    def _build_channels_group(self):
        gb = QGroupBox("DAQ channels"); grid = QGridLayout(); r = 0
        grid.setVerticalSpacing(4)
        headers = ['AI', 'on', 'name', 'kind', 'sens (V)', 'gain']
        for c, h in enumerate(headers):
            lbl = QLabel(h)
            lbl.setStyleSheet(
                f"color: {CT_OVERLAY}; font-size: 11px; font-weight: bold; "
                f"text-transform: uppercase; padding-bottom: 2px;")
            grid.addWidget(lbl, 0, c)
        r = 1
        self.ch_enable_checks: List[QCheckBox] = []
        self.ch_name_inputs:   List[QLineEdit] = []
        self.ch_kind_combos:   List[QComboBox] = []
        self.ch_sens_inputs:   List[QLineEdit] = []
        self.ch_gain_inputs:   List[QLineEdit] = []
        default_kinds = ['dIdV_X', 'dIdV_Y', 'dIdV_R', 'theta',
                         'disabled', 'disabled', 'disabled', 'disabled']
        default_names = ['dIdV_X', 'dIdV_Y', 'dIdV_R', 'theta',
                         'AI4', 'AI5', 'AI6', 'AI7']
        for ai in range(NUM_AI):
            grid.addWidget(QLabel(f'ai{ai}'), r, 0)
            chk = QCheckBox(); chk.setChecked(default_kinds[ai] != 'disabled')
            self.ch_enable_checks.append(chk); grid.addWidget(chk, r, 1)
            name = QLineEdit(default_names[ai]); self.ch_name_inputs.append(name)
            grid.addWidget(name, r, 2)
            kind = NoWheelComboBox()
            kind.addItems(['disabled', 'dIdV_X', 'dIdV_Y', 'dIdV_R',
                           'theta', 'Voltage'])
            kind.setCurrentText(default_kinds[ai]); self.ch_kind_combos.append(kind)
            grid.addWidget(kind, r, 3)
            sens = QLineEdit('10e-3'); self.ch_sens_inputs.append(sens)
            grid.addWidget(sens, r, 4)
            gain = QLineEdit('100'); self.ch_gain_inputs.append(gain)
            grid.addWidget(gain, r, 5)
            r += 1
        gb.setLayout(grid); return gb

    def _build_output_group(self):
        gb = QGroupBox("Output"); grid = QGridLayout(); r = 0
        grid.addWidget(QLabel("Folder"), r, 0)
        self.le_folder = QLineEdit(os.getcwd())
        grid.addWidget(self.le_folder, r, 1, 1, 2)
        btn = QPushButton("..."); btn.clicked.connect(self._browse_folder)
        grid.addWidget(btn, r, 3); r += 1
        self.lbl_filename = QLabel("(auto)"); grid.addWidget(self.lbl_filename, r, 0, 1, 4); r += 1
        gb.setLayout(grid); return gb

    def _build_controls_group(self):
        gb = QGroupBox("Controls"); h = QHBoxLayout()
        h.setSpacing(10)
        self.btn_start = QPushButton("  START  ")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop  = QPushButton("  STOP  ")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)
        self.btn_test  = QPushButton("Test hardware")
        self.btn_test.clicked.connect(self._on_test)
        h.addWidget(self.btn_start); h.addWidget(self.btn_stop); h.addWidget(self.btn_test)
        gb.setLayout(h); return gb

    # -----------------------------------------------------------------
    # GUI callbacks
    # -----------------------------------------------------------------
    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder",
                                              self.le_folder.text())
        if d:
            self.le_folder.setText(d)

    def _log_event(self, text: str, level: str = 'info'):
        color = {'info': CT_TEXT, 'success': CT_GREEN,
                 'warning': CT_YELLOW, 'error': CT_RED}.get(level, CT_TEXT)
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        self.log.appendHtml(
            f'<span style="color:{CT_OVERLAY};">{ts}</span> '
            f'<span style="color:{color};">{text}</span>')
        self.log.ensureCursorVisible()

    def _parse_config(self) -> Tuple[DIDVConfig, List[ChannelConfig]]:
        def f(s, name):
            try: return float(s)
            except ValueError: raise ValueError(f"{name}: cannot parse {s!r} as float.")
        def i(s, name):
            try: return int(s)
            except ValueError: raise ValueError(f"{name}: cannot parse {s!r} as int.")

        bias = BiasConfig(
            mode=self.cb_bias_mode.currentText(),
            channel=self.cb_bias_channel.currentText(),
            compliance_a=f(self.le_bias_compl.text(), 'bias_compliance'),
            voltage_range_v=f(self.le_bias_vrange.text(), 'bias_vrange'),
        )
        gate = GateConfig(
            enabled=self.cb_gate_enabled.isChecked(),
            channel=self.cb_gate_channel.currentText(),
            compliance_a=f(self.le_gate_compl.text(), 'gate_compliance'),
            voltage_range_v=f(self.le_gate_vrange.text(), 'gate_vrange'),
            v_fixed=f(self.le_gate_vfixed.text(), 'gate_vfixed'),
            v_outer_min=f(self.le_gate_vmin.text(), 'gate_vmin'),
            v_outer_max=f(self.le_gate_vmax.text(), 'gate_vmax'),
            num_outer=i(self.le_gate_nouter.text(), 'gate_nouter'),
        )
        sr = SR830Config(
            frequency_hz=f(self.le_sr830_freq.text(), 'sr830_freq'),
            v_osc_rms=f(self.le_sr830_vosc.text(), 'sr830_vosc'),
            sensitivity_index=int(self.cb_sr830_sens.currentData()),
            time_constant_index=int(self.cb_sr830_tc.currentData()),
            slope_db_oct=int(self.cb_sr830_slope.currentText()),
            reserve=self.cb_sr830_reserve.currentText(),
            sync_filter=self.cb_sr830_sync.isChecked(),
            input_source=self.cb_sr830_src.currentText(),
            input_coupling=self.cb_sr830_cpl.currentText(),
            input_ground=self.cb_sr830_gnd.currentText(),
            auto_configure=self.cb_sr830_auto.isChecked(),
        )
        res = ResistorNetwork(
            r_series_dc=f(self.le_r_dc.text(), 'r_series_dc'),
            r_series_ac=f(self.le_r_ac.text(), 'r_series_ac'),
            r_sample_estimate=f(self.le_r_sample.text(), 'r_sample_est'),
            ac_attenuator=f(self.le_ac_atten.text(), 'ac_attenuator'),
        )
        sw = SweepConfig(
            v_dc_min=f(self.le_vdc_min.text(), 'v_dc_min'),
            v_dc_max=f(self.le_vdc_max.text(), 'v_dc_max'),
            num_points=i(self.le_vdc_n.text(), 'num_points'),
            bidirectional=self.cb_bidir.isChecked(),
            t_settle_point=f(self.le_t_settle.text(), 't_settle'),
            n_avg=i(self.le_n_avg.text(), 'n_avg'),
            outer_axis=self.cb_outer.currentText(),
            v_dc_limit_cmd_v=f(self.le_vdc_limit_cmd.text(), 'v_dc_limit_cmd'),
            v_dc_limit_sample_v=f(self.le_vdc_limit_sample.text(), 'v_dc_limit_sample'),
        )
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = self.le_run_name.text().strip() or 'didv'
        sample = self.le_sample.text().strip() or 'sample'
        filename = f'{ts}_{sample}_{run_name}.csv'
        save_path = os.path.join(self.le_folder.text().strip() or '.', filename)
        # Collision avoidance
        stem, ext = os.path.splitext(save_path)
        if (os.path.exists(save_path) or os.path.exists(stem + '.txt')
                or os.path.exists(stem + '.json')):
            for k in range(2, 999):
                cand = f'{stem}_{k}{ext}'
                cstem = os.path.splitext(cand)[0]
                if not (os.path.exists(cand) or os.path.exists(cstem+'.txt')
                        or os.path.exists(cstem+'.json')):
                    save_path = cand; break

        cfg = DIDVConfig(
            sample=sample,
            device=self.le_device.text().strip(),
            operator=self.le_operator.text().strip(),
            run_name=run_name,
            sample_T_k=f(self.le_sample_T.text(), 'sample_T'),
            k2636b_visa_address=self.le_k2636b_addr.text().strip(),
            bias=bias, gate=gate,
            sr830_visa_address=self.le_sr830_addr.text().strip(),
            sr830=sr, resistors=res, sweep=sw,
            daq_device_name=self.le_daq_device.text().strip() or 'Dev1',
            daq_ai_rate_hz=f(self.le_daq_rate.text(), 'daq_ai_rate'),
            save_path=save_path,
        )
        # Channels
        channels = []
        for ai in range(NUM_AI):
            channels.append(ChannelConfig(
                ai_index=ai,
                name=self.ch_name_inputs[ai].text().strip() or f'AI{ai}',
                enabled=self.ch_enable_checks[ai].isChecked(),
                kind=self.ch_kind_combos[ai].currentText(),
                sens=f(self.ch_sens_inputs[ai].text(), f'AI{ai}_sens'),
                gain=f(self.ch_gain_inputs[ai].text(), f'AI{ai}_gain'),
            ))
        self.lbl_filename.setText(os.path.basename(save_path))
        return cfg, channels

    def _on_start(self):
        try:
            cfg, channels = self._parse_config()
            cfg.validate()
            for ch in channels: ch.validate()
        except Exception as e:
            QMessageBox.critical(self, "Parse/validate error", str(e))
            self._log_event(f"Parse error: {e}", "error")
            return
        self._log_event(
            f"Starting: V_DC ∈ [{cfg.sweep.v_dc_min:+.4g}, "
            f"{cfg.sweep.v_dc_max:+.4g}] V × {cfg.sweep.num_points} pts, "
            f"{'fwd+bwd' if cfg.sweep.bidirectional else 'fwd only'}, "
            f"outer={cfg.sweep.outer_axis}",
            "info")
        # Reset plot (clear curves AND legend entries so repeated START
        # runs don't accumulate stale legend rows)
        self.plot.clear()
        self._plot_curves.clear()
        try:
            if self._plot_legend is not None:
                self._plot_legend.clear()
        except Exception:
            pass
        self._scan_data = {'v_dc_cmd': [], 'v_dc_sample': []}
        self._channel_series = {ch.name: [] for ch in channels if ch.enabled}
        for ch in channels:
            if ch.enabled and ch.kind in ('dIdV_X', 'dIdV_R'):
                color = {'dIdV_X': CT_BLUE, 'dIdV_Y': CT_TEAL,
                         'dIdV_R': CT_MAUVE, 'theta': CT_PEACH,
                         'Voltage': CT_GREEN}.get(ch.kind, CT_TEXT)
                self._plot_curves[ch.name] = self.plot.plot(
                    [], [], pen=pg.mkPen(color=color, width=2), name=ch.name)

        self._thread = MeasurementThread(cfg, channels)
        self._thread.log_msg.connect(self._log_event)
        self._thread.point_ready.connect(self._on_point)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished_ok.connect(self._on_finished_ok)
        self._thread.error_occurred.connect(self._on_error)
        self._thread.finished.connect(self._on_thread_done)
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self.btn_test.setEnabled(False)
        self._thread.start()

    def _on_stop(self):
        if self._thread is not None and self._thread.isRunning():
            self._thread.stop()
            self._log_event("Stop requested — ramping to safe state...", "warning")

    def _on_test(self):
        """Quick instrument probe: *IDN? each device and report."""
        try:
            cfg, _ = self._parse_config()
        except Exception as e:
            QMessageBox.critical(self, "Parse error", str(e)); return
        # K2636B
        k = K2636B_DualChannel(cfg.k2636b_visa_address)
        try:
            k.open(); self._log_event(f"K2636B: {k._idn}", "success")
        except Exception as e:
            self._log_event(f"K2636B test FAILED: {e}", "error")
        finally:
            try: k.close()
            except Exception: pass
        # SR830
        sr = SR830Controller(cfg.sr830_visa_address, cfg.sr830)
        try:
            sr.open(); self._log_event(f"SR830: {sr._idn}", "success")
        except Exception as e:
            self._log_event(f"SR830 test FAILED: {e}", "error")
        finally:
            try: sr.close()
            except Exception: pass
        # DAQ
        daq = DAQHardware(device_name=cfg.daq_device_name,
                          ai_rate_hz=cfg.daq_ai_rate_hz,
                          n_avg=cfg.sweep.n_avg)
        try:
            daq.open(); self._log_event(f"DAQ: {daq.device_name}  OK", "success")
        except Exception as e:
            self._log_event(f"DAQ test FAILED: {e}", "error")
        finally:
            try: daq.close()
            except Exception: pass

    def _on_point(self, pinfo: PointInfo):
        self._scan_data['v_dc_cmd'].append(pinfo.v_dc_cmd)
        self._scan_data['v_dc_sample'].append(pinfo.v_dc_sample_est)
        for name, val in pinfo.didv_values.items():
            self._channel_series.setdefault(name, []).append(val)
        # Update curves
        xs = np.array(self._scan_data['v_dc_sample'])
        for name, curve in self._plot_curves.items():
            ys = np.array(self._channel_series.get(name, []))
            if len(ys) == len(xs):
                curve.setData(xs, ys)

    def _on_progress(self, done: int, total: int):
        if total > 0:
            self._progress_bar.setMaximum(total)
            self._progress_bar.setValue(done)
            self.status.showMessage(f"{done} / {total}")

    def _on_finished_ok(self, path):
        self._log_event(f"✓ Completed: {os.path.basename(path)}", "success")
        QMessageBox.information(self, "Done", f"Saved:\n{path}")

    def _on_error(self, tb):
        self._log_event("Error traceback:", "error")
        for line in tb.strip().split('\n')[-5:]:
            self._log_event(f"  {line}", "error")

    def _on_thread_done(self):
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.btn_test.setEnabled(True)

    # -----------------------------------------------------------------
    # Settings persistence
    # -----------------------------------------------------------------
    def _persistent_items(self):
        items = [
            ('lineedit', 'sample',       self.le_sample),
            ('lineedit', 'device',       self.le_device),
            ('lineedit', 'operator',     self.le_operator),
            ('lineedit', 'run_name',     self.le_run_name),
            ('lineedit', 'sample_T',     self.le_sample_T),
            ('lineedit', 'k2636b_addr',  self.le_k2636b_addr),
            ('combo',    'bias_mode',    self.cb_bias_mode),
            ('combo',    'bias_channel', self.cb_bias_channel),
            ('lineedit', 'bias_compl',   self.le_bias_compl),
            ('lineedit', 'bias_vrange',  self.le_bias_vrange),
            ('checkbox', 'gate_enabled', self.cb_gate_enabled),
            ('combo',    'gate_channel', self.cb_gate_channel),
            ('lineedit', 'gate_compl',   self.le_gate_compl),
            ('lineedit', 'gate_vfixed',  self.le_gate_vfixed),
            ('lineedit', 'gate_vrange',  self.le_gate_vrange),
            ('lineedit', 'gate_vmin',    self.le_gate_vmin),
            ('lineedit', 'gate_vmax',    self.le_gate_vmax),
            ('lineedit', 'gate_nouter',  self.le_gate_nouter),
            ('lineedit', 'sr830_addr',   self.le_sr830_addr),
            ('lineedit', 'sr830_freq',   self.le_sr830_freq),
            ('lineedit', 'sr830_vosc',   self.le_sr830_vosc),
            ('combo',    'sr830_slope',  self.cb_sr830_slope),
            ('combo',    'sr830_reserve',self.cb_sr830_reserve),
            ('combo',    'sr830_src',    self.cb_sr830_src),
            ('combo',    'sr830_cpl',    self.cb_sr830_cpl),
            ('combo',    'sr830_gnd',    self.cb_sr830_gnd),
            ('checkbox', 'sr830_sync',   self.cb_sr830_sync),
            ('checkbox', 'sr830_auto',   self.cb_sr830_auto),
            ('lineedit', 'r_dc',         self.le_r_dc),
            ('lineedit', 'r_ac',         self.le_r_ac),
            ('lineedit', 'r_sample',     self.le_r_sample),
            ('lineedit', 'ac_atten',     self.le_ac_atten),
            ('lineedit', 'vdc_min',      self.le_vdc_min),
            ('lineedit', 'vdc_max',      self.le_vdc_max),
            ('lineedit', 'vdc_n',        self.le_vdc_n),
            ('checkbox', 'bidir',        self.cb_bidir),
            ('lineedit', 't_settle',     self.le_t_settle),
            ('lineedit', 'n_avg',        self.le_n_avg),
            ('combo',    'outer',        self.cb_outer),
            ('lineedit', 'vdc_limit_cmd',    self.le_vdc_limit_cmd),
            ('lineedit', 'vdc_limit_sample', self.le_vdc_limit_sample),
            ('lineedit', 'daq_device',   self.le_daq_device),
            ('lineedit', 'daq_rate',     self.le_daq_rate),
            ('lineedit', 'folder',       self.le_folder),
        ]
        for ai in range(NUM_AI):
            items.append(('checkbox', f'ch{ai}_enabled', self.ch_enable_checks[ai]))
            items.append(('lineedit', f'ch{ai}_name',    self.ch_name_inputs[ai]))
            items.append(('combo',    f'ch{ai}_kind',    self.ch_kind_combos[ai]))
            items.append(('lineedit', f'ch{ai}_sens',    self.ch_sens_inputs[ai]))
            items.append(('lineedit', f'ch{ai}_gain',    self.ch_gain_inputs[ai]))
        return items

    def _load_settings(self):
        s = QSettings(ORG_NAME, SETTINGS_NAME)
        for kind, key, w in self._persistent_items():
            v = s.value(key)
            if v is None: continue
            if kind == 'lineedit':
                w.setText(str(v))
            elif kind == 'checkbox':
                w.setChecked(str(v).lower() in ('true', '1', 'yes'))
            elif kind == 'combo':
                idx = w.findText(str(v))
                if idx >= 0: w.setCurrentIndex(idx)
        # Settings for sensitivity/TC index combos (stored as data, not text)
        for key, combo, default_idx in [
            ('sr830_sens_idx', self.cb_sr830_sens, 20),
            ('sr830_tc_idx',   self.cb_sr830_tc,   10),
        ]:
            v = s.value(key)
            if v is not None:
                try:
                    combo.setCurrentIndex(int(v))
                except Exception:
                    combo.setCurrentIndex(default_idx)

    def _save_settings(self):
        s = QSettings(ORG_NAME, SETTINGS_NAME)
        for kind, key, w in self._persistent_items():
            if kind == 'lineedit':
                s.setValue(key, w.text())
            elif kind == 'checkbox':
                s.setValue(key, w.isChecked())
            elif kind == 'combo':
                s.setValue(key, w.currentText())
        s.setValue('sr830_sens_idx', self.cb_sr830_sens.currentIndex())
        s.setValue('sr830_tc_idx',   self.cb_sr830_tc.currentIndex())

    def closeEvent(self, ev):
        if self._thread is not None and self._thread.isRunning():
            ans = QMessageBox.question(self, "Scan running",
                "A scan is active. Abort and quit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes: ev.ignore(); return
            self._thread.stop()
            ok = self._thread.wait(15000)
            if not ok:
                self._log_event(
                    "Worker did not stop in 15s — CHECK K2636B OUTPUT MANUALLY",
                    "error")
        try: self._save_settings()
        except Exception as e: print(f"save settings failed: {e}", file=sys.stderr)
        ev.accept()


# =====================================================================
# 7. Entry point
# =====================================================================
def main():
    global DEMO_MODE, nidaqmx, TerminalConfiguration, AcquisitionType
    global pyvisa, VisaIOError

    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} {APP_VERSION}")
    parser.add_argument('--demo', action='store_true',
        help="Run in DEMO mode (no hardware, synthetic BCS-like spectrum).")
    args = parser.parse_args()
    DEMO_MODE = bool(args.demo)

    if not DEMO_MODE:
        try:
            import nidaqmx as _nidaqmx
            from nidaqmx.constants import (
                TerminalConfiguration as _Term,
                AcquisitionType as _Acq)
            nidaqmx = _nidaqmx
            TerminalConfiguration = _Term
            AcquisitionType = _Acq
        except ImportError as e:
            sys.stderr.write(
                f"ERROR: nidaqmx not available ({e}).\n"
                f"Install:  python -m pip install nidaqmx\n"
                f"Or run with --demo.\n")
            sys.exit(1)
        try:
            import pyvisa as _pyvisa
            from pyvisa.errors import VisaIOError as _VIOE
            pyvisa = _pyvisa; VisaIOError = _VIOE
        except ImportError as e:
            sys.stderr.write(
                f"ERROR: pyvisa not available ({e}).\n"
                f"Install:  python -m pip install pyvisa pyvisa-py\n"
                f"Or run with --demo.\n")
            sys.exit(1)

    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    pg.setConfigOption('background', CT_BASE)
    pg.setConfigOption('foreground', CT_TEXT)
    pg.setConfigOptions(antialias=True)

    gui = DIDVGui(); gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()