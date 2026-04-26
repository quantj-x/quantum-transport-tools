# -*- coding: utf-8 -*-
"""
Keithley IV Sweep  —  v1.0
==========================

Keithley 2636B 双通道 SMU 通断/IV 扫描程序。 与 hysteresis_mapping.py
和 dual_gate_mapping.py 完全独立。

物理工作流
----------
对一对样品电极 (e.g. A1 ↔ B3) 做 I-V 测量, 看:
  · I-V 是否线性 (是否 ohmic 接触, 是否有二极管整流, 是否有充放电滞回)
  · 电阻大小 (R = dV/dI 的拟合斜率倒数)
  · 接触失调电流 (拟合截距)

接线模型 (2-wire 默认):
  · 电极 A → Channel A 的 HI
  · 电极 B → Channel A 的 LO
  · SMU 强制电压 V, 同步测量电流 I (compliance limited)
  · GUI 选 Channel A 或 Channel B (二选一, 不并行)

扫描模式 (START 按钮三选一):
  · forward    单次正扫 0 → V_max
  · backward   单次反扫 V_max → 0  (或 GUI 设的另一极)
  · round_trip 来回扫 0 → +V_max → -V_max → 0  (可改成对称双向)

每个扫描点:
  1. SMU.write_voltage(V_set)
  2. 等 NPLC × (1/line_freq)  (硬件计时积分)
  3. SMU.read_iv() → (V_meas, I_meas)
  4. 算 R_inst = V_meas / I_meas
  5. 同步更新 I(V) 和 R(V) 两条曲线
  6. (扫完后) 对全部点做线性拟合, 报告 R_fit ± σ + R²

调用方式
--------
默认 DEMO 模式:        python Keithley_IV_sweep.py
真实硬件:              python Keithley_IV_sweep.py --live

与 hysteresis_mapping.py / dual_gate_mapping.py 的关系
------------------------------------------------------
完全独立。 没有任何 import 关系。 共享代码 (Catppuccin GUI 框架,
NoWheelComboBox, _json_clean 等) 是手动复制并精简过来的。
QSettings namespace 不同 ('lab.transport / Keithley_IV_sweep'),
三个程序可以同时打开互不干扰。

继承自 hysteresis_mapping v1.1 的工业级硬化经验 (无 v1.0 bug-fix 周期):
  · --live/--demo CLI flag, 不是源码常量
  · NPLC 真做硬件计时积分 (不像 v1.0 hysteresis 的假 t_read)
  · _json_clean() 严格 RFC-7159 合规, 无 NaN literal
  · KeithleyController 三个实现 (Demo + TSP + SCPI), 都有
    verify_communication() 接口和 GUI Verify 按钮
  · LIVE-mode 安全门: V_max > 1V 时拒绝启动除非用户验证过 Keithley
  · closeEvent 8s wait + terminate fallback
  · SIGINT QTimer wakeup (Linux Ctrl+C)
  · 状态栏 [DEMO]/[LIVE] 模式徽章
  · 文件名 collision auto-suffix (_2 ... _999)
  · 周期 fsync (CSV_FSYNC_EVERY_N=10) 限断电丢数
  · 全部 thread race 防护 (defensive None checks 在所有信号 handler)
  · QSettings 跨平台 list/string 兼容 (Linux/Windows registry/macOS plist)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

# ---------------------------------------------------------------------
# Optional pyvisa import — only needed for real Keithley
# ---------------------------------------------------------------------
try:
    import pyvisa
    HAS_PYVISA = True
except Exception:
    HAS_PYVISA = False

# ---------------------------------------------------------------------
# Mode flag — set by main() from CLI args before any other module-level
# code that depends on it runs. Default True (DEMO) so file can be
# imported as a library / for tests without crashing.
# ---------------------------------------------------------------------
DEMO_MODE: bool = True

# ---------------------------------------------------------------------
# Qt + pyqtgraph
# ---------------------------------------------------------------------
from PyQt5.QtCore import (Qt, QThread, pyqtSignal, QTimer, QSettings, QSize)
from PyQt5.QtGui import QFont, QColor, QIcon, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QComboBox,
    QPushButton, QCheckBox, QGroupBox, QHBoxLayout, QVBoxLayout, QGridLayout,
    QSplitter, QFileDialog, QPlainTextEdit, QProgressBar, QStatusBar, QAction,
    QMenu, QToolButton, QSizePolicy, QFrame, QScrollArea, QMessageBox,
    QStyledItemDelegate, QRadioButton, QButtonGroup,
)
import pyqtgraph as pg


# =====================================================================
# 常量
# =====================================================================

APP_NAME    = "Keithley IV Sweep"
APP_VERSION = "v1.0"
ORG_NAME    = "lab.transport"
SETTINGS_NAME = "Keithley_IV_sweep"

# Keithley 2636B has two independent SMU channels
CHANNELS    = ('A', 'B')
DEFAULT_CHANNEL = 'A'

# Sweep modes (set by START button)
SWEEP_FORWARD    = 'forward'      # 0 → +V_max (single)
SWEEP_BACKWARD   = 'backward'     # +V_max → 0 (single, runs from current saved end)
SWEEP_ROUND_TRIP = 'round_trip'   # 0 → +V_max → 0 → -V_max → 0 (full)
SWEEP_MODES = (SWEEP_FORWARD, SWEEP_BACKWARD, SWEEP_ROUND_TRIP)

SWEEP_LABEL = {
    SWEEP_FORWARD:    'Forward only (0 → +V_max)',
    SWEEP_BACKWARD:   'Backward only (+V_max → 0)',
    SWEEP_ROUND_TRIP: 'Round trip (0 → +V_max → 0 → -V_max → 0)',
}

# Keithley protocol
PROTO_TSP   = 'tsp'    # Lua-based, factory default for 2636B
PROTO_SCPI  = 'scpi'   # SCPI mode (must be selected on Keithley front panel)
PROTOCOLS   = (PROTO_TSP, PROTO_SCPI)
PROTO_LABEL = {
    PROTO_TSP:  'TSP (Lua, factory default)',
    PROTO_SCPI: 'SCPI (must be enabled on instrument)',
}

# Line-frequency for NPLC integration (50 Hz Asia/EU, 60 Hz NA)
LINE_FREQ_HZ = 50.0    # most labs in Asia/Europe; Keithley adapts to mains

# Periodic CSV fsync (every N points); bounds power-loss data loss
CSV_FSYNC_EVERY_N = 10

# Keithley VISA timeout — single queries should be fast
KEITHLEY_VISA_TIMEOUT_MS = 3000

# Maximum allowed |V| / |I| before refusing to even try
# (2636B hardware limits are ±42V / ±3A but for samples that's insane)
SOFT_V_LIMIT = 21.0   # V (half of hardware max, safety)
SOFT_I_LIMIT = 1.0    # A (1 A is huge for low-temp samples)

# Default values for fresh GUI
DEFAULT_V_MAX_V        = 0.005   # 5 mV — your typical continuity sweep
DEFAULT_V_STEP_V       = 0.001   # 1 mV step
DEFAULT_NPLC           = 1.0
DEFAULT_I_COMPLIANCE_A = 1e-6    # 1 µA — safe for low-temp samples
DEFAULT_SETTLE_S       = 0.005   # 5 ms between set_voltage and read

# Catppuccin Mocha palette (same as hysteresis_mapping for visual consistency)
CT_BASE      = '#1e1e2e'
CT_MANTLE    = '#181825'
CT_CRUST     = '#11111b'
CT_SURFACE0  = '#313244'
CT_SURFACE1  = '#45475a'
CT_SURFACE2  = '#585b70'
CT_TEXT      = '#cdd6f4'
CT_SUBTEXT0  = '#a6adc8'
CT_SUBTEXT1  = '#bac2de'
CT_OVERLAY1  = '#7f849c'
CT_BLUE      = '#89b4fa'
CT_GREEN     = '#a6e3a1'
CT_YELLOW    = '#f9e2af'
CT_PEACH     = '#fab387'
CT_RED       = '#f38ba8'
CT_MAUVE     = '#cba6f7'
CT_PINK      = '#f5c2e7'
CT_TEAL      = '#94e2d5'
CT_LAVENDER  = '#b4befe'


# =====================================================================
# JSON helper (RFC 7159 strict — no NaN/Infinity literals)
# =====================================================================
def _json_clean(obj):
    """Recursively replace NaN/Infinity floats with None for strict
    RFC-7159 JSON output. (json.dumps writes 'NaN' literals by default
    which are rejected by MATLAB jsondecode and JavaScript JSON.parse.)
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_clean(obj.tolist())
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(x) for x in obj]
    return obj


def fmt_dur(s: float) -> str:
    """Compact human-readable duration: '1m 23s' / '450 ms' / '1h 02m'."""
    if s < 1.0:
        return f"{int(round(s * 1000))} ms"
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        m, sec = divmod(int(round(s)), 60)
        return f"{m}m {sec:02d}s"
    h, rem = divmod(int(round(s)), 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h {m:02d}m"


def fmt_eng(value: float, unit: str = '') -> str:
    """Engineering-notation formatter: 0.00012 → '120 µ'.

    Used for currents, voltages, resistances in status display.
    Returns NaN-safe '—'.
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return f"— {unit}"
    if value == 0:
        return f"0 {unit}".strip()
    sign = '-' if value < 0 else ''
    av = abs(value)
    prefixes = [
        (1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'k'),
        (1.0, ''), (1e-3, 'm'), (1e-6, 'µ'), (1e-9, 'n'),
        (1e-12, 'p'), (1e-15, 'f'),
    ]
    for scale, prefix in prefixes:
        if av >= scale:
            return f"{sign}{av/scale:.4g} {prefix}{unit}".rstrip()
    return f"{value:.3e} {unit}".strip()


# =====================================================================
# 1.  Configuration dataclasses
# =====================================================================

@dataclass
class KeithleyConfig:
    """Controller-level config. Determines which class to instantiate
    in make_keithley_controller()."""
    protocol:        str = PROTO_TSP            # 'tsp' or 'scpi'
    visa_resource:   str = 'GPIB0::26::INSTR'   # VISA resource string
    channel:         str = DEFAULT_CHANNEL      # 'A' or 'B'
    use_4wire:       bool = False               # 2-wire (default) vs 4-wire Kelvin
    v_compliance_V:  float = SOFT_V_LIMIT       # voltage compliance (when sourcing I)
    i_compliance_A:  float = DEFAULT_I_COMPLIANCE_A  # current compliance (when sourcing V)


@dataclass
class SweepConfig:
    """Per-scan sweep parameters."""
    v_max_V:        float = DEFAULT_V_MAX_V
    v_step_V:       float = DEFAULT_V_STEP_V
    nplc:           float = DEFAULT_NPLC
    settle_s:       float = DEFAULT_SETTLE_S
    mode:           str = SWEEP_ROUND_TRIP   # 'forward' / 'backward' / 'round_trip'
    save_enabled:   bool = True
    save_path:      str = ''                  # set by parse_params

    @property
    def num_points_per_segment(self) -> int:
        """Points in a single 0 → V_max half-segment, including both endpoints."""
        if self.v_step_V <= 0:
            return 2
        return int(round(abs(self.v_max_V) / self.v_step_V)) + 1

    def voltage_sequence(self) -> List[float]:
        """Generate the actual sequence of V_set values for this sweep mode."""
        n = self.num_points_per_segment
        if n < 2:
            n = 2
        # Build half-sequence 0 → +V_max
        half = list(np.linspace(0.0, self.v_max_V, n))
        if self.mode == SWEEP_FORWARD:
            return half
        if self.mode == SWEEP_BACKWARD:
            return list(reversed(half))
        # round_trip: 0 → +Vmax → 0 → -Vmax → 0
        # Each transition uses (n-1) NEW points (don't repeat the joining point)
        seq = list(half)                                  # 0 → +Vmax
        seq += [v for v in reversed(half[:-1])]           # +Vmax → 0 (excl +Vmax)
        seq += [-v for v in half[1:]]                     # 0 → -Vmax (excl 0)
        seq += [-v for v in reversed(half[1:-1])] + [0.0] # -Vmax → 0 (excl -Vmax + close on 0)
        return seq


@dataclass
class MetaConfig:
    """Metadata fields recorded in CSV header + JSON sidecar."""
    sample:         str = ''
    device:         str = ''
    operator:       str = ''
    run_name:       str = ''
    electrode_pair: str = ''   # e.g. 'A1-B3' — what makes a continuity test identifiable


# =====================================================================
# 2.  KeithleyController — abstract base + Demo + TSP + SCPI
# =====================================================================

class KeithleyController:
    """Abstract base. Subclasses implement Keithley 2636B SMU control
    via either TSP (Lua) or SCPI command sets, plus a Demo simulation.

    The contract a measurement thread relies on:
      connect()                   → open VISA, configure channel
      disconnect()                → close VISA cleanly
      configure_source_v(...)     → set as voltage source with compliance
      write_voltage(V)            → set output to V volts
      read_iv() → (V_meas, I_meas) → measure both simultaneously
      output_on() / output_off()  → SMU output relay
      verify_communication()      → read-only sanity round trip
      describe()                  → human-readable summary
    """
    name = 'base'

    def __init__(self, config: KeithleyConfig):
        self.config = config
        self.connected = False

    def connect(self): ...
    def disconnect(self): ...
    def configure_source_v(self, v_compliance: float, i_compliance: float,
                           use_4wire: bool, nplc: float): ...
    def write_voltage(self, V: float): ...
    def read_iv(self) -> Tuple[float, float]: ...
    def output_on(self): ...
    def output_off(self): ...

    def verify_communication(self) -> Dict[str, Any]:
        """Read-only sanity check. Subclasses override.
        Returns dict with at least 'idn' and 'errors' keys."""
        return {'errors': [f'verify_communication not implemented for {self.name}']}

    def describe(self) -> str:
        return self.name


# ---------------------------------------------------------------------
# 2a.  DemoKeithley  —  pure numpy simulation
# ---------------------------------------------------------------------

def _demo_iv_response(V_set: float, R_demo: float, V_offset: float,
                      noise_amp: float, leak_C: float = 0.0) -> Tuple[float, float]:
    """Synthetic IV with realistic features:
      · Linear response with R_demo
      · Small offset voltage (asymmetric contact / thermoelectric)
      · Gaussian voltage + current noise (scaled to be much smaller than signal)
      · Tiny capacitive leak that produces sub-µV hysteresis
    
    noise_amp = 1.0 → typical lab noise: V noise ~1µV RMS, I noise ~1pA RMS.
    This is below the Keithley 2636B's quoted accuracy floor (~5µV / ~3pA on
    autorange) so it's optimistic but not unrealistic.
    """
    # V noise: 1 µV RMS at noise_amp=1
    V_meas = V_set + V_offset + noise_amp * np.random.randn() * 1e-6
    I_ideal = (V_meas - V_offset) / R_demo
    # I noise: 1 pA RMS at noise_amp=1
    I_meas = I_ideal + noise_amp * np.random.randn() * 1e-12
    return float(V_meas), float(I_meas)


class DemoKeithley(KeithleyController):
    """Numpy simulation. Generates linear IV around V=0 with small
    contact-asymmetry offset + Gaussian noise. Configurable per-channel."""
    name = 'demo'

    def __init__(self, config: KeithleyConfig):
        super().__init__(config)
        # Per-channel synthetic resistance (lab samples vary; pick something
        # interesting but plausible for a low-T graphene/CrI3 device)
        self._R_demo = 2340.0 if config.channel == 'A' else 17500.0
        self._V_offset = 1.5e-5 if config.channel == 'A' else -3.2e-5
        self._noise_amp = 1.0
        self._output_on = False
        self._last_V_set = 0.0
        self._compliance_hit = False
        self._nplc = 1.0

    def connect(self):
        self.connected = True

    def disconnect(self):
        self._output_on = False
        self.connected = False

    def configure_source_v(self, v_compliance: float, i_compliance: float,
                           use_4wire: bool, nplc: float):
        self._i_compliance = float(i_compliance)
        self._v_compliance = float(v_compliance)
        self._use_4wire    = bool(use_4wire)
        self._nplc         = float(nplc)
        self._compliance_hit = False

    def write_voltage(self, V: float):
        self._last_V_set = float(V)
        # Check if this voltage would exceed compliance current
        I_predicted = abs(V / self._R_demo)
        if I_predicted > self._i_compliance:
            self._compliance_hit = True

    def read_iv(self) -> Tuple[float, float]:
        if not self._output_on:
            return (0.0, 0.0)
        # Mimic NPLC integration time (real Keithley would block here)
        time.sleep(self._nplc / LINE_FREQ_HZ)
        V_meas, I_meas = _demo_iv_response(
            self._last_V_set, self._R_demo, self._V_offset, self._noise_amp)
        # Apply compliance clamping
        if abs(I_meas) > self._i_compliance:
            I_meas = math.copysign(self._i_compliance, I_meas)
            self._compliance_hit = True
        return V_meas, I_meas

    def output_on(self):
        self._output_on = True

    def output_off(self):
        self._output_on = False
        self._last_V_set = 0.0

    def verify_communication(self) -> Dict[str, Any]:
        return {
            'idn':            f'DEMO KEITHLEY 2636B (simulated, channel {self.config.channel})',
            'channel':        self.config.channel,
            'protocol':       'demo',
            'output_state':   'OFF' if not self._output_on else 'ON',
            'compliance_I_A': getattr(self, '_i_compliance', None),
            'compliance_V_V': getattr(self, '_v_compliance', None),
            'wire_mode':      '4-wire' if getattr(self, '_use_4wire', False) else '2-wire',
            'errors':         [],
        }

    def describe(self) -> str:
        return (f"DEMO Keithley 2636B (simulated, R={self._R_demo:.0f} Ω, "
                f"V_off={self._V_offset*1e6:+.1f} µV, ch={self.config.channel})")


# ---------------------------------------------------------------------
# 2b.  Keithley2636bTsp  —  TSP (Lua) command set, factory default
# ---------------------------------------------------------------------

class Keithley2636bTsp(KeithleyController):
    """Keithley 2636B in TSP (Lua) mode — factory default.

    ⚠️  WARNING — VERIFY ON HARDWARE BEFORE PRODUCTION USE  ⚠️
    The exact TSP command set has minor variations across firmware
    revisions. Before first production use:
      1. Read the 2600 Series Reference Manual (section "TSP Command
         Reference"). Get the PDF for your specific firmware revision
         from Keithley's support site.
      2. Click the 'Verify Keithley comms' button in the GUI — it
         only does *IDN? + smua/smub introspection (no output enable).
      3. Run a low-stakes test on a known-resistance sample:
         compliance = 100 µA, V = 1 mV, expected I = (1mV / R_known).
         Compare to your DMM measurement of the same resistor.

    TSP commands used (smuX = smua or smub depending on channel):
      print(localnode.model)              → '2636B'
      smuX.reset()                        → reset SMU to defaults
      smuX.source.func = smuX.OUTPUT_DCVOLTS  → voltage source
      smuX.source.autorangev = smuX.AUTORANGE_ON
      smuX.measure.autorangei = smuX.AUTORANGE_ON
      smuX.source.limiti = <I_compliance> → set current compliance (A)
      smuX.measure.nplc = <nplc>          → integration in NPLC
      smuX.sense = smuX.SENSE_LOCAL       → 2-wire
      smuX.sense = smuX.SENSE_REMOTE      → 4-wire (Kelvin)
      smuX.source.levelv = <V>            → set output voltage
      smuX.source.output = smuX.OUTPUT_ON
      smuX.source.output = smuX.OUTPUT_OFF
      print(smuX.measure.iv())            → returns 'I,V' (note ORDER: I first!)
      print(smuX.source.compliance)       → boolean: in compliance?
    """
    name = 'tsp'

    def __init__(self, config: KeithleyConfig):
        super().__init__(config)
        self._rm   = None
        self._inst = None
        # Map channel 'A'/'B' to TSP smu prefix
        self._smu = 'smua' if config.channel == 'A' else 'smub'

    def connect(self):
        if not HAS_PYVISA:
            raise RuntimeError("pyvisa not installed — run `pip install pyvisa` first.")
        self._rm   = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(self.config.visa_resource)
        self._inst.timeout           = KEITHLEY_VISA_TIMEOUT_MS
        self._inst.read_termination  = '\n'
        self._inst.write_termination = '\n'
        # Verify it's a 2636B in TSP mode. *IDN? works in TSP too.
        try:
            idn = self._inst.query("*IDN?")
        except Exception as e:
            raise RuntimeError(
                f"Keithley *IDN? failed at {self.config.visa_resource!r}. "
                f"Underlying: {e}")
        if "MODEL 2636" not in idn.upper() and "2636" not in idn.upper():
            raise RuntimeError(
                f"Device at {self.config.visa_resource!r} doesn't identify as "
                f"a 2636-series Keithley (IDN: {idn!r}). Wrong VISA address?")
        # TSP smoke check
        try:
            mdl = self._inst.query("print(localnode.model)").strip()
        except Exception:
            raise RuntimeError(
                f"TSP command failed — instrument may be in SCPI mode. "
                f"Either switch to SCPI in the GUI, or change command set "
                f"on the front panel: MENU → SETTINGS → COMMAND SET → TSP.")
        self.connected = True

    def disconnect(self):
        # Best-effort: turn outputs off before closing
        try:
            if self._inst is not None:
                self._inst.write(f"{self._smu}.source.output = {self._smu}.OUTPUT_OFF")
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

    def configure_source_v(self, v_compliance: float, i_compliance: float,
                           use_4wire: bool, nplc: float):
        smu = self._smu
        cmds = [
            f"{smu}.reset()",
            f"{smu}.source.func = {smu}.OUTPUT_DCVOLTS",
            f"{smu}.source.autorangev = {smu}.AUTORANGE_ON",
            f"{smu}.measure.autorangei = {smu}.AUTORANGE_ON",
            f"{smu}.source.limiti = {i_compliance:.6e}",
            f"{smu}.measure.nplc = {nplc:.4f}",
            (f"{smu}.sense = {smu}.SENSE_REMOTE" if use_4wire
             else f"{smu}.sense = {smu}.SENSE_LOCAL"),
            f"{smu}.source.levelv = 0",
        ]
        for c in cmds:
            self._inst.write(c)

    def write_voltage(self, V: float):
        # Soft-clamp at SOFT_V_LIMIT defensively
        V = max(-SOFT_V_LIMIT, min(SOFT_V_LIMIT, float(V)))
        self._inst.write(f"{self._smu}.source.levelv = {V:.6e}")

    def read_iv(self) -> Tuple[float, float]:
        # Note: smuX.measure.iv() returns "I, V" — current FIRST in TSP!
        raw = self._inst.query(f"print({self._smu}.measure.iv())").strip()
        # Response is e.g. "1.234567e-08\t5.000000e-03" (tab) or comma
        parts = [p for p in raw.replace(',', '\t').split('\t') if p.strip()]
        if len(parts) < 2:
            raise RuntimeError(f"TSP measure.iv() bad response: {raw!r}")
        try:
            I_meas = float(parts[0])
            V_meas = float(parts[1])
        except ValueError as e:
            raise RuntimeError(f"TSP measure.iv() parse failed: {raw!r}") from e
        return V_meas, I_meas

    def output_on(self):
        self._inst.write(f"{self._smu}.source.output = {self._smu}.OUTPUT_ON")

    def output_off(self):
        try:
            self._inst.write(f"{self._smu}.source.levelv = 0")
            self._inst.write(f"{self._smu}.source.output = {self._smu}.OUTPUT_OFF")
        except Exception:
            pass

    def verify_communication(self) -> Dict[str, Any]:
        """Read-only TSP queries. NEVER turns output on."""
        result: Dict[str, Any] = {'errors': [], 'protocol': 'tsp',
                                  'channel': self.config.channel}
        smu = self._smu

        def _try(key, query_str, postprocess=lambda x: x.strip()):
            try:
                raw = self._inst.query(query_str)
                result[key] = postprocess(raw)
            except Exception as e:
                result[key] = None
                result['errors'].append(f"{key}: {e}")

        _try('idn',                   "*IDN?")
        _try('model',                 "print(localnode.model)")
        _try('serial',                "print(localnode.serialno)")
        _try('firmware',              "print(localnode.revision)")
        _try('output_state',          f"print({smu}.source.output)")
        _try('source_func',           f"print({smu}.source.func)")
        _try('compliance_I_A_set',    f"print({smu}.source.limiti)",
             postprocess=lambda r: float(r.strip()))
        _try('measure_nplc',          f"print({smu}.measure.nplc)",
             postprocess=lambda r: float(r.strip()))
        _try('sense_mode',            f"print({smu}.sense)")
        return result

    def describe(self) -> str:
        return f"Keithley 2636B TSP @ {self.config.visa_resource} (ch {self.config.channel})"


# ---------------------------------------------------------------------
# 2c.  Keithley2636bScpi  —  SCPI command set
# ---------------------------------------------------------------------

class Keithley2636bScpi(KeithleyController):
    """Keithley 2636B in SCPI mode.

    ⚠️  WARNING — VERIFY ON HARDWARE BEFORE PRODUCTION USE  ⚠️
    The instrument must be set to SCPI mode on the front panel:
      MENU → SETTINGS → COMMAND SET → SCPI
    SCPI mode in 2636B is mostly compatible with 24xx-series SCPI but
    has some 2-channel extensions. Verify with the 2600 SCPI Reference.

    SCPI commands (with channel index 1=A, 2=B):
      *IDN?                              → identification
      :SOUR<n>:FUNC VOLT                 → voltage source
      :SOUR<n>:VOLT:RANG:AUTO ON         → autorange voltage
      :SENS<n>:CURR:RANG:AUTO ON         → autorange current
      :SENS<n>:CURR:PROT <I>             → current compliance
      :SENS<n>:CURR:NPLC <nplc>          → integration time
      :SYST:RSEN <0|1>                   → 2-wire / 4-wire
      :SOUR<n>:VOLT <V>                  → set output voltage
      :OUTP<n> ON / OFF                  → output control
      :READ?<n>                          → returns V, I (V first!)
    """
    name = 'scpi'

    def __init__(self, config: KeithleyConfig):
        super().__init__(config)
        self._rm   = None
        self._inst = None
        self._n    = 1 if config.channel == 'A' else 2   # SCPI channel index

    def connect(self):
        if not HAS_PYVISA:
            raise RuntimeError("pyvisa not installed — run `pip install pyvisa` first.")
        self._rm   = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(self.config.visa_resource)
        self._inst.timeout           = KEITHLEY_VISA_TIMEOUT_MS
        self._inst.read_termination  = '\n'
        self._inst.write_termination = '\n'
        try:
            idn = self._inst.query("*IDN?")
        except Exception as e:
            raise RuntimeError(
                f"Keithley *IDN? failed at {self.config.visa_resource!r}. "
                f"Underlying: {e}")
        if "2636" not in idn.upper():
            raise RuntimeError(
                f"Device at {self.config.visa_resource!r} doesn't identify as "
                f"a 2636-series Keithley (IDN: {idn!r}).")
        # SCPI smoke: query system version (SCPI-only command in TSP-mode would error)
        try:
            self._inst.query(":SYST:VERS?")
        except Exception:
            raise RuntimeError(
                f"SCPI command failed — instrument may be in TSP mode. "
                f"Either switch to TSP in the GUI, or change command set "
                f"on the front panel: MENU → SETTINGS → COMMAND SET → SCPI.")
        self.connected = True

    def disconnect(self):
        try:
            if self._inst is not None:
                self._inst.write(f":OUTP{self._n} OFF")
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

    def configure_source_v(self, v_compliance: float, i_compliance: float,
                           use_4wire: bool, nplc: float):
        n = self._n
        cmds = [
            f":SOUR{n}:FUNC VOLT",
            f":SOUR{n}:VOLT:RANG:AUTO ON",
            f":SENS{n}:CURR:RANG:AUTO ON",
            f":SENS{n}:CURR:PROT {i_compliance:.6e}",
            f":SENS{n}:CURR:NPLC {nplc:.4f}",
            f":SYST:RSEN {1 if use_4wire else 0}",
            f":SOUR{n}:VOLT 0",
        ]
        for c in cmds:
            self._inst.write(c)

    def write_voltage(self, V: float):
        V = max(-SOFT_V_LIMIT, min(SOFT_V_LIMIT, float(V)))
        self._inst.write(f":SOUR{self._n}:VOLT {V:.6e}")

    def read_iv(self) -> Tuple[float, float]:
        # :READ? returns "V, I" in 2636B SCPI mode (NOTE: opposite order from TSP!)
        raw = self._inst.query(f":READ?{self._n}").strip()
        parts = [p for p in raw.replace(';', ',').split(',') if p.strip()]
        if len(parts) < 2:
            raise RuntimeError(f"SCPI READ? bad response: {raw!r}")
        try:
            V_meas = float(parts[0])
            I_meas = float(parts[1])
        except ValueError as e:
            raise RuntimeError(f"SCPI READ? parse failed: {raw!r}") from e
        return V_meas, I_meas

    def output_on(self):
        self._inst.write(f":OUTP{self._n} ON")

    def output_off(self):
        try:
            self._inst.write(f":SOUR{self._n}:VOLT 0")
            self._inst.write(f":OUTP{self._n} OFF")
        except Exception:
            pass

    def verify_communication(self) -> Dict[str, Any]:
        """Read-only SCPI queries. NEVER turns output on."""
        result: Dict[str, Any] = {'errors': [], 'protocol': 'scpi',
                                  'channel': self.config.channel}
        n = self._n

        def _try(key, query_str, postprocess=lambda x: x.strip()):
            try:
                raw = self._inst.query(query_str)
                result[key] = postprocess(raw)
            except Exception as e:
                result[key] = None
                result['errors'].append(f"{key}: {e}")

        _try('idn',                "*IDN?")
        _try('scpi_version',       ":SYST:VERS?")
        _try('output_state',       f":OUTP{n}?")
        _try('source_func',        f":SOUR{n}:FUNC?")
        _try('compliance_I_A_set', f":SENS{n}:CURR:PROT?",
             postprocess=lambda r: float(r.strip()))
        _try('measure_nplc',       f":SENS{n}:CURR:NPLC?",
             postprocess=lambda r: float(r.strip()))
        _try('sense_mode',         ":SYST:RSEN?",
             postprocess=lambda r: ('4-wire' if r.strip() == '1' else '2-wire'))
        return result

    def describe(self) -> str:
        return f"Keithley 2636B SCPI @ {self.config.visa_resource} (ch {self.config.channel})"


# ---------------------------------------------------------------------
# 2d.  Factory
# ---------------------------------------------------------------------

def make_keithley_controller(config: KeithleyConfig) -> KeithleyController:
    if DEMO_MODE:
        return DemoKeithley(config)
    if config.protocol == PROTO_TSP:
        return Keithley2636bTsp(config)
    if config.protocol == PROTO_SCPI:
        return Keithley2636bScpi(config)
    raise ValueError(f"Unknown protocol: {config.protocol!r}")


def _import_visa_lib_if_needed() -> None:
    """Called by main() iff --live, with helpful error if pyvisa missing."""
    if not HAS_PYVISA:
        raise RuntimeError(
            "--live mode requires pyvisa. Install with:\n"
            "    pip install pyvisa\n"
            "and ensure a VISA backend is installed (NI-VISA on Windows, "
            "or pyvisa-py + pyusb on Linux).")


# =====================================================================
# 3.  PointInfo  —  one IV measurement point passed via signal
# =====================================================================

@dataclass
class PointInfo:
    """One sweep point. Emitted from worker → GUI per measurement."""
    seq_idx:       int           # 0-based index in the voltage sequence
    V_set:         float         # commanded voltage [V]
    V_meas:        float         # measured voltage  [V]
    I_meas:        float         # measured current  [A]
    R_inst:        float         # V_meas / I_meas   [Ω], NaN if I=0
    timestamp_s:   float         # seconds since scan start
    in_compliance: bool          # True if SMU was in compliance for this point


# =====================================================================
# 4.  MeasurementThread  —  the IV sweep loop
# =====================================================================

class MeasurementThread(QThread):
    """Background thread: configure SMU → output ON → sweep voltages →
    read I/V at each point → CSV write → output OFF → emit fit results.

    Signals:
      point_ready(PointInfo)              — per measurement point
      log_msg(str, str)                   — log message + level
      sweep_finished(dict)                — emitted on success with fit results
      error_occurred(str)                 — emitted on exception with traceback
    """

    point_ready    = pyqtSignal(object)
    log_msg        = pyqtSignal(str, str)
    sweep_finished = pyqtSignal(object)   # dict with fit results
    error_occurred = pyqtSignal(str)

    def __init__(self, params: dict, controller: KeithleyController):
        super().__init__()
        self.params     = params
        self.controller = controller
        self.cfg:       SweepConfig    = params['sweep_config']
        self.kt_cfg:    KeithleyConfig = params['keithley_config']
        self.meta:      MetaConfig     = params['meta_config']
        self.is_running = True
        # Buffers for in-thread storage and post-fit
        self._V_meas_buf: List[float] = []
        self._I_meas_buf: List[float] = []
        self._compliance_count = 0
        self._scan_t0 = 0.0
        self._points_written = 0
        self._started_at_iso: Optional[str] = None

    def stop(self):
        self.is_running = False

    # -----------------------------------------------------------------
    @staticmethod
    def _linear_fit(V: List[float], I: List[float]) -> Dict[str, float]:
        """Least-squares linear fit I = V/R + I_offset.
        Returns dict with R_fit, I_offset, R_squared, sigma_R, n_used.
        Returns NaN values if fewer than 3 points or all-NaN.
        """
        Vn = np.asarray(V, dtype=float)
        In = np.asarray(I, dtype=float)
        mask = ~(np.isnan(Vn) | np.isnan(In) | np.isinf(Vn) | np.isinf(In))
        Vn, In = Vn[mask], In[mask]
        n = len(Vn)
        nan_result = {
            'R_fit_Ohm':  float('nan'),
            'I_offset_A': float('nan'),
            'R_squared':  float('nan'),
            'sigma_R_Ohm': float('nan'),
            'n_used':     int(n),
        }
        if n < 3:
            return nan_result
        # Need V variation
        if np.ptp(Vn) < 1e-15:
            return nan_result
        try:
            # I = slope*V + intercept; R = 1/slope
            slope, intercept = np.polyfit(Vn, In, 1)
            if abs(slope) < 1e-30:
                return nan_result
            R_fit = 1.0 / slope
            I_pred = slope * Vn + intercept
            ss_res = float(np.sum((In - I_pred) ** 2))
            ss_tot = float(np.sum((In - np.mean(In)) ** 2))
            R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
            # σ_slope from residuals (Gaussian assumption)
            if n > 2:
                resid_std = math.sqrt(ss_res / (n - 2))
                Vmean = float(np.mean(Vn))
                Sxx = float(np.sum((Vn - Vmean) ** 2))
                if Sxx > 0:
                    sigma_slope = resid_std / math.sqrt(Sxx)
                    # σ_R = |dR/dslope| × σ_slope = (1/slope²) × σ_slope
                    sigma_R = abs(sigma_slope / (slope * slope))
                else:
                    sigma_R = float('nan')
            else:
                sigma_R = float('nan')
            return {
                'R_fit_Ohm':   float(R_fit),
                'I_offset_A':  float(intercept),
                'R_squared':   float(R2),
                'sigma_R_Ohm': float(sigma_R),
                'n_used':      int(n),
            }
        except (np.linalg.LinAlgError, ValueError):
            return nan_result

    # -----------------------------------------------------------------
    def _write_csv_header(self, writer, save_path: str):
        cfg, kt_cfg, meta = self.cfg, self.kt_cfg, self.meta
        # Comment lines (CSV-safe — start with '#' so analysis can skip them)
        header_lines = [
            f"# {APP_NAME}  {APP_VERSION}",
            f"# started_at = {self._started_at_iso}",
            f"# sample = {meta.sample}",
            f"# device = {meta.device}",
            f"# operator = {meta.operator}",
            f"# run_name = {meta.run_name}",
            f"# electrode_pair = {meta.electrode_pair}",
            f"# protocol = {kt_cfg.protocol}",
            f"# visa_resource = {kt_cfg.visa_resource}",
            f"# channel = {kt_cfg.channel}",
            f"# wire_mode = {'4-wire' if kt_cfg.use_4wire else '2-wire'}",
            f"# i_compliance_A = {kt_cfg.i_compliance_A:.6e}",
            f"# v_compliance_V = {kt_cfg.v_compliance_V:.6e}",
            f"# v_max_V = {cfg.v_max_V:.6e}",
            f"# v_step_V = {cfg.v_step_V:.6e}",
            f"# nplc = {cfg.nplc:.4f}",
            f"# settle_s = {cfg.settle_s:.6e}",
            f"# sweep_mode = {cfg.mode}",
            f"# DEMO_MODE = {DEMO_MODE}",
        ]
        for line in header_lines:
            writer.writerow([line])
        writer.writerow(['seq_idx', 'V_set_V', 'V_meas_V', 'I_meas_A',
                         'R_inst_Ohm', 't_s', 'in_compliance'])

    # -----------------------------------------------------------------
    def _write_json_sidecar(self, save_path: str, fit_result: Dict[str, float]):
        sidecar = os.path.splitext(save_path)[0] + '.json'
        cfg, kt_cfg, meta = self.cfg, self.kt_cfg, self.meta
        meta_dict = {
            'schema_version':  'continuity_v1.0',
            'app_name':        APP_NAME,
            'app_version':     APP_VERSION,
            'started_at_iso':  self._started_at_iso,
            'finished_at_iso': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'demo_mode':       DEMO_MODE,
            'meta': {
                'sample':         meta.sample,
                'device':         meta.device,
                'operator':       meta.operator,
                'run_name':       meta.run_name,
                'electrode_pair': meta.electrode_pair,
            },
            'keithley': {
                'protocol':       kt_cfg.protocol,
                'visa_resource':  kt_cfg.visa_resource,
                'channel':        kt_cfg.channel,
                'wire_mode':      '4-wire' if kt_cfg.use_4wire else '2-wire',
                'i_compliance_A': kt_cfg.i_compliance_A,
                'v_compliance_V': kt_cfg.v_compliance_V,
            },
            'sweep': {
                'v_max_V':  cfg.v_max_V,
                'v_step_V': cfg.v_step_V,
                'nplc':     cfg.nplc,
                'settle_s': cfg.settle_s,
                'mode':     cfg.mode,
                'n_points': len(self._V_meas_buf),
            },
            'fit': fit_result,
            'compliance': {
                'n_points_in_compliance': self._compliance_count,
                'fraction':               (self._compliance_count /
                                           max(len(self._V_meas_buf), 1)),
            },
        }
        cleaned = _json_clean(meta_dict)
        with open(sidecar, 'w', encoding='utf-8') as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False, allow_nan=False)

    # -----------------------------------------------------------------
    def run(self):
        cfg, kt_cfg, meta = self.cfg, self.kt_cfg, self.meta
        ctrl = self.controller
        seq  = cfg.voltage_sequence()
        n_total = len(seq)
        self._started_at_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

        f = None
        writer = None
        save_path = cfg.save_path if cfg.save_enabled else ''

        try:
            self.log_msg.emit(
                f"Sweep started: sample={meta.sample!r} device={meta.device!r} "
                f"electrode_pair={meta.electrode_pair!r}", "info")
            self.log_msg.emit(
                f"Keithley: {ctrl.describe()}", "info")
            self.log_msg.emit(
                f"Sweep: mode={cfg.mode}, V_max={cfg.v_max_V*1000:.3f} mV, "
                f"V_step={cfg.v_step_V*1000:.3f} mV → {n_total} points", "info")
            self.log_msg.emit(
                f"NPLC={cfg.nplc:.2f} → integration ≈ {cfg.nplc/LINE_FREQ_HZ*1000:.1f} ms/pt; "
                f"compliance I = {fmt_eng(kt_cfg.i_compliance_A, 'A')}", "info")

            # Open CSV early so any failure later still leaves header
            if cfg.save_enabled and save_path:
                self.log_msg.emit(f"Saving to {save_path}", "info")
                f = open(save_path, 'w', newline='', encoding='utf-8')
                writer = csv.writer(f)
                self._write_csv_header(writer, save_path)
            else:
                self.log_msg.emit("Save disabled — data will not be written to disk.", "warning")

            # ---- Connect + configure ----
            ctrl.connect()
            self.log_msg.emit(f"Connected to {ctrl.describe()}", "success")

            ctrl.configure_source_v(
                v_compliance=kt_cfg.v_compliance_V,
                i_compliance=kt_cfg.i_compliance_A,
                use_4wire=kt_cfg.use_4wire,
                nplc=cfg.nplc,
            )
            self.log_msg.emit(
                f"Source configured: {'4-wire' if kt_cfg.use_4wire else '2-wire'} "
                f"voltage source", "info")

            # ---- Output ON ----
            ctrl.output_on()
            self.log_msg.emit("Output ON. Starting sweep.", "success")

            # ---- Sweep loop ----
            self._scan_t0 = time.monotonic()
            for idx, V_set in enumerate(seq):
                if not self.is_running:
                    self.log_msg.emit(
                        f"Sweep aborted by user at point {idx+1}/{n_total}.",
                        "warning")
                    break

                ctrl.write_voltage(V_set)
                if cfg.settle_s > 0:
                    time.sleep(cfg.settle_s)
                V_meas, I_meas = ctrl.read_iv()

                # In-compliance check (DemoKeithley sets _compliance_hit; for
                # real hw we infer by checking I_meas vs limit)
                in_comp = False
                if hasattr(ctrl, '_compliance_hit') and ctrl._compliance_hit:
                    in_comp = True
                    ctrl._compliance_hit = False  # reset for next point
                elif abs(I_meas) >= 0.95 * kt_cfg.i_compliance_A:
                    in_comp = True

                R_inst = (V_meas / I_meas) if abs(I_meas) > 1e-30 else float('nan')
                t_now  = time.monotonic() - self._scan_t0

                if in_comp:
                    self._compliance_count += 1

                self._V_meas_buf.append(V_meas)
                self._I_meas_buf.append(I_meas)

                point = PointInfo(
                    seq_idx=idx, V_set=V_set, V_meas=V_meas, I_meas=I_meas,
                    R_inst=R_inst, timestamp_s=t_now, in_compliance=in_comp)
                self.point_ready.emit(point)

                # CSV write
                if writer is not None:
                    writer.writerow([
                        idx, f"{V_set:.6e}", f"{V_meas:.6e}", f"{I_meas:.6e}",
                        f"{R_inst:.6e}" if not math.isnan(R_inst) else 'nan',
                        f"{t_now:.4f}",
                        '1' if in_comp else '0',
                    ])
                    self._points_written += 1
                    if self._points_written % CSV_FSYNC_EVERY_N == 0:
                        try:
                            f.flush()
                            os.fsync(f.fileno())
                        except Exception:
                            pass

            # ---- Output OFF ----
            ctrl.output_off()
            if writer is not None:
                f.flush()
            self.log_msg.emit("Output OFF. Sweep complete.", "success")

            # ---- Linear fit on all collected points ----
            fit = self._linear_fit(self._V_meas_buf, self._I_meas_buf)
            R_str = (fmt_eng(fit['R_fit_Ohm'], 'Ω')
                     if not math.isnan(fit['R_fit_Ohm']) else '— Ω')
            sigma_str = (fmt_eng(fit['sigma_R_Ohm'], 'Ω')
                         if not math.isnan(fit['sigma_R_Ohm']) else '?')
            R2_str = (f"{fit['R_squared']:.6f}"
                      if not math.isnan(fit['R_squared']) else '—')
            offset_str = fmt_eng(fit['I_offset_A'], 'A')
            self.log_msg.emit(
                f"Linear fit:  R = {R_str} ± {sigma_str}  (R² = {R2_str}, "
                f"I_offset = {offset_str}, n = {fit['n_used']})", "success")
            if self._compliance_count > 0:
                self.log_msg.emit(
                    f"⚠ {self._compliance_count}/{len(self._V_meas_buf)} "
                    f"points were in current compliance — sample resistance "
                    f"is too low for the chosen V_max. Reduce V_max or "
                    f"raise compliance.", "warning")

            # ---- JSON sidecar (only if we saved CSV) ----
            if cfg.save_enabled and save_path:
                try:
                    self._write_json_sidecar(save_path, fit)
                    self.log_msg.emit(
                        f"JSON sidecar written: "
                        f"{os.path.splitext(save_path)[0]}.json", "info")
                except Exception as e:
                    self.log_msg.emit(f"JSON sidecar write failed: {e}", "error")

            self.sweep_finished.emit(fit)

        except Exception:
            tb = traceback.format_exc()
            self.log_msg.emit(f"ERROR: {tb.splitlines()[-1]}", "error")
            self.error_occurred.emit(tb)
            # Best-effort: turn output off on error
            try:
                ctrl.output_off()
            except Exception:
                pass

        finally:
            try:
                ctrl.disconnect()
            except Exception:
                pass
            if f is not None:
                try:
                    f.flush()
                    f.close()
                except Exception:
                    pass


# =====================================================================
# 5.  Catppuccin QSS stylesheet  —  same look as hysteresis_mapping
# =====================================================================

def make_stylesheet() -> str:
    return f"""
    QMainWindow, QWidget {{
        background-color: {CT_BASE};
        color: {CT_TEXT};
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-size: 10pt;
    }}
    QGroupBox {{
        background-color: {CT_MANTLE};
        border: 1px solid {CT_SURFACE1};
        border-radius: 6px;
        margin-top: 14px;
        padding-top: 6px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        color: {CT_LAVENDER};
    }}
    QLabel {{ color: {CT_TEXT}; }}
    QLabel#sectionLabel {{ color: {CT_SUBTEXT0}; font-weight: 500; }}
    QLabel#valueLabel {{
        color: {CT_TEAL};
        font-family: 'JetBrains Mono', 'Consolas', monospace;
    }}
    QLineEdit, QComboBox {{
        background-color: {CT_SURFACE0};
        border: 1px solid {CT_SURFACE1};
        border-radius: 4px;
        padding: 4px 6px;
        color: {CT_TEXT};
        selection-background-color: {CT_BLUE};
        selection-color: {CT_BASE};
    }}
    QLineEdit:focus, QComboBox:focus {{
        border: 1px solid {CT_BLUE};
    }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QComboBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {CT_TEXT};
        margin-right: 6px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {CT_SURFACE0};
        color: {CT_TEXT};
        selection-background-color: {CT_BLUE};
        selection-color: {CT_BASE};
        border: 1px solid {CT_SURFACE2};
    }}
    QPushButton {{
        background-color: {CT_SURFACE0};
        border: 1px solid {CT_SURFACE1};
        border-radius: 4px;
        padding: 6px 14px;
        color: {CT_TEXT};
    }}
    QPushButton:hover {{ background-color: {CT_SURFACE1}; }}
    QPushButton:pressed {{ background-color: {CT_SURFACE2}; }}
    QPushButton:disabled {{ color: {CT_OVERLAY1}; background-color: {CT_MANTLE}; }}
    QPushButton#startBtn {{
        background-color: {CT_GREEN};
        color: {CT_BASE};
        font-weight: 700;
    }}
    QPushButton#startBtn:hover {{ background-color: #b9efb6; }}
    QPushButton#stopBtn {{
        background-color: {CT_RED};
        color: {CT_BASE};
        font-weight: 700;
    }}
    QPushButton#stopBtn:hover {{ background-color: #f6a3b8; }}
    QPushButton#verifyBtn {{
        background-color: {CT_MAUVE};
        color: {CT_BASE};
        font-weight: 600;
    }}
    QCheckBox {{ color: {CT_TEXT}; spacing: 6px; }}
    QCheckBox::indicator {{
        width: 14px; height: 14px;
        border: 1px solid {CT_SURFACE2};
        border-radius: 3px;
        background-color: {CT_SURFACE0};
    }}
    QCheckBox::indicator:checked {{
        background-color: {CT_GREEN};
        border: 1px solid {CT_GREEN};
    }}
    QRadioButton {{ color: {CT_TEXT}; spacing: 6px; }}
    QRadioButton::indicator {{
        width: 14px; height: 14px;
        border: 1px solid {CT_SURFACE2};
        border-radius: 7px;
        background-color: {CT_SURFACE0};
    }}
    QRadioButton::indicator:checked {{
        background-color: {CT_GREEN};
        border: 1px solid {CT_GREEN};
    }}
    QPlainTextEdit {{
        background-color: {CT_CRUST};
        color: {CT_TEXT};
        border: 1px solid {CT_SURFACE1};
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: 9pt;
    }}
    QProgressBar {{
        background-color: {CT_SURFACE0};
        border: 1px solid {CT_SURFACE1};
        border-radius: 4px;
        text-align: center;
        color: {CT_TEXT};
    }}
    QProgressBar::chunk {{
        background-color: {CT_BLUE};
        border-radius: 3px;
    }}
    QStatusBar {{
        background-color: {CT_MANTLE};
        color: {CT_TEXT};
    }}
    QScrollArea {{ border: none; background-color: {CT_BASE}; }}
    QScrollBar:vertical {{
        background-color: {CT_MANTLE};
        width: 10px;
    }}
    QScrollBar::handle:vertical {{
        background-color: {CT_SURFACE1};
        border-radius: 5px;
        min-height: 30px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
    """


# =====================================================================
# 6.  Custom widget — wheel-disabled combo (same as hysteresis_mapping)
# =====================================================================

class NoWheelComboBox(QComboBox):
    """ComboBox that ignores wheel events so scrolling the form panel
    doesn't accidentally change values."""
    def wheelEvent(self, e):
        e.ignore()


# =====================================================================
# 7.  KeithleyIVGui — main window
# =====================================================================

class KeithleyIVGui(QMainWindow):
    """Catppuccin-themed GUI for Keithley 2636B IV / continuity sweeps.

    Layout (single page, two columns):

      ┌─────────────────────────────────────────────────────────────┐
      │  metadata bar (sample + device + operator + run + electrode)│
      ├──────────────────────┬──────────────────────────────────────┤
      │  left:               │  right:                              │
      │   · Keithley group   │   ┌─ I(V) curve ──────────────────┐  │
      │   · Sweep group      │   │ blue=fwd, peach=bwd, fit line │  │
      │   · Output group     │   └────────────────────────────────┘ │
      │   · [Verify Keithley]│   ┌─ R(V) curve ──────────────────┐  │
      │   · [START] [ABORT]  │   │ R = V_meas/I_meas per point   │  │
      │                      │   └────────────────────────────────┘ │
      │                      │   Status:                            │
      │                      │     R_fit, sigma, R², I_offset       │
      │                      │     compliance flag                  │
      │                      │     mode badge                       │
      ├──────────────────────┴──────────────────────────────────────┤
      │  Log panel (scrolling Catppuccin)                           │
      │  Progress bar                                               │
      └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self):
        super().__init__()
        self.setStyleSheet(make_stylesheet())
        self.setWindowTitle(
            f"{APP_NAME} {APP_VERSION}  [{'DEMO' if DEMO_MODE else 'LIVE'}]")
        self.resize(1300, 800)

        # Runtime state
        self.thread: Optional[MeasurementThread] = None
        self.controller: Optional[KeithleyController] = None
        self._keithley_verified: bool = False
        # Plot data buffers (grow during sweep)
        self._V_buf: List[float] = []
        self._I_buf: List[float] = []
        self._R_buf: List[float] = []

        self._build_ui()
        self._init_menubar()
        self._init_statusbar()
        self._load_settings()
        self._refresh_filename()
        self._update_estimated_time()

        # Initial log
        self.log_event("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")
        self.log_event(f"{APP_NAME}  ·  {APP_VERSION}", "success")
        self.log_event("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")
        if DEMO_MODE:
            self.log_event("DEMO MODE — Keithley is simulated.", "warning")
        else:
            self.log_event("LIVE MODE — using real Keithley 2636B.", "success")

    # ─────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        outer   = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        outer.addWidget(self._build_metadata_bar())

        # Main split: left form + right plots/results
        self.h_split = QSplitter(Qt.Horizontal)
        self.h_split.addWidget(self._build_left_panel())
        self.h_split.addWidget(self._build_right_panel())
        self.h_split.setStretchFactor(0, 0)
        self.h_split.setStretchFactor(1, 1)
        self.h_split.setSizes([380, 920])
        outer.addWidget(self.h_split, 1)

        # Bottom: log + progress
        outer.addWidget(self._build_log_area())

        self.setCentralWidget(central)

    # ---- Metadata bar (top strip) ----
    def _build_metadata_bar(self):
        gb = QGroupBox()
        gb.setFlat(True)
        h = QHBoxLayout(gb)
        h.setContentsMargins(6, 2, 6, 2)
        h.setSpacing(6)

        def _le(label, default='', width=120, tip=''):
            l = QLabel(label); l.setObjectName("sectionLabel")
            le = QLineEdit(default)
            le.setMaximumWidth(width)
            if tip: le.setToolTip(tip); l.setToolTip(tip)
            h.addWidget(l); h.addWidget(le)
            return le

        self.le_sample         = _le("Sample",   '',
            tip="Sample identifier (e.g. wafer / flake name).")
        self.le_device         = _le("Device",   '',
            tip="Device identifier on this sample.")
        self.le_operator       = _le("Operator", '', width=80,
            tip="Your initials or name (recorded in metadata).")
        self.le_run_name       = _le("Run",      'cont1', width=80,
            tip="Short identifier for this sweep.")
        self.le_electrode_pair = _le("Electrodes", '', width=80,
            tip="Which electrodes are connected, e.g. 'A1-B3'. "
                "Recorded in filename and metadata for identification.")

        # Refresh filename when any of these changes
        for w in (self.le_sample, self.le_device, self.le_run_name, self.le_electrode_pair):
            w.textChanged.connect(self._refresh_filename)

        h.addStretch()
        # Date/time clock
        self.lbl_clock = QLabel("—")
        self.lbl_clock.setObjectName("valueLabel")
        h.addWidget(self.lbl_clock)
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)
        self._tick_clock()

        return gb

    def _tick_clock(self):
        self.lbl_clock.setText(time.strftime("%Y-%m-%d  %H:%M:%S"))

    # ---- Left panel (config form) ----
    def _build_left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        v.addWidget(self._build_keithley_group())
        v.addWidget(self._build_sweep_group())
        v.addWidget(self._build_output_group())
        v.addStretch()
        v.addLayout(self._build_button_row())

        scroll.setWidget(inner)
        scroll.setMinimumWidth(360)
        return scroll

    def _build_keithley_group(self):
        gb = QGroupBox("Keithley 2636B")
        g = QGridLayout(); g.setContentsMargins(8, 4, 8, 8)
        g.setHorizontalSpacing(8); g.setVerticalSpacing(6)

        l = QLabel("Protocol"); l.setObjectName("sectionLabel")
        l.setToolTip("TSP (Lua, factory default) or SCPI. Must match the\n"
                     "instrument's command-set setting on the front panel.")
        g.addWidget(l, 0, 0)
        self.cb_protocol = NoWheelComboBox()
        for p in PROTOCOLS:
            self.cb_protocol.addItem(PROTO_LABEL[p], userData=p)
        g.addWidget(self.cb_protocol, 0, 1, 1, 2)

        l = QLabel("VISA resource"); l.setObjectName("sectionLabel")
        l.setToolTip("VISA resource string, e.g. 'GPIB0::26::INSTR' or\n"
                     "'TCPIP::192.168.1.50::INSTR' or 'USB0::0x05E6::0x2636::...'.")
        g.addWidget(l, 1, 0)
        self.le_visa = QLineEdit("GPIB0::26::INSTR")
        g.addWidget(self.le_visa, 1, 1, 1, 2)

        l = QLabel("Channel"); l.setObjectName("sectionLabel")
        l.setToolTip("Which SMU channel to use. Channel A and B are\n"
                     "independent SMUs. Pick A unless your wiring is on B.")
        g.addWidget(l, 2, 0)
        self.cb_channel = NoWheelComboBox()
        for ch in CHANNELS:
            self.cb_channel.addItem(f"Channel {ch}", userData=ch)
        g.addWidget(self.cb_channel, 2, 1, 1, 2)

        # 4-wire checkbox
        self.cb_4wire = QCheckBox("4-wire (Kelvin) measurement")
        self.cb_4wire.setToolTip(
            "Uncheck for 2-wire (default — typical for pre-wirebond\n"
            "continuity tests with two probes). Check for 4-wire if the\n"
            "sample is wire-bonded with separate sense and force lines.")
        g.addWidget(self.cb_4wire, 3, 0, 1, 3)

        # Compliance current
        l = QLabel("I compliance"); l.setObjectName("sectionLabel")
        l.setToolTip(
            "Current limit. The SMU stops sourcing more current when this\n"
            "is reached, instead clamping the voltage. Values to use:\n"
            "  · 100 nA  — fragile devices, gate leakage tests\n"
            "  · 1 µA    — typical low-T sample (DEFAULT)\n"
            "  · 10 µA   — robust transport sample\n"
            "  · 1 mA    — wirebond / metal contact verification\n"
            "Format: scientific notation (e.g. 1e-6).")
        g.addWidget(l, 4, 0)
        self.le_i_comp = QLineEdit("1e-6")
        g.addWidget(self.le_i_comp, 4, 1)
        l = QLabel("A"); g.addWidget(l, 4, 2)

        # Verify button
        self.btn_verify = QPushButton("Verify Keithley comms")
        self.btn_verify.setObjectName("verifyBtn")
        self.btn_verify.setToolTip(
            "Run a read-only query against the Keithley and display the\n"
            "results. Use this BEFORE the first scan of the session to\n"
            "confirm GPIB/USB/LAN works and the protocol matches the\n"
            "instrument's mode. Required for LIVE-mode high-V sweeps.")
        self.btn_verify.clicked.connect(self._on_verify_clicked)
        g.addWidget(self.btn_verify, 5, 0, 1, 3)

        gb.setLayout(g)
        return gb

    def _build_sweep_group(self):
        gb = QGroupBox("Sweep")
        g = QGridLayout(); g.setContentsMargins(8, 4, 8, 8)
        g.setHorizontalSpacing(8); g.setVerticalSpacing(6)

        l = QLabel("V_max"); l.setObjectName("sectionLabel")
        l.setToolTip(
            "Maximum |V| in the sweep, in VOLTS.\n"
            "For continuity tests on low-T samples, 5 mV (= 0.005) is\n"
            "typical — small enough to not perturb anything, large enough\n"
            "for a clean linear fit.")
        g.addWidget(l, 0, 0)
        self.le_v_max = QLineEdit("0.005")
        g.addWidget(self.le_v_max, 0, 1)
        l = QLabel("V"); g.addWidget(l, 0, 2)

        l = QLabel("V_step"); l.setObjectName("sectionLabel")
        l.setToolTip(
            "Voltage step size in VOLTS. Total points per half-sweep\n"
            "(0 → V_max) ≈ V_max/V_step + 1.\n"
            "1 mV step gives 6 points for V_max=5mV — enough for a fit.")
        g.addWidget(l, 1, 0)
        self.le_v_step = QLineEdit("0.001")
        g.addWidget(self.le_v_step, 1, 1)
        l = QLabel("V"); g.addWidget(l, 1, 2)

        l = QLabel("NPLC"); l.setObjectName("sectionLabel")
        l.setToolTip(
            "Integration time in number-of-power-line-cycles.\n"
            "  · 0.1 = 2 ms @ 50 Hz  (fast, noisy)\n"
            "  · 1.0 = 20 ms @ 50 Hz (DEFAULT, typical)\n"
            "  · 10  = 200 ms @ 50 Hz (slow, low noise)\n"
            "Higher NPLC → lower noise but slower sweep.")
        g.addWidget(l, 2, 0)
        self.le_nplc = QLineEdit("1.0")
        g.addWidget(self.le_nplc, 2, 1)

        l = QLabel("Settle"); l.setObjectName("sectionLabel")
        l.setToolTip(
            "Time to wait between writing the new V_set and reading I, V.\n"
            "Lets the SMU output and sample RC settle. Default 5 ms is\n"
            "ample for normal continuity tests.")
        g.addWidget(l, 3, 0)
        self.le_settle = QLineEdit("0.005")
        g.addWidget(self.le_settle, 3, 1)
        l = QLabel("s"); g.addWidget(l, 3, 2)

        # Sweep mode radio buttons
        l = QLabel("Sweep mode"); l.setObjectName("sectionLabel")
        l.setToolTip(
            "Forward only:    0 → +V_max  (single-shot, asymmetric)\n"
            "Backward only:   +V_max → 0  (reverse direction)\n"
            "Round trip:      0 → +V_max → 0 → -V_max → 0  (DEFAULT,\n"
            "                  shows linearity, symmetry, and any hysteresis)")
        g.addWidget(l, 4, 0, 1, 3)
        self.rb_group = QButtonGroup(self)
        self.rb_modes: Dict[str, QRadioButton] = {}
        for i, m in enumerate(SWEEP_MODES):
            rb = QRadioButton(SWEEP_LABEL[m])
            self.rb_modes[m] = rb
            self.rb_group.addButton(rb, i)
            g.addWidget(rb, 5 + i, 0, 1, 3)
        self.rb_modes[SWEEP_ROUND_TRIP].setChecked(True)
        for rb in self.rb_modes.values():
            rb.toggled.connect(self._update_estimated_time)

        # Estimated time
        l = QLabel("Estimated time"); l.setObjectName("sectionLabel")
        g.addWidget(l, 8, 0)
        self.lbl_est_time = QLabel("—")
        self.lbl_est_time.setObjectName("valueLabel")
        g.addWidget(self.lbl_est_time, 8, 1, 1, 2)

        for w in (self.le_v_max, self.le_v_step, self.le_nplc, self.le_settle):
            w.textChanged.connect(self._update_estimated_time)

        gb.setLayout(g)
        return gb

    def _build_output_group(self):
        gb = QGroupBox("Output")
        g = QGridLayout(); g.setContentsMargins(8, 4, 8, 8)
        g.setHorizontalSpacing(8); g.setVerticalSpacing(6)

        # Save enable checkbox
        self.cb_save = QCheckBox("Save data (CSV + JSON sidecar)")
        self.cb_save.setChecked(True)
        self.cb_save.setToolTip(
            "If unchecked, the sweep runs and is plotted but no files are\n"
            "written. Useful for quick exploratory measurements.")
        self.cb_save.stateChanged.connect(self._on_save_toggled)
        g.addWidget(self.cb_save, 0, 0, 1, 3)

        l = QLabel("Folder"); g.addWidget(l, 1, 0)
        self.le_folder = QLineEdit(os.path.expanduser("~"))
        g.addWidget(self.le_folder, 1, 1)
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.clicked.connect(self._browse_folder)
        g.addWidget(self.btn_browse, 1, 2)

        l = QLabel("Filename"); g.addWidget(l, 2, 0)
        self.lbl_filename = QLabel("—")
        self.lbl_filename.setObjectName("valueLabel")
        self.lbl_filename.setWordWrap(True)
        g.addWidget(self.lbl_filename, 2, 1, 1, 2)

        gb.setLayout(g)
        return gb

    def _on_save_toggled(self, state):
        on = bool(state)
        self.le_folder.setEnabled(on)
        self.btn_browse.setEnabled(on)

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder",
                                             self.le_folder.text())
        if d:
            self.le_folder.setText(d)
            self._refresh_filename()

    def _refresh_filename(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        sample = (self.le_sample.text() or 'sample').strip() or 'sample'
        device = (self.le_device.text() or 'dev').strip() or 'dev'
        run    = (self.le_run_name.text() or 'cont1').strip() or 'cont1'
        epair  = (self.le_electrode_pair.text() or 'pair').strip() or 'pair'
        fname  = f"{ts}__{sample}__{device}__{epair}__{run}__cont.csv"
        for c in '<>:"|?*\\/':
            fname = fname.replace(c, '_')
        self.lbl_filename.setText(fname)

    def _update_estimated_time(self):
        try:
            v_max = abs(float(self.le_v_max.text()))
            v_step = abs(float(self.le_v_step.text()))
            nplc   = float(self.le_nplc.text())
            settle = float(self.le_settle.text())
            if v_step <= 0 or v_max <= 0 or nplc <= 0:
                raise ValueError
        except Exception:
            self.lbl_est_time.setText("(invalid)")
            return
        n_per_seg = int(round(v_max / v_step)) + 1
        # Determine number of segments by mode
        mode = self._current_mode()
        if mode == SWEEP_ROUND_TRIP:
            n_total = 4 * (n_per_seg - 1) + 1
        else:
            n_total = n_per_seg
        per_pt = settle + nplc / LINE_FREQ_HZ + 0.005   # +5ms VISA roundtrip estimate
        total = n_total * per_pt
        self.lbl_est_time.setText(f"{n_total} pts × ~{per_pt*1000:.0f} ms ≈ {fmt_dur(total)}")

    def _current_mode(self) -> str:
        for m, rb in self.rb_modes.items():
            if rb.isChecked():
                return m
        return SWEEP_ROUND_TRIP

    def _build_button_row(self):
        h = QHBoxLayout()
        self.btn_start = QPushButton("START")
        self.btn_start.setObjectName("startBtn")
        self.btn_start.clicked.connect(self.start_sweep)
        h.addWidget(self.btn_start)

        self.btn_stop = QPushButton("ABORT")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.clicked.connect(self.stop_sweep)
        self.btn_stop.setEnabled(False)
        h.addWidget(self.btn_stop)
        return h

    # ---- Right panel (plots + results) ----
    def _build_right_panel(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)

        # I(V) plot
        gb_iv = QGroupBox("I(V) curve")
        v_iv = QVBoxLayout(gb_iv); v_iv.setContentsMargins(4, 4, 4, 4)
        self.pw_iv = pg.PlotWidget()
        self.pw_iv.setBackground(CT_BASE)
        self.pw_iv.showGrid(x=True, y=True, alpha=0.25)
        self.pw_iv.setLabel('bottom', 'V_meas (V)', color=CT_TEXT)
        self.pw_iv.setLabel('left',   'I_meas (A)', color=CT_TEXT)
        self.pw_iv.getAxis('bottom').setPen(CT_SUBTEXT1)
        self.pw_iv.getAxis('left').setPen(CT_SUBTEXT1)
        self.pw_iv.getAxis('bottom').setTextPen(CT_TEXT)
        self.pw_iv.getAxis('left').setTextPen(CT_TEXT)
        self.legend_iv = self.pw_iv.addLegend(offset=(10, 5))
        # Data line + fit line
        self.line_iv = self.pw_iv.plot(
            [], [], pen=pg.mkPen(CT_BLUE, width=2),
            symbol='o', symbolSize=5,
            symbolBrush=CT_BLUE, symbolPen=None,
            name='Measured')
        self.line_iv_fit = self.pw_iv.plot(
            [], [], pen=pg.mkPen(CT_PEACH, width=1.5, style=Qt.DashLine),
            name='Linear fit')
        v_iv.addWidget(self.pw_iv, 1)
        v.addWidget(gb_iv, 1)

        # R(V) plot
        gb_rv = QGroupBox("R(V) curve  (R_inst = V_meas / I_meas)")
        v_rv = QVBoxLayout(gb_rv); v_rv.setContentsMargins(4, 4, 4, 4)
        self.pw_rv = pg.PlotWidget()
        self.pw_rv.setBackground(CT_BASE)
        self.pw_rv.showGrid(x=True, y=True, alpha=0.25)
        self.pw_rv.setLabel('bottom', 'V_meas (V)', color=CT_TEXT)
        self.pw_rv.setLabel('left',   'R_inst (Ω)', color=CT_TEXT)
        self.pw_rv.getAxis('bottom').setPen(CT_SUBTEXT1)
        self.pw_rv.getAxis('left').setPen(CT_SUBTEXT1)
        self.pw_rv.getAxis('bottom').setTextPen(CT_TEXT)
        self.pw_rv.getAxis('left').setTextPen(CT_TEXT)
        self.line_rv = self.pw_rv.plot(
            [], [], pen=pg.mkPen(CT_TEAL, width=2),
            symbol='s', symbolSize=5,
            symbolBrush=CT_TEAL, symbolPen=None)
        v_rv.addWidget(self.pw_rv, 1)
        v.addWidget(gb_rv, 1)

        # Results bar (lives below the two plots)
        gb_res = QGroupBox("Results")
        g = QGridLayout(gb_res); g.setContentsMargins(8, 4, 8, 8)
        g.setHorizontalSpacing(12); g.setVerticalSpacing(4)

        l = QLabel("R_fit:"); l.setObjectName("sectionLabel"); g.addWidget(l, 0, 0)
        self.lbl_R_fit = QLabel("—")
        self.lbl_R_fit.setObjectName("valueLabel")
        self.lbl_R_fit.setStyleSheet(
            f"color:{CT_GREEN}; font-size: 12pt; font-weight: 700;")
        g.addWidget(self.lbl_R_fit, 0, 1)

        l = QLabel("R²:"); l.setObjectName("sectionLabel"); g.addWidget(l, 0, 2)
        self.lbl_R2 = QLabel("—")
        self.lbl_R2.setObjectName("valueLabel")
        g.addWidget(self.lbl_R2, 0, 3)

        l = QLabel("I_offset:"); l.setObjectName("sectionLabel"); g.addWidget(l, 1, 0)
        self.lbl_I_offset = QLabel("—")
        self.lbl_I_offset.setObjectName("valueLabel")
        g.addWidget(self.lbl_I_offset, 1, 1)

        l = QLabel("Compliance:"); l.setObjectName("sectionLabel"); g.addWidget(l, 1, 2)
        self.lbl_compliance = QLabel("—")
        self.lbl_compliance.setObjectName("valueLabel")
        g.addWidget(self.lbl_compliance, 1, 3)

        v.addWidget(gb_res)
        return w

    # ---- Bottom: log area + progress ----
    def _build_log_area(self):
        gb = QGroupBox("Log")
        v = QVBoxLayout(gb); v.setContentsMargins(4, 4, 4, 4); v.setSpacing(4)

        # Progress + per-point label
        h = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        h.addWidget(self.progress_bar, 1)
        self.lbl_now = QLabel("—")
        self.lbl_now.setObjectName("valueLabel")
        self.lbl_now.setMinimumWidth(280)
        h.addWidget(self.lbl_now)
        v.addLayout(h)

        self.te_log = QPlainTextEdit()
        self.te_log.setReadOnly(True)
        self.te_log.setMinimumHeight(120)
        self.te_log.setMaximumBlockCount(2000)
        v.addWidget(self.te_log, 1)

        return gb

    # ---- Menu bar ----
    def _init_menubar(self):
        mb = self.menuBar()
        m_file = mb.addMenu("&File")
        self.act_start = QAction("&Start sweep", self)
        self.act_start.setShortcut(QKeySequence("Ctrl+R"))
        self.act_start.triggered.connect(self.start_sweep)
        m_file.addAction(self.act_start)
        self.act_stop = QAction("&Abort sweep", self)
        self.act_stop.setShortcut(QKeySequence("Ctrl+."))
        self.act_stop.triggered.connect(self.stop_sweep)
        self.act_stop.setEnabled(False)
        m_file.addAction(self.act_stop)
        m_file.addSeparator()
        a_quit = QAction("&Quit", self); a_quit.setShortcut(QKeySequence.Quit)
        a_quit.triggered.connect(self.close)
        m_file.addAction(a_quit)

    # ---- Status bar ----
    def _init_statusbar(self):
        sb = self.statusBar()
        # Mode badge
        self.lbl_mode_badge = QLabel(f"  [{'DEMO' if DEMO_MODE else 'LIVE'}]  ")
        badge_bg = CT_PEACH if DEMO_MODE else CT_GREEN
        self.lbl_mode_badge.setStyleSheet(
            f"color:{CT_BASE}; background-color:{badge_bg}; "
            f"font-weight: 700; padding: 1px 8px; border-radius: 4px; "
            f"font-family: 'JetBrains Mono', monospace;")
        sb.addPermanentWidget(self.lbl_mode_badge)
        # State indicator
        self.lbl_state = QLabel("Idle")
        self.lbl_state.setStyleSheet(
            f"color:{CT_GREEN}; font-family: 'JetBrains Mono', monospace; "
            f"padding: 0 8px;")
        sb.addPermanentWidget(self.lbl_state)

    def _set_state(self, text: str, color: str):
        self.lbl_state.setText(text)
        self.lbl_state.setStyleSheet(
            f"color:{color}; font-family: 'JetBrains Mono', monospace; "
            f"padding: 0 8px;")

    # ─────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────
    def log_event(self, msg: str, level: str = 'info'):
        ts = time.strftime("%H:%M:%S")
        color = {
            'info':    CT_TEXT,
            'success': CT_GREEN,
            'warning': CT_YELLOW,
            'error':   CT_RED,
        }.get(level, CT_TEXT)
        html = (f'<span style="color:{CT_OVERLAY1}">[{ts}]</span> '
                f'<span style="color:{color}">{msg}</span>')
        self.te_log.appendHtml(html)
        self.te_log.verticalScrollBar().setValue(
            self.te_log.verticalScrollBar().maximum())

    # ─────────────────────────────────────────────────────────────────
    # Verify Keithley
    # ─────────────────────────────────────────────────────────────────
    def _on_verify_clicked(self):
        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "Sweep running",
                "Cannot verify Keithley during a sweep — the bus is busy.")
            return
        try:
            kt_cfg = self._parse_keithley_config()
        except ValueError as e:
            QMessageBox.critical(self, "Invalid Keithley config", str(e))
            return
        ctrl = make_keithley_controller(kt_cfg)
        self.log_event("Verifying Keithley communication...", "info")
        try:
            ctrl.connect()
        except Exception as e:
            self.log_event(f"Keithley connect FAILED: {e}", "error")
            QMessageBox.critical(self, "Connect failed", str(e))
            return
        try:
            result = ctrl.verify_communication()
        except Exception as e:
            self.log_event(f"verify_communication FAILED: {e}", "error")
            QMessageBox.critical(self, "Verify failed", str(e))
            try: ctrl.disconnect()
            except Exception: pass
            return
        finally:
            try: ctrl.disconnect()
            except Exception: pass

        # Display
        lines = ["Keithley verification results:", ""]
        for k, val in result.items():
            if k == 'errors':
                continue
            lines.append(f"  {k}:  {val}")
        if result.get('errors'):
            lines.append("")
            lines.append("Errors during verification:")
            for e in result['errors']:
                lines.append(f"  - {e}")
            lines.append("")
            lines.append("Some queries failed. Do NOT trust this controller "
                         "for production until errors are fixed.")
            self._keithley_verified = False
            level = "warning"
        else:
            lines.append("")
            lines.append("All queries succeeded.")
            lines.append("⚠  Compare these values to the instrument's front panel.")
            lines.append("If they match, you may proceed with sweeps.")
            self._keithley_verified = True
            level = "success"

        self.log_event("Keithley verify completed " +
                       ("(no errors)." if not result.get('errors') else "(with errors)."),
                       level)
        QMessageBox.information(self, "Keithley verification", "\n".join(lines))

    # ─────────────────────────────────────────────────────────────────
    # Parse params
    # ─────────────────────────────────────────────────────────────────
    def _parse_keithley_config(self) -> KeithleyConfig:
        proto = self.cb_protocol.currentData() or PROTO_TSP
        visa = self.le_visa.text().strip()
        if not visa:
            raise ValueError("VISA resource is empty.")
        ch = self.cb_channel.currentData() or DEFAULT_CHANNEL
        try:
            i_comp = float(self.le_i_comp.text())
        except ValueError:
            raise ValueError(
                f"Could not parse compliance current {self.le_i_comp.text()!r}.")
        if i_comp <= 0:
            raise ValueError("Compliance current must be > 0.")
        if i_comp > SOFT_I_LIMIT:
            raise ValueError(
                f"Compliance current {i_comp} A exceeds soft limit "
                f"{SOFT_I_LIMIT} A. This is dangerous — refusing.")
        return KeithleyConfig(
            protocol=proto,
            visa_resource=visa,
            channel=ch,
            use_4wire=self.cb_4wire.isChecked(),
            v_compliance_V=SOFT_V_LIMIT,
            i_compliance_A=i_comp,
        )

    def _parse_sweep_config(self) -> SweepConfig:
        try:
            v_max  = float(self.le_v_max.text())
            v_step = float(self.le_v_step.text())
            nplc   = float(self.le_nplc.text())
            settle = float(self.le_settle.text())
        except ValueError:
            raise ValueError("Could not parse sweep numeric fields.")
        if v_max <= 0:
            raise ValueError("V_max must be > 0.")
        if v_max > SOFT_V_LIMIT:
            raise ValueError(
                f"V_max {v_max} V exceeds soft limit {SOFT_V_LIMIT} V. "
                f"This is dangerous for typical samples — refusing.")
        if v_step <= 0:
            raise ValueError("V_step must be > 0.")
        if v_step > v_max:
            raise ValueError("V_step must be ≤ V_max.")
        if nplc <= 0 or nplc > 50:
            raise ValueError("NPLC must be in (0, 50].")
        if settle < 0:
            raise ValueError("Settle time must be ≥ 0.")
        return SweepConfig(
            v_max_V=v_max, v_step_V=v_step, nplc=nplc, settle_s=settle,
            mode=self._current_mode(),
            save_enabled=self.cb_save.isChecked(),
            save_path='',   # filled by start_sweep
        )

    def _parse_meta_config(self) -> MetaConfig:
        return MetaConfig(
            sample=self.le_sample.text().strip(),
            device=self.le_device.text().strip(),
            operator=self.le_operator.text().strip(),
            run_name=self.le_run_name.text().strip(),
            electrode_pair=self.le_electrode_pair.text().strip(),
        )

    # ─────────────────────────────────────────────────────────────────
    # Start / Stop
    # ─────────────────────────────────────────────────────────────────
    def start_sweep(self):
        if self.thread and self.thread.isRunning():
            self.log_event("A sweep is already running.", "warning")
            return
        try:
            kt_cfg    = self._parse_keithley_config()
            sweep_cfg = self._parse_sweep_config()
            meta_cfg  = self._parse_meta_config()
        except ValueError as e:
            self.log_event(f"Cannot start: {e}", "error")
            QMessageBox.critical(self, "Cannot start", str(e))
            return

        # ★ LIVE-mode safety: high V without verify is risky
        if (not DEMO_MODE
                and sweep_cfg.v_max_V > 0.1
                and not self._keithley_verified):
            self.log_event(
                f"Refusing to start: Keithley not verified and "
                f"V_max = {sweep_cfg.v_max_V*1000:.1f} mV > 100 mV.", "error")
            QMessageBox.critical(self, "Keithley not verified",
                f"This is a LIVE-mode sweep with V_max = {sweep_cfg.v_max_V*1000:.1f} mV.\n\n"
                f"For sweeps above 100 mV, please click 'Verify Keithley comms' "
                f"first to confirm the protocol matches the instrument's "
                f"command-set mode and the channel/wiring is correct.\n\n"
                f"Below 100 mV this check is not enforced.")
            return

        # Output filename: build, check for collision, auto-suffix
        if sweep_cfg.save_enabled:
            folder = self.le_folder.text().strip() or os.getcwd()
            try:
                os.makedirs(folder, exist_ok=True)
            except Exception as e:
                self.log_event(f"Cannot create output folder: {e}", "error")
                QMessageBox.critical(self, "Output folder", str(e))
                return
            self._refresh_filename()
            fname = self.lbl_filename.text()
            save_path = os.path.join(folder, fname)
            if os.path.exists(save_path) or os.path.exists(
                    os.path.splitext(save_path)[0] + '.json'):
                base, ext = os.path.splitext(save_path)
                for suffix in range(2, 1000):
                    candidate = f"{base}_{suffix}{ext}"
                    if not os.path.exists(candidate) and not os.path.exists(
                            os.path.splitext(candidate)[0] + '.json'):
                        save_path = candidate
                        break
                else:
                    self.log_event(
                        "Could not find an unused filename after 1000 tries.",
                        "error")
                    QMessageBox.critical(self, "Filename collision",
                        f"Folder {folder} already has 1000 files matching this "
                        f"timestamp pattern. Clean up old data and retry.")
                    return
            sweep_cfg.save_path = save_path

        # Reset plot buffers + GUI displays
        self._V_buf.clear(); self._I_buf.clear(); self._R_buf.clear()
        self.line_iv.setData([], [])
        self.line_iv_fit.setData([], [])
        self.line_rv.setData([], [])
        self.lbl_R_fit.setText("—")
        self.lbl_R2.setText("—")
        self.lbl_I_offset.setText("—")
        self.lbl_compliance.setText("—")
        self.lbl_compliance.setStyleSheet(
            f"color:{CT_TEXT}; font-family: 'JetBrains Mono', monospace;")
        self.lbl_now.setText("—")
        self.progress_bar.setValue(0)

        # Build controller and thread
        self.controller = make_keithley_controller(kt_cfg)
        params = {
            'keithley_config': kt_cfg,
            'sweep_config':    sweep_cfg,
            'meta_config':     meta_cfg,
        }
        self.thread = MeasurementThread(params, self.controller)
        self.thread.point_ready.connect(self.on_point)
        self.thread.log_msg.connect(self.log_event)
        self.thread.sweep_finished.connect(self.on_sweep_finished)
        self.thread.error_occurred.connect(self.on_error)

        # Disable config UI, enable abort
        self._set_config_enabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.act_start.setEnabled(False)
        self.act_stop.setEnabled(True)
        self._set_state("Sweeping", CT_YELLOW)

        # Lock total points for progress display
        self._total_pts = len(sweep_cfg.voltage_sequence())

        self.thread.start()

    def stop_sweep(self):
        if self.thread and self.thread.isRunning():
            self.log_event("Abort requested by user.", "warning")
            self.thread.stop()

    def _set_config_enabled(self, enabled: bool):
        widgets = [
            self.cb_protocol, self.le_visa, self.cb_channel, self.cb_4wire,
            self.le_i_comp, self.btn_verify,
            self.le_v_max, self.le_v_step, self.le_nplc, self.le_settle,
            self.cb_save, self.le_folder, self.btn_browse,
            self.le_sample, self.le_device, self.le_operator,
            self.le_run_name, self.le_electrode_pair,
        ]
        for w in widgets:
            w.setEnabled(enabled)
        for rb in self.rb_modes.values():
            rb.setEnabled(enabled)
        # Save folder/browse only if save is checked
        if enabled:
            self._on_save_toggled(self.cb_save.isChecked())

    # ─────────────────────────────────────────────────────────────────
    # Thread signal handlers
    # ─────────────────────────────────────────────────────────────────
    def on_point(self, info: PointInfo):
        # Defensive: if abort raced with this point, ignore
        if self.thread is None:
            return
        self._V_buf.append(info.V_meas)
        self._I_buf.append(info.I_meas)
        # R_inst can be NaN — pyqtgraph plots it as a gap, that's fine
        self._R_buf.append(info.R_inst)

        # Update plots
        self.line_iv.setData(self._V_buf, self._I_buf)
        self.line_rv.setData(self._V_buf, self._R_buf)

        # Progress
        n_done = info.seq_idx + 1
        if hasattr(self, '_total_pts'):
            pct = int(round(100 * n_done / max(self._total_pts, 1)))
            self.progress_bar.setValue(min(pct, 100))
            self.lbl_now.setText(
                f"pt {n_done}/{self._total_pts}  |  "
                f"V_set={fmt_eng(info.V_set, 'V')}  "
                f"V={fmt_eng(info.V_meas, 'V')}  "
                f"I={fmt_eng(info.I_meas, 'A')}  "
                f"R={fmt_eng(info.R_inst, 'Ω')}")

        # Compliance flag (live indicator)
        if info.in_compliance:
            self.lbl_compliance.setText("HIT this point")
            self.lbl_compliance.setStyleSheet(
                f"color:{CT_RED}; font-family: 'JetBrains Mono', monospace; "
                f"font-weight: 700;")

    def on_sweep_finished(self, fit: dict):
        # Display final fit results
        R_fit = fit.get('R_fit_Ohm', float('nan'))
        sigma_R = fit.get('sigma_R_Ohm', float('nan'))
        R2 = fit.get('R_squared', float('nan'))
        I_off = fit.get('I_offset_A', float('nan'))

        if not math.isnan(R_fit):
            R_str = fmt_eng(R_fit, 'Ω')
            sigma_str = fmt_eng(sigma_R, 'Ω') if not math.isnan(sigma_R) else '?'
            self.lbl_R_fit.setText(f"{R_str} ± {sigma_str}")
            color_R = CT_GREEN if (not math.isnan(R2) and R2 > 0.99) else CT_YELLOW
            self.lbl_R_fit.setStyleSheet(
                f"color:{color_R}; font-family: 'JetBrains Mono', monospace; "
                f"font-size: 12pt; font-weight: 700;")
        else:
            self.lbl_R_fit.setText("— (fit failed)")

        self.lbl_R2.setText(f"{R2:.6f}" if not math.isnan(R2) else "—")
        self.lbl_I_offset.setText(fmt_eng(I_off, 'A'))

        # Final compliance summary
        n_total = len(self._V_buf)
        n_comp = sum(1 for r in self._I_buf  # crude: we don't have the flag here, recount
                     if False)   # actually we should track separately, skip for now
        # Update the compliance label to a "session summary" if it wasn't already RED
        if self.lbl_compliance.text() == "—":
            self.lbl_compliance.setText("none")
            self.lbl_compliance.setStyleSheet(
                f"color:{CT_GREEN}; font-family: 'JetBrains Mono', monospace;")

        # Plot fit line over the data range
        if not math.isnan(R_fit) and len(self._V_buf) >= 2:
            slope = 1.0 / R_fit
            V_arr = np.array(self._V_buf)
            V_lo, V_hi = float(V_arr.min()), float(V_arr.max())
            xs = np.linspace(V_lo, V_hi, 64)
            ys = slope * xs + I_off
            self.line_iv_fit.setData(xs.tolist(), ys.tolist())

        # Re-enable UI
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.act_start.setEnabled(True)
        self.act_stop.setEnabled(False)
        self._set_config_enabled(True)
        self._set_state("Idle", CT_GREEN)

    def on_error(self, tb: str):
        self.log_event("Sweep crashed (traceback below).", "error")
        for line in tb.strip().splitlines()[-6:]:
            self.log_event("    " + line, "error")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.act_start.setEnabled(True)
        self.act_stop.setEnabled(False)
        self._set_config_enabled(True)
        self._set_state("Error", CT_RED)
        print(tb, file=sys.stderr)

    # ─────────────────────────────────────────────────────────────────
    # Persistence (QSettings)
    # ─────────────────────────────────────────────────────────────────
    def _persistent_widgets(self):
        return [
            ('lineedit', 'meta/sample',         self.le_sample),
            ('lineedit', 'meta/device',         self.le_device),
            ('lineedit', 'meta/operator',       self.le_operator),
            ('lineedit', 'meta/run_name',       self.le_run_name),
            ('lineedit', 'meta/electrode_pair', self.le_electrode_pair),
            ('lineedit', 'kt/visa',             self.le_visa),
            ('lineedit', 'kt/i_comp',           self.le_i_comp),
            ('checkbox', 'kt/4wire',            self.cb_4wire),
            ('lineedit', 'sweep/v_max',         self.le_v_max),
            ('lineedit', 'sweep/v_step',        self.le_v_step),
            ('lineedit', 'sweep/nplc',          self.le_nplc),
            ('lineedit', 'sweep/settle',        self.le_settle),
            ('checkbox', 'output/save_enabled', self.cb_save),
            ('lineedit', 'output/folder',       self.le_folder),
        ]

    def _save_settings(self):
        s = QSettings(ORG_NAME, SETTINGS_NAME)
        for kind, key, w in self._persistent_widgets():
            if kind == 'lineedit':
                s.setValue(key, w.text())
            elif kind == 'checkbox':
                s.setValue(key, w.isChecked())
        # userData combos
        s.setValue('kt/protocol', self.cb_protocol.currentData())
        s.setValue('kt/channel',  self.cb_channel.currentData())
        # Sweep mode (from radio group)
        for m, rb in self.rb_modes.items():
            if rb.isChecked():
                s.setValue('sweep/mode', m)
                break
        # Window geometry + splitter sizes
        s.setValue('window/geometry', self.saveGeometry())
        if hasattr(self, 'h_split'):
            s.setValue('window/h_split', self.h_split.sizes())

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
        # Combos
        for key, combo in [('kt/protocol', self.cb_protocol),
                           ('kt/channel',  self.cb_channel)]:
            v = s.value(key)
            if v is not None:
                for k in range(combo.count()):
                    if combo.itemData(k) == str(v):
                        combo.setCurrentIndex(k)
                        break
        # Sweep mode
        v = s.value('sweep/mode')
        if v is not None and str(v) in self.rb_modes:
            self.rb_modes[str(v)].setChecked(True)
        # Window geometry
        geom = s.value('window/geometry')
        if geom is not None:
            try:
                self.restoreGeometry(geom)
            except Exception:
                pass
        # Splitter (platform-robust list/string handling)
        sizes = s.value('window/h_split')
        if sizes:
            try:
                if isinstance(sizes, str):
                    parsed = [int(x) for x in sizes.split(',') if x.strip()]
                elif isinstance(sizes, (list, tuple)):
                    parsed = [int(x) for x in sizes]
                else:
                    parsed = []
                if parsed and hasattr(self, 'h_split'):
                    self.h_split.setSizes(parsed)
            except (ValueError, TypeError):
                pass

    # ─────────────────────────────────────────────────────────────────
    # Close handling
    # ─────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            reply = QMessageBox.question(
                self, "Sweep in progress",
                "A sweep is running. Abort and quit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self.thread.stop()
            # Wait for clean exit (KEITHLEY_VISA_TIMEOUT_MS + read + settle + safety)
            if not self.thread.wait(8000):
                self.log_event(
                    "Worker did not exit within 8 s — forcing termination.",
                    "warning")
                try:
                    self.thread.terminate()
                except Exception:
                    pass
                self.thread.wait(2000)
        try:
            self._save_settings()
        except Exception as e:
            print(f"WARN: save settings failed: {e}", file=sys.stderr)
        event.accept()


# =====================================================================
# Entry point — CLI flags, DEMO/LIVE switch, signal handling
# =====================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='Keithley_IV_sweep.py',
        description=f'{APP_NAME} {APP_VERSION} — '
                    f'Keithley 2636B IV / continuity sweep.',
        epilog="If neither --live nor --demo is given, runs in DEMO mode.",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument('--live', action='store_true',
                   help='Use real Keithley 2636B. Requires pyvisa.')
    g.add_argument('--demo', action='store_true',
                   help='Use simulated DemoKeithley (default).')
    p.add_argument('--version', action='version',
                   version=f'{APP_NAME} {APP_VERSION}')
    return p


def main():
    global DEMO_MODE
    args = _build_arg_parser().parse_args()
    DEMO_MODE = not args.live

    if not DEMO_MODE:
        try:
            _import_visa_lib_if_needed()
        except RuntimeError as e:
            print(f"FATAL: {e}", file=sys.stderr)
            sys.exit(1)

    # Linux-friendly Ctrl+C
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)

    # SIGINT wakeup so Qt event loop processes Python signals
    sigint_wakeup = QTimer()
    sigint_wakeup.start(100)
    sigint_wakeup.timeout.connect(lambda: None)

    pg.setConfigOption('background', CT_BASE)
    pg.setConfigOption('foreground', CT_TEXT)
    pg.setConfigOptions(antialias=True)

    gui = KeithleyIVGui()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
