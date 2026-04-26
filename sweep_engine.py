# -*- coding: utf-8 -*-
"""
Sweep Engine  —  v1.0
=====================

通用多通道扫描测量平台, 集成 NI DAQ + Oxford Mercury iPS + Mercury iTC +
Keithley 2636B SMU.  与 hysteresis_mapping.py / Keithley_IV_sweep.py /
dual_gate_mapping.py 完全独立。

Channel IN  (8 个可配置读取通道):
    DAQ AI 0–7 · VTI Temp · PT2 Temp · Magnet Temp (iPS) ·
    Magnet Temp (iTC) · B-field · Keithley I · Keithley R · Keithley V

Channel OUT (8 个可配置输出通道):
    DAQ AO 0–1 · Keithley V · B-field setpoint · T setpoint ·
    Time (wait)

Scan Channel (1 个扫描通道, Linear):
    与 Channel OUT 相同的物理类型 + Time
    Min / Max / Step / Dwell / 方向 (forward / reverse / bidirectional)

采样模型 (Two-tier):
    · DAQ AI: hardware-timed burst at 10 kS/s, averaged per scan point
    · 其它仪器: 每个 scan point 单次 VISA query
    · InstrumentManager.acquire_all() 合并一次调用全读

调用方式
--------
默认 DEMO 模式:        python sweep_engine.py
真实硬件:              python sweep_engine.py --live
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal as _signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

# ── Optional imports ─────────────────────────────────────────────
try:
    import pyvisa
    HAS_PYVISA = True
except Exception:
    HAS_PYVISA = False

try:
    import nidaqmx
    import nidaqmx.constants as ni_const
    from nidaqmx.stream_readers import AnalogMultiChannelReader
    HAS_NIDAQMX = True
except Exception:
    HAS_NIDAQMX = False

DEMO_MODE: bool = True

from PyQt5.QtCore import (Qt, QThread, pyqtSignal, QTimer, QSettings, QSize)
from PyQt5.QtGui import QFont, QColor, QIcon, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QComboBox,
    QPushButton, QCheckBox, QGroupBox, QHBoxLayout, QVBoxLayout, QGridLayout,
    QSplitter, QFileDialog, QPlainTextEdit, QProgressBar, QStatusBar, QAction,
    QSizePolicy, QFrame, QScrollArea, QMessageBox, QRadioButton, QButtonGroup,
    QTabWidget,
)
import pyqtgraph as pg

# =====================================================================
# 常量
# =====================================================================
APP_NAME      = "Sweep Engine"
APP_VERSION   = "v1.0"
ORG_NAME      = "lab.transport"
SETTINGS_NAME = "sweep_engine"

NUM_AI = 8;  NUM_AO = 2
NUM_IN_SLOTS  = 8;  NUM_OUT_SLOTS = 8;  NUM_PLOT_SLOTS = 3
DAQ_AI_SAMPLE_RATE_HZ     = 10000.0
DAQ_AI_DEFAULT_INTEG_S    = 0.05
LINE_FREQ_HZ              = 50.0
MERCURY_VISA_TIMEOUT_MS   = 1000
KEITHLEY_VISA_TIMEOUT_MS  = 3000
RAMP_CHECK_INTERVAL_S     = 0.5
CSV_FSYNC_EVERY_N         = 10

SOFT_AO_LIMIT      = 10.0
SOFT_V_LIMIT_KT    = 21.0
SOFT_I_LIMIT_KT    = 1.0
SOFT_B_LIMIT_T     = 15.0
SOFT_T_LIMIT_K     = 400.0

DEFAULT_KT_I_COMP  = 1e-6
DEFAULT_KT_NPLC    = 1.0
PROTO_TSP = 'tsp'; PROTO_SCPI = 'scpi'

# ── Channel type keys ────────────────────────────────────────────
CH_NONE = 'none'
CH_DAQ_AI = [f'daq_ai_{i}' for i in range(NUM_AI)]
CH_DAQ_AO = [f'daq_ao_{i}' for i in range(NUM_AO)]
CH_IPS_FIELD      = 'ips_field'
CH_IPS_FIELD_SET  = 'ips_field_set'
CH_IPS_MAG_TEMP   = 'ips_mag_temp'
CH_ITC_VTI        = 'itc_vti_temp'
CH_ITC_PT2        = 'itc_pt2_temp'
CH_ITC_MAG        = 'itc_mag_temp'
CH_ITC_TEMP_SET   = 'itc_temp_set'
CH_KT_V_SRC       = 'kt_v_source'
CH_KT_V_MEAS      = 'kt_v_meas'
CH_KT_I_MEAS      = 'kt_i_meas'
CH_KT_R_MEAS      = 'kt_r_meas'
CH_TIME            = 'time'

CHANNEL_IN_TYPES = (
    [(CH_NONE,          'None (disabled)')] +
    [(k, f'DAQ AI {i}') for i, k in enumerate(CH_DAQ_AI)] +
    [(CH_ITC_VTI,       'VTI Temperature'),
     (CH_ITC_PT2,       'PT2 Temperature'),
     (CH_IPS_MAG_TEMP,  'Magnet Temp (iPS)'),
     (CH_ITC_MAG,       'Magnet Temp (iTC)'),
     (CH_IPS_FIELD,     'Magnetic Field'),
     (CH_KT_I_MEAS,     'Keithley Current'),
     (CH_KT_R_MEAS,     'Keithley Resistance'),
     (CH_KT_V_MEAS,     'Keithley Voltage')]
)
CHANNEL_OUT_TYPES = (
    [(CH_NONE,          'None (disabled)'),
     (CH_TIME,          'Time (wait)')] +
    [(k, f'DAQ AO {i}') for i, k in enumerate(CH_DAQ_AO)] +
    [(CH_KT_V_SRC,      'Keithley V source'),
     (CH_IPS_FIELD_SET, 'B-field setpoint'),
     (CH_ITC_TEMP_SET,  'T setpoint')]
)
SCAN_CH_TYPES = CHANNEL_OUT_TYPES

_UNITS: Dict[str, str] = {CH_NONE: ''}
for k in CH_DAQ_AI: _UNITS[k] = 'V'
for k in CH_DAQ_AO: _UNITS[k] = 'V'
_UNITS.update({
    CH_IPS_FIELD: 'T', CH_IPS_FIELD_SET: 'T', CH_IPS_MAG_TEMP: 'K',
    CH_ITC_VTI: 'K', CH_ITC_PT2: 'K', CH_ITC_MAG: 'K', CH_ITC_TEMP_SET: 'K',
    CH_KT_V_SRC: 'V', CH_KT_V_MEAS: 'V', CH_KT_I_MEAS: 'A', CH_KT_R_MEAS: 'Ω',
    CH_TIME: 's',
})

_INST: Dict[str, str] = {CH_NONE: 'none', CH_TIME: 'none'}
for k in CH_DAQ_AI: _INST[k] = 'daq'
for k in CH_DAQ_AO: _INST[k] = 'daq'
_INST.update({
    CH_IPS_FIELD: 'ips', CH_IPS_FIELD_SET: 'ips', CH_IPS_MAG_TEMP: 'ips',
    CH_ITC_VTI: 'itc', CH_ITC_PT2: 'itc', CH_ITC_MAG: 'itc', CH_ITC_TEMP_SET: 'itc',
    CH_KT_V_SRC: 'keithley', CH_KT_V_MEAS: 'keithley',
    CH_KT_I_MEAS: 'keithley', CH_KT_R_MEAS: 'keithley',
})

DIR_FWD = 'forward'; DIR_REV = 'reverse'; DIR_BIDI = 'bidirectional'
DIRECTIONS = (DIR_FWD, DIR_REV, DIR_BIDI)
DIR_LABEL = {DIR_FWD: 'Forward (Min → Max)', DIR_REV: 'Reverse (Max → Min)',
             DIR_BIDI: 'Bidirectional (Min → Max → Min)'}

# ── Catppuccin Mocha ─────────────────────────────────────────────
CT_BASE     = '#1e1e2e';  CT_MANTLE   = '#181825';  CT_CRUST    = '#11111b'
CT_SURFACE0 = '#313244';  CT_SURFACE1 = '#45475a';  CT_SURFACE2 = '#585b70'
CT_TEXT     = '#cdd6f4';   CT_SUBTEXT0 = '#a6adc8';  CT_SUBTEXT1 = '#bac2de'
CT_OVERLAY1 = '#7f849c'
CT_BLUE     = '#89b4fa';   CT_GREEN    = '#a6e3a1';  CT_YELLOW   = '#f9e2af'
CT_PEACH    = '#fab387';   CT_RED      = '#f38ba8';  CT_MAUVE    = '#cba6f7'
CT_PINK     = '#f5c2e7';   CT_TEAL     = '#94e2d5';  CT_LAVENDER = '#b4befe'
CT_SKY      = '#89dceb';   CT_SAPPHIRE = '#74c7ec';  CT_MAROON   = '#eba0ac'
CT_FLAMINGO = '#f2cdcd';   CT_ROSEWATER= '#f5e0dc'
PLOT_COLORS = [CT_BLUE, CT_GREEN, CT_PEACH, CT_MAUVE,
               CT_TEAL, CT_PINK, CT_YELLOW, CT_RED]

# =====================================================================
# Helpers
# =====================================================================
def _json_clean(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (np.floating,)):
        v = float(obj); return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.bool_,)):   return bool(obj)
    if isinstance(obj, np.ndarray):    return _json_clean(obj.tolist())
    if isinstance(obj, dict):  return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_json_clean(x) for x in obj]
    return obj

def fmt_eng(value: float, unit: str = '') -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return f"— {unit}".strip()
    if value == 0: return f"0 {unit}".strip()
    sign = '-' if value < 0 else ''; av = abs(value)
    for sc, pf in [(1e12,'T'),(1e9,'G'),(1e6,'M'),(1e3,'k'),(1,''),
                   (1e-3,'m'),(1e-6,'µ'),(1e-9,'n'),(1e-12,'p'),(1e-15,'f')]:
        if av >= sc: return f"{sign}{av/sc:.4g} {pf}{unit}".rstrip()
    return f"{value:.3e} {unit}".strip()

def fmt_dur(s: float) -> str:
    if s < 1: return f"{int(round(s*1000))} ms"
    if s < 60: return f"{s:.1f}s"
    if s < 3600: m, sec = divmod(int(round(s)), 60); return f"{m}m {sec:02d}s"
    h, rem = divmod(int(round(s)), 3600); m, _ = divmod(rem, 60); return f"{h}h {m:02d}m"

def _mercury_parse_float(raw: str, expected_unit: str = '') -> float:
    s = raw.strip()
    for u in ('T', 'T/m', 'K', 'A', 'V', 'Ohm', 'W', 'mB'):
        if s.endswith(u): s = s[:-len(u)]; break
    parts = s.split(':')
    try: return float(parts[-1])
    except ValueError: raise ValueError(f"Cannot parse Mercury float from {raw!r}")

# =====================================================================
# Config dataclasses
# =====================================================================
@dataclass
class HardwareConfig:
    daq_device:           str   = 'Dev1'
    daq_ai_integ_s:       float = DAQ_AI_DEFAULT_INTEG_S
    ips_visa:             str   = 'GPIB0::10::INSTR'
    ips_psu_uid:          str   = 'GRPZ'
    itc_visa:             str   = 'GPIB0::20::INSTR'
    itc_vti_uid:          str   = 'MB1.T1'
    itc_pt2_uid:          str   = 'DB6.T1'
    itc_mag_uid:          str   = 'DB8.T1'
    kt_visa:              str   = 'GPIB0::26::INSTR'
    kt_proto:             str   = PROTO_TSP
    kt_channel:           str   = 'A'
    kt_4wire:             bool  = False
    kt_nplc:              float = DEFAULT_KT_NPLC
    kt_i_comp:            float = DEFAULT_KT_I_COMP

@dataclass
class ChInSlot:
    ch_type: str   = CH_NONE
    factor:  float = 1.0
    name:    str   = ''
    @property
    def enabled(self) -> bool: return self.ch_type != CH_NONE
    @property
    def unit(self) -> str: return _UNITS.get(self.ch_type, '')

@dataclass
class ChOutSlot:
    ch_type:    str   = CH_NONE
    value:      float = 0.0
    factor:     float = 1.0
    name:       str   = ''
    safe_min:   float = -10.0
    safe_max:   float = 10.0
    ramp_step:  float = 0.1
    ramp_speed: float = 1.0
    @property
    def enabled(self) -> bool: return self.ch_type != CH_NONE
    @property
    def unit(self) -> str: return _UNITS.get(self.ch_type, '')

@dataclass
class ScanConfig:
    ch_type:   str   = CH_NONE
    scan_min:  float = 0.0
    scan_max:  float = 0.005
    scan_step: float = 0.001
    dwell_s:   float = 0.1
    direction: str   = DIR_FWD
    factor:    float = 1.0
    name:      str   = 'Scan'
    @property
    def enabled(self) -> bool: return self.ch_type != CH_NONE
    def voltage_sequence(self) -> List[float]:
        if not self.enabled or self.scan_step <= 0: return []
        n = max(2, int(round(abs(self.scan_max - self.scan_min) / self.scan_step)) + 1)
        fwd = list(np.linspace(self.scan_min, self.scan_max, n))
        if self.direction == DIR_FWD: return fwd
        if self.direction == DIR_REV: return list(reversed(fwd))
        return fwd + list(reversed(fwd[:-1]))

@dataclass
class MetaConfig:
    sample: str = ''; device: str = ''; operator: str = ''; run_name: str = ''

# =====================================================================
# 1. DAQHardware
# =====================================================================
class DAQHardware:
    def __init__(self, device: str = 'Dev1'):
        self.device = device
        self.ao_chans = [f'{device}/ao{i}' for i in range(NUM_AO)]
        self.ai_chans = [f'{device}/ai{i}' for i in range(NUM_AI)]
        self._ao_tasks: Dict[str, Any] = {}
        self._ai_task = None; self._ai_n_cached = 0
        self._last_ao = {ch: 0.0 for ch in self.ao_chans}
        self.connected = False

    def open(self):
        if DEMO_MODE: self.connected = True; return
        if not HAS_NIDAQMX: raise RuntimeError("nidaqmx not installed")
        for ch in self.ao_chans:
            t = nidaqmx.Task(f"ao_{ch}"); t.ao_channels.add_ao_voltage_chan(ch)
            t.start(); self._ao_tasks[ch] = t
        self.connected = True

    def close(self):
        for t in self._ao_tasks.values():
            try: t.stop(); t.close()
            except Exception: pass
        self._ao_tasks.clear()
        if self._ai_task:
            try: self._ai_task.stop(); self._ai_task.close()
            except Exception: pass
        self._ai_task = None; self._ai_n_cached = 0; self.connected = False

    def write_ao(self, idx: int, value: float):
        value = max(-SOFT_AO_LIMIT, min(SOFT_AO_LIMIT, float(value)))
        ch = self.ao_chans[idx]
        if DEMO_MODE: self._last_ao[ch] = value; return
        task = self._ao_tasks.get(ch)
        if task: task.write(value, auto_start=False)
        self._last_ao[ch] = value

    def read_ai(self, t_read: float = 0.05) -> List[float]:
        n = max(2, int(round(t_read * DAQ_AI_SAMPLE_RATE_HZ)))
        if DEMO_MODE:
            time.sleep(t_read)
            sc = 1.0 / math.sqrt(n)
            return [float(0.5 + 0.1*i + sc*np.random.randn()*0.001) for i in range(NUM_AI)]
        if self._ai_task is None or self._ai_n_cached != n:
            if self._ai_task:
                try: self._ai_task.stop(); self._ai_task.close()
                except Exception: pass
            t = nidaqmx.Task("ai_burst")
            for ch in self.ai_chans: t.ai_channels.add_ai_voltage_chan(ch)
            t.timing.cfg_samp_clk_timing(DAQ_AI_SAMPLE_RATE_HZ,
                sample_mode=ni_const.AcquisitionType.FINITE, samps_per_chan=n)
            self._ai_task = t; self._ai_n_cached = n
        self._ai_task.start()
        buf = np.zeros((NUM_AI, n), dtype=np.float64)
        AnalogMultiChannelReader(self._ai_task.in_stream).read_many_sample(buf, n)
        self._ai_task.stop()
        return [float(np.mean(buf[i])) for i in range(NUM_AI)]

    def get_ao(self, idx: int) -> float:
        return self._last_ao.get(self.ao_chans[idx], 0.0)

# =====================================================================
# 2. MercuryIpsController
# =====================================================================
class MercuryIpsController:
    """Oxford Mercury iPS (SCPI). ⚠️ VERIFY ON HARDWARE ⚠️"""
    def __init__(self, visa_res: str, uid: str = 'GRPZ'):
        self.visa_res = visa_res; self.uid = uid
        self._rm = None; self._inst = None; self.connected = False
        self._demo_B = 0.0; self._demo_Tset = 0.0; self._demo_Tmag = 3.8

    def connect(self):
        if DEMO_MODE: self.connected = True; return
        if not HAS_PYVISA: raise RuntimeError("pyvisa required")
        self._rm = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(self.visa_res)
        self._inst.timeout = MERCURY_VISA_TIMEOUT_MS
        self._inst.read_termination = '\n'; self._inst.write_termination = '\n'
        self._inst.query("*IDN?"); self.connected = True

    def disconnect(self):
        for o in (self._inst, self._rm):
            try:
                if o: o.close()
            except Exception: pass
        self._inst = None; self._rm = None; self.connected = False

    def read_field(self) -> float:
        if DEMO_MODE: return self._demo_B + np.random.randn()*1e-5
        return _mercury_parse_float(
            self._inst.query(f"READ:DEV:{self.uid}:PSU:SIG:FLD"), 'T')

    def read_mag_temp(self) -> float:
        if DEMO_MODE: return self._demo_Tmag + np.random.randn()*0.01
        return _mercury_parse_float(
            self._inst.query(f"READ:DEV:{self.uid}:PSU:SIG:TEMP"), 'K')

    def set_field(self, B: float, rate: float = 0.5):
        B = max(-SOFT_B_LIMIT_T, min(SOFT_B_LIMIT_T, float(B)))
        if DEMO_MODE: self._demo_B = B; return
        u = self.uid
        self._inst.write(f"SET:DEV:{u}:PSU:SIG:RFST:{rate:.5f}")
        self._inst.write(f"SET:DEV:{u}:PSU:SIG:FSET:{B:.5f}")
        self._inst.write(f"SET:DEV:{u}:PSU:ACTN:RTOS")

    def verify(self) -> Dict[str, Any]:
        r: Dict[str, Any] = {'errors': []}
        if DEMO_MODE:
            r['idn'] = 'DEMO iPS'; r['field_T'] = self._demo_B; return r
        u = self.uid
        for k, q in [('idn','*IDN?'),
                      ('field_T', f"READ:DEV:{u}:PSU:SIG:FLD"),
                      ('fset_T',  f"READ:DEV:{u}:PSU:SIG:FSET")]:
            try:
                raw = self._inst.query(q)
                r[k] = _mercury_parse_float(raw) if k != 'idn' else raw.strip()
            except Exception as e:
                r[k] = None; r['errors'].append(f"{k}: {e}")
        return r

# =====================================================================
# 3. MercuryItcController
# =====================================================================
class MercuryItcController:
    """Oxford Mercury iTC (SCPI). ⚠️ VERIFY ON HARDWARE ⚠️
    UIDs are cryostat-specific. Verify by clicking 'Verify iTC' button."""
    def __init__(self, visa_res: str, vti_uid='MB1.T1',
                 pt2_uid='DB6.T1', mag_uid='DB8.T1'):
        self.visa_res = visa_res
        self.vti_uid = vti_uid; self.pt2_uid = pt2_uid; self.mag_uid = mag_uid
        self._rm = None; self._inst = None; self.connected = False
        self._demo = {'vti': 4.2, 'pt2': 3.1, 'mag': 3.8, 'tset': 4.2}

    def connect(self):
        if DEMO_MODE: self.connected = True; return
        if not HAS_PYVISA: raise RuntimeError("pyvisa required")
        self._rm = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(self.visa_res)
        self._inst.timeout = MERCURY_VISA_TIMEOUT_MS
        self._inst.read_termination = '\n'; self._inst.write_termination = '\n'
        self._inst.query("*IDN?"); self.connected = True

    def disconnect(self):
        for o in (self._inst, self._rm):
            try:
                if o: o.close()
            except Exception: pass
        self._inst = None; self._rm = None; self.connected = False

    def _read_t(self, uid: str, demo_key: str) -> float:
        if DEMO_MODE: return self._demo[demo_key] + np.random.randn()*0.005
        return _mercury_parse_float(
            self._inst.query(f"READ:DEV:{uid}:TEMP:SIG:TEMP"), 'K')

    def read_vti(self)  -> float: return self._read_t(self.vti_uid, 'vti')
    def read_pt2(self)  -> float: return self._read_t(self.pt2_uid, 'pt2')
    def read_mag(self)  -> float: return self._read_t(self.mag_uid, 'mag')

    def set_temp(self, T: float, rate_K_min: float = 1.0):
        T = max(0.0, min(SOFT_T_LIMIT_K, float(T)))
        if DEMO_MODE: self._demo['tset'] = T; self._demo['vti'] = T; return
        u = self.vti_uid
        self._inst.write(f"SET:DEV:{u}:TEMP:LOOP:RAMP:ENAB:ON")
        self._inst.write(f"SET:DEV:{u}:TEMP:LOOP:RAMP:RATE:{rate_K_min:.3f}")
        self._inst.write(f"SET:DEV:{u}:TEMP:LOOP:TSET:{T:.4f}")

    def verify(self) -> Dict[str, Any]:
        r: Dict[str, Any] = {'errors': []}
        if DEMO_MODE:
            r['idn'] = 'DEMO iTC'; r.update(self._demo); return r
        for k, q in [('idn', '*IDN?')] + [
            (f'{lbl}_K', f"READ:DEV:{uid}:TEMP:SIG:TEMP")
            for lbl, uid in [('vti', self.vti_uid), ('pt2', self.pt2_uid),
                              ('mag', self.mag_uid)]]:
            try:
                raw = self._inst.query(q)
                r[k] = _mercury_parse_float(raw) if k != 'idn' else raw.strip()
            except Exception as e:
                r[k] = None; r['errors'].append(f"{k}: {e}")
        return r

# =====================================================================
# 4. KeithleyController
# =====================================================================
class KeithleyController:
    """Keithley 2636B SMU (TSP + SCPI). ⚠️ VERIFY ON HARDWARE ⚠️"""
    def __init__(self, visa_res: str, proto: str = PROTO_TSP,
                 channel: str = 'A', wire4: bool = False,
                 nplc: float = 1.0, i_comp: float = 1e-6):
        self.visa_res = visa_res; self.proto = proto
        self.channel = channel; self.wire4 = wire4
        self.nplc = nplc; self.i_comp = i_comp
        self._rm = None; self._inst = None; self.connected = False
        self._smu = 'smua' if channel == 'A' else 'smub'
        self._n   = 1 if channel == 'A' else 2
        self._output_on = False; self._demo_V = 0.0

    def connect(self):
        if DEMO_MODE: self.connected = True; return
        if not HAS_PYVISA: raise RuntimeError("pyvisa required")
        self._rm = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(self.visa_res)
        self._inst.timeout = KEITHLEY_VISA_TIMEOUT_MS
        self._inst.read_termination = '\n'; self._inst.write_termination = '\n'
        self._inst.query("*IDN?"); self.connected = True

    def disconnect(self):
        try: self.output_off()
        except Exception: pass
        for o in (self._inst, self._rm):
            try:
                if o: o.close()
            except Exception: pass
        self._inst = None; self._rm = None; self.connected = False

    def configure(self):
        if DEMO_MODE: return
        if self.proto == PROTO_TSP:
            s = self._smu
            for c in [f"{s}.reset()", f"{s}.source.func = {s}.OUTPUT_DCVOLTS",
                      f"{s}.source.autorangev = {s}.AUTORANGE_ON",
                      f"{s}.measure.autorangei = {s}.AUTORANGE_ON",
                      f"{s}.source.limiti = {self.i_comp:.6e}",
                      f"{s}.measure.nplc = {self.nplc:.4f}",
                      (f"{s}.sense = {s}.SENSE_REMOTE" if self.wire4
                       else f"{s}.sense = {s}.SENSE_LOCAL"),
                      f"{s}.source.levelv = 0"]:
                self._inst.write(c)
        else:
            n = self._n
            for c in [f":SOUR{n}:FUNC VOLT", f":SOUR{n}:VOLT:RANG:AUTO ON",
                      f":SENS{n}:CURR:RANG:AUTO ON",
                      f":SENS{n}:CURR:PROT {self.i_comp:.6e}",
                      f":SENS{n}:CURR:NPLC {self.nplc:.4f}",
                      f":SYST:RSEN {1 if self.wire4 else 0}",
                      f":SOUR{n}:VOLT 0"]:
                self._inst.write(c)

    def write_v(self, V: float):
        V = max(-SOFT_V_LIMIT_KT, min(SOFT_V_LIMIT_KT, float(V)))
        if DEMO_MODE: self._demo_V = V; return
        if self.proto == PROTO_TSP:
            self._inst.write(f"{self._smu}.source.levelv = {V:.6e}")
        else:
            self._inst.write(f":SOUR{self._n}:VOLT {V:.6e}")

    def read_iv(self) -> Tuple[float, float]:
        if DEMO_MODE:
            R = 2340.0 if self.channel == 'A' else 17500.0
            Voff = 1.5e-5 if self.channel == 'A' else -3.2e-5
            Vm = self._demo_V + Voff + np.random.randn()*1e-6
            Ii = (Vm - Voff) / R + np.random.randn()*1e-12
            if abs(Ii) > self.i_comp: Ii = math.copysign(self.i_comp, Ii)
            return float(Vm), float(Ii)
        if self.proto == PROTO_TSP:
            raw = self._inst.query(f"print({self._smu}.measure.iv())").strip()
            parts = [p for p in raw.replace(',','\t').split('\t') if p.strip()]
            return float(parts[1]), float(parts[0])  # TSP: I,V → return V,I
        else:
            raw = self._inst.query(f":READ?{self._n}").strip()
            parts = [p for p in raw.replace(';',',').split(',') if p.strip()]
            return float(parts[0]), float(parts[1])  # SCPI: V,I

    def output_on(self):
        self._output_on = True
        if DEMO_MODE: return
        if self.proto == PROTO_TSP:
            self._inst.write(f"{self._smu}.source.output = {self._smu}.OUTPUT_ON")
        else:
            self._inst.write(f":OUTP{self._n} ON")

    def output_off(self):
        self._output_on = False; self._demo_V = 0.0
        if DEMO_MODE: return
        try:
            if self.proto == PROTO_TSP:
                self._inst.write(f"{self._smu}.source.levelv = 0")
                self._inst.write(f"{self._smu}.source.output = {self._smu}.OUTPUT_OFF")
            else:
                self._inst.write(f":SOUR{self._n}:VOLT 0")
                self._inst.write(f":OUTP{self._n} OFF")
        except Exception: pass

    def verify(self) -> Dict[str, Any]:
        r: Dict[str, Any] = {'errors': []}
        if DEMO_MODE: r['idn'] = f'DEMO KT ch {self.channel}'; return r
        try: r['idn'] = self._inst.query("*IDN?").strip()
        except Exception as e: r['idn'] = None; r['errors'].append(str(e))
        return r

# =====================================================================
# 5. InstrumentManager
# =====================================================================
class InstrumentManager:
    def __init__(self, hw: HardwareConfig):
        self.hw = hw
        self.daq:      Optional[DAQHardware]           = None
        self.ips:      Optional[MercuryIpsController]   = None
        self.itc:      Optional[MercuryItcController]   = None
        self.keithley: Optional[KeithleyController]     = None
        self._t0 = 0.0; self._kt_out = False

    def needed(self, ch_in: List[ChInSlot], ch_out: List[ChOutSlot],
               scan: ScanConfig) -> set:
        s = set()
        for sl in ch_in:
            if sl.enabled: s.add(_INST.get(sl.ch_type, 'none'))
        for sl in ch_out:
            if sl.enabled: s.add(_INST.get(sl.ch_type, 'none'))
        if scan.enabled: s.add(_INST.get(scan.ch_type, 'none'))
        s.discard('none'); return s

    def connect(self, need: set):
        self._t0 = time.monotonic()
        if 'daq' in need:
            self.daq = DAQHardware(self.hw.daq_device); self.daq.open()
        if 'ips' in need:
            self.ips = MercuryIpsController(self.hw.ips_visa, self.hw.ips_psu_uid)
            self.ips.connect()
        if 'itc' in need:
            self.itc = MercuryItcController(
                self.hw.itc_visa, self.hw.itc_vti_uid,
                self.hw.itc_pt2_uid, self.hw.itc_mag_uid)
            self.itc.connect()
        if 'keithley' in need:
            self.keithley = KeithleyController(
                self.hw.kt_visa, self.hw.kt_proto, self.hw.kt_channel,
                self.hw.kt_4wire, self.hw.kt_nplc, self.hw.kt_i_comp)
            self.keithley.connect(); self.keithley.configure()

    def disconnect(self):
        for inst in (self.keithley, self.ips, self.itc):
            try:
                if inst: inst.disconnect()
            except Exception: pass
        if self.daq:
            try: self.daq.close()
            except Exception: pass
        self.daq = None; self.ips = None; self.itc = None; self.keithley = None

    def kt_on(self):
        if self.keithley: self.keithley.output_on(); self._kt_out = True
    def kt_off(self):
        if self.keithley: self.keithley.output_off(); self._kt_out = False

    def acquire_all(self) -> Dict[str, float]:
        r: Dict[str, float] = {CH_TIME: time.monotonic() - self._t0}
        if self.daq and self.daq.connected:
            vals = self.daq.read_ai(self.hw.daq_ai_integ_s)
            for i in range(NUM_AI): r[CH_DAQ_AI[i]] = vals[i]
            for i in range(NUM_AO): r[CH_DAQ_AO[i]] = self.daq.get_ao(i)
        if self.ips and self.ips.connected:
            try: r[CH_IPS_FIELD] = self.ips.read_field()
            except Exception: r[CH_IPS_FIELD] = float('nan')
            try: r[CH_IPS_MAG_TEMP] = self.ips.read_mag_temp()
            except Exception: r[CH_IPS_MAG_TEMP] = float('nan')
        if self.itc and self.itc.connected:
            for key, fn in [(CH_ITC_VTI, self.itc.read_vti),
                            (CH_ITC_PT2, self.itc.read_pt2),
                            (CH_ITC_MAG, self.itc.read_mag)]:
                try: r[key] = fn()
                except Exception: r[key] = float('nan')
        if self.keithley and self.keithley.connected:
            try:
                V, I = self.keithley.read_iv()
                r[CH_KT_V_MEAS] = V; r[CH_KT_I_MEAS] = I
                r[CH_KT_R_MEAS] = V/I if abs(I) > 1e-30 else float('nan')
            except Exception:
                r[CH_KT_V_MEAS] = r[CH_KT_I_MEAS] = r[CH_KT_R_MEAS] = float('nan')
        return r

    def write_ch(self, ch: str, val: float):
        if ch == CH_TIME: time.sleep(max(0, val))
        elif ch == CH_DAQ_AO[0] and self.daq: self.daq.write_ao(0, val)
        elif ch == CH_DAQ_AO[1] and self.daq: self.daq.write_ao(1, val)
        elif ch == CH_KT_V_SRC:
            if self.keithley:
                if not self._kt_out: self.kt_on()
                self.keithley.write_v(val)
        elif ch == CH_IPS_FIELD_SET and self.ips: self.ips.set_field(val)
        elif ch == CH_ITC_TEMP_SET and self.itc: self.itc.set_temp(val)

# =====================================================================
# 6. ScanEngine (QThread)
# =====================================================================
class ScanEngine(QThread):
    point_ready   = pyqtSignal(object)
    log_msg       = pyqtSignal(str, str)
    scan_finished = pyqtSignal(object)
    error_occurred= pyqtSignal(str)

    def __init__(self, im: InstrumentManager, ch_in: List[ChInSlot],
                 ch_out: List[ChOutSlot], scan: ScanConfig,
                 meta: MetaConfig, save_path: str = ''):
        super().__init__()
        self.im = im
        self.ch_in  = [s for s in ch_in if s.enabled]
        self.ch_out = [s for s in ch_out if s.enabled]
        self.scan = scan; self.meta = meta
        self.save_path = save_path; self.is_running = True

    def stop(self): self.is_running = False

    def _ramp_outs(self):
        for slot in self.ch_out:
            if not self.is_running: return
            raw = slot.value / slot.factor if slot.factor else slot.value
            self.log_msg.emit(
                f"Set OUT '{slot.name or slot.ch_type}' → "
                f"{fmt_eng(slot.value, slot.unit)}", 'info')
            if slot.ch_type in (CH_IPS_FIELD_SET, CH_ITC_TEMP_SET):
                self.im.write_ch(slot.ch_type, raw)
                for _ in range(240):
                    if not self.is_running: return
                    time.sleep(RAMP_CHECK_INTERVAL_S)
                    rd = self.im.acquire_all()
                    cur = rd.get(CH_IPS_FIELD if slot.ch_type == CH_IPS_FIELD_SET
                                else CH_ITC_VTI, float('nan'))
                    if not math.isnan(cur) and abs(cur - raw) < 0.01: break
            elif slot.ch_type == CH_TIME:
                time.sleep(max(0, slot.value))
            else:
                self.im.write_ch(slot.ch_type, raw)

    def _csv_cols(self) -> List[str]:
        return (['seq', 'scan_val'] +
                [s.name or s.ch_type for s in self.ch_in] + ['t_s'])

    def run(self):
        f = None; writer = None
        try:
            self.log_msg.emit("━━ Scan starting ━━", 'info')
            if self.ch_out:
                self.log_msg.emit("Ramping static outputs...", 'info')
                self._ramp_outs()
                if not self.is_running:
                    self.scan_finished.emit({}); return

            needs_kt = (self.scan.ch_type == CH_KT_V_SRC or
                        any(s.ch_type == CH_KT_V_SRC for s in self.ch_out))
            if needs_kt: self.im.kt_on(); self.log_msg.emit("Keithley ON", 'success')

            if self.scan.ch_type == CH_TIME:
                dur = self.scan.scan_max - self.scan.scan_min
                n = max(1, int(round(dur / self.scan.dwell_s)))
                seq = list(np.linspace(self.scan.scan_min, self.scan.scan_max, n))
            else:
                seq = self.scan.voltage_sequence()
            n_total = len(seq)
            self.log_msg.emit(
                f"Scan: {n_total} pts, {self.scan.name} "
                f"[{self.scan.scan_min} → {self.scan.scan_max}]", 'info')

            if self.save_path:
                f = open(self.save_path, 'w', newline='', encoding='utf-8')
                writer = csv.writer(f)
                for line in [
                    f"# {APP_NAME} {APP_VERSION}",
                    f"# {time.strftime('%Y-%m-%dT%H:%M:%S')}",
                    f"# sample={self.meta.sample} device={self.meta.device}",
                    f"# scan={self.scan.ch_type} dir={self.scan.direction}",
                    f"# DEMO={DEMO_MODE}"]:
                    writer.writerow([line])
                for i, s in enumerate(self.ch_in):
                    writer.writerow([f"# in_{i}={s.ch_type} factor={s.factor} name={s.name}"])
                for i, s in enumerate(self.ch_out):
                    writer.writerow([f"# out_{i}={s.ch_type} val={s.value} name={s.name}"])
                writer.writerow(self._csv_cols())

            t0 = time.monotonic(); pw = 0; last_idx = 0
            for idx, sv in enumerate(seq):
                if not self.is_running:
                    self.log_msg.emit(f"Aborted {idx+1}/{n_total}", 'warning'); break
                if self.scan.ch_type == CH_TIME:
                    elapsed = time.monotonic() - t0
                    if elapsed < sv: time.sleep(sv - elapsed)
                else:
                    raw = sv / self.scan.factor if self.scan.factor else sv
                    self.im.write_ch(self.scan.ch_type, raw)
                if self.scan.dwell_s > 0 and self.scan.ch_type != CH_TIME:
                    time.sleep(self.scan.dwell_s)
                readings = self.im.acquire_all()
                pt: Dict[str, Any] = {'_idx': idx, '_sv': sv,
                                       '_t': time.monotonic() - t0}
                for s in self.ch_in:
                    pt[s.ch_type] = readings.get(s.ch_type, float('nan')) * s.factor
                self.point_ready.emit(pt)
                if writer:
                    row = [idx, f"{sv:.6e}"]
                    for s in self.ch_in:
                        v = pt.get(s.ch_type, float('nan'))
                        row.append(f"{v:.6e}" if not (isinstance(v, float) and math.isnan(v)) else 'nan')
                    row.append(f"{pt['_t']:.4f}"); writer.writerow(row)
                    pw += 1
                    if pw % CSV_FSYNC_EVERY_N == 0 and f:
                        try: f.flush(); os.fsync(f.fileno())
                        except Exception: pass
                last_idx = idx

            if needs_kt: self.im.kt_off(); self.log_msg.emit("Keithley OFF", 'success')
            el = time.monotonic() - t0
            summary = {'n_points': last_idx + 1, 'elapsed_s': el}
            self.log_msg.emit(f"Done: {summary['n_points']} pts in {fmt_dur(el)}", 'success')
            if self.save_path:
                sidecar = os.path.splitext(self.save_path)[0] + '.json'
                try:
                    meta_d = _json_clean({
                        'schema_version': 'sweep_v1.0', 'app': APP_VERSION,
                        'demo': DEMO_MODE,
                        'meta': {'sample': self.meta.sample, 'device': self.meta.device,
                                 'operator': self.meta.operator, 'run': self.meta.run_name},
                        'scan': {'ch': self.scan.ch_type, 'min': self.scan.scan_min,
                                 'max': self.scan.scan_max, 'step': self.scan.scan_step,
                                 'dir': self.scan.direction, 'dwell': self.scan.dwell_s},
                        'ch_in': [{'type': s.ch_type, 'factor': s.factor, 'name': s.name}
                                  for s in self.ch_in],
                        'ch_out': [{'type': s.ch_type, 'val': s.value, 'name': s.name}
                                   for s in self.ch_out],
                        'summary': summary})
                    with open(sidecar, 'w', encoding='utf-8') as jf:
                        json.dump(meta_d, jf, indent=2, allow_nan=False)
                    self.log_msg.emit(f"JSON: {sidecar}", 'info')
                except Exception as e:
                    self.log_msg.emit(f"JSON failed: {e}", 'error')
            self.scan_finished.emit(summary)
        except Exception:
            tb = traceback.format_exc()
            self.log_msg.emit(f"ERROR: {tb.splitlines()[-1]}", 'error')
            self.error_occurred.emit(tb)
            try: self.im.kt_off()
            except Exception: pass
        finally:
            if f:
                try: f.flush(); f.close()
                except Exception: pass


# =====================================================================
# 7. GUI — Stylesheet + Widgets + MainWindow
# =====================================================================
class NoWheelComboBox(QComboBox):
    def wheelEvent(self, e): e.ignore()

def _qss() -> str:
    return f"""
    QMainWindow, QWidget {{ background:{CT_BASE}; color:{CT_TEXT}; font-family:'Inter','Segoe UI',sans-serif; font-size:10pt; }}
    QGroupBox {{ background:{CT_MANTLE}; border:1px solid {CT_SURFACE1}; border-radius:6px; margin-top:14px; padding-top:6px; font-weight:600; }}
    QGroupBox::title {{ subcontrol-origin:margin; left:10px; padding:0 6px; color:{CT_LAVENDER}; }}
    QLabel {{ color:{CT_TEXT}; }}
    QLabel#sec {{ color:{CT_SUBTEXT0}; font-weight:500; }}
    QLabel#val {{ color:{CT_TEAL}; font-family:'JetBrains Mono','Consolas',monospace; }}
    QLineEdit, QComboBox {{ background:{CT_SURFACE0}; border:1px solid {CT_SURFACE1}; border-radius:4px; padding:3px 5px; color:{CT_TEXT}; }}
    QLineEdit:focus, QComboBox:focus {{ border:1px solid {CT_BLUE}; }}
    QComboBox::drop-down {{ border:none; width:18px; }}
    QComboBox::down-arrow {{ image:none; border-left:4px solid transparent; border-right:4px solid transparent; border-top:5px solid {CT_TEXT}; margin-right:4px; }}
    QComboBox QAbstractItemView {{ background:{CT_SURFACE0}; color:{CT_TEXT}; selection-background-color:{CT_BLUE}; selection-color:{CT_BASE}; }}
    QPushButton {{ background:{CT_SURFACE0}; border:1px solid {CT_SURFACE1}; border-radius:4px; padding:5px 12px; color:{CT_TEXT}; }}
    QPushButton:hover {{ background:{CT_SURFACE1}; }}
    QPushButton:disabled {{ color:{CT_OVERLAY1}; background:{CT_MANTLE}; }}
    QPushButton#startBtn {{ background:{CT_GREEN}; color:{CT_BASE}; font-weight:700; }}
    QPushButton#stopBtn {{ background:{CT_RED}; color:{CT_BASE}; font-weight:700; }}
    QPushButton#verifyBtn {{ background:{CT_MAUVE}; color:{CT_BASE}; font-weight:600; }}
    QCheckBox {{ color:{CT_TEXT}; spacing:5px; }}
    QCheckBox::indicator {{ width:13px; height:13px; border:1px solid {CT_SURFACE2}; border-radius:3px; background:{CT_SURFACE0}; }}
    QCheckBox::indicator:checked {{ background:{CT_GREEN}; border:1px solid {CT_GREEN}; }}
    QRadioButton {{ color:{CT_TEXT}; spacing:5px; }}
    QRadioButton::indicator {{ width:13px; height:13px; border:1px solid {CT_SURFACE2}; border-radius:7px; background:{CT_SURFACE0}; }}
    QRadioButton::indicator:checked {{ background:{CT_GREEN}; border:1px solid {CT_GREEN}; }}
    QPlainTextEdit {{ background:{CT_CRUST}; color:{CT_TEXT}; border:1px solid {CT_SURFACE1}; font-family:'JetBrains Mono','Consolas',monospace; font-size:9pt; }}
    QProgressBar {{ background:{CT_SURFACE0}; border:1px solid {CT_SURFACE1}; border-radius:4px; text-align:center; color:{CT_TEXT}; }}
    QProgressBar::chunk {{ background:{CT_BLUE}; border-radius:3px; }}
    QStatusBar {{ background:{CT_MANTLE}; color:{CT_TEXT}; }}
    QScrollArea {{ border:none; background:{CT_BASE}; }}
    QScrollBar:vertical {{ background:{CT_MANTLE}; width:10px; }}
    QScrollBar::handle:vertical {{ background:{CT_SURFACE1}; border-radius:5px; min-height:30px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0px; }}
    QTabWidget::pane {{ border:1px solid {CT_SURFACE1}; background:{CT_MANTLE}; }}
    QTabBar::tab {{ background:{CT_SURFACE0}; color:{CT_TEXT}; padding:5px 12px; border:1px solid {CT_SURFACE1}; border-bottom:none; border-top-left-radius:4px; border-top-right-radius:4px; }}
    QTabBar::tab:selected {{ background:{CT_MANTLE}; color:{CT_BLUE}; }}
    """

# ── Channel row widgets ──────────────────────────────────────────
class ChInRow(QWidget):
    """One Channel IN slot: [Type dropdown] [Value label] [Factor] [Name]"""
    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx
        h = QHBoxLayout(self); h.setContentsMargins(2,2,2,2); h.setSpacing(4)
        lbl = QLabel(f"IN {idx}"); lbl.setFixedWidth(32); lbl.setObjectName("sec")
        h.addWidget(lbl)
        self.cb_type = NoWheelComboBox(); self.cb_type.setMinimumWidth(130)
        for key, label in CHANNEL_IN_TYPES:
            self.cb_type.addItem(label, userData=key)
        h.addWidget(self.cb_type)
        self.lbl_val = QLabel("—"); self.lbl_val.setObjectName("val")
        self.lbl_val.setFixedWidth(90); self.lbl_val.setAlignment(Qt.AlignRight)
        h.addWidget(self.lbl_val)
        self.le_factor = QLineEdit("1.0"); self.le_factor.setFixedWidth(60)
        self.le_factor.setToolTip("Multiplicative factor: physical = raw × factor")
        h.addWidget(QLabel("×")); h.addWidget(self.le_factor)
        self.le_name = QLineEdit(f"in_{idx}"); self.le_name.setFixedWidth(80)
        self.le_name.setToolTip("Column name in CSV and plot label")
        h.addWidget(self.le_name)

    def to_slot(self) -> ChInSlot:
        try: f = float(self.le_factor.text())
        except ValueError: f = 1.0
        return ChInSlot(
            ch_type=self.cb_type.currentData() or CH_NONE,
            factor=f, name=self.le_name.text().strip() or f"in_{self.idx}")

    def set_value(self, v: float):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            self.lbl_val.setText("—")
        else:
            self.lbl_val.setText(fmt_eng(v, ''))


class ChOutRow(QWidget):
    """One Channel OUT slot: [Type] [Value] [Factor] [Name] [Min] [Max] [Step] [Speed]"""
    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx
        h = QHBoxLayout(self); h.setContentsMargins(2,2,2,2); h.setSpacing(3)
        lbl = QLabel(f"OUT {idx}"); lbl.setFixedWidth(38); lbl.setObjectName("sec")
        h.addWidget(lbl)
        self.cb_type = NoWheelComboBox(); self.cb_type.setMinimumWidth(120)
        for key, label in CHANNEL_OUT_TYPES:
            self.cb_type.addItem(label, userData=key)
        h.addWidget(self.cb_type)
        def _le(w, default, tip=''):
            le = QLineEdit(default); le.setFixedWidth(w)
            if tip: le.setToolTip(tip)
            return le
        self.le_val   = _le(65, '0',     'Target value (physical units)')
        self.le_factor= _le(50, '1.0',   'Factor: raw = physical / factor')
        self.le_name  = _le(65, f'out_{idx}', 'Name')
        self.le_min   = _le(50, '-10',   'Safe min limit')
        self.le_max   = _le(50, '10',    'Safe max limit')
        self.le_step  = _le(45, '0.1',   'Ramp step size')
        self.le_speed = _le(45, '1.0',   'Ramp rate (units/s)')
        for w, lbl_t in [(self.le_val, 'Val'), (self.le_factor, '×'),
                          (self.le_name, ''), (self.le_min, 'Min'),
                          (self.le_max, 'Max'), (self.le_step, 'Stp'),
                          (self.le_speed, 'Spd')]:
            if lbl_t: h.addWidget(QLabel(lbl_t))
            h.addWidget(w)

    def to_slot(self) -> ChOutSlot:
        def _f(le, default=0.0):
            try: return float(le.text())
            except ValueError: return default
        return ChOutSlot(
            ch_type=self.cb_type.currentData() or CH_NONE,
            value=_f(self.le_val), factor=_f(self.le_factor, 1.0),
            name=self.le_name.text().strip() or f"out_{self.idx}",
            safe_min=_f(self.le_min, -10), safe_max=_f(self.le_max, 10),
            ramp_step=_f(self.le_step, 0.1), ramp_speed=_f(self.le_speed, 1.0))

# ── Plot slot widget ─────────────────────────────────────────────
class PlotSlot(QWidget):
    """One configurable plot: dropdown to pick Y-axis channel, pyqtgraph plot."""
    def __init__(self, idx: int, ch_in_rows: List[ChInRow]):
        super().__init__()
        self.idx = idx; self.ch_in_rows = ch_in_rows
        v = QVBoxLayout(self); v.setContentsMargins(2,2,2,2); v.setSpacing(3)
        hdr = QHBoxLayout(); hdr.setSpacing(4)
        hdr.addWidget(QLabel(f"Plot {idx+1}"))
        self.cb_y = NoWheelComboBox(); self.cb_y.setMinimumWidth(140)
        self._refresh_channels()
        hdr.addWidget(self.cb_y); hdr.addStretch()
        v.addLayout(hdr)
        self.pw = pg.PlotWidget(); self.pw.setBackground(CT_BASE)
        self.pw.showGrid(x=True, y=True, alpha=0.25)
        self.pw.setLabel('bottom', 'Scan', color=CT_TEXT)
        self.pw.setLabel('left', 'Value', color=CT_TEXT)
        for ax in ('bottom', 'left'):
            self.pw.getAxis(ax).setPen(CT_SUBTEXT1)
            self.pw.getAxis(ax).setTextPen(CT_TEXT)
        c = PLOT_COLORS[idx % len(PLOT_COLORS)]
        self.line = self.pw.plot([], [], pen=pg.mkPen(c, width=2),
                                 symbol='o', symbolSize=4,
                                 symbolBrush=c, symbolPen=None)
        v.addWidget(self.pw, 1)
        self._xs: List[float] = []; self._ys: List[float] = []

    def _refresh_channels(self):
        self.cb_y.clear()
        self.cb_y.addItem('(none)', userData=CH_NONE)
        for row in self.ch_in_rows:
            s = row.to_slot()
            if s.enabled:
                self.cb_y.addItem(s.name or s.ch_type, userData=s.ch_type)

    def reset(self):
        self._xs.clear(); self._ys.clear()
        self.line.setData([], [])

    def add_point(self, scan_val: float, readings: Dict[str, float]):
        ch = self.cb_y.currentData()
        if ch == CH_NONE or ch is None: return
        y = readings.get(ch, float('nan'))
        if isinstance(y, float) and math.isnan(y): return
        self._xs.append(scan_val); self._ys.append(y)
        self.line.setData(self._xs, self._ys)


# =====================================================================
# 8. Main Window
# =====================================================================
class SweepGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(_qss())
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}  [{'DEMO' if DEMO_MODE else 'LIVE'}]")
        self.resize(1400, 900)
        self.thread: Optional[ScanEngine] = None
        self.im: Optional[InstrumentManager] = None
        self._verified: Dict[str, bool] = {}
        self._build_ui()
        self._init_statusbar()
        self._load_settings()
        self.log_event("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")
        self.log_event(f"{APP_NAME}  ·  {APP_VERSION}", "success")
        if DEMO_MODE: self.log_event("DEMO MODE", "warning")
        else: self.log_event("LIVE MODE", "success")

    def _build_ui(self):
        central = QWidget(); outer = QVBoxLayout(central)
        outer.setContentsMargins(6,6,6,6); outer.setSpacing(4)
        outer.addWidget(self._build_meta_bar())
        self.h_split = QSplitter(Qt.Horizontal)
        self.h_split.addWidget(self._build_left())
        self.h_split.addWidget(self._build_right())
        self.h_split.setStretchFactor(0, 0); self.h_split.setStretchFactor(1, 1)
        self.h_split.setSizes([420, 980])
        outer.addWidget(self.h_split, 1)
        outer.addWidget(self._build_log())
        self.setCentralWidget(central)

    # ── Metadata ──────────────────────────────────────────────────
    def _build_meta_bar(self):
        gb = QGroupBox(); gb.setFlat(True)
        h = QHBoxLayout(gb); h.setContentsMargins(4,2,4,2); h.setSpacing(4)
        def _le(lbl, w=100):
            l = QLabel(lbl); l.setObjectName("sec"); h.addWidget(l)
            le = QLineEdit(); le.setMaximumWidth(w); h.addWidget(le); return le
        self.le_sample = _le("Sample", 120)
        self.le_device = _le("Device", 100)
        self.le_operator = _le("Operator", 70)
        self.le_run = _le("Run", 70)
        h.addStretch()
        self.lbl_clock = QLabel("—"); self.lbl_clock.setObjectName("val"); h.addWidget(self.lbl_clock)
        self._ct = QTimer(self); self._ct.timeout.connect(lambda: self.lbl_clock.setText(time.strftime("%Y-%m-%d  %H:%M:%S")))
        self._ct.start(1000)
        return gb

    # ── Left panel ────────────────────────────────────────────────
    def _build_left(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); v = QVBoxLayout(inner)
        v.setContentsMargins(6,6,6,6); v.setSpacing(6)
        v.addWidget(self._build_hw_tabs())
        v.addWidget(self._build_scan_group())
        v.addWidget(self._build_out_group())
        v.addStretch()
        v.addLayout(self._build_buttons())
        scroll.setWidget(inner); scroll.setMinimumWidth(400)
        return scroll

    def _build_hw_tabs(self):
        gb = QGroupBox("Hardware")
        v = QVBoxLayout(gb); v.setContentsMargins(4,4,4,4)
        tabs = QTabWidget()
        # DAQ tab
        w = QWidget(); g = QGridLayout(w); g.setContentsMargins(4,4,4,4)
        g.addWidget(QLabel("Device"), 0, 0)
        self.le_daq_dev = QLineEdit("Dev1"); g.addWidget(self.le_daq_dev, 0, 1)
        g.addWidget(QLabel("AI integration"), 1, 0)
        self.le_daq_integ = QLineEdit("0.05"); g.addWidget(self.le_daq_integ, 1, 1)
        g.addWidget(QLabel("s"), 1, 2)
        tabs.addTab(w, "DAQ")
        # iPS tab
        w = QWidget(); g = QGridLayout(w); g.setContentsMargins(4,4,4,4)
        g.addWidget(QLabel("VISA"), 0, 0)
        self.le_ips_visa = QLineEdit("GPIB0::10::INSTR"); g.addWidget(self.le_ips_visa, 0, 1)
        g.addWidget(QLabel("PSU UID"), 1, 0)
        self.le_ips_uid = QLineEdit("GRPZ"); g.addWidget(self.le_ips_uid, 1, 1)
        b = QPushButton("Verify iPS"); b.setObjectName("verifyBtn")
        b.clicked.connect(lambda: self._verify('ips')); g.addWidget(b, 2, 0, 1, 2)
        tabs.addTab(w, "iPS")
        # iTC tab
        w = QWidget(); g = QGridLayout(w); g.setContentsMargins(4,4,4,4)
        g.addWidget(QLabel("VISA"), 0, 0)
        self.le_itc_visa = QLineEdit("GPIB0::20::INSTR"); g.addWidget(self.le_itc_visa, 0, 1)
        g.addWidget(QLabel("VTI UID"), 1, 0)
        self.le_itc_vti = QLineEdit("MB1.T1"); g.addWidget(self.le_itc_vti, 1, 1)
        g.addWidget(QLabel("PT2 UID"), 2, 0)
        self.le_itc_pt2 = QLineEdit("DB6.T1"); g.addWidget(self.le_itc_pt2, 2, 1)
        g.addWidget(QLabel("Mag UID"), 3, 0)
        self.le_itc_mag = QLineEdit("DB8.T1"); g.addWidget(self.le_itc_mag, 3, 1)
        b = QPushButton("Verify iTC"); b.setObjectName("verifyBtn")
        b.clicked.connect(lambda: self._verify('itc')); g.addWidget(b, 4, 0, 1, 2)
        tabs.addTab(w, "iTC")
        # Keithley tab
        w = QWidget(); g = QGridLayout(w); g.setContentsMargins(4,4,4,4)
        g.addWidget(QLabel("VISA"), 0, 0)
        self.le_kt_visa = QLineEdit("GPIB0::26::INSTR"); g.addWidget(self.le_kt_visa, 0, 1)
        g.addWidget(QLabel("Protocol"), 1, 0)
        self.cb_kt_proto = NoWheelComboBox()
        self.cb_kt_proto.addItem("TSP (factory)", userData=PROTO_TSP)
        self.cb_kt_proto.addItem("SCPI", userData=PROTO_SCPI)
        g.addWidget(self.cb_kt_proto, 1, 1)
        g.addWidget(QLabel("Channel"), 2, 0)
        self.cb_kt_ch = NoWheelComboBox()
        self.cb_kt_ch.addItem("A", userData='A'); self.cb_kt_ch.addItem("B", userData='B')
        g.addWidget(self.cb_kt_ch, 2, 1)
        self.cb_kt_4w = QCheckBox("4-wire"); g.addWidget(self.cb_kt_4w, 3, 0)
        g.addWidget(QLabel("NPLC"), 4, 0)
        self.le_kt_nplc = QLineEdit("1.0"); g.addWidget(self.le_kt_nplc, 4, 1)
        g.addWidget(QLabel("I compliance"), 5, 0)
        self.le_kt_icomp = QLineEdit("1e-6"); g.addWidget(self.le_kt_icomp, 5, 1)
        g.addWidget(QLabel("A"), 5, 2)
        b = QPushButton("Verify Keithley"); b.setObjectName("verifyBtn")
        b.clicked.connect(lambda: self._verify('keithley')); g.addWidget(b, 6, 0, 1, 2)
        tabs.addTab(w, "Keithley")
        v.addWidget(tabs)
        return gb

    def _build_scan_group(self):
        gb = QGroupBox("Scan Channel")
        g = QGridLayout(gb); g.setContentsMargins(6,4,6,6); g.setHorizontalSpacing(6)
        g.addWidget(QLabel("Type"), 0, 0)
        self.cb_scan_type = NoWheelComboBox()
        for k, lbl in SCAN_CH_TYPES: self.cb_scan_type.addItem(lbl, userData=k)
        g.addWidget(self.cb_scan_type, 0, 1, 1, 3)
        for r, lbl, attr, default in [
            (1, "Min", "le_sc_min", "0"), (2, "Max", "le_sc_max", "0.005"),
            (3, "Step", "le_sc_step", "0.001"), (4, "Dwell(s)", "le_sc_dwell", "0.1")]:
            g.addWidget(QLabel(lbl), r, 0)
            le = QLineEdit(default); setattr(self, attr, le)
            g.addWidget(le, r, 1, 1, 3)
        g.addWidget(QLabel("Factor"), 5, 0)
        self.le_sc_factor = QLineEdit("1.0"); g.addWidget(self.le_sc_factor, 5, 1)
        g.addWidget(QLabel("Name"), 5, 2)
        self.le_sc_name = QLineEdit("Scan"); g.addWidget(self.le_sc_name, 5, 3)
        g.addWidget(QLabel("Direction"), 6, 0)
        self.rb_dir = QButtonGroup(self); self.rb_dirs: Dict[str, QRadioButton] = {}
        for i, d in enumerate(DIRECTIONS):
            rb = QRadioButton(DIR_LABEL[d]); self.rb_dirs[d] = rb
            self.rb_dir.addButton(rb, i); g.addWidget(rb, 7+i, 0, 1, 4)
        self.rb_dirs[DIR_FWD].setChecked(True)
        # Save
        self.cb_save = QCheckBox("Save data"); self.cb_save.setChecked(True)
        g.addWidget(self.cb_save, 10, 0, 1, 2)
        g.addWidget(QLabel("Folder"), 11, 0)
        self.le_folder = QLineEdit(os.path.expanduser("~"))
        g.addWidget(self.le_folder, 11, 1, 1, 2)
        self.btn_browse = QPushButton("…"); self.btn_browse.setFixedWidth(30)
        self.btn_browse.clicked.connect(self._browse)
        g.addWidget(self.btn_browse, 11, 3)
        return gb

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.le_folder.text())
        if d: self.le_folder.setText(d)

    def _build_out_group(self):
        gb = QGroupBox("Channel OUT (static outputs)")
        v = QVBoxLayout(gb); v.setContentsMargins(4,4,4,4); v.setSpacing(2)
        self.out_rows: List[ChOutRow] = []
        for i in range(NUM_OUT_SLOTS):
            row = ChOutRow(i); self.out_rows.append(row); v.addWidget(row)
        return gb

    def _build_buttons(self):
        h = QHBoxLayout()
        self.btn_start = QPushButton("START"); self.btn_start.setObjectName("startBtn")
        self.btn_start.clicked.connect(self.start_scan); h.addWidget(self.btn_start)
        self.btn_stop = QPushButton("ABORT"); self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.clicked.connect(self.stop_scan)
        self.btn_stop.setEnabled(False); h.addWidget(self.btn_stop)
        return h

    # ── Right panel ───────────────────────────────────────────────
    def _build_right(self):
        w = QWidget(); v = QVBoxLayout(w)
        v.setContentsMargins(4,4,4,4); v.setSpacing(4)
        # Plots
        self.in_rows: List[ChInRow] = []
        # Build IN rows first (needed by PlotSlots)
        for i in range(NUM_IN_SLOTS):
            self.in_rows.append(ChInRow(i))
        self.plot_slots: List[PlotSlot] = []
        for i in range(NUM_PLOT_SLOTS):
            ps = PlotSlot(i, self.in_rows); self.plot_slots.append(ps)
            v.addWidget(ps, 1)
        # Channel IN monitor
        gb = QGroupBox("Channel IN (live monitor)")
        vv = QVBoxLayout(gb); vv.setContentsMargins(4,4,4,4); vv.setSpacing(2)
        for row in self.in_rows: vv.addWidget(row)
        v.addWidget(gb)
        return w

    # ── Log + progress ────────────────────────────────────────────
    def _build_log(self):
        gb = QGroupBox("Log"); v = QVBoxLayout(gb); v.setContentsMargins(4,4,4,4); v.setSpacing(3)
        h = QHBoxLayout()
        self.progress = QProgressBar(); self.progress.setRange(0,100)
        h.addWidget(self.progress, 1)
        self.lbl_pt = QLabel("—"); self.lbl_pt.setObjectName("val")
        self.lbl_pt.setMinimumWidth(250); h.addWidget(self.lbl_pt)
        v.addLayout(h)
        self.te_log = QPlainTextEdit(); self.te_log.setReadOnly(True)
        self.te_log.setMaximumBlockCount(2000); self.te_log.setMinimumHeight(100)
        v.addWidget(self.te_log, 1)
        return gb

    def _init_statusbar(self):
        sb = self.statusBar()
        bg = CT_PEACH if DEMO_MODE else CT_GREEN
        badge = QLabel(f"  [{'DEMO' if DEMO_MODE else 'LIVE'}]  ")
        badge.setStyleSheet(f"color:{CT_BASE}; background:{bg}; font-weight:700; "
                            f"padding:1px 8px; border-radius:4px; font-family:monospace;")
        sb.addPermanentWidget(badge)
        self.lbl_state = QLabel("Idle")
        self.lbl_state.setStyleSheet(f"color:{CT_GREEN}; font-family:monospace; padding:0 8px;")
        sb.addPermanentWidget(self.lbl_state)

    def _set_state(self, txt: str, color: str):
        self.lbl_state.setText(txt)
        self.lbl_state.setStyleSheet(f"color:{color}; font-family:monospace; padding:0 8px;")

    # ── Logging ───────────────────────────────────────────────────
    def log_event(self, msg: str, level: str = 'info'):
        ts = time.strftime("%H:%M:%S")
        c = {
            'info': CT_TEXT, 'success': CT_GREEN, 'warning': CT_YELLOW, 'error': CT_RED
        }.get(level, CT_TEXT)
        self.te_log.appendHtml(
            f'<span style="color:{CT_OVERLAY1}">[{ts}]</span> '
            f'<span style="color:{c}">{msg}</span>')
        self.te_log.verticalScrollBar().setValue(self.te_log.verticalScrollBar().maximum())

    # ── Parse config ──────────────────────────────────────────────
    def _parse_hw(self) -> HardwareConfig:
        def _f(le, d): 
            try: return float(le.text())
            except ValueError: return d
        return HardwareConfig(
            daq_device=self.le_daq_dev.text().strip() or 'Dev1',
            daq_ai_integ_s=_f(self.le_daq_integ, DAQ_AI_DEFAULT_INTEG_S),
            ips_visa=self.le_ips_visa.text().strip(),
            ips_psu_uid=self.le_ips_uid.text().strip() or 'GRPZ',
            itc_visa=self.le_itc_visa.text().strip(),
            itc_vti_uid=self.le_itc_vti.text().strip() or 'MB1.T1',
            itc_pt2_uid=self.le_itc_pt2.text().strip() or 'DB6.T1',
            itc_mag_uid=self.le_itc_mag.text().strip() or 'DB8.T1',
            kt_visa=self.le_kt_visa.text().strip(),
            kt_proto=self.cb_kt_proto.currentData() or PROTO_TSP,
            kt_channel=self.cb_kt_ch.currentData() or 'A',
            kt_4wire=self.cb_kt_4w.isChecked(),
            kt_nplc=_f(self.le_kt_nplc, DEFAULT_KT_NPLC),
            kt_i_comp=_f(self.le_kt_icomp, DEFAULT_KT_I_COMP))

    def _parse_scan(self) -> ScanConfig:
        def _f(le, d):
            try: return float(le.text())
            except ValueError: return d
        d = DIR_FWD
        for k, rb in self.rb_dirs.items():
            if rb.isChecked(): d = k; break
        return ScanConfig(
            ch_type=self.cb_scan_type.currentData() or CH_NONE,
            scan_min=_f(self.le_sc_min, 0), scan_max=_f(self.le_sc_max, 0.005),
            scan_step=_f(self.le_sc_step, 0.001), dwell_s=_f(self.le_sc_dwell, 0.1),
            direction=d, factor=_f(self.le_sc_factor, 1.0),
            name=self.le_sc_name.text().strip() or 'Scan')

    def _parse_meta(self) -> MetaConfig:
        return MetaConfig(
            sample=self.le_sample.text().strip(),
            device=self.le_device.text().strip(),
            operator=self.le_operator.text().strip(),
            run_name=self.le_run.text().strip())

    # ── Verify ────────────────────────────────────────────────────
    def _verify(self, inst_key: str):
        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "Busy", "Cannot verify during a scan.")
            return
        hw = self._parse_hw()
        im = InstrumentManager(hw)
        self.log_event(f"Verifying {inst_key}...", 'info')
        try:
            im.connect({inst_key})
        except Exception as e:
            self.log_event(f"Connect failed: {e}", 'error')
            QMessageBox.critical(self, "Connect failed", str(e)); return
        try:
            r = im.verify_instrument(inst_key) if hasattr(im, 'verify_instrument') else {}
            # Fallback: verify directly
            if not r:
                obj = getattr(im, inst_key, None)
                if obj and hasattr(obj, 'verify'): r = obj.verify()
                else: r = {'errors': ['no verify method']}
        except Exception as e:
            self.log_event(f"Verify failed: {e}", 'error')
            QMessageBox.critical(self, "Verify failed", str(e)); return
        finally:
            im.disconnect()
        lines = [f"Verification results for {inst_key}:", ""]
        for k, val in r.items():
            if k != 'errors': lines.append(f"  {k}: {val}")
        errs = r.get('errors', [])
        if errs:
            lines += ["", "Errors:"] + [f"  - {e}" for e in errs]
            self._verified[inst_key] = False
        else:
            lines += ["", "All queries OK."]
            self._verified[inst_key] = True
        self.log_event(f"Verify {inst_key}: {'OK' if not errs else 'ERRORS'}",
                       'success' if not errs else 'warning')
        QMessageBox.information(self, f"Verify {inst_key}", "\n".join(lines))

    # ── Start / Stop ──────────────────────────────────────────────
    def start_scan(self):
        if self.thread and self.thread.isRunning():
            self.log_event("Scan already running.", "warning"); return
        hw = self._parse_hw()
        scan = self._parse_scan()
        meta = self._parse_meta()
        ch_in = [r.to_slot() for r in self.in_rows]
        ch_out = [r.to_slot() for r in self.out_rows]

        if not scan.enabled:
            self.log_event("No scan channel selected.", "error")
            QMessageBox.critical(self, "No scan channel", "Select a scan channel type.")
            return

        # Refresh plot channel dropdowns
        for ps in self.plot_slots: ps._refresh_channels()

        # Build save path
        save_path = ''
        if self.cb_save.isChecked():
            folder = self.le_folder.text().strip() or os.getcwd()
            try: os.makedirs(folder, exist_ok=True)
            except Exception as e:
                self.log_event(f"Folder error: {e}", 'error'); return
            ts = time.strftime("%Y%m%d_%H%M%S")
            s = meta.sample or 'sample'; d = meta.device or 'dev'
            r = meta.run_name or 'run'
            fname = f"{ts}__{s}__{d}__{r}__scan.csv"
            for c in '<>:"|?*\\/': fname = fname.replace(c, '_')
            save_path = os.path.join(folder, fname)
            # Auto-suffix collision avoidance
            if os.path.exists(save_path):
                base, ext = os.path.splitext(save_path)
                for sfx in range(2, 1000):
                    cand = f"{base}_{sfx}{ext}"
                    if not os.path.exists(cand):
                        save_path = cand; break

        # Connect instruments
        self.im = InstrumentManager(hw)
        need = self.im.needed(ch_in, ch_out, scan)
        self.log_event(f"Connecting: {', '.join(sorted(need)) or 'none'}", 'info')
        try:
            self.im.connect(need)
        except Exception as e:
            self.log_event(f"Connect failed: {e}", 'error')
            QMessageBox.critical(self, "Connect failed", str(e))
            self.im.disconnect(); return

        # Reset plots
        for ps in self.plot_slots: ps.reset()
        self.progress.setValue(0); self.lbl_pt.setText("—")

        # Thread
        self._n_total = len(scan.voltage_sequence()) if scan.ch_type != CH_TIME else max(
            1, int(round((scan.scan_max - scan.scan_min) / scan.dwell_s)))
        self.thread = ScanEngine(self.im, ch_in, ch_out, scan, meta, save_path)
        self.thread.point_ready.connect(self._on_point)
        self.thread.log_msg.connect(self.log_event)
        self.thread.scan_finished.connect(self._on_done)
        self.thread.error_occurred.connect(self._on_error)
        self._set_config_enabled(False)
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self._set_state("Scanning", CT_YELLOW)
        self.thread.start()

    def stop_scan(self):
        if self.thread and self.thread.isRunning():
            self.log_event("Abort requested.", "warning"); self.thread.stop()

    def _set_config_enabled(self, en: bool):
        for w in [self.le_sample, self.le_device, self.le_operator, self.le_run,
                  self.le_daq_dev, self.le_daq_integ,
                  self.le_ips_visa, self.le_ips_uid,
                  self.le_itc_visa, self.le_itc_vti, self.le_itc_pt2, self.le_itc_mag,
                  self.le_kt_visa, self.cb_kt_proto, self.cb_kt_ch, self.cb_kt_4w,
                  self.le_kt_nplc, self.le_kt_icomp,
                  self.cb_scan_type, self.le_sc_min, self.le_sc_max, self.le_sc_step,
                  self.le_sc_dwell, self.le_sc_factor, self.le_sc_name,
                  self.cb_save, self.le_folder, self.btn_browse]:
            w.setEnabled(en)
        for rb in self.rb_dirs.values(): rb.setEnabled(en)
        for row in self.out_rows:
            for child in row.findChildren(QWidget): child.setEnabled(en)
        for row in self.in_rows:
            for child in row.findChildren(QWidget): child.setEnabled(en)

    # ── Signal handlers ───────────────────────────────────────────
    def _on_point(self, pt: dict):
        if self.thread is None: return
        idx = pt.get('_idx', 0); sv = pt.get('_sv', 0)
        pct = int(round(100 * (idx + 1) / max(self._n_total, 1)))
        self.progress.setValue(min(pct, 100))
        # Update IN row live values
        for row in self.in_rows:
            s = row.to_slot()
            if s.enabled:
                v = pt.get(s.ch_type, float('nan'))
                row.set_value(v)
        # Update plots
        for ps in self.plot_slots:
            ps.add_point(sv, pt)
        # Status
        self.lbl_pt.setText(f"pt {idx+1}/{self._n_total}  scan={fmt_eng(sv, '')}")

    def _on_done(self, summary: dict):
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self._set_config_enabled(True); self._set_state("Idle", CT_GREEN)
        if self.im:
            try: self.im.disconnect()
            except Exception: pass

    def _on_error(self, tb: str):
        self.log_event("Scan crashed.", "error")
        for line in tb.strip().splitlines()[-5:]:
            self.log_event("  " + line, "error")
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self._set_config_enabled(True); self._set_state("Error", CT_RED)
        if self.im:
            try: self.im.disconnect()
            except Exception: pass

    # ── Settings ──────────────────────────────────────────────────
    def _save_settings(self):
        s = QSettings(ORG_NAME, SETTINGS_NAME)
        for key, w in [('meta/sample', self.le_sample), ('meta/device', self.le_device),
                       ('meta/operator', self.le_operator), ('meta/run', self.le_run),
                       ('hw/daq_dev', self.le_daq_dev), ('hw/daq_integ', self.le_daq_integ),
                       ('hw/ips_visa', self.le_ips_visa), ('hw/ips_uid', self.le_ips_uid),
                       ('hw/itc_visa', self.le_itc_visa), ('hw/itc_vti', self.le_itc_vti),
                       ('hw/itc_pt2', self.le_itc_pt2), ('hw/itc_mag', self.le_itc_mag),
                       ('hw/kt_visa', self.le_kt_visa), ('hw/kt_nplc', self.le_kt_nplc),
                       ('hw/kt_icomp', self.le_kt_icomp),
                       ('scan/min', self.le_sc_min), ('scan/max', self.le_sc_max),
                       ('scan/step', self.le_sc_step), ('scan/dwell', self.le_sc_dwell),
                       ('scan/factor', self.le_sc_factor), ('scan/name', self.le_sc_name),
                       ('output/folder', self.le_folder)]:
            s.setValue(key, w.text())
        s.setValue('hw/kt_proto', self.cb_kt_proto.currentData())
        s.setValue('hw/kt_ch', self.cb_kt_ch.currentData())
        s.setValue('hw/kt_4w', self.cb_kt_4w.isChecked())
        s.setValue('output/save', self.cb_save.isChecked())
        s.setValue('scan/type', self.cb_scan_type.currentData())
        for d, rb in self.rb_dirs.items():
            if rb.isChecked(): s.setValue('scan/dir', d); break
        s.setValue('window/geometry', self.saveGeometry())
        # Channel IN/OUT slot configs
        for i, row in enumerate(self.in_rows):
            s.setValue(f'in/{i}/type', row.cb_type.currentData())
            s.setValue(f'in/{i}/factor', row.le_factor.text())
            s.setValue(f'in/{i}/name', row.le_name.text())
        for i, row in enumerate(self.out_rows):
            s.setValue(f'out/{i}/type', row.cb_type.currentData())
            s.setValue(f'out/{i}/val', row.le_val.text())
            s.setValue(f'out/{i}/factor', row.le_factor.text())
            s.setValue(f'out/{i}/name', row.le_name.text())
            s.setValue(f'out/{i}/min', row.le_min.text())
            s.setValue(f'out/{i}/max', row.le_max.text())
            s.setValue(f'out/{i}/step', row.le_step.text())
            s.setValue(f'out/{i}/speed', row.le_speed.text())

    def _load_settings(self):
        s = QSettings(ORG_NAME, SETTINGS_NAME)
        for key, w in [('meta/sample', self.le_sample), ('meta/device', self.le_device),
                       ('meta/operator', self.le_operator), ('meta/run', self.le_run),
                       ('hw/daq_dev', self.le_daq_dev), ('hw/daq_integ', self.le_daq_integ),
                       ('hw/ips_visa', self.le_ips_visa), ('hw/ips_uid', self.le_ips_uid),
                       ('hw/itc_visa', self.le_itc_visa), ('hw/itc_vti', self.le_itc_vti),
                       ('hw/itc_pt2', self.le_itc_pt2), ('hw/itc_mag', self.le_itc_mag),
                       ('hw/kt_visa', self.le_kt_visa), ('hw/kt_nplc', self.le_kt_nplc),
                       ('hw/kt_icomp', self.le_kt_icomp),
                       ('scan/min', self.le_sc_min), ('scan/max', self.le_sc_max),
                       ('scan/step', self.le_sc_step), ('scan/dwell', self.le_sc_dwell),
                       ('scan/factor', self.le_sc_factor), ('scan/name', self.le_sc_name),
                       ('output/folder', self.le_folder)]:
            v = s.value(key)
            if v is not None: w.setText(str(v))
        # Combo restores
        for key, combo in [('hw/kt_proto', self.cb_kt_proto),
                           ('hw/kt_ch', self.cb_kt_ch),
                           ('scan/type', self.cb_scan_type)]:
            v = s.value(key)
            if v is not None:
                for k in range(combo.count()):
                    if combo.itemData(k) == str(v):
                        combo.setCurrentIndex(k); break
        v = s.value('hw/kt_4w')
        if v is not None:
            self.cb_kt_4w.setChecked(str(v).lower() in ('true','1'))
        v = s.value('output/save')
        if v is not None:
            self.cb_save.setChecked(str(v).lower() in ('true','1'))
        v = s.value('scan/dir')
        if v and str(v) in self.rb_dirs:
            self.rb_dirs[str(v)].setChecked(True)
        geom = s.value('window/geometry')
        if geom is not None:
            try: self.restoreGeometry(geom)
            except Exception: pass
        # Channel slots
        for i, row in enumerate(self.in_rows):
            v = s.value(f'in/{i}/type')
            if v:
                for k in range(row.cb_type.count()):
                    if row.cb_type.itemData(k) == str(v):
                        row.cb_type.setCurrentIndex(k); break
            for attr, key in [('le_factor', 'factor'), ('le_name', 'name')]:
                v = s.value(f'in/{i}/{key}')
                if v is not None: getattr(row, attr).setText(str(v))
        for i, row in enumerate(self.out_rows):
            v = s.value(f'out/{i}/type')
            if v:
                for k in range(row.cb_type.count()):
                    if row.cb_type.itemData(k) == str(v):
                        row.cb_type.setCurrentIndex(k); break
            for attr, key in [('le_val','val'),('le_factor','factor'),('le_name','name'),
                              ('le_min','min'),('le_max','max'),('le_step','step'),
                              ('le_speed','speed')]:
                v = s.value(f'out/{i}/{key}')
                if v is not None: getattr(row, attr).setText(str(v))

    # ── Close ─────────────────────────────────────────────────────
    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            reply = QMessageBox.question(self, "Scan running",
                "Abort and quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes: event.ignore(); return
            self.thread.stop()
            if not self.thread.wait(8000):
                try: self.thread.terminate()
                except Exception: pass
                self.thread.wait(2000)
        try: self._save_settings()
        except Exception: pass
        event.accept()


# =====================================================================
# 9. Entry point
# =====================================================================
def main():
    global DEMO_MODE
    p = argparse.ArgumentParser(prog='sweep_engine.py',
        description=f'{APP_NAME} {APP_VERSION}')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--live', action='store_true')
    g.add_argument('--demo', action='store_true')
    args = p.parse_args()
    DEMO_MODE = not args.live

    if not DEMO_MODE:
        if not HAS_PYVISA:
            print("FATAL: --live requires pyvisa.", file=sys.stderr); sys.exit(1)

    _signal.signal(_signal.SIGINT, _signal.SIG_DFL)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME); app.setOrganizationName(ORG_NAME)
    wakeup = QTimer(); wakeup.start(100); wakeup.timeout.connect(lambda: None)
    pg.setConfigOption('background', CT_BASE)
    pg.setConfigOption('foreground', CT_TEXT)
    pg.setConfigOptions(antialias=True)
    gui = SweepGui(); gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
