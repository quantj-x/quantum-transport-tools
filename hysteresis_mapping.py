# -*- coding: utf-8 -*-
"""
Hysteresis Mapping  —  v1.1
===========================

磁场扫描 + R(B) 磁滞测量专用程序。 与 dual_gate_mapping.py 完全独立。

物理工作流
----------
对于一个双栅 vdW 异质结 (CrI3 / TMD 双层 / 石墨烯 双层 etc.):

  1. 用户选定 outer 变量 (n 或 D) 和它的扫描范围。 另一个变量
     被固定在 "fixed value" 上。 GUI 会用 hBN 几何反推每一个外层点
     对应的 (Vtg, Vbg)。 任何超出 |Vtg|_max / |Vbg|_max 的外层点
     在 preflight 阶段就拒绝, 不允许扫描启动。

  2. 每个外层点扫描时:
       a) ramp gates 到目标 (Vtg, Vbg) 并等 RC 稳
       b) **fwd**: 命令磁体 ramp 到 B_max, 在 ramp 过程中以
          固定节拍 t_sample = B_step / ramp_rate 持续采样 DAQ,
          每个采样点在 t_read 时间窗内做 hardware-timed 平均,
          同时记录 B_actual (磁体实测) 和 B_requested
          (量化到 canonical B 网格的最近格点)
       c) magnet.hold(), 等 t_settle_after_fwd
       d) **bwd**: 同样反向, ramp 到 B_min
       e) magnet.hold(), 等 t_settle_after_bwd
       f) 计算这个外层点的滞回面积 ∫|R_fwd − R_bwd|dB, 显示到状态栏

  3. 全部外层点扫完后, gates ramp 回 0, 磁体 hold + disconnect,
     CSV 落盘, JSON sidecar 写元数据。

调用方式
--------
默认 DEMO 模式: python hysteresis_mapping.py
真实硬件:       python hysteresis_mapping.py --live

与 dual_gate_mapping.py 的关系
-------------------------------
完全独立。 没有任何 import 关系。 共享代码 (DAQHardware, ChannelConfig,
GeometryConfig, Catppuccin GUI 框架等) 是手动复制过来并精简过的。
两个程序可以同时打开 (QSettings namespace 不同)。

v1.1 修复了 v1.0 的 23 个已知 bug, 详见 hysteresis_mapping_BUGFIX_CHECKLIST.md。
关键修复:
  · t_read 现在真的做硬件计时平均 (v1.0 是空字段, 元数据是假的)
  · NI-DAQmx Task 显式 start/stop
  · JSON sidecar 用 None 替代 NaN, 严格 RFC 7159 合规
  · DEMO/LIVE 通过 --live CLI flag 切换, 而不是源码常量
  · Mercury 控制器命令保留警告 + verify_communication() 接口
  · 多个 thread safety, GUI race, 边界量化 bug 修复
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
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

# ---------------------------------------------------------------------
# Optional pyvisa import — only needed for Mercury iPS controllers
# ---------------------------------------------------------------------
try:
    import pyvisa
    HAS_PYVISA = True
except Exception:
    HAS_PYVISA = False

# ---------------------------------------------------------------------
# Mode flag — set by main() from CLI args before any other module-level
# code that depends on it runs.  Default is True (DEMO) so the file can
# also be imported as a library / for tests without crashing.
# ---------------------------------------------------------------------
DEMO_MODE: bool = True

# nidaqmx is imported lazily in main() if --live is given. References
# to nidaqmx outside DEMO_MODE branches must use the module-level names
# set below by _import_daq_libs().
nidaqmx = None
TerminalConfiguration = None
AcquisitionType = None


def _import_daq_libs() -> None:
    """Import nidaqmx into module globals. Called by main() iff --live.
    Raises a helpful RuntimeError if nidaqmx isn't installed."""
    global nidaqmx, TerminalConfiguration, AcquisitionType
    try:
        import nidaqmx as _ndmx
        from nidaqmx.constants import (
            TerminalConfiguration as _TC,
            AcquisitionType as _AT,
        )
    except ImportError as e:
        raise RuntimeError(
            "--live mode requires nidaqmx. Install with:\n"
            "    pip install nidaqmx\n"
            "and ensure NI-DAQmx Runtime is installed on this machine.\n"
            f"Underlying ImportError: {e}"
        )
    nidaqmx = _ndmx
    TerminalConfiguration = _TC
    AcquisitionType = _AT


# ---------------------------------------------------------------------
# Qt + pyqtgraph
# ---------------------------------------------------------------------
from PyQt5.QtCore import (Qt, QThread, pyqtSignal, QTimer, QSettings, QSize,
                          QRectF)
from PyQt5.QtGui import QFont, QColor, QIcon, QKeySequence, QTransform
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QComboBox,
    QPushButton, QCheckBox, QGroupBox, QHBoxLayout, QVBoxLayout, QGridLayout,
    QSplitter, QFileDialog, QPlainTextEdit, QProgressBar, QStatusBar, QAction,
    QMenu, QToolButton, QSizePolicy, QFrame, QScrollArea, QMessageBox,
    QStyledItemDelegate,
)
import pyqtgraph as pg


# =====================================================================
# 常量
# =====================================================================

APP_NAME    = "Hysteresis Mapping"
APP_VERSION = "v1.1"
ORG_NAME    = "lab.transport"
SETTINGS_NAME = "hysteresis_mapping"

NUM_AI         = 8
NUM_LINE_SLOTS = 2     # 1D R(B) 曲线显示槽 (上下排列)
NUM_MAP_SLOTS  = 2     # 2D map 显示槽 (slot 0 永远 fwd, slot 1 永远 bwd)

# DAQ AI sample rate for hardware-timed t_read averaging
DAQ_AI_SAMPLE_RATE_HZ = 10000.0   # 10 kS/s per channel

# Mercury iPS GPIB timeout (single-query reads should be fast).
# Reduced from 5000 ms so abort can respond within ~1 s.
MERCURY_VISA_TIMEOUT_MS = 1000

# Stall detection grace period at start of each ramp direction.
# Real Mercury iPS takes 100ms-1s to begin moving after start_ramp().
RAMP_STARTUP_GRACE_S = 2.0

# Periodic CSV fsync (every N points). Loss-bound: < N * t_sample of data
# on power loss. Default N=10 → ≤ 10 sample times of loss.
CSV_FSYNC_EVERY_N = 10

# 物理常量
ELEM_CHARGE = 1.602176634e-19   # C
EPS0        = 8.8541878128e-12  # F/m
N_DISPLAY_TO_SI = 1.0e16        # 1e12 cm^-2 → m^-2
D_DISPLAY_TO_SI = 1.0e9         # V/nm → V/m

# 磁体 controller 类型
MAGNET_DEMO         = 'demo'
MAGNET_MERCURY_SCPI = 'mercury_scpi'
MAGNET_MERCURY_ISO  = 'mercury_iso'

# 通道 kind
CHANNEL_KINDS = ('R', 'Phase', 'Voltage')
KIND_UNIT     = {'R': 'Ω', 'Phase': '°', 'Voltage': 'V'}
KIND_CSV_UNIT = {'R': 'ohm', 'Phase': 'deg', 'Voltage': 'V'}

DIRECTIONS = ('fwd', 'bwd')


# =====================================================================
# Catppuccin Mocha 配色 + QSS
# =====================================================================

CT_BASE       = "#1e1e2e"
CT_MANTLE     = "#181825"
CT_CRUST      = "#11111b"
CT_SURFACE0   = "#313244"
CT_SURFACE1   = "#45475a"
CT_SURFACE2   = "#585b70"
CT_OVERLAY0   = "#6c7086"
CT_TEXT       = "#cdd6f4"
CT_SUBTEXT1   = "#bac2de"
CT_SUBTEXT0   = "#a6adc8"
CT_BLUE       = "#89b4fa"
CT_LAVENDER   = "#b4befe"
CT_SAPPHIRE   = "#74c7ec"
CT_TEAL       = "#94e2d5"
CT_GREEN      = "#a6e3a1"
CT_YELLOW     = "#f9e2af"
CT_PEACH      = "#fab387"
CT_RED        = "#f38ba8"
CT_MAUVE      = "#cba6f7"
CT_PINK       = "#f5c2e7"

# 1D 多通道叠加时, 每个通道一个 dash pattern (颜色 fwd=红 bwd=蓝固定)
CHANNEL_DASH_PATTERNS = [Qt.SolidLine, Qt.DashLine, Qt.DotLine, Qt.DashDotLine]

APP_QSS = f"""
QMainWindow, QWidget {{
    background-color: {CT_BASE};
    color: {CT_TEXT};
    font-family: "SF Pro Text", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 11pt;
}}
QGroupBox {{
    border: 1px solid {CT_SURFACE1};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: {CT_LAVENDER};
}}
QLineEdit, QComboBox, QPlainTextEdit {{
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
QLineEdit:disabled, QComboBox:disabled {{
    color: {CT_OVERLAY0};
    background-color: {CT_MANTLE};
}}
QPushButton {{
    background-color: {CT_SURFACE1};
    color: {CT_TEXT};
    border: 1px solid {CT_SURFACE2};
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {CT_SURFACE2};
}}
QPushButton:disabled {{
    color: {CT_OVERLAY0};
    background-color: {CT_MANTLE};
}}
QPushButton#startBtn {{
    background-color: {CT_GREEN};
    color: {CT_BASE};
}}
QPushButton#startBtn:hover {{
    background-color: {CT_TEAL};
}}
QPushButton#stopBtn {{
    background-color: {CT_RED};
    color: {CT_BASE};
}}
QPushButton#stopBtn:hover {{
    background-color: {CT_PEACH};
}}
QCheckBox {{ color: {CT_TEXT}; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {CT_SURFACE2};
    border-radius: 3px;
    background-color: {CT_SURFACE0};
}}
QCheckBox::indicator:checked {{
    background-color: {CT_BLUE};
    border: 1px solid {CT_BLUE};
}}
QLabel#sectionLabel {{ color: {CT_SUBTEXT1}; }}
QLabel#valueLabel   {{ color: {CT_GREEN}; font-family: "JetBrains Mono", "Consolas", monospace; }}
QLabel#stateLabel   {{ font-weight: 700; padding: 2px 10px; border-radius: 10px; }}
QProgressBar {{
    background-color: {CT_SURFACE0};
    border: 1px solid {CT_SURFACE1};
    border-radius: 4px;
    color: {CT_TEXT};
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {CT_BLUE};
    border-radius: 3px;
}}
QStatusBar {{
    background-color: {CT_MANTLE};
    color: {CT_SUBTEXT0};
    border-top: 1px solid {CT_SURFACE0};
}}
QMenuBar {{
    background-color: {CT_MANTLE};
    color: {CT_TEXT};
    border-bottom: 1px solid {CT_SURFACE0};
}}
QMenuBar::item:selected {{ background-color: {CT_SURFACE1}; }}
QMenu {{
    background-color: {CT_MANTLE};
    color: {CT_TEXT};
    border: 1px solid {CT_SURFACE1};
}}
QMenu::item:selected {{ background-color: {CT_SURFACE1}; }}
QScrollArea {{ background-color: {CT_BASE}; border: none; }}
QSplitter::handle {{ background-color: {CT_SURFACE0}; }}
QSplitter::handle:horizontal {{ width: 4px; }}
QSplitter::handle:vertical   {{ height: 4px; }}
QToolButton {{
    background-color: {CT_SURFACE0};
    color: {CT_TEXT};
    border: 1px solid {CT_SURFACE1};
    border-radius: 4px;
    padding: 4px 8px;
}}
QToolButton:hover {{ background-color: {CT_SURFACE1}; }}
"""


# =====================================================================
# 小工具函数
# =====================================================================

def fmt_dur(seconds: float) -> str:
    """秒数 → 'Xh Ym Zs' 风格字符串。"""
    if not (seconds == seconds):  # NaN
        return "—"
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _trapz(y, x):
    """numpy.trapezoid on numpy>=2, numpy.trapz on older. Tiny wrapper."""
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _json_clean(obj):
    """Recursively replace NaN/Infinity floats with None for RFC-7159
    compliant JSON output. (json.dumps writes 'NaN' literals by default
    which are rejected by strict parsers like MATLAB jsondecode and
    JavaScript JSON.parse.)
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
    if isinstance(obj, np.ndarray):
        return _json_clean(obj.tolist())
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(x) for x in obj]
    return obj


# =====================================================================
# 数据类
# =====================================================================

@dataclass
class ChannelConfig:
    """8 路 AI 中的一路。 kind 决定单位和转换公式。"""
    ai_index: int
    name:     str
    enabled:  bool
    kind:     str       # one of CHANNEL_KINDS
    sens:     float     # lockin sensitivity (V), only used when kind='R'
    gain:     float     # preamp gain, only used when kind='R'

    @property
    def unit(self) -> str:
        return KIND_UNIT.get(self.kind, '')

    @property
    def csv_unit(self) -> str:
        return KIND_CSV_UNIT.get(self.kind, '')

    @property
    def csv_col_name(self) -> str:
        u = self.csv_unit
        return f'{self.name}_{u}' if u else self.name


@dataclass
class GeometryConfig:
    """hBN 几何 + CNP + 限压。 用来在 (n, D) 和 (Vtg, Vbg) 之间双向映射。
    
    数学模型 (vdW 双栅):
        n = C_t (Vtg − Vtg0) + C_b (Vbg − Vbg0)        [SI: m^-2 charge density]
        2 ε0 D = C_t (Vtg − Vtg0) − C_b (Vbg − Vbg0)   [SI: V/m displacement field]
    """
    d_t_nm:  float       # top hBN thickness, nm
    d_b_nm:  float       # bottom hBN thickness, nm
    eps_hBN: float       # out-of-plane dielectric constant
    Vtg0:    float       # CNP, V
    Vbg0:    float       # CNP, V
    Vtg_max: float       # |Vtg| limit, V
    Vbg_max: float       # |Vbg| limit, V

    @property
    def C_t(self) -> float:
        """top gate capacitance per area, F/m^2."""
        return self.eps_hBN * EPS0 / (self.d_t_nm * 1e-9)

    @property
    def C_b(self) -> float:
        return self.eps_hBN * EPS0 / (self.d_b_nm * 1e-9)

    def vgates_to_nD(self, Vtg: float, Vbg: float) -> Tuple[float, float]:
        """(Vtg, Vbg) [V] → (n_charge_density [m^-2], D [V/m])."""
        n_SI = self.C_t * (Vtg - self.Vtg0) + self.C_b * (Vbg - self.Vbg0)
        n_charge = n_SI / ELEM_CHARGE
        D_SI = (self.C_t * (Vtg - self.Vtg0)
                - self.C_b * (Vbg - self.Vbg0)) / (2.0 * EPS0)
        return n_charge, D_SI

    def nD_to_vgates(self, n_SI: float, D_SI: float) -> Tuple[float, float]:
        """(n_charge_density [m^-2], D [V/m]) → (Vtg, Vbg) [V]."""
        ne = n_SI * ELEM_CHARGE
        twoDeps0 = 2.0 * D_SI * EPS0
        Vtg = self.Vtg0 + (ne + twoDeps0) / (2.0 * self.C_t)
        Vbg = self.Vbg0 + (ne - twoDeps0) / (2.0 * self.C_b)
        return Vtg, Vbg

    def is_within_limits(self, Vtg: float, Vbg: float) -> bool:
        eps = 1e-9
        return abs(Vtg) <= self.Vtg_max + eps and abs(Vbg) <= self.Vbg_max + eps


@dataclass
class MagnetConfig:
    """磁体控制器的全部配置。"""
    controller_type:     str        # one of MAGNET_*
    visa_resource:       str        # pyvisa resource string, ignored for DEMO
    magnet_uid:          str        # SCPI device UID, e.g. 'GRPZ' for main z
    B_min_T:             float
    B_max_T:             float
    ramp_rate_T_per_min: float
    B_step_T:            float       # logical sampling interval in T
    field_tolerance_T:   float       # for is_at_target judgement

    @property
    def t_sample_s(self) -> float:
        """每个采样点之间的目标时间间隔 (s)。"""
        rate = max(self.ramp_rate_T_per_min, 1e-9)
        return self.B_step_T / (rate / 60.0)

    @property
    def num_B_points(self) -> int:
        """fwd 方向上 canonical B grid 的点数 (= bwd 的点数)。"""
        if self.B_step_T <= 0:
            return 1
        return int(round((self.B_max_T - self.B_min_T) / self.B_step_T)) + 1


@dataclass
class HysteresisConfig:
    """整个磁滞扫描任务的高层描述。 GUI 解析后构造一个这个对象交给 thread。"""
    geometry:      GeometryConfig
    magnet_config: MagnetConfig
    outer_axis:    str            # 'n' or 'D'
    outer_min:     float          # display units (1e12 cm^-2 if 'n', V/nm if 'D')
    outer_max:     float          # ditto
    num_outer:     int
    fixed_value:   float          # display units of the *other* axis
    # 时序参数 (s)
    t_read:               float = 0.05
    t_settle_slow:        float = 0.5
    t_settle_after_fwd:   float = 0.5
    t_settle_after_bwd:   float = 0.5
    # 物理 AO 通道
    ao_top: str = 'Dev1/ao0'
    ao_bot: str = 'Dev1/ao1'
    # 滞回面积算哪个 AI 通道 (索引 0..NUM_AI-1)
    hysteresis_channel_ai: int = 0

    @property
    def fixed_axis(self) -> str:
        return 'D' if self.outer_axis == 'n' else 'n'

    def outer_array(self) -> np.ndarray:
        """外层变量的 display-unit 数组。"""
        return np.linspace(self.outer_min, self.outer_max, self.num_outer)

    def outer_to_gates(self, outer_value: float) -> Tuple[float, float]:
        """把一个外层 display-unit 值反推成 (Vtg, Vbg) [V]。"""
        if self.outer_axis == 'n':
            n_disp, D_disp = outer_value, self.fixed_value
        else:
            n_disp, D_disp = self.fixed_value, outer_value
        n_SI = n_disp * N_DISPLAY_TO_SI
        D_SI = D_disp * D_DISPLAY_TO_SI
        return self.geometry.nD_to_vgates(n_SI, D_SI)

    def precheck_any_out_of_limits(self) -> Optional[Tuple[int, float, float, float]]:
        """检查全部外层点是否都在 hBN 限压内。
        
        返回 None 表示全部合法; 否则返回第一个超界点的
        (i_idx, outer_value_disp, Vtg, Vbg)。
        """
        for i, v in enumerate(self.outer_array()):
            Vtg, Vbg = self.outer_to_gates(float(v))
            if not self.geometry.is_within_limits(Vtg, Vbg):
                return (i, float(v), Vtg, Vbg)
        return None


@dataclass
class PointInfo:
    """单个采样点的载荷, MeasurementThread → GUI 信号。"""
    outer_target: float        # display units of outer axis
    B_requested:  float        # quantized to canonical B grid (T)
    B_actual:     float        # raw read from magnet (T)
    Vtg:          float        # V
    Vbg:          float        # V
    values:       List[float]  # length NUM_AI, converted (R/Phase/V)
    j_idx:        int          # B grid index in [0, num_B-1]
    i_idx:        int          # outer index in [0, num_outer-1]
    direction:    str          # 'fwd' or 'bwd'


# =====================================================================
# Preisach-like hysteresis model for DEMO data synthesis
# =====================================================================

def preisach_R_of_B(B: float, direction: str,
                    H_c: float = 1.5,
                    M_s: float = 1.0,
                    R_bg: float = 0.5,
                    R_lin: float = 0.0,
                    seed_ai: int = 0) -> float:
    """合成一个 R(B) 值, 模拟有矫顽力的铁磁/反铁磁滞回。

    数学:
        M_fwd(B) = M_s * tanh((B − H_c) / w)
        M_bwd(B) = M_s * tanh((B + H_c) / w)
        R(B, dir) = R_bg + R_lin * B + alpha * M_dir(B)

    其中 w = H_c / 3 是过渡宽度 (软化 step), alpha 是磁阻耦合系数。
    fwd 和 bwd 的差异只在 |B| < ~2 H_c 范围内显著, 远场重合, 这正是
    真实铁磁/AFM skyrmion 系统的滞回行为。

    seed_ai 用来给不同 AI 通道一点点不同的 alpha, 让 8 个通道的曲线
    长得不一样, 方便 GUI 测试。
    """
    w = H_c / 3.0
    alpha = 0.4 + 0.05 * seed_ai
    if direction == 'fwd':
        M = M_s * math.tanh((B - H_c) / w)
    else:
        M = M_s * math.tanh((B + H_c) / w)
    return R_bg + R_lin * B + alpha * M


# =====================================================================
# 1.  磁体控制器
# =====================================================================
#
# 三个实现:
#   · DemoMagnet         — 纯 Python 模拟
#   · MercuryIpsScpi     — Oxford Mercury iPS, SCPI 命令集
#   · MercuryIpsIsobus   — Oxford Mercury iPS, 旧 ISOBUS 命令集
#
# 选择哪个由 GUI 下拉决定。 真实硬件试 SCPI 不响应再切 ISOBUS。
#
# 关键约定:
#   · connect(): 建立通信, 任何失败都 raise
#   · read_field_T(): 实时读当前 B (T)。 真机走 GPIB query, DEMO 走内部模型
#   · set_target / set_ramp_rate: 只设值, 不启动 ramp
#   · start_ramp(): 命令磁体开始向 target ramp
#   · hold(): 立即停在当前位置 (HOLD 模式), ramp 中断
#   · is_at_target(tol): True 表示 |current - target| < tol
#   · disconnect(): 释放资源, 永远不在 finally 之外被调用
# ---------------------------------------------------------------------


class MagnetController:
    """抽象基类。 子类必须实现 connect / disconnect / read_field_T /
    set_ramp_rate / set_target / start_ramp / hold / is_at_target。
    """
    name = "base"

    def __init__(self, config: MagnetConfig = None):
        self.config = config
        self.connected = False
        self._target = 0.0

    def connect(self):
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError

    def read_field_T(self) -> float:
        raise NotImplementedError

    def set_ramp_rate(self, rate_T_per_min: float):
        raise NotImplementedError

    def set_target(self, B_T: float):
        self._target = float(B_T)

    def start_ramp(self):
        raise NotImplementedError

    def hold(self):
        """Stop wherever we are. Idempotent."""
        raise NotImplementedError

    def is_at_target(self, tol_T: float = None) -> bool:
        """Return True if |B_now - target| < tol。
        默认 tol 用 config.field_tolerance_T。"""
        if tol_T is None:
            tol_T = self.config.field_tolerance_T if self.config else 0.01
        try:
            B = self.read_field_T()
        except Exception:
            return False
        return abs(B - self._target) <= tol_T

    def verify_communication(self) -> Dict[str, Any]:
        """★ v1.1: read-only sanity check, must be implemented by subclasses
        for safe pre-flight on real hardware."""
        return {'errors': ['verify_communication not implemented for ' + self.name]}

    def describe(self) -> str:
        """Human-readable controller summary for log lines."""
        return self.name


# ---------------------------------------------------------------------
# 1a. DemoMagnet — pure-Python linear-ramp simulation
# ---------------------------------------------------------------------

class DemoMagnet(MagnetController):
    """Demo: 模拟一个理想线性 ramp 磁体。 connect() 自动 pre-position
    到 B_min, 这样 strict start check 在 DEMO 下永远通过。
    
    不计入加速度/惯性, ramp 是真的瞬时启动到目标 rate。
    """
    name = "demo"

    def __init__(self, config=None):
        super().__init__(config)
        self._B_current       = 0.0
        self._B_at_ramp_start = 0.0
        self._rate_T_per_min  = 0.5     # default
        self._ramp_start_t    = None    # time.monotonic() when current ramp began
        self._holding         = True

    def connect(self):
        # Pre-position to B_min so strict check passes
        if self.config is not None:
            try:
                self._B_current       = float(self.config.B_min_T)
                self._target          = self._B_current
                self._B_at_ramp_start = self._B_current
                self._rate_T_per_min  = float(self.config.ramp_rate_T_per_min)
            except Exception:
                pass
        self._holding      = True
        self._ramp_start_t = None
        self.connected     = True

    def disconnect(self):
        self.connected = False

    def _update_B(self):
        """Recompute self._B_current based on elapsed time since ramp start."""
        if self._holding or self._ramp_start_t is None:
            return
        now = time.monotonic()
        elapsed_min = (now - self._ramp_start_t) / 60.0
        rate = max(self._rate_T_per_min, 1e-6)
        delta = rate * elapsed_min
        if self._target >= self._B_at_ramp_start:
            new_B = min(self._B_at_ramp_start + delta, self._target)
        else:
            new_B = max(self._B_at_ramp_start - delta, self._target)
        self._B_current = new_B
        if abs(new_B - self._target) < 1e-9:
            self._holding      = True
            self._ramp_start_t = None

    def read_field_T(self) -> float:
        self._update_B()
        # Tiny noise to mimic real sensor jitter (~10 µT)
        return self._B_current + np.random.randn() * 1e-5

    def set_ramp_rate(self, rate_T_per_min: float):
        self._rate_T_per_min = float(rate_T_per_min)

    def start_ramp(self):
        self._update_B()
        self._B_at_ramp_start = self._B_current
        self._ramp_start_t    = time.monotonic()
        self._holding         = False

    def hold(self):
        self._update_B()
        self._holding      = True
        self._ramp_start_t = None
        self._target       = self._B_current

    def verify_communication(self) -> Dict[str, Any]:
        """★ v1.1: DEMO returns canonical 'ok' values."""
        return {
            'idn':                  'DEMO MAGNET (simulated, no hardware)',
            'field_T':              self.read_field_T(),
            'setpoint_T':           self._target,
            'ramp_rate_T_per_min':  self._rate_T_per_min,
            'persistent_switch_raw': 'N/A (DEMO has no persistent switch)',
            'errors':               [],
        }

    def describe(self) -> str:
        return f"DEMO magnet (rate={self._rate_T_per_min} T/min, B_now={self._B_current:+.4f} T)"


# ---------------------------------------------------------------------
# 1b. MercuryIpsScpi — Oxford Mercury iPS, modern SCPI command set
# ---------------------------------------------------------------------

class MercuryIpsScpi(MagnetController):
    """Oxford Mercury iPS, modern SCPI firmware.

    ⚠️  WARNING — UNVERIFIED ON HARDWARE  ⚠️
    The exact SCPI command set differs between Mercury iPS firmware
    versions. Before first production use:
      1. Read your unit's serial-port command reference manual
         (Oxford ships it as a PDF; ask your facility manager if missing).
      2. Verify each command listed below against your manual. The
         commands here are based on the Mercury iPS SCPI guide as of
         mid-2024 firmware.
      3. Run a low-stakes test FIRST: in the GUI, click the
         "Verify magnet comms" button (it calls verify_communication()
         which only does READs, no writes). Visually compare the values
         to the Mercury front panel.
      4. Then do a tiny ramp test: set ramp_rate=0.05 T/min,
         B_min=current_field, B_max=current_field+0.005 T, watch the
         front panel and confirm it ramps the right direction at the
         right rate.
      5. If anything doesn't match, the magnet may quench or the
         persistent switch may behave unexpectedly. DO NOT skip
         verification.

    SCPI command pattern (UID = 'GRPZ' for main z magnet, configurable):
        READ:DEV:<UID>:PSU:SIG:FLD          → query measured field [T]
        READ:DEV:<UID>:PSU:SIG:FSET         → query field setpoint [T]
        READ:DEV:<UID>:PSU:SIG:RFST         → query ramp-rate setpoint [T/min]
        READ:DEV:<UID>:PSU:SIG:SWHT         → query persistent-switch state
        SET:DEV:<UID>:PSU:SIG:RFST:<rate>   → set ramp-rate
        SET:DEV:<UID>:PSU:SIG:FSET:<target> → set target field
        SET:DEV:<UID>:PSU:ACTN:RTOS         → ramp to set
        SET:DEV:<UID>:PSU:ACTN:HOLD         → hold

    Responses are like:
        STAT:DEV:GRPZ:PSU:SIG:FLD:1.2345T
    The parser strips the 'STAT:...:' prefix and trailing unit suffix
    (e.g., 'T' or 'T/m') before float-converting.
    """
    name = "mercury_scpi"

    def __init__(self, config: MagnetConfig):
        super().__init__(config)
        self._rm   = None
        self._inst = None

    def connect(self):
        if not HAS_PYVISA:
            raise RuntimeError("pyvisa not installed — run `pip install pyvisa` first.")
        self._rm   = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(self.config.visa_resource)
        # ★ v1.1: shorter timeout for responsiveness (Bug #19)
        self._inst.timeout       = MERCURY_VISA_TIMEOUT_MS
        self._inst.read_termination  = '\n'
        self._inst.write_termination = '\n'
        try:
            idn = self._inst.query("*IDN?")
        except Exception as e:
            raise RuntimeError(f"Mercury *IDN? failed: {e}")
        if "MERCURY" not in idn.upper():
            raise RuntimeError(
                f"Device at {self.config.visa_resource!r} does not identify "
                f"as Mercury (IDN: {idn!r}). Try the ISOBUS controller, "
                f"or check the VISA resource string.")
        self.connected = True

    def disconnect(self):
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
        self._inst    = None
        self._rm      = None
        self.connected = False

    def _strip_response(self, raw: str, suffix_unit: str) -> float:
        """Extract a float from a Mercury response, e.g.:
          raw='STAT:DEV:GRPZ:PSU:SIG:FLD:1.2345T', suffix_unit='T' → 1.2345

        ★ v1.1 (Bug #3): uses literal-suffix strip instead of rstrip()
        which would otherwise strip any character in the suffix set
        ('T', '/', 'm') and over-eat the value. Also defensively handles
        the 'mT' (millitesla) case by converting to T.
        """
        s = raw.strip()
        # Defensive: handle mT misconfiguration (Mercury can return mT
        # if front panel units are wrong — convert to T silently)
        unit_in_response = None
        for cand in ('mT', 'T/m', 'T', 'A/s', 'A', 'V'):
            if s.endswith(cand):
                unit_in_response = cand
                s = s[:-len(cand)].strip()
                break

        # Take the part after the last ':' as the value
        idx = s.rfind(':')
        if idx >= 0:
            s = s[idx+1:].strip()

        try:
            val = float(s)
        except ValueError as e:
            raise RuntimeError(
                f"Could not parse Mercury response {raw!r}: expected suffix "
                f"{suffix_unit!r}, got value {s!r}. Check the magnet's "
                f"current units configuration on the front panel."
            ) from e

        # Convert mT → T if needed
        if unit_in_response == 'mT' and suffix_unit == 'T':
            val *= 1e-3

        return val

    def read_field_T(self) -> float:
        uid = self.config.magnet_uid
        raw = self._inst.query(f"READ:DEV:{uid}:PSU:SIG:FLD")
        return self._strip_response(raw, 'T')

    def set_ramp_rate(self, rate_T_per_min: float):
        uid = self.config.magnet_uid
        cmd = f"SET:DEV:{uid}:PSU:SIG:RFST:{rate_T_per_min:.5f}"
        self._inst.write(cmd)

    def set_target(self, B_T: float):
        super().set_target(B_T)
        uid = self.config.magnet_uid
        cmd = f"SET:DEV:{uid}:PSU:SIG:FSET:{B_T:.5f}"
        self._inst.write(cmd)

    def start_ramp(self):
        uid = self.config.magnet_uid
        self._inst.write(f"SET:DEV:{uid}:PSU:ACTN:RTOS")

    def hold(self):
        uid = self.config.magnet_uid
        try:
            self._inst.write(f"SET:DEV:{uid}:PSU:ACTN:HOLD")
        except Exception:
            pass

    def verify_communication(self) -> Dict[str, Any]:
        """★ v1.1 Bug #2: read-only round-trip query for safety testing.
        
        Reads field, setpoint, ramp rate, persistent switch state and
        returns them in a dict. Safe to call before first production run
        — does no SET commands. The user should compare these values to
        the Mercury front panel before trusting any SET commands.

        Returns dict with keys: 'idn', 'field_T', 'setpoint_T',
        'ramp_rate_T_per_min', 'persistent_switch_raw'. Any field that
        fails to query is set to None and an 'errors' key is added.
        """
        uid = self.config.magnet_uid
        result: Dict[str, Any] = {'errors': []}

        def _try(key, query_fn):
            try:
                result[key] = query_fn()
            except Exception as e:
                result[key] = None
                result['errors'].append(f"{key}: {e}")

        _try('idn',
             lambda: self._inst.query("*IDN?").strip())
        _try('field_T',
             lambda: self._strip_response(
                 self._inst.query(f"READ:DEV:{uid}:PSU:SIG:FLD"), 'T'))
        _try('setpoint_T',
             lambda: self._strip_response(
                 self._inst.query(f"READ:DEV:{uid}:PSU:SIG:FSET"), 'T'))
        _try('ramp_rate_T_per_min',
             lambda: self._strip_response(
                 self._inst.query(f"READ:DEV:{uid}:PSU:SIG:RFST"), 'T/m'))
        _try('persistent_switch_raw',
             lambda: self._inst.query(f"READ:DEV:{uid}:PSU:SIG:SWHT").strip())

        return result

    def describe(self) -> str:
        return (f"Mercury iPS SCPI @ {self.config.visa_resource} "
                f"(UID={self.config.magnet_uid})")


# ---------------------------------------------------------------------
# 1c. MercuryIpsIsobus — legacy ISOBUS short command set
# ---------------------------------------------------------------------

class MercuryIpsIsobus(MagnetController):
    """Oxford Mercury iPS / IPS120, legacy ISOBUS short commands.

    ⚠️  WARNING — UNVERIFIED ON HARDWARE  ⚠️
    Same caveat as MercuryIpsScpi above. ISOBUS commands have varied
    historically across IPS120, IPS180, Mercury-in-legacy-mode, etc.
    Especially for the field setpoint command:
      - Some firmware: J<val> sets field setpoint
      - Other firmware: J<val> sets persistent-mode field (DANGEROUS if
        misinterpreted — could trigger persistent switch heater)
      - Some firmware uses I<val> instead
    VERIFY YOUR UNIT'S MANUAL before first use. Click the
    "Verify magnet comms" button to do read-only checks first.

    Command pattern (1-letter ASCII):
        V       → query version (sanity ping)
        R7      → read measured field (T)
        R8      → read field setpoint (T)
        R9      → read ramp rate (T/min)
        R16     → read persistent-switch heater status
        T<rate> → set ramp rate (T/min)
        J<targ> → set target field (T)  [VERIFY this on your unit]
        A1      → "to setpoint" (start ramp)
        A0      → "hold"
        C3      → remote + unlocked
        C0      → local

    If commands fail or behave unexpectedly, switch to the SCPI variant.
    """
    name = "mercury_iso"

    def __init__(self, config: MagnetConfig):
        super().__init__(config)
        self._rm   = None
        self._inst = None

    def connect(self):
        if not HAS_PYVISA:
            raise RuntimeError("pyvisa not installed — run `pip install pyvisa` first.")
        self._rm   = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(self.config.visa_resource)
        # ★ v1.1: shorter timeout (Bug #19)
        self._inst.timeout           = MERCURY_VISA_TIMEOUT_MS
        self._inst.read_termination  = '\r'
        self._inst.write_termination = '\r'
        try:
            self._inst.write("C3")          # remote + unlocked
            ver = self._inst.query("V")
        except Exception as e:
            raise RuntimeError(f"Mercury ISOBUS V command failed: {e}")
        if not ver:
            raise RuntimeError(f"No response from {self.config.visa_resource!r} on V command.")
        self.connected = True

    def disconnect(self):
        try:
            if self._inst is not None:
                self._inst.write("A0")      # hold
                self._inst.write("C0")      # back to local
                self._inst.close()
        except Exception:
            pass
        try:
            if self._rm is not None:
                self._rm.close()
        except Exception:
            pass
        self._inst    = None
        self._rm      = None
        self.connected = False

    def _parse_R(self, raw: str) -> float:
        """ISOBUS responses to R<n> look like 'R+1.2345' — strip leading 'R'."""
        s = raw.strip()
        if s.startswith('R'):
            s = s[1:]
        try:
            return float(s)
        except ValueError as e:
            raise RuntimeError(
                f"Could not parse ISOBUS response {raw!r} as float."
            ) from e

    def read_field_T(self) -> float:
        return self._parse_R(self._inst.query("R7"))

    def set_ramp_rate(self, rate_T_per_min: float):
        self._inst.write(f"T{rate_T_per_min:.4f}")

    def set_target(self, B_T: float):
        super().set_target(B_T)
        self._inst.write(f"J{B_T:.4f}")

    def start_ramp(self):
        self._inst.write("A1")

    def hold(self):
        try:
            self._inst.write("A0")
        except Exception:
            pass

    def verify_communication(self) -> Dict[str, Any]:
        """★ v1.1 Bug #2: read-only verification round-trip."""
        result: Dict[str, Any] = {'errors': []}

        def _try(key, fn):
            try:
                result[key] = fn()
            except Exception as e:
                result[key] = None
                result['errors'].append(f"{key}: {e}")

        _try('version',     lambda: self._inst.query("V").strip())
        _try('field_T',     lambda: self._parse_R(self._inst.query("R7")))
        _try('setpoint_T',  lambda: self._parse_R(self._inst.query("R8")))
        _try('ramp_rate_T_per_min',
                            lambda: self._parse_R(self._inst.query("R9")))
        _try('persistent_switch_raw',
                            lambda: self._inst.query("R16").strip())

        return result

    def describe(self) -> str:
        return f"Mercury iPS ISOBUS @ {self.config.visa_resource}"


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

def make_magnet_controller(config: MagnetConfig) -> MagnetController:
    """Construct the right MagnetController for a given config."""
    t = config.controller_type
    if t == MAGNET_DEMO:
        return DemoMagnet(config)
    if t == MAGNET_MERCURY_SCPI:
        return MercuryIpsScpi(config)
    if t == MAGNET_MERCURY_ISO:
        return MercuryIpsIsobus(config)
    raise ValueError(f"Unknown magnet controller type: {t!r}")


# =====================================================================
# 2.  DAQHardware  —  NI DAQ wrapper with hardware-timed averaging
# =====================================================================

class DAQHardware:
    """NI DAQ AO + AI 封装。 在 DEMO_MODE 下完全用 numpy 合成数据。

    AI 采集模型 (v1.1):
      · 用户在 GUI 里指定 t_read (默认 50 ms) — 每次 read_ai(t_read) 调用
        会在 t_read 时间窗内做硬件计时采样, 然后每个通道求平均, 返回
        长度 NUM_AI 的列表。
      · 真机模式: 用 nidaqmx hardware-timed FINITE acquisition,
        sample rate = DAQ_AI_SAMPLE_RATE_HZ, samples/chan = round(t_read*rate)。
      · DEMO 模式: 在 t_read 窗内多次合成样本然后求平均, 噪声按
        1/sqrt(N) 缩减, 模拟真实 lockin 噪声减小。

    DEMO 数据合成:
      · 偶数 AI (0, 2, 4, 6) → R-like, Preisach hysteresis + 栅压调制
      · 奇数 AI (1, 3, 5, 7) → Phase-like, 平滑 sin 调制 + 弱磁滞
      · MeasurementThread 在每次 read_ai 之前更新 self._demo_magnet_B_T
        和 self._demo_direction 来注入磁场状态 (DEMO 模式专用)。

    线程安全: open/close/read_ai/write_ao 必须只从 measurement worker
    线程调用; GUI 线程不可直接调。 _last_ao 字典在两个线程中只读取, 写
    入只在 worker 中, 64-bit float 的赋值在 CPython GIL 下是原子的。
    """

    def __init__(self,
                 ao_chans=('Dev1/ao0', 'Dev1/ao1'),
                 ai_chans=('Dev1/ai0', 'Dev1/ai1', 'Dev1/ai2', 'Dev1/ai3',
                           'Dev1/ai4', 'Dev1/ai5', 'Dev1/ai6', 'Dev1/ai7')):
        self.ao_chans  = list(ao_chans)
        self.ai_chans  = list(ai_chans)
        self._ao_tasks = {}
        self._ai_task  = None
        self._ai_sample_rate_hz = DAQ_AI_SAMPLE_RATE_HZ
        # Last requested samples-per-channel — to avoid reconfiguring the
        # task every read when t_read hasn't changed
        self._ai_n_samps_cached = 0
        self._last_ao  = {ch: 0.0 for ch in self.ao_chans}
        self.connected = False
        # DEMO state — set by MeasurementThread before each read_ai (DEMO only)
        self._demo_magnet_B_T = 0.0
        self._demo_direction  = 'fwd'

    def open(self):
        """Allocate + start NI tasks. Call once before any read/write."""
        if DEMO_MODE:
            self.connected = True
            return
        # AO tasks: one per channel, started immediately for on-demand writes
        for ch in self.ao_chans:
            t = nidaqmx.Task()
            t.ao_channels.add_ao_voltage_chan(ch, min_val=-10.0, max_val=10.0)
            t.start()                                     # ★ explicit start
            self._ao_tasks[ch] = t
        # AI task: hardware-timed FINITE acquisition. We re-arm it every
        # read_ai call by updating samps_per_chan and calling start/stop.
        self._ai_task = nidaqmx.Task()
        for ch in self.ai_chans:
            self._ai_task.ai_channels.add_ai_voltage_chan(
                ch, terminal_config=TerminalConfiguration.RSE,
                min_val=-10.0, max_val=10.0)
        # Configure timing once with a reasonable default; will be re-armed
        # per read_ai call to match requested t_read.
        default_n = int(round(0.05 * self._ai_sample_rate_hz))   # 50ms default
        self._ai_task.timing.cfg_samp_clk_timing(
            rate=self._ai_sample_rate_hz,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=default_n,
        )
        self._ai_n_samps_cached = default_n
        # Note: do NOT start the AI task here — it will be started+stopped
        # per read_ai call (FINITE acquisitions auto-stop at samps_per_chan).
        self.connected = True

    def close(self):
        """Stop + close all tasks. Idempotent."""
        if DEMO_MODE:
            self.connected = False
            return
        for t in self._ao_tasks.values():
            try:
                t.stop()                                   # ★ stop before close
            except Exception:
                pass
            try:
                t.close()
            except Exception:
                pass
        self._ao_tasks.clear()
        if self._ai_task is not None:
            try:
                self._ai_task.stop()
            except Exception:
                pass
            try:
                self._ai_task.close()
            except Exception:
                pass
            self._ai_task = None
        self.connected = False

    def write_ao(self, channel: str, value: float):
        v = float(np.clip(value, -10.0, 10.0))
        if abs(v - self._last_ao.get(channel, 0.0)) < 1e-9:
            return
        self._last_ao[channel] = v
        if DEMO_MODE:
            return
        self._ao_tasks[channel].write(v)

    def ramp_ao(self, channel: str, v_target: float,
                rate_v_per_s: float = 2.0, step: float = 0.02):
        """Software-timed AO ramp. NOT guaranteed deterministic under heavy
        CPU load. For safety-critical ramping (e.g., shutdown to 0V) consider
        a hardware-timed AO buffer instead."""
        v0 = self._last_ao.get(channel, 0.0)
        v_target = float(np.clip(v_target, -10.0, 10.0))
        if abs(v_target - v0) < 1e-6:
            return
        n = max(2, int(np.ceil(abs(v_target - v0) / step)))
        dwell = abs(v_target - v0) / max(rate_v_per_s, 1e-3) / n
        for v in np.linspace(v0, v_target, n):
            self.write_ao(channel, v)
            time.sleep(dwell)

    def read_ai(self, t_read: float = 0.05) -> List[float]:
        """Acquire averaged AI readings over t_read seconds.

        Returns: list of NUM_AI floats, each = mean over the t_read window.

        For LIVE mode: hardware-timed FINITE acquisition at
        DAQ_AI_SAMPLE_RATE_HZ, samples_per_chan = round(t_read * rate).
        Total wall time per call ≈ t_read + ~5 ms task overhead.

        For DEMO mode: synthesizes N = max(1, round(t_read * 1000)) samples
        and averages, with noise scaled by 1/sqrt(N) to mimic real lockin
        SNR improvement with longer integration.

        t_read floor = 1 ms (minimum 10 hardware samples at 10 kS/s).
        """
        t_read = max(float(t_read), 1e-3)

        if DEMO_MODE:
            # Software-timed averaging in DEMO; noise scales as 1/sqrt(N)
            n_samp = max(1, int(round(t_read * 1000.0)))
            ao0 = self._last_ao.get(self.ao_chans[0], 0.0)
            ao1 = self._last_ao.get(self.ao_chans[1], 0.0)
            B = self._demo_magnet_B_T
            d = self._demo_direction
            noise_scale = 1.0 / math.sqrt(n_samp)
            data = []
            for ai in range(NUM_AI):
                if ai % 2 == 0:
                    # R-like
                    base = preisach_R_of_B(B, d, seed_ai=ai)
                    base += 0.05 * math.sin(2.0 * ao0 + ai // 2) * math.cos(1.3 * ao1)
                    val = base + 0.005 * noise_scale * np.random.randn()
                else:
                    # Phase-like
                    val = 0.1 * math.sin(ao0 + ai // 2)
                    val += 0.02 * preisach_R_of_B(B, d, seed_ai=ai)
                    val += 0.005 * noise_scale * np.random.randn()
                data.append(float(val))
            # Mimic the real measurement's wall-time so DEMO timing matches
            time.sleep(t_read)
            return data

        # ---- LIVE mode ----
        n_samps = max(10, int(round(t_read * self._ai_sample_rate_hz)))

        # Re-arm timing if t_read changed since last call
        if n_samps != self._ai_n_samps_cached:
            # Must stop before reconfiguring timing
            try:
                self._ai_task.stop()
            except Exception:
                pass
            self._ai_task.timing.cfg_samp_clk_timing(
                rate=self._ai_sample_rate_hz,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=n_samps,
            )
            self._ai_n_samps_cached = n_samps

        # Start, read, stop. Timeout = t_read + generous overhead.
        self._ai_task.start()
        try:
            raw = self._ai_task.read(
                number_of_samples_per_channel=n_samps,
                timeout=t_read * 2.0 + 1.0,
            )
        finally:
            try:
                self._ai_task.stop()
            except Exception:
                pass

        # raw shape: list-of-lists [NUM_AI][n_samps] (or list of floats if
        # NUM_AI=1, but we always have ≥2 channels so it's nested).
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 1:
            # Defensive — single-channel edge case
            return [float(arr.mean())]
        return [float(x) for x in arr.mean(axis=1)]


# =====================================================================
# 3.  MeasurementThread  —  the magnetic sweep + sampling loop
# =====================================================================

class MeasurementThread(QThread):
    """背景线程: 严格起点检查 + 外层循环 + 双向连续 ramp + 节拍采样 +
    滞回面积计算 + CSV/JSON 落盘。
    
    Signals:
      · point_ready(PointInfo)            — 每个采样点
      · outer_point_finished(int, float)  — 一个外层点扫完, 带 (i_idx, hyst_area)
      · log_msg(str, str)                 — (text, level) 给 GUI log panel
      · finished_ok()
      · error_occurred(str)               — full traceback string
    """
    point_ready          = pyqtSignal(object)
    outer_point_finished = pyqtSignal(int, float)
    log_msg              = pyqtSignal(str, str)
    finished_ok          = pyqtSignal()
    error_occurred       = pyqtSignal(str)

    def __init__(self, params: dict, hw: DAQHardware):
        super().__init__()
        self.params = params
        self.hw     = hw
        self.is_running = True

        self.cfg:      HysteresisConfig = params['hysteresis_config']
        self.magnet:   MagnetController = params['magnet']
        self.channels: List[ChannelConfig] = params['channels']
        self.v_osc      = float(params['v_osc'])
        self.r_series   = float(params['r_series'])
        # AC bias current
        self.i_ac = self.v_osc / max(self.r_series, 1e-12)

        self._started_at_iso = None
        self._initial_B_at_start = float('nan')
        self._hyst_areas: List[float] = []   # filled per outer point
        self._points_written = 0             # ★ v1.1: for periodic fsync

    # -----------------------------------------------------------------
    def _convert(self, raw_ai: List[float]) -> List[float]:
        """raw V → physical units according to each channel's kind."""
        out = []
        for ch, raw in zip(self.channels, raw_ai):
            if not ch.enabled:
                out.append(float('nan'))
                continue
            if ch.kind == 'R':
                # R = (V_lockin / sens) × sens_full × (1/gain) / I_ac
                # Simplification used here: V_meas = raw * (sens / 10) / gain,
                # then R = V_meas / I_ac. (Same as dual_gate's formula.)
                v_meas = raw * (ch.sens / 10.0) / max(ch.gain, 1e-12)
                R = v_meas / max(self.i_ac, 1e-18)
                out.append(R)
            elif ch.kind == 'Phase':
                # 18°/V is SR830 default phase output scaling
                out.append(raw * 18.0)
            else:  # Voltage
                out.append(raw)
        return out

    # -----------------------------------------------------------------
    def _write_csv_header(self, writer):
        p   = self.params
        cfg = self.cfg
        mc  = cfg.magnet_config
        g   = cfg.geometry
        writer.writerow([f"# {APP_NAME} {APP_VERSION}"])
        writer.writerow([f"# Started at {self._started_at_iso}"])
        writer.writerow([f"# Sample={p.get('sample','')!r} | Device={p.get('device','')!r} | "
                         f"Operator={p.get('operator','')!r} | Run={p.get('run_name','')!r}"])
        writer.writerow([f"# I_ac [A] = {self.i_ac:.6e}"])
        writer.writerow([f"# Magnet: {mc.controller_type} @ {mc.visa_resource!r} (UID={mc.magnet_uid!r})"])
        writer.writerow([f"# Magnet sweep: B in [{mc.B_min_T}, {mc.B_max_T}] T, "
                         f"rate={mc.ramp_rate_T_per_min} T/min, "
                         f"B_step={mc.B_step_T} T, tol={mc.field_tolerance_T} T"])
        writer.writerow([f"# Outer: {cfg.outer_axis} in [{cfg.outer_min}, {cfg.outer_max}] "
                         f"({cfg.num_outer} pts), fixed {cfg.fixed_axis}={cfg.fixed_value}"])
        writer.writerow([f"# Geometry: d_t={g.d_t_nm}nm d_b={g.d_b_nm}nm eps={g.eps_hBN} "
                         f"Vtg0={g.Vtg0} Vbg0={g.Vbg0} Vtg_max={g.Vtg_max} Vbg_max={g.Vbg_max}"])
        writer.writerow([f"# Initial B at scan start = {self._initial_B_at_start:+.5f} T"])
        chnames = [c.csv_col_name for c in self.channels]
        writer.writerow([f"# Channels: {', '.join(chnames)}"])
        # Column header
        col_names = [
            f"{cfg.outer_axis}_target",
            f"{cfg.fixed_axis}_fixed",
            "B_requested_T",
            "B_actual_T",
            "Vtg_V",
            "Vbg_V",
            "direction",
        ] + chnames
        writer.writerow(col_names)

    # -----------------------------------------------------------------
    def _write_metadata_sidecar(self):
        """JSON sidecar next to the CSV file. schema_version is a string
        unique to this program, not a number — we will never share a schema
        with dual_gate_mapping."""
        p   = self.params
        cfg = self.cfg
        mc  = cfg.magnet_config
        g   = cfg.geometry
        sidecar_path = os.path.splitext(p['save_path'])[0] + '.json'
        meta = {
            'schema_version': 'hysteresis_v1.0',
            'app': {'name': APP_NAME, 'version': APP_VERSION},
            'started_at_iso': self._started_at_iso,
            'sample':   p.get('sample',''),
            'device':   p.get('device',''),
            'operator': p.get('operator',''),
            'run_name': p.get('run_name',''),
            'data_file': os.path.basename(p['save_path']),
            'magnet': {
                'controller_type':     mc.controller_type,
                'visa_resource':       mc.visa_resource,
                'magnet_uid':          mc.magnet_uid,
                'B_min_T':             mc.B_min_T,
                'B_max_T':             mc.B_max_T,
                'ramp_rate_T_per_min': mc.ramp_rate_T_per_min,
                'B_step_T':            mc.B_step_T,
                'field_tolerance_T':   mc.field_tolerance_T,
                't_sample_s':          mc.t_sample_s,
                'num_B_points':        mc.num_B_points,
                'initial_B_at_start_T': self._initial_B_at_start,
            },
            'outer_sweep': {
                'outer_axis':  cfg.outer_axis,
                'outer_min':   cfg.outer_min,
                'outer_max':   cfg.outer_max,
                'num_outer':   cfg.num_outer,
                'fixed_axis':  cfg.fixed_axis,
                'fixed_value': cfg.fixed_value,
            },
            'geometry': {
                'd_t_nm':  g.d_t_nm,
                'd_b_nm':  g.d_b_nm,
                'eps_hBN': g.eps_hBN,
                'Vtg0':    g.Vtg0,
                'Vbg0':    g.Vbg0,
                'Vtg_max': g.Vtg_max,
                'Vbg_max': g.Vbg_max,
            },
            'timing': {
                't_read_s':              cfg.t_read,
                't_settle_slow_s':       cfg.t_settle_slow,
                't_settle_after_fwd_s':  cfg.t_settle_after_fwd,
                't_settle_after_bwd_s':  cfg.t_settle_after_bwd,
            },
            'lockin': {
                'v_osc_v':      self.v_osc,
                'r_series_ohm': self.r_series,
                'i_ac_a':       self.i_ac,
            },
            'channels': [
                {
                    'ai_index':       c.ai_index,
                    'name':           c.name,
                    'enabled':        bool(c.enabled),
                    'kind':           c.kind,
                    'unit':           c.unit,
                    'sensitivity_v':  c.sens if c.kind == 'R' else None,
                    'gain':           c.gain if c.kind == 'R' else None,
                } for c in self.channels
            ],
            'hysteresis_analysis': {
                'channel_ai':       cfg.hysteresis_channel_ai,
                'channel_name':     self.channels[cfg.hysteresis_channel_ai].name,
                'channel_unit':     self.channels[cfg.hysteresis_channel_ai].unit,
                'area_unit':        f"{self.channels[cfg.hysteresis_channel_ai].unit}*T",
                'area_per_outer':   self._hyst_areas,
            },
        }
        # ★ v1.1: clean NaN/Inf to None and use allow_nan=False as a
        # belt-and-braces guard against any straggler. Produces strict
        # RFC-7159 compliant JSON readable by MATLAB jsondecode etc.
        cleaned = _json_clean(meta)
        with open(sidecar_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False, allow_nan=False)
        return sidecar_path

    # -----------------------------------------------------------------
    def _emit_point(self, writer, outer_target, B_requested, B_actual,
                    Vtg, Vbg, values, j_idx, i_idx, direction):
        cfg = self.cfg
        row = [
            outer_target,
            cfg.fixed_value,
            B_requested,
            B_actual,
            Vtg,
            Vbg,
            direction,
        ] + values
        writer.writerow(row)
        info = PointInfo(
            outer_target=outer_target,
            B_requested=B_requested,
            B_actual=B_actual,
            Vtg=Vtg, Vbg=Vbg,
            values=values,
            j_idx=j_idx, i_idx=i_idx,
            direction=direction,
        )
        self.point_ready.emit(info)

    # -----------------------------------------------------------------
    @staticmethod
    def _compute_hysteresis_area(fwd_B: List[float], fwd_R: List[float],
                                 bwd_B: List[float], bwd_R: List[float]) -> float:
        """∫ |R_fwd(B) − R_bwd(B)| dB over the common B range.

        Both fwd and bwd are sampled at slightly different B values (because
        the magnet is freely ramping, not stepping). We interpolate both onto
        a common dense grid, then trapezoidal integrate the absolute diff.

        Returns NaN if either array is too short, all-NaN, or B-ranges are
        disjoint (Bug #23 v1.1: distinguish "couldn't compute" from "really
        zero hysteresis"). Downstream JSON serialization converts NaN → None.
        """
        if len(fwd_B) < 4 or len(bwd_B) < 4:
            return float('nan')
        fwd_B = np.asarray(fwd_B, dtype=float)
        fwd_R = np.asarray(fwd_R, dtype=float)
        bwd_B = np.asarray(bwd_B, dtype=float)
        bwd_R = np.asarray(bwd_R, dtype=float)

        # Drop NaN
        fmask = ~(np.isnan(fwd_B) | np.isnan(fwd_R))
        bmask = ~(np.isnan(bwd_B) | np.isnan(bwd_R))
        fwd_B, fwd_R = fwd_B[fmask], fwd_R[fmask]
        bwd_B, bwd_R = bwd_B[bmask], bwd_R[bmask]
        if len(fwd_B) < 4 or len(bwd_B) < 4:
            return float('nan')

        # Sort ascending in B (bwd was traversed high → low so it's descending)
        f_order = np.argsort(fwd_B)
        b_order = np.argsort(bwd_B)
        fwd_B, fwd_R = fwd_B[f_order], fwd_R[f_order]
        bwd_B, bwd_R = bwd_B[b_order], bwd_R[b_order]

        # Common range
        B_lo = max(fwd_B[0], bwd_B[0])
        B_hi = min(fwd_B[-1], bwd_B[-1])
        if B_hi <= B_lo:
            return float('nan')
        n_grid = 501
        B_common = np.linspace(B_lo, B_hi, n_grid)
        Rf = np.interp(B_common, fwd_B, fwd_R)
        Rb = np.interp(B_common, bwd_B, bwd_R)
        return _trapz(np.abs(Rf - Rb), B_common)

    # -----------------------------------------------------------------
    def run(self):
        p   = self.params
        cfg = self.cfg
        mc  = cfg.magnet_config
        mag = self.magnet
        ao_top = cfg.ao_top
        ao_bot = cfg.ao_bot
        tol    = mc.field_tolerance_T

        self._started_at_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

        try:
            self.log_msg.emit(
                f"Scan started: sample={p.get('sample','')!r} "
                f"device={p.get('device','')!r} run={p.get('run_name','')!r}",
                "info")
            self.log_msg.emit(
                f"I_ac = {self.i_ac:.3e} A | top gate AO = {ao_top} | bot gate AO = {ao_bot}",
                "info")
            self.log_msg.emit(
                f"Outer: {cfg.outer_axis} in [{cfg.outer_min}, {cfg.outer_max}] "
                f"({cfg.num_outer} pts), fixed {cfg.fixed_axis}={cfg.fixed_value}",
                "info")
            self.log_msg.emit(
                f"Magnet: {mc.controller_type} @ {mc.visa_resource!r}, "
                f"B in [{mc.B_min_T}, {mc.B_max_T}] T, rate={mc.ramp_rate_T_per_min} T/min, "
                f"t_sample={mc.t_sample_s*1000:.0f} ms",
                "info")
            self.log_msg.emit(f"Saving to {p['save_path']}", "info")

            # ---- 1) Ramp gates to safety position ----
            self.log_msg.emit("Ramping gates to 0 V safety position...", "info")
            self.hw.ramp_ao(ao_top, 0.0)
            self.hw.ramp_ao(ao_bot, 0.0)
            time.sleep(0.3)

            # ---- 2) Connect magnet ----
            self.log_msg.emit("Connecting to magnet controller...", "info")
            mag.connect()
            self.log_msg.emit(f"Magnet: {mag.describe()}", "success")

            # ---- 3) STRICT start-point check ----
            B_now = mag.read_field_T()
            self._initial_B_at_start = B_now
            delta = abs(B_now - mc.B_min_T)
            if delta > mc.field_tolerance_T:
                msg = (f"Start-point check FAILED: magnet at {B_now:+.4f} T but scan "
                       f"requires start at B_min = {mc.B_min_T:+.4f} T "
                       f"(tolerance {mc.field_tolerance_T} T, actual deviation {delta:.4f} T).\n"
                       f"Please ramp the magnet manually to B_min and retry.")
                self.log_msg.emit(msg, "error")
                raise RuntimeError(msg)
            self.log_msg.emit(
                f"Start-point OK: B_now = {B_now:+.4f} T "
                f"(|ΔB| = {delta*1000:.1f} mT ≤ {mc.field_tolerance_T*1000:.0f} mT tol)",
                "success")

            # ---- 4) Set ramp rate once for the whole scan ----
            mag.set_ramp_rate(mc.ramp_rate_T_per_min)
            self.log_msg.emit(f"Magnet ramp rate set to {mc.ramp_rate_T_per_min} T/min.", "info")

            # ---- 5) Open CSV and write header ----
            with open(p['save_path'], mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                self._write_csv_header(writer)
                f.flush()

                outer_array = cfg.outer_array()
                num_outer   = len(outer_array)
                t_sample    = mc.t_sample_s
                hyst_ai     = cfg.hysteresis_channel_ai

                # ---- 6) Outer loop ----
                for i, outer_target in enumerate(outer_array):
                    if not self.is_running:
                        break
                    outer_target = float(outer_target)

                    # 6a) Reverse-map and ramp gates
                    Vtg, Vbg = cfg.outer_to_gates(outer_target)
                    self.log_msg.emit(
                        f"Outer {i+1}/{num_outer}: {cfg.outer_axis}={outer_target:+.4g} "
                        f"→ Vtg={Vtg:+.3f}V, Vbg={Vbg:+.3f}V", "info")
                    self.hw.ramp_ao(ao_top, Vtg)
                    self.hw.ramp_ao(ao_bot, Vbg)
                    time.sleep(cfg.t_settle_slow)

                    # buffers for hysteresis area calculation (for one direction pair)
                    fwd_B_buf, fwd_R_buf = [], []
                    bwd_B_buf, bwd_R_buf = [], []

                    # 6b/c) Two directions
                    for direction, B_target, settle in (
                        ('fwd', mc.B_max_T, cfg.t_settle_after_fwd),
                        ('bwd', mc.B_min_T, cfg.t_settle_after_bwd),
                    ):
                        if not self.is_running:
                            break

                        # ★ v1.1 Bug #7: read current field BEFORE issuing
                        # set_target/start_ramp. This becomes B_at_dir_start
                        # so we can detect "magnet has actually started moving"
                        # and skip the redundant first-sample at the turnaround.
                        B_at_dir_start = mag.read_field_T()

                        mag.set_target(B_target)
                        mag.start_ramp()
                        self.log_msg.emit(
                            f"  {direction}: ramping B → {B_target:+.3f} T "
                            f"(starting from {B_at_dir_start:+.4f} T)", "info")

                        # ★ v1.1 Bug #7+#9: wait for magnet to actually begin
                        # moving (Mercury can take 100ms-1s after start_ramp
                        # before the field starts changing). During this grace
                        # period we don't sample (avoids duplicate at turnaround
                        # and false stall detection).
                        t_dir_start = time.monotonic()
                        moving_thresh_T = max(5 * tol, 0.005)  # 5*tol or 5 mT, larger
                        while self.is_running:
                            if time.monotonic() - t_dir_start > RAMP_STARTUP_GRACE_S:
                                # Grace expired — proceed regardless. Magnet
                                # may legitimately not have moved if B_target
                                # is very close to current (already at target).
                                break
                            if abs(mag.read_field_T() - B_at_dir_start) > moving_thresh_T:
                                break  # magnet has begun moving
                            time.sleep(0.05)

                        # ★ v1.1 Bug #9: better stall threshold. tol/100 is
                        # too tight; use max(tol/10, 1 mT) which is comfortably
                        # above sensor noise (~10 µT) but still sensitive
                        # enough to catch a real stall.
                        stall_threshold_T = max(tol / 10.0, 0.001)
                        stall_window      = 30  # ticks of no progress before abort
                        stall_count = 0
                        B_prev = None

                        # Continuous timer-paced sampling loop
                        while self.is_running:
                            tick_start = time.monotonic()

                            # Read magnet first (so DEMO synthesis sees fresh B)
                            read_t0 = time.monotonic()
                            B_actual = mag.read_field_T()
                            read_dt = time.monotonic() - read_t0
                            if read_dt > 0.5:
                                # Bug #19 watchdog warning — slow GPIB
                                self.log_msg.emit(
                                    f"  WARN: magnet read took {read_dt*1000:.0f} ms "
                                    f"— GPIB may be slow or controller unresponsive.",
                                    "warning")
                            # Inject B + direction into DAQ for DEMO synthesis only
                            if DEMO_MODE:
                                self.hw._demo_magnet_B_T = float(B_actual)
                                self.hw._demo_direction  = direction

                            # ★ v1.1: pass t_read so read_ai does proper averaging
                            raw    = self.hw.read_ai(cfg.t_read)
                            values = self._convert(raw)

                            # Quantize to canonical B grid
                            if mc.B_step_T > 0:
                                j_idx = int(round((B_actual - mc.B_min_T) / mc.B_step_T))
                            else:
                                j_idx = 0
                            j_idx = max(0, min(mc.num_B_points - 1, j_idx))
                            B_requested = mc.B_min_T + j_idx * mc.B_step_T

                            self._emit_point(writer, outer_target, B_requested, B_actual,
                                             Vtg, Vbg, values, j_idx, i, direction)
                            self._points_written += 1

                            # ★ v1.1 E1: periodic fsync — bound power-loss data loss
                            if self._points_written % CSV_FSYNC_EVERY_N == 0:
                                try:
                                    f.flush()
                                    os.fsync(f.fileno())
                                except Exception:
                                    pass

                            # Buffer for hysteresis area
                            R_for_area = values[hyst_ai] if hyst_ai < len(values) else float('nan')
                            if direction == 'fwd':
                                fwd_B_buf.append(B_actual)
                                fwd_R_buf.append(R_for_area)
                            else:
                                bwd_B_buf.append(B_actual)
                                bwd_R_buf.append(R_for_area)

                            # ---- Termination: use magnet.is_at_target ----
                            # (Fix #1: terminate when the magnet itself reports
                            #  it has reached the target, not when B_actual crosses
                            #  some heuristic margin. This fills in the last few
                            #  cells of the canonical grid.)
                            if mag.is_at_target(tol):
                                break

                            # ---- Stall detection (with v1.1 better threshold) ----
                            if B_prev is not None and abs(B_actual - B_prev) < stall_threshold_T:
                                stall_count += 1
                                if stall_count > stall_window:
                                    self.log_msg.emit(
                                        f"  WARN: magnet appears stalled at "
                                        f"{B_actual:+.4f} T "
                                        f"(no progress in {stall_window} samples). "
                                        f"Aborting this direction.",
                                        "warning")
                                    break
                            else:
                                stall_count = 0
                            B_prev = B_actual

                            # ---- Pace next sample (Fix #3: drift-free) ----
                            elapsed = time.monotonic() - tick_start
                            sleep_for = t_sample - elapsed
                            if sleep_for > 0:
                                time.sleep(sleep_for)
                            # If elapsed > t_sample we just plough on — no backlog
                            # accumulation because each loop measures from its own
                            # tick_start, not from a global next_tick variable.

                        mag.hold()
                        f.flush()
                        time.sleep(settle)

                    # 6f) Compute hysteresis area for this outer point
                    area = self._compute_hysteresis_area(
                        fwd_B_buf, fwd_R_buf, bwd_B_buf, bwd_R_buf)
                    self._hyst_areas.append(area)
                    ch = self.channels[hyst_ai]
                    self.log_msg.emit(
                        f"Outer {i+1}/{num_outer} done. Hysteresis area "
                        f"({ch.name}) = {area:.4g} {ch.unit}·T",
                        "success")
                    self.outer_point_finished.emit(i, area)

            self.log_msg.emit("Scan finished normally.", "success")
            self.finished_ok.emit()

        except Exception:
            tb = traceback.format_exc()
            self.log_msg.emit(f"ERROR: {tb.splitlines()[-1]}", "error")
            self.error_occurred.emit(tb)
        finally:
            # Always: gates → 0 V, magnet hold + disconnect, write sidecar
            try:
                self.log_msg.emit("Ramping gates back to 0 V...", "info")
                self.hw.ramp_ao(ao_top, 0.0)
                self.hw.ramp_ao(ao_bot, 0.0)
                self.log_msg.emit("Gates safely at 0 V.", "success")
            except Exception as e:
                self.log_msg.emit(f"WARN: ramp-down failed: {e}", "warning")
            try:
                self.magnet.hold()
                self.magnet.disconnect()
                self.log_msg.emit("Magnet hold + disconnect.", "info")
            except Exception as e:
                self.log_msg.emit(f"WARN: magnet cleanup failed: {e}", "warning")
            try:
                sc = self._write_metadata_sidecar()
                self.log_msg.emit(f"Metadata sidecar written: {os.path.basename(sc)}", "info")
            except Exception as e:
                self.log_msg.emit(f"WARN: metadata write failed: {e}", "warning")

    def stop(self):
        self.is_running = False


# =====================================================================
# 4.  Helper: NoWheelComboBox + StayOpenMenu
# =====================================================================

class NoWheelComboBox(QComboBox):
    """ComboBox that ignores wheel events (so scrolling the form panel
    doesn't accidentally change values)."""
    def wheelEvent(self, e):
        e.ignore()


class StayOpenMenu(QMenu):
    """Menu that doesn't close after a checkable action is toggled.
    Used for the line slot multi-channel picker."""
    def mouseReleaseEvent(self, e):
        action = self.activeAction()
        if action is not None and action.isCheckable() and action.isEnabled():
            action.trigger()
            return
        super().mouseReleaseEvent(e)


# =====================================================================
# 5.  HysteresisGUI — main window
# =====================================================================

class HysteresisGUI(QMainWindow):
    """Catppuccin-themed GUI for hysteresis mapping.

    Layout:
      ┌──────────────────────────────────────────────────────────────┐
      │  metadata bar (sample / device / operator / run + state)     │
      ├─────────────────────────────────┬────────────────────────────┤
      │  left panel (scrollable form)   │  right panel:              │
      │   · Outer sweep                  │   ┌─ line slot 0 ────┐    │
      │   · Gates (top/bot AO)           │   ├─ line slot 1 ────┤    │
      │   · Geometry                     │   ├─ map slot 0 fwd ─┤    │
      │   · Magnet                       │   └─ map slot 1 bwd ─┘    │
      │   · Timing                       │  ───────────────────────  │
      │   · Lockin / source              │   log panel               │
      │   · Channels (×8)                │   progress bar            │
      │   · Output                       │   status bar (hyst area)  │
      │   · START / ABORT                │                            │
      └─────────────────────────────────┴────────────────────────────┘
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}  [{'DEMO' if DEMO_MODE else 'LIVE'}]")
        self.resize(1700, 1000)
        self.setStyleSheet(APP_QSS)

        # Background-thread / runtime state
        self.thread: Optional[MeasurementThread] = None
        self.locked_channels: Optional[List[ChannelConfig]] = None
        self.locked_cfg:      Optional[HysteresisConfig]    = None
        self.map_data = None     # dict[ai] → dict[direction] → 2D ndarray
        self.hyst_area_per_outer: List[float] = []
        # ★ v1.1 Bug #2: track whether user has run "Verify magnet comms"
        # at least once this session. LIVE-mode scans with B-range > 1 T
        # require this to be True before they can start.
        self._magnet_verified: bool = False

        # DAQ
        self.hw = DAQHardware()
        self.hw.open()

        # 1D buffer for the currently-displayed outer point
        self.curr_outer_idx = -1
        self.curr_buf = self._make_empty_1d_buffers()

        # Build UI
        self._build_ui()
        self._init_menubar()
        self._init_statusbar()

        # ★ v1.1 Bug #14: Populate channel-name-derived widgets BEFORE
        # _load_settings, so int-userData restorations (hyst_channel etc.)
        # find existing items to match.
        self._update_iac_label()
        for ai in range(NUM_AI):
            self._refresh_kind_widgets(ai)
        self._rebuild_hyst_channel_combo()

        # Load saved settings (combos are now populated)
        self._load_settings()

        # Clock timer in metadata bar
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)
        self._tick_clock()

        # Final refreshes that depend on loaded values
        self._refresh_filename()
        self._on_outer_axis_changed()
        self._update_t_sample_label()
        for slot_idx in range(NUM_LINE_SLOTS):
            self._on_line_slot_changed(slot_idx)

        # Initial log
        self.log_event("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")
        self.log_event(f"{APP_NAME}  ·  {APP_VERSION}", "success")
        self.log_event("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")
        if DEMO_MODE:
            self.log_event("DEMO MODE — DAQ + magnet are simulated.", "warning")
        else:
            self.log_event("LIVE MODE — using real NI DAQ + magnet hardware.", "success")

    def _make_empty_1d_buffers(self):
        """Per-direction buffers of (B_array, [val_array per AI])."""
        return {
            d: {'B': [], 'vals': [[] for _ in range(NUM_AI)]}
            for d in DIRECTIONS
        }

    # -----------------------------------------------------------------
    # UI build — top-level layout
    # -----------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        v.addLayout(self._build_top_bar())

        self.h_split = QSplitter(Qt.Horizontal)
        self.h_split.addWidget(self._build_left_panel())
        self.h_split.addWidget(self._build_right_panel())
        self.h_split.setStretchFactor(0, 0)
        self.h_split.setStretchFactor(1, 1)
        self.h_split.setSizes([520, 1180])
        v.addWidget(self.h_split, 1)

    def _build_top_bar(self):
        h = QHBoxLayout()
        h.setSpacing(8)

        for label_text, attr, default, width in [
            ("Sample",  'le_sample',   '',   140),
            ("Device",  'le_device',   '',   140),
            ("Operator",'le_operator', '',   100),
            ("Run",     'le_run_name', 'r1', 120),
        ]:
            lbl = QLabel(label_text + ':')
            lbl.setObjectName("sectionLabel")
            le = QLineEdit(default)
            le.setMaximumWidth(width)
            le.textChanged.connect(self._refresh_filename)
            setattr(self, attr, le)
            h.addWidget(lbl)
            h.addWidget(le)
        h.addStretch()

        self.lbl_state = QLabel("Idle")
        self.lbl_state.setObjectName("stateLabel")
        self.lbl_state.setStyleSheet(f"background-color:{CT_GREEN}; color:{CT_BASE};")
        h.addWidget(self.lbl_state)

        self.lbl_clock = QLabel("--:--:--")
        self.lbl_clock.setStyleSheet(f"color:{CT_SUBTEXT0}; font-family: monospace;")
        h.addWidget(self.lbl_clock)
        return h

    def _set_state(self, text, color_hex):
        self.lbl_state.setText(text)
        self.lbl_state.setStyleSheet(f"background-color:{color_hex}; color:{CT_BASE};")

    def _tick_clock(self):
        self.lbl_clock.setText(time.strftime("%H:%M:%S"))

    # -----------------------------------------------------------------
    # Left panel
    # -----------------------------------------------------------------
    def _build_left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(10)

        v.addWidget(self._build_outer_group())
        v.addWidget(self._build_gates_group())
        v.addWidget(self._build_geometry_group())
        v.addWidget(self._build_magnet_group())
        v.addWidget(self._build_timing_group())
        v.addWidget(self._build_lockin_group())
        v.addWidget(self._build_channels_group())
        v.addWidget(self._build_output_group())
        v.addStretch()
        v.addLayout(self._build_button_row())

        scroll.setWidget(inner)
        scroll.setMinimumWidth(500)
        return scroll

    # ---- Outer sweep group ----
    def _build_outer_group(self):
        gb = QGroupBox("Outer sweep")
        grid = QGridLayout()
        grid.setContentsMargins(8, 4, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        l = QLabel("Outer axis")
        l.setObjectName("sectionLabel")
        grid.addWidget(l, 0, 0)
        self.cb_outer_axis = NoWheelComboBox()
        self.cb_outer_axis.addItem("n  (carrier density)",     userData='n')
        self.cb_outer_axis.addItem("D  (displacement field)",  userData='D')
        self.cb_outer_axis.setToolTip(
            "Which of (n, D) is the outer sweep variable. The other one is held\n"
            "fixed at the value in 'Fixed value'.")
        self.cb_outer_axis.currentIndexChanged.connect(self._on_outer_axis_changed)
        grid.addWidget(self.cb_outer_axis, 0, 1, 1, 2)

        # Outer min / max / num
        self.lbl_outer_min = QLabel("Outer min (×10¹² cm⁻²)")
        grid.addWidget(self.lbl_outer_min, 1, 0)
        self.le_outer_min = QLineEdit("-3.0")
        grid.addWidget(self.le_outer_min, 1, 1, 1, 2)

        self.lbl_outer_max = QLabel("Outer max (×10¹² cm⁻²)")
        grid.addWidget(self.lbl_outer_max, 2, 0)
        self.le_outer_max = QLineEdit("3.0")
        grid.addWidget(self.le_outer_max, 2, 1, 1, 2)

        l = QLabel("Num outer points")
        grid.addWidget(l, 3, 0)
        self.le_num_outer = QLineEdit("11")
        grid.addWidget(self.le_num_outer, 3, 1, 1, 2)

        self.lbl_fixed = QLabel("Fixed D (V/nm)")
        grid.addWidget(self.lbl_fixed, 4, 0)
        self.le_fixed = QLineEdit("0.0")
        grid.addWidget(self.le_fixed, 4, 1, 1, 2)

        gb.setLayout(grid)
        return gb

    def _on_outer_axis_changed(self, _idx=None):
        if not hasattr(self, 'cb_outer_axis'):
            return
        outer = self.cb_outer_axis.currentData()
        if outer == 'n':
            self.lbl_outer_min.setText("Outer min n (×10¹² cm⁻²)")
            self.lbl_outer_max.setText("Outer max n (×10¹² cm⁻²)")
            self.lbl_fixed.setText("Fixed D (V/nm)")
        else:
            self.lbl_outer_min.setText("Outer min D (V/nm)")
            self.lbl_outer_max.setText("Outer max D (V/nm)")
            self.lbl_fixed.setText("Fixed n (×10¹² cm⁻²)")

    # ---- Gates group ----
    def _build_gates_group(self):
        gb = QGroupBox("Gate AOs")
        grid = QGridLayout()
        grid.setContentsMargins(8, 4, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        l = QLabel("Top gate AO")
        l.setObjectName("sectionLabel")
        grid.addWidget(l, 0, 0)
        self.cb_top_ao = NoWheelComboBox()
        self.cb_top_ao.addItem("AO0", userData='AO0')
        self.cb_top_ao.addItem("AO1", userData='AO1')
        self.cb_top_ao.setToolTip(
            "Which physical NI DAQ AO channel drives the top gate (Vtg).\n"
            "Bottom gate gets the other AO automatically.")
        self.cb_top_ao.currentIndexChanged.connect(self._on_top_ao_changed)
        grid.addWidget(self.cb_top_ao, 0, 1, 1, 2)

        l = QLabel("Bot gate AO")
        l.setObjectName("sectionLabel")
        grid.addWidget(l, 1, 0)
        self.lbl_bot_ao = QLabel("AO1")
        self.lbl_bot_ao.setObjectName("valueLabel")
        grid.addWidget(self.lbl_bot_ao, 1, 1, 1, 2)

        gb.setLayout(grid)
        return gb

    def _on_top_ao_changed(self, _idx=None):
        if not hasattr(self, 'cb_top_ao'):
            return
        top = self.cb_top_ao.currentData()
        self.lbl_bot_ao.setText('AO1' if top == 'AO0' else 'AO0')

    # ---- Geometry group ----
    def _build_geometry_group(self):
        gb = QGroupBox("Geometry  (hBN dual gate)")
        grid = QGridLayout()
        grid.setContentsMargins(8, 4, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.geom_inputs = {}
        # ★ v1.1 Bug #17: explicit tooltip showing the D-field ↔ Vtg math
        Vmax_tip = (
            "Hard upper bound on |Vtg|. Outer points whose reverse-mapped Vtg\n"
            "exceeds this will REJECT the scan at preflight.\n\n"
            "Rule of thumb: at n=0, the gate voltage required to reach a given\n"
            "displacement field D is approximately\n"
            "    |Vtg| ≈ D × d_t / ε_hBN\n"
            "For d_t=30 nm, ε=3.0:\n"
            "    D = 0.5 V/nm  →  |Vtg| ≈ 5.0 V\n"
            "    D = 1.0 V/nm  →  |Vtg| ≈ 10.0 V\n"
            "    D = 2.0 V/nm  →  |Vtg| ≈ 20.0 V\n"
            "Default 5 V works for D ≲ 0.5 V/nm only. Increase to your hBN\n"
            "breakdown limit (typically up to ~1 V/nm sustained) only if the\n"
            "sample tolerates it.")
        rows = [
            ('d_t',     'd_t  (top hBN, nm)',    '30.0',
             "Top hBN dielectric thickness in nm. Determines C_t."),
            ('d_b',     'd_b  (bottom hBN, nm)', '30.0',
             "Bottom hBN dielectric thickness in nm. Determines C_b."),
            ('eps',     'ε_hBN (out-of-plane)',  '3.0',
             "hBN out-of-plane dielectric constant. Typical 3.0–3.9."),
            ('Vtg0',    'Vtg₀  (CNP, V)',        '0.0',
             "Top gate charge neutrality point."),
            ('Vbg0',    'Vbg₀  (CNP, V)',        '0.0',
             "Bottom gate charge neutrality point."),
            ('Vtg_max', '|Vtg|_max (V)',         '5.0', Vmax_tip),
            ('Vbg_max', '|Vbg|_max (V)',         '5.0',
             "Hard upper bound on |Vbg|. Same logic as |Vtg|_max."),
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

    # ---- Magnet group ----
    def _build_magnet_group(self):
        gb = QGroupBox("Magnet")
        grid = QGridLayout()
        grid.setContentsMargins(8, 4, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        l = QLabel("Controller")
        l.setObjectName("sectionLabel")
        grid.addWidget(l, 0, 0)
        self.cb_magnet_type = NoWheelComboBox()
        self.cb_magnet_type.addItem("DEMO (simulated)",         userData=MAGNET_DEMO)
        self.cb_magnet_type.addItem("Mercury iPS (SCPI)",       userData=MAGNET_MERCURY_SCPI)
        self.cb_magnet_type.addItem("Mercury iPS (ISOBUS legacy)", userData=MAGNET_MERCURY_ISO)
        self.cb_magnet_type.setToolTip(
            "DEMO: simulated, no hardware needed. Auto-positions to B_min on connect.\n"
            "Mercury SCPI: modern firmware, READ:DEV:UID:PSU:SIG:... commands.\n"
            "Mercury ISOBUS: legacy short commands R7/J/T/A1.\n"
            "If unsure, try SCPI first; if *IDN? doesn't respond, switch to ISOBUS.")
        grid.addWidget(self.cb_magnet_type, 0, 1, 1, 2)

        l = QLabel("VISA resource")
        grid.addWidget(l, 1, 0)
        self.le_magnet_visa = QLineEdit("GPIB0::25::INSTR")
        self.le_magnet_visa.setToolTip("Ignored in DEMO mode.")
        grid.addWidget(self.le_magnet_visa, 1, 1, 1, 2)

        l = QLabel("Magnet UID")
        grid.addWidget(l, 2, 0)
        self.le_magnet_uid = QLineEdit("GRPZ")
        self.le_magnet_uid.setToolTip("SCPI device identifier; only used in Mercury SCPI mode.")
        grid.addWidget(self.le_magnet_uid, 2, 1, 1, 2)

        l = QLabel("B_min (T)")
        grid.addWidget(l, 3, 0)
        self.le_B_min = QLineEdit("-5.0")
        self.le_B_min.setToolTip(
            "Starting field. MUST manually pre-ramp the magnet to this value\n"
            "before clicking START — the program strict-checks and rejects\n"
            "the scan if the current field is more than tol away.")
        grid.addWidget(self.le_B_min, 3, 1, 1, 2)

        l = QLabel("B_max (T)")
        grid.addWidget(l, 4, 0)
        self.le_B_max = QLineEdit("5.0")
        grid.addWidget(self.le_B_max, 4, 1, 1, 2)

        l = QLabel("Ramp rate (T/min)")
        grid.addWidget(l, 5, 0)
        self.le_ramp_rate = QLineEdit("0.5")
        self.le_ramp_rate.setToolTip("Typical SC magnet: 0.1–1 T/min.")
        self.le_ramp_rate.textChanged.connect(self._update_t_sample_label)
        grid.addWidget(self.le_ramp_rate, 5, 1, 1, 2)

        l = QLabel("B_step (T)")
        grid.addWidget(l, 6, 0)
        self.le_B_step = QLineEdit("0.02")
        self.le_B_step.setToolTip("Logical sampling interval in T. t_sample = B_step / ramp_rate.")
        self.le_B_step.textChanged.connect(self._update_t_sample_label)
        grid.addWidget(self.le_B_step, 6, 1, 1, 2)

        l = QLabel("Field tol (T)")
        grid.addWidget(l, 7, 0)
        self.le_B_tol = QLineEdit("0.05")
        self.le_B_tol.setToolTip(
            "Tolerance for strict start check + magnet.is_at_target() termination.")
        grid.addWidget(self.le_B_tol, 7, 1, 1, 2)

        l = QLabel("t_sample (derived)")
        l.setToolTip("Computed from B_step / ramp_rate. Read-only.")
        grid.addWidget(l, 8, 0)
        self.lbl_t_sample = QLabel("— ms")
        self.lbl_t_sample.setObjectName("valueLabel")
        grid.addWidget(self.lbl_t_sample, 8, 1, 1, 2)

        l = QLabel("Hyst channel")
        l.setObjectName("sectionLabel")
        l.setToolTip("Which AI channel to use for ∫|R_fwd-R_bwd|dB area calculation.")
        grid.addWidget(l, 9, 0)
        self.cb_hyst_channel = NoWheelComboBox()
        # populated later by _rebuild_hyst_channel_combo()
        grid.addWidget(self.cb_hyst_channel, 9, 1, 1, 2)

        # ★ v1.1 Bug #2: Verify magnet communication button.
        # Read-only round-trip query that confirms commands work BEFORE
        # any production scan. Required at least once per session before
        # large-range LIVE scans can start (enforced in start_measurement).
        self.btn_verify_magnet = QPushButton("Verify magnet comms")
        self.btn_verify_magnet.setToolTip(
            "Run a read-only query against the magnet controller and display\n"
            "the results. Use this BEFORE the first scan of the session to\n"
            "confirm GPIB communication works and the units shown match the\n"
            "Mercury front panel. Required for LIVE-mode large-range scans.")
        self.btn_verify_magnet.clicked.connect(self._on_verify_magnet_clicked)
        grid.addWidget(self.btn_verify_magnet, 10, 0, 1, 3)

        gb.setLayout(grid)
        return gb

    def _on_verify_magnet_clicked(self):
        """★ v1.1 Bug #2: Verify magnet communication via verify_communication().
        Constructs a temporary controller (does NOT touch the running
        thread), connects, queries, disconnects."""
        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "Scan running",
                "Cannot verify magnet during a scan — the GPIB bus is busy.")
            return
        try:
            params_partial = {
                'controller_type': self.cb_magnet_type.currentData(),
                'visa_resource':   self.le_magnet_visa.text().strip(),
                'magnet_uid':      self.le_magnet_uid.text().strip(),
                'B_min_T':         float(self.le_B_min.text() or 0),
                'B_max_T':         float(self.le_B_max.text() or 1),
                'ramp_rate_T_per_min': float(self.le_ramp_rate.text() or 0.5),
                'B_step_T':        float(self.le_B_step.text() or 0.02),
                'field_tolerance_T': float(self.le_B_tol.text() or 0.05),
            }
        except ValueError as e:
            QMessageBox.critical(self, "Invalid magnet config",
                f"Could not parse magnet fields: {e}")
            return

        mc = MagnetConfig(**params_partial)
        mag = make_magnet_controller(mc)
        self.log_event("Verifying magnet communication...", "info")
        try:
            mag.connect()
        except Exception as e:
            self.log_event(f"Magnet connect FAILED: {e}", "error")
            QMessageBox.critical(self, "Magnet connect failed", str(e))
            return
        try:
            result = mag.verify_communication()
        except Exception as e:
            self.log_event(f"Magnet verify FAILED: {e}", "error")
            QMessageBox.critical(self, "Magnet verify failed", str(e))
            try: mag.disconnect()
            except Exception: pass
            return
        finally:
            try: mag.disconnect()
            except Exception: pass

        # Format and display
        lines = ["Magnet verification results:", ""]
        for k, v in result.items():
            if k == 'errors':
                continue
            lines.append(f"  {k}:  {v}")
        if result.get('errors'):
            lines.append("")
            lines.append("Errors during verification:")
            for e in result['errors']:
                lines.append(f"  - {e}")
            lines.append("")
            lines.append("Some queries failed. Do NOT trust this controller "
                         "for production use until errors are fixed.")
        else:
            lines.append("")
            lines.append("All queries succeeded.")
            lines.append("⚠️  Compare these values to the magnet's front panel.")
            lines.append("If they match, you may proceed with scans.")
            lines.append("If they don't match, check VISA resource string and UID.")

        msg = "\n".join(lines)
        self.log_event("Magnet verify completed " +
                       ("(no errors)." if not result.get('errors') else "(with errors)."),
                       "success" if not result.get('errors') else "warning")
        # Mark verified for this session if no errors
        if not result.get('errors'):
            self._magnet_verified = True
        QMessageBox.information(self, "Magnet verification", msg)

    def _update_t_sample_label(self):
        try:
            rate = float(self.le_ramp_rate.text())
            step = float(self.le_B_step.text())
            if rate > 0 and step > 0:
                t = step / (rate / 60.0)
                self.lbl_t_sample.setText(f"{t*1000:.0f} ms")
            else:
                self.lbl_t_sample.setText("(invalid)")
        except Exception:
            self.lbl_t_sample.setText("(invalid)")

    def _rebuild_hyst_channel_combo(self):
        if not hasattr(self, 'cb_hyst_channel'):
            return
        prev = self.cb_hyst_channel.currentData()
        self.cb_hyst_channel.blockSignals(True)
        self.cb_hyst_channel.clear()
        for ai in range(NUM_AI):
            name = self._current_channel_name(ai)
            self.cb_hyst_channel.addItem(f"AI{ai}  {name}", userData=ai)
        # Restore previous selection
        if prev is not None:
            for k in range(self.cb_hyst_channel.count()):
                if self.cb_hyst_channel.itemData(k) == prev:
                    self.cb_hyst_channel.setCurrentIndex(k)
                    break
        self.cb_hyst_channel.blockSignals(False)

    def _on_channel_name_changed(self):
        """★ v1.1 Bug #24: when any channel name changes, rebuild ALL
        channel-name-derived widgets:
          · Hyst channel combo (in Magnet group)
          · Each line slot's "Channels ▾" menu action labels
          · Each map slot's combo entries

        Called from textChanged signals on ch_name_inputs."""
        self._rebuild_hyst_channel_combo()
        # Rebuild line-slot menu action labels (preserve checked state)
        if hasattr(self, 'line_slot_actions'):
            for slot_idx, actions in enumerate(self.line_slot_actions):
                for ai, act in actions.items():
                    act.setText(f"AI{ai}  {self._current_channel_name(ai)}")
        # Rebuild map-slot combo entries (preserve current AI selection)
        if hasattr(self, 'map_slot_combos'):
            for combo in self.map_slot_combos:
                prev_ai = combo.currentData()
                combo.blockSignals(True)
                combo.clear()
                for ai in range(NUM_AI):
                    combo.addItem(f"AI{ai}  {self._current_channel_name(ai)}",
                                  userData=ai)
                if prev_ai is not None:
                    for k in range(combo.count()):
                        if combo.itemData(k) == prev_ai:
                            combo.setCurrentIndex(k)
                            break
                combo.blockSignals(False)
        # Rebuild line-slot legend labels (rebuild lines so legend re-pulls names)
        if hasattr(self, 'line_slot_plots'):
            for slot_idx in range(NUM_LINE_SLOTS):
                self._rebuild_line_slot_lines(slot_idx)

    # ---- Timing group ----
    def _build_timing_group(self):
        gb = QGroupBox("Timing")
        grid = QGridLayout()
        grid.setContentsMargins(8, 4, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.timing_inputs = {}
        rows = [
            ('t_read',             't_read (s)',             '0.05',
             "DAQ AI integration window per sample. Typical 0.03–0.1 s."),
            ('t_settle_slow',      't_settle_slow (s)',      '0.5',
             "Wait time after gate ramp at the start of each outer point."),
            ('t_settle_after_fwd', 't_settle_after_fwd (s)', '0.5',
             "Pause after fwd ramp finishes, before bwd starts."),
            ('t_settle_after_bwd', 't_settle_after_bwd (s)', '0.5',
             "Pause after bwd ramp finishes, before next outer point."),
        ]
        for i, (key, label_text, default, tip) in enumerate(rows):
            l = QLabel(label_text)
            l.setToolTip(tip)
            grid.addWidget(l, i, 0)
            le = QLineEdit(default)
            le.setToolTip(tip)
            self.timing_inputs[key] = le
            grid.addWidget(le, i, 1)
        gb.setLayout(grid)
        return gb

    # ---- Lockin group ----
    def _build_lockin_group(self):
        gb = QGroupBox("Lockin / Source")
        grid = QGridLayout()
        grid.setContentsMargins(8, 4, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.hw_inputs = {}
        l = QLabel("V_osc (V)")
        grid.addWidget(l, 0, 0)
        self.hw_inputs['v_osc'] = QLineEdit("1.0")
        self.hw_inputs['v_osc'].textChanged.connect(self._update_iac_label)
        grid.addWidget(self.hw_inputs['v_osc'], 0, 1)

        l = QLabel("R_series (Ω)")
        grid.addWidget(l, 1, 0)
        self.hw_inputs['r_series'] = QLineEdit("1.0e6")
        self.hw_inputs['r_series'].textChanged.connect(self._update_iac_label)
        grid.addWidget(self.hw_inputs['r_series'], 1, 1)

        l = QLabel("→ I_ac")
        l.setObjectName("sectionLabel")
        grid.addWidget(l, 2, 0)
        self.lbl_iac = QLabel("— A")
        self.lbl_iac.setObjectName("valueLabel")
        grid.addWidget(self.lbl_iac, 2, 1)

        gb.setLayout(grid)
        return gb

    def _update_iac_label(self):
        try:
            v = float(self.hw_inputs['v_osc'].text())
            r = float(self.hw_inputs['r_series'].text())
            self.lbl_iac.setText(f"{v/r:.3e} A" if r > 0 else "(invalid)")
        except Exception:
            self.lbl_iac.setText("(invalid)")

    # ---- Channels group ----
    def _build_channels_group(self):
        gb = QGroupBox("Channels")
        grid = QGridLayout()
        grid.setContentsMargins(8, 4, 8, 8)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)

        # Header row
        for col, text in enumerate(["AI", "Name", "On", "Kind", "Sens (V)", "Gain"]):
            lbl = QLabel(text)
            lbl.setObjectName("sectionLabel")
            grid.addWidget(lbl, 0, col)

        self.ch_name_inputs   = []
        self.ch_enable_checks = []
        self.ch_kind_combos   = []
        self.ch_sens_inputs   = []
        self.ch_gain_inputs   = []

        default_names = ['Rxx', 'θxx', 'Rxy', 'θxy', 'Rxx2', 'θxx2', 'Rxy2', 'θxy2']
        default_kinds = ['R',   'Phase','R',   'Phase','R',    'Phase','R',    'Phase']

        for ai in range(NUM_AI):
            row = ai + 1
            grid.addWidget(QLabel(f"AI{ai}"), row, 0)

            le_name = QLineEdit(default_names[ai])
            le_name.setMaximumWidth(70)
            le_name.textChanged.connect(self._on_channel_name_changed)
            self.ch_name_inputs.append(le_name)
            grid.addWidget(le_name, row, 1)

            cb_en = QCheckBox()
            cb_en.setChecked(True)
            self.ch_enable_checks.append(cb_en)
            grid.addWidget(cb_en, row, 2)

            cb_kind = NoWheelComboBox()
            for k in CHANNEL_KINDS:
                cb_kind.addItem(k)
            cb_kind.setCurrentText(default_kinds[ai])
            cb_kind.currentIndexChanged.connect(lambda _i, a=ai: self._refresh_kind_widgets(a))
            self.ch_kind_combos.append(cb_kind)
            grid.addWidget(cb_kind, row, 3)

            le_sens = QLineEdit("1.0e-3")
            le_sens.setMaximumWidth(80)
            self.ch_sens_inputs.append(le_sens)
            grid.addWidget(le_sens, row, 4)

            le_gain = QLineEdit("1.0e3")
            le_gain.setMaximumWidth(70)
            self.ch_gain_inputs.append(le_gain)
            grid.addWidget(le_gain, row, 5)

        gb.setLayout(grid)
        return gb

    def _refresh_kind_widgets(self, ai):
        kind = self.ch_kind_combos[ai].currentText()
        is_R = (kind == 'R')
        self.ch_sens_inputs[ai].setEnabled(is_R)
        self.ch_gain_inputs[ai].setEnabled(is_R)

    def _current_channel_name(self, ai: int) -> str:
        return (self.ch_name_inputs[ai].text() or f"AI{ai}").strip() or f"AI{ai}"

    def _current_channel_unit(self, ai: int) -> str:
        return KIND_UNIT.get(self.ch_kind_combos[ai].currentText(), '')

    # ---- Output group ----
    def _build_output_group(self):
        gb = QGroupBox("Output")
        grid = QGridLayout()
        grid.setContentsMargins(8, 4, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        l = QLabel("Folder")
        grid.addWidget(l, 0, 0)
        self.le_folder = QLineEdit(os.path.expanduser("~"))
        grid.addWidget(self.le_folder, 0, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(self._browse_folder)
        grid.addWidget(btn, 0, 2)

        l = QLabel("Filename")
        grid.addWidget(l, 1, 0)
        self.lbl_filename = QLabel("—")
        self.lbl_filename.setObjectName("valueLabel")
        grid.addWidget(self.lbl_filename, 1, 1, 1, 2)

        gb.setLayout(grid)
        return gb

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
        run    = (self.le_run_name.text() or 'r1').strip() or 'r1'
        fname  = f"{ts}__{sample}__{device}__{run}__hyst.csv"
        # sanitize
        for c in '<>:"|?*\\/':
            fname = fname.replace(c, '_')
        self.lbl_filename.setText(fname)

    # ---- Button row ----
    def _build_button_row(self):
        h = QHBoxLayout()
        self.btn_start = QPushButton("START")
        self.btn_start.setObjectName("startBtn")
        self.btn_start.clicked.connect(self.start_measurement)
        h.addWidget(self.btn_start)

        self.btn_stop = QPushButton("ABORT")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.clicked.connect(self.stop_measurement)
        self.btn_stop.setEnabled(False)
        h.addWidget(self.btn_stop)
        return h

    # =================================================================
    # Right panel — plots + log
    # =================================================================
    def _build_right_panel(self):
        self.v_split = QSplitter(Qt.Vertical)
        self.v_split.addWidget(self._build_plots_area())
        self.v_split.addWidget(self._build_log_area())
        self.v_split.setStretchFactor(0, 1)
        self.v_split.setStretchFactor(1, 0)
        self.v_split.setSizes([720, 280])
        return self.v_split

    def _build_plots_area(self):
        """Left column = 2 line slots stacked, right column = 2 map slots stacked."""
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(4, 4, 4, 4)
        h.setSpacing(6)

        # Line slots (left column)
        self.line_slot_plots   = []
        self.line_slot_lines   = [{} for _ in range(NUM_LINE_SLOTS)]
        self.line_slot_actions = []   # per slot: dict[ai] -> QAction
        self.line_slot_legends = []
        line_col = QVBoxLayout()
        line_col.setSpacing(4)
        for i in range(NUM_LINE_SLOTS):
            line_col.addWidget(self._build_line_slot(i), 1)
        line_col_w = QWidget()
        line_col_w.setLayout(line_col)
        h.addWidget(line_col_w, 1)

        # Map slots (right column)
        self.map_slot_plots  = []
        self.map_slot_images = []
        self.map_slot_combos = []   # AI selector (direction is fixed per slot)
        map_col = QVBoxLayout()
        map_col.setSpacing(4)
        for i in range(NUM_MAP_SLOTS):
            map_col.addWidget(self._build_map_slot(i), 1)
        map_col_w = QWidget()
        map_col_w.setLayout(map_col)
        h.addWidget(map_col_w, 1)

        return w

    def _build_line_slot(self, slot_idx: int):
        """One R(B) line plot with a 'Channels ▾' multi-select menu button."""
        gb = QGroupBox(f"Line slot {slot_idx + 1}")
        v = QVBoxLayout(gb)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(2)

        # Header row: title + channels button
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        btn = QToolButton()
        btn.setText("Channels ▾")
        btn.setPopupMode(QToolButton.InstantPopup)
        menu = StayOpenMenu(btn)
        actions = {}
        for ai in range(NUM_AI):
            act = QAction(f"AI{ai}  {self._current_channel_name(ai)}", menu)
            act.setCheckable(True)
            # Default: slot 0 has AI0 checked, slot 1 nothing
            act.setChecked(slot_idx == 0 and ai == 0)
            act.triggered.connect(lambda _checked=False, s=slot_idx: self._on_line_slot_changed(s))
            menu.addAction(act)
            actions[ai] = act
        btn.setMenu(menu)
        self.line_slot_actions.append(actions)
        hdr.addWidget(btn)
        hdr.addStretch()
        v.addLayout(hdr)

        # Plot widget
        pw = pg.PlotWidget()
        pw.setBackground(CT_BASE)
        pw.showGrid(x=True, y=True, alpha=0.25)
        pw.setLabel('bottom', 'B (T)', color=CT_TEXT)
        pw.getAxis('bottom').setPen(CT_SUBTEXT1)
        pw.getAxis('left').setPen(CT_SUBTEXT1)
        pw.getAxis('bottom').setTextPen(CT_TEXT)
        pw.getAxis('left').setTextPen(CT_TEXT)
        legend = pw.addLegend(offset=(10, 5))
        self.line_slot_plots.append(pw)
        self.line_slot_legends.append(legend)
        v.addWidget(pw, 1)
        return gb

    def _build_map_slot(self, slot_idx: int):
        """One 2D map. slot 0 is permanently fwd, slot 1 is permanently bwd."""
        direction = 'fwd' if slot_idx == 0 else 'bwd'
        dir_color = CT_RED if direction == 'fwd' else CT_BLUE
        gb = QGroupBox(f"Map slot {slot_idx + 1}  —  {direction.upper()}")
        gb.setStyleSheet(f"QGroupBox::title {{ color: {dir_color}; }}")
        v = QVBoxLayout(gb)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(2)

        # Header: AI channel selector only (direction is fixed)
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.addWidget(QLabel("Channel:"))
        cb = NoWheelComboBox()
        for ai in range(NUM_AI):
            cb.addItem(f"AI{ai}  {self._current_channel_name(ai)}", userData=ai)
        cb.currentIndexChanged.connect(lambda _i, s=slot_idx: self._refresh_map_slot(s))
        self.map_slot_combos.append(cb)
        hdr.addWidget(cb)
        hdr.addStretch()
        v.addLayout(hdr)

        # PlotWidget + ImageItem
        pw = pg.PlotWidget()
        pw.setBackground(CT_BASE)
        pw.getAxis('bottom').setPen(CT_SUBTEXT1)
        pw.getAxis('left').setPen(CT_SUBTEXT1)
        pw.getAxis('bottom').setTextPen(CT_TEXT)
        pw.getAxis('left').setTextPen(CT_TEXT)
        pw.setLabel('bottom', 'B (T)', color=CT_TEXT)
        pw.setLabel('left', 'outer', color=CT_TEXT)
        img = pg.ImageItem()
        # Viridis-ish lookup table
        try:
            cmap = pg.colormap.get('viridis')
            img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        except Exception:
            pass
        pw.addItem(img)
        self.map_slot_plots.append(pw)
        self.map_slot_images.append(img)
        v.addWidget(pw, 1)
        return gb

    def _on_line_slot_changed(self, slot_idx: int):
        self._rebuild_line_slot_lines(slot_idx)

    def _rebuild_line_slot_lines(self, slot_idx: int):
        """Rebuild the line items for one slot based on its current selection.
        
        Color rule for hysteresis: fwd=red, bwd=blue, both solid. Multi-channel
        overlays use dash patterns (solid/dash/dot/dashdot) to disambiguate.
        """
        pw = self.line_slot_plots[slot_idx]
        legend = self.line_slot_legends[slot_idx]
        # Clear old items
        for item in list(pw.listDataItems()):
            pw.removeItem(item)
        try:
            legend.clear()
        except Exception:
            pass

        self.line_slot_lines[slot_idx] = {}
        actions = self.line_slot_actions[slot_idx]
        selected = [ai for ai, a in actions.items() if a.isChecked()]
        if not selected:
            pw.setLabel('left', '')
            return

        # Y label = unit (or 'mixed')
        units = {self._current_channel_unit(ai) for ai in selected}
        ylabel = list(units)[0] if len(units) == 1 else '(mixed)'
        pw.setLabel('left', ylabel)

        styles = [Qt.SolidLine, Qt.DashLine, Qt.DotLine, Qt.DashDotLine]
        for slot_pos, ai in enumerate(selected):
            name = self._current_channel_name(ai)
            for d in DIRECTIONS:
                color = CT_RED if d == 'fwd' else CT_BLUE
                style = styles[slot_pos % len(styles)] if len(selected) > 1 else Qt.SolidLine
                pen = pg.mkPen(color=color, width=2, style=style)
                line = pw.plot(name=f'{name} ({d})', pen=pen)
                self.line_slot_lines[slot_idx][(ai, d)] = line

        # Replay current buffer
        for (ai, d), line in self.line_slot_lines[slot_idx].items():
            buf = self.curr_buf[d]
            if buf['B']:
                line.setData(buf['B'], buf['vals'][ai])

    def _refresh_map_slot(self, slot_idx: int):
        if self.map_data is None or self.locked_cfg is None:
            return
        direction = 'fwd' if slot_idx == 0 else 'bwd'
        combo = self.map_slot_combos[slot_idx]
        ai = combo.currentData()
        if ai is None:
            return
        img = self.map_slot_images[slot_idx]
        arr = self.map_data[ai][direction]
        # arr shape = (num_outer, num_B); ImageItem wants (cols, rows) → transpose
        img.setImage(arr.T, autoLevels=True)
        # Map pixels → (B, outer) coordinates
        cfg = self.locked_cfg
        mc = cfg.magnet_config
        num_B    = mc.num_B_points
        num_outer = cfg.num_outer
        B_min, B_max = mc.B_min_T, mc.B_max_T
        outer_min, outer_max = cfg.outer_min, cfg.outer_max
        if num_B > 1 and num_outer > 1:
            # ★ v1.1 Bug #25: half-pixel alignment.
            # ImageItem pixel (i, j) covers extent [i, i+1] x [j, j+1] in raw
            # image space. We want pixel CENTERS to land on the canonical
            # grid points, so the image's lower-left corner must be at
            # (grid[0] - dx/2, grid[0] - dy/2) and pixel size = (dx, dy).
            dx = (B_max - B_min) / (num_B - 1)
            dy = (outer_max - outer_min) / (num_outer - 1)
            tr = QTransform()
            tr.translate(B_min - dx / 2.0, outer_min - dy / 2.0)
            tr.scale(dx, dy)
            img.setTransform(tr)

    # =================================================================
    # Log area
    # =================================================================
    def _build_log_area(self):
        gb = QGroupBox("Log")
        v = QVBoxLayout(gb)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        # Progress + step labels
        prog_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        prog_row.addWidget(self.progress_bar, 1)
        self.lbl_step    = QLabel("Step 0 / 0")
        self.lbl_step.setObjectName("valueLabel")
        prog_row.addWidget(self.lbl_step)
        self.lbl_elapsed = QLabel("Elapsed  0s")
        prog_row.addWidget(self.lbl_elapsed)
        self.lbl_eta     = QLabel("ETA  —")
        prog_row.addWidget(self.lbl_eta)
        v.addLayout(prog_row)

        # Now-line
        self.lbl_now = QLabel("—")
        self.lbl_now.setStyleSheet(f"color:{CT_SUBTEXT1}; font-family: monospace;")
        v.addWidget(self.lbl_now)

        # Event log
        self.event_log = QPlainTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumBlockCount(2000)
        self.event_log.setStyleSheet(
            f"QPlainTextEdit {{ background-color:{CT_MANTLE}; color:{CT_TEXT}; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; font-size: 10pt; }}")
        v.addWidget(self.event_log, 1)
        return gb

    def log_event(self, text: str, level: str = "info"):
        ts = time.strftime("%H:%M:%S")
        color = {
            'info':    CT_TEXT,
            'success': CT_GREEN,
            'warning': CT_YELLOW,
            'error':   CT_RED,
        }.get(level, CT_TEXT)
        # Escape HTML
        safe = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html = f'<span style="color:{CT_OVERLAY0}">[{ts}]</span> <span style="color:{color}">{safe}</span>'
        self.event_log.appendHtml(html)

    # =================================================================
    # Menu bar + status bar
    # =================================================================
    def _init_menubar(self):
        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        act_save = QAction("Save settings", self)
        act_save.triggered.connect(self._save_settings)
        file_menu.addAction(act_save)
        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        scan_menu = mb.addMenu("&Scan")
        self.act_start = QAction("Start scan", self)
        self.act_start.setShortcut(QKeySequence("F5"))
        self.act_start.triggered.connect(self.start_measurement)
        scan_menu.addAction(self.act_start)
        self.act_stop = QAction("Abort scan", self)
        self.act_stop.setShortcut(QKeySequence("Esc"))
        self.act_stop.triggered.connect(self.stop_measurement)
        self.act_stop.setEnabled(False)
        scan_menu.addAction(self.act_stop)

    def _init_statusbar(self):
        sb = self.statusBar()
        # ★ v1.1 Bug #18: persistent DEMO/LIVE mode badge.
        # Always visible so the user can never forget which mode they're in.
        self.lbl_mode_badge = QLabel(f"  [{'DEMO' if DEMO_MODE else 'LIVE'}]  ")
        badge_bg = CT_PEACH if DEMO_MODE else CT_GREEN
        self.lbl_mode_badge.setStyleSheet(
            f"color:{CT_BASE}; background-color:{badge_bg}; "
            f"font-weight: 700; padding: 1px 8px; border-radius: 4px; "
            f"font-family: 'JetBrains Mono', monospace;")
        sb.addPermanentWidget(self.lbl_mode_badge)

        self.lbl_hyst_area = QLabel("Hysteresis area: —")
        self.lbl_hyst_area.setStyleSheet(
            f"color:{CT_GREEN}; font-family: 'JetBrains Mono', monospace; padding: 0 8px;")
        sb.addPermanentWidget(self.lbl_hyst_area)

    # =================================================================
    # Param parsing
    # =================================================================
    def _parse_params(self) -> dict:
        """Parse the entire GUI into a params dict for MeasurementThread."""
        # ---- Outer sweep ----
        outer_axis = self.cb_outer_axis.currentData() or 'n'
        try:
            outer_min = float(self.le_outer_min.text())
            outer_max = float(self.le_outer_max.text())
            num_outer = int(self.le_num_outer.text())
            fixed_value = float(self.le_fixed.text())
        except ValueError:
            raise ValueError("Outer sweep parsing failed.")
        if outer_max <= outer_min:
            raise ValueError("Outer max must be > outer min.")
        if num_outer < 2:
            raise ValueError("Num outer points must be ≥ 2.")

        # ---- Geometry ----
        try:
            d_t  = float(self.geom_inputs['d_t'].text())
            d_b  = float(self.geom_inputs['d_b'].text())
            eps  = float(self.geom_inputs['eps'].text())
            Vtg0 = float(self.geom_inputs['Vtg0'].text())
            Vbg0 = float(self.geom_inputs['Vbg0'].text())
            Vtg_max = float(self.geom_inputs['Vtg_max'].text())
            Vbg_max = float(self.geom_inputs['Vbg_max'].text())
        except ValueError:
            raise ValueError("Geometry parameter parsing failed.")
        if d_t <= 0 or d_b <= 0 or eps <= 0:
            raise ValueError("Geometry: d_t, d_b, eps must be > 0.")
        if Vtg_max <= 0 or Vbg_max <= 0:
            raise ValueError("Vtg_max, Vbg_max must be > 0.")
        geometry = GeometryConfig(
            d_t_nm=d_t, d_b_nm=d_b, eps_hBN=eps,
            Vtg0=Vtg0, Vbg0=Vbg0,
            Vtg_max=Vtg_max, Vbg_max=Vbg_max,
        )

        # ---- Magnet ----
        try:
            B_min     = float(self.le_B_min.text())
            B_max     = float(self.le_B_max.text())
            ramp_rate = float(self.le_ramp_rate.text())
            B_step    = float(self.le_B_step.text())
            field_tol = float(self.le_B_tol.text())
        except ValueError:
            raise ValueError("Magnet parameter parsing failed.")
        if B_max <= B_min:
            raise ValueError("B_max must be > B_min.")
        if ramp_rate <= 0:
            raise ValueError("Ramp rate must be > 0 T/min.")
        if B_step <= 0:
            raise ValueError("B_step must be > 0 T.")
        if field_tol <= 0:
            raise ValueError("Field tolerance must be > 0 T.")
        magnet_config = MagnetConfig(
            controller_type=self.cb_magnet_type.currentData(),
            visa_resource=self.le_magnet_visa.text().strip(),
            magnet_uid=self.le_magnet_uid.text().strip(),
            B_min_T=B_min, B_max_T=B_max,
            ramp_rate_T_per_min=ramp_rate,
            B_step_T=B_step,
            field_tolerance_T=field_tol,
        )

        # ---- Timing ----
        try:
            t_read              = float(self.timing_inputs['t_read'].text())
            t_settle_slow       = float(self.timing_inputs['t_settle_slow'].text())
            t_settle_after_fwd  = float(self.timing_inputs['t_settle_after_fwd'].text())
            t_settle_after_bwd  = float(self.timing_inputs['t_settle_after_bwd'].text())
        except ValueError:
            raise ValueError("Timing parameter parsing failed.")
        # ★ v1.1 Bug #32: timing must be physically meaningful
        if t_read <= 0:
            raise ValueError("t_read must be > 0 s.")
        if t_read < 1e-3:
            raise ValueError("t_read < 1 ms is below the DAQ minimum.")
        if t_settle_slow < 0 or t_settle_after_fwd < 0 or t_settle_after_bwd < 0:
            raise ValueError("Settle times must be ≥ 0 s.")
        # Warn (don't reject) if t_read > t_sample — the sampling rate will
        # be limited by t_read in that case, but it's still a valid scan.
        t_sample_target = magnet_config.t_sample_s
        if t_read > t_sample_target:
            self.log_event(
                f"NOTE: t_read ({t_read*1000:.0f} ms) > t_sample target "
                f"({t_sample_target*1000:.0f} ms). Effective sampling rate "
                f"will be limited by t_read; B_step in metadata may not "
                f"reflect actual coverage. Consider lowering t_read or "
                f"raising B_step.", "warning")

        # ---- Lockin / source ----
        try:
            v_osc    = float(self.hw_inputs['v_osc'].text())
            r_series = float(self.hw_inputs['r_series'].text())
        except ValueError:
            raise ValueError("Lockin parsing failed.")
        if v_osc == 0 or r_series <= 0:
            raise ValueError("V_osc must be nonzero and R_series > 0.")

        # ---- Channels ----
        channels = []
        for ai in range(NUM_AI):
            kind = self.ch_kind_combos[ai].currentText()
            sens, gain = 0.0, 0.0
            if kind == 'R':
                try:
                    sens = float(self.ch_sens_inputs[ai].text())
                    gain = float(self.ch_gain_inputs[ai].text())
                except ValueError:
                    raise ValueError(f"AI{ai}: sens/gain parsing failed.")
                if sens <= 0 or gain <= 0:
                    raise ValueError(f"AI{ai}: sens and gain must be > 0.")
            channels.append(ChannelConfig(
                ai_index=ai,
                name=self._current_channel_name(ai),
                enabled=self.ch_enable_checks[ai].isChecked(),
                kind=kind, sens=sens, gain=gain,
            ))
        # ★ v1.1 Bug #33: at least one channel must be enabled
        if not any(c.enabled for c in channels):
            raise ValueError(
                "All channels are disabled — nothing would be recorded. "
                "Enable at least one AI channel in the Channels group.")
        # Warn if hyst channel is disabled
        hyst_ai_check = self.cb_hyst_channel.currentData()
        if hyst_ai_check is not None:
            try:
                if not channels[int(hyst_ai_check)].enabled:
                    raise ValueError(
                        f"Hysteresis channel AI{hyst_ai_check} is disabled. "
                        f"Either enable it in the Channels group, or pick a "
                        f"different Hyst channel in the Magnet group.")
            except (IndexError, TypeError):
                pass

        # ---- AO physical channels ----
        top_ao = self.cb_top_ao.currentData() or 'AO0'
        bot_ao = 'AO1' if top_ao == 'AO0' else 'AO0'
        ao_map = {'AO0': self.hw.ao_chans[0], 'AO1': self.hw.ao_chans[1]}
        ao_top_ch = ao_map[top_ao]
        ao_bot_ch = ao_map[bot_ao]

        # ---- Hysteresis channel ----
        hyst_ai = self.cb_hyst_channel.currentData()
        if hyst_ai is None:
            hyst_ai = 0

        hysteresis_config = HysteresisConfig(
            geometry=geometry,
            magnet_config=magnet_config,
            outer_axis=outer_axis,
            outer_min=outer_min,
            outer_max=outer_max,
            num_outer=num_outer,
            fixed_value=fixed_value,
            t_read=t_read,
            t_settle_slow=t_settle_slow,
            t_settle_after_fwd=t_settle_after_fwd,
            t_settle_after_bwd=t_settle_after_bwd,
            ao_top=ao_top_ch,
            ao_bot=ao_bot_ch,
            hysteresis_channel_ai=int(hyst_ai),
        )

        # ---- Output path ----
        folder = self.le_folder.text().strip() or os.getcwd()
        os.makedirs(folder, exist_ok=True)
        self._refresh_filename()
        save_path = os.path.join(folder, self.lbl_filename.text())
        # ★ v1.1 Bug #34: prevent silent overwrite. The filename includes
        # a YYYYMMDD_HHMMSS timestamp so two scans started in the same
        # second would otherwise overwrite each other. Append _2, _3, ...
        # until we find an unused name. Also check for the JSON sidecar.
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
                raise RuntimeError(
                    f"Could not find an unused filename in {folder} after "
                    f"trying 1000 suffixes. Clean up old data files.")

        # ---- Magnet controller ----
        magnet = make_magnet_controller(magnet_config)

        return dict(
            hysteresis_config=hysteresis_config,
            magnet=magnet,
            channels=channels,
            v_osc=v_osc, r_series=r_series,
            save_path=save_path,
            sample=self.le_sample.text().strip(),
            device=self.le_device.text().strip(),
            operator=self.le_operator.text().strip(),
            run_name=self.le_run_name.text().strip(),
        )

    # =================================================================
    # Start / Stop
    # =================================================================
    def start_measurement(self):
        if self.thread and self.thread.isRunning():
            self.log_event("A scan is already running.", "warning")
            return
        try:
            params = self._parse_params()
        except Exception as e:
            self.log_event(f"Cannot start: {e}", "error")
            QMessageBox.critical(self, "Cannot start", str(e))
            return

        cfg = params['hysteresis_config']

        # ---- Strict preflight: every outer point must be within hBN limits ----
        bad = cfg.precheck_any_out_of_limits()
        if bad is not None:
            i_idx, v, Vtg, Vbg = bad
            unit = '×10¹² cm⁻²' if cfg.outer_axis == 'n' else 'V/nm'
            msg = (f"Outer point {i_idx} ({cfg.outer_axis}={v:+.4g} {unit}) "
                   f"reverse-maps to (Vtg={Vtg:+.3f}V, Vbg={Vbg:+.3f}V), "
                   f"outside hBN limits "
                   f"(|Vtg|≤{cfg.geometry.Vtg_max}V, |Vbg|≤{cfg.geometry.Vbg_max}V).")
            self.log_event("Preflight FAIL: " + msg, "error")
            QMessageBox.critical(self, "Preflight failed",
                "Cannot start hysteresis scan.\n\n" + msg + "\n\n"
                "Adjust outer range, fixed value, geometry, or voltage limits.")
            return
        self.log_event("Preflight: all outer points within hBN limits.", "success")

        # ★ v1.1 Bug #2: LIVE-mode safety — for large field ranges, require
        # the user to have run "Verify magnet comms" at least once this
        # session. Refuses to start otherwise.
        mc_check = cfg.magnet_config
        B_range = abs(mc_check.B_max_T - mc_check.B_min_T)
        if (not DEMO_MODE
                and mc_check.controller_type != MAGNET_DEMO
                and B_range > 1.0
                and not self._magnet_verified):
            self.log_event(
                f"Refusing to start: magnet not verified this session and "
                f"B-range = {B_range:.2f} T > 1 T threshold.", "error")
            QMessageBox.critical(self, "Magnet not verified",
                f"This is a LIVE-mode scan with a large magnetic field range "
                f"({B_range:.2f} T).\n\n"
                f"Before starting, please click the 'Verify magnet comms' "
                f"button in the Magnet group to confirm the controller "
                f"responds correctly. This is a one-time-per-session check "
                f"to catch GPIB or firmware-version issues that could "
                f"otherwise damage the magnet.\n\n"
                f"For small ramps (B-range ≤ 1 T) this check is not enforced.")
            return

        # ---- Lock state ----
        self.locked_channels = list(params['channels'])
        self.locked_cfg      = cfg
        self.hyst_area_per_outer = [float('nan')] * cfg.num_outer

        # ---- Allocate 2D map data ----
        num_B = cfg.magnet_config.num_B_points
        self.map_data = {
            ai: {d: np.full((cfg.num_outer, num_B), np.nan) for d in DIRECTIONS}
            for ai in range(NUM_AI)
        }

        # ---- Lock plot ranges ----
        mc = cfg.magnet_config
        for pw in self.line_slot_plots:
            pw.setLabel('bottom', 'B (T)', color=CT_TEXT)
            pw.setXRange(mc.B_min_T, mc.B_max_T, padding=0)
            pw.enableAutoRange(axis='x', enable=False)
            pw.enableAutoRange(axis='y', enable=True)
        outer_label = 'n (×10¹² cm⁻²)' if cfg.outer_axis == 'n' else 'D (V/nm)'
        for pw in self.map_slot_plots:
            pw.setLabel('bottom', 'B (T)', color=CT_TEXT)
            pw.setLabel('left', outer_label, color=CT_TEXT)
            pw.setXRange(mc.B_min_T, mc.B_max_T, padding=0)
            pw.setYRange(cfg.outer_min, cfg.outer_max, padding=0)
            pw.enableAutoRange(axis='x', enable=False)
            pw.enableAutoRange(axis='y', enable=False)
        for img in self.map_slot_images:
            img.clear()

        # ---- Rebuild line slot lines (so colors apply) ----
        for slot_idx in range(NUM_LINE_SLOTS):
            self._rebuild_line_slot_lines(slot_idx)

        # ---- Reset 1D buffer + counters ----
        self.curr_outer_idx = -1
        self.curr_buf = self._make_empty_1d_buffers()
        self._total_points = cfg.num_outer * 2 * num_B
        self._points_done  = 0
        self._scan_started_at = time.monotonic()
        self.progress_bar.setValue(0)
        self.lbl_step.setText(f"Step 0 / {self._total_points}")
        self.lbl_elapsed.setText("Elapsed  0s")
        self.lbl_eta.setText("ETA  —")
        self.lbl_now.setText("—")
        self.lbl_hyst_area.setText("Hysteresis area: —")

        # ---- Launch thread ----
        self.thread = MeasurementThread(params, self.hw)
        self.thread.point_ready.connect(self.on_point)
        self.thread.outer_point_finished.connect(self.on_outer_point_finished)
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
            self.log_event("Abort requested by user.", "warning")
            self.thread.stop()

    # =================================================================
    # Thread signal handlers
    # =================================================================
    def on_point(self, info: PointInfo):
        # ★ v1.1 Bug #31: defensive — if scan finished and cleared
        # locked_cfg/map_data before this signal was processed, ignore.
        if (self.locked_cfg is None
                or self.map_data is None
                or self.locked_channels is None):
            return
        # If outer index changed, clear the 1D buffer (new R(B) pair)
        if info.i_idx != self.curr_outer_idx:
            self.curr_outer_idx = info.i_idx
            self.curr_buf = self._make_empty_1d_buffers()
            for slot_idx in range(NUM_LINE_SLOTS):
                for line in self.line_slot_lines[slot_idx].values():
                    line.setData([], [])

        # Append to 1D buffer
        buf = self.curr_buf[info.direction]
        buf['B'].append(info.B_actual)
        for ai in range(NUM_AI):
            buf['vals'][ai].append(info.values[ai])

        # Write to 2D map at j_idx
        if 0 <= info.j_idx < self.map_data[0][info.direction].shape[1]:
            for ai in range(NUM_AI):
                self.map_data[ai][info.direction][info.i_idx, info.j_idx] = info.values[ai]

        # Update line items for this direction (only the relevant ones)
        for slot_idx in range(NUM_LINE_SLOTS):
            for (ai, d), line in self.line_slot_lines[slot_idx].items():
                if d == info.direction:
                    line.setData(buf['B'], buf['vals'][ai])

        # Progress
        self._points_done += 1
        pct = int(round(100 * self._points_done / max(self._total_points, 1)))
        self.progress_bar.setValue(min(pct, 100))
        self.lbl_step.setText(f"Step {self._points_done} / {self._total_points}")
        elapsed = time.monotonic() - self._scan_started_at
        self.lbl_elapsed.setText(f"Elapsed  {fmt_dur(elapsed)}")
        if self._points_done > 0:
            eta = elapsed * (self._total_points - self._points_done) / self._points_done
            self.lbl_eta.setText(f"ETA  {fmt_dur(eta)}")
        unit = '×10¹² cm⁻²' if self.locked_cfg.outer_axis == 'n' else 'V/nm'
        self.lbl_now.setText(
            f"outer[{info.i_idx}]={info.outer_target:+.4g} {unit}  |  "
            f"B={info.B_actual:+.4f} T  ({info.direction})  |  "
            f"j={info.j_idx}")

    def on_outer_point_finished(self, i_idx: int, hyst_area: float):
        # ★ v1.1 Bug #13: defensive — if scan already finished and
        # cleared locked_cfg before this signal was processed, ignore.
        if self.locked_cfg is None or self.locked_channels is None:
            return
        # Refresh both 2D maps (they get a new row of data)
        for slot_idx in range(NUM_MAP_SLOTS):
            self._refresh_map_slot(slot_idx)
        # Store + display hysteresis area
        if 0 <= i_idx < len(self.hyst_area_per_outer):
            self.hyst_area_per_outer[i_idx] = hyst_area
        # Status bar
        cfg = self.locked_cfg
        ai = cfg.hysteresis_channel_ai
        ch_name = self._current_channel_name(ai)
        ch_kind = self.locked_channels[ai].kind if self.locked_channels else 'R'
        unit = 'Ω·T' if ch_kind == 'R' else f'{KIND_UNIT.get(ch_kind, "")}·T'
        # NaN-safe display: don't print "nan Ω·T" — show em-dash
        if math.isnan(hyst_area):
            area_text = "—"
        else:
            area_text = f"{hyst_area:.4g}"
        self.lbl_hyst_area.setText(
            f"Hysteresis area [{ch_name}]: {area_text} {unit}  "
            f"(outer {i_idx + 1}/{cfg.num_outer})")

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
        self.locked_cfg = None

    def on_error(self, tb: str):
        self.log_event("Scan crashed (traceback below).", "error")
        for line in tb.strip().splitlines()[-6:]:
            self.log_event("    " + line, "error")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.act_start.setEnabled(True)
        self.act_stop.setEnabled(False)
        self._set_config_enabled(True)
        self._set_state("Error", CT_RED)
        self.locked_cfg = None
        print(tb, file=sys.stderr)

    # =================================================================
    # Config widget freeze
    # =================================================================
    def _config_widgets(self):
        widgets = [
            self.le_sample, self.le_device, self.le_operator, self.le_run_name,
            self.cb_outer_axis, self.le_outer_min, self.le_outer_max,
            self.le_num_outer, self.le_fixed,
            self.cb_top_ao,
            self.cb_magnet_type, self.le_magnet_visa, self.le_magnet_uid,
            self.le_B_min, self.le_B_max, self.le_ramp_rate, self.le_B_step,
            self.le_B_tol, self.cb_hyst_channel,
            self.le_folder,
        ]
        widgets += list(self.geom_inputs.values())
        widgets += list(self.timing_inputs.values())
        widgets += list(self.hw_inputs.values())
        widgets += list(self.ch_name_inputs)
        widgets += list(self.ch_enable_checks)
        widgets += list(self.ch_kind_combos)
        widgets += list(self.ch_sens_inputs)
        widgets += list(self.ch_gain_inputs)
        return widgets

    def _set_config_enabled(self, enabled: bool):
        for w in self._config_widgets():
            w.setEnabled(enabled)
        if enabled:
            for ai in range(NUM_AI):
                self._refresh_kind_widgets(ai)

    # =================================================================
    # Persistence (QSettings)
    # =================================================================
    def _persistent_widgets(self):
        items = [
            ('lineedit', 'meta/sample',       self.le_sample),
            ('lineedit', 'meta/device',       self.le_device),
            ('lineedit', 'meta/operator',     self.le_operator),
            ('lineedit', 'meta/run_name',     self.le_run_name),
            ('lineedit', 'outer/min',         self.le_outer_min),
            ('lineedit', 'outer/max',         self.le_outer_max),
            ('lineedit', 'outer/num',         self.le_num_outer),
            ('lineedit', 'outer/fixed',       self.le_fixed),
            ('lineedit', 'magnet/visa',       self.le_magnet_visa),
            ('lineedit', 'magnet/uid',        self.le_magnet_uid),
            ('lineedit', 'magnet/B_min',      self.le_B_min),
            ('lineedit', 'magnet/B_max',      self.le_B_max),
            ('lineedit', 'magnet/ramp_rate',  self.le_ramp_rate),
            ('lineedit', 'magnet/B_step',     self.le_B_step),
            ('lineedit', 'magnet/B_tol',      self.le_B_tol),
            ('lineedit', 'output/folder',     self.le_folder),
        ]
        for k, le in self.geom_inputs.items():
            items.append(('lineedit', f'geometry/{k}', le))
        for k, le in self.timing_inputs.items():
            items.append(('lineedit', f'timing/{k}', le))
        for k, le in self.hw_inputs.items():
            items.append(('lineedit', f'lockin/{k}', le))
        for ai in range(NUM_AI):
            items.append(('lineedit', f'channels/{ai}/name',    self.ch_name_inputs[ai]))
            items.append(('checkbox', f'channels/{ai}/enabled', self.ch_enable_checks[ai]))
            items.append(('combo',    f'channels/{ai}/kind',    self.ch_kind_combos[ai]))
            items.append(('lineedit', f'channels/{ai}/sens',    self.ch_sens_inputs[ai]))
            items.append(('lineedit', f'channels/{ai}/gain',    self.ch_gain_inputs[ai]))
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

        # userData combos
        for key, combo in [
            ('outer/axis',           self.cb_outer_axis),
            ('gates/top_ao',         self.cb_top_ao),
            ('magnet/controller_type', self.cb_magnet_type),
        ]:
            v = s.value(key)
            if v is not None:
                for k in range(combo.count()):
                    if combo.itemData(k) == str(v):
                        combo.setCurrentIndex(k)
                        break

        # Hyst channel (int userData)
        v = s.value('magnet/hyst_channel')
        if v is not None:
            try:
                target = int(v)
                for k in range(self.cb_hyst_channel.count()):
                    if self.cb_hyst_channel.itemData(k) == target:
                        self.cb_hyst_channel.setCurrentIndex(k)
                        break
            except Exception:
                pass

        # Line slot selections
        for slot_idx in range(NUM_LINE_SLOTS):
            v = s.value(f'line_slots/{slot_idx}/sel', '')
            if v:
                try:
                    sel = {int(x) for x in str(v).split(',') if x}
                    actions = self.line_slot_actions[slot_idx]
                    for ai, a in actions.items():
                        a.setChecked(ai in sel)
                except Exception:
                    pass

        # Map slot AI selections (direction is fixed per slot)
        for slot_idx in range(NUM_MAP_SLOTS):
            v = s.value(f'map_slots/{slot_idx}/ai')
            if v is not None:
                try:
                    target = int(v)
                    combo = self.map_slot_combos[slot_idx]
                    for k in range(combo.count()):
                        if combo.itemData(k) == target:
                            combo.setCurrentIndex(k)
                            break
                except Exception:
                    pass

        # Window geometry
        geom = s.value('window/geometry')
        if geom is not None:
            self.restoreGeometry(geom)
        # ★ v1.1 Bug #26: splitter sizes round-trip differs by platform.
        # Linux: stores Python list → returns Python list of ints.
        # Windows registry: stores list as comma-string → returns string,
        # OR with type=list returns ['720','280'] (list of strings).
        # macOS plist: returns list of strings.
        # Handle all three cases:
        for name, split in (('h_split', getattr(self, 'h_split', None)),
                            ('v_split', getattr(self, 'v_split', None))):
            if split is None:
                continue
            sizes = s.value(f'window/{name}')
            if not sizes:
                continue
            try:
                if isinstance(sizes, str):
                    parsed = [int(x) for x in sizes.split(',') if x.strip()]
                elif isinstance(sizes, (list, tuple)):
                    parsed = [int(x) for x in sizes]
                else:
                    parsed = []
                if parsed:
                    split.setSizes(parsed)
            except (ValueError, TypeError):
                pass    # malformed, ignore — defaults will apply

    def _save_settings(self):
        s = QSettings(ORG_NAME, SETTINGS_NAME)
        for kind, key, w in self._persistent_widgets():
            if kind == 'lineedit':
                s.setValue(key, w.text())
            elif kind == 'checkbox':
                s.setValue(key, w.isChecked())
            elif kind == 'combo':
                s.setValue(key, w.currentText())
        # userData combos
        s.setValue('outer/axis',             self.cb_outer_axis.currentData())
        s.setValue('gates/top_ao',           self.cb_top_ao.currentData())
        s.setValue('magnet/controller_type', self.cb_magnet_type.currentData())
        # Hyst channel as int
        v = self.cb_hyst_channel.currentData()
        if v is not None:
            s.setValue('magnet/hyst_channel', int(v))
        # Line slots
        for slot_idx, actions in enumerate(self.line_slot_actions):
            sel = ','.join(str(ai) for ai, a in actions.items() if a.isChecked())
            s.setValue(f'line_slots/{slot_idx}/sel', sel)
        # Map slot AI
        for slot_idx, combo in enumerate(self.map_slot_combos):
            ai = combo.currentData()
            if ai is not None:
                s.setValue(f'map_slots/{slot_idx}/ai', int(ai))
        # Window geometry
        s.setValue('window/geometry', self.saveGeometry())
        if hasattr(self, 'h_split'):
            s.setValue('window/h_split', self.h_split.sizes())
        if hasattr(self, 'v_split'):
            s.setValue('window/v_split', self.v_split.sizes())

    # =================================================================
    # Close handling
    # =================================================================
    def closeEvent(self, event):
        # ★ v1.1 Bug #15: wait long enough for any pending GPIB read to time
        # out (Mercury VISA timeout = MERCURY_VISA_TIMEOUT_MS = 1000ms in v1.1)
        # plus a generous safety margin. If the thread still hasn't finished
        # after that, forcibly terminate to avoid race with hw.close().
        if self.thread and self.thread.isRunning():
            reply = QMessageBox.question(
                self, "Scan in progress",
                "A measurement is running. Abort and quit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self.thread.stop()
            # Wait for the worker to acknowledge stop and exit cleanly.
            # MERCURY_VISA_TIMEOUT_MS (1s) + magnet hold/disconnect (~1s) +
            # gate-down ramp (~2s) + safety = 8s.
            if not self.thread.wait(8000):
                self.log_event(
                    "Worker did not exit within 8 s — forcing termination. "
                    "Magnet may not have been cleanly disconnected.",
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
        try:
            self.hw.close()
        except Exception as e:
            print(f"WARN: hw close failed: {e}", file=sys.stderr)
        event.accept()


# =====================================================================
# Entry point — CLI flags, DEMO/LIVE switch, signal handling
# =====================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='hysteresis_mapping.py',
        description=f'{APP_NAME} {APP_VERSION} — magnetic hysteresis R(B) measurement.',
        epilog="If neither --live nor --demo is given, runs in DEMO (simulated) mode.",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument('--live', action='store_true',
                   help='Use real NI DAQ + magnet hardware. Requires nidaqmx + pyvisa.')
    g.add_argument('--demo', action='store_true',
                   help='Use simulated DAQ + DemoMagnet (default).')
    p.add_argument('--version', action='version',
                   version=f'{APP_NAME} {APP_VERSION}')
    return p


def main():
    global DEMO_MODE

    args = _build_arg_parser().parse_args()
    # Default DEMO if neither flag is given. --live overrides.
    DEMO_MODE = not args.live

    # Lazy-load nidaqmx only in LIVE mode, with a helpful error if missing
    if not DEMO_MODE:
        try:
            _import_daq_libs()
        except RuntimeError as e:
            print(f"FATAL: {e}", file=sys.stderr)
            sys.exit(1)

    # ★ v1.1 Bug #16: Make Ctrl+C work reliably across platforms.
    # SIG_DFL alone is not enough on Linux because Qt's event loop blocks
    # in C and Python signals are only handled between bytecode ops. A
    # 100 ms QTimer that does nothing wakes the event loop periodically
    # so the SIGINT handler gets a chance to fire.
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)

    sigint_wakeup = QTimer()
    sigint_wakeup.start(100)
    sigint_wakeup.timeout.connect(lambda: None)

    # pyqtgraph defaults: white background → override at widget level
    pg.setConfigOption('background', CT_BASE)
    pg.setConfigOption('foreground', CT_TEXT)
    pg.setConfigOptions(antialias=True)

    gui = HysteresisGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
