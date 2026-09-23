# -*- coding: utf-8 -*-
"""
main_app.py
===========
吸收光譜 Elliott formula 擬合工具 (PyQt5 GUI)

功能:
  1. 匯入實驗數據 (txt / csv / dat，兩欄: 能量(eV), 吸收度)
  2. 於圖上以滑鼠拖曳選擇擬合的能量範圍 (或手動輸入)
  3. 設定每個參數的初始值 / 上下限 / 是否參與擬合 (vary)
  4. 執行非線性最小平方擬合 (lmfit)
  5. 顯示擬合後的重要參數 (Eg, E0/Eb, Gamma_ex, Gamma_c, b, alpha0) 與統計量 (R^2, reduced chi-square)
  6. 圖上疊加畫出: 實驗數據、激子吸收項、連續態吸收項、總合擬合曲線 (各自不同顏色)
  7. 匯出結果為 tab-delimited 文字檔，可直接被 Origin 匯入作圖 (同一 X 欄, 多個 Y 欄)

執行方式:
    python3 main_app.py
"""

import sys
import os
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QDoubleSpinBox, QSpinBox, QCheckBox, QGroupBox, QSplitter, QMessageBox,
    QHeaderView, QComboBox, QTextEdit, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from fitting_engine import PARAM_INFO, run_fit, goodness_of_fit, get_component_curves
from elliott_model import total_absorption
from estimator import estimate_initial_parameters


# ----------------------------------------------------------------------
# 數據匯入輔助函式
# ----------------------------------------------------------------------
def load_spectrum_file(path):
    """
    嘗試讀取兩欄數值資料 (能量, 吸收度)。
    自動略過無法轉為數字的表頭行，自動偵測分隔符號 (逗號 / Tab / 空白)。
    回傳依能量遞增排序後的 (E, A) numpy array。
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for sep in ("\t", ",", ";"):
                if sep in line:
                    parts = [p.strip() for p in line.split(sep) if p.strip() != ""]
                    break
            else:
                parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            rows.append((x, y))
    if not rows:
        raise ValueError("無法從檔案中解析出任何數值資料，請確認檔案格式為兩欄數值 (能量, 吸收度)。")
    arr = np.array(rows, dtype=float)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return arr[:, 0], arr[:, 1]


# ----------------------------------------------------------------------
# 主視窗
# ----------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("吸收光譜擬合工具 — Elliott Formula (激子 + 連續態)")
        self.resize(1400, 850)

        # ---- 資料狀態 ----
        self.E_full = None
        self.A_full = None
        self.data_filename = None
        self.fit_range = None          # (Emin, Emax)
        self.fit_result = None         # lmfit MinimizerResult
        self.last_curves = None        # (E_used, A_used, exc, cont, total)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI 建構
    # ------------------------------------------------------------------
    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # ============ 左側控制面板 (含捲軸) ============
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(430)
        left_scroll.setMaximumWidth(480)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_scroll.setWidget(left_panel)

        left_layout.addWidget(self._build_import_group())
        left_layout.addWidget(self._build_range_group())
        left_layout.addWidget(self._build_param_group())
        left_layout.addWidget(self._build_fit_group())
        left_layout.addWidget(self._build_result_group())
        left_layout.addWidget(self._build_export_group())
        left_layout.addStretch(1)

        # ============ 右側繪圖區 ============
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.figure = Figure(figsize=(7, 6))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        self.status_label = QLabel("請先匯入實驗數據 (File > 匯入 或 左側按鈕)。")
        self.status_label.setStyleSheet("color: #555; padding: 4px;")
        right_layout.addWidget(self.status_label)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._init_plot()

        # span selector 於 _init_plot 內建立 (需要 ax 存在)
        self.span_selector = SpanSelector(
            self.ax, self._on_span_select, "horizontal",
            useblit=True, interactive=True,
            props=dict(alpha=0.2, facecolor="tab:blue"),
        )
        self.span_selector.set_active(False)

    # ---- 匯入區塊 ----
    def _build_import_group(self):
        box = QGroupBox("1. 匯入實驗數據")
        layout = QVBoxLayout(box)

        btn_import = QPushButton("匯入數據檔案 (.txt / .csv / .dat)")
        btn_import.clicked.connect(self.on_import_data)
        layout.addWidget(btn_import)

        self.label_filename = QLabel("尚未載入檔案")
        self.label_filename.setWordWrap(True)
        self.label_filename.setStyleSheet("color: #666;")
        layout.addWidget(self.label_filename)

        return box

    # ---- 範圍選擇區塊 ----
    def _build_range_group(self):
        box = QGroupBox("2. 選擇擬合的能量範圍")
        layout = QGridLayout(box)

        self.btn_toggle_select = QPushButton("啟動「拖曳選擇範圍」")
        self.btn_toggle_select.setCheckable(True)
        self.btn_toggle_select.clicked.connect(self.on_toggle_span_select)
        layout.addWidget(self.btn_toggle_select, 0, 0, 1, 3)

        layout.addWidget(QLabel("Emin (eV):"), 1, 0)
        self.spin_emin = QDoubleSpinBox()
        self.spin_emin.setRange(-1000, 1000)
        self.spin_emin.setDecimals(4)
        self.spin_emin.setSingleStep(0.01)
        layout.addWidget(self.spin_emin, 1, 1)

        layout.addWidget(QLabel("Emax (eV):"), 2, 0)
        self.spin_emax = QDoubleSpinBox()
        self.spin_emax.setRange(-1000, 1000)
        self.spin_emax.setDecimals(4)
        self.spin_emax.setSingleStep(0.01)
        layout.addWidget(self.spin_emax, 2, 1)

        btn_apply_range = QPushButton("套用範圍")
        btn_apply_range.clicked.connect(self.on_apply_manual_range)
        layout.addWidget(btn_apply_range, 1, 2, 2, 1)

        btn_reset_range = QPushButton("重設為全部數據範圍")
        btn_reset_range.clicked.connect(self.on_reset_range)
        layout.addWidget(btn_reset_range, 3, 0, 1, 3)

        return box

    # ---- 參數設定區塊 ----
    def _build_param_group(self):
        box = QGroupBox("3. 擬合參數設定 (初始值 / 下限 / 上限 / 參與擬合)")
        layout = QVBoxLayout(box)

        btn_auto = QPushButton("自動估算初始值 (依目前選擇範圍的資料)")
        btn_auto.setStyleSheet("font-weight: bold; color: #1a5d1a;")
        btn_auto.clicked.connect(self.on_auto_estimate)
        layout.addWidget(btn_auto)

        hint = QLabel(
            "提示：Eg（能隙）與 E0（激子束縛能）最關鍵，請先目視資料抓大概位置；\n"
            "Γex/Γc 影響曲線寬窄；b 影響高能側的形狀，通常較不敏感，可先用預設值。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(hint)

        self.table_params = QTableWidget(len(PARAM_INFO), 5)
        self.table_params.setHorizontalHeaderLabels(
            ["參數", "初始值", "下限", "上限", "擬合"]
        )
        self.table_params.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_params.verticalHeader().setVisible(False)

        self.param_widgets = {}
        for row, (key, label, default_val, default_min, default_max) in enumerate(PARAM_INFO):
            item_name = QTableWidgetItem(label)
            item_name.setFlags(Qt.ItemIsEnabled)
            self.table_params.setItem(row, 0, item_name)

            sp_val = QDoubleSpinBox()
            sp_val.setDecimals(6)
            sp_val.setRange(-1e6, 1e6)
            sp_val.setSingleStep(0.001)
            sp_val.setValue(default_val)
            self.table_params.setCellWidget(row, 1, sp_val)

            sp_min = QDoubleSpinBox()
            sp_min.setDecimals(6)
            sp_min.setRange(-1e6, 1e6)
            sp_min.setSingleStep(0.001)
            sp_min.setValue(default_min)
            self.table_params.setCellWidget(row, 2, sp_min)

            sp_max = QDoubleSpinBox()
            sp_max.setDecimals(6)
            sp_max.setRange(-1e6, 1e6)
            sp_max.setSingleStep(0.001)
            sp_max.setValue(default_max)
            self.table_params.setCellWidget(row, 3, sp_max)

            chk_vary = QCheckBox()
            chk_vary.setChecked(True)
            # 置中
            cell_widget = QWidget()
            cell_layout = QHBoxLayout(cell_widget)
            cell_layout.addWidget(chk_vary)
            cell_layout.setAlignment(Qt.AlignCenter)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            self.table_params.setCellWidget(row, 4, cell_widget)

            self.param_widgets[key] = dict(value=sp_val, min=sp_min, max=sp_max, vary=chk_vary)

        self.table_params.setMinimumHeight(230)
        layout.addWidget(self.table_params)

       # 額外設定 n_max, x_upper 以及材料維度
        extra_layout = QGridLayout()
        
        extra_layout.addWidget(QLabel("模型維度:"), 0, 0)
        self.combo_dim = QComboBox()
        self.combo_dim.addItems(["2D / Quasi-2D (二維/準二維)", "3D Bulk (三維塊材)"])
        extra_layout.addWidget(self.combo_dim, 0, 1)

        extra_layout.addWidget(QLabel("n_max (激子求和項數):"), 1, 0)
        self.spin_nmax = QSpinBox()
        self.spin_nmax.setRange(1, 30)
        self.spin_nmax.setValue(8)
        extra_layout.addWidget(self.spin_nmax, 1, 1)

        extra_layout.addWidget(QLabel("x_upper (積分上限, eV):"), 2, 0)
        self.spin_xupper = QDoubleSpinBox()
        self.spin_xupper.setRange(0.1, 100)
        self.spin_xupper.setValue(10.0)
        extra_layout.addWidget(self.spin_xupper, 2, 1)

        layout.addLayout(extra_layout)

        return box

    # ---- 擬合執行區塊 ----
    def _build_fit_group(self):
        box = QGroupBox("4. 執行擬合")
        layout = QHBoxLayout(box)

        layout.addWidget(QLabel("演算法:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems(["leastsq", "least_squares", "nelder"])
        layout.addWidget(self.combo_method)

        btn_fit = QPushButton("開始擬合 (Fit)")
        btn_fit.setStyleSheet("font-weight: bold;")
        btn_fit.clicked.connect(self.on_run_fit)
        layout.addWidget(btn_fit)

        return box

    # ---- 結果顯示區塊 ----
    def _build_result_group(self):
        box = QGroupBox("5. 擬合結果")
        layout = QVBoxLayout(box)

        self.table_results = QTableWidget(len(PARAM_INFO), 3)
        self.table_results.setHorizontalHeaderLabels(["參數", "擬合值", "標準誤差"])
        self.table_results.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_results.verticalHeader().setVisible(False)
        for row, (key, label, *_r) in enumerate(PARAM_INFO):
            item_name = QTableWidgetItem(label)
            item_name.setFlags(Qt.ItemIsEnabled)
            self.table_results.setItem(row, 0, item_name)
            self.table_results.setItem(row, 1, QTableWidgetItem("-"))
            self.table_results.setItem(row, 2, QTableWidgetItem("-"))
        self.table_results.setMinimumHeight(230)
        layout.addWidget(self.table_results)

        self.text_stats = QTextEdit()
        self.text_stats.setReadOnly(True)
        self.text_stats.setMaximumHeight(90)
        self.text_stats.setPlainText("R^2 = -\nReduced chi-square = -\n擬合狀態: 尚未執行")
        layout.addWidget(self.text_stats)

        return box

    # ---- 匯出區塊 ----
    def _build_export_group(self):
        box = QGroupBox("6. 匯出結果 (可匯入 Origin)")
        layout = QVBoxLayout(box)

        btn_export_curves = QPushButton("匯出數據 + 擬合曲線 (.txt, tab 分隔)")
        btn_export_curves.clicked.connect(self.on_export_curves)
        layout.addWidget(btn_export_curves)

        btn_export_params = QPushButton("匯出擬合參數摘要 (.txt)")
        btn_export_params.clicked.connect(self.on_export_params)
        layout.addWidget(btn_export_params)

        return box

    # ------------------------------------------------------------------
    # 繪圖
    # ------------------------------------------------------------------
    def _init_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("Photon Energy  (eV)")
        self.ax.set_ylabel("Absorbance  (a.u.)")
        self.ax.set_title("吸收光譜")
        self.canvas.draw_idle()

    def _redraw(self, show_fit=False):
        self.ax.clear()
        self.ax.set_xlabel("Photon Energy  (eV)")
        self.ax.set_ylabel("Absorbance  (a.u.)")
        self.ax.set_title("吸收光譜")

        if self.E_full is not None:
            self.ax.scatter(self.E_full, self.A_full, s=12, color="black",
                             label="Experimental data", zorder=2)

        if self.fit_range is not None:
            self.ax.axvspan(self.fit_range[0], self.fit_range[1],
                             color="tab:blue", alpha=0.12, label="Fit range", zorder=1)

        if show_fit and self.last_curves is not None:
            E_used, A_used, exc, cont, total = self.last_curves
            self.ax.plot(E_used, exc, "--", color="tab:orange", lw=2,
                         label=r"Excitonic absorption $\alpha_{ext}$", zorder=3)
            self.ax.plot(E_used, cont, "-.", color="tab:green", lw=2,
                         label=r"Continuum absorption $\alpha_{cont}$", zorder=3)
            self.ax.plot(E_used, total, "-", color="tab:red", lw=2.2,
                         label="Total fit", zorder=4)

        self.ax.legend(loc="best", fontsize=9)
        self.canvas.draw_idle()

        # 重新建立 span selector (clear() 會移除舊的 artist 綁定)
        self.span_selector = SpanSelector(
            self.ax, self._on_span_select, "horizontal",
            useblit=True, interactive=True,
            props=dict(alpha=0.2, facecolor="tab:blue"),
        )
        self.span_selector.set_active(self.btn_toggle_select.isChecked())

    # ------------------------------------------------------------------
    # 事件處理：匯入
    # ------------------------------------------------------------------
    def on_import_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇實驗數據檔案", "",
            "數據檔案 (*.txt *.csv *.dat);;所有檔案 (*)"
        )
        if not path:
            return
        try:
            E, A = load_spectrum_file(path)
        except Exception as e:
            QMessageBox.critical(self, "匯入失敗", str(e))
            return

        self.E_full = E
        self.A_full = A
        self.data_filename = path
        self.fit_range = (float(E.min()), float(E.max()))
        self.spin_emin.setValue(self.fit_range[0])
        self.spin_emax.setValue(self.fit_range[1])
        self.label_filename.setText(f"已載入: {os.path.basename(path)}  ({len(E)} 個數據點)")
        self.status_label.setText("數據已匯入。可拖曳圖上範圍或手動輸入 Emin/Emax，接著設定初始參數並執行擬合。")

        self.fit_result = None
        self.last_curves = None
        self._redraw(show_fit=False)

    # ------------------------------------------------------------------
    # 事件處理：範圍選擇
    # ------------------------------------------------------------------
    def on_toggle_span_select(self, checked):
        if self.E_full is None:
            QMessageBox.information(self, "提示", "請先匯入實驗數據。")
            self.btn_toggle_select.setChecked(False)
            return
        self.span_selector.set_active(checked)
        if checked:
            self.btn_toggle_select.setText("停止「拖曳選擇範圍」(拖曳滑鼠選取)")
            self.status_label.setText("請在圖上以滑鼠拖曳，選擇欲擬合的能量範圍。")
        else:
            self.btn_toggle_select.setText("啟動「拖曳選擇範圍」")

    def _on_span_select(self, xmin, xmax):
        if xmin == xmax:
            return
        self.fit_range = (float(min(xmin, xmax)), float(max(xmin, xmax)))
        self.spin_emin.setValue(self.fit_range[0])
        self.spin_emax.setValue(self.fit_range[1])
        self.status_label.setText(
            f"已選擇擬合範圍: {self.fit_range[0]:.4f} — {self.fit_range[1]:.4f} eV"
        )
        self._redraw(show_fit=False)

    def on_apply_manual_range(self):
        if self.E_full is None:
            QMessageBox.information(self, "提示", "請先匯入實驗數據。")
            return
        emin, emax = self.spin_emin.value(), self.spin_emax.value()
        if emin >= emax:
            QMessageBox.warning(self, "範圍錯誤", "Emin 必須小於 Emax。")
            return
        self.fit_range = (emin, emax)
        self.status_label.setText(f"已套用擬合範圍: {emin:.4f} — {emax:.4f} eV")
        self._redraw(show_fit=False)

    def on_reset_range(self):
        if self.E_full is None:
            return
        self.fit_range = (float(self.E_full.min()), float(self.E_full.max()))
        self.spin_emin.setValue(self.fit_range[0])
        self.spin_emax.setValue(self.fit_range[1])
        self._redraw(show_fit=False)

    # ------------------------------------------------------------------
    # 事件處理：自動估算初始值
    # ------------------------------------------------------------------
    def on_auto_estimate(self):
        if self.E_full is None:
            QMessageBox.information(self, "提示", "請先匯入實驗數據。")
            return

        rng = self.fit_range or (float(self.E_full.min()), float(self.E_full.max()))
        mask = (self.E_full >= rng[0]) & (self.E_full <= rng[1])
        E_used = self.E_full[mask]
        A_used = self.A_full[mask]
        if len(E_used) < 5:
            QMessageBox.warning(self, "資料不足", "目前選擇範圍內數據點過少，無法估算，請先擴大範圍。")
            return

        # 取得目前的維度與使用者填寫的 E0 初始值
        dim_str = "2D" if "2D" in self.combo_dim.currentText() else "3D"
        current_e0 = self.param_widgets["E0"]["value"].value()

        # 傳入估算器
        est = estimate_initial_parameters(
            E_used, A_used, 
            dim=dim_str, 
            user_E0_guess=current_e0
        )
        peak_found = est.pop("_peak_found", False)

        for key, val in est.items():
            widgets = self.param_widgets.get(key)
            if widgets is None:
                continue
            widgets["value"].setValue(val)

            if key == "Eg":
                # Eg 是絕對能量值，不適合用「估計值 x 倍數」設定邊界
                # (例如 val*0.1~val*5 可能得到 0.17~8.5 eV 這種離譜範圍，
                #  會讓擬合演算法在不合理的空間內遊走、極度變慢甚至卡住)
                # 改用「以估計值為中心，加減目前選擇範圍的一半（至少 0.3 eV）」
                span = float(rng[1] - rng[0])
                half_window = max(span * 0.5, 0.3)
                widgets["min"].setValue(val - half_window)
                widgets["max"].setValue(val + half_window)
            elif val > 0:
                widgets["min"].setValue(max(val * 0.1, 1e-5))
                widgets["max"].setValue(val * 5)
            else:
                widgets["min"].setValue(val - 2.0)
                widgets["max"].setValue(val + 2.0)

        msg = "已根據目前範圍自動估算初始值。"
        if peak_found:
            msg += f" 偵測到激子峰，已結合您填寫的 E0 ({current_e0} eV) 與 {dim_str} 模型推算 Eg。"
        else:
            msg += " 未偵測到明顯峰值，Eg 改以吸收邊最大斜率位置估算。"
        self.status_label.setText(msg)

    # ------------------------------------------------------------------
    # 事件處理：擬合
    # ------------------------------------------------------------------
    def _collect_user_param_values(self):
        user_values = {}
        for key, widgets in self.param_widgets.items():
            user_values[key] = dict(
                value=widgets["value"].value(),
                min=widgets["min"].value(),
                max=widgets["max"].value(),
                vary=widgets["vary"].isChecked(),
            )
        return user_values

    def on_run_fit(self):
        if self.E_full is None:
            QMessageBox.information(self, "提示", "請先匯入實驗數據。")
            return
        if self.fit_range is None:
            self.fit_range = (float(self.E_full.min()), float(self.E_full.max()))

        mask = (self.E_full >= self.fit_range[0]) & (self.E_full <= self.fit_range[1])
        E_used = self.E_full[mask]
        A_used = self.A_full[mask]
        if len(E_used) < 6:
            QMessageBox.warning(self, "數據不足", "選擇的範圍內數據點過少，請擴大擬合範圍。")
            return

        # 取得參數
        user_values = self._collect_user_param_values()
        n_max = self.spin_nmax.value()
        x_upper = self.spin_xupper.value()
        method = self.combo_method.currentText()
        # 判斷維度
        dim_str = "2D" if "2D" in self.combo_dim.currentText() else "3D"

        self.status_label.setText("擬合執行中，請稍候...(數值積分可能需要數秒到數十秒)")
        QApplication.processEvents()

        try:
            result = run_fit(E_used, A_used, user_values, n_max=n_max,
                              x_upper=x_upper, method=method, dim=dim_str)
        except Exception as e:
            QMessageBox.critical(self, "擬合失敗", f"擬合過程發生錯誤:\n{e}")
            self.status_label.setText("擬合失敗，請檢查初始參數與範圍設定。")
            return

        self.fit_result = result
        pvals = result.params.valuesdict()
        model_curve = total_absorption(
            E_used, pvals["alpha0"], pvals["E0"], pvals["Eg"],
            pvals["Gamma_ex"], pvals["Gamma_c"], pvals["b"],
            n_max=n_max, x_upper=x_upper, dim=dim_str
        )
        r2, ss_res = goodness_of_fit(A_used, model_curve)
        exc, cont, total = get_component_curves(E_used, pvals, n_max=n_max, x_upper=x_upper, dim=dim_str)
        self.last_curves = (E_used, A_used, exc, cont, total)

        # 更新結果表格
        for row, (key, label, *_r) in enumerate(PARAM_INFO):
            p = result.params[key]
            self.table_results.setItem(row, 1, QTableWidgetItem(f"{p.value:.6g}"))
            stderr_txt = f"{p.stderr:.3g}" if p.stderr is not None else "N/A"
            self.table_results.setItem(row, 2, QTableWidgetItem(stderr_txt))

        # 額外列出重要衍生量:
        #   Eb^n = E0/(n-1/2)^2，故基態 (n=1) 束縛能 Eb(n=1) = E0/(0.5)^2 = 4*E0
        #   n=1 激子峰理論位置 Eex(n=1) = Eg - Eb(n=1) = Eg - 4*E0
        # 額外列出重要衍生量 (依據維度判斷基態束縛能)
        redchi = result.redchi if hasattr(result, "redchi") else float("nan")
        if dim_str == "2D":
            Eb_n1 = 4.0 * pvals["E0"]
            eb_str = "4×E0"
        else:
            Eb_n1 = pvals["E0"]
            eb_str = "E0"
            
        Eex_n1 = pvals["Eg"] - Eb_n1
        
        stats_text = (
            f"模型維度: {dim_str}\n"
            f"擬合是否成功: {'是' if result.success else '否'}\n"
            f"R^2 = {r2:.6f}\n"
            f"Reduced chi-square = {redchi:.6g}\n"
            f"殘差平方和 (SSR) = {ss_res:.6g}\n"
            f"基態(n=1)激子束縛能 Eb(n=1) = {eb_str} = {Eb_n1:.6g} eV\n"
            f"基態(n=1)激子峰理論位置 Eex(n=1) = Eg-Eb(n=1) = {Eex_n1:.6g} eV\n"
            f"擬合範圍: {self.fit_range[0]:.4f} - {self.fit_range[1]:.4f} eV, "
            f"點數: {len(E_used)}\n"
            f"lmfit 訊息: {result.message}"
        )
        self.text_stats.setPlainText(stats_text)

        self.status_label.setText("擬合完成！結果已顯示於左側表格與右側圖表。")
        self._redraw(show_fit=True)

    # ------------------------------------------------------------------
    # 事件處理：匯出
    # ------------------------------------------------------------------
    def on_export_curves(self):
        if self.last_curves is None:
            QMessageBox.information(self, "提示", "請先完成擬合，才能匯出曲線。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "匯出數據與擬合曲線", "fit_curves.txt",
            "Text Files (*.txt);;CSV Files (*.csv)"
        )
        if not path:
            return

        E_used, A_used, exc, cont, total = self.last_curves
        sep = "," if path.lower().endswith(".csv") else "\t"

        header = sep.join([
            "Energy(eV)", "Absorbance_exp", "Fit_Total",
            "Excitonic_absorption", "Continuum_absorption"
        ])
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(header + "\n")
                for i in range(len(E_used)):
                    row = [E_used[i], A_used[i], total[i], exc[i], cont[i]]
                    f.write(sep.join(f"{v:.8g}" for v in row) + "\n")
        except Exception as e:
            QMessageBox.critical(self, "匯出失敗", str(e))
            return

        QMessageBox.information(
            self, "匯出成功",
            f"已匯出至:\n{path}\n\n"
            "在 Origin 中可直接以 Import > Single ASCII 匯入此檔案，"
            "第一欄 (Energy) 設為 X，其餘四欄設為 Y，"
            "即可將 Data / Total fit / Exciton / Continuum 疊加繪製，並分別指定不同顏色。"
        )

    def on_export_params(self):
        if self.fit_result is None:
            QMessageBox.information(self, "提示", "請先完成擬合，才能匯出參數摘要。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "匯出擬合參數摘要", "fit_parameters.txt", "Text Files (*.txt)"
        )
        if not path:
            return

        # 1. 取得目前選擇的模型維度
        dim_str = "2D" if "2D" in self.combo_dim.currentText() else "3D"

        lines = ["吸收光譜 Elliott formula 擬合結果摘要", "=" * 40]
        if self.data_filename:
            lines.append(f"數據檔案: {self.data_filename}")
        if self.fit_range:
            lines.append(f"擬合範圍: {self.fit_range[0]:.6g} - {self.fit_range[1]:.6g} eV")
        
        # 在摘要中記錄本次擬合使用的維度
        lines.append(f"模型維度: {dim_str}")  
        lines.append(f"n_max = {self.spin_nmax.value()}, x_upper = {self.spin_xupper.value()} eV")
        lines.append(f"擬合演算法: {self.combo_method.currentText()}")
        lines.append("")
        lines.append(f"{'參數':<24}{'擬合值':<18}{'標準誤差':<18}")
        for key, label, *_r in PARAM_INFO:
            p = self.fit_result.params[key]
            stderr_txt = f"{p.stderr:.6g}" if p.stderr is not None else "N/A"
            lines.append(f"{label:<24}{p.value:<18.6g}{stderr_txt:<18}")

        pvals = self.fit_result.params.valuesdict()
        E_used, A_used, exc, cont, total = self.last_curves
        r2, ss_res = goodness_of_fit(A_used, total)
        redchi = self.fit_result.redchi if hasattr(self.fit_result, "redchi") else float("nan")
        
        # 2. 依據維度動態計算基態束縛能與峰值位置
        if dim_str == "2D":
            Eb_n1 = 4.0 * pvals["E0"]
            eb_str = "4×E0"
        else:
            Eb_n1 = pvals["E0"]
            eb_str = "E0"
            
        Eex_n1 = pvals["Eg"] - Eb_n1
        
        # 3. 將動態轉換後的字串寫入報告中
        lines.append("")
        lines.append(f"基態(n=1)激子束縛能 Eb(n=1) = {eb_str} = {Eb_n1:.6g} eV")
        lines.append(f"基態(n=1)激子峰理論位置 Eex(n=1) = Eg-Eb(n=1) = {Eex_n1:.6g} eV")
        lines.append(f"R^2 = {r2:.6f}")
        lines.append(f"Reduced chi-square = {redchi:.6g}")
        lines.append(f"殘差平方和 (SSR) = {ss_res:.6g}")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            QMessageBox.critical(self, "匯出失敗", str(e))
            return

        QMessageBox.information(self, "匯出成功", f"參數摘要已匯出至:\n{path}")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
