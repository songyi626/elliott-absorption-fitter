# -*- coding: utf-8 -*-
"""
fitting_engine.py
==================
使用 lmfit 對 Elliott formula 進行非線性最小平方擬合。
"""

import numpy as np
from lmfit import Parameters, Minimizer

from elliott_model import total_absorption, component_curves

# 參數的中文顯示名稱、預設初始值、預設下限/上限
PARAM_INFO = [
    # key,        中文名稱,                 初始值,  下限,     上限
    ("alpha0",    "α0 (整體振幅)",           1.0,     0.0,      1e6),
    ("E0",        "E0 (激子雷德堡能量, eV)", 0.03,    0.0001,   1.0),
    ("Eg",        "Eg (能隙, eV)",           1.7,     0.0,      10.0),
    ("Gamma_ex",  "Γex (激子展寬, eV)",      0.02,    0.0001,   1.0),
    ("Gamma_c",   "Γc (連續態展寬, eV)",     0.05,    0.0001,   1.0),
    ("b",         "b (非拋物線參數)",        0.3,     -5.0,     5.0),
]

FIXED_INFO = [
    # key,      中文名稱,                   預設值
    ("n_max",   "n_max (激子求和上限)",      8),
    ("x_upper", "x_upper (連續態積分上限, eV)", 10.0),
]


def build_parameters(user_values):
    """
    user_values: dict，格式為
        { key: {"value":..., "min":..., "max":..., "vary": bool} }
    回傳 lmfit.Parameters 物件。
    """
    params = Parameters()
    for key, _, default_val, default_min, default_max in PARAM_INFO:
        cfg = user_values.get(key, {})
        val = cfg.get("value", default_val)
        vmin = cfg.get("min", default_min)
        vmax = cfg.get("max", default_max)
        vary = cfg.get("vary", True)
        params.add(key, value=val, min=vmin, max=vmax, vary=vary)
    return params


def _residual(params, E, data, n_max, x_upper, dim):
    p = params.valuesdict()
    model = total_absorption(
        E, p["alpha0"], p["E0"], p["Eg"], p["Gamma_ex"], p["Gamma_c"], p["b"],
        n_max=n_max, x_upper=x_upper, dim=dim
    )
    return model - data


def run_fit(E, data, user_values, n_max=8, x_upper=10.0, method="leastsq", dim="2D",
            max_nfev=None):
    """
    dim: "2D" 或 "3D"
    max_nfev: 最大函式評估次數上限。若為 None，預設依參數數量給一個合理上限
              (避免參數邊界設定不良時，優化器長時間在不合理空間內遊走，
              造成 GUI 長時間無回應)。
    """
    params = build_parameters(user_values)
    if max_nfev is None:
        n_vary = sum(1 for p in params.values() if p.vary)
        max_nfev = max(500 * (n_vary + 1), 2000)
    minimizer = Minimizer(_residual, params, fcn_args=(E, data, n_max, x_upper, dim))
    result = minimizer.minimize(method=method, max_nfev=max_nfev)
    return result


def goodness_of_fit(data, model):
    """回傳 R^2 與 reduced chi-square 所需的殘差平方和"""
    ss_res = np.sum((data - model) ** 2)
    ss_tot = np.sum((data - np.mean(data)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return r2, ss_res


def get_component_curves(E, params_valuesdict, n_max=8, x_upper=10.0, dim="2D"):
    p = params_valuesdict
    return component_curves(
        E, p["alpha0"], p["E0"], p["Eg"], p["Gamma_ex"], p["Gamma_c"], p["b"],
        n_max=n_max, x_upper=x_upper, dim=dim
    )
